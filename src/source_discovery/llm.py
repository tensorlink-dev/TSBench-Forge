"""The LLM boundary for source discovery — where the agent actually runs.

This LLM call is **not** in the benchmark's scoring or consensus path. It runs
offline, its output is a candidate list that deterministic code (``vet.py`` then
``quality.py``) vets automatically before anything enters rotation, and it never
touches a forecast or a score. So it needs none of the determinism machinery a
consensus component would: a non-deterministic model is exactly right for
open-ended discovery.

Provider-agnostic via the OpenRouter chat-completions API over stdlib ``urllib``
(no extra dependency). Point ``OPENROUTER_MODEL`` at any model; a strong general
model is recommended since the task is judgement, not pattern completion.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_PROMPT_PATH = Path(__file__).with_name("system_prompt.md")

# A ``finish_reason="error"`` choice is an upstream provider failure (rate limit,
# timeout, transient outage) returned mid-generation, not budget exhaustion.
# OpenRouter may re-route to a healthy provider on a fresh attempt, so retry a
# few times with exponential backoff before giving up.
_MAX_ERROR_RETRIES = 3
_ERROR_BACKOFF_BASE = 2.0  # seconds; sleep = base * 2**(attempt - 1)


def system_prompt() -> str:
    return _PROMPT_PATH.read_text()


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str | None = None
    model: str = "z-ai/glm-5.2"
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    temperature: float = 0.4
    max_tokens: int = 8000
    timeout: float = 120.0
    # Cap on hidden reasoning tokens (OpenRouter-normalized `reasoning.max_tokens`).
    # 0 disables the cap. Without one, reasoning models can burn the whole
    # completion budget deliberating and return content=None.
    reasoning_max_tokens: int = 0
    # Set false to disable hidden reasoning entirely (`reasoning.enabled: false`).
    # z-ai/GLM ignores the max_tokens cap, so this is the only reliable control
    # for it; verified to zero reasoning tokens where the cap did not.
    reasoning_enabled: bool = True

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> OpenRouterConfig:
        e = env if env is not None else os.environ

        def _get(key: str, default):
            # GitHub Actions substitutes an unset `${{ vars.X }}` as an *empty
            # string*, so the var is present-but-blank and os.environ.get's default
            # never fires. Treat blank (or whitespace) as absent, so we fall back to
            # the real default instead of sending e.g. model="" — which OpenRouter
            # rejects with "No models provided" (HTTP 400) — or crashing on
            # float("")/int("") for the numeric knobs.
            v = e.get(key)
            return v if v is not None and v.strip() != "" else default

        return cls(
            api_key=_get("OPENROUTER_API_KEY", None),
            model=_get("OPENROUTER_MODEL", cls.model),
            base_url=_get("OPENROUTER_BASE_URL", cls.base_url),
            temperature=float(_get("OPENROUTER_TEMPERATURE", cls.temperature)),
            max_tokens=int(_get("OPENROUTER_MAX_TOKENS", cls.max_tokens)),
            timeout=float(_get("OPENROUTER_TIMEOUT", cls.timeout)),
            reasoning_max_tokens=int(
                _get("OPENROUTER_REASONING_MAX_TOKENS", cls.reasoning_max_tokens)
            ),
            reasoning_enabled=_get("OPENROUTER_REASONING_ENABLED", "true").strip().lower()
            not in ("false", "0", "off", "no"),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


def build_user_message(inputs: dict) -> str:
    """Render the four agent inputs as one message block.

    ``inputs`` keys: ``current_sources`` (list), ``coverage_summary`` (dict),
    ``target_coverage`` (dict), ``contamination_denylist`` (list),
    ``model_cutoffs`` (dict).
    """
    def block(title: str, obj) -> str:
        return f"## {title}\n```json\n{json.dumps(obj, indent=2, default=str)}\n```"

    parts = [
        "Run the three-phase task on the following inputs.",
        block("CURRENT_SOURCES", inputs["current_sources"]),
        block("COVERAGE_SUMMARY (precomputed)", inputs["coverage_summary"]),
        block("TARGET_COVERAGE", inputs["target_coverage"]),
        block("CONTAMINATION_DENYLIST", inputs["contamination_denylist"]),
        block("MODEL_CUTOFFS", inputs["model_cutoffs"]),
    ]
    if inputs.get("already_proposed"):
        parts.append(block(
            "ALREADY_PROPOSED (do NOT re-propose these hosts/datasets — "
            "every one is auto-rejected; find genuinely NEW sources)",
            inputs["already_proposed"],
        ))
    return "\n\n".join(parts)


def assemble_messages(inputs: dict) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": build_user_message(inputs)},
    ]


def build_request_body(inputs: dict, cfg: OpenRouterConfig, max_tokens: int) -> dict:
    """Assemble the OpenRouter chat-completions request body.

    Kept module-level (not buried in ``propose``) so the parameter interplay is
    testable without a network call — in particular the temperature/reasoning
    interaction below, which was a silent source of HTTP 400s.
    """
    body: dict = {
        "model": cfg.model,
        "messages": assemble_messages(inputs),
        "max_tokens": max_tokens,
    }
    reasoning_enabled_here = False
    if not cfg.reasoning_enabled:
        body["reasoning"] = {"enabled": False}
    elif cfg.reasoning_max_tokens:
        body["reasoning"] = {"max_tokens": cfg.reasoning_max_tokens}
        reasoning_enabled_here = True

    # Providers whose reasoning OpenRouter normalizes to Anthropic-style
    # "thinking" (Claude — the default model here — among them) reject any
    # temperature other than 1 while thinking is enabled: sending 0.4 next to a
    # reasoning budget is a hard HTTP 400 ("temperature may only be set to 1 when
    # thinking is enabled"). Only pin a custom temperature when we are NOT asking
    # for a reasoning budget; otherwise let the provider default (1) stand.
    if not reasoning_enabled_here:
        body["temperature"] = cfg.temperature
    return body


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    """Best-effort extraction of the provider's error message from a 4xx/5xx body.

    A bare ``HTTP Error 400: Bad Request`` says nothing about *why* the request
    was rejected (bad model slug, unsupported parameter, out-of-range value).
    OpenRouter puts the real reason in the JSON response body, which ``HTTPError``
    otherwise discards. Read it back so the surfaced error is actionable. Returns
    a leading ``": <detail>"`` fragment, or ``""`` if nothing useful is present.
    """
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - defensive; body already consumed/closed
        return ""
    raw = raw.strip()
    if not raw:
        return ""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return f": {raw[:500]}"
    msg = obj.get("error")
    if isinstance(msg, dict):
        msg = msg.get("message") or msg.get("code")
    if not msg:
        msg = obj.get("message")
    return f": {msg}" if msg else f": {raw[:500]}"


def _extract_error(body: dict, choice: dict, msg: dict) -> str | None:
    """Pull the provider error message out of an OpenRouter response.

    When a provider errors mid-generation OpenRouter returns HTTP 200 with
    ``finish_reason="error"`` and attaches the real reason to the *choice*
    (``choice["error"]``) or the message — not always at the top level, which is
    why the top-level ``error`` reads ``None`` for these failures. Check all
    three, most-specific first, so the surfaced error is actionable instead of
    ``error=None``.
    """
    for src in (choice.get("error"), msg.get("error"), body.get("error")):
        if isinstance(src, dict):
            detail = src.get("message") or src.get("code")
            if detail:
                return str(detail)
        elif isinstance(src, str) and src.strip():
            return src.strip()
    return None


def parse_response(text: str) -> tuple[str, list[dict]]:
    """Split the model reply into (gap_analysis_prose, candidates list).

    The candidates are the last fenced ```json array (Block 2); everything before
    the first fence is the human-readable gap analysis (Block 1). Falls back to the
    last bare ``[...]`` array if no fence is found. Returns ``[]`` candidates if
    nothing parseable is present (the caller decides how loudly to complain).
    """
    fences = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    candidates: list[dict] = []
    for blob in reversed(fences):
        try:
            obj = json.loads(blob.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            candidates = obj
            break
    if not candidates:
        m = re.search(r"\[\s*\{.*\}\s*\]", text, flags=re.DOTALL)
        if m:
            try:
                candidates = json.loads(m.group(0))
            except json.JSONDecodeError:
                candidates = []
    gap_analysis = text.split("```", 1)[0].strip() if "```" in text else text.strip()
    return gap_analysis, candidates


def propose(inputs: dict, cfg: OpenRouterConfig) -> tuple[str, list[dict]]:
    """Call the model and return (gap_analysis, candidates). Raises on API error.

    There is no heuristic fallback: discovery is inherently the model's job, so a
    missing key or a failed call is a hard error the caller surfaces — it does not
    silently degrade to a fake proposal list.
    """
    if not cfg.enabled:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set — discovery needs a model. "
            "Use `--dry-run` to emit the assembled prompt, or `--vet <file>` to vet "
            "candidates produced elsewhere."
        )
    def _call(max_tokens: int) -> dict:
        payload = json.dumps(build_request_body(inputs, cfg, max_tokens)).encode()
        req = urllib.request.Request(
            cfg.base_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            raise RuntimeError(
                f"OpenRouter call failed: HTTP {exc.code} "
                f"(model={cfg.model}){_http_error_detail(exc)}"
            ) from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"OpenRouter call failed: {exc}") from exc

    # Three recoverable failure modes:
    # - reasoning models (GLM, o-series, R1...) can spend the entire completion
    #   budget on hidden reasoning and return content=None with
    #   finish_reason="length" -> retry once with a doubled budget;
    # - GLM (non-thinking) sometimes ends cleanly right after the "Block 2"
    #   header without emitting the candidate JSON -> one fresh retry;
    # - finish_reason="error" is a transient upstream provider failure (rate
    #   limit, timeout, outage) -> retry a few times with backoff, since
    #   OpenRouter may re-route to a healthy provider.
    max_tokens = cfg.max_tokens
    budget_retried = False
    empty_retried = False
    error_retries = 0
    while True:
        body = _call(max_tokens)
        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        if text.strip():
            gap_analysis, candidates = parse_response(text)
            if candidates or empty_retried:
                return gap_analysis, candidates
            empty_retried = True
            continue
        finish = choice.get("finish_reason")
        if finish == "length" and not budget_retried:
            budget_retried = True
            max_tokens *= 2
            continue
        if finish == "error" and error_retries < _MAX_ERROR_RETRIES:
            error_retries += 1
            time.sleep(_ERROR_BACKOFF_BASE * 2 ** (error_retries - 1))
            continue
        err = _extract_error(body, choice, msg)
        if finish == "error":
            # Budget is not the problem here; don't send the reader chasing
            # OPENROUTER_MAX_TOKENS. This is an upstream provider fault.
            retried = f" after {error_retries} retries" if error_retries else ""
            hint = (
                f"Upstream provider error{retried}; not a budget issue. See the "
                "error detail above, retry later, or set OPENROUTER_MODEL to a "
                "different model/provider."
            )
        else:
            hint = (
                "For reasoning models raise OPENROUTER_MAX_TOKENS or pick a "
                "non-reasoning OPENROUTER_MODEL."
            )
        raise RuntimeError(
            f"OpenRouter returned no content (model={cfg.model}, "
            f"finish_reason={finish!r}, error={err!r}, max_tokens={max_tokens}). "
            + hint
        )
