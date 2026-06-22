"""The forge LLM proposer, backed by OpenRouter.

This is the production implementation of the ``>>> THE LLM GOES HERE <<<``
boundary documented in :func:`forge_loop.propose_mutation` and ``program.md``.
Each epoch the forge hands the model the optimization brief (``program.md``), the
current ``GeneratorState``, the legal envelope (knob bounds), and the recent
``(state, metrics, decision)`` history, and asks for **exactly one** knob change.

Consensus / determinism contract
--------------------------------
The model is non-deterministic, so -- exactly as the architecture requires -- it
runs **once per epoch** to propose the state the forge will commit. Everything
downstream (the concrete challenges) is still a pure function of the committed
manifest + revealed seed, so validators never call the LLM. Two further
guarantees make the LLM safe to put in the loop:

* **Structural one-knob enforcement.** We read a single ``(knob, value)`` from
  the model and apply *only* that field via :func:`dataclasses.replace`. Even a
  misbehaving model that returns ten fields can change at most one, then the
  result is ``normalized().clamped()`` back into the legal envelope. The
  ``program.md`` invariants (never zero the live share, respect bounds) are thus
  enforced by construction, not by trusting the model.
* **Fail-safe fallback.** Any error -- no API key, network failure, malformed
  output, illegal knob -- falls back to the deterministic heuristic proposer
  (:func:`forge_loop.propose_mutation`). The forge degrades gracefully to the
  numpy-only behaviour rather than crashing or stalling.

No third-party SDK is required: the client uses the stdlib ``urllib`` so the
numpy-only install keeps working. Set ``OPENROUTER_API_KEY`` to enable it.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from config import (
    PHI_BOUNDS,
    PROB_BOUNDS,
    SEVERITY_BOUNDS,
    GeneratorState,
)
from forge_loop import ForgeStep, propose_mutation

# The knobs the forge may change, mirrored from ``forge_loop._STEP`` so the LLM
# and the heuristic agree on the action space. Blend weights are raw (the state
# renormalises them); difficulty priors are absolute values within their bounds.
_BLEND_KNOBS = ("w_synth", "w_spliced", "w_aug_live")
_DIFFICULTY_BOUNDS = {
    "changepoint_prob": PROB_BOUNDS,
    "regime_switch_prob": PROB_BOUNDS,
    "aug_severity": SEVERITY_BOUNDS,
    "noise_ar_phi": PHI_BOUNDS,
}
ALLOWED_KNOBS = (*_BLEND_KNOBS, *_DIFFICULTY_BOUNDS)

DEFAULT_MODEL = "anthropic/claude-opus-4.8"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_PROGRAM_MD = Path(__file__).with_name("program.md")

# A transport maps (url, headers, body) -> (status_code, response_bytes). Injected
# so tests can exercise the full proposer without network or an API key.
Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, bytes]]


class ForgeLLMError(RuntimeError):
    """Raised internally when the model response cannot be used; always caught."""


@dataclass(frozen=True)
class OpenRouterConfig:
    """OpenRouter connection + sampling configuration, all env-overridable."""

    api_key: str | None = None
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.4
    max_tokens: int = 1024
    timeout: float = 60.0
    max_retries: int = 4
    # Optional attribution headers OpenRouter surfaces on its dashboard.
    referer: str = "https://github.com/tensorlink-dev/tsbench-forge"
    title: str = "tsbench-forge"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> OpenRouterConfig:
        """Build a config from ``OPENROUTER_*`` environment variables."""
        e = env if env is not None else os.environ

        def num(key: str, default: float, cast: Callable[[str], float]) -> float:
            raw = e.get(key)
            if raw is None or raw == "":
                return default
            try:
                return cast(raw)
            except ValueError:
                return default

        return cls(
            api_key=e.get("OPENROUTER_API_KEY") or None,
            model=e.get("OPENROUTER_MODEL") or DEFAULT_MODEL,
            base_url=e.get("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL,
            temperature=num("OPENROUTER_TEMPERATURE", 0.4, float),
            max_tokens=int(num("OPENROUTER_MAX_TOKENS", 1024, lambda s: int(float(s)))),
            timeout=num("OPENROUTER_TIMEOUT", 60.0, float),
            max_retries=int(num("OPENROUTER_MAX_RETRIES", 4, lambda s: int(float(s)))),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #


def _load_program_md() -> str:
    try:
        return _PROGRAM_MD.read_text(encoding="utf-8")
    except OSError:
        return "(program.md not found; optimize fitness = spread * max(0, ordering) * gate)"


def _system_prompt(program_md: str) -> str:
    return (
        program_md
        + "\n\n---\n\n"
        + "You are being called programmatically. Respond with a SINGLE JSON "
        "object and nothing else (no markdown, no prose), of the exact form:\n"
        '  {"knob": "<one knob name>", "value": <number>, "rationale": "<short>"}\n\n'
        f"`knob` MUST be one of: {', '.join(ALLOWED_KNOBS)}.\n"
        "For blend knobs (w_synth, w_spliced, w_aug_live) `value` is the new raw "
        "weight >= 0 (the system renormalises the three to sum to 1). For "
        "difficulty knobs `value` is the new absolute value; it will be clamped "
        "to its legal range. Propose exactly ONE knob change per call."
    )


def _state_dict(state: GeneratorState) -> dict[str, float]:
    n = state.normalized()
    return {
        "w_synth": round(n.w_synth, 4),
        "w_spliced": round(n.w_spliced, 4),
        "w_aug_live": round(n.w_aug_live, 4),
        "changepoint_prob": round(n.changepoint_prob, 4),
        "regime_switch_prob": round(n.regime_switch_prob, 4),
        "aug_severity": round(n.aug_severity, 4),
        "noise_ar_phi": round(n.noise_ar_phi, 4),
    }


def _user_prompt(state: GeneratorState, log: list[ForgeStep], history: int = 8) -> str:
    bounds = {
        "blend_weights": "raw >= 0, renormalised to sum to 1",
        **{k: list(v) for k, v in _DIFFICULTY_BOUNDS.items()},
    }
    recent = [
        {
            "epoch": s.epoch,
            "knob": s.knob,
            "decision": s.decision,
            "fitness": round(s.fitness, 4),
            "spread": round(s.spread, 4),
            "ordering": round(s.ordering, 4),
            "gate": round(s.gate, 4),
        }
        for s in log[-history:]
    ]
    payload = {
        "current_state": _state_dict(state),
        "bounds": bounds,
        "recent_history": recent,
        "task": "Propose one knob change that most improves fitness next epoch.",
    }
    return json.dumps(payload, indent=2)


# --------------------------------------------------------------------------- #
# Transport + response parsing
# --------------------------------------------------------------------------- #


def _urllib_transport(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:  # 4xx/5xx still carry a body
        return exc.code, exc.read()


_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def chat_completion(
    config: OpenRouterConfig,
    messages: list[dict[str, str]],
    *,
    transport: Transport = _urllib_transport,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Call OpenRouter chat-completions and return the assistant message content.

    Retries transient failures (timeouts, 429, 5xx) with exponential backoff
    (2s, 4s, 8s, 16s). Raises :class:`ForgeLLMError` on a non-retryable or
    exhausted failure; the caller turns that into a heuristic fallback.
    """
    if not config.api_key:
        raise ForgeLLMError("no OPENROUTER_API_KEY configured")

    url = f"{config.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": config.referer,
        "X-Title": config.title,
    }
    body = json.dumps(
        {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")

    last_err = "unknown error"
    for attempt in range(config.max_retries):
        try:
            status, raw = transport(url, headers, body, config.timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            status, raw, last_err = -1, b"", f"transport error: {exc}"
        else:
            if status == 200:
                return _extract_content(raw)
            last_err = f"HTTP {status}: {raw[:200]!r}"
            if status not in _RETRYABLE_STATUS:
                raise ForgeLLMError(last_err)
        if attempt < config.max_retries - 1:
            sleep(2.0 * (2**attempt))
    raise ForgeLLMError(f"exhausted {config.max_retries} retries; last: {last_err}")


def _extract_content(raw: bytes) -> str:
    try:
        data = json.loads(raw.decode("utf-8"))
        return str(data["choices"][0]["message"]["content"])
    except (json.JSONDecodeError, KeyError, IndexError, UnicodeDecodeError) as exc:
        raise ForgeLLMError(f"malformed completion response: {exc}") from exc


def parse_mutation(content: str) -> tuple[str, float]:
    """Parse the model's JSON into a validated ``(knob, value)`` pair.

    Tolerant of a model that wraps JSON in prose or fences: we locate the first
    balanced ``{...}`` object. Raises :class:`ForgeLLMError` if the knob is not
    in :data:`ALLOWED_KNOBS` or the value is not a finite number.
    """
    obj = _first_json_object(content)
    knob = obj.get("knob")
    if knob not in ALLOWED_KNOBS:
        raise ForgeLLMError(f"knob {knob!r} not in allowed set")
    try:
        value = float(obj["value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ForgeLLMError(f"missing/invalid value: {exc}") from exc
    if not np.isfinite(value):
        raise ForgeLLMError("value is not finite")
    return knob, value


def _first_json_object(content: str) -> dict:
    """Extract the first balanced JSON object from ``content``."""
    start = content.find("{")
    if start == -1:
        raise ForgeLLMError("no JSON object in response")
    depth = 0
    for i in range(start, len(content)):
        c = content[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(content[start : i + 1])
                except json.JSONDecodeError as exc:
                    raise ForgeLLMError(f"unparseable JSON object: {exc}") from exc
                if not isinstance(obj, dict):
                    raise ForgeLLMError("top-level JSON is not an object")
                return obj
    raise ForgeLLMError("unbalanced JSON object in response")


def apply_mutation(state: GeneratorState, knob: str, value: float) -> GeneratorState:
    """Apply a single-knob change and project back into the legal envelope.

    Only the named field is replaced, so the one-knob invariant holds by
    construction; ``normalized().clamped()`` enforces the blend-sum and
    difficulty-bound invariants regardless of what the model proposed.
    """
    if knob not in ALLOWED_KNOBS:
        raise ForgeLLMError(f"knob {knob!r} not in allowed set")
    return replace(state, **{knob: value}).normalized().clamped()


# --------------------------------------------------------------------------- #
# The proposer
# --------------------------------------------------------------------------- #


def make_openrouter_proposer(
    config: OpenRouterConfig | None = None,
    *,
    transport: Transport = _urllib_transport,
    program_md: str | None = None,
    on_fallback: Callable[[str], None] | None = None,
) -> Callable[[GeneratorState, list[ForgeStep], np.random.Generator], tuple[GeneratorState, str]]:
    """Build a forge proposer that asks OpenRouter for a one-knob mutation.

    The returned callable matches :func:`forge_loop.propose_mutation`'s signature,
    so it drops straight into :func:`forge_loop.run_forge` via its ``proposer``
    argument. On any failure it transparently falls back to the deterministic
    heuristic proposer, optionally reporting why via ``on_fallback``.
    """
    cfg = config or OpenRouterConfig.from_env()
    system = _system_prompt(program_md if program_md is not None else _load_program_md())

    def proposer(
        state: GeneratorState, log: list[ForgeStep], rng: np.random.Generator
    ) -> tuple[GeneratorState, str]:
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": _user_prompt(state, log)},
            ]
            content = chat_completion(cfg, messages, transport=transport)
            knob, value = parse_mutation(content)
            return apply_mutation(state, knob, value), knob
        except ForgeLLMError as exc:
            if on_fallback is not None:
                on_fallback(str(exc))
            # Fail safe: deterministic heuristic keeps the forge moving.
            return propose_mutation(state, log, rng)

    return proposer
