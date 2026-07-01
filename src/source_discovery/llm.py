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
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_PROMPT_PATH = Path(__file__).with_name("system_prompt.md")


def system_prompt() -> str:
    return _PROMPT_PATH.read_text()


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str | None = None
    model: str = "anthropic/claude-opus-4.8"
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    temperature: float = 0.4
    max_tokens: int = 8000
    timeout: float = 120.0

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "OpenRouterConfig":
        e = env if env is not None else os.environ
        return cls(
            api_key=e.get("OPENROUTER_API_KEY"),
            model=e.get("OPENROUTER_MODEL", cls.model),
            base_url=e.get("OPENROUTER_BASE_URL", cls.base_url),
            temperature=float(e.get("OPENROUTER_TEMPERATURE", cls.temperature)),
            max_tokens=int(e.get("OPENROUTER_MAX_TOKENS", cls.max_tokens)),
            timeout=float(e.get("OPENROUTER_TIMEOUT", cls.timeout)),
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

    return "\n\n".join(
        [
            "Run the three-phase task on the following inputs.",
            block("CURRENT_SOURCES", inputs["current_sources"]),
            block("COVERAGE_SUMMARY (precomputed)", inputs["coverage_summary"]),
            block("TARGET_COVERAGE", inputs["target_coverage"]),
            block("CONTAMINATION_DENYLIST", inputs["contamination_denylist"]),
            block("MODEL_CUTOFFS", inputs["model_cutoffs"]),
        ]
    )


def assemble_messages(inputs: dict) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": build_user_message(inputs)},
    ]


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
    payload = json.dumps(
        {
            "model": cfg.model,
            "messages": assemble_messages(inputs),
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
    ).encode()
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
            body = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:  # pragma: no cover - network path
        raise RuntimeError(f"OpenRouter call failed: {exc}") from exc
    text = body["choices"][0]["message"]["content"]
    return parse_response(text)
