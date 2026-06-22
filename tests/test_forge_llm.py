"""OpenRouter forge proposer: parsing, one-knob enforcement, and fail-safe."""

from __future__ import annotations

import json

import pytest

from config import WEAK_STATE, GeneratorState
from forge_llm import (
    ALLOWED_KNOBS,
    ForgeLLMError,
    OpenRouterConfig,
    apply_mutation,
    chat_completion,
    make_openrouter_proposer,
    parse_mutation,
)
from forge_loop import ForgeStep
from seed import rng_for


def _init_log() -> list[ForgeStep]:
    """A one-row log like the one run_forge always seeds before proposing."""
    return [
        ForgeStep(
            epoch=0, knob=None, decision="init", fitness=0.05, spread=1.0,
            ordering=0.3, difficulty=2.0, gate=0.15, state=WEAK_STATE,
        )
    ]


def _completion(content: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode()


def _transport_returning(content: str):
    def transport(url, headers, body, timeout):
        # The Authorization header must carry the configured key.
        assert headers["Authorization"].startswith("Bearer ")
        return 200, _completion(content)

    return transport


def test_parse_mutation_accepts_valid_json() -> None:
    knob, value = parse_mutation('{"knob": "w_aug_live", "value": 0.4, "rationale": "lift gate"}')
    assert knob == "w_aug_live"
    assert value == 0.4


def test_parse_mutation_tolerates_prose_and_fences() -> None:
    content = 'Sure!\n```json\n{"knob": "aug_severity", "value": 0.5}\n```\nDone.'
    knob, value = parse_mutation(content)
    assert knob == "aug_severity"
    assert value == 0.5


def test_parse_mutation_rejects_unknown_knob() -> None:
    with pytest.raises(ForgeLLMError):
        parse_mutation('{"knob": "not_a_knob", "value": 1}')


def test_parse_mutation_rejects_nonfinite() -> None:
    with pytest.raises(ForgeLLMError):
        parse_mutation('{"knob": "noise_ar_phi", "value": 1e999}')


def test_apply_mutation_changes_at_most_one_field_and_clamps() -> None:
    out = apply_mutation(WEAK_STATE, "changepoint_prob", 99.0)
    # Clamped back into the legal envelope ...
    assert out.changepoint_prob <= 0.6
    # ... and no difficulty knob other than the target moved.
    assert out.regime_switch_prob == WEAK_STATE.regime_switch_prob
    assert out.aug_severity == WEAK_STATE.aug_severity


def test_apply_mutation_blend_renormalises() -> None:
    out = apply_mutation(WEAK_STATE, "w_aug_live", 1.0)
    assert abs(sum(out.blend_weights()) - 1.0) < 1e-9
    assert out.normalized().w_aug_live > WEAK_STATE.normalized().w_aug_live


def test_chat_completion_requires_api_key() -> None:
    cfg = OpenRouterConfig(api_key=None)
    with pytest.raises(ForgeLLMError):
        chat_completion(cfg, [{"role": "user", "content": "hi"}])


def test_chat_completion_retries_then_raises_on_5xx() -> None:
    calls = {"n": 0}
    slept: list[float] = []

    def transport(url, headers, body, timeout):
        calls["n"] += 1
        return 503, b"unavailable"

    cfg = OpenRouterConfig(api_key="k", max_retries=3)
    with pytest.raises(ForgeLLMError):
        chat_completion(cfg, [{"role": "user", "content": "x"}], transport=transport,
                        sleep=slept.append)
    assert calls["n"] == 3  # retried the full budget
    assert slept == [2.0, 4.0]  # exponential backoff between attempts


def test_chat_completion_does_not_retry_4xx() -> None:
    calls = {"n": 0}

    def transport(url, headers, body, timeout):
        calls["n"] += 1
        return 400, b"bad request"

    cfg = OpenRouterConfig(api_key="k", max_retries=4)
    with pytest.raises(ForgeLLMError):
        chat_completion(cfg, [{"role": "user", "content": "x"}], transport=transport,
                        sleep=lambda _: None)
    assert calls["n"] == 1  # non-retryable, failed fast


def test_proposer_uses_llm_output() -> None:
    cfg = OpenRouterConfig(api_key="k")
    proposer = make_openrouter_proposer(
        cfg, transport=_transport_returning('{"knob": "w_spliced", "value": 0.9}')
    )
    state, knob = proposer(WEAK_STATE, [], rng_for("a", 1, "m"))
    assert knob == "w_spliced"
    assert isinstance(state, GeneratorState)
    assert state.normalized().w_spliced > WEAK_STATE.normalized().w_spliced


def test_proposer_falls_back_on_error() -> None:
    reasons: list[str] = []

    def failing_transport(url, headers, body, timeout):
        return 500, b"boom"

    cfg = OpenRouterConfig(api_key="k", max_retries=1)
    proposer = make_openrouter_proposer(
        cfg, transport=failing_transport, on_fallback=reasons.append
    )
    state, knob = proposer(WEAK_STATE, _init_log(), rng_for("a", 1, "m"))
    # Fell back to the heuristic: still a legal one-knob move, and we recorded why.
    assert knob in ALLOWED_KNOBS
    assert abs(sum(state.blend_weights()) - 1.0) < 1e-9
    assert reasons and "HTTP 500" in reasons[0]


def test_proposer_falls_back_without_api_key() -> None:
    cfg = OpenRouterConfig(api_key=None)
    proposer = make_openrouter_proposer(cfg)
    state, knob = proposer(WEAK_STATE, _init_log(), rng_for("a", 1, "m"))
    assert knob in ALLOWED_KNOBS


def test_config_from_env_defaults_to_opus() -> None:
    cfg = OpenRouterConfig.from_env({})
    assert cfg.model == "anthropic/claude-opus-4.8"
    assert not cfg.enabled
    cfg2 = OpenRouterConfig.from_env({"OPENROUTER_API_KEY": "k", "OPENROUTER_MODEL": "x/y"})
    assert cfg2.enabled and cfg2.model == "x/y"
