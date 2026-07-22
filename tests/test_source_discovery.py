"""Tests for the source-discovery agent — deterministic halves only (no network).

Covers registry loading + coverage/gap analysis, the vetting rules (denylist,
duplicate, schema, contamination sanity), and the two-block response parser. The
LLM call itself is not exercised (it needs a key); its output *shape* is tested
via ``llm.parse_response``.
"""

from __future__ import annotations

import io
import json
import os
import urllib.error

from source_discovery import config, coverage, llm, runner, vet

CATALOG = os.path.join(os.path.dirname(__file__), os.pardir, "src", "sources", "sources.yaml")


def _clean_candidate(**over) -> dict:
    base = {
        "name": "Elhovo regional river gauge",
        "domain": "nature",
        "frequency": "PT15M",
        "access_method": "open_api",
        "url_or_endpoint": "https://data.example-gov.bg/api/river/elhovo",
        "license": "CC-BY",
        "estimated_series_count": "40",
        "estimated_length": "ongoing",
        "first_available_date": "2025-06-01",
        "supports_live_future_tasks": True,
        "contamination_risk": "low",
        "contamination_reasoning": "regional gov feed, future values do not exist yet",
        "gap_filled": "nature/few-min live",
        "difficulty_note": "flash-flood regime shifts, not periodic",
        "adapter_notes": "REST JSON, no auth, poll 15min",
        "confidence": "medium",
        "verify": "confirm endpoint schema and station ids",
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #


def test_load_registry_normalises_entries() -> None:
    reg = coverage.load_registry(CATALOG)
    assert len(reg) >= 80
    s = reg[0]
    for field in ("id", "domain", "frequency", "cadence", "access_method",
                  "contamination_risk", "url_or_endpoint"):
        assert field in s
    assert all(s["contamination_risk"] in config.RISK_LEVELS for s in reg)
    assert all(s["cadence"] in config.CADENCE_BANDS for s in reg)


def test_coverage_matrix_and_summary() -> None:
    reg = coverage.load_registry(CATALOG)
    summary = coverage.summarize(reg)
    assert summary["n_sources"] == len(reg)
    assert sum(summary["by_domain"].values()) == len(reg)
    # Every catalog domain is one of the taxonomy domains.
    assert set(summary["by_domain"]) <= set(config.DOMAINS)


def test_gap_cells_rank_high_value_bands_first() -> None:
    reg = coverage.load_registry(CATALOG)
    gaps = coverage.gap_cells(reg)
    assert gaps, "a 92-source catalog should still have unfilled (domain x cadence) cells"
    # Among the worst (largest-deficit) gaps, a high-value band must appear early.
    top = gaps[:8]
    assert any(g["high_value"] for g in top)
    # Deficits are non-increasing (sorted worst-first).
    deficits = [g["deficit"] for g in gaps]
    assert deficits == sorted(deficits, reverse=True)


# --------------------------------------------------------------------------- #
# Vetting
# --------------------------------------------------------------------------- #


def test_clean_candidate_accepts() -> None:
    reg = coverage.load_registry(CATALOG)
    r = vet.vet_candidate(_clean_candidate(), reg)
    assert r.ok and r.verdict == "accept", r.reasons


def test_denylist_dataset_rejected() -> None:
    reg = coverage.load_registry(CATALOG)
    for bad_name in ("ETTh1 transformer temperature", "Monash aggregated archive",
                     "M4 competition monthly", "Electricity ECL load"):
        r = vet.vet_candidate(_clean_candidate(name=bad_name), reg)
        assert r.verdict == "reject" and any("denylist" in x for x in r.reasons), bad_name


def test_real_weather_feed_not_falsely_denylisted() -> None:
    # "weather" is a denylist token, but a real NOAA/open-meteo feed must pass.
    reg = coverage.load_registry(CATALOG)
    r = vet.vet_candidate(
        _clean_candidate(name="NOAA NDBC buoy weather station 41008",
                         url_or_endpoint="https://www.ndbc.noaa.gov/data/realtime2/41008.txt"),
        reg,
    )
    assert r.ok, r.reasons


def test_duplicate_of_existing_source_flagged() -> None:
    reg = coverage.load_registry(CATALOG)
    # Reuse an existing catalog source's host + domain.
    existing = next(s for s in reg if s["url_or_endpoint"])
    r = vet.vet_candidate(
        _clean_candidate(domain=existing["domain"], url_or_endpoint=existing["url_or_endpoint"]),
        reg,
    )
    assert r.verdict == "flag" and any("existing source" in x for x in r.reasons)


def test_schema_incomplete_rejected() -> None:
    reg = coverage.load_registry(CATALOG)
    c = _clean_candidate()
    del c["verify"]
    r = vet.vet_candidate(c, reg)
    assert r.verdict == "reject" and any("verify" in x for x in r.reasons)


def test_invalid_enum_rejected() -> None:
    reg = coverage.load_registry(CATALOG)
    r = vet.vet_candidate(_clean_candidate(access_method="ftp"), reg)
    assert r.verdict == "reject" and any("access_method" in x for x in r.reasons)


def test_low_risk_non_live_without_postcutoff_date_flagged() -> None:
    reg = coverage.load_registry(CATALOG)
    r = vet.vet_candidate(
        _clean_candidate(supports_live_future_tasks=False, first_available_date="2015-01-01"),
        reg,
    )
    assert r.verdict == "flag" and any("cutoff" in x for x in r.reasons)


def test_vet_all_orders_accept_first_reject_last() -> None:
    reg = coverage.load_registry(CATALOG)
    cands = [
        _clean_candidate(name="ETTh1 bundle"),           # reject (denylist)
        _clean_candidate(),                               # accept
        _clean_candidate(name="M3 monthly"),             # reject (denylist)
    ]
    results = vet.vet_all(cands, reg)
    assert [r.verdict for r in results][0] == "accept"
    assert results[-1].verdict == "reject"


# --------------------------------------------------------------------------- #
# LLM boundary (parsing / prompt assembly only — no network)
# --------------------------------------------------------------------------- #


def test_parse_response_extracts_gap_analysis_and_candidates() -> None:
    reply = (
        "## Gap analysis\n\nEnergy sub-minute live is empty; healthcare is monoculture.\n\n"
        "## Candidates\n```json\n"
        + json.dumps([_clean_candidate()])
        + "\n```\n"
    )
    prose, cands = llm.parse_response(reply)
    assert "sub-minute" in prose
    assert isinstance(cands, list) and len(cands) == 1
    assert cands[0]["name"].startswith("Elhovo")


def test_parse_response_tolerates_no_candidates() -> None:
    prose, cands = llm.parse_response("Just prose, the pool looks fine, no gaps.")
    assert cands == []
    assert prose


def test_build_inputs_and_prompt_render() -> None:
    inputs = runner.build_inputs(CATALOG)
    assert set(inputs) >= {"current_sources", "coverage_summary", "target_coverage",
                           "contamination_denylist", "model_cutoffs"}
    msg = llm.build_user_message(inputs)
    assert "CURRENT_SOURCES" in msg and "MODEL_CUTOFFS" in msg
    assert "CONTAMINATION_DENYLIST" in msg


def test_run_vet_writes_outputs(tmp_path) -> None:
    cand_file = tmp_path / "cands.json"
    cand_file.write_text(json.dumps([_clean_candidate(), _clean_candidate(name="ETTh2")]))
    res = runner.run_vet(str(cand_file), CATALOG, str(tmp_path / "out"))
    assert res["proposed"] == 2
    assert res["accept"] == 1 and res["reject"] == 1
    written = json.loads((tmp_path / "out" / "candidates.json").read_text())
    assert all("_verdict" in c for c in written)


def _inputs() -> dict:
    return runner.build_inputs(CATALOG)


def test_from_env_blank_model_falls_back_to_default() -> None:
    # GitHub Actions passes an unset `${{ vars.OPENROUTER_MODEL }}` as "" — the
    # var is present-but-blank, so it must NOT override the code default (an empty
    # model is a 400 "No models provided").
    cfg = llm.OpenRouterConfig.from_env({"OPENROUTER_API_KEY": "k", "OPENROUTER_MODEL": ""})
    assert cfg.model == llm.OpenRouterConfig.model
    assert cfg.model  # non-empty


def test_from_env_explicit_model_overrides_default() -> None:
    cfg = llm.OpenRouterConfig.from_env(
        {"OPENROUTER_API_KEY": "k", "OPENROUTER_MODEL": "vendor/some-model"}
    )
    assert cfg.model == "vendor/some-model"


def test_from_env_blank_numeric_envs_fall_back() -> None:
    # Blank numeric knobs must fall back to defaults, not crash on float("")/int("").
    cfg = llm.OpenRouterConfig.from_env({
        "OPENROUTER_API_KEY": "k",
        "OPENROUTER_TEMPERATURE": "",
        "OPENROUTER_MAX_TOKENS": "  ",
        "OPENROUTER_TIMEOUT": "",
        "OPENROUTER_REASONING_MAX_TOKENS": "",
    })
    assert cfg.temperature == llm.OpenRouterConfig.temperature
    assert cfg.max_tokens == llm.OpenRouterConfig.max_tokens
    assert cfg.timeout == llm.OpenRouterConfig.timeout
    assert cfg.reasoning_max_tokens == llm.OpenRouterConfig.reasoning_max_tokens


def test_from_env_blank_api_key_is_disabled() -> None:
    cfg = llm.OpenRouterConfig.from_env({"OPENROUTER_API_KEY": ""})
    assert not cfg.enabled


def test_request_body_drops_temperature_when_reasoning_budget_set() -> None:
    # A reasoning budget enables Anthropic-style "thinking"; sending a custom
    # temperature alongside it is a hard HTTP 400. The body must omit temperature.
    cfg = llm.OpenRouterConfig(api_key="k", reasoning_max_tokens=3000, temperature=0.4)
    body = llm.build_request_body(_inputs(), cfg, cfg.max_tokens)
    assert "temperature" not in body
    assert body["reasoning"] == {"max_tokens": 3000}


def test_request_body_keeps_temperature_without_reasoning_budget() -> None:
    cfg = llm.OpenRouterConfig(api_key="k", reasoning_max_tokens=0, temperature=0.4)
    body = llm.build_request_body(_inputs(), cfg, cfg.max_tokens)
    assert body["temperature"] == 0.4
    assert "reasoning" not in body


def test_request_body_disables_reasoning_explicitly() -> None:
    cfg = llm.OpenRouterConfig(api_key="k", reasoning_enabled=False, temperature=0.4)
    body = llm.build_request_body(_inputs(), cfg, cfg.max_tokens)
    assert body["reasoning"] == {"enabled": False}
    # Reasoning is off, so the custom temperature is safe to send.
    assert body["temperature"] == 0.4


def _http_error(code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://openrouter.ai/api/v1/chat/completions",
        code=code,
        msg="Bad Request",
        hdrs=None,
        fp=io.BytesIO(body.encode()),
    )


def test_http_error_detail_extracts_openrouter_message() -> None:
    exc = _http_error(400, json.dumps({"error": {"message": "not a valid model id"}}))
    assert llm._http_error_detail(exc) == ": not a valid model id"


def test_http_error_detail_falls_back_to_raw_body() -> None:
    exc = _http_error(400, "upstream is on fire")
    assert llm._http_error_detail(exc) == ": upstream is on fire"


def test_http_error_detail_handles_empty_body() -> None:
    exc = _http_error(400, "")
    assert llm._http_error_detail(exc) == ""


def test_propose_without_key_raises() -> None:
    cfg = llm.OpenRouterConfig.from_env({})  # no OPENROUTER_API_KEY
    assert not cfg.enabled
    try:
        llm.propose({"current_sources": []}, cfg)
    except RuntimeError as e:
        assert "OPENROUTER_API_KEY" in str(e)
    else:
        raise AssertionError("expected RuntimeError when no API key is set")


# --------------------------------------------------------------------------- #
# propose: recovery from a mid-generation provider error (finish_reason=error)
# --------------------------------------------------------------------------- #


def _error_choice(message: str | None = "provider timed out") -> dict:
    """An OpenRouter choice for a mid-generation upstream failure.

    The real reason rides on the *choice*, not the top-level ``error`` — mirror
    that so the extraction path is exercised.
    """
    choice: dict = {"message": {"content": None}, "finish_reason": "error"}
    if message is not None:
        choice["error"] = {"code": 502, "message": message}
    return {"choices": [choice]}


def _content_choice(text: str) -> dict:
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


class _FakeResp:
    def __init__(self, body: dict) -> None:
        self._body = json.dumps(body).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *exc) -> None:
        return None


def _patch_calls(monkeypatch, bodies: list[dict]) -> list[int]:
    """Feed ``bodies`` to successive urlopen calls; no real sleeping."""
    seq = iter(bodies)
    calls = [0]

    def fake_urlopen(req, timeout=None):
        calls[0] += 1
        return _FakeResp(next(seq))

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)
    return calls


def test_extract_error_prefers_choice_level_detail() -> None:
    body = _error_choice("rate limited by upstream")
    choice = body["choices"][0]
    assert llm._extract_error(body, choice, choice["message"]) == "rate limited by upstream"


def test_extract_error_falls_back_to_top_level_and_none() -> None:
    assert llm._extract_error({"error": {"message": "top"}}, {}, {}) == "top"
    assert llm._extract_error({}, {}, {}) is None


def test_propose_retries_finish_reason_error_then_succeeds(monkeypatch) -> None:
    reply = 'Gap analysis.\n\n```json\n[{"name": "x"}]\n```'
    calls = _patch_calls(monkeypatch, [_error_choice(), _content_choice(reply)])
    cfg = llm.OpenRouterConfig(api_key="k")

    gap, cands = llm.propose(_inputs(), cfg)

    assert gap == "Gap analysis."
    assert cands == [{"name": "x"}]
    assert calls[0] == 2  # errored once, retried, then succeeded


def test_propose_surfaces_choice_error_after_exhausting_retries(monkeypatch) -> None:
    bodies = [_error_choice("host is on fire")] * (llm._MAX_ERROR_RETRIES + 1)
    calls = _patch_calls(monkeypatch, bodies)
    cfg = llm.OpenRouterConfig(api_key="k")

    try:
        llm.propose(_inputs(), cfg)
    except RuntimeError as e:
        text = str(e)
        assert "host is on fire" in text  # choice-level detail, not error=None
        assert "Upstream provider error" in text
        assert "OPENROUTER_MAX_TOKENS" not in text  # don't misdirect to the budget
    else:
        raise AssertionError("expected RuntimeError after retries are exhausted")
    assert calls[0] == llm._MAX_ERROR_RETRIES + 1  # first attempt + N retries
