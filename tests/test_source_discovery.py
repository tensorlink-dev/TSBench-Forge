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

import pytest
import yaml

from source_discovery import __main__, audit, config, coverage, duration, llm, runner, vet

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


ENGY = "https://api.engy.ai/v1/chat/completions"


def test_engy_must_be_sent_the_reasoning_disable() -> None:
    """Measured against engy.ai on 2026-08-13: glm-5.2 thinks by default there,
    and without `reasoning: {enabled: false}` it spends the whole budget in
    reasoning_content and returns content="" with finish_reason="length" — an
    HTTP 200 that reads as truncation rather than misconfiguration.
    `enable_thinking` and `thinking.type` are both silently ignored."""
    cfg = llm.OpenRouterConfig.from_env({
        "LLM_BASE_URL": ENGY, "LLM_MODEL": "glm-5.2", "ENGY_API_KEY": "k",
        "LLM_REASONING_ENABLED": "false",
    })
    assert cfg.send_reasoning
    body = llm.build_request_body(_inputs(), cfg, cfg.max_tokens)
    assert body["reasoning"] == {"enabled": False}
    assert body["model"] == "glm-5.2"
    assert body["temperature"] == cfg.temperature


def test_unknown_provider_gets_a_plain_openai_body() -> None:
    """An endpoint not measured to accept `reasoning` gets the standard fields
    only; an unknown key can be a hard 400."""
    cfg = llm.OpenRouterConfig.from_env({
        "LLM_BASE_URL": "https://api.example.com/v1/chat/completions",
        "LLM_API_KEY": "k", "LLM_REASONING_MAX_TOKENS": "3000",
    })
    assert not cfg.send_reasoning
    body = llm.build_request_body(_inputs(), cfg, cfg.max_tokens)
    assert set(body) == {"model", "messages", "max_tokens", "temperature"}
    # The temperature omission exists only for OpenRouter's Anthropic-style
    # normalisation; elsewhere it would silently drop an operator's setting.
    assert body["temperature"] == cfg.temperature


def test_send_reasoning_follows_the_endpoint() -> None:
    for env in ({"OPENROUTER_API_KEY": "k"},
                {"LLM_BASE_URL": ENGY, "LLM_API_KEY": "k"}):
        assert llm.OpenRouterConfig.from_env(env).send_reasoning
    unknown = {"LLM_BASE_URL": "https://api.example.com/v1/chat/completions",
               "LLM_API_KEY": "k"}
    assert not llm.OpenRouterConfig.from_env(unknown).send_reasoning
    # ...and either default can be overridden.
    assert llm.OpenRouterConfig.from_env(
        {**unknown, "LLM_SEND_REASONING": "true"}).send_reasoning
    assert not llm.OpenRouterConfig.from_env(
        {"LLM_BASE_URL": ENGY, "LLM_API_KEY": "k",
         "LLM_SEND_REASONING": "false"}).send_reasoning


def test_openrouter_config_still_works_unchanged() -> None:
    """The OPENROUTER_* names predate this and are still in .env and CI."""
    cfg = llm.OpenRouterConfig.from_env({
        "OPENROUTER_API_KEY": "k", "OPENROUTER_MODEL": "z-ai/glm-5.2",
        "OPENROUTER_REASONING_ENABLED": "false", "OPENROUTER_MAX_TOKENS": "32000",
    })
    assert (cfg.model, cfg.max_tokens) == ("z-ai/glm-5.2", 32000)
    assert cfg.enabled and cfg.send_reasoning and not cfg.reasoning_enabled
    body = llm.build_request_body(_inputs(), cfg, cfg.max_tokens)
    assert body["reasoning"] == {"enabled": False}


def test_new_names_win_over_the_legacy_ones() -> None:
    cfg = llm.OpenRouterConfig.from_env({
        "OPENROUTER_MODEL": "old", "LLM_MODEL": "glm-5.2",
        "OPENROUTER_API_KEY": "old-key", "LLM_API_KEY": "new-key",
    })
    assert cfg.model == "glm-5.2" and cfg.api_key == "new-key"


def test_a_provider_key_is_never_sent_to_another_provider() -> None:
    """The likely half-finished migration: LLM_BASE_URL is repointed but the
    old key is still sitting in .env. Falling back to it would hand the
    OpenRouter credential to whoever runs the new endpoint."""
    cfg = llm.OpenRouterConfig.from_env({
        "LLM_BASE_URL": ENGY, "LLM_MODEL": "glm-5.2",
        "OPENROUTER_API_KEY": "sk-or-secret",
    })
    assert cfg.api_key is None, "must not leak the OpenRouter key to engy.ai"
    assert not cfg.enabled, "and must fail closed rather than send nothing useful"

    # ...and the mirror image.
    other = llm.OpenRouterConfig.from_env({"ENGY_API_KEY": "engy-secret"})
    assert other.base_url.startswith("https://openrouter.ai")
    assert other.api_key is None, "must not leak the engy key to OpenRouter"


def test_llm_api_key_is_honoured_for_any_endpoint() -> None:
    """The deliberate 'I mean this key for this endpoint' spelling."""
    for base in (ENGY, "https://api.example.com/v1/chat/completions"):
        cfg = llm.OpenRouterConfig.from_env(
            {"LLM_BASE_URL": base, "LLM_API_KEY": "k"})
        assert cfg.api_key == "k" and cfg.enabled


def test_provider_key_matches_on_subdomains_not_lookalikes() -> None:
    ok = llm.OpenRouterConfig.from_env(
        {"LLM_BASE_URL": "https://eu.api.engy.ai/v1/chat/completions",
         "ENGY_API_KEY": "k"})
    assert ok.api_key == "k"
    # A host merely ENDING in the brand is a different party.
    evil = llm.OpenRouterConfig.from_env(
        {"LLM_BASE_URL": "https://engy.ai.attacker.example/v1/chat/completions",
         "ENGY_API_KEY": "k"})
    assert evil.api_key is None


def test_key_resolution_order_and_blank_handling() -> None:
    # Blank is absent, not a value: CI substitutes unset vars as "".
    cfg = llm.OpenRouterConfig.from_env(
        {"LLM_BASE_URL": ENGY, "LLM_API_KEY": "", "ENGY_API_KEY": "engy"})
    assert cfg.api_key == "engy", "blank LLM_API_KEY falls through to the host's own"
    assert llm.OpenRouterConfig.from_env(
        {"LLM_API_KEY": "", "OPENROUTER_API_KEY": "or"}).api_key == "or"
    assert not llm.OpenRouterConfig.from_env({}).enabled


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


# --------------------------------------------------------------------------- #
# coverage: banding, disabled filtering, provider diversity
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("freq,band", [
    ("PT10S", "sub-min"), ("PT1M", "sub-min"), ("PT4M", "few-min"),
    ("PT15M", "few-min"), ("PT20M", "half-hour"),   # config convention
    ("PT1H", "hourly"), ("P3D", "daily"),
    ("P2W", "weekly"), ("P30D", "monthly"), ("P3M", "quarterly"),
    ("P1Y", "yearly"), ("irregular", "irregular"), ("", "irregular"),
])
def test_band_for_parses_durations_not_just_known_strings(freq, band) -> None:
    assert coverage.band_for(freq) == band


def test_load_registry_skips_disabled_sources(tmp_path) -> None:
    cat = tmp_path / "sources.yaml"
    cat.write_text(yaml.safe_dump([
        {"id": "live_one", "domain": "energy", "frequency": "P1D",
         "endpoint": {"url": "https://a.example/x"}},
        {"id": "dead_one", "domain": "energy", "frequency": "P1D",
         "disabled": True, "endpoint": {"url": "https://b.example/x"}},
    ]))
    assert [s["id"] for s in coverage.load_registry(cat)] == ["live_one"]
    assert len(coverage.load_registry(cat, include_disabled=True)) == 2


def test_diversity_discounts_single_host_domains() -> None:
    reg = [{"domain": "energy", "url_or_endpoint": f"https://one.example/{i}"}
           for i in range(10)]
    reg += [{"domain": "nature", "url_or_endpoint": f"https://h{i}.example/x"}
            for i in range(10)]
    div = coverage.diversity(reg, cap=5)
    assert div["energy"]["sources"] == div["nature"]["sources"] == 10
    # ten sources on one host count as five; ten hosts count as ten
    assert div["energy"]["effective"] == 5
    assert div["nature"]["effective"] == 10
    assert div["energy"]["top_host_share"] == 1.0
    assert div["nature"]["hosts"] == 10


# --------------------------------------------------------------------------- #
# CLI contracts the autosearch workflow depends on
# --------------------------------------------------------------------------- #

def test_coverage_stdout_is_json_only(capsys) -> None:
    """`--coverage`'s stdout must parse as JSON on its own.

    The workflow pipes stdout straight into ``json.load``. When the human
    coverage matrix was printed to stdout alongside the summary, that pipe died
    with "Expecting value: line 1 column 1" — the matrix, not the JSON, was what
    the parser saw first.
    """
    assert __main__.main(["--coverage", "--catalog", CATALOG]) == 0
    out = capsys.readouterr()
    summary = json.loads(out.out)
    assert "gap_cells" in summary and "n_sources" in summary
    # the matrix is still shown, just on the stream a pipeline can drop
    assert "EFFCTV" in out.err


def test_coverage_needs_no_scraper_dependencies(monkeypatch) -> None:
    """Banding a duration must not import the scraper (httpx/pyarrow at import).

    The autosearch job installs pyyaml + numpy only. Pulling ``scraper`` in to
    parse an ISO-8601 duration made every unlisted frequency an ImportError, so
    ``--coverage`` printed nothing at all and the workflow's parse failed with
    an empty stream.
    """
    import builtins

    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name in ("scraper", "httpx", "pyarrow"):
            raise ImportError(f"{name} is not installed in this job")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert coverage.band_for("PT10S") == "sub-min"   # not in config.FREQ_BAND
    assert coverage.band_for("P3M") == "quarterly"
    assert audit.staleness_threshold("PT5M").total_seconds() > 0


@pytest.mark.parametrize("freq", [
    "PT1S", "PT10S", "PT2M30S", "PT4M", "PT15M", "PT30M", "PT3H",
    "P1D", "P3D", "P2W", "P30D", "P3M", "P1Y", "irregular", "P1Q", "",
])
def test_period_seconds_matches_scraper(freq) -> None:
    """The local parser must stay bit-identical to the scraper's original."""
    scraper = pytest.importorskip(
        "scraper", reason="scraper needs httpx/pyarrow; parity check is opt-in")
    assert duration.period_seconds(freq) == scraper._period_seconds(freq)
