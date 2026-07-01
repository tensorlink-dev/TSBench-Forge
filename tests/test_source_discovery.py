"""Tests for the source-discovery agent — deterministic halves only (no network).

Covers registry loading + coverage/gap analysis, the vetting rules (denylist,
duplicate, schema, contamination sanity), and the two-block response parser. The
LLM call itself is not exercised (it needs a key); its output *shape* is tested
via ``llm.parse_response``.
"""

from __future__ import annotations

import json
import os

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


def test_propose_without_key_raises() -> None:
    cfg = llm.OpenRouterConfig.from_env({})  # no OPENROUTER_API_KEY
    assert not cfg.enabled
    try:
        llm.propose({"current_sources": []}, cfg)
    except RuntimeError as e:
        assert "OPENROUTER_API_KEY" in str(e)
    else:
        raise AssertionError("expected RuntimeError when no API key is set")
