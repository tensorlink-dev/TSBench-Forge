"""Tests for scripts/publish_round.py — the public leaderboard-round feed.

The round files under ``docs/data/rounds/`` are a published contract: the
cascade-frontend leaderboard tracker fetches them cross-origin from GitHub
Pages. These tests pin the envelope shape for freshly published rounds and
validate the committed seed rounds against the same contract.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("publish_round", REPO / "scripts/publish_round.py")
publish_round = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(publish_round)

MERGED = {
    "leaderboard": [
        {"model": "toto2", "rank": 1, "crps_rel": 0.5, "mase_rel": 0.7, "crps": 0.4, "mase": 2.0},
        {"model": "seasonal_naive", "rank": 2, "crps_rel": 1.0, "mase_rel": 1.0,
         "crps": 0.8, "mase": 3.5},
    ],
    "friedman_crps": {"statistic": 10.0, "df": 1, "p_value": 0.001},
    "vs_baseline_crps": [
        {"model": "toto2", "median_diff": -0.2, "win_rate": 0.9, "wilcoxon_p": 1e-9,
         "wilcoxon_p_holm": 1e-8, "t_p": 1e-9, "t_p_holm": 1e-8, "significant": True},
    ],
}


def test_build_round_envelope():
    doc = publish_round.build_round(
        MERGED, "2026-08-01", config={"seed": "s1", "n_challenges": 256}, note="test run"
    )
    assert doc["schema_version"] == publish_round.SCHEMA_VERSION
    assert doc["round_id"] == "2026-08-01"
    assert doc["config"]["seed"] == "s1"
    assert doc["note"] == "test run"
    assert [r["model"] for r in doc["leaderboard"]] == ["toto2", "seasonal_naive"]
    assert set(doc["leaderboard"][0]) == {"rank", "model", "crps_rel", "mase_rel", "crps", "mase"}
    # vs_baseline is trimmed to the tracker's fields — raw p-values stay out.
    assert set(doc["vs_baseline_crps"][0]) == set(publish_round._VS_BASELINE_KEYS)


def test_rebuild_index_orders_and_summarises(tmp_path):
    rounds = tmp_path / "rounds"
    rounds.mkdir()
    for rid in ("2026-08-02", "2026-08-01"):
        doc = publish_round.build_round(MERGED, rid)
        (rounds / f"{rid}.json").write_text(json.dumps(doc))
    index = publish_round.rebuild_index(rounds)
    assert [e["round_id"] for e in index["rounds"]] == ["2026-08-01", "2026-08-02"]
    entry = index["rounds"][0]
    assert entry["n_models"] == 2
    assert entry["n_tsfms"] == 1
    assert entry["top"] == ["toto2", "seasonal_naive"]
    assert entry["has_metrics"] is True
    assert json.loads((rounds / "index.json").read_text())["rounds"] == index["rounds"]


def test_rebuild_history_is_column_aligned(tmp_path):
    rounds = tmp_path / "rounds"
    rounds.mkdir()

    def write(rid, merged):
        (rounds / f"{rid}.json").write_text(json.dumps(publish_round.build_round(merged, rid)))

    # round A has both models; round B drops toto2 and adds a newcomer
    write("2026-08-01", MERGED)
    write("2026-08-02", {"leaderboard": [
        {"model": "seasonal_naive", "rank": 1, "crps_rel": 1.0, "mase_rel": 1.0,
         "crps": 0.8, "mase": 3.5},
        {"model": "newcomer", "rank": 2, "crps_rel": None, "mase_rel": None,
         "crps": None, "mase": None},
    ]})

    hist = publish_round.rebuild_history(rounds)
    assert hist["models"] == ["toto2", "seasonal_naive", "newcomer"]
    assert [r["round_id"] for r in hist["rounds"]] == ["2026-08-01", "2026-08-02"]
    # every per-round array is aligned with `models`, padded with None
    for r in hist["rounds"]:
        assert len(r["ranks"]) == len(hist["models"])
        assert len(r["crps_rel"]) == len(hist["models"])
    assert hist["rounds"][0]["ranks"] == [1, 2, None]
    assert hist["rounds"][1]["ranks"] == [None, 1, 2]
    assert hist["rounds"][1]["crps_rel"] == [None, 1.0, None]
    assert json.loads((rounds / "history.json").read_text())["models"] == hist["models"]


def test_derived_files_are_not_read_back_as_rounds(tmp_path):
    """index.json / history.json live beside the round files; rebuilding twice
    must not fold them into the history."""
    rounds = tmp_path / "rounds"
    rounds.mkdir()
    doc = publish_round.build_round(MERGED, "2026-08-01")
    (rounds / "2026-08-01.json").write_text(json.dumps(doc))
    publish_round.rebuild_index(rounds)
    publish_round.rebuild_history(rounds)
    hist = publish_round.rebuild_history(rounds)
    index = publish_round.rebuild_index(rounds)
    assert [r["round_id"] for r in hist["rounds"]] == ["2026-08-01"]
    assert [e["round_id"] for e in index["rounds"]] == ["2026-08-01"]


def test_committed_seed_rounds_match_contract():
    rounds_dir = REPO / "docs/data/rounds"
    paths = [p for p in rounds_dir.glob("*.json") if p.name not in publish_round._DERIVED]
    assert paths, "no committed rounds under docs/data/rounds/"
    models = json.loads((REPO / "docs/data/models.json").read_text())["models"]
    for p in paths:
        doc = json.loads(p.read_text())
        assert doc["round_id"] == p.stem
        assert doc["schema_version"] == publish_round.SCHEMA_VERSION
        ranks = [r["rank"] for r in doc["leaderboard"]]
        assert ranks == sorted(ranks) and len(set(ranks)) == len(ranks)
        for row in doc["leaderboard"]:
            assert row["model"] in models, f"{row['model']} missing from models.json"
    index = json.loads((rounds_dir / "index.json").read_text())
    assert [e["round_id"] for e in index["rounds"]] == sorted(p.stem for p in paths)
    hist = json.loads((rounds_dir / "history.json").read_text())
    assert [r["round_id"] for r in hist["rounds"]] == sorted(p.stem for p in paths)
    assert set(hist["models"]) <= set(models)


def test_cli_publish_and_index(tmp_path):
    merged_path = tmp_path / "merged.json"
    merged_path.write_text(json.dumps(MERGED))
    docs_data = tmp_path / "data"
    rc = publish_round.main([
        "--merged", str(merged_path), "--round-id", "2026-08-03",
        "--n-series", "43", "--docs-data", str(docs_data),
    ])
    assert rc == 0
    doc = json.loads((docs_data / "rounds/2026-08-03.json").read_text())
    assert doc["config"] == {"n_series": 43}
    index = json.loads((docs_data / "rounds/index.json").read_text())
    assert [e["round_id"] for e in index["rounds"]] == ["2026-08-03"]
