"""pool_report: composition of the eval pool, not of the catalog.

The whole point of the module is that these two answers differ, so the tests
pin the ways they differ: depth-ineligible sources, panel expansion inflating
one domain's series count, and the equal-weight draw flattening both.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from source_discovery import pool_report


def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df), path)


@pytest.fixture
def pool_tree(tmp_path: Path) -> tuple[Path, Path]:
    """A catalog whose pool shape is deliberately unlike its source counts.

      web_a/web_b/web_c : 3 sources, plain, deep enough      -> 3 series
      trn_a             : 1 source, 40-way panel             -> 40 series
      nat_short         : live but only 10 observations      -> 0 series
      off_a             : disabled, deep enough              -> 0 series
    """
    catalog = [
        {"id": "web_a", "domain": "web_cloudops", "dgp_class": "req_rate",
         "frequency": "PT1H"},
        {"id": "web_b", "domain": "web_cloudops", "dgp_class": "req_rate",
         "frequency": "PT1H"},
        {"id": "web_c", "domain": "web_cloudops", "dgp_class": "req_rate",
         "frequency": "P1D"},
        {"id": "trn_a", "domain": "transport", "dgp_class": "vehicle_position",
         "frequency": "PT1M"},
        {"id": "nat_short", "domain": "nature", "dgp_class": "weather_field",
         "frequency": "PT1H"},
        {"id": "off_a", "domain": "energy", "dgp_class": "grid_load",
         "frequency": "PT1H", "disabled": True},
    ]
    src_dir = tmp_path / "src" / "sources"
    src_dir.mkdir(parents=True)
    catalog_path = src_dir / "sources.yaml"
    catalog_path.write_text(yaml.safe_dump(catalog))
    data_dir = src_dir / "data"

    def plain(sid: str, n: int) -> None:
        _write(data_dir / sid / "2026-01-01.parquet", pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
            "value": np.arange(float(n)),
        }))

    for sid in ("web_a", "web_b", "web_c", "off_a"):
        plain(sid, 300)
    plain("nat_short", 10)
    _write(data_dir / "trn_a" / "2026-01-01.parquet", pd.DataFrame({
        "timestamp": np.repeat(
            pd.date_range("2026-01-01", periods=200, freq="min"), 40),
        "value": np.arange(8000.0),
        "_panel_vehicle": [f"v{i}" for i in range(40)] * 200,
    }))
    return catalog_path, data_dir


def test_pool_is_not_the_catalog(pool_tree):
    catalog_path, data_dir = pool_tree
    rep = pool_report.build(catalog_path, data_dir, sample_n=70)
    t = rep["totals"]

    assert t["catalog_entries"] == 6
    assert t["live_sources"] == 5, "the disabled source is not live"
    assert t["eligible_sources"] == 4, "nat_short is short of min_series_length"
    assert t["ineligible_on_depth"] == 1
    assert rep["ineligible_on_depth_by_domain"] == {"nature": 1}

    # Sources say web_cloudops leads 3:1. Series say transport leads 40:3.
    assert rep["eligible_sources_by_domain"] == {"web_cloudops": 3, "transport": 1}
    assert rep["eligible_series_by_domain"] == {"transport": 40, "web_cloudops": 3}


def test_disabled_sources_never_enter_the_pool(pool_tree):
    catalog_path, data_dir = pool_tree
    rep = pool_report.build(catalog_path, data_dir, sample_n=70)
    assert "energy" not in rep["eligible_sources_by_domain"]
    assert "energy" not in rep["eligible_series_by_domain"]


def test_sampled_mix_is_levelled_across_domains(pool_tree):
    """Transport holds 40 of 43 eligible series. The equal-weight draw must
    still split the sample evenly, which is the property the eval relies on."""
    catalog_path, data_dir = pool_tree
    rep = pool_report.build(catalog_path, data_dir, sample_n=70)
    sampled = rep["sampled_by_domain"]
    assert rep["sample_error"] is None
    assert set(sampled) == {"web_cloudops", "transport"}
    assert sampled["web_cloudops"] == sampled["transport"] == 35


def test_cadence_grid_counts_sources_per_band(pool_tree):
    catalog_path, data_dir = pool_tree
    grid = pool_report.build(pool_tree[0], pool_tree[1],
                             sample_n=70)["eligible_sources_by_domain_cadence"]
    assert grid["web_cloudops"] == {"hourly": 2, "daily": 1}
    assert grid["transport"] == {"sub-min": 1}
    assert "nature" not in grid, "depth-ineligible sources are not in the grid"


def test_summary_line_is_greppable(pool_tree):
    rep = pool_report.build(pool_tree[0], pool_tree[1], sample_n=70)
    line = pool_report.summarize(rep)
    assert line.startswith("pool ")
    assert "4/5 sources eligible" in line
    assert "43 series" in line
    assert "1 short of 128 obs" in line
    assert "\n" not in line


def test_render_table_has_a_row_per_domain_and_totals(pool_tree):
    rep = pool_report.build(pool_tree[0], pool_tree[1], sample_n=70)
    table = pool_report.render_table(rep)
    lines = table.splitlines()
    assert lines[0].startswith("domain")
    assert [ln.split()[0] for ln in lines[1:]] == ["transport", "web_cloudops",
                                                   "TOTAL"]
    assert lines[-1].split()[-1] == "4"


def test_seed_is_fixed_so_daily_drift_is_real(pool_tree):
    """A moving seed would make sampling noise look like composition drift."""
    a = pool_report.build(pool_tree[0], pool_tree[1], sample_n=70)
    b = pool_report.build(pool_tree[0], pool_tree[1], sample_n=70)
    assert a["sampled_by_domain"] == b["sampled_by_domain"]
    assert a["sampled_by_cadence"] == b["sampled_by_cadence"]


def test_empty_pool_does_not_crash(tmp_path):
    src_dir = tmp_path / "src" / "sources"
    src_dir.mkdir(parents=True)
    (src_dir / "sources.yaml").write_text(yaml.safe_dump([
        {"id": "x", "domain": "nature", "dgp_class": "y", "frequency": "PT1H"},
    ]))
    (src_dir / "data").mkdir()
    rep = pool_report.build(src_dir / "sources.yaml", src_dir / "data")
    assert rep["totals"]["eligible_sources"] == 0
    assert rep["sampled_by_domain"] == {}
    assert pool_report.render_table(rep) == "(no eligible sources)"
    assert "0/1 sources eligible" in pool_report.summarize(rep)
