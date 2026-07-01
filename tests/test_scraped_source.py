"""ScrapedLiveSource: parquet → FreshBuffer with DGP-class + cadence labels.

Uses a fixture parquet tree so the test is self-contained (no network, no real
scraper run). Verifies:
  - Motifs get pulled with the expected shape and length
  - Each MotifMeta carries domain + dgp_class + cadence + source_id from the catalog
  - Equal-weight sampling doesn't over-represent nature just because it has
    more sources in the fixture (the very property that motivated pool_sampler)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingest import FreshBuffer, MotifMeta
from scraped_source import ScrapedLiveSource


@pytest.fixture
def catalog_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Build a minimal (sources.yaml, data_dir) pair the adapter can walk.

    Two domains, three sources, one panel-expanded — deliberately unbalanced so
    the equal-weight sampler has something to enforce against.
    """
    catalog = [
        {"id": "src_nature_a",  "domain": "nature",     "dgp_class": "weather_field",
         "frequency": "PT1H"},
        {"id": "src_nature_b",  "domain": "nature",     "dgp_class": "weather_field",
         "frequency": "PT1H"},
        {"id": "src_health_a",  "domain": "healthcare", "dgp_class": "vital_statistics",
         "frequency": "P1Y"},
    ]
    yaml_path = tmp_path / "sources.yaml"
    yaml_path.write_text(yaml.safe_dump(catalog))

    data_dir = tmp_path / "data"
    for entry in catalog:
        sid = entry["id"]
        d = data_dir / sid
        d.mkdir(parents=True)
        rng = np.random.default_rng(hash(sid) & 0xFFFF)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=500, freq="h"),
            "value": rng.normal(0, 1, size=500).cumsum(),
        })
        pq.write_table(pa.Table.from_pandas(df), d / "2026-07-01.parquet")
    return yaml_path, data_dir


def test_pull_meta_returns_labeled_motifs(catalog_tree):
    catalog_path, data_dir = catalog_tree
    src = ScrapedLiveSource(catalog_path, data_dir, min_series_length=128)
    rng = np.random.default_rng(0)
    metas = src.pull_meta(n=20, length=64, rng=rng)
    assert len(metas) == 20
    for m in metas:
        assert isinstance(m, MotifMeta)
        assert m.motif.shape == (64,)
        assert np.all(np.isfinite(m.motif))
        assert m.domain in {"nature", "healthcare"}
        assert m.dgp_class in {"weather_field", "vital_statistics"}
        assert m.cadence in {"hourly", "yearly"}
        assert m.source_id in {"src_nature_a", "src_nature_b", "src_health_a"}


def test_equal_weight_prevents_source_count_domination(catalog_tree):
    """With 2 nature sources : 1 healthcare source, a source-count-weighted
    sampler would give nature ~2× the share. Equal-weight per DGP class per
    domain must give both domains equal share."""
    catalog_path, data_dir = catalog_tree
    src = ScrapedLiveSource(catalog_path, data_dir, min_series_length=128)
    metas = src.pull_meta(n=200, length=64, rng=np.random.default_rng(42))
    domain_counts = Counter(m.domain for m in metas)
    # Not source-count-weighted (which would be ~133/67); equal-weight gives ~100/100.
    assert 80 <= domain_counts["nature"] <= 120
    assert 80 <= domain_counts["healthcare"] <= 120


def test_buffer_exposes_dgp_classes_and_cadences(catalog_tree):
    catalog_path, data_dir = catalog_tree
    src = ScrapedLiveSource(catalog_path, data_dir, min_series_length=128)
    buffer = FreshBuffer(src, pool_size=40, motif_len=128)
    buffer.refresh(np.random.default_rng(7))
    assert len(buffer.pool_domains) == 40
    assert len(buffer.pool_dgp_classes) == 40
    assert len(buffer.pool_cadences) == 40
    # Every motif has a real (non-None) DGP class + cadence — the reward-hacking
    # breadth gates need these to activate.
    assert all(c is not None for c in buffer.pool_dgp_classes)
    assert all(c is not None for c in buffer.pool_cadences)


def test_no_data_raises_clearly(tmp_path):
    """A catalog pointing at an empty data dir gives a clear error rather than
    a mysterious empty-pool downstream."""
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump([
        {"id": "src_x", "domain": "nature", "dgp_class": "x", "frequency": "PT1H"},
    ]))
    (tmp_path / "data").mkdir()
    src = ScrapedLiveSource(tmp_path / "sources.yaml", tmp_path / "data")
    with pytest.raises(RuntimeError, match="no series available"):
        src.pull_meta(n=4, length=32, rng=np.random.default_rng(0))
