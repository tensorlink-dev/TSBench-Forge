"""ScrapedLiveSource: parquet → FreshBuffer with DGP-class + cadence labels.

Uses a fixture parquet tree so the test is self-contained (no network, no real
scraper run). Verifies:
  - Motifs get pulled with the expected shape and length
  - Each MotifMeta carries domain + dgp_class + cadence + source_id from the catalog
  - Equal-weight sampling doesn't over-represent nature just because it has
    more sources in the fixture (the equal-weight-per-domain-per-class property)
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


def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df), path)


def _shape_via_full_read(src: ScrapedLiveSource, paths):
    """What _series_shape must agree with: the old full-frame index path."""
    df = src._read_series_frame(paths)
    if df is None or df.empty:
        return 0, [], None
    return len(df), [c for c in df.columns if c.startswith("_panel_")], df


def test_series_shape_matches_a_full_read_across_schema_drift(tmp_path):
    """Sources gain panel keys mid-life: citibike ran 16 days before it emitted
    _panel_station_id. Reading only the newest file's schema off every file
    drops the earlier days entirely, under-counting the series."""
    d = tmp_path / "data" / "src_drift"
    stamps = pd.date_range("2026-01-01", periods=60, freq="h")
    _write(d / "2026-01-01.parquet",
           pd.DataFrame({"timestamp": stamps, "value": np.arange(60.0)}))
    _write(d / "2026-01-02.parquet",
           pd.DataFrame({"timestamp": stamps.repeat(2),
                         "value": np.arange(120.0),
                         "_panel_station_id": ["a", "b"] * 60}))
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump([
        {"id": "src_drift", "domain": "nature", "dgp_class": "x",
         "frequency": "PT1H"},
    ]))
    src = ScrapedLiveSource(tmp_path / "sources.yaml", tmp_path / "data")
    paths = sorted(d.glob("*.parquet"))

    n, panel_cols, panel_df = src._series_shape(paths)
    n_old, panel_old, df_old = _shape_via_full_read(src, paths)

    assert (n, panel_cols) == (n_old, panel_old)
    assert n == 180, "the pre-panel day's 60 rows must still be counted"
    groups = panel_df.groupby(panel_cols).size().sort_index()
    assert groups.equals(df_old.groupby(panel_old).size().sort_index())


def test_series_shape_reads_no_value_column(tmp_path, monkeypatch):
    """The index needs row counts and panel keys, never a measured value.
    Loading values here is what made a catalog walk cost the whole corpus."""
    d = tmp_path / "data" / "src_wide"
    _write(d / "2026-01-01.parquet", pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=40, freq="h"),
        "value": np.arange(40.0),
        "other": np.arange(40.0),
        "_panel_station_id": ["a", "b"] * 20,
    }))
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump([
        {"id": "src_wide", "domain": "nature", "dgp_class": "x",
         "frequency": "PT1H"},
    ]))
    src = ScrapedLiveSource(tmp_path / "sources.yaml", tmp_path / "data")

    seen: list[list[str] | None] = []
    real_read = pq.ParquetFile.read

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("columns"))
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read", spy)
    n, panel_cols, _ = src._series_shape(sorted(d.glob("*.parquet")))

    assert n == 40 and panel_cols == ["_panel_station_id"]
    assert seen, "expected a narrow read for the panel keys"
    for cols in seen:
        assert cols is not None, "a full read defeats the point"
        assert set(cols) <= {"timestamp", "_panel_station_id"}


def test_series_shape_uses_metadata_when_there_is_nothing_to_dedup(tmp_path,
                                                                   monkeypatch):
    """One file, no panel: parquet's footer already carries the row count, so
    the index should not touch the data at all."""
    d = tmp_path / "data" / "src_plain"
    _write(d / "2026-01-01.parquet", pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=250, freq="h"),
        "value": np.arange(250.0),
    }))
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump([
        {"id": "src_plain", "domain": "nature", "dgp_class": "x",
         "frequency": "PT1H"},
    ]))
    src = ScrapedLiveSource(tmp_path / "sources.yaml", tmp_path / "data")

    def boom(*args, **kwargs):
        raise AssertionError("row count must come from metadata, not a read")

    monkeypatch.setattr(pq.ParquetFile, "read", boom)
    monkeypatch.setattr(pq, "read_table", boom)
    assert src._series_shape(sorted(d.glob("*.parquet"))) == (250, [], None)


def test_frame_cache_is_bounded(tmp_path):
    """An unbounded frame cache is what OOM-killed a full catalog walk: the
    largest sources run to millions of rows and nothing was ever evicted."""
    catalog = []
    data_dir = tmp_path / "data"
    for i in range(6):
        sid = f"src_{i}"
        catalog.append({"id": sid, "domain": "nature", "dgp_class": "x",
                        "frequency": "PT1H"})
        _write(data_dir / sid / "2026-01-01.parquet", pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=200, freq="h"),
            "value": np.arange(200.0),
        }))
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump(catalog))
    src = ScrapedLiveSource(tmp_path / "sources.yaml", data_dir,
                            min_series_length=128, frame_cache_size=2)
    for i in range(6):
        src._read_series_frame([data_dir / f"src_{i}" / "2026-01-01.parquet"])
    assert len(src.__dict__["_frame_cache"]) == 2
    # The two most recent survive; the earliest were evicted.
    assert [k[0].split("/")[-2] for k in src.__dict__["_frame_cache"]] == \
        ["src_4", "src_5"]


def test_frame_cache_honours_its_byte_ceiling(tmp_path):
    """A count-only cap still admits N giants. The byte bound is the one that
    keeps a pull inside memory when the frames are millions of rows."""
    catalog = []
    data_dir = tmp_path / "data"
    for i in range(4):
        sid = f"src_{i}"
        catalog.append({"id": sid, "domain": "nature", "dgp_class": "x",
                        "frequency": "PT1H"})
        _write(data_dir / sid / "2026-01-01.parquet", pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=5000, freq="h"),
            "value": np.arange(5000.0),
        }))
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump(catalog))
    # Generous count, tiny byte budget: eviction must come from the bytes.
    src = ScrapedLiveSource(tmp_path / "sources.yaml", data_dir,
                            min_series_length=128, frame_cache_size=100,
                            frame_cache_bytes=1)
    for i in range(4):
        src._read_series_frame([data_dir / f"src_{i}" / "2026-01-01.parquet"])
    cache = src.__dict__["_frame_cache"]
    assert len(cache) == 1, "byte ceiling should hold only the newest frame"
    assert next(iter(cache))[0].endswith("src_3/2026-01-01.parquet")
    # A frame is still returned even when it alone exceeds the budget.
    df = src._read_series_frame([data_dir / "src_0" / "2026-01-01.parquet"])
    assert df is not None and len(df) == 5000


def test_frame_cache_still_serves_repeat_reads(tmp_path):
    """Panel rows of one source share a path tuple, so the cache must actually
    hit — bounding it must not turn every motif into a re-read."""
    data_dir = tmp_path / "data"
    _write(data_dir / "src_a" / "2026-01-01.parquet", pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=300, freq="h"),
        "value": np.arange(300.0),
    }))
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump([
        {"id": "src_a", "domain": "nature", "dgp_class": "x",
         "frequency": "PT1H"},
    ]))
    src = ScrapedLiveSource(tmp_path / "sources.yaml", data_dir,
                            min_series_length=128)
    paths = [data_dir / "src_a" / "2026-01-01.parquet"]
    first = src._read_series_frame(paths)
    assert src._read_series_frame(paths) is first, "second read should be cached"


def test_indexing_does_not_populate_the_frame_cache(catalog_tree):
    """Indexing must stay off the full-read path entirely — that coupling is
    what made the cost of a catalog walk scale with corpus size."""
    catalog_path, data_dir = catalog_tree
    src = ScrapedLiveSource(catalog_path, data_dir, min_series_length=128)
    series = src._catalog()
    assert series, "fixture should yield series"
    assert not src.__dict__.get("_frame_cache")


def test_disabled_sources_are_excluded_from_the_pool(tmp_path):
    """A disabled source keeps its parquet on disk, so nothing else stops the
    eval serving windows from a feed the pipeline retired. Every other consumer
    of the catalog honours the flag; this one must too."""
    data_dir = tmp_path / "data"
    for sid in ("live_one", "retired_one"):
        _write(data_dir / sid / "2026-01-01.parquet", pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=300, freq="h"),
            "value": np.arange(300.0),
        }))
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump([
        {"id": "live_one", "domain": "nature", "dgp_class": "x",
         "frequency": "PT1H"},
        {"id": "retired_one", "domain": "nature", "dgp_class": "x",
         "frequency": "PT1H", "disabled": True,
         "disabled_reason": "stale past 3x threshold"},
    ]))
    src = ScrapedLiveSource(tmp_path / "sources.yaml", data_dir,
                            min_series_length=128)
    assert {s["source_id"] for s in src._catalog()} == {"live_one"}


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


# ── honest windows: never pad a short series into a long one ────────────────
#
# _extract_motif used to tile-pad — repeat the longest contiguous segment until
# it filled the requested length — on two separate paths. Measured 2026-08-18
# that fired for 15.8% of draws, in one case repeating 16 real points 20 times.
# A tiled window is exactly periodic, so any model that finds the period scores
# perfectly on data that never happened.


def _one_source(tmp_path: Path, values, *, sid="s", domain="nature",
                freq="PT1H", gap_after=None, periods=None) -> tuple[Path, Path]:
    """A catalog of one source whose single series is ``values``.

    ``gap_after`` inserts a jump far larger than the sampling interval at that
    index, so the 8x-median segmentation splits there.
    """
    n = len(values)
    ts = list(pd.date_range("2026-01-01", periods=n, freq="h"))
    if gap_after is not None:
        shift = pd.Timedelta(days=30)
        ts = ts[:gap_after] + [t + shift for t in ts[gap_after:]]
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump([
        {"id": sid, "domain": domain, "dgp_class": "x", "frequency": freq},
    ]))
    d = tmp_path / "data" / sid
    d.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame({"timestamp": ts, "value": values})),
        d / "2026-07-01.parquet")
    return tmp_path / "sources.yaml", tmp_path / "data"


def _period(a: np.ndarray) -> int | None:
    """Smallest exact repeating period, or None. Detects tile-padding."""
    for p in range(1, len(a) // 2 + 1):
        if np.array_equal(a[p:], a[:-p]):
            return p
    return None


def test_a_short_segment_is_served_short_rather_than_padded(tmp_path):
    rng = np.random.default_rng(0)
    vals = rng.normal(size=200).cumsum()
    src = ScrapedLiveSource(*_one_source(tmp_path, vals))
    metas = src.pull_meta(3, 320, np.random.default_rng(1))

    assert len(metas) == 3, "the pool is still filled"
    for m in metas:
        assert len(m.motif) == 200, "served the real segment, not 320 padded"
        assert _period(m.motif) is None, "no repetition"
        assert m.ts is not None, "a real window carries real timestamps"


def test_no_motif_is_ever_tile_padded(tmp_path):
    """The regression in one line: 16 real points must never become 320."""
    src = ScrapedLiveSource(*_one_source(
        tmp_path, np.random.default_rng(2).normal(size=140).cumsum()))
    for m in src.pull_meta(5, 320, np.random.default_rng(3)):
        assert len(m.motif) == 140
        assert _period(m.motif) is None


def test_the_segment_is_measured_after_gap_splitting(tmp_path):
    """300 rows but a month-long hole at 150: the honest window is 150, not 300
    and certainly not 320. Straddling the gap is a fake level shift."""
    vals = np.random.default_rng(4).normal(size=300).cumsum()
    src = ScrapedLiveSource(*_one_source(tmp_path, vals, gap_after=150))
    m = src.pull_meta(1, 320, np.random.default_rng(5))[0]
    assert len(m.motif) == 150


def test_a_fragmented_series_is_refused_and_replaced(tmp_path):
    """The case the floor exists for, and it is NOT "too few rows".

    Eligibility counts total observations; extraction needs CONTIGUOUS ones. A
    series with 300 rows broken into 60-point chunks sails past
    min_series_length and then has no honest 128-point window -- which is how
    tile-padding got reached in production. It must leave the pool, and the
    sampler must replace it so the pool size is unchanged.
    """
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump([
        {"id": "gappy", "domain": "nature", "dgp_class": "x", "frequency": "PT1H"},
        {"id": "good",  "domain": "nature", "dgp_class": "x", "frequency": "PT1H"},
    ]))
    # 300 rows, a month-long hole every 60 -> longest contiguous run is 60.
    ts = [pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i)
          + pd.Timedelta(days=30 * (i // 60)) for i in range(300)]
    for sid, stamps, n in (("gappy", ts, 300),
                           ("good", list(pd.date_range("2026-01-01", periods=400,
                                                       freq="h")), 400)):
        d = tmp_path / "data" / sid
        d.mkdir(parents=True)
        pq.write_table(pa.Table.from_pandas(pd.DataFrame({
            "timestamp": stamps,
            "value": np.random.default_rng(6).normal(size=n).cumsum(),
        })), d / "2026-07-01.parquet")

    src = ScrapedLiveSource(tmp_path / "sources.yaml", tmp_path / "data")
    assert {c["source_id"] for c in src._catalog()} == {"gappy", "good"}, (
        "both pass the row-count index -- that is the whole problem")

    metas = src.pull_meta(8, 320, np.random.default_rng(7))
    assert len(metas) == 8, "refusals are replaced, not silently dropped"
    assert {m.source_id for m in metas} == {"good"}
    assert any("floor" in r for r in src._rejected.values())


def test_the_floor_never_exceeds_what_was_asked_for(tmp_path):
    """A caller wanting 32 points must not be refused a 50-point series just
    because 50 is under the 128 floor -- the floor guards against serving a
    stub in place of the request, and here it IS the request."""
    cat, data = _one_source(
        tmp_path, np.random.default_rng(8).normal(size=50).cumsum())
    src = ScrapedLiveSource(cat, data, min_series_length=32)
    metas = src.pull_meta(2, 32, np.random.default_rng(9))
    assert len(metas) == 2 and all(len(m.motif) == 32 for m in metas)


def test_a_constant_window_is_rejected(tmp_path):
    """Zero variance is a free win for predict-the-last-value. Measured
    2026-08-18: 2.5% of draws."""
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump([
        {"id": "flat", "domain": "nature", "dgp_class": "x", "frequency": "PT1H"},
        {"id": "live", "domain": "nature", "dgp_class": "x", "frequency": "PT1H"},
    ]))
    for sid, vals in (("flat", np.full(400, 7.0)),
                      ("live", np.random.default_rng(10).normal(size=400).cumsum())):
        d = tmp_path / "data" / sid
        d.mkdir(parents=True)
        pq.write_table(pa.Table.from_pandas(pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=400, freq="h"),
            "value": vals,
        })), d / "2026-07-01.parquet")

    src = ScrapedLiveSource(tmp_path / "sources.yaml", tmp_path / "data")
    metas = src.pull_meta(6, 320, np.random.default_rng(11))
    assert {m.source_id for m in metas} == {"live"}
    assert any("constant" in r for r in src._rejected.values())
    assert "constant" in " ".join(src.rejection_summary())


def test_a_mostly_flat_window_is_kept(tmp_path):
    """A step function sitting on one value 95% of the time is real data about
    a real process. Rejecting it needs the fuller battery in quality.py, not
    this path -- so it must survive here."""
    # The step sits at 200 so that EVERY 320-window out of 400 contains it;
    # putting it at 380 would leave most windows flat and the test would be
    # asserting on where the sampler happened to land.
    vals = np.full(400, 3.0)
    vals[200:] = 9.0
    src = ScrapedLiveSource(*_one_source(tmp_path, vals))
    assert len(src.pull_meta(2, 320, np.random.default_rng(12))) == 2


def test_a_pool_that_runs_dry_returns_short_and_says_so(tmp_path):
    """Bounded redraw: an all-unusable pool must surface as a short return the
    caller can see, not an infinite loop and not a fabricated window."""
    src = ScrapedLiveSource(*_one_source(tmp_path, np.full(400, 1.0)))
    with pytest.warns(RuntimeWarning, match="served 0"):
        metas = src.pull_meta(4, 320, np.random.default_rng(13))
    assert metas == []
    assert src.rejection_summary()
