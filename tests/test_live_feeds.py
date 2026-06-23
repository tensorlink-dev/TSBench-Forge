"""Tests for the real public-feed adapters — all offline via an injected fetch.

A network-gated smoke test (real URLs) is included but skipped unless
``TSBENCH_LIVE_TESTS=1``, so the default suite stays hermetic.
"""

from __future__ import annotations

import os
from collections import Counter
from datetime import UTC

import numpy as np
import pytest

from feeds import AsOfLiveSource
from ingest import FreshBuffer
from live_feeds import (
    REGISTRY,
    CsvFeed,
    DatedCsvFeed,
    build_real_live_source,
    cached_fetch,
    make_feed,
    parse_csv_column,
)

# A tiny dated CSV with a header, used by most tests.
CSV = "Date,Temp\n" + "\n".join(
    f"19{81 + i // 365:02d}-{(i % 12) + 1:02d}-{(i % 27) + 1:02d},{10.0 + (i % 13)}"
    for i in range(600)
)


def _fetch_const(_url: str) -> str:
    return CSV


def test_parse_by_name_and_index_agree():
    by_name, _ = parse_csv_column(CSV, "Temp")
    by_index, _ = parse_csv_column(CSV, 1)
    assert by_name.size == 600
    assert np.array_equal(by_name, by_index)


def test_parse_skips_nonnumeric_rows():
    text = "d,v\n2020-01-01,1.0\n2020-01-02,oops\n2020-01-03,3.0\n"
    vals, _ = parse_csv_column(text, "v")
    assert np.array_equal(vals, np.array([1.0, 3.0]))


def test_csvfeed_windows_are_zscored_and_right_length():
    feed = CsvFeed(url="x", domain="climate", value_column="Temp", fetch=_fetch_const)
    motifs = feed.pull(4, 256, np.random.default_rng(0))
    assert len(motifs) == 4
    for m in motifs:
        assert m.shape == (256,)
        assert abs(float(m.mean())) < 1e-6  # centered
        assert abs(float(m.std()) - 1.0) < 1e-6  # unit scale


def test_csvfeed_is_deterministic_under_seed():
    feed = CsvFeed(url="x", value_column="Temp", fetch=_fetch_const)
    a = feed.pull(3, 200, np.random.default_rng(7))
    b = feed.pull(3, 200, np.random.default_rng(7))
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_csvfeed_raises_when_series_too_short():
    feed = CsvFeed(url="x", value_column="Temp", fetch=_fetch_const)
    with pytest.raises(ValueError):
        feed.pull(1, 10_000, np.random.default_rng(0))


def test_csvfeed_fetches_once_and_caches_in_memory():
    calls = {"n": 0}

    def counting_fetch(_url: str) -> str:
        calls["n"] += 1
        return CSV

    feed = CsvFeed(url="x", value_column="Temp", fetch=counting_fetch)
    feed.pull(2, 100, np.random.default_rng(0))
    feed.pull(2, 100, np.random.default_rng(1))
    assert calls["n"] == 1  # parsed series is memoised after the first pull


def test_dated_feed_stamps_window_end_time():
    feed = DatedCsvFeed(
        url="x", domain="climate", value_column="Temp", date_column="Date", fetch=_fetch_const
    )
    out = feed.pull_dated(5, 200, np.random.default_rng(0))
    assert len(out) == 5
    for motif, domain, ts in out:
        assert motif.shape == (200,)
        assert domain == "climate"
        assert ts > 0.0


def test_as_of_gate_admits_only_post_commit_windows():
    from datetime import datetime

    feed = DatedCsvFeed(url="x", value_column="Temp", date_column="Date", fetch=_fetch_const)
    commit = datetime(1982, 1, 1, tzinfo=UTC).timestamp()
    gated = AsOfLiveSource(inner=feed, commit_time=commit)
    out = gated.pull_dated(6, 200, np.random.default_rng(0))
    assert out  # the feed runs to ~1982, so some windows end after the commit
    assert all(ts > commit for _, _, ts in out)


def _fetch_for_registry(url: str) -> str:
    """Return a CSV whose header matches whichever registry feed asked for ``url``."""
    spec = next(s for s in REGISTRY.values() if s.url == url)
    col = spec.value_column
    rows = "\n".join(f"2020-01-{(i % 27) + 1:02d},{10.0 + (i % 13)}" for i in range(600))
    return f"Date,{col}\n{rows}"


def test_build_real_live_source_is_multidomain_and_seeded():
    src = build_real_live_source(fetch=_fetch_for_registry)
    buf = FreshBuffer(src, pool_size=40, motif_len=256)
    buf.refresh(np.random.default_rng(0xC0FFEE))
    # Every registry domain is reachable; the constant fetch gives them all the
    # same series but distinct labels, so coverage wiring is exercised.
    assert set(buf.pool_domains) == {REGISTRY[n].domain for n in REGISTRY}
    counts = Counter(buf.pool_domains)
    assert sum(counts.values()) == 40


def test_make_feed_unknown_name_raises():
    with pytest.raises(KeyError):
        make_feed("not-a-feed", fetch=_fetch_const)


def test_cached_fetch_writes_then_reads(tmp_path):
    calls = {"n": 0}

    def inner(url: str) -> str:
        calls["n"] += 1
        return f"body-of-{url}"

    fetch = cached_fetch(cache_dir=str(tmp_path), inner=inner)
    assert fetch("http://example/a") == "body-of-http://example/a"
    assert fetch("http://example/a") == "body-of-http://example/a"  # served from disk
    assert calls["n"] == 1
    assert fetch("http://example/b") == "body-of-http://example/b"
    assert calls["n"] == 2


@pytest.mark.skipif(
    os.environ.get("TSBENCH_LIVE_TESTS") != "1",
    reason="network smoke test; set TSBENCH_LIVE_TESTS=1 to run",
)
def test_live_network_smoke():  # pragma: no cover - network
    src = build_real_live_source()
    motifs = src.pull(5, 304, np.random.default_rng(0))
    assert len(motifs) == 5
    assert all(np.isfinite(m).all() for m in motifs)
