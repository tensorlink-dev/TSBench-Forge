"""Tests for the daily JSON-feed adapters — all offline via an injected fetch.

Mirrors ``test_live_feeds.py``: the path engine, timestamp coercion, the dated
adapter, as-of gating, and the multi-domain builder are all exercised hermetically.
A network smoke test against the real catalogue runs only when
``TSBENCH_LIVE_TESTS=1``.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import UTC, datetime

import numpy as np
import pytest

from daily_feeds import (
    DAILY_REGISTRY,
    DailyFeedSpec,
    DatedJsonFeed,
    _to_posix,
    _walk,
    build_daily_live_source,
    make_daily_feed,
    parse_json_series,
)
from feeds import AsOfLiveSource
from ingest import FreshBuffer

# --------------------------------------------------------------------------- #
# Path engine + timestamp coercion
# --------------------------------------------------------------------------- #


def test_walk_key_index_and_iteration():
    obj = {"a": {"b": [{"v": 1}, {"v": 2}, {"v": 3}]}}
    assert _walk(obj, "a.b[].v") == [1, 2, 3]
    assert _walk(obj, "a.b[0].v") == 1
    assert _walk(obj, "a.b[-1].v") == 3
    assert _walk([{"date": 10}, {"date": 20}], "[].date") == [10, 20]
    # nested explicit index then iterate (the USGS shape)
    nested = {"value": {"timeSeries": [{"values": [{"value": [{"x": 5}, {"x": 6}]}]}]}}
    assert _walk(nested, "value.timeSeries[0].values[0].value[].x") == [5, 6]


def test_walk_missing_paths_return_none():
    assert _walk({"a": 1}, "a.b.c") is None
    assert _walk({"a": [1, 2]}, "a[5]") is None
    assert _walk(None, "a") is None


def test_to_posix_handles_epoch_and_iso():
    # epoch seconds round-trips
    assert _to_posix(1_577_836_800) == pytest.approx(1_577_836_800.0)
    # epoch milliseconds is divided down to seconds
    assert _to_posix(1_577_836_800_000) == pytest.approx(1_577_836_800.0)
    # plain date string, interpreted as UTC midnight
    assert _to_posix("2020-01-01") == pytest.approx(
        datetime(2020, 1, 1, tzinfo=UTC).timestamp()
    )
    # full ISO with offset
    assert _to_posix("2020-01-01T00:00:00.000-00:00") == pytest.approx(
        datetime(2020, 1, 1, tzinfo=UTC).timestamp()
    )
    # garbage -> None
    assert _to_posix("not-a-date") is None
    assert _to_posix(None) is None


def test_parse_json_series_aligns_sorts_and_skips_bad_rows():
    payload = json.dumps(
        {
            "data": [
                {"date": "2020-01-03", "v": 3.0},
                {"date": "2020-01-01", "v": 1.0},
                {"date": "2020-01-02", "v": None},  # non-numeric value -> dropped
                {"date": "bad", "v": 9.0},  # unparseable date -> dropped
                {"date": "2020-01-04", "v": 4.0},
            ]
        }
    )
    vals, times = parse_json_series(payload, "data[].date", "data[].v")
    # sorted ascending by time, the two bad rows dropped
    assert np.array_equal(vals, np.array([1.0, 3.0, 4.0]))
    assert np.all(np.diff(times) > 0)


def test_parse_json_series_raises_on_non_list_paths():
    with pytest.raises(ValueError):
        parse_json_series(json.dumps({"a": 1}), "a", "a")


# --------------------------------------------------------------------------- #
# DatedJsonFeed adapter
# --------------------------------------------------------------------------- #

_SERIES = json.dumps(
    {
        "data": [
            {"date": f"2020-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}", "v": 10.0 + (i % 13)}
            for i in range(600)
        ]
    }
)


def _fetch_const(_url: str) -> str:
    return _SERIES


def _feed(**kw) -> DatedJsonFeed:
    kw.setdefault("fetch", _fetch_const)
    return DatedJsonFeed(
        url="x", domain="d", timestamp_field="data[].date", value_field="data[].v", **kw,
    )


def test_feed_windows_are_zscored_and_right_length():
    motifs = _feed().pull(4, 256, np.random.default_rng(0))
    assert len(motifs) == 4
    for m in motifs:
        assert m.shape == (256,)
        assert abs(float(m.mean())) < 1e-6
        assert abs(float(m.std()) - 1.0) < 1e-6


def test_feed_is_deterministic_under_seed():
    a = _feed().pull(3, 200, np.random.default_rng(7))
    b = _feed().pull(3, 200, np.random.default_rng(7))
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_feed_fetches_once_and_caches_in_memory():
    calls = {"n": 0}

    def counting_fetch(_url: str) -> str:
        calls["n"] += 1
        return _SERIES

    feed = _feed(fetch=counting_fetch)
    feed.pull(2, 100, np.random.default_rng(0))
    feed.pull(2, 100, np.random.default_rng(1))
    assert calls["n"] == 1


def test_feed_raises_when_series_too_short():
    with pytest.raises(ValueError):
        _feed().pull(1, 10_000, np.random.default_rng(0))


def test_dated_feed_stamps_window_end_time():
    out = _feed().pull_dated(5, 200, np.random.default_rng(0))
    assert len(out) == 5
    for motif, domain, ts in out:
        assert motif.shape == (200,)
        assert domain == "d"
        assert ts > 0.0


def test_as_of_gate_admits_only_post_commit_windows():
    feed = _feed()
    commit = datetime(2020, 6, 1, tzinfo=UTC).timestamp()
    gated = AsOfLiveSource(inner=feed, commit_time=commit)
    out = gated.pull_dated(6, 200, np.random.default_rng(0))
    assert out
    assert all(ts > commit for _, _, ts in out)


# --------------------------------------------------------------------------- #
# Registry + multi-domain builder
# --------------------------------------------------------------------------- #


def _payload_for_spec(spec: DailyFeedSpec, n: int = 400) -> str:
    """Build a JSON body matching ``spec``'s path shape, for offline builder tests."""
    dates = [f"2020-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}" for i in range(n)]
    vals = [float(10 + (i % 13)) for i in range(n)]
    ts, vf = spec.timestamp_field, spec.value_field
    if ts == "[].date":  # DefiLlama: top-level array, epoch-second dates
        return json.dumps([{"date": 1_577_836_800 + i * 86_400, "tvl": vals[i]} for i in range(n)])
    if ts.startswith("daily"):  # Open-Meteo: parallel arrays under `daily`
        return json.dumps({"daily": {"time": dates, vf.split(".")[-1]: vals}})
    if ts.startswith("value.timeSeries"):  # USGS: deep nesting
        recs = [
            {"dateTime": dates[i] + "T00:00:00.000-00:00", "value": str(vals[i])}
            for i in range(n)
        ]
        return json.dumps({"value": {"timeSeries": [{"values": [{"value": recs}]}]}})
    if ts.startswith("data[]"):  # RKI: list under `data`
        rows = [{"date": dates[i], vf.split(".")[-1]: vals[i]} for i in range(n)]
        return json.dumps({"data": rows})
    raise AssertionError(f"no fixture shape for {ts!r}")


def _fetch_for_registry(url: str) -> str:
    spec = next(s for s in DAILY_REGISTRY.values() if s.url == url)
    return _payload_for_spec(spec)


def test_every_registry_feed_parses_offline():
    for name in DAILY_REGISTRY:
        feed = make_daily_feed(name, fetch=_fetch_for_registry)
        motifs = feed.pull(2, 256, np.random.default_rng(0))
        assert all(m.shape == (256,) for m in motifs), name


def test_build_daily_live_source_is_multidomain_and_seeded():
    src = build_daily_live_source(fetch=_fetch_for_registry)
    buf = FreshBuffer(src, pool_size=40, motif_len=256)
    buf.refresh(np.random.default_rng(0xDA1147))
    assert set(buf.pool_domains) == {s.domain for s in DAILY_REGISTRY.values()}
    assert sum(Counter(buf.pool_domains).values()) == 40


def test_make_daily_feed_unknown_name_raises():
    with pytest.raises(KeyError):
        make_daily_feed("not-a-feed", fetch=_fetch_const)


@pytest.mark.skipif(
    os.environ.get("TSBENCH_LIVE_TESTS") != "1",
    reason="network smoke test; set TSBENCH_LIVE_TESTS=1 to run",
)
def test_live_network_smoke():  # pragma: no cover - network
    src = build_daily_live_source()
    motifs = src.pull(5, 256, np.random.default_rng(0))
    assert len(motifs) == 5
    assert all(np.isfinite(m).all() for m in motifs)
