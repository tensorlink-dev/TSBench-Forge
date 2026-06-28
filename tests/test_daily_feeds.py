"""Tests for the daily public-feed adapters — all offline via an injected fetch.

Covers the JSON path engine (incl. nested flattening + record filtering), the
four payload kinds (list JSON, date-keyed JSON, CSV, whitespace text), panel
expansion, as-of gating, and the full-catalogue multi-domain builder — all
hermetic. A network smoke test runs only when ``TSBENCH_LIVE_TESTS=1``.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from daily_feeds import (
    DAILY_REGISTRY,
    EXCLUDED,
    DatedJsonFeed,
    DatedTextFeed,
    _flatten,
    _to_posix,
    _walk,
    build_daily_live_source,
    expand_daily_feeds,
    make_daily_feed,
    parse_json_dict_series,
    parse_json_series,
    parse_whitespace_series,
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
    nested = {"value": {"timeSeries": [{"values": [{"value": [{"x": 5}, {"x": 6}]}]}]}}
    assert _walk(nested, "value.timeSeries[0].values[0].value[].x") == [5, 6]


def test_walk_missing_paths_return_none():
    assert _walk({"a": 1}, "a.b.c") is None
    assert _walk({"a": [1, 2]}, "a[5]") is None
    assert _walk(None, "a") is None


def test_flatten_collapses_nested_lists():
    assert _flatten([[[1, 2], [3]], [[4]]]) == [1, 2, 3, 4]
    assert _flatten(5) == [5]


def test_to_posix_handles_epoch_iso_and_wiki_stamps():
    assert _to_posix(1_577_836_800) == pytest.approx(1_577_836_800.0)
    assert _to_posix(1_577_836_800_000) == pytest.approx(1_577_836_800.0)  # ms -> s
    assert _to_posix("2020-01-01") == pytest.approx(datetime(2020, 1, 1, tzinfo=UTC).timestamp())
    assert _to_posix("2020-01-01T00:00:00.000-00:00") == pytest.approx(
        datetime(2020, 1, 1, tzinfo=UTC).timestamp()
    )
    assert _to_posix("2020010100") == pytest.approx(  # Wikipedia YYYYMMDD00 stamp
        datetime(2020, 1, 1, tzinfo=UTC).timestamp()
    )
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


def test_parse_json_series_flattens_nested_paths():
    # The USDA/SNOTEL shape: [].data[].values[].{date,value} (double iteration).
    payload = json.dumps(
        [{"data": [{"values": [{"date": "2020-01-01", "value": 1.0},
                               {"date": "2020-01-02", "value": 2.0}]}]}]
    )
    vals, _ = parse_json_series(payload, "[].data[].values[].date", "[].data[].values[].value")
    assert np.array_equal(vals, np.array([1.0, 2.0]))


def test_parse_json_series_filters_by_record_field():
    # The PyPI shape: two categories per date; keep only with_mirrors.
    payload = json.dumps(
        {
            "data": [
                {"category": "with_mirrors", "date": "2020-01-01", "downloads": 100},
                {"category": "without_mirrors", "date": "2020-01-01", "downloads": 70},
                {"category": "with_mirrors", "date": "2020-01-02", "downloads": 110},
                {"category": "without_mirrors", "date": "2020-01-02", "downloads": 80},
            ]
        }
    )
    vals, times = parse_json_series(
        payload, "data[].date", "data[].downloads",
        filter_field="data[].category", filter_value="with_mirrors",
    )
    assert np.array_equal(vals, np.array([100.0, 110.0]))
    assert times.size == 2


def test_parse_json_series_raises_on_non_list_paths():
    with pytest.raises(ValueError):
        parse_json_series(json.dumps({"a": 1}), "a", "a")


def test_parse_json_dict_series():
    # The Frankfurter shape: {"rates": {date: {"EUR": rate}}}.
    payload = json.dumps(
        {"rates": {"2020-01-03": {"EUR": 0.91}, "2020-01-01": {"EUR": 0.89}}}
    )
    vals, times = parse_json_dict_series(payload, "rates", "EUR")
    assert np.array_equal(vals, np.array([0.89, 0.91]))  # sorted by date
    assert np.all(np.diff(times) > 0)


def test_parse_whitespace_series_skips_comments_and_missing():
    text = "# header comment\n2020 1 1 0.0 410.0\n2020 1 2 0.0 -999.99\n2020 1 3 0.0 411.0\n"
    vals, times = parse_whitespace_series(text, date_cols=(0, 1, 2), value_col=4)
    assert np.array_equal(vals, np.array([410.0, 411.0]))  # missing sentinel dropped
    assert times.size == 2


# --------------------------------------------------------------------------- #
# Adapters
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
    commit = datetime(2020, 6, 1, tzinfo=UTC).timestamp()
    gated = AsOfLiveSource(inner=_feed(), commit_time=commit)
    out = gated.pull_dated(6, 200, np.random.default_rng(0))
    assert out
    assert all(ts > commit for _, _, ts in out)


def test_text_feed_parses_and_serves_windows():
    text = "# co2\n" + "\n".join(
        f"2020 {(i % 12) + 1} {(i % 27) + 1} 0.0 {400.0 + (i % 7)}" for i in range(400)
    )
    feed = DatedTextFeed(url="x", domain="co2", value_col=4, fetch=lambda _u: text)
    motifs = feed.pull(3, 256, np.random.default_rng(0))
    assert all(m.shape == (256,) for m in motifs)


# --------------------------------------------------------------------------- #
# Catalogue: per-kind offline fixtures, panel expansion, multi-domain builder
# --------------------------------------------------------------------------- #


_BASE = datetime(2016, 1, 1, tzinfo=UTC)


def _seq(n: int) -> list[datetime]:
    return [_BASE + timedelta(days=i) for i in range(n)]


def _fetch_for_registry(url: str, n: int = 400) -> str:
    """Return a payload matching whichever catalogue feed asked for ``url``.

    Dispatch is by host/path substring so it survives panel URL templating.
    Uses sequential (unique) dates so date-keyed feeds don't collapse.
    """
    days = _seq(n)
    iso = [d.strftime("%Y-%m-%d") for d in days]
    vals = [float(10 + (i % 13)) for i in range(n)]
    if "llama.fi" in url:  # defi_tvl: top-level list, epoch-second dates
        return json.dumps([{"date": int(days[i].timestamp()), "tvl": vals[i]} for i in range(n)])
    if "open-meteo" in url:  # parallel arrays under `daily`
        return json.dumps({"daily": {"time": iso, "temperature_2m_mean": vals}})
    if "waterservices.usgs" in url:  # deep nesting
        recs = [
            {"dateTime": iso[i] + "T00:00:00.000-00:00", "value": str(vals[i])} for i in range(n)
        ]
        return json.dumps({"value": {"timeSeries": [{"values": [{"value": recs}]}]}})
    if "egov.usda" in url:  # SNOTEL double iteration
        recs = [{"date": iso[i], "value": vals[i]} for i in range(n)]
        return json.dumps([{"data": [{"values": recs}]}])
    if "corona-zahlen" in url:  # RKI list under `data`
        key = "incidence7Days" if "hospitalization" in url else "cases"
        return json.dumps({"data": [{"date": iso[i], key: vals[i]} for i in range(n)]})
    if "npmjs.org" in url:  # npm range
        return json.dumps({"downloads": [{"day": iso[i], "downloads": vals[i]} for i in range(n)]})
    if "frankfurter" in url:  # date-keyed object
        return json.dumps({"rates": {iso[i]: {"EUR": vals[i]} for i in range(n)}})
    if "treasury" in url:  # CSV (MM/DD/YYYY)
        rows = "\n".join(f"{days[i].strftime('%m/%d/%Y')},{vals[i]}" for i in range(n))
        return f"Date,10 Yr\n{rows}"
    if "noaa.gov" in url:  # whitespace text
        body = "\n".join(f"{d.year} {d.month} {d.day} 0.0 {vals[i]}" for i, d in enumerate(days))
        return f"# comment\n{body}"
    if "wikimedia" in url:  # YYYYMMDD00 stamps
        items = [
            {"timestamp": days[i].strftime("%Y%m%d") + "00", "views": vals[i]} for i in range(n)
        ]
        return json.dumps({"items": items})
    if "pypistats" in url:  # two categories per date -> filter
        recs = []
        for i in range(n):
            recs.append({"category": "with_mirrors", "date": iso[i], "downloads": vals[i]})
            recs.append({"category": "without_mirrors", "date": iso[i], "downloads": vals[i] + 1})
        return json.dumps({"data": recs})
    if "crates.io" in url:
        recs = [{"date": iso[i], "downloads": vals[i]} for i in range(n)]
        return json.dumps({"meta": {"extra_downloads": recs}})
    raise AssertionError(f"no fixture for {url}")


def test_registry_and_exclusions_account_for_all_18_daily_sources():
    # 15 ported + 3 documented exclusions == the catalogue's 18 P1D sources.
    assert len(DAILY_REGISTRY) == 15
    assert len(EXCLUDED) == 3
    assert set(DAILY_REGISTRY).isdisjoint(EXCLUDED)


def test_every_registry_feed_parses_offline():
    for name in DAILY_REGISTRY:
        feed = make_daily_feed(name, fetch=_fetch_for_registry)
        motifs = feed.pull(2, 256, np.random.default_rng(0))
        assert all(m.shape == (256,) for m in motifs), name


def test_panel_specs_expand_to_one_feed_per_instance():
    for name, spec in DAILY_REGISTRY.items():
        components = expand_daily_feeds(name, fetch=_fetch_for_registry)
        expected = len(spec.panel) if spec.panel else 1
        assert len(components) == expected, name
        # each spec contributes unit total weight regardless of panel size
        assert sum(w for _, w in components) == pytest.approx(1.0)


def test_build_daily_live_source_is_multidomain_and_seeded():
    src = build_daily_live_source(fetch=_fetch_for_registry)
    buf = FreshBuffer(src, pool_size=300, motif_len=256)
    buf.refresh(np.random.default_rng(0xDA1147))
    assert set(buf.pool_domains) == {s.domain for s in DAILY_REGISTRY.values()}
    assert sum(Counter(buf.pool_domains).values()) == 300


def test_make_daily_feed_unknown_name_raises():
    with pytest.raises(KeyError):
        make_daily_feed("not-a-feed", fetch=_fetch_const)


def test_make_daily_feed_selects_panel_instance():
    a = make_daily_feed("pypi_downloads", instance=0, fetch=_fetch_for_registry)
    b = make_daily_feed("pypi_downloads", instance=5, fetch=_fetch_for_registry)
    assert a.url != b.url  # different templated package URLs
    assert "numpy" in a.url


@pytest.mark.skipif(
    os.environ.get("TSBENCH_LIVE_TESTS") != "1",
    reason="network smoke test; set TSBENCH_LIVE_TESTS=1 to run",
)
def test_live_network_smoke():  # pragma: no cover - network
    # Long feeds only — the short-history feeds can't serve a 256-window.
    src = build_daily_live_source(
        names=["defi_tvl", "climate_open_meteo", "epidemic_cases", "fx_usd_eur"]
    )
    motifs = src.pull(5, 256, np.random.default_rng(0))
    assert len(motifs) == 5
    assert all(np.isfinite(m).all() for m in motifs)
