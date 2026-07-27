"""Unit tests for source_discovery.wire (verify-and-wire) and .audit (freshness)."""

from __future__ import annotations

import datetime as dt
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from source_discovery import audit, wire

UTC = dt.timezone.utc


# --------------------------------------------------------------------------- #
# wire.safe_unescape
# --------------------------------------------------------------------------- #

def test_safe_unescape_handles_agent_escaping() -> None:
    assert wire.safe_unescape("a=1&amp;b=2&lt;3&gt;&quot;x&quot;&#39;y&#x27;") == \
        'a=1&b=2<3>"x"\'y\''


def test_safe_unescape_never_mangles_timespan() -> None:
    # html.unescape turns "&times" (legacy semicolon-less entity) into "×".
    url = "https://api.example.com/chart?format=json&timespan=30d"
    assert wire.safe_unescape(url) == url


# --------------------------------------------------------------------------- #
# wire.verify_entry — the admission gate, against a stub scraper
# --------------------------------------------------------------------------- #

class _StubScraper:
    def __init__(self, records, fail: Exception | None = None):
        self._records = records
        self._fail = fail
        self.fetches: list = []

    def fetch_payload(self, entry, panel_row=None):
        if self._fail:
            raise self._fail
        self.fetches.append(panel_row)
        return b"{}", "application/json"

    def parse_payload(self, entry, blob, ctype):
        return list(self._records)


def _series(n: int, numeric: bool = True, end: dt.datetime | None = None):
    """n hourly records ending at `end` (default: now) — fresh by construction."""
    end = end or dt.datetime.now(UTC)
    return [{"timestamp": (end - dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S"),
             "value": str(i) if numeric else "n/a"} for i in range(n)]


def test_verify_passes_long_numeric_series() -> None:
    res = wire.verify_entry({"endpoint": {"url": "u"}}, _StubScraper(_series(30)))
    assert res.ok and res.rows == 30 and res.distinct_ts >= 20


def test_verify_rejects_short_series() -> None:
    res = wire.verify_entry({"endpoint": {"url": "u"}}, _StubScraper(_series(5)))
    assert not res.ok


def test_verify_rejects_non_numeric() -> None:
    res = wire.verify_entry({"endpoint": {"url": "u"}},
                            _StubScraper(_series(30, numeric=False)))
    assert not res.ok


def test_verify_accepts_wide_panel_snapshot() -> None:
    # One timestamp but 25 panel entities with numeric values (station snapshot).
    recs = [{"timestamp": "2026-07-26T00:00:00", "value": str(i),
             "_panel_station": f"s{i}"} for i in range(25)]
    res = wire.verify_entry({"endpoint": {"url": "u"}}, _StubScraper(recs))
    assert res.ok and res.panel_entities == 25


def test_verify_iterates_panel_rows() -> None:
    stub = _StubScraper(_series(30))
    entry = {"endpoint": {"url": "u/{K}"}, "panel": [{"K": "a"}, {"K": "b"}]}
    res = wire.verify_entry(entry, stub)
    assert res.ok and stub.fetches == [{"K": "a"}, {"K": "b"}]


def test_verify_fetch_error_is_verdict_not_crash() -> None:
    res = wire.verify_entry({"endpoint": {"url": "u"}},
                            _StubScraper([], fail=RuntimeError("HTTP 403: nope")))
    assert not res.ok and "HTTP 403" in res.error


def test_verify_rejects_stale_window_even_with_volume() -> None:
    # The oldest-first-pagination failure class: plenty of rows, all historical.
    old = _series(30, end=dt.datetime.now(UTC) - dt.timedelta(days=400))
    res = wire.verify_entry({"frequency": "PT1H", "endpoint": {"url": "u"}},
                            _StubScraper(old))
    assert not res.ok and "stale at wire time" in res.error


# --------------------------------------------------------------------------- #
# wire.wire_batch — end-to-end against a tmp catalog
# --------------------------------------------------------------------------- #

_EXISTING = [{"id": "existing_src", "domain": "energy", "frequency": "P1D",
              "endpoint": {"url": "https://old.example/api", "type": "rest_json"},
              "schema": {"timestamp_field": "t", "value_field": "v"}}]


def _grind_block(sid: str, url: str, wireable: bool = True, reason: str = "") -> dict:
    entry = {"id": sid, "domain": "energy", "frequency": "PT1H",
             "endpoint": {"url": url, "type": "rest_json"},
             "schema": {"timestamp_field": "t", "value_field": "v"}}
    return {"candidate_name": sid, "wireable": wireable,
            "yaml_block": "- " + yaml.safe_dump(entry).replace("\n", "\n  ").rstrip(),
            "cron_cadence": "PT1H", "reason": reason}


@pytest.fixture()
def catalog(tmp_path):
    cat = tmp_path / "sources.yaml"
    cat.write_text(yaml.safe_dump(_EXISTING, sort_keys=False))
    (tmp_path / "cron.yaml").write_text(yaml.safe_dump(
        {"schedules": [{"every": "P1D", "sources": ["existing_src"]}]}))
    ledger_dir = tmp_path / "discovery"
    ledger_dir.mkdir()
    (ledger_dir / "proposal_ledger.json").write_text(json.dumps([
        {"key": "energy|new.example", "name": "new_src", "status": "proposed"},
        {"key": "energy|gated.example", "name": "gated_src", "status": "proposed"},
        {"key": "energy|dead.example", "name": "dead_src", "status": "proposed"},
    ]))
    return cat


def test_wire_batch_dry_run_writes_nothing(catalog, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(wire, "_scraper", lambda: _StubScraper(_series(30)))
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps([_grind_block("new_src", "https://new.example/api")]))
    rep = wire.wire_batch(batch, catalog, label="t", apply=False)
    assert rep["wired_ids"] == ["new_src"] and not rep["applied"]
    assert len(yaml.safe_load(catalog.read_text())) == 1  # untouched


def test_wire_batch_apply_wires_cron_and_ledger(catalog, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(wire, "_scraper", lambda: _StubScraper(_series(30)))
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps([
        _grind_block("new_src", "https://new.example/api"),
        _grind_block("gated_src", "x", wireable=False, reason="needs API key"),
        _grind_block("dead_src", "x", wireable=False, reason="endpoint returns 404"),
        _grind_block("existing_src", "https://old.example/api"),  # dup by id+url
    ]))
    rep = wire.wire_batch(batch, catalog, label="t", apply=True)
    assert rep["applied"] and rep["wired_ids"] == ["new_src"]

    entries = yaml.safe_load(catalog.read_text())
    assert [e["id"] for e in entries] == ["existing_src", "new_src"]

    cron = yaml.safe_load((catalog.parent / "cron.yaml").read_text())
    hourly = [g for g in cron["schedules"] if g["every"] == "PT1H"]
    assert hourly and hourly[0]["sources"] == ["new_src"]

    led = {e["name"]: e["status"] for e in json.loads(
        (catalog.parent / "discovery" / "proposal_ledger.json").read_text())}
    assert led == {"new_src": "wired", "gated_src": "key-gated",
                   "dead_src": "rejected"}


def test_wire_batch_rejects_verify_failures(catalog, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(wire, "_scraper", lambda: _StubScraper(_series(3)))
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps([_grind_block("new_src", "https://new.example/api")]))
    rep = wire.wire_batch(batch, catalog, label="t", apply=True)
    assert rep["wired_ids"] == [] and rep["counts"]["rejected"] == 1
    assert len(yaml.safe_load(catalog.read_text())) == 1


def test_wire_batch_bad_domain_rejected(catalog, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(wire, "_scraper", lambda: _StubScraper(_series(30)))
    blk = _grind_block("new_src", "https://new.example/api")
    blk["yaml_block"] = blk["yaml_block"].replace("domain: energy", "domain: crypto")
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps([blk]))
    rep = wire.wire_batch(batch, catalog, label="t")
    assert rep["detail"][0]["verdict"] == "bad-domain"


# --------------------------------------------------------------------------- #
# audit.parse_ts — the formats actually stored on disk
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw,expect", [
    ("2026-07-26T07:00:00+00:00", dt.datetime(2026, 7, 26, 7, tzinfo=UTC)),
    ("2026-07-26T07:00:00Z", dt.datetime(2026, 7, 26, 7, tzinfo=UTC)),
    ("2026-07-26T07:00Z", dt.datetime(2026, 7, 26, 7, tzinfo=UTC)),
    ("2026-07-26 07:00:00.123", dt.datetime(2026, 7, 26, 7, 0, 0, 123000, tzinfo=UTC)),
    ("07/26/2026 14:05:00", dt.datetime(2026, 7, 26, 14, 5, tzinfo=UTC)),  # NYISO
    ("2026/07/26 14:05:00", dt.datetime(2026, 7, 26, 14, 5, tzinfo=UTC)),  # AEMO
    ("01.2026", dt.datetime(2026, 1, 1, tzinfo=UTC)),                      # MM.YYYY
    ("26/07/2026", dt.datetime(2026, 7, 26, tzinfo=UTC)),                  # DD/MM
    ("2026-Q2", dt.datetime(2026, 4, 1, tzinfo=UTC)),
    ("2025", dt.datetime(2025, 1, 1, tzinfo=UTC)),
    ("202607", dt.datetime(2026, 7, 1, tzinfo=UTC)),
    ("20260726", dt.datetime(2026, 7, 26, tzinfo=UTC)),
    ("1784998860", dt.datetime.fromtimestamp(1784998860, UTC)),
    ("1784998860000", dt.datetime.fromtimestamp(1784998860, UTC)),
    ("2026M05", dt.datetime(2026, 5, 1, tzinfo=UTC)),                      # DST
    ("2026MM05", dt.datetime(2026, 5, 1, tzinfo=UTC)),                     # CBS
    ("2026-W22", dt.datetime(2026, 5, 25, tzinfo=UTC)),                    # ECB ILM
    ("2026W01", dt.datetime(2025, 12, 29, tzinfo=UTC)),                    # W1 backs into Dec
])
def test_parse_ts_formats(raw, expect) -> None:
    assert audit.parse_ts(raw) == expect


def test_parse_ts_garbage_is_none() -> None:
    assert audit.parse_ts("not a date") is None
    assert audit.parse_ts("") is None
    assert audit.parse_ts(None) is None
    assert audit.parse_ts("2025-W53") is None      # 2025 has only 52 ISO weeks


# --------------------------------------------------------------------------- #
# audit thresholds + catalog audit
# --------------------------------------------------------------------------- #

def test_staleness_threshold_bands() -> None:
    assert audit.staleness_threshold("PT5M") == dt.timedelta(hours=72)
    assert audit.staleness_threshold("P1D") == dt.timedelta(seconds=86400 * 8)
    assert audit.staleness_threshold("P1W") == dt.timedelta(days=35)
    assert audit.staleness_threshold("P1M") == dt.timedelta(days=100)
    assert audit.staleness_threshold("P1Y") == dt.timedelta(days=550)
    assert audit.staleness_threshold("irregular") == dt.timedelta(days=45)


def _write_parquet(data_dir, sid: str, stamps: list[str]) -> None:
    d = data_dir / sid
    d.mkdir(parents=True)
    tbl = pa.table({"timestamp": stamps, "value": ["1.0"] * len(stamps)})
    pq.write_table(tbl, d / "2026-07-26.parquet")


def _catalog_file(tmp_path, entries):
    cat = tmp_path / "sources.yaml"
    cat.write_text(yaml.safe_dump(entries, sort_keys=False))
    return cat


def test_audit_catalog_statuses(tmp_path) -> None:
    now = dt.datetime(2026, 7, 26, tzinfo=UTC)
    data = tmp_path / "data"
    _write_parquet(data, "fresh_src", ["2026-07-25T12:00:00", "2026-07-25T13:00:00"])
    _write_parquet(data, "stale_src", ["2026-01-01T00:00:00"])
    _write_parquet(data, "weird_src", ["??", "!!"])
    cat = _catalog_file(tmp_path, [
        {"id": "fresh_src", "frequency": "P1D"},
        {"id": "stale_src", "frequency": "P1D"},
        {"id": "weird_src", "frequency": "P1D"},
        {"id": "empty_src", "frequency": "P1D"},
        {"id": "off_src", "frequency": "P1D", "disabled": True},
    ])
    by_id = {f["id"]: f for f in audit.audit_catalog(cat, data, now=now)}
    assert by_id["fresh_src"]["status"] == "ok"
    assert by_id["stale_src"]["status"] == "stale"
    assert by_id["weird_src"]["status"] == "unparsed"
    assert by_id["empty_src"]["status"] == "nodata"
    assert by_id["off_src"]["status"] == "disabled"


def test_apply_disables_only_hard_stale_and_validates(tmp_path) -> None:
    now = dt.datetime(2026, 7, 26, tzinfo=UTC)
    data = tmp_path / "data"
    _write_parquet(data, "long_dead", ["2025-01-01T00:00:00"])   # ~570d > 3x 8d
    _write_parquet(data, "soft_stale", ["2026-07-05T00:00:00"])  # 21d: stale, < 3x
    cat = _catalog_file(tmp_path, [
        {"id": "long_dead", "frequency": "P1D", "notes": "x"},
        {"id": "soft_stale", "frequency": "P1D"},
    ])
    findings = audit.audit_catalog(cat, data, now=now)
    disabled = audit.apply_disables(cat, findings, today="2026-07-26")
    assert disabled == ["long_dead"]
    entries = {e["id"]: e for e in yaml.safe_load(cat.read_text())}
    assert entries["long_dead"]["disabled"] is True
    assert "2026-07-26" in entries["long_dead"]["disabled_reason"]
    assert entries["long_dead"]["notes"] == "x"          # rest of entry intact
    assert "disabled" not in entries["soft_stale"]


def test_audit_slack_days_override_silences_known_lag(tmp_path) -> None:
    now = dt.datetime(2026, 7, 26, tzinfo=UTC)
    data = tmp_path / "data"
    _write_parquet(data, "lagged_src", ["2026-03-01T00:00:00"])  # 147d old
    cat = _catalog_file(tmp_path, [
        {"id": "lagged_src", "frequency": "P1D", "audit_slack_days": 200},
    ])
    by_id = {f["id"]: f for f in audit.audit_catalog(cat, data, now=now)}
    assert by_id["lagged_src"]["status"] == "ok"
    assert by_id["lagged_src"]["limit_days"] == 200.0


@pytest.mark.parametrize("raw,expect", [
    # RFC 2822 / HTTP-date — the RSS lingua franca (GDACS, iTunes, AWS health)
    ("Fri, 03 Jul 2026 12:57:24 GMT", dt.datetime(2026, 7, 3, 12, 57, 24, tzinfo=UTC)),
    ("Fri, 03 Jul 2026 14:39:49", dt.datetime(2026, 7, 3, 14, 39, 49, tzinfo=UTC)),
    ("Mon Jul 27 00:06:02 2026 GMT", dt.datetime(2026, 7, 27, 0, 6, 2, tzinfo=UTC)),  # FAA
    ("1/1/2026 12:00:00 AM", dt.datetime(2026, 1, 1, tzinfo=UTC)),                    # NRC
    ("010012Z APR 2023", dt.datetime(2023, 4, 1, 0, 12, tzinfo=UTC)),                 # NGA DTG
    ("01-01-2026 01:00", dt.datetime(2026, 1, 1, 1, tzinfo=UTC)),                     # DEFRA
    ("31-05-2026 24:00", dt.datetime(2026, 6, 1, tzinfo=UTC)),      # hour-ending 24
])
def test_parse_ts_wire_formats(raw, expect) -> None:
    assert audit.parse_ts(raw) == expect


def test_parse_ts_named_offset_zone_keeps_offset() -> None:
    got = audit.parse_ts("Mon, 02 Mar 2026 00:52:00 PST")
    assert got is not None and got.utcoffset() == dt.timedelta(hours=-8)


def test_parse_ts_dateless_strings_stay_none() -> None:
    # Time-of-day only (BART) and month-day only (GLERL) cannot yield a date.
    assert audit.parse_ts("05:02:47 PM PDT") is None
    assert audit.parse_ts("01-01") is None


def test_parse_ts_slash_date_with_hour_ending() -> None:
    """AESO's ETS reports put date and hour-ending 1..24 in one column."""
    assert audit.parse_ts("07/26/2026 24") == dt.datetime(2026, 7, 26, 23, tzinfo=UTC)
    assert audit.parse_ts("07/26/2026 01") == dt.datetime(2026, 7, 26, 0, tzinfo=UTC)
    # the bare form is unchanged, and a real clock time still takes the
    # strptime path rather than being read as an hour-ending
    assert audit.parse_ts("07/26/2026") == dt.datetime(2026, 7, 26, tzinfo=UTC)
    assert audit.parse_ts("07/26/2026 14:05:00") == \
        dt.datetime(2026, 7, 26, 14, 5, tzinfo=UTC)
