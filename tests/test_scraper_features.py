"""Unit tests for the scraper's composite-timestamp and field-paneling features."""
import datetime as dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src" / "sources"))
import scraper  # noqa: E402


def test_compose_ts_date_plus_hour():
    # hour-ENDING 1..24 -> hour-of-day 0..23 (grid CSV convention)
    assert scraper._compose_ts(["2026-01-01", "1"]) == "2026-01-01T00:00:00"
    assert scraper._compose_ts(["2026-01-01", "24"]) == "2026-01-01T23:00:00"
    assert scraper._compose_ts(["2026/07/24", "13"]) == "2026-07-24T12:00:00"


def test_compose_ts_numeric_ndbc_unchanged():
    assert scraper._compose_ts(["2026", "07", "24", "13", "30"]) == "2026-07-24T13:30:00"
    assert scraper._compose_ts(["2026", "07", "24"]) == "2026-07-24"


def test_csv_composite_timestamp_hourly():
    csv_text = "Date,Hour,Ontario Demand\n2026-01-01,1,16526\n2026-01-01,2,16374\n"
    recs = scraper._records_from_csv(
        csv_text, {"timestamp_field": "Date Hour", "value_field": "Ontario Demand"}
    )
    ts = [r["timestamp"] for r in recs]
    assert ts == ["2026-01-01T00:00:00", "2026-01-01T01:00:00"]  # distinct, not day-collapsed


def test_json_field_paneling_splits_series():
    data = [
        {"d": "2026-01-01", "v": 0.9, "reg": "A"},
        {"d": "2026-01-08", "v": 0.8, "reg": "A"},
        {"d": "2026-01-01", "v": 0.5, "reg": "B"},
    ]
    recs = scraper._records_from_json(
        data, {"timestamp_field": "[].d", "value_field": "[].v", "panel_field": "[].reg"}
    )
    assert all("_panel_reg" in r for r in recs)
    assert {r["_panel_reg"] for r in recs} == {"A", "B"}


def test_csv_field_paneling():
    csv_text = "station,ts,level\nX,2026-01-01,10\nY,2026-01-01,20\n"
    recs = scraper._records_from_csv(
        csv_text, {"timestamp_field": "ts", "value_field": "level", "panel_field": "station"}
    )
    assert {r["_panel_station"] for r in recs} == {"X", "Y"}


def test_graphql_parse_data_path():
    # GraphQL results nest under data.*; the rest_json path handles it.
    payload = {"data": {"transactions": {"nodes": [
        {"createdAt": "2026-07-01T00:00:00Z", "amount": {"value": 10}},
        {"createdAt": "2026-07-01T01:00:00Z", "amount": {"value": 20}},
    ]}}}
    recs = scraper._records_from_json(payload, {
        "timestamp_field": "data.transactions.nodes[].createdAt",
        "value_field": "data.transactions.nodes[].amount.value",
    })
    assert [r["timestamp"] for r in recs] == ["2026-07-01T00:00:00Z", "2026-07-01T01:00:00Z"]
    assert [r["value"] for r in recs] == [10, 20]


def test_paginate_single_page_noop():
    import json
    first = json.dumps({"data": [{"t": 1}, {"t": 2}], "next": None}).encode()
    out = scraper._paginate_json(first, "http://x", {}, {"items": "data", "next": "next", "max_pages": 3})
    import json as j
    assert len(j.loads(out)["data"]) == 2


def test_aggregate_count_per_hour():
    recs = [
        {"timestamp": "2026-07-01T00:05:00Z", "value": 1},
        {"timestamp": "2026-07-01T00:40:00Z", "value": 1},
        {"timestamp": "2026-07-01T01:10:00Z", "value": 1},
    ]
    out = scraper._aggregate_records(recs, {"op": "count", "bin": "PT1H"})
    counts = {r["timestamp"][:13]: r["value"] for r in out}
    assert sum(counts.values()) == 3
    assert len(out) == 2  # two distinct hours


def test_aggregate_count_preserves_panel():
    recs = [
        {"timestamp": "2026-07-01T00:05:00Z", "value": 1, "_panel_region": "A"},
        {"timestamp": "2026-07-01T00:40:00Z", "value": 1, "_panel_region": "B"},
        {"timestamp": "2026-07-01T00:50:00Z", "value": 1, "_panel_region": "A"},
    ]
    out = scraper._aggregate_records(recs, {"op": "count", "bin": "PT1H"})
    by = {(r["_panel_region"]): r["value"] for r in out}
    assert by == {"A": 2, "B": 1}


def test_jsonstat2_single_series():
    # JSON-stat 2.0: one time dim, one size-1 metric -> flat value[] maps to time
    data = {
        "class": "dataset", "version": "2.0",
        "id": ["ContentsCode", "Tid"], "size": [1, 3],
        "role": {"time": ["Tid"], "metric": ["ContentsCode"]},
        "dimension": {
            "ContentsCode": {"category": {"index": {"CPI": 0}}},
            "Tid": {"category": {"index": {"1980M01": 0, "1980M02": 1, "1980M03": 2}}},
        },
        "value": [95.3, 96.8, 97.2],
    }
    recs = scraper._records_from_jsonstat2(data, {})
    assert [r["timestamp"] for r in recs] == ["1980-01", "1980-02", "1980-03"]  # normalised
    assert [r["value"] for r in recs] == [95.3, 96.8, 97.2]


def test_jsonstat2_panels_nontime_dims():
    # two regions x two months -> 4 flat values, split into _panel_Region
    data = {
        "class": "dataset",
        "id": ["Region", "Tid"], "size": [2, 2],
        "role": {"time": ["Tid"]},
        "dimension": {
            "Region": {"category": {"index": {"SE": 0, "NO": 1}}},
            "Tid": {"category": {"index": {"2020M01": 0, "2020M02": 1}}},
        },
        "value": [10, 11, 20, 21],  # SE:[10,11], NO:[20,21]  (C-order)
    }
    recs = scraper._records_from_jsonstat2(data, {})
    se = {r["timestamp"]: r["value"] for r in recs if r["_panel_Region"] == "SE"}
    no = {r["timestamp"]: r["value"] for r in recs if r["_panel_Region"] == "NO"}
    assert se == {"2020-01": 10, "2020-02": 11}
    assert no == {"2020-01": 20, "2020-02": 21}


def test_jsonstat2_skips_null_and_normalises_quarter():
    data = {
        "class": "dataset", "id": ["Tid"], "size": [3],
        "role": {"time": ["Tid"]},
        "dimension": {"Tid": {"category": {"index": {"2020K1": 0, "2020K2": 1, "2020K3": 2}}}},
        "value": [1.0, None, 3.0],  # middle obs missing
    }
    recs = scraper._records_from_jsonstat2(data, {})
    assert [(r["timestamp"], r["value"]) for r in recs] == [("2020-Q1", 1.0), ("2020-Q3", 3.0)]


def test_compact_date_offset_token():
    """`{YYYYMMDD-Nd}` — for hosts whose daily file name lags UTC (NYISO et al)."""
    now = dt.datetime(2026, 7, 26, 1, 12, tzinfo=dt.UTC)
    assert scraper.expand_url("p/{YYYYMMDD-1d}pal.csv", now) == "p/20260725pal.csv"
    assert scraper.expand_url("p/{YYYYMMDD}pal.csv", now) == "p/20260726pal.csv"
    # crossing a month boundary
    assert scraper.expand_url("{YYYYMMDD-26d}", dt.datetime(2026, 7, 26, tzinfo=dt.UTC)) == "20260630"


def test_month_offset_token():
    """`{YYYYMM-Nm}` — calendar-month arithmetic, incl. year wrap."""
    now = dt.datetime(2026, 7, 26, tzinfo=dt.UTC)
    assert scraper.expand_url("{YYYYMM-0m}", now) == "202607"
    assert scraper.expand_url("{YYYYMM-1m}", now) == "202606"
    assert scraper.expand_url("{YYYYMM-7m}", now) == "202512"
    assert scraper.expand_url("{YYYYMM-19m}", now) == "202412"


def test_offset_tokens_do_not_shadow_literal_tokens():
    """The literal {YYYYMMDD}/{YYYYMM} subs must not partially consume offset tokens."""
    now = dt.datetime(2026, 7, 26, 1, 0, tzinfo=dt.UTC)
    assert scraper.expand_url("{YYYY-MM-DD-1d}|{YYYYMMDD-2d}|{YYYYMM-1m}", now) == \
        "2026-07-25|20260724|202606"


def test_csv_column_name_containing_slash():
    """Literal column names with '/' (NYISO "LBMP ($/MWHr)") must not be
    truncated by the '/'-path-shortening convention."""
    csv_text = (
        '"Time Stamp","Name","LBMP ($/MWHr)"\n'
        '"07/25/2026 00:05:00","CAPITL",43.64\n'
        '"07/25/2026 00:10:00","CAPITL",41.02\n'
    )
    schema = {"timestamp_field": "Time Stamp", "value_field": "LBMP ($/MWHr)",
              "panel_field": "Name"}
    rows = scraper._records_from_csv(csv_text, schema)
    assert len(rows) == 2
    assert rows[0]["LBMP ($/MWHr)"] == "43.64"
    assert rows[0]["_panel_Name"] == "CAPITL"


def test_csv_slash_path_shortening_still_works():
    """Backwards-compat: a genuine '/'-path still resolves to its last segment."""
    csv_text = "date,value\n2026-01-01,5\n2026-01-02,6\n"
    rows = scraper._records_from_csv(csv_text, {"timestamp_field": "date",
                                                "value_field": "some/path/value"})
    assert [r["value"] for r in rows] == ["5", "6"]


def test_sniff_delim_prefers_semicolon_when_dominant():
    """European CSV: ';' separates, ',' is the decimal mark."""
    assert scraper._sniff_delim('"Date";"D0";"Value"') == ";"
    assert scraper._sniff_delim("1988-01;1J;2,887") == ";"
    # a normal comma CSV is unaffected
    assert scraper._sniff_delim("date,station,value") == ","
    assert scraper._sniff_delim("a\tb\tc") == "\t"


def test_semicolon_csv_with_quoted_fields():
    csv_text = (
        '"CubeId";"rendoblim"\n'
        '"Date";"D0";"Value"\n'
        '"1988-01";"1J";"2.887"\n'
        '"1988-02";"1J";"3.218"\n'
    )
    rows = scraper._records_from_csv(csv_text, {"timestamp_field": "Date",
                                                "value_field": "Value",
                                                "panel_field": "D0"})
    assert [r["timestamp"] for r in rows] == ["1988-01", "1988-02"]
    assert [r["Value"] for r in rows] == ["2.887", "3.218"]
    assert rows[0]["_panel_D0"] == "1J"


def test_pipe_delimited_csv():
    """NRC reactor status and several gov feeds are '|'-delimited."""
    assert scraper._sniff_delim("ReportDt|Unit|Power") == "|"
    txt = ("ReportDt|Unit|Power\n"
           "7/24/2026 12:00:00 AM|Arkansas Nuclear 1|100\n"
           "7/24/2026 12:00:00 AM|Beaver Valley 1|97\n")
    rows = scraper._records_from_csv(txt, {"timestamp_field": "ReportDt",
                                           "value_field": "Power",
                                           "panel_field": "Unit"})
    assert [r["Power"] for r in rows] == ["100", "97"]
    assert {r["_panel_Unit"] for r in rows} == {"Arkansas Nuclear 1", "Beaver Valley 1"}


def test_wide_csv_melt():
    """Zillow-style pivot CSV: one row per region, one column per date."""
    csv_text = (
        "RegionID,SizeRank,RegionName,2000-01-31,2000-02-29,2000-03-31\n"
        "394913,0,New York NY,220000,221500,\n"
        "753899,1,Los Angeles CA,190000,,192000\n"
    )
    schema = {"wide": {"id_fields": ["RegionName"]}}
    rows = scraper._records_from_csv(csv_text, schema)
    # blanks are dropped, not emitted as empty observations
    assert len(rows) == 4
    ny = {r["timestamp"]: r["value"] for r in rows if r["_panel_RegionName"] == "New York NY"}
    assert ny == {"2000-01-31": "220000", "2000-02-29": "221500"}
    la = {r["timestamp"]: r["value"] for r in rows if r["_panel_RegionName"] == "Los Angeles CA"}
    assert la == {"2000-01-31": "190000", "2000-03-31": "192000"}


def test_wide_csv_ignores_non_date_columns():
    """Non-date id/metadata columns must never become timestamps."""
    csv_text = "RegionName,StateName,SizeRank,2024-01-31\nAustin TX,TX,25,500000\n"
    rows = scraper._records_from_csv(csv_text, {"wide": {"id_fields": ["RegionName"]}})
    assert [r["timestamp"] for r in rows] == ["2024-01-31"]
    assert rows[0]["value"] == "500000"


def test_capped_response_rejects_oversized_body():
    """A body past the cap must raise, not truncate — a half-read JSON/CSV
    payload would parse into silently wrong records."""
    import httpx

    def handler(request):
        return httpx.Response(200, content=b"x" * 5000)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        with client.stream("GET", "http://x/big") as resp:
            try:
                scraper._capped_response(resp, "http://x/big", max_bytes=1000)
            except RuntimeError as e:
                assert "exceeded" in str(e)
            else:
                raise AssertionError("expected RuntimeError for oversized body")


def test_capped_response_passes_normal_body_and_keeps_content_type():
    import httpx

    def handler(request):
        return httpx.Response(200, content=b'{"ok":1}',
                              headers={"content-type": "application/json"})

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        with client.stream("GET", "http://x/small") as resp:
            out = scraper._capped_response(resp, "http://x/small", max_bytes=1000)
    assert out.status_code == 200
    assert out.content == b'{"ok":1}'
    assert "json" in out.headers.get("content-type", "")
    # transfer-encoding headers must not survive onto the decoded body
    assert "content-encoding" not in out.headers


def test_period_seconds_parses_iso_durations():
    assert scraper._period_seconds("PT1M") == 60
    assert scraper._period_seconds("PT2M30S") == 150
    assert scraper._period_seconds("PT30M") == 1800
    assert scraper._period_seconds("PT1H") == 3600
    assert scraper._period_seconds("P1D") == 86400
    assert scraper._period_seconds("P1W") == 604800
    assert scraper._period_seconds("irregular") is None
    assert scraper._period_seconds("") is None


def test_is_due_fast_sources_always_run(monkeypatch):
    """Hourly-or-faster feeds are never cadence-skipped."""
    monkeypatch.setattr(scraper, "_last_scraped_age", lambda sid: 1.0)
    for freq in ("PT1M", "PT5M", "PT30M", "PT1H"):
        assert scraper.is_due({"id": "x", "frequency": freq}) is True


def test_is_due_slow_sources_skipped_until_stale(monkeypatch):
    """A daily feed scraped 10 min ago is not due; one from 7 h ago is."""
    monkeypatch.setattr(scraper, "_last_scraped_age", lambda sid: 600.0)
    assert scraper.is_due({"id": "x", "frequency": "P1D"}) is False
    monkeypatch.setattr(scraper, "_last_scraped_age", lambda sid: 7 * 3600.0)
    assert scraper.is_due({"id": "x", "frequency": "P1D"}) is True
    # monthly is capped at the same 6 h refresh, not 30 days
    monkeypatch.setattr(scraper, "_last_scraped_age", lambda sid: 7 * 3600.0)
    assert scraper.is_due({"id": "x", "frequency": "P1M"}) is True


def test_is_due_never_scraped_source_runs(monkeypatch):
    monkeypatch.setattr(scraper, "_last_scraped_age", lambda sid: None)
    assert scraper.is_due({"id": "brand-new", "frequency": "P1M"}) is True


def test_is_due_unparseable_frequency_runs(monkeypatch):
    monkeypatch.setattr(scraper, "_last_scraped_age", lambda sid: 60.0)
    assert scraper.is_due({"id": "x", "frequency": "irregular"}) is True
