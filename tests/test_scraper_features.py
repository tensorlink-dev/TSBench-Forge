"""Unit tests for the scraper's composite-timestamp and field-paneling features."""
import datetime as dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src" / "sources"))
import pytest  # noqa: E402

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


def test_write_records_is_atomic(tmp_path, monkeypatch):
    """A killed write must not leave a truncated parquet behind."""
    monkeypatch.setattr(scraper, "DATA_DIR", tmp_path)
    scraper.write_records("atomic_src", [{"timestamp": "2026-01-01", "value": "1"}])
    out = tmp_path / "atomic_src"
    files = list(out.glob("*.parquet"))
    assert len(files) == 1
    # no temp files left over
    assert not list(out.glob("*.tmp*"))
    import pyarrow.parquet as pq
    assert pq.read_table(files[0]).num_rows == 1


def test_write_records_recovers_from_corrupt_existing_file(tmp_path, monkeypatch):
    """A truncated day-file is quarantined, not fatal — otherwise the source is
    wedged forever, because every run re-reads today's file."""
    monkeypatch.setattr(scraper, "DATA_DIR", tmp_path)
    scraper.write_records("corrupt_src", [{"timestamp": "2026-01-01", "value": "1"}])
    out = tmp_path / "corrupt_src"
    day_file = next(out.glob("*.parquet"))
    day_file.write_bytes(b"PAR1garbage-not-a-real-footer")   # simulate a killed write

    n = scraper.write_records("corrupt_src", [{"timestamp": "2026-01-02", "value": "2"}])

    assert n == 1                                    # wrote the new observation
    assert list(out.glob("*.parquet.corrupt"))       # bad file preserved for inspection
    import pyarrow.parquet as pq
    assert pq.read_table(day_file).num_rows == 1     # and the live file is readable again


def test_epoch_string_timestamps_normalised():
    """APIs that quote their epochs (OKX, KuCoin, Bitstamp) must not leave raw
    numeric strings in the same column as ISO stamps."""
    assert scraper._epoch_to_iso("1784998860").startswith("2026-")      # seconds
    assert scraper._epoch_to_iso("1785028800000").startswith("2026-")   # millis
    assert scraper._epoch_to_iso(1784998860).startswith("2026-")        # int form


def test_date_labels_not_mistaken_for_epochs():
    """Guard the range: these are date *labels*, not epochs."""
    assert scraper._epoch_to_iso("20260501") == "20260501"        # YYYYMMDD (2.0e7)
    assert scraper._epoch_to_iso("2026050100") == "2026050100"    # YYYYMMDDHH (2.03e9)
    assert scraper._epoch_to_iso("2003") == "2003"                # a year
    assert scraper._epoch_to_iso("202606") == "202606"            # YYYYMM


def test_double_suffixed_timezone_cleaned():
    """CityBikes emits '...+00:00Z' — both an offset and a Z."""
    assert scraper._epoch_to_iso("2026-07-26T04:21:36.968136+00:00Z") == \
        "2026-07-26T04:21:36.968136+00:00"
    # a plain Z stamp is left alone
    assert scraper._epoch_to_iso("2026-07-26T04:21:36Z") == "2026-07-26T04:21:36Z"


def test_integer_year_not_converted_to_1970():
    """WHO GHO ships TimeDim as an integer year — it must stay a year label,
    not become a 1970 epoch timestamp."""
    assert scraper._epoch_to_iso(2003) == "2003"
    assert scraper._epoch_to_iso(2026) == "2026"
    assert scraper._epoch_to_iso(202606) == "202606"      # YYYYMM as int
    assert scraper._epoch_to_iso(20260501) == "20260501"  # YYYYMMDD as int


def test_real_epochs_still_converted_both_types():
    for v in (1784998860, "1784998860", 1784998860.0):
        assert scraper._epoch_to_iso(v).startswith("2026-")
    for v in (1785028800000, "1785028800000"):
        assert scraper._epoch_to_iso(v).startswith("2026-")


# ── 2026-07-26 feature batch: headers / form-POST / csv_delimiter / XML-attrs /
#    epoch URL tokens ──────────────────────────────────────────────────────────

def test_expand_url_epoch_tokens():
    now = dt.datetime(2026, 7, 26, 12, 0, 0, tzinfo=dt.timezone.utc)
    epoch = int(now.timestamp())
    got = scraper.expand_url("https://x/api?begin={EPOCH-2h}&end={EPOCH}", now=now)
    assert got == f"https://x/api?begin={epoch - 7200}&end={epoch}"


def test_custom_headers_merged_and_env_expanded(monkeypatch):
    seen = {}

    def fake_get(url, headers, cap=None, rto=None):
        seen.update(headers)
        class R:
            status_code = 200
            content = b"{}"
            headers = {"content-type": "application/json"}
        return R()

    monkeypatch.setenv("TEST_CONTACT", "ops@example.com")
    monkeypatch.setattr(scraper, "_http_get", fake_get)
    src = {"endpoint": {"url": "https://x/api", "type": "rest_json",
                        "headers": {"Origin": "https://x",
                                    "User-Agent": "bench/1 ({TEST_CONTACT})"}}}
    scraper.fetch_payload(src)
    assert seen["Origin"] == "https://x"
    assert seen["User-Agent"] == "bench/1 (ops@example.com)"


def test_body_form_posts_urlencoded(monkeypatch):
    calls = {}

    def fake_post(url, headers, body, cap=None, rto=None, form=False):
        calls["body"], calls["form"] = body, form
        class R:
            status_code = 200
            content = b"{}"
            headers = {"content-type": "application/json"}
        return R()

    monkeypatch.setattr(scraper, "_http_post", fake_post)
    src = {"endpoint": {"url": "https://x/api", "type": "rest_json",
                        "body_form": {"table": "kpi", "fmt": "json"}}}
    scraper.fetch_payload(src)
    assert calls["form"] is True and calls["body"] == {"table": "kpi", "fmt": "json"}


def test_csv_forced_delimiter_with_quoted_commas():
    text = 'ts;val\n"2026-07-26";"1,5"\n"2026-07-25";"2,5"\n'
    recs = scraper._records_from_csv(text, {"timestamp_field": "ts",
                                            "value_field": "val",
                                            "csv_delimiter": ";"})
    assert recs[0]["timestamp"] == "2026-07-26"
    assert recs[0]["val"] == "1,5"      # cell comma survives (decimal mark)


def test_xml_record_path_attributes():
    blob = (b'<root xmlns="urn:x"><series k="DE">'
            b'<obs date="2026-07-25" value="1.1"/>'
            b'<obs date="2026-07-26" value="2.2"/>'
            b'</series></root>')
    recs = scraper._records_from_xml(blob, {"record_path": "obs",
                                            "timestamp_field": "@date",
                                            "value_field": ["@value"]})
    assert len(recs) == 2
    assert recs[1] == {"timestamp": "2026-07-26", "value": "2.2"}


def test_xml_record_path_child_and_panel():
    blob = (b'<r><row><t>2026-07-26T00:00:00</t><v>5</v><st id="a1"/></row>'
            b'<row><t>2026-07-26T01:00:00</t><v>6</v><st id="a2"/></row></r>')
    recs = scraper._records_from_xml(blob, {"record_path": "row",
                                            "timestamp_field": "t",
                                            "value_field": ["v"],
                                            "panel_field": "st/@id"})
    assert recs[0]["timestamp"] == "2026-07-26T00:00:00"
    assert recs[0]["v"] == "5" and recs[0]["_panel_id"] == "a1"


def test_compose_ts_year_month_pair():
    # DWD regional averages: 'Jahr','Monat' columns
    assert scraper._compose_ts(["2026", "3"]) == "2026-03"
    assert scraper._compose_ts(["2026", "12"]) == "2026-12"
    # not a month -> untouched join
    assert scraper._compose_ts(["2026", "13"]) == "2026 13"


def test_decimal_year_to_iso_month():
    # NOAA GML trends files use (month-0.5)/12 decimal years
    assert scraper._epoch_to_iso("2026.042") == "2026-01-01"
    assert scraper._epoch_to_iso("2026.125") == "2026-02-01"
    assert scraper._epoch_to_iso("2025.958") == "2025-12-01"
    # epochs with fractions are NOT decimal years (10 digits before the dot);
    # they pass through unchanged, as before
    assert scraper._epoch_to_iso("1784998860.5") == "1784998860.5"


def test_compose_ts_hour_of_day_convention():
    # default: hour-ENDING 1..24 (IESO) — hour '1' means 00:00-01:00
    assert scraper._compose_ts(["2026-07-26", "1"]) == "2026-07-26T00:00:00"
    # Melbourne pedestrian counts: hour is already hour-of-day 0..23
    assert scraper._compose_ts(["2026-07-26", "0"], hour_of_day=True) == "2026-07-26T00:00:00"
    assert scraper._compose_ts(["2026-07-26", "1"], hour_of_day=True) == "2026-07-26T01:00:00"
    assert scraper._compose_ts(["2026-07-26", "23"], hour_of_day=True) == "2026-07-26T23:00:00"


def test_compose_ts_year_month_name():
    # CMS Medicare enrollment: YEAR + English month name columns
    assert scraper._compose_ts(["2026", "January"]) == "2026-01"
    assert scraper._compose_ts(["2026", "SEPTEMBER"]) == "2026-09"
    assert scraper._compose_ts(["2026", "Dec"]) == "2026-12"
    assert scraper._compose_ts(["2026", "Notamonth"]) == "2026 Notamonth"


def test_body_token_expansion():
    now = scraper.dt.datetime.now(scraper.UTC)
    body = {"query": "traffic(from: \"{YYYY-MM-DD-1d}\", to: \"{YYYY-MM-DD}\")",
            "n": 5, "nested": [{"begin": "{EPOCH-2h}"}]}
    out = scraper._expand_body(body)
    assert body["query"].count("{") == 2          # original untouched
    assert "{YYYY" not in out["query"]
    assert now.strftime("%Y-%m-%d") in out["query"]
    assert out["n"] == 5
    begin = int(out["nested"][0]["begin"])
    assert abs(begin - (int(now.timestamp()) - 7200)) < 60


def test_body_panel_expansion():
    out = scraper._expand_body({"variables": {"id": "{STATION}"}},
                               panel_row={"STATION": "ABC123"})
    assert out["variables"]["id"] == "ABC123"


def test_drop_null_values():
    src = {"endpoint": {"type": "rest_csv"},
           "schema": {"timestamp_field": "t", "value_field": "v",
                      "drop_null_values": True}}
    text = "t,v\n2026-07-20,100\n2026-07-27,\n2026-08-03,\n"
    recs = scraper.parse_payload(src, text.encode(), "text/csv")
    assert [r["timestamp"] for r in recs] == ["2026-07-20"]


def test_drop_null_values_keeps_panel_only_null():
    # panel columns don't count as values
    src = {"endpoint": {"type": "rest_csv"},
           "schema": {"timestamp_field": "t", "value_field": "v",
                      "panel_field": "z", "drop_null_values": True}}
    text = "t,v,z\n2026-07-20,1,DE\n2026-07-27,,DE\n"
    recs = scraper.parse_payload(src, text.encode(), "text/csv")
    assert len(recs) == 1 and recs[0]["_panel_z"] == "DE"


def test_compose_hour_of_day_via_schema():
    src = {"endpoint": {"type": "rest_csv"},
           "schema": {"timestamp_field": "Date Hour", "value_field": "Count",
                      "compose_hour_of_day": True}}
    text = "Date,Hour,Count\n2026-07-26,0,11\n2026-07-26,1,22\n"
    recs = scraper.parse_payload(src, text.encode(), "text/csv")
    assert recs[0]["timestamp"] == "2026-07-26T00:00:00"
    assert recs[1]["timestamp"] == "2026-07-26T01:00:00"


def test_csv_columns_mixed_delimiter_header():
    # NMDB NEST: whitespace-delimited header line over ';'-delimited data rows
    src = {"endpoint": {"type": "rest_csv"},
           "schema": {"timestamp_field": "start_date_time", "value_field": "RCORR_E",
                      "csv_delimiter": ";", "csv_columns": ["start_date_time", "RCORR_E"]}}
    text = ("start_date_time   RCORR_E\n"
            "2026-07-26 00:00:00;105.3\n"
            "2026-07-26 00:01:00;105.1\n")
    recs = scraper.parse_payload(src, text.encode(), "text/plain")
    assert len(recs) == 2
    assert recs[0]["timestamp"] == "2026-07-26 00:00:00"
    assert recs[0]["RCORR_E"] == "105.3"


def test_csv_columns_headerless():
    src = {"endpoint": {"type": "rest_csv"},
           "schema": {"timestamp_field": "Year Week", "value_field": "Extent",
                      "csv_columns": ["Year", "Week", "Extent"]}}
    text = "2026 28 431.2\n2026 29 405.7\n"
    recs = scraper.parse_payload(src, text.encode(), "text/plain")
    assert len(recs) == 2
    assert recs[0]["Extent"] == "431.2"


def test_endpoint_encoding_shift_jis():
    src = {"endpoint": {"type": "rest_csv", "encoding": "shift_jis"},
           "schema": {"timestamp_field": "t", "value_field": "v"}}
    text = "t,v\n2026-07-26,100\n"
    recs = scraper.parse_payload(src, text.encode("shift_jis"), "text/csv")
    assert recs[0]["v"] == "100"


def test_compose_year_week():
    # Rutgers snow lab: 'Year Week' rows; flag required since weeks 1..12
    # would otherwise read as months
    assert scraper._compose_ts(["2026", "3"], year_week=True) == "2026-01-12"
    assert scraper._compose_ts(["2026", "30"], year_week=True) == "2026-07-20"
    assert scraper._compose_ts(["2026", "3"]) == "2026-03"          # default unchanged
    assert scraper._compose_ts(["2026", "60"], year_week=True) == "2026 60"  # invalid week


def test_compose_year_week_via_schema():
    src = {"endpoint": {"type": "rest_csv"},
           "schema": {"timestamp_field": "Year Week", "value_field": "Extent",
                      "csv_columns": ["Year", "Week", "Extent"],
                      "compose_year_week": True}}
    text = "2026 29 405.7\n2026 30 398.2\n"
    recs = scraper.parse_payload(src, text.encode(), "text/plain")
    assert recs[0]["timestamp"] == "2026-07-13"


def test_decimal_comma():
    src = {"endpoint": {"type": "rest_csv"},
           "schema": {"timestamp_field": "t", "value_field": "v",
                      "csv_delimiter": ";", "decimal_comma": True}}
    text = "t;v\n2026-07-26;1.234,56\n2026-07-27;0,899\n"
    recs = scraper.parse_payload(src, text.encode(), "text/csv")
    assert recs[0]["v"] == "1234.56"
    assert recs[1]["v"] == "0.899"
    # timestamps and plain values untouched
    assert recs[0]["timestamp"] == "2026-07-26"


def test_snapshot_now():
    src = {"endpoint": {"type": "rest_json"},
           "schema": {"record_path_unused": None, "timestamp_field": "none",
                      "value_field": ["trains[].late"], "snapshot_now": True}}
    import json as _json
    blob = _json.dumps({"trains": [{"late": 3}, {"late": 0}]}).encode()
    recs = scraper.parse_payload(src, blob, "application/json")
    assert len(recs) == 2
    now = scraper.dt.datetime.now(scraper.UTC)
    for r in recs:
        got = scraper.dt.datetime.fromisoformat(r["timestamp"]).replace(tzinfo=scraper.UTC)
        assert abs((now - got).total_seconds()) < 60


def test_snapshot_now_panel_explode():
    src = {"endpoint": {"type": "rest_json"},
           "schema": {"value_field": ["trains[].late"],
                      "panel_field": "trains[].trainno",
                      "snapshot_now": True}}
    import json as _json
    blob = _json.dumps({"trains": [{"late": 3, "trainno": "501"},
                                   {"late": 0, "trainno": "502"}]}).encode()
    recs = scraper.parse_payload(src, blob, "application/json")
    assert len(recs) == 2
    assert recs[0]["_panel_trainno"] == "501" and recs[0]["late"] == 3
    assert recs[1]["_panel_trainno"] == "502"
    assert recs[0]["timestamp"] == recs[1]["timestamp"]


def test_date_keyed_map_unions_metrics_on_date_keys():
    src = {"endpoint": {"type": "rest_json"},
           "schema": {"date_keyed_map": True,
                      "value_field": ["hits.dates", "bandwidth.dates"]}}
    import json as _json
    blob = _json.dumps({"hits": {"total": 9, "dates": {"2026-07-02": 5, "2026-07-01": 4}},
                        "bandwidth": {"dates": {"2026-07-01": 40, "2026-07-02": 50}}}).encode()
    recs = scraper.parse_payload(src, blob, "application/json")
    assert [r["timestamp"] for r in recs] == ["2026-07-01", "2026-07-02"]
    # colliding leaf names ("dates") are disambiguated by their parent
    assert recs[0]["hits_dates"] == 4 and recs[0]["bandwidth_dates"] == 40


def test_date_keyed_map_missing_date_in_one_metric_is_none():
    src = {"endpoint": {"type": "rest_json"},
           "schema": {"date_keyed_map": True, "value_field": ["a.dates", "b.dates"]}}
    import json as _json
    blob = _json.dumps({"a": {"dates": {"2026-07-01": 1, "2026-07-02": 2}},
                        "b": {"dates": {"2026-07-01": 9}}}).encode()
    recs = scraper.parse_payload(src, blob, "application/json")
    assert len(recs) == 2
    assert recs[1]["b_dates"] is None


def test_json_timestamp_composed_from_two_paths():
    """BLS splits every observation into year + periodName ('2026' + 'June')."""
    src = {"endpoint": {"type": "rest_json"},
           "schema": {"timestamp_field": "series[].data[].year series[].data[].periodName",
                      "value_field": ["series[].data[].value"]}}
    import json as _json
    blob = _json.dumps({"series": [{"data": [
        {"year": "2026", "periodName": "June", "value": "333.9"},
        {"year": "2026", "periodName": "May", "value": "335.1"}]}]}).encode()
    recs = scraper.parse_payload(src, blob, "application/json")
    assert [r["timestamp"] for r in recs] == ["2026-06", "2026-05"]
    assert recs[0]["value"] == "333.9"


def test_json_single_path_timestamp_still_walks_verbatim():
    src = {"endpoint": {"type": "rest_json"},
           "schema": {"timestamp_field": "obs[].t", "value_field": ["obs[].v"]}}
    import json as _json
    blob = _json.dumps({"obs": [{"t": "2026-07-01T00:00:00", "v": 1}]}).encode()
    recs = scraper.parse_payload(src, blob, "application/json")
    assert recs[0]["timestamp"] == "2026-07-01T00:00:00"


def test_timestamp_pattern_filters_mixed_cadence_rows():
    """ONS generator CSVs stack annual, quarterly and monthly rows in one column."""
    src = {"endpoint": {"type": "rest_csv"},
           "schema": {"csv_columns": ["period", "value"],
                      "timestamp_field": "period", "value_field": ["value"],
                      "timestamp_pattern": r"^\d{4} [A-Z]{3}$"}}
    text = '"2026","3.1"\n"2026 Q1","3.2"\n"2026 JAN","3.3"\n"2026 FEB","3.4"\n'
    recs = scraper.parse_payload(src, text.encode(), "text/csv")
    assert [r["timestamp"] for r in recs] == ["2026 JAN", "2026 FEB"]


def test_timestamp_pattern_absent_keeps_every_row():
    src = {"endpoint": {"type": "rest_csv"},
           "schema": {"csv_columns": ["period", "value"],
                      "timestamp_field": "period", "value_field": ["value"]}}
    text = '"2026","3.1"\n"2026 JAN","3.3"\n'
    assert len(scraper.parse_payload(src, text.encode(), "text/csv")) == 2


def test_value_and_timestamp_regex_extract_from_prose():
    """Geoscience Australia buries the magnitude in the title and the time in
    the description."""
    src = {"endpoint": {"type": "rss"},
           "schema": {"timestamp_field": "description", "value_field": "title",
                      "timestamp_regex": r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})",
                      "value_regex": r"Magnitude ([0-9.]+)"}}
    xml = ("<rss><channel>"
           "<item><title>Magnitude 5.1, Tonga Trench</title>"
           "<description>2026-07-27T03:36:40.319Z(UTC)</description></item>"
           "</channel></rss>")
    recs = scraper.parse_payload(src, xml.encode(), "application/rss+xml")
    assert recs == [{"timestamp": "2026-07-27T03:36:40", "value": "5.1"}]


def test_rss_falls_back_to_pubdate_when_field_absent():
    src = {"endpoint": {"type": "rss"},
           "schema": {"timestamp_field": "nosuchfield", "value_field": "alsomissing"}}
    xml = ("<rss><channel><item><title>Quake</title>"
           "<pubDate>Fri, 03 Jul 2026 12:57:24 GMT</pubDate></item></channel></rss>")
    recs = scraper.parse_payload(src, xml.encode(), "application/rss+xml")
    assert recs[0]["timestamp"] == "Fri, 03 Jul 2026 12:57:24 GMT"
    assert recs[0]["value"] == "Quake"


def test_value_regex_leaves_non_matching_values_alone():
    src = {"endpoint": {"type": "rest_csv"},
           "schema": {"timestamp_field": "t", "value_field": ["v"],
                      "value_regex": r"Magnitude ([0-9.]+)"}}
    text = "t,v\n2026-07-01,Magnitude 4.2 near X\n2026-07-02,no reading\n"
    recs = scraper.parse_payload(src, text.encode(), "text/csv")
    assert recs[0]["v"] == "4.2" and recs[1]["v"] == "no reading"


def test_wide_csv_skips_preamble_and_honours_delimiter():
    """NOAA CPC degree-day files: 3 preamble lines, pipe-delimited, YYYYMMDD headers."""
    src = {"endpoint": {"type": "rest_csv"},
           "schema": {"skip_rows": 3, "csv_delimiter": "|",
                      "wide": {"id_fields": ["Region"]}}}
    text = ("Product: Daily Heating Degree Days\nRegions: x\nWeights: Population\n"
            "Region|20260101|20260102\n1|41|47\nCONUS|29|28\n")
    recs = scraper.parse_payload(src, text.encode(), "text/csv")
    assert len(recs) == 4
    assert recs[0] == {"timestamp": "20260101", "value": "41", "_panel_Region": "1"}
    assert {r["_panel_Region"] for r in recs} == {"1", "CONUS"}


def test_wide_csv_accepts_compact_date_headers():
    src = {"endpoint": {"type": "rest_csv"},
           "schema": {"wide": {"id_fields": ["id"]}}}
    text = "id,202601,20260201,2026-03,notadate\na,1,2,3,9\n"
    recs = scraper.parse_payload(src, text.encode(), "text/csv")
    assert [r["timestamp"] for r in recs] == ["202601", "20260201", "2026-03"]


def test_xlsx_sheet_selector_picks_named_tab():
    import io
    from openpyxl import Workbook
    wb = Workbook()
    wb.active.title = "cover"
    wb.active["A1"] = "this is a cover page"
    ws = wb.create_sheet("data-42191")
    for row in [("when", "howmany"), ("2026-07-01", 5), ("2026-07-02", 6)]:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    src = {"endpoint": {"type": "rest_xlsx"},
           "schema": {"sheet": "data-42191", "timestamp_field": "when",
                      "value_field": "howmany"}}
    recs = scraper.parse_payload(src, buf.getvalue(), "application/vnd.ms-excel")
    assert recs == [{"timestamp": "2026-07-01", "value": 5.0},
                    {"timestamp": "2026-07-02", "value": 6.0}]


def test_resolve_follows_metadata_to_hashed_url(monkeypatch):
    """GOV.UK mints a new random URL per release; resolve reads the stable API."""
    calls = []

    class _Resp:
        def __init__(self, payload, text=""):
            self.status_code = 200
            self._payload = payload
            self.text = text
            self.content = b"t,v\n2026-07-01,1\n"
            self.headers = {"content-type": "text/csv"}

        def json(self):
            return self._payload

    def fake_get(url, headers, cap, rto=None):
        calls.append(url)
        if "api/content" in url:
            return _Resp({"details": {"attachments": [
                {"url": "https://assets.example/media/aaa/prices_010126.xlsx"},
                {"url": "https://assets.example/media/bbb/prices_010126.csv"},
            ]}})
        return _Resp(None)

    monkeypatch.setattr(scraper, "_http_get", fake_get)
    src = {"endpoint": {"url": "unused", "type": "rest_csv",
                        "resolve": {"url": "https://www.gov.uk/api/content/x",
                                    "select": "details.attachments[].url",
                                    "match": r"prices_\d+\.csv$"}}}
    scraper.fetch_payload(src)
    assert calls[1] == "https://assets.example/media/bbb/prices_010126.csv"


def test_resolve_raises_when_nothing_matches(monkeypatch):
    class _Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"details": {"attachments": []}}

    monkeypatch.setattr(scraper, "_http_get", lambda *a, **k: _Resp())
    src = {"endpoint": {"url": "u", "type": "rest_csv",
                        "resolve": {"url": "https://meta", "select": "details.attachments[].url"}}}
    with pytest.raises(RuntimeError, match="matched no URL"):
        scraper.fetch_payload(src)


def test_scalar_timestamp_broadcasts_over_list_values():
    """GBFS shape: one top-level `last_updated` beside a list of stations."""
    import json as _json
    src = {"endpoint": {"type": "rest_json"},
           "schema": {"timestamp_field": "last_updated",
                      "value_field": "data.stations[].num_bikes_available",
                      "panel_field": "data.stations[].station_id"}}
    blob = _json.dumps({"last_updated": 1784000000, "data": {"stations": [
        {"station_id": "a", "num_bikes_available": 3},
        {"station_id": "b", "num_bikes_available": 0}]}}).encode()
    recs = scraper.parse_payload(src, blob, "application/json")
    assert len(recs) == 2
    assert recs[0]["value"] == 3 and recs[0]["_panel_station_id"] == "a"
    assert recs[0]["timestamp"] == recs[1]["timestamp"] != "null"
    # never the raw-payload dump that this used to fall through to
    assert all("stations" not in str(r.get("value")) for r in recs)


def test_csv_timestamp_column_containing_slash():
    """BPA's column is literally "Date/Time"; the path-leaf reading is "Time"."""
    src = {"endpoint": {"type": "rest_csv"},
           "schema": {"timestamp_field": "Date/Time",
                      "value_field": ["Load", "Fossil/Biomass"]}}
    text = "Date/Time\tLoad\tFossil/Biomass\n07/20/2026 00:00\t6476\t1126\n"
    recs = scraper.parse_payload(src, text.encode(), "text/csv")
    assert recs == [{"timestamp": "07/20/2026 00:00", "Load": "6476",
                     "Fossil/Biomass": "1126"}]


def test_indexed_value_paths_get_clean_column_names():
    import json as _json
    src = {"endpoint": {"type": "rest_json"},
           "schema": {"timestamp_field": "time",
                      "value_field": ["states[][6]", "states[][9]"]}}
    blob = _json.dumps({"time": 1784000000,
                        "states": [["a", "b", 0, 0, 0, 0, 40.6, 0, 0, 4.6]]}).encode()
    recs = scraper.parse_payload(src, blob, "application/json")
    assert recs[0]["states_6"] == 40.6 and recs[0]["states_9"] == 4.6


def test_where_keeps_only_matching_rows():
    """Malaysia's fuel feed returns level and change_weekly rows on one date."""
    import json as _json
    src = {"endpoint": {"type": "rest_json"},
           "schema": {"timestamp_field": "[].date", "value_field": ["[].ron95"],
                      "panel_field": "[].series_type",
                      "where": {"series_type": "level"}}}
    blob = _json.dumps([
        {"date": "2026-07-23", "ron95": 3.62, "series_type": "level"},
        {"date": "2026-07-23", "ron95": 0.20, "series_type": "change_weekly"},
    ]).encode()
    recs = scraper.parse_payload(src, blob, "application/json")
    assert len(recs) == 1 and recs[0]["ron95"] == 3.62


def test_where_matches_plain_columns_too():
    src = {"endpoint": {"type": "rest_csv"},
           "schema": {"timestamp_field": "t", "value_field": ["v", "kind"],
                      "where": {"kind": "actual"}}}
    text = "t,v,kind\n2026-07-01,1,actual\n2026-07-01,9,forecast\n"
    recs = scraper.parse_payload(src, text.encode(), "text/csv")
    assert [r["v"] for r in recs] == ["1"]


# --------------------------------------------------------------------------- #
# schema.timestamp_dims — json-stat time axis named explicitly
# --------------------------------------------------------------------------- #
def test_jsonstat_timestamp_dims_non_latin_dimension():
    """North Macedonia labels its time dimension `Месец`, so neither role.time
    nor the English-word fallback finds it and the whole dataset is lost."""
    data = {
        "class": "dataset",
        "id": ["Показател", "Месец"],
        "size": [1, 3],
        "dimension": {
            "Показател": {"category": {"index": {"Индекс": 0}}},
            "Месец": {"category": {"index": {"2026-01": 0, "2026-02": 1, "2026-03": 2}}},
        },
        "value": [101.0, 102.5, 103.1],
    }
    rows = scraper._records_from_jsonstat2(data, {"timestamp_dims": ["Месец"]})
    assert [r["timestamp"] for r in rows] == ["2026-01", "2026-02", "2026-03"]
    assert [r["value"] for r in rows] == [101.0, 102.5, 103.1]


def test_jsonstat_timestamp_dims_composes_split_year_and_month():
    """Several offices split the period into separate year and month dimensions,
    with month labels carrying ordinal punctuation ('1.')."""
    data = {
        "class": "dataset",
        "id": ["Godina", "Mesec"],
        "size": [2, 2],
        "dimension": {
            "Godina": {"category": {"index": {"2025": 0, "2026": 1}}},
            "Mesec": {"category": {"index": {"1.": 0, "2.": 1}}},
        },
        "value": [1, 2, 3, 4],
    }
    rows = scraper._records_from_jsonstat2(
        data, {"timestamp_dims": ["Godina", "Mesec"]}
    )
    assert [r["timestamp"] for r in rows] == [
        "2025-01", "2025-02", "2026-01", "2026-02"
    ]


def test_compose_pxweb_parts_drops_annual_aggregate_range():
    """'1. - 12.' is the annual total, not a month — composing it as a period
    would invent a 12th-of-January observation that duplicates the year."""
    assert scraper._compose_pxweb_parts(["2026", "1. - 12."]) == "2026"
    assert scraper._compose_pxweb_parts(["2026", "3."]) == "2026-03"


def test_jsonstat_timestamp_dims_ignored_when_dimension_absent():
    """A stale override must not silently produce zero rows — fall through to
    the normal role.time detection instead."""
    data = {
        "class": "dataset",
        "id": ["Tid"],
        "size": [2],
        "dimension": {"Tid": {"category": {"index": {"2026M01": 0, "2026M02": 1}}}},
        "value": [5, 6],
    }
    rows = scraper._records_from_jsonstat2(data, {"timestamp_dims": ["NotThere"]})
    assert [r["timestamp"] for r in rows] == ["2026-01", "2026-02"]


# --------------------------------------------------------------------------- #
# schema.series_start / series_step — time axis as metadata
# --------------------------------------------------------------------------- #
def test_stepped_series_expands_start_plus_step():
    """Statnett sends one start instant, a step, and a bare value array; without
    expansion the array has nothing to align to and becomes one snapshot row."""
    data = {"data": {"startPointUTC": 1769500800000, "Frequency": [50.01, 49.98, 50.02]}}
    rows = scraper._records_from_json(data, {
        "series_start": "data.startPointUTC",
        "series_step": "PT1S",
        "value_field": ["data.Frequency"],
    })
    assert len(rows) == 3
    assert [r["value"] for r in rows] == [50.01, 49.98, 50.02]
    stamps = [dt.datetime.fromisoformat(r["timestamp"]) for r in rows]
    assert (stamps[1] - stamps[0]).total_seconds() == 1.0


def test_stepped_series_accepts_iso_start_and_step_from_payload():
    data = {"start": "2026-07-27T00:00:00Z", "stepMs": 900000,
            "vals": [1.0, 2.0, 3.0, 4.0]}
    rows = scraper._records_from_json(data, {
        "series_start": "start", "series_step": "stepMs", "value_field": ["vals"],
    })
    assert len(rows) == 4
    stamps = [dt.datetime.fromisoformat(r["timestamp"]) for r in rows]
    assert (stamps[1] - stamps[0]).total_seconds() == 900.0
    assert stamps[0].isoformat().startswith("2026-07-27T00:00:00")


def test_stepped_series_multiple_value_paths_named():
    data = {"t0": 1769500800000, "a": [1, 2], "b": [3, 4]}
    rows = scraper._records_from_json(data, {
        "series_start": "t0", "series_step": "PT1M", "value_field": ["a", "b"],
    })
    assert rows[0]["a"] == 1 and rows[0]["b"] == 3
    assert "value" not in rows[0]


def test_stepped_series_without_resolvable_start_yields_nothing():
    """Better to yield no rows than to date real observations from 1970."""
    rows = scraper._records_from_json(
        {"vals": [1, 2, 3]},
        {"series_start": "missing.path", "series_step": "PT1S", "value_field": ["vals"]},
    )
    assert rows == []


def test_parse_epoch_or_iso_distinguishes_ms_from_seconds():
    ms = scraper._parse_epoch_or_iso(1769500800000)
    s = scraper._parse_epoch_or_iso(1769500800)
    assert ms.year == 2026 and s.year == 2026


# --------------------------------------------------------------------------- #
# {FY_JP} — Japanese fiscal year (April-March, named for the start year)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("when,expect", [
    (dt.datetime(2026, 7, 28, tzinfo=dt.timezone.utc), "2026"),   # mid-FY
    (dt.datetime(2026, 4, 1, tzinfo=dt.timezone.utc), "2026"),    # first day
    (dt.datetime(2026, 3, 31, tzinfo=dt.timezone.utc), "2025"),   # last day
    (dt.datetime(2027, 1, 15, tzinfo=dt.timezone.utc), "2026"),   # Jan: prior FY
])
def test_fy_jp_token(when, expect):
    """JEPX names its spot file by fiscal year, so a plain {YYYY} fetches a
    non-existent file every January-March — months after the source was wired
    and verified, which is exactly when nobody is looking."""
    url = scraper.expand_url("https://x/spot_summary_{FY_JP}.csv", now=when)
    assert url == f"https://x/spot_summary_{expect}.csv"


def test_fy_jp_uses_japan_local_date():
    """The boundary is JST midnight — at 2026-03-31T20:00Z it is already
    April 1 in Japan and the new fiscal year's file is the live one."""
    url = scraper.expand_url("https://x/{FY_JP}.csv",
                             now=dt.datetime(2026, 3, 31, 20, 0, tzinfo=dt.timezone.utc))
    assert url == "https://x/2026.csv"
