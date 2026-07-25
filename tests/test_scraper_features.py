"""Unit tests for the scraper's composite-timestamp and field-paneling features."""
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
