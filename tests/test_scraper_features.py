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
