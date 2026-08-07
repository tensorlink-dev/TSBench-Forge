"""Tests for bulk.py — the deterministic Socrata candidate generator.

Everything here is offline: the column-picking, cadence-inference, id-minting
and URL-building logic is pure, and the network layer is exercised through
`synthesize` with a hand-built probe dict. The point of these tests is that a
bad guess in the *generator* is expensive — it produces dozens of catalog
entries at once — so the mechanical choices need pinning down.
"""

from __future__ import annotations

import datetime as dt
import re

import pytest
import yaml

from source_discovery import bulk


# --------------------------------------------------------------------------- #
# timestamp column choice
# --------------------------------------------------------------------------- #
def test_picks_the_only_date_column():
    cols = [("Case ID", "case_id", "number"),
            ("Reported", "reported_date", "calendar date"),
            ("Type", "type", "text")]
    assert bulk.pick_timestamp_column(cols) == "reported_date"


def test_prefers_event_time_over_row_bookkeeping():
    """`updated_at` is when the ROW changed; `created_date` is when the thing
    happened. Choosing the former yields a series of ETL runs, not of events."""
    cols = [("Updated", "updated_at", "floating timestamp"),
            ("Created", "created_date", "floating timestamp")]
    assert bulk.pick_timestamp_column(cols) == "created_date"


def test_no_date_column_returns_none():
    cols = [("Ward", "ward", "number"), ("Name", "name", "text")]
    assert bulk.pick_timestamp_column(cols) is None


def test_text_column_named_date_is_not_a_timestamp():
    """Datatype governs, not the name — a text 'date' column may hold anything
    and SoQL cannot $order it meaningfully."""
    cols = [("Date", "date", "text")]
    assert bulk.pick_timestamp_column(cols) is None


# --------------------------------------------------------------------------- #
# value column choice
# --------------------------------------------------------------------------- #
def test_picks_numeric_observation_columns():
    cols = [("When", "when", "calendar date"),
            ("Gallons", "gallons", "number"),
            ("Cost", "cost", "money")]
    assert bulk.pick_value_columns(cols, "when") == ["gallons", "cost"]


@pytest.mark.parametrize("field", [
    "incident_id", "zip", "latitude", "longitude", "council_district",
    "beat", "objectid", "fips", "year", "x_coordinate", "police_district",
])
def test_identifiers_and_coordinates_are_not_observations(field):
    """These are numbers but not measurements. A series of zip codes ordered by
    time is noise that would still pass a 'is it numeric' gate."""
    cols = [("When", "when", "calendar date"), ("F", field, "number")]
    assert bulk.pick_value_columns(cols, "when") == []


def test_value_columns_capped():
    cols = [("When", "when", "calendar date")] + [
        (f"V{i}", f"amount_{i}", "number") for i in range(9)
    ]
    assert len(bulk.pick_value_columns(cols, "when", limit=3)) == 3


def test_timestamp_column_never_doubles_as_a_value():
    cols = [("When", "when", "number"), ("Amt", "amt", "number")]
    assert bulk.pick_value_columns(cols, "when") == ["amt"]


# --------------------------------------------------------------------------- #
# cadence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gap_s,expected", [
    (30, "PT1M"), (60, "PT1M"), (300, "PT5M"), (900, "PT15M"),
    (1800, "PT30M"), (3600, "PT1H"), (86400, "P1D"),
    (604800, "P1W"), (2592000, "P1M"), (7776000, "P1Q"), (31536000, "P1Y"),
])
def test_freq_from_delta(gap_s, expected):
    assert bulk.freq_from_delta(gap_s) == expected


def test_freq_bands_match_the_coverage_vocabulary():
    """Generated frequencies must be labels config.FREQ_BAND knows, or the
    sources land outside the coverage matrix that steers the whole build."""
    from source_discovery import config
    for gap in (10, 100, 500, 1000, 2000, 5000, 90000, 700000, 3000000,
                9000000, 40000000):
        assert bulk.freq_from_delta(gap) in config.FREQ_BAND


def test_poll_cadence_follows_publication_lag_not_series_cadence():
    """A 5-minute series published in a nightly batch must not be polled every
    5 minutes — that is ~288 identical fetches a day."""
    assert bulk.cron_cadence_for("PT5M", age_days=0.0) == "PT5M"
    assert bulk.cron_cadence_for("PT5M", age_days=9.0) == "P1D"
    assert bulk.cron_cadence_for("PT1M", age_days=30.0) == "P1D"


def test_slow_series_still_polled_daily():
    """A weekly series needs a daily poll to catch its publication promptly."""
    assert bulk.cron_cadence_for("P1W", age_days=0.0) == "P1D"
    assert bulk.cron_cadence_for("P1M", age_days=0.0) == "P1D"


# --------------------------------------------------------------------------- #
# ids
# --------------------------------------------------------------------------- #
def test_host_label_skips_platform_labels():
    """Every city portal is 'data.<city>.gov'; keying on the first label would
    collapse them all to 'data_' and collide."""
    assert bulk.host_label("data.everettwa.gov") == "everettwa"
    assert bulk.host_label("cos-data.seattle.gov") == "seattle"
    assert bulk.host_label("data.cityofnewyork.us") == "cityofnewyork"
    assert bulk.host_label("opendata.maryland.gov") == "maryland"


def test_entry_ids_differ_across_portals_for_the_same_title():
    a = bulk.entry_id("data.everettwa.gov", "911 Calls For Service")
    b = bulk.entry_id("data.kcmo.org", "911 Calls For Service")
    assert a != b and a and b


def test_entry_id_is_slug_safe():
    sid = bulk.entry_id("data.foo.gov", "Fire/EMS Calls (2024 & forward!)")
    assert sid.replace("_", "").isalnum()
    assert len(sid) <= 64


# --------------------------------------------------------------------------- #
# url building
# --------------------------------------------------------------------------- #
def test_build_url_orders_newest_first():
    """$order is mandatory alongside $limit on Socrata: without it the page is
    arbitrary, which is how a source ends up storing 2016 data forever."""
    url = bulk.build_url("data.x.gov", "ab12-cd34", "created_date", ["units"])
    assert "$order=created_date%20DESC" in url
    assert "$limit=5000" in url
    assert url.startswith("https://data.x.gov/resource/ab12-cd34.json?")


def test_build_url_selects_only_needed_columns():
    url = bulk.build_url("d.gov", "a1b2-c3d4", "ts", ["v1", "v2"])
    assert "$select=ts,v1,v2" in url


def test_probe_url_is_timestamp_only():
    url = bulk.probe_url("d.gov", "a1b2-c3d4", "ts")
    assert "$select=ts&" in url and "$limit=60" in url


# --------------------------------------------------------------------------- #
# synthesis
# --------------------------------------------------------------------------- #
def _result(cols, host="data.testcity.gov", rid="ab12-cd34", title="Test Feed"):
    names, fields, types = zip(*cols) if cols else ([], [], [])
    return {
        "metadata": {"domain": host},
        "resource": {
            "id": rid, "name": title,
            "columns_name": list(names),
            "columns_field_name": list(fields),
            "columns_datatype": list(types),
            "updatedAt": "2026-07-27T06:00:00.000Z",
        },
    }


_PROBE = {"rows": 60, "distinct": 60, "future": 0,
          "newest": "2026-07-27T06:00:00+00:00", "age_days": 0.1,
          "median_gap_s": 300.0}
_KLASS = ("911 calls", "healthcare", "emergency_dispatch")


def test_synthesize_emits_a_parseable_wire_block():
    res = _result([("When", "created_date", "calendar date"),
                   ("Units", "units_sent", "number")])
    block = bulk.synthesize(res, _KLASS, _PROBE)
    assert block["wireable"] is True
    entry = yaml.safe_load(block["yaml_block"])[0]
    assert entry["domain"] == "healthcare"
    assert entry["frequency"] == "PT5M"
    assert entry["schema"]["timestamp_field"] == "[].created_date"
    assert entry["schema"]["value_field"] == ["[].units_sent"]
    assert entry["endpoint"]["auth"] == "none"


def test_synthesize_bins_a_count_when_there_is_no_numeric_column():
    """An event log with no measurement column is still a time series — the
    event rate. Left unaggregated it would store text and contribute nothing."""
    res = _result([("When", "created_date", "calendar date"),
                   ("Type", "call_type", "text")])
    entry = yaml.safe_load(bulk.synthesize(res, _KLASS, _PROBE)["yaml_block"])[0]
    assert entry["schema"]["aggregate"] == {"op": "count", "bin": "PT1H"}
    assert entry["frequency"] == "PT1H"


def test_synthesize_slack_is_never_tighter_than_observed_age():
    """Declaring a slack window tighter than the age already observed is the
    classic self-inflicted rejection."""
    probe = {**_PROBE, "age_days": 180.0}
    res = _result([("When", "created_date", "calendar date"),
                   ("N", "n", "number")])
    entry = yaml.safe_load(bulk.synthesize(res, _KLASS, probe)["yaml_block"])[0]
    assert entry["audit_slack_days"] >= 180 + 45


def test_synthesize_returns_none_without_a_timestamp():
    res = _result([("Ward", "ward", "number")])
    assert bulk.synthesize(res, _KLASS, _PROBE) is None


def test_synthesize_avoids_id_collisions():
    res = _result([("When", "created_date", "calendar date"),
                   ("N", "n", "number")])
    first = yaml.safe_load(bulk.synthesize(res, _KLASS, _PROBE)["yaml_block"])[0]
    second = yaml.safe_load(
        bulk.synthesize(res, _KLASS, _PROBE, taken={first["id"]})["yaml_block"]
    )[0]
    assert second["id"] != first["id"]
    assert len(second["id"]) <= 64


def test_generated_entry_has_every_field_the_catalog_requires():
    res = _result([("When", "created_date", "calendar date"),
                   ("N", "n", "number")])
    entry = yaml.safe_load(bulk.synthesize(res, _KLASS, _PROBE)["yaml_block"])[0]
    for key in ("id", "name", "domain", "dgp_class", "archetypes", "frequency",
                "endpoint", "schema", "pretraining_novelty", "license"):
        assert key in entry, f"missing {key}"


# --------------------------------------------------------------------------- #
# dedupe helpers
# --------------------------------------------------------------------------- #
def test_wired_resource_ids_extracts_socrata_4x4s(tmp_path):
    cat = tmp_path / "sources.yaml"
    cat.write_text(yaml.dump([
        {"id": "a", "endpoint": {"url": "https://data.x.gov/resource/ab12-cd34.json?$limit=5"}},
        {"id": "b", "endpoint": {"url": "https://example.org/feed.csv"}},
    ]))
    assert bulk.wired_resource_ids(str(cat)) == {"ab12-cd34"}


def test_wired_hosts_includes_resolve_urls(tmp_path):
    cat = tmp_path / "sources.yaml"
    cat.write_text(yaml.dump([
        {"id": "a", "endpoint": {"url": "https://one.example/x.json",
                                 "resolve": {"url": "https://two.example/meta"}}},
    ]))
    assert bulk.wired_hosts(str(cat)) == {"one.example", "two.example"}


# --------------------------------------------------------------------------- #
# keyword classes
# --------------------------------------------------------------------------- #
def test_keyword_classes_use_the_catalog_domain_vocabulary():
    allowed = {"energy", "econ_fin", "web_cloudops", "healthcare", "nature",
               "transport", "sales"}
    for kw, dom, dgp in bulk.KEYWORD_CLASSES:
        assert dom in allowed, f"{kw} -> unknown domain {dom}"
        assert kw and dgp


def test_keyword_classes_avoid_contaminated_dataset_names():
    """The sweep must not go fishing in the keyword families that name known
    TSFM pretraining sets, or bulk volume quietly imports contamination."""
    from source_discovery import config
    hot = {"traffic", "electricity", "weather", "exchange rate", "illness"}
    for kw, _dom, _dgp in bulk.KEYWORD_CLASSES:
        assert not (hot & set(kw.lower().split())), kw
        assert kw.lower() not in config.CONTAMINATION_DENYLIST


# --------------------------------------------------------------------------- #
# probe timestamp parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("s,expected", [
    ("2026-07-27T06:00:00.000", dt.datetime(2026, 7, 27, 6, tzinfo=dt.timezone.utc)),
    ("2026-07-27", dt.datetime(2026, 7, 27, tzinfo=dt.timezone.utc)),
    ("2026-07-27 06:30", dt.datetime(2026, 7, 27, 6, 30, tzinfo=dt.timezone.utc)),
])
def test_parse_iso(s, expected):
    assert bulk._parse_iso(s) == expected


@pytest.mark.parametrize("s", ["", None, "not a date", "27/07/2026"])
def test_parse_iso_rejects_junk(s):
    assert bulk._parse_iso(s) is None


# --------------------------------------------------------------------------- #
# relevance / classification
# --------------------------------------------------------------------------- #
def test_off_topic_result_is_reclassified_not_inherited():
    """Full-text relevance decays past the first page: a 'short term rental'
    query returns campaign-finance filings. Inheriting the query's domain would
    poison the domain x cadence matrix that steers the entire build."""
    query = ("short term rental registrations", "sales", "registration_stream")
    got = bulk.resolve_class(query, "Campaign Contributions Received By Candidates")
    assert got is not None and got[1] == "econ_fin"


def test_on_topic_result_keeps_the_query_class():
    query = ("911 calls", "healthcare", "emergency_dispatch")
    assert bulk.resolve_class(query, "911 Calls For Service 2024") == query


def test_police_data_is_classified_not_discarded():
    """Crime/CAD feeds arrive under all sorts of queries and are good series —
    they just need the right domain rather than the bin."""
    query = ("call center wait times", "web_cloudops", "service_latency")
    got = bulk.resolve_class(query, "Richmond Police Department - CAD Events")
    assert got is not None and got[1] == "healthcare"


def test_unclassifiable_result_returns_none():
    query = ("api requests", "web_cloudops", "api_traffic")
    assert bulk.resolve_class(query, "Zzzz Qqqq") is None


def test_resolve_class_only_returns_known_domains():
    allowed = {"energy", "econ_fin", "web_cloudops", "healthcare", "nature",
               "transport", "sales"}
    for kw, dom, dgp in bulk.EXTRA_CLASSES:
        assert dom in allowed, f"{kw} -> {dom}"
        assert kw and dgp


def test_is_relevant_ignores_stopwords():
    """Matching on 'data' or 'city' would let anything through."""
    assert not bulk.is_relevant("transit ridership data", "City Data Portal")
    assert bulk.is_relevant("transit ridership data", "Daily Ridership Totals")


# --------------------------------------------------------------------------- #
# Opendatasoft
# --------------------------------------------------------------------------- #
def _ods_rec(fields, host="opendata.tpg.ch", dsid="freq-par-tranche",
             title="Ridership per day", records=64191,
             processed="2026-07-26T00:02:30.510000+00:00"):
    return {
        "dataset_id": f"{dsid}@tpg",
        "has_records": True,
        "fields": [{"name": n, "type": t, "label": lab} for lab, n, t in fields],
        "metas": {"default": {
            "title": title,
            "source_domain_address": host,
            "source_dataset": dsid,
            "records_count": records,
            "data_processed": processed,
        }},
    }


_ODS_PROBE = {"rows": 100, "distinct": 100, "future": 0,
              "newest": "2026-07-26T00:00:00+00:00", "age_days": 0.5,
              "median_gap_s": 86400.0}
_ODS_KLASS = ("transit ridership", "transport", "ridership")


def test_ods_picks_date_field_and_numeric_values():
    rec = _ods_rec([("Date", "date", "date"),
                    ("Day Week", "jour_semaine", "text"),
                    ("Boardings", "nb_de_montees", "double")])
    fields = bulk._ods_fields(rec)
    assert bulk.ods_pick_timestamp(fields) == "date"
    assert bulk.ods_pick_values(fields, "date") == ["nb_de_montees"]


def test_ods_rejects_french_identifier_columns():
    """`code_insee` and `numero_commune` are numbers, not measurements."""
    rec = _ods_rec([("D", "date", "date"),
                    ("INSEE", "code_insee", "int"),
                    ("Num", "numero_ligne", "int")])
    assert bulk.ods_pick_values(bulk._ods_fields(rec), "date") == []


def test_ods_timestamp_picker_understands_non_english_names():
    rec = _ods_rec([("Maj", "derniere_modification", "datetime"),
                    ("Horodate", "horodate", "datetime")])
    assert bulk.ods_pick_timestamp(bulk._ods_fields(rec)) == "horodate"


def test_ods_entry_points_at_the_publishers_own_host():
    """All ODS datasets are reachable through data.opendatasoft.com, but wiring
    them there would collapse every European publisher into one host — and the
    coverage metric credits at most a handful of sources per host."""
    rec = _ods_rec([("Date", "date", "date"), ("N", "nb_de_montees", "double")])
    block = bulk.ods_synthesize(rec, _ODS_KLASS, _ODS_PROBE)
    entry = yaml.safe_load(block["yaml_block"])[0]
    assert "opendata.tpg.ch" in entry["endpoint"]["url"]
    assert "data.opendatasoft.com" not in entry["endpoint"]["url"]


def test_ods_export_url_is_newest_first_and_bounded():
    url = bulk.ods_export_url("opendata.tpg.ch", "ds", "date", ["v"])
    assert "order_by=date%20desc" in url
    assert f"limit={bulk.ODS_EXPORT_ROWS}" in url
    assert "/exports/json?" in url
    assert "select=date,v" in url


def test_ods_synthesize_emits_a_valid_wire_block():
    rec = _ods_rec([("Date", "date", "date"), ("N", "nb_de_montees", "double")])
    block = bulk.ods_synthesize(rec, _ODS_KLASS, _ODS_PROBE)
    entry = yaml.safe_load(block["yaml_block"])[0]
    assert entry["domain"] == "transport"
    assert entry["frequency"] == "P1D"
    assert entry["schema"]["timestamp_field"] == "[].date"
    assert entry["schema"]["value_field"] == ["[].nb_de_montees"]
    assert entry["audit_slack_days"] >= 45


def test_ods_synthesize_bins_count_without_numeric_fields():
    rec = _ods_rec([("Date", "date", "date"), ("Type", "type", "text")])
    entry = yaml.safe_load(
        bulk.ods_synthesize(rec, _ODS_KLASS, _ODS_PROBE)["yaml_block"]
    )[0]
    assert entry["schema"]["aggregate"]["op"] == "count"


def test_ods_synthesize_needs_a_publisher_host():
    rec = _ods_rec([("Date", "date", "date")], host="")
    assert bulk.ods_synthesize(rec, _ODS_KLASS, _ODS_PROBE) is None


def test_wired_ods_datasets_dedupe(tmp_path):
    cat = tmp_path / "sources.yaml"
    cat.write_text(yaml.dump([{"id": "a", "endpoint": {"url":
        "https://opendata.tpg.ch/api/explore/v2.1/catalog/datasets/freq-x/exports/json?limit=10"}}]))
    assert bulk.wired_ods_datasets(str(cat)) == {"freq-x"}


# --------------------------------------------------------------------------- #
# CKAN — linked file resources
# --------------------------------------------------------------------------- #
def test_ckan_text_handles_multilingual_fields():
    """Several national portals return {'en': ..., 'nl': ...} where CKAN's
    schema says string; calling .strip() on that crashed a whole sweep."""
    assert bulk._text({"nl": "Luchtkwaliteit", "en": "Air quality"}) == "Air quality"
    assert bulk._text({"fr": "Qualité"}) == "Qualité"
    assert bulk._text("plain") == "plain"
    assert bulk._text(None) == ""


def test_csv_timestamp_column_chosen_from_the_data():
    """A raw file has no declared types, so the values must answer for
    themselves — a text column merely NAMED 'date' proves nothing."""
    header = ["station", "date", "value"]
    body = [["A", f"2026-07-{d:02d}", str(d)] for d in range(1, 21)]
    assert bulk._csv_timestamp_column(header, body) == "date"


def test_csv_timestamp_column_rejects_unparseable():
    header = ["name", "note"]
    body = [["A", "hello"] for _ in range(20)]
    assert bulk._csv_timestamp_column(header, body) is None


def test_csv_numeric_column_detection():
    header = ["d", "n", "t"]
    body = [[f"2026-07-{i:02d}", str(i * 1.5), "text"] for i in range(1, 21)]
    assert bulk._csv_column_is_numeric(header, body, "n")
    assert not bulk._csv_column_is_numeric(header, body, "t")


def test_sniff_csv_detects_semicolon_delimiter():
    text = "date;value\n2026-07-01;1\n2026-07-02;2\n"
    rows = bulk._sniff_csv_rows(text)
    assert rows[0] == ["date", "value"]
    assert rows[1] == ["2026-07-01", "1"]


def test_sniff_csv_drops_truncated_last_line():
    """The head fetch cuts mid-record; a partial row would corrupt the sniff."""
    text = "date,value,comment\n2026-07-01,1,aaaaaaaaaaaaaaaaaaaa\n2026-0"
    rows = bulk._sniff_csv_rows(text)
    assert len(rows) == 2


def test_ckan_file_candidate_rejects_non_csv():
    got = bulk.ckan_file_candidate({"format": "PDF", "url": "https://x/y.pdf"}, 21)
    assert "error" in got and "CSV" in got["error"]


def test_ckan_file_candidate_rejects_stale_metadata():
    got = bulk.ckan_file_candidate(
        {"format": "CSV", "url": "https://x/y.csv",
         "last_modified": "2020-01-01T00:00:00"}, 21)
    assert "stale" in got["error"]


def test_ckan_file_candidate_rejects_oversized():
    got = bulk.ckan_file_candidate(
        {"format": "CSV", "url": "https://x/y.csv", "size": 99_000_000,
         "last_modified": "2026-07-27T00:00:00"}, 21)
    assert "too large" in got["error"]


def test_ckan_file_entry_reads_the_file_directly():
    """A linked file is fetched as CSV with its own column names — the
    result.records[] wrapper only exists for the DataStore API."""
    res = {"id": "r1", "name": "Air quality", "format": "CSV",
           "url": "https://portal.example/aq.csv"}
    fields = [("date", "date", "timestamp"), ("pm25", "pm25", "numeric")]
    probe = {"rows": 100, "distinct": 100, "future": 0,
             "newest": "2026-07-27T00:00:00+00:00", "age_days": 1.0,
             "median_gap_s": 3600.0}
    block = bulk.ckan_synthesize("data.gov.ie", "Ireland", {"title": "AQ"}, res,
                                 ("air quality monitoring", "nature", "air_quality"),
                                 fields, probe)
    entry = yaml.safe_load(block["yaml_block"])[0]
    assert entry["endpoint"]["type"] == "rest_csv"
    assert entry["endpoint"]["url"] == "https://portal.example/aq.csv"
    assert entry["schema"]["timestamp_field"] == "date"
    assert entry["schema"]["value_field"] == ["pm25"]


def test_ckan_datastore_entry_keeps_the_records_wrapper():
    res = {"id": "r1", "name": "AQ", "datastore_active": True}
    fields = [("date", "date", "timestamp"), ("pm25", "pm25", "numeric")]
    probe = {"rows": 100, "distinct": 100, "future": 0,
             "newest": "2026-07-27T00:00:00+00:00", "age_days": 1.0,
             "median_gap_s": 3600.0}
    block = bulk.ckan_synthesize("data.gov.ie", "Ireland", {"title": "AQ"}, res,
                                 ("air quality monitoring", "nature", "air_quality"),
                                 fields, probe)
    entry = yaml.safe_load(block["yaml_block"])[0]
    assert entry["schema"]["timestamp_field"] == "result.records[].date"
    assert entry["schema"]["value_field"] == ["result.records[].pm25"]


def test_resource_title_ignores_format_names():
    """Resource names are frequently just the format, which would produce ids
    like `ie_csv` that say nothing and collide across every package."""
    pkg = {"title": "Air Quality Monitoring"}
    assert bulk._resource_title({"name": "CSV"}, pkg) == "Air Quality Monitoring"
    assert bulk._resource_title({"name": "Download"}, pkg) == "Air Quality Monitoring"
    assert bulk._resource_title({}, pkg) == "Air Quality Monitoring"


def test_resource_title_keeps_distinguishing_names():
    """When one package holds several series, the resource name is what tells
    them apart and must survive."""
    pkg = {"title": "Air Quality Monitoring"}
    got = bulk._resource_title({"name": "Station Dublin PM10"}, pkg)
    assert got == "Air Quality Monitoring — Station Dublin PM10"


def test_resource_title_avoids_duplicating_the_package_name():
    pkg = {"title": "Air Quality Monitoring"}
    got = bulk._resource_title({"name": "air quality monitoring"}, pkg)
    assert got == "Air Quality Monitoring"


def test_host_counts_seed_the_cap_from_the_catalog(tmp_path):
    """The cap must count what the catalog already holds. Counting only within a
    run lets an already-full host collect more sources past the point where the
    coverage metric credits them — volume that reads as breadth and is not."""
    cat = tmp_path / "sources.yaml"
    cat.write_text(yaml.dump([
        {"id": "a", "endpoint": {"url": "https://data.x.gov/resource/aa11-bb22.json"}},
        {"id": "b", "endpoint": {"url": "https://data.x.gov/resource/cc33-dd44.json"}},
        {"id": "c", "endpoint": {"url": "https://other.gov/f.csv"}},
        {"id": "d", "disabled": True,
         "endpoint": {"url": "https://data.x.gov/resource/ee55-ff66.json"}},
    ]))
    counts = bulk.host_counts(str(cat))
    assert counts["data.x.gov"] == 2      # the disabled one does not count
    assert counts["other.gov"] == 1


@pytest.mark.parametrize("raw,expect", [
    ("2026-07-27", (2026, 7, 27)),
    ("27/07/2026", (2026, 7, 27)),
    ("27-07-2026", (2026, 7, 27)),
    ("01.03.2026", (2026, 3, 1)),
    ("20260727", (2026, 7, 27)),
])
def test_parse_loose_date(raw, expect):
    """European portals overwhelmingly write DD/MM/YYYY; an ISO-only sniffer
    finds no timestamp column on most of them and the portal reads as unusable."""
    got = bulk._parse_loose_date(raw)
    assert (got.year, got.month, got.day) == expect


@pytest.mark.parametrize("raw", ["", None, "hello", "13/13/2026"])
def test_parse_loose_date_rejects_junk(raw):
    assert bulk._parse_loose_date(raw) is None


def test_csv_timestamp_column_accepts_day_first_dates():
    header = ["station", "datum", "waarde"]
    body = [["A", f"{d:02d}/07/2026", str(d)] for d in range(1, 21)]
    assert bulk._csv_timestamp_column(header, body) == "datum"


def test_ckan_data_url_sorts_newest_first():
    url = bulk.ckan_data_url("data.gov.ie", "abc-123", "date", ["v"])
    assert "sort=date%20desc" in url
    assert "resource_id=abc-123" in url
    assert "fields=date,v" in url


def test_ckan_file_candidate_accepts_a_fresh_csv(monkeypatch):
    """End-to-end on the file path: a fresh CSV whose head parses should yield
    a timestamp column, numeric fields and a usable probe."""
    csv_text = "station;datum;pm25\n" + "".join(
        f"A;{d:02d}/07/2026;{d}.5\n" for d in range(1, 26)
    )
    monkeypatch.setattr(bulk, "ckan_fetch_head", lambda url, timeout=30: csv_text)
    got = bulk.ckan_file_candidate(
        {"format": "CSV", "url": "https://p.example/aq.csv",
         "last_modified": "2026-07-27T00:00:00"}, 3650)
    assert "error" not in got
    assert got["ts"] == "datum"
    assert ("pm25", "pm25", "numeric") in got["fields"]
    assert got["probe"]["distinct"] == 25


def test_ckan_file_candidate_uses_content_length_when_size_missing(monkeypatch):
    """CKAN's `size` is frequently absent. Without a HEAD check the file gets
    downloaded twice before anyone learns it is over the scraper's 48MB cap."""
    monkeypatch.setattr(bulk, "_content_length", lambda url, timeout=20: 60_000_000)
    got = bulk.ckan_file_candidate(
        {"format": "CSV", "url": "https://p.example/big.csv",
         "last_modified": "2026-07-27T00:00:00"}, 3650)
    assert "48MB cap" in got["error"]


# --------------------------------------------------------------------------- #
# ERDDAP
# --------------------------------------------------------------------------- #
def _erddap_payload(names, rows):
    return {"table": {"columnNames": names, "rows": rows}}


def test_erddap_pick_values_drops_qc_and_coordinates():
    """On a typical mooring the QC flags OUTNUMBER the measurements, and every
    one of them is numeric. Wiring a flag column yields a constant series."""
    variables = [
        ("time", "double"), ("station", "String"),
        ("latitude", "float"), ("longitude", "float"), ("depth", "double"),
        ("air_temperature", "float"), ("air_temperature_qc", "byte"),
        ("wind_speed", "float"), ("wind_speed_qc_agg", "byte"),
        ("crs", "double"), ("instrument_1", "byte"), ("time_modified", "double"),
    ]
    assert bulk.erddap_pick_values(variables) == ["air_temperature", "wind_speed"]


def test_erddap_pick_values_caps_the_width():
    variables = [(f"var_{i}", "float") for i in range(10)]
    assert len(bulk.erddap_pick_values(variables, limit=4)) == 4


def test_erddap_data_url_is_server_relative():
    """`now-Nd` is evaluated server-side, so a generated entry never needs a
    date token re-templated and cannot silently pin to a fixed past window."""
    url = bulk.erddap_data_url("https://e.example/erddap", "A01_met",
                               ["air_temperature"], 2)
    assert url == ("https://e.example/erddap/tabledap/A01_met.json"
                   "?time,air_temperature&time%3E=now-2days")


def test_erddap_rejects_model_output_titles():
    for title in ("Irish Marine Institute NE Atlantic Model",
                  "Wave forecast, 7 day", "SST climatology 1991-2020",
                  "Data from a local source."):
        assert bulk._ERDDAP_REJECT_TITLE.search(title), title
    for title in ("A01 Met - Meteorology", "Halifax (Herring Cove) Buoy"):
        assert not bulk._ERDDAP_REJECT_TITLE.search(title), title


def test_erddap_rejects_one_off_deployments():
    """A glider deployment is fresh today and dead when the glider is recovered;
    the id carries its launch stamp."""
    assert bulk._ERDDAP_DEPLOYMENT.search("AsterSEA068-20260804T0908")
    assert not bulk._ERDDAP_DEPLOYMENT.search("A01_met")


def test_erddap_platform_key_groups_one_buoys_instruments():
    """Three depths of the same mooring are the same water. Taking all three
    buys volume and no information."""
    k = bulk.erddap_platform_key
    assert k("A01 Directional Waves", "A01_waves") == "a01"
    assert k("A01 Ocean - 1 meter", "A01_ocean_001m") == "a01"
    assert k("(41024 / SUN2) Sunset Nearshore", "org_cormp_sun2") == "41024"
    # Prose titles carry the distinguishing part at the END, so keying on the
    # leading token there would collapse genuinely separate radar sites.
    assert (k("Near Real Time Surface Velocity by HFR-Granitola", "a")
            != k("Near Real Time Surface Velocity by HFR-Galicia", "b"))


def test_erddap_slack_scales_with_cadence():
    """A flat 45-day floor lets a 10-minute buoy stay 'fresh' for six weeks
    after it goes dark."""
    assert bulk._erddap_slack(600, 0.02) == 3          # 10-minute mooring
    assert bulk._erddap_slack(86_400, 1.0) == 14       # daily
    assert bulk._erddap_slack(2_600_000, 10.0) == 55   # monthly


def test_erddap_datasets_filters_grids_and_the_catalog_row(monkeypatch):
    payload = _erddap_payload(
        ["datasetID", "title", "institution", "minTime", "maxTime", "dataStructure"],
        [["allDatasets", "*", "*", "", "", "table"],
         ["gridThing", "SST grid", "X", "", "", "grid"],
         ["A01_met", "A01 Met", "UMaine", "2001-01-01T00:00:00Z",
          "2026-08-07T00:00:00Z", "table"]],
    )
    monkeypatch.setattr(bulk, "erddap_get",
                        lambda url, timeout=60, max_bytes=0: payload)
    got = bulk.erddap_datasets("https://e.example/erddap")
    assert [d["datasetID"] for d in got] == ["A01_met"]


def test_erddap_probe_rejects_a_panel(monkeypatch):
    """Several stations sharing a timestamp is a panel; the gate reads one
    series per entry, so the extra rows silently overwrite each other."""
    base = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)
    rows = [[(base + dt.timedelta(days=i // 3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
             float(i)] for i in range(60)]
    monkeypatch.setattr(bulk, "erddap_get",
                        lambda url, timeout=60: _erddap_payload(["time", "v"], rows))
    got = bulk.erddap_probe("https://e.example/erddap", "ds", ["v"])
    assert "panel" in got["error"]


def test_erddap_probe_rejects_future_dated_rows(monkeypatch):
    future = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=30)
    rows = [[(future + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
             float(i)] for i in range(40)]
    monkeypatch.setattr(bulk, "erddap_get",
                        lambda url, timeout=60: _erddap_payload(["time", "v"], rows))
    got = bulk.erddap_probe("https://e.example/erddap", "ds", ["v"])
    assert "future-dated" in got["error"]


def test_erddap_probe_drops_all_null_columns(monkeypatch):
    now = dt.datetime.now(dt.timezone.utc)
    rows = [[(now - dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
             float(i), None] for i in range(40)]
    monkeypatch.setattr(
        bulk, "erddap_get",
        lambda url, timeout=60: _erddap_payload(["time", "good", "empty"], rows))
    got = bulk.erddap_probe("https://e.example/erddap", "ds", ["good", "empty"])
    assert got["values"] == ["good"]
    assert got["distinct"] == 40


def test_erddap_probe_widens_the_window_for_slow_series(monkeypatch):
    """A monthly series has nothing in a 2-day window; giving up there would
    drop every low-cadence dataset on the federation."""
    now = dt.datetime.now(dt.timezone.utc)
    calls: list[int] = []

    def fake(url, timeout=60):
        days = int(re.search(r"now-(\d+)days", url).group(1))
        calls.append(days)
        if days < 120:
            return _erddap_payload(["time", "v"], [])
        rows = [[(now - dt.timedelta(days=3 * i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 float(i)] for i in range(30)]
        return _erddap_payload(["time", "v"], rows)

    monkeypatch.setattr(bulk, "erddap_get", fake)
    got = bulk.erddap_probe("https://e.example/erddap", "ds", ["v"])
    assert calls == [2, 14, 120]
    assert got["window_days"] == 120


def test_erddap_synthesize_uses_positional_fields():
    """ERDDAP answers with parallel arrays, so the schema is positional: column
    0 is always the requested `time`."""
    probe = {"values": ["air_temperature", "wind_speed"], "window_days": 2,
             "distinct": 288, "rows": 288, "newest": "2026-08-07T00:00:00+00:00",
             "age_days": 0.01, "median_gap_s": 600.0}
    block = bulk.erddap_synthesize(
        "https://data.neracoos.org/erddap", "NERACOOS",
        {"datasetID": "A01_met", "title": "A01 Met", "institution": "UMaine",
         "minTime": "2001-01-01T00:00:00Z"},
        bulk.ERDDAP_DEFAULT_CLASS, probe)
    entry = yaml.safe_load(block["yaml_block"])[0]
    assert entry["schema"]["timestamp_field"] == "table.rows[][0]"
    assert entry["schema"]["value_field"] == ["table.rows[][1]", "table.rows[][2]"]
    assert entry["id"].startswith("neracoos_")
    assert entry["frequency"] == "PT15M"   # nearest rung of the ladder
    assert entry["audit_slack_days"] == 3


def test_wired_erddap_datasets_reads_host_and_id():
    import tempfile, os
    reg = [{"id": "x", "endpoint": {"url": "https://data.neracoos.org/erddap/"
                                           "tabledap/A01_met.json?time&x=1"}}]
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as fh:
        yaml.dump(reg, fh)
    try:
        assert bulk.wired_erddap_datasets(path) == {"data.neracoos.org|A01_met"}
    finally:
        os.unlink(path)


def test_host_label_walks_past_country_tlds():
    """Without this a European host stops on its TLD: 'eu' names a continent,
    not a publisher, and every .eu source would collide."""
    assert bulk.host_label("erddap.emodnet-physics.eu") == "emodnet"
    assert bulk.host_label("erddap.marine.ie") == "marine"


def test_erddap_class_never_leaves_the_nature_domain():
    """An ocean federation has one domain, so every cross-domain match the
    open-vocabulary classifier finds is a false one: the token 'real' in 'Near
    Real Time' filed NOAA ship met data under sales, and 'RADS Altimeter' under
    econ_fin. A mislabelled domain corrupts the matrix that steers the build."""
    for title in ("NOAA Ship Bell M. Shimada Underway Meteorological Data, "
                  "Near Real Time",
                  "RADS Altimeter Along-Track Data",
                  "028 - Santa Monica Bay, CA (46221)"):
        assert bulk.erddap_class(title)[1] == "nature", title


def test_erddap_class_picks_the_sub_class():
    assert bulk.erddap_class("A01 Directional Waves")[2] == "ocean_waves"
    assert bulk.erddap_class("Halifax tide gauge water level")[2] == "sea_level"


def test_erddap_datasets_treats_404_as_an_empty_catalog(monkeypatch):
    """ERDDAP answers an empty result set with HTTP 404. Reported as an error it
    reads as 'server down' and hides the real finding: the freshness window was
    too tight."""
    def boom(url, timeout=60, max_bytes=0):
        raise RuntimeError("HTTP 404")
    monkeypatch.setattr(bulk, "erddap_get", boom)
    assert bulk.erddap_datasets("https://e.example/erddap") == []

    def other(url, timeout=60, max_bytes=0):
        raise RuntimeError("HTTP 500")
    monkeypatch.setattr(bulk, "erddap_get", other)
    with pytest.raises(RuntimeError):
        bulk.erddap_datasets("https://e.example/erddap")


# --------------------------------------------------------------------------- #
# GBFS
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, payload, status=200, text=""):
        self._payload = payload
        self.status_code = status
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")


def _stations(n, field="num_bikes_available", clock=True, docks=True):
    out = []
    for i in range(n):
        s = {"station_id": f"s{i}", field: i % 7}
        if clock:
            s["last_reported"] = 1_770_000_000 + i
        if docks:
            s["num_docks_available"] = 10 - (i % 7)
        out.append(s)
    return {"data": {"stations": out}, "ttl": 60}


def test_gbfs_feeds_reads_both_layouts(monkeypatch):
    """Pre-3.0 nests the feed list under a language key, 3.0 does not. Assuming
    either one silently loses half the registry."""
    pre = {"data": {"en": {"feeds": [{"name": "station_status",
                                      "url": "https://a/ss.json"}]}}}
    v3 = {"data": {"feeds": [{"name": "station_status",
                              "url": "https://b/ss.json"}]}}
    for payload, want in ((pre, "https://a/ss.json"), (v3, "https://b/ss.json")):
        monkeypatch.setattr(bulk.requests, "get",
                            lambda *a, **k: _FakeResp(payload))
        assert bulk.gbfs_feeds("https://x/gbfs.json")["station_status"] == want


def test_gbfs_feeds_falls_back_to_an_unlisted_language(monkeypatch):
    payload = {"data": {"sv": {"feeds": [{"name": "station_status",
                                          "url": "https://c/ss.json"}]}}}
    monkeypatch.setattr(bulk.requests, "get", lambda *a, **k: _FakeResp(payload))
    assert bulk.gbfs_feeds("https://x/gbfs.json")["station_status"] == "https://c/ss.json"


def test_gbfs_probe_accepts_the_v3_availability_spelling(monkeypatch):
    monkeypatch.setattr(bulk.requests, "get", lambda *a, **k: _FakeResp(
        _stations(30, field="num_vehicles_available")))
    got = bulk.gbfs_probe("https://x/ss.json")
    assert got["stations"] == 30
    assert got["value_fields"][0] == "num_vehicles_available"


def test_gbfs_probe_rejects_a_panel_below_the_gate(monkeypatch):
    """The wire gate admits a panel at >=20 entities; 19 docks is a rejection,
    not a thin candidate."""
    monkeypatch.setattr(bulk.requests, "get",
                        lambda *a, **k: _FakeResp(_stations(19)))
    assert "20 entities" in bulk.gbfs_probe("https://x/ss.json")["error"]


def test_gbfs_probe_requires_a_clock(monkeypatch):
    """Without last_reported every dock shares the fetch time, so the series has
    no timestamp of its own."""
    monkeypatch.setattr(bulk.requests, "get",
                        lambda *a, **k: _FakeResp(_stations(40, clock=False)))
    assert "last_reported" in bulk.gbfs_probe("https://x/ss.json")["error"]


def test_gbfs_probe_rejects_a_free_floating_payload(monkeypatch):
    monkeypatch.setattr(bulk.requests, "get", lambda *a, **k: _FakeResp(
        {"data": {"bikes": [{"bike_id": "b1"}]}}))
    assert "stations" in bulk.gbfs_probe("https://x/ss.json")["error"]


def test_gbfs_synthesize_keys_the_panel_on_station_id():
    system = {"Name": "Test Bikes", "Location": "Springfield",
              "Country Code": "US", "System ID": "test_bikes"}
    block = bulk.gbfs_synthesize(system, "https://gbfs.example/ss.json",
                                 {"stations": 40, "ttl": 60,
                                  "value_fields": ["num_bikes_available",
                                                   "num_docks_available"]})
    entry = yaml.safe_load(block["yaml_block"])[0]
    assert entry["schema"]["panel_field"] == "data.stations[].station_id"
    assert entry["schema"]["timestamp_field"] == "data.stations[].last_reported"
    assert entry["schema"]["variates"] == 2
    assert entry["domain"] == "transport"
    assert entry["frequency"] == "PT5M"
    assert len(entry["id"]) <= 64


def test_gbfs_synthesize_disambiguates_a_taken_id():
    """Two systems can share an operator host and a city name; a colliding id
    would overwrite a wired source rather than add one."""
    system = {"Name": "Test Bikes", "Location": "Springfield",
              "Country Code": "US", "System ID": "test_bikes_2"}
    first = yaml.safe_load(bulk.gbfs_synthesize(
        system, "https://gbfs.example/ss.json",
        {"stations": 40, "ttl": None, "value_fields": ["num_bikes_available"]},
    )["yaml_block"])[0]["id"]
    second = yaml.safe_load(bulk.gbfs_synthesize(
        system, "https://gbfs.example/ss.json",
        {"stations": 40, "ttl": None, "value_fields": ["num_bikes_available"]},
        taken={first},
    )["yaml_block"])[0]["id"]
    assert second != first and len(second) <= 64


# --------------------------------------------------------------------------- #
# ODS, publisher-first
# --------------------------------------------------------------------------- #
def _facet_payload(names):
    return {"facets": [{"name": "source_domain_address",
                        "facets": [{"name": n, "count": 1} for n in names]}]}


def test_ods_publishers_recurses_only_into_full_prefixes(monkeypatch):
    """A page of exactly 100 means the facet was truncated; anything less is the
    whole prefix and recursing into it would cost 37 requests for nothing."""
    calls = []

    def fake(url, params=None, headers=None, timeout=0):
        where = (params or {}).get("where", "")
        calls.append(where)
        if not where:                                  # root: full, so truncated
            return _FakeResp(_facet_payload([f"h{i}.example" for i in range(100)]))
        if 'startswith(source_domain_address, "a")' == where:
            return _FakeResp(_facet_payload(["only-one.example"]))
        return _FakeResp(_facet_payload([]))

    monkeypatch.setattr(bulk.requests, "get", fake)
    monkeypatch.setattr(bulk.time, "sleep", lambda s: None)
    got = bulk.ods_publishers(sleep_s=0, log=lambda m: None)
    assert "only-one.example" in got
    assert len(got) == 101                              # 100 root + the a-prefix one
    # One request per alphabet letter at depth 1, and no deeper.
    assert len(calls) == 1 + len(bulk.ODS_PREFIX_ALPHABET)


def test_ods_publishers_stops_at_max_depth(monkeypatch):
    """A prefix that is still full at max depth must return what it has rather
    than recurse forever."""
    monkeypatch.setattr(bulk.requests, "get", lambda *a, **k: _FakeResp(
        _facet_payload([f"h{i}.example" for i in range(bulk.ODS_FACET_CAP)])))
    monkeypatch.setattr(bulk.time, "sleep", lambda s: None)
    warned = []
    got = bulk.ods_publishers(sleep_s=0, max_requests=200, log=warned.append)
    assert len(got) == bulk.ODS_FACET_CAP
    # Every prefix full at every level is the pathological case: 37 per level.
    # It must come back with a budget warning, not 1.8M requests.
    assert any("budget" in w for w in warned)


def test_ods_publishers_survives_a_dead_facet_call(monkeypatch):
    monkeypatch.setattr(bulk.requests, "get", lambda *a, **k: _FakeResp({}, status=500))
    assert bulk.ods_publishers(sleep_s=0, log=lambda m: None) == []


def test_ods_publisher_datasets_orders_newest_first(monkeypatch):
    seen = {}

    def fake(url, params=None, headers=None, timeout=0):
        seen.update(params or {})
        return _FakeResp({"results": [{"dataset_id": "d1"}]})

    monkeypatch.setattr(bulk.requests, "get", fake)
    bulk.ods_publisher_datasets("portal.example", want=7)
    assert seen["where"] == 'source_domain_address="portal.example"'
    # `metas.default.data_processed desc` is a 400 on this endpoint.
    assert seen["order_by"] == "data_processed desc"
    assert seen["limit"] == 7


def test_resolve_class_without_a_query_class():
    """Publisher-first has no keyword to inherit a domain from, so query_class is
    None and the title must carry it alone."""
    got = bulk.resolve_class(None, "Hourly electricity consumption by feeder")
    assert got is not None and got[1] in {
        "energy", "nature", "econ_fin", "transport", "sales", "healthcare",
        "web_cloudops"}
    assert bulk.resolve_class(None, "") is None


def test_ods_publisher_class_uses_the_theme_not_the_title():
    """The failures that motivated the closed map, all from one real sweep:
    without a theme these titles scored a single accidental token against an
    English keyword list and landed in the wrong domain."""
    assert bulk.ods_publisher_class(["Energy"],
        "Hourly electricity consumption by feeder")[1] == "energy"
    assert bulk.ods_publisher_class(["Hydrography"],
        "Daily river water level gauge readings")[1] == "nature"
    assert bulk.ods_publisher_class(["Santé"],
        "Passages aux urgences par jour")[1] == "healthcare"


def test_ods_publisher_class_skips_what_it_cannot_place():
    # Unmapped theme: real data, but no class in this build describes it.
    assert bulk.ods_publisher_class(["Orthoimagery"], "Aerial photos 2019") is None
    assert bulk.ods_publisher_class([], "No theme") is None
    assert bulk.ods_publisher_class(None, "No theme") is None
    # Two themes pointing at different domains is a coin flip, so don't flip it.
    assert bulk.ods_publisher_class(["Economy", "Transport"], "Mixed") is None
    # Two spellings of ONE domain is not a disagreement.
    assert bulk.ods_publisher_class(["Transport", "Mobilité"], "Bus")[1] == "transport"


def test_ods_publisher_class_accepts_a_bare_string_theme():
    assert bulk.ods_publisher_class("Energy", "Grid load")[1] == "energy"


def test_ods_publisher_class_falls_back_to_a_domain_default():
    """A theme the map knows plus a title no keyword class matches is still a
    usable source; it just gets the generic class for its domain."""
    got = bulk.ods_publisher_class(["Electricity"], "Zzzz qqqq wwww")
    assert got[1] == "energy"
    assert got[2] == bulk.ODS_DOMAIN_DEFAULT_CLASS["energy"]


def test_ods_theme_map_only_emits_known_domains():
    DOMAINS = {"nature", "econ_fin", "transport", "energy", "sales",
               "healthcare", "web_cloudops"}
    assert set(bulk.ODS_THEME_DOMAIN.values()) <= DOMAINS
    assert set(bulk.ODS_DOMAIN_DEFAULT_CLASS) == DOMAINS
    # Keys are matched lowercased; a capitalised key would never fire.
    assert all(k == k.lower() for k in bulk.ODS_THEME_DOMAIN)


def test_ods_rejects_games_of_chance():
    """A lottery draw passes every structural check -- clean timestamp, dense
    numeric column, daily cadence -- and is i.i.d. uniform by construction."""
    for t in ("Résultats Loto", "Keno winning numbers", "EuroMillions tirage",
              "dedicated-integers", "random walk sample_data"):
        assert bulk._ODS_REJECT_TITLE.search(t), t
    # Only the draw itself is the problem. A grants ledger that happens to be
    # lottery-funded is a real spending series and several UK portals publish
    # one, so "lottery" on its own must not be a rejection.
    for t in ("Lottery Fund grant payments to community groups",
              "National Lottery Community Fund awards by month",
              "Random sample survey of household energy use"):
        assert not bulk._ODS_REJECT_TITLE.search(t), t


def test_ods_theme_patterns_cover_a_portals_own_vocabulary():
    """Portals define their own themes, so exact values are not enough: these
    are real values that the exact-match map alone skipped."""
    cases = [("Solidarité et soutien à l'activité", "healthcare"),
             ("Petite enfance", "healthcare"),
             ("Handicap et autonomie", "healthcare"),
             ("Logement", "sales"),
             ("Mobiliteit", "transport"),
             ("Itinerarios y horarios", "transport"),
             ("Hydrographie", "nature")]
    for theme, want in cases:
        got = bulk.ods_publisher_class([theme], "a title")
        assert got is not None and got[1] == want, (theme, got)


def test_ods_theme_patterns_still_skip_the_contentless_ones():
    """A theme that names no subject must stay a skip; the whole point of a
    closed vocabulary is that it declines rather than guesses."""
    for theme in ("Données générales", "International", "Institution",
                  "Données de référence", "Orthoimagery", "Elevation",
                  "No Theme"):
        assert bulk.ods_publisher_class([theme], "a title") is None, theme


def test_ods_theme_patterns_are_valid_and_on_known_domains():
    DOMAINS = {"nature", "econ_fin", "transport", "energy", "sales",
               "healthcare", "web_cloudops"}
    for pat, dom in bulk.ODS_THEME_PATTERNS:
        re.compile(pat)
        assert dom in DOMAINS, pat


def test_ods_theme_pattern_ambiguity_is_a_skip():
    # One theme string hitting two domains is a coin flip, not a signal.
    assert bulk.ods_publisher_class(["Transport et Énergie"], "x") is None


def test_ods_rejects_registry_timestamps():
    """A publisher walk sees the whole portal, and an open-data portal is mostly
    registries. A registry built incrementally has a per-row created/updated
    column that looks exactly like an observation clock -- half of the first
    publisher-first batch got in this way."""
    for ts in ("date_der_maj", "c_date_instal", "date_mise_en_service",
               "date_creation", "creationtime", "creationdate", "date_demande",
               "published"):
        assert bulk._ODS_RECORD_TS.search(ts), ts
    for ts in ("date_time", "moment", "depart", "timestamp", "periode",
               "date_visite_diagnostique", "datenotification",
               "dernier_jour_du_trimestre"):
        assert not bulk._ODS_RECORD_TS.search(ts), ts


def test_ods_rejects_identifier_columns_as_values():
    for v in ("x", "y", "gid", "c_xy_precis", "said", "organisationnumber",
              "uiccountrycode", "numbershort", "code_insee"):
        assert bulk._ODS_ID_VAL.search(v), v
    for v in ("solar_radiation", "air_temperature", "production_biomethane",
              "aqi", "reg_avg_value", "nbre_pdc", "montant", "conso_chauffage",
              "number_of_passengers", "nombre_de_velos"):
        assert not bulk._ODS_ID_VAL.search(v), v


def test_ods_title_override_beats_a_departmental_theme():
    """Publishers file by department: a health agency tags its air-quality
    monitors 'Santé'. The measured process is still air quality."""
    got = bulk.ods_publisher_class(["Santé"], "Qualité de l'air par heure")
    assert got[1] == "nature" and got[2] == "air_quality"
    got = bulk.ods_publisher_class(["Environnement"], "Prix des Énergies en France")
    assert got[1] == "econ_fin"


def test_ods_record_ts_catches_a_glued_maj():
    """`datemajobs` is a last-updated column with no separators. It got a river
    obstacle registry through the first pass of these filters."""
    for ts in ("datemajobs", "majdate", "date_maj", "equip_maj_date"):
        assert bulk._ODS_RECORD_TS.search(ts), ts
    # ...but "maj" glued into an unrelated word must not fire.
    for ts in ("majoration_tarifaire", "major_incident_time"):
        assert not bulk._ODS_RECORD_TS.search(ts), ts


def test_ods_id_val_rejects_dublin_core_and_area_codes():
    for v in ("dc_coverage_temporal_periodoftime", "dc_date", "cdcommune",
              "code_departement", "wmo"):
        assert bulk._ODS_ID_VAL.search(v), v


def test_ods_epc_and_dpe_land_in_the_same_domain():
    """The same process in two languages was landing in two domains -- the
    English EPC in nature, the French DPE in sales -- purely on theme."""
    en = bulk.ods_publisher_class(
        ["Environment"], "Domestic Energy Performance Certificate (EPC) Data")
    fr = bulk.ods_publisher_class(
        ["Logement"], "Diagnostics de Performances Énergétiques (DPE)")
    assert en[1] == fr[1] == "energy"
    assert en[2] == fr[2] == "building_energy_rating"
