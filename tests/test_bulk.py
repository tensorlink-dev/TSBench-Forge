"""Tests for bulk.py — the deterministic Socrata candidate generator.

Everything here is offline: the column-picking, cadence-inference, id-minting
and URL-building logic is pure, and the network layer is exercised through
`synthesize` with a hand-built probe dict. The point of these tests is that a
bad guess in the *generator* is expensive — it produces dozens of catalog
entries at once — so the mechanical choices need pinning down.
"""

from __future__ import annotations

import datetime as dt

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
