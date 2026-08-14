"""Unit tests for source_discovery.monitor (pipeline health, not data freshness).

The checks here are all "does the report say something *true* about a tree of
files on disk", so each test builds the smallest tree that provokes the case.
"""

from __future__ import annotations

import datetime as dt

import pytest
import yaml

from source_discovery import monitor

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 8, 14, 12, 0, tzinfo=UTC)


def _entry(sid: str, url: str = "https://api.example.com/x", **kw) -> dict:
    return {"id": sid, "domain": kw.pop("domain", "nature"),
            "endpoint": {"url": url}, **kw}


def _write_errors(data_dir, sid: str, lines: list[str]) -> None:
    d = data_dir / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / "_errors.log").write_text("\n".join(lines) + "\n")


def _touch_parquet(data_dir, sid: str, day: dt.date) -> None:
    d = data_dir / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{day}.parquet").write_bytes(b"")


# --------------------------------------------------------------------------- #
# _host_of — the bug that collapsed every host into one nameless bucket
# --------------------------------------------------------------------------- #

def test_host_of_reads_the_raw_catalog_shape() -> None:
    # coverage.host_of wants the flattened registry key and returns "" here;
    # the per-host rollup is worthless if this regresses to that.
    assert monitor._host_of(
        {"endpoint": {"url": "https://api.energy-charts.info/price?bzn=DE"}}
    ) == "api.energy-charts.info"


def test_host_of_still_accepts_the_registry_shape() -> None:
    assert monitor._host_of(
        {"url_or_endpoint": "https://example.gov/a"}) == "example.gov"


def test_host_of_never_returns_empty() -> None:
    # An unresolved {TEMPLATE} url has no netloc; it must still group somewhere
    # rather than silently merging with every other hostless entry as "".
    assert monitor._host_of({"endpoint": {"url": "{BASE}/data"}}) == "?"
    assert monitor._host_of({}) == "?"


# --------------------------------------------------------------------------- #
# _reason — failures only aggregate if the label collapses
# --------------------------------------------------------------------------- #

def test_reason_collapses_http_status_across_distinct_urls() -> None:
    a = monitor._reason("HTTP 429: Too Many Requests")
    b = monitor._reason("HTTP 429: Too Many Requests for url https://x/y?z=1")
    assert a == b == "HTTP 429"


def test_reason_labels_transport_failures() -> None:
    assert monitor._reason("The read operation timed out") == "timeout"
    assert monitor._reason("ConnectTimeout") == "timeout"
    # An errno string is already short and already groupable, and says more
    # than a generic bucket would; it is kept rather than relabelled.
    assert monitor._reason("[Errno 101] Network is unreachable") == \
        "[Errno 101] Network is unreachable"


def test_reason_keeps_the_leading_clause_of_other_exceptions() -> None:
    assert monitor._reason("Expecting value: line 1 column 1") == "Expecting value"


# --------------------------------------------------------------------------- #
# heartbeats
# --------------------------------------------------------------------------- #

def test_heartbeat_flags_a_job_that_stopped(tmp_path) -> None:
    log = tmp_path / "_cron_all.log"
    log.write_text("x\n")
    import os
    old = (NOW - dt.timedelta(hours=3)).timestamp()
    os.utime(log, (old, old))

    got = {h["job"]: h for h in monitor.heartbeats(tmp_path, None, now=NOW)}
    assert got["scrape_all"]["status"] == "late"
    # The every-minute job has no log at all here.
    assert got["scrape_fast"]["status"] == "missing"


def test_heartbeat_accepts_a_job_inside_its_grace(tmp_path) -> None:
    log = tmp_path / "_cron_all.log"
    log.write_text("x\n")
    import os
    recent = (NOW - dt.timedelta(minutes=20)).timestamp()
    os.utime(log, (recent, recent))
    got = {h["job"]: h for h in monitor.heartbeats(tmp_path, None, now=NOW)}
    # 20 min > the 15 min period, but one period of grace covers it.
    assert got["scrape_all"]["status"] == "ok"


def test_heartbeat_age_is_never_negative(tmp_path) -> None:
    # A job writing its log during the report would otherwise print "-0.0m ago".
    log = tmp_path / "_cron_all.log"
    log.write_text("x\n")
    import os
    ahead = (NOW + dt.timedelta(seconds=30)).timestamp()
    os.utime(log, (ahead, ahead))
    got = {h["job"]: h for h in monitor.heartbeats(tmp_path, None, now=NOW)}
    assert got["scrape_all"]["age_s"] == 0.0


def test_ops_jobs_are_skipped_when_there_is_no_ops_log_dir(tmp_path) -> None:
    # Off the scrape host those three logs do not exist, and reporting them
    # missing would make every developer machine look broken.
    names = {h["job"] for h in monitor.heartbeats(tmp_path, None, now=NOW)}
    assert "grind" not in names and "audit" not in names


# --------------------------------------------------------------------------- #
# sweeps
# --------------------------------------------------------------------------- #

SWEEP = ("2026-08-14 02:12:06,935 INFO sweep finished: "
         "1768 ok, 0 failed, 27 skipped of 1795 in 709s (8 workers)")


def test_sweeps_parses_the_finish_line(tmp_path) -> None:
    (tmp_path / "_cron_all.log").write_text(SWEEP + "\n")
    got = monitor.sweeps(tmp_path)
    assert got["recent"] == [{"at": "2026-08-14 02:12:06", "ok": 1768,
                              "failed": 0, "skipped": 27, "total": 1795,
                              "seconds": 709.0}]
    assert got["skipped_last"] == 27


def test_sweeps_finds_the_line_behind_a_sweep_in_flight(tmp_path) -> None:
    # The regression this guards: a running sweep writes megabytes of INFO
    # after the last finish line, so a fixed 400KB tail sees none of them and
    # the report used to declare the pipeline dead.
    noise = "2026-08-14 02:20:00,000 INFO fetch something -> https://x/y\n"
    (tmp_path / "_cron_all.log").write_text(
        SWEEP + "\n" + noise * 20_000)
    assert monitor.sweeps(tmp_path)["recent"][-1]["ok"] == 1768


def test_sweeps_falls_back_to_the_rotated_log(tmp_path) -> None:
    # Just after the hourly rotation the live log is empty or tiny.
    (tmp_path / "_cron_all.log").write_text("2026-08-14 03:00:00,0 INFO hi\n")
    (tmp_path / "_cron_all.log.1").write_text(SWEEP + "\n")
    assert monitor.sweeps(tmp_path)["recent"][-1]["skipped"] == 27


def test_sweeps_reports_nothing_rather_than_guessing(tmp_path) -> None:
    (tmp_path / "_cron_all.log").write_text("no sweeps here\n")
    got = monitor.sweeps(tmp_path)
    assert got["recent"] == [] and got["skipped_last"] is None


# --------------------------------------------------------------------------- #
# failures
# --------------------------------------------------------------------------- #

def test_failures_group_by_host_not_by_source(tmp_path) -> None:
    # The whole value of the check: 60 errors on one API must read as one
    # rate-limited host, not as 30 unrelated broken sources.
    catalog = [_entry(f"ec_{i}", "https://api.energy-charts.info/price")
               for i in range(30)]
    for i in range(30):
        _write_errors(tmp_path, f"ec_{i}", [
            f"{(NOW - dt.timedelta(minutes=5)).isoformat()} FETCH "
            "HTTP 429: Too Many Requests" for _ in range(2)])

    got = monitor.failures(catalog, tmp_path, now=NOW)
    assert got["total"] == 60
    assert got["sources_affected"] == 30
    assert [h["host"] for h in got["culprit_hosts"]] == ["api.energy-charts.info"]
    assert got["culprit_hosts"][0]["reasons"] == {"HTTP 429": 60}


def test_failures_ignore_entries_outside_the_window(tmp_path) -> None:
    catalog = [_entry("old")]
    _write_errors(tmp_path, "old", [
        f"{(NOW - dt.timedelta(days=9)).isoformat()} FETCH HTTP 500: boom"])
    assert monitor.failures(catalog, tmp_path, now=NOW)["total"] == 0


def test_failures_skip_logs_untouched_since_the_cutoff(tmp_path) -> None:
    # The mtime pre-filter is an optimisation; it must not drop live errors.
    catalog = [_entry("fresh")]
    _write_errors(tmp_path, "fresh", [
        f"{(NOW - dt.timedelta(minutes=1)).isoformat()} FETCH HTTP 503: down"])
    # mtime is "now" (just written), so the file is read despite the filter.
    assert monitor.failures(catalog, tmp_path, now=NOW)["total"] == 1


def test_failures_tolerate_a_disabled_source_with_no_log(tmp_path) -> None:
    assert monitor.failures([_entry("nothing")], tmp_path, now=NOW)["total"] == 0


# --------------------------------------------------------------------------- #
# silence — the check that catches what audit.py structurally cannot
# --------------------------------------------------------------------------- #

def _cron(tmp_path, groups: dict[str, list[str]]):
    p = tmp_path / "cron.yaml"
    p.write_text(yaml.safe_dump(
        {"schedules": [{"every": k, "sources": v} for k, v in groups.items()]}))
    return p


def test_silence_finds_a_source_that_never_wrote(tmp_path) -> None:
    # A never-written source has no observation to be stale, so the freshness
    # audit passes over it and the catalog counts it live forever.
    cron = _cron(tmp_path, {"PT15M": ["ghost"]})
    got = monitor.silence([_entry("ghost")], tmp_path, cron, now=NOW)
    assert got["never_written"] == 1
    assert got["never_written_ids"] == ["ghost"]
    assert got["writing"] == 0


def test_silence_accepts_a_file_written_today_or_yesterday(tmp_path) -> None:
    cron = _cron(tmp_path, {"PT15M": ["a", "b"]})
    _touch_parquet(tmp_path, "a", NOW.date())
    _touch_parquet(tmp_path, "b", NOW.date() - dt.timedelta(days=1))
    got = monitor.silence([_entry("a"), _entry("b")], tmp_path, cron, now=NOW)
    assert got["writing"] == 2 and got["not_writing"] == 0


def test_silence_reports_the_gap_in_days(tmp_path) -> None:
    cron = _cron(tmp_path, {"PT15M": ["slow"]})
    _touch_parquet(tmp_path, "slow", NOW.date() - dt.timedelta(days=17))
    got = monitor.silence([_entry("slow")], tmp_path, cron, now=NOW)
    assert got["not_writing_worst"] == [
        {"id": "slow", "days": 17, "domain": "nature", "host": "api.example.com"}]


def test_silence_gives_a_slower_group_its_own_period(tmp_path) -> None:
    # A P7D source two days without a file is on schedule; the same gap on a
    # 15-minute source is not.
    cron = _cron(tmp_path, {"P7D": ["weekly"], "PT15M": ["fast"]})
    for sid in ("weekly", "fast"):
        _touch_parquet(tmp_path, sid, NOW.date() - dt.timedelta(days=2))
    got = monitor.silence([_entry("weekly"), _entry("fast")], tmp_path, cron,
                          now=NOW)
    assert [r["id"] for r in got["not_writing_worst"]] == ["fast"]


def test_a_source_missing_from_cron_is_still_judged(tmp_path) -> None:
    # Regression: this module used to call ids absent from cron.yaml
    # "unscheduled — will never be polled" and exempt them from every check.
    # The 15-min sweep walks sources.yaml and gates on `frequency`, so those
    # ids are polled like any other; exempting them hid real breakage behind
    # a fault that did not exist.
    cron = _cron(tmp_path, {"PT15M": ["scheduled"]})
    _touch_parquet(tmp_path, "scheduled", NOW.date())
    got = monitor.silence(
        [_entry("scheduled"), _entry("orphan", frequency="PT5M")],
        tmp_path, cron, now=NOW)
    assert got["never_written"] == 1
    assert got["never_written_ids"] == ["orphan"]


def test_a_source_missing_from_cron_uses_its_own_frequency(tmp_path) -> None:
    # A yearly source with no cron entry must get a yearly allowance, not the
    # one-day default that would flag it the moment it is a week old.
    cron = _cron(tmp_path, {"PT15M": ["other"]})
    _touch_parquet(tmp_path, "yearly", NOW.date() - dt.timedelta(days=200))
    _touch_parquet(tmp_path, "other", NOW.date())
    got = monitor.silence(
        [_entry("yearly", frequency="P1Y"), _entry("other")],
        tmp_path, cron, now=NOW)
    assert got["not_writing"] == 0 and got["writing"] == 2


# --------------------------------------------------------------------------- #
# undersampled — snapshot feeds recording slower than they claim
# --------------------------------------------------------------------------- #

def test_undersampled_flags_a_snapshot_outside_the_fast_band(tmp_path) -> None:
    # No timestamp field: the scraper stamps arrival, so one poll is one point
    # and the source's resolution is exactly its polling rate.
    p = tmp_path / "cron.yaml"
    p.write_text(yaml.safe_dump({"schedules": [
        {"every": "PT1M", "cron": "* * * * *", "sources": ["in_band"]},
        {"every": "PT5M", "cron": "*/5 * * * *", "sources": ["out_of_band"]}]}))
    catalog = [_entry("in_band", frequency="PT1M"),
               _entry("out_of_band", frequency="PT1M")]
    got = monitor.undersampled(catalog, p)
    assert got["count"] == 1
    assert got["sources"][0]["id"] == "out_of_band"


def test_undersampled_ignores_feeds_that_carry_history(tmp_path) -> None:
    # A response with a real timestamp path holds the last N minutes, so a
    # 15-minute poll loses nothing — those must not be flagged.
    p = tmp_path / "cron.yaml"
    p.write_text(yaml.safe_dump({"schedules": [
        {"every": "PT1M", "cron": "* * * * *", "sources": []}]}))
    catalog = [_entry("windowed", frequency="PT1M",
                      schema={"timestamp_field": "[].time_tag"})]
    assert monitor.undersampled(catalog, p)["count"] == 0


def test_undersampled_ignores_slow_sources(tmp_path) -> None:
    p = tmp_path / "cron.yaml"
    p.write_text(yaml.safe_dump({"schedules": [
        {"every": "PT1M", "cron": "* * * * *", "sources": []}]}))
    assert monitor.undersampled(
        [_entry("hourly", frequency="PT1H")], p)["count"] == 0


def test_fast_band_reads_only_the_every_minute_group(tmp_path) -> None:
    p = tmp_path / "cron.yaml"
    p.write_text(yaml.safe_dump({"schedules": [
        {"every": "PT1M", "cron": "* * * * *", "sources": ["a", "b"]},
        {"every": "PT5M", "cron": "*/5 * * * *", "sources": ["c"]},
        {"every": "P1D", "sources": ["d"]}]}))
    assert monitor.fast_band(p) == {"a", "b"}


# --------------------------------------------------------------------------- #
# build / findings / severity
# --------------------------------------------------------------------------- #

def _healthy_tree(tmp_path):
    import os

    data = tmp_path / "data"
    data.mkdir()
    (data / "_cron_all.log").write_text(SWEEP.replace("27 skipped", "0 skipped")
                                        + "\n")
    (data / "_cron_fast.log").write_text("ok\n")
    (data / "_cron_sync.log").write_text("ok\n")
    # Heartbeats are read from mtimes, so they must be pinned to the fake
    # clock — otherwise the tree is "healthy" only while the wall clock
    # happens to sit near NOW.
    for name in ("_cron_all.log", "_cron_fast.log", "_cron_sync.log"):
        os.utime(data / name, (NOW.timestamp(), NOW.timestamp()))
    catalog = tmp_path / "sources.yaml"
    catalog.write_text(yaml.safe_dump([_entry("a")]))
    cron = _cron(tmp_path, {"PT15M": ["a"]})
    _touch_parquet(data, "a", NOW.date())
    return catalog, data, cron


def test_build_reports_ok_on_a_healthy_tree(tmp_path) -> None:
    catalog, data, cron = _healthy_tree(tmp_path)
    report = monitor.build(catalog, data, cron, ops_log_dir=None, now=NOW)
    assert report["status"] == "ok"
    assert report["findings"] == []
    assert "no findings" in monitor.render(report)


def test_build_calls_a_stopped_cron_critical(tmp_path) -> None:
    catalog, data, cron = _healthy_tree(tmp_path)
    import os
    old = (NOW - dt.timedelta(days=2)).timestamp()
    os.utime(data / "_cron_fast.log", (old, old))
    report = monitor.build(catalog, data, cron, ops_log_dir=None, now=NOW)
    assert report["status"] == "critical"
    assert report["findings"][0]["check"] == "heartbeat"


def test_a_sweep_in_flight_is_not_an_outage(tmp_path) -> None:
    # Fresh deploy: the job is running but has not finished a pass yet.
    import os

    catalog, data, cron = _healthy_tree(tmp_path)
    (data / "_cron_all.log").write_text("2026-08-14 11:59:00,0 INFO starting\n")
    os.utime(data / "_cron_all.log", (NOW.timestamp(), NOW.timestamp()))
    report = monitor.build(catalog, data, cron, ops_log_dir=None, now=NOW)
    sweep = [f for f in report["findings"] if f["check"] == "sweep"]
    assert sweep and sweep[0]["severity"] == "warn"
    assert report["status"] == "warn"


def test_findings_name_the_worst_hosts_and_count_the_rest(tmp_path) -> None:
    # A long tail of marginal hosts must not crowd out the real culprit.
    catalog, data, cron = _healthy_tree(tmp_path)
    entries = [_entry("a")]
    for i in range(monitor.FINDING_HOSTS + 3):
        sid = f"h{i}"
        entries.append(_entry(sid, f"https://host{i}.example/x"))
        _touch_parquet(data, sid, NOW.date())
        _write_errors(data, sid, [
            f"{(NOW - dt.timedelta(minutes=3)).isoformat()} FETCH HTTP 500: x"
            for _ in range(monitor.HOST_CALLOUT + i)])
    catalog.write_text(yaml.safe_dump(entries))
    cron.write_text(yaml.safe_dump({"schedules": [
        {"every": "PT15M", "sources": [e["id"] for e in entries]}]}))

    report = monitor.build(catalog, data, cron, ops_log_dir=None, now=NOW)
    named = [f for f in report["findings"] if f["check"] == "failures"]
    assert len(named) == monitor.FINDING_HOSTS + 1
    assert "3 further hosts" in named[-1]["detail"]
    # Worst first, so the top line is the host with the most errors.
    assert "host7.example" in named[0]["detail"]


def test_findings_put_critical_before_warn(tmp_path) -> None:
    catalog, data, cron = _healthy_tree(tmp_path)
    import os
    old = (NOW - dt.timedelta(days=2)).timestamp()
    os.utime(data / "_cron_fast.log", (old, old))
    _write_errors(data, "a", [
        f"{(NOW - dt.timedelta(minutes=3)).isoformat()} FETCH HTTP 500: x"
        for _ in range(monitor.HOST_CALLOUT)])
    report = monitor.build(catalog, data, cron, ops_log_dir=None, now=NOW)
    sev = [f["severity"] for f in report["findings"]]
    assert sev == sorted(sev, key=lambda s: {"critical": 0, "warn": 1}[s])


def test_summarize_is_one_line(tmp_path) -> None:
    catalog, data, cron = _healthy_tree(tmp_path)
    line = monitor.summarize(
        monitor.build(catalog, data, cron, ops_log_dir=None, now=NOW))
    assert "\n" not in line and line.startswith("monitor 2026-08-14")


def test_poll_periods_reads_cron_groups(tmp_path) -> None:
    cron = _cron(tmp_path, {"PT1M": ["fast"], "P1D": ["slow"]})
    assert monitor.poll_periods(cron) == {"fast": 60, "slow": 86_400}


def test_poll_periods_skips_unparseable_cadences(tmp_path) -> None:
    p = tmp_path / "cron.yaml"
    p.write_text(yaml.safe_dump({"schedules": [
        {"every": "irregular", "sources": ["x"]},
        {"every": "PT5M", "sources": ["y"]}]}))
    assert monitor.poll_periods(p) == {"y": 300}
