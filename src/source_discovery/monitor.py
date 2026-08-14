"""monitor.py — is the PIPELINE running? (``audit.py`` asks: is the DATA fresh?)

The two questions look alike and fail differently. A source can be perfectly
fresh on disk while the scraper has stopped reaching it: the last good fetch is
still the newest observation, and stays that way until it drifts past the
cadence threshold days later. Conversely a source can be scraped faithfully
every fifteen minutes and still be stale, because the upstream stopped
publishing. ``audit.py`` catches the second. Nothing caught the first.

What makes that gap load-bearing is that the sweep's own tally cannot report
it. ``scraper.run_one`` catches fetch and parse failures per panel-row and
``continue``s, so the exception never reaches ``_run_target``, ``counts
["failed"]`` stays 0, and the process exits 0. A sweep that logged 574 HTTP
429s in one log rotation still signs off as ``1768 ok, 0 failed``. Every
failure signal here is therefore read from the per-source ``_errors.log`` that
``scraper.log_error`` appends to — never from the sweep line.

Four checks, ordered by how things actually break:

* ``heartbeats`` — did each cron job run at all? A skipped job is invisible in
  the data for hours, and its log mtime is the cheapest available witness.
* ``sweeps`` — recent sweep outcomes, including how many sources the deadline
  left unstarted. Chronic skipping is a scheduling fault, not an upstream one:
  the sweep no longer fits its window.
* ``failures`` — FETCH/PARSE errors in the window, grouped by source and by
  host. Grouping by host is what turns "1200 scattered errors" into "one
  rate-limited API", which is the form you can act on.
* ``silence`` — live sources the scraper has not written a byte for. This is
  the class that hid 69 dead energy-charts sources: a source that never wrote
  has no observation to be stale, so the freshness audit passes over it and
  the catalog counts it as live forever.

Deliberately stat-only. It lists directories and tails logs; it never opens a
parquet. ``audit.py`` reads all ~3000 of them and takes minutes, which is right
for a daily job and wrong for "is it running?". This runs in about three
seconds over a 3000-source catalog, most of it the YAML parse, which keeps it
worth typing ad hoc and cheap enough to schedule often.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import duration

UTC = dt.timezone.utc

# Default window for the failure rollup. One day covers ~96 full sweeps, so a
# host that fails intermittently still shows a count rather than a coin flip.
DEFAULT_WINDOW_HOURS = 24.0

# A host needs this many failures in the window before it is called out by
# name. Below it, one flaky endpoint would bury the one that is really down.
HOST_CALLOUT = 20

# How many culprit hosts to name in `findings` before collapsing to a count.
FINDING_HOSTS = 5


@dataclass(frozen=True)
class Job:
    """A cron job and the log it is expected to touch.

    ``period_s`` is the crontab schedule, not the tolerance — the grace period
    is derived in ``heartbeats`` so this table stays a statement of intent.
    """

    name: str
    log: str
    period_s: int
    ops: bool = False   # log lives in the ops log dir, not alongside the data


JOBS = (
    Job("scrape_fast", "_cron_fast.log", 60),
    Job("scrape_all", "_cron_all.log", 900),
    Job("sync_hippius", "_cron_sync.log", 900),
    Job("audit", "audit.log", 86_400, ops=True),
    Job("pool_report", "pool.log", 86_400, ops=True),
    Job("grind", "grind.log", 86_400, ops=True),
)


def _grace_s(period_s: int) -> float:
    """How late a job may be before it counts as missed.

    One whole extra period, capped at an hour of additional slack, so the
    every-minute job is called late at two minutes while the daily one gets
    until 25 hours rather than 48 — a daily job that skipped a day should be
    reported the same day, not the next one.
    """
    return period_s + min(period_s, 3600)


_SWEEP_RE = re.compile(
    r"^(?P<ts>\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d)[,.]\d+ \w+ sweep finished: "
    r"(?P<ok>\d+) ok, (?P<failed>\d+) failed, (?P<skipped>\d+) skipped "
    r"of (?P<total>\d+) in (?P<secs>[\d.]+)s"
)

# `scraper.log_error` writes "<iso> FETCH <exception> <panel row>". The kind is
# the only structured field; everything after it is an exception's str().
_ERR_RE = re.compile(r"^(?P<ts>\S+) (?P<kind>FETCH|PARSE|UNCAUGHT) (?P<msg>.*)$")

_HTTP_RE = re.compile(r"\bHTTP (\d{3})\b")
_PARQUET_RE = re.compile(r"^(\d{4}-\d\d-\d\d)\.parquet$")


def _tail(path: Path, nbytes: int = 400_000) -> list[str]:
    """Last lines of a possibly-huge log.

    ``_cron_all.log`` reaches 53MB before the hourly rotation, and the fast
    log grows past 49MB, so this seeks from the end instead of reading. The
    first line of the window is usually partial and is dropped.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return []
    try:
        with path.open("rb") as fh:
            if size > nbytes:
                fh.seek(size - nbytes)
            blob = fh.read()
    except OSError:
        return []
    text = blob.decode("utf-8", "replace")
    lines = text.splitlines()
    return lines[1:] if size > nbytes and lines else lines


def _reason(msg: str) -> str:
    """Collapse an exception string to a groupable label.

    The point is to make counts add up across a host: forty distinct
    ``HTTP 429: Too Many Requests`` strings carrying different URLs are one
    fact, and the URL belongs in the log, not in the tally.
    """
    m = _HTTP_RE.search(msg)
    if m:
        return f"HTTP {m.group(1)}"
    low = msg.lower()
    for needle, label in (("timed out", "timeout"), ("timeout", "timeout"),
                          ("connect", "connect error"), ("ssl", "TLS error"),
                          ("dns", "DNS failure"),
                          ("name or service not known", "DNS failure")):
        if needle in low:
            return label
    # Otherwise keep the exception's leading clause — enough to distinguish
    # a KeyError from a ValueError without carrying the whole payload.
    head = msg.split(":", 1)[0].strip()
    return head[:48] if head else "unknown"


def _parse_err_ts(raw: str) -> dt.datetime | None:
    try:
        stamp = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)


def load_catalog(catalog_path: str | Path) -> list[dict]:
    """Live (non-disabled) catalog entries.

    libyaml where available: ``sources.yaml`` is ~3000 entries and the pure
    Python loader spends 12s on it, which would be an order of magnitude more
    than every other check in this module put together.
    """
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    entries = yaml.load(Path(catalog_path).read_text(), Loader=loader) or []
    return [e for e in entries if not e.get("disabled")]


def _host_of(entry: dict) -> str:
    """Hostname from a RAW catalog entry.

    ``coverage.host_of`` reads ``url_or_endpoint``, which only exists after
    ``coverage.load_registry`` has flattened the entry; feeding it the catalog
    shape returns "" for every source and silently collapses the per-host
    rollup into one nameless bucket. Read ``endpoint.url`` directly, and accept
    the registry key too so either shape works.
    """
    from urllib.parse import urlparse

    url = (entry.get("endpoint") or {}).get("url") or entry.get("url_or_endpoint")
    return urlparse(url or "").netloc or "?"


def fast_band(cron_path: str | Path) -> set[str]:
    """Ids in cron.yaml's every-minute group.

    This is the only part of cron.yaml that schedules anything. ``scrape_fast
    .sh`` selects exactly the group whose ``cron`` is ``* * * * *``; every
    other group is descriptive.
    """
    doc = yaml.safe_load(Path(cron_path).read_text()) or {}
    return {s for g in doc.get("schedules") or []
            if g.get("cron") == "* * * * *" for s in g.get("sources") or []}


def poll_periods(cron_path: str | Path) -> dict[str, int]:
    """source id -> declared polling period in seconds, from cron.yaml.

    Read this as documentation, not as schedule. The 15-minute sweep runs
    ``scraper.py --all``, which walks *sources.yaml* and gates each entry on
    ``is_due(frequency)``; it never opens cron.yaml. So an id missing from
    every group here is still polled — a fact worth stating plainly, because
    an earlier version of this module reported those ids as "will never be
    polled" when all of them were being scraped on schedule.
    """
    doc = yaml.safe_load(Path(cron_path).read_text()) or {}
    out: dict[str, int] = {}
    for group in doc.get("schedules") or []:
        secs = duration.period_seconds(group.get("every") or "")
        if secs is None:
            continue
        for sid in group.get("sources") or []:
            # Last definition wins, matching how the scraper resolves an id
            # that appears in more than one group.
            out[sid] = secs
    return out


def heartbeats(data_dir: str | Path, ops_log_dir: str | Path | None,
               now: dt.datetime | None = None) -> list[dict]:
    """Per-job liveness from log mtimes."""
    now = now or dt.datetime.now(UTC)
    data_dir, out = Path(data_dir), []
    for job in JOBS:
        if job.ops and ops_log_dir is None:
            continue
        path = (Path(ops_log_dir) / job.log) if job.ops else (data_dir / job.log)
        try:
            mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, UTC)
        except OSError:
            out.append({"job": job.name, "log": str(path), "status": "missing",
                        "age_s": None, "period_s": job.period_s})
            continue
        # Clamped: a job writing to its log while this runs can stamp it a
        # fraction ahead of the `now` captured at the top of the report, and
        # "-0.0m ago" reads like a bug.
        age = max(0.0, (now - mtime).total_seconds())
        out.append({
            "job": job.name,
            "log": str(path),
            "status": "late" if age > _grace_s(job.period_s) else "ok",
            "age_s": round(age, 1),
            "last_run": mtime.isoformat(timespec="seconds"),
            "period_s": job.period_s,
        })
    return out


def _scan_sweeps(path: Path, window: int) -> list[dict]:
    rows = []
    for line in _tail(path, nbytes=window):
        m = _SWEEP_RE.match(line)
        if m:
            rows.append({
                "at": m.group("ts"),
                "ok": int(m.group("ok")),
                "failed": int(m.group("failed")),
                "skipped": int(m.group("skipped")),
                "total": int(m.group("total")),
                "seconds": float(m.group("secs")),
            })
    return rows


def sweeps(data_dir: str | Path, keep: int = 8) -> dict:
    """The last few full-sweep outcomes.

    ``failed`` is reproduced from the log verbatim, but it is very nearly
    always 0 for the reason in the module docstring — read ``failures`` for
    what actually went wrong, and read ``skipped`` here for whether the sweep
    is outgrowing its deadline.

    The window grows until it finds something. A sweep in flight writes tens of
    thousands of lines, so a fixed tail can sit entirely inside the current
    pass and conclude, wrongly, that no sweep has ever finished; and just after
    the hourly rotation the last finished sweep is in ``.1``, not in the live
    file. Both cases used to read as a dead pipeline.
    """
    log = Path(data_dir) / "_cron_all.log"
    rows: list[dict] = []
    for window in (400_000, 4_000_000, 24_000_000):
        rows = _scan_sweeps(log, window)
        if len(rows) >= keep:
            break
    if not rows:
        rows = _scan_sweeps(log.with_suffix(".log.1"), 24_000_000)
    recent = rows[-keep:]
    return {
        "recent": recent,
        "sweeps_seen": len(rows),
        "skipping": sum(1 for r in recent if r["skipped"]),
        "skipped_last": recent[-1]["skipped"] if recent else None,
    }


def failures(catalog: list[dict], data_dir: str | Path,
             window_hours: float = DEFAULT_WINDOW_HOURS,
             now: dt.datetime | None = None) -> dict:
    """Fetch/parse failures in the window, grouped by source, host and reason."""
    now = now or dt.datetime.now(UTC)
    cutoff = now - dt.timedelta(hours=window_hours)
    data_dir = Path(data_dir)

    host_of = {e["id"]: _host_of(e) for e in catalog}
    by_source: Counter[str] = Counter()
    by_host: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    host_reasons: dict[str, Counter[str]] = defaultdict(Counter)
    total = 0

    for sid, host in host_of.items():
        # Most sources have no error log, and most that do have not been
        # touched inside the window. One stat rules those out; without it this
        # reads and decodes ~3000 files to find the handful that matter.
        err_path = data_dir / sid / "_errors.log"
        try:
            if err_path.stat().st_mtime < cutoff.timestamp():
                continue
        except OSError:
            continue
        # Append-only per source, but a source failing every minute for a week
        # is not small, so tail rather than read.
        for line in _tail(err_path, nbytes=60_000):
            m = _ERR_RE.match(line)
            if not m:
                continue
            stamp = _parse_err_ts(m.group("ts"))
            if stamp is None or stamp < cutoff:
                continue
            reason = _reason(m.group("msg"))
            total += 1
            by_source[sid] += 1
            by_host[host] += 1
            by_reason[f"{m.group('kind')} {reason}"] += 1
            host_reasons[host][reason] += 1

    culprits = [
        {"host": h, "errors": n,
         "sources": sum(1 for s, c in by_source.items() if host_of[s] == h and c),
         "reasons": dict(host_reasons[h].most_common(3))}
        for h, n in by_host.most_common() if n >= HOST_CALLOUT
    ]
    return {
        "window_hours": window_hours,
        "total": total,
        "sources_affected": len(by_source),
        "hosts_affected": len(by_host),
        "by_reason": dict(by_reason.most_common(10)),
        "worst_sources": dict(by_source.most_common(10)),
        "culprit_hosts": culprits,
    }


def silence(catalog: list[dict], data_dir: str | Path,
            cron_path: str | Path | None = None,
            now: dt.datetime | None = None) -> dict:
    """Live sources the scraper is not writing.

    Judged on the parquet *filename* — the scrape date — not on the timestamps
    inside it. That is the whole point: a file written today proves the
    scraper reached the source today, whatever the observations say.
    """
    now = now or dt.datetime.now(UTC)
    today = now.date()
    data_dir = Path(data_dir)
    live = catalog
    periods = poll_periods(cron_path) if cron_path else {}

    never, stale, ok = [], [], 0
    for entry in live:
        sid = entry["id"]
        # Fall back to the entry's own frequency, which is what the full sweep
        # actually gates on. Sources absent from cron.yaml are polled exactly
        # like the rest; treating them as unscheduled invented a fault.
        period_s = periods.get(sid) or duration.period_seconds(
            str(entry.get("frequency") or "")) or 86_400
        # A daily poll writes today's file or yesterday's depending on the
        # hour, so one day of gap is normal for everything up to P1D; slower
        # feeds get their own period plus that same day of slack.
        period_days = max(1, round(period_s / 86_400))
        sdir = data_dir / sid

        # Fast path: the healthy case is a file dated within the allowed gap,
        # and probing those few names costs a stat each. Listing the directory
        # instead means reading every dirent of ~3000 dirs that hold hundreds
        # of parquet files apiece — the difference is a minute of wall clock.
        healthy = False
        for back in range(period_days + 1):
            if (sdir / f"{today - dt.timedelta(days=back)}.parquet").exists():
                healthy = True
                break
        if healthy:
            ok += 1
            continue

        newest: dt.date | None = None
        try:
            with os.scandir(sdir) as it:
                for item in it:
                    m = _PARQUET_RE.match(item.name)
                    if m:
                        day = dt.date.fromisoformat(m.group(1))
                        if newest is None or day > newest:
                            newest = day
        except OSError:
            newest = None
        if newest is None:
            never.append({"id": sid, "domain": entry.get("domain", "?"),
                          "host": _host_of(entry)})
            continue
        stale.append({"id": sid, "days": (today - newest).days,
                      "domain": entry.get("domain", "?"),
                      "host": _host_of(entry)})

    stale.sort(key=lambda r: -r["days"])
    never_hosts = Counter(r["host"] for r in never)
    return {
        "live": len(live),
        "writing": ok,
        "never_written": len(never),
        "not_writing": len(stale),
        "never_written_hosts": dict(never_hosts.most_common(5)),
        "never_written_ids": [r["id"] for r in never[:40]],
        "not_writing_worst": stale[:20],
    }


# A payload with no history is one point per poll, so its resolution IS its
# polling rate. Anything declared this fast has to be in the every-minute band
# or it silently degrades to the 15-minute sweep's cadence.
SNAPSHOT_MAX_PERIOD_S = 300


def undersampled(catalog: list[dict], cron_path: str | Path | None) -> dict:
    """Snapshot feeds polled slower than they are declared.

    ``schema.timestamp_field`` of ``now()`` — or absent, which makes the
    scraper stamp arrival time — means the response carries no history to
    backfill from. One poll is one observation, whatever the source's declared
    frequency claims, so a PT1M snapshot outside the fast band produces PT15M
    data and the catalog has no way to notice: the rows are current, the file
    is fresh, and only the spacing is wrong.
    """
    if not cron_path:
        return {"count": 0, "sources": []}
    fast = fast_band(cron_path)
    out = []
    for entry in catalog:
        if entry["id"] in fast:
            continue
        period = duration.period_seconds(str(entry.get("frequency") or ""))
        if period is None or period > SNAPSHOT_MAX_PERIOD_S:
            continue
        ts = (entry.get("schema") or {}).get("timestamp_field")
        if ts is None or str(ts).strip() in ("", "now()", "current-state"):
            out.append({"id": entry["id"], "frequency": entry.get("frequency"),
                        "host": _host_of(entry)})
    return {"count": len(out), "sources": out[:20]}


def build(catalog_path: str | Path, data_dir: str | Path,
          cron_path: str | Path | None = None,
          ops_log_dir: str | Path | None = None,
          window_hours: float = DEFAULT_WINDOW_HOURS,
          now: dt.datetime | None = None) -> dict:
    """Run every check and return a JSON-ready report."""
    now = now or dt.datetime.now(UTC)
    catalog = load_catalog(catalog_path)   # parsed once; both checks walk it
    report = {
        "at": now.isoformat(timespec="seconds"),
        "heartbeats": heartbeats(data_dir, ops_log_dir, now=now),
        "sweeps": sweeps(data_dir),
        "failures": failures(catalog, data_dir, window_hours, now=now),
        "silence": silence(catalog, data_dir, cron_path, now=now),
        "undersampled": undersampled(catalog, cron_path),
    }
    report["findings"] = _findings(report)
    report["status"] = ("critical" if any(f["severity"] == "critical"
                                          for f in report["findings"])
                        else "warn" if report["findings"] else "ok")
    return report


def _findings(report: dict) -> list[dict]:
    """Turn the raw checks into ranked, actionable statements.

    Severity is about what a human should do now: ``critical`` means the
    pipeline is not running and data is being lost that no later sweep will
    recover; ``warn`` means it is running but degraded, which is a backlog to
    work through rather than an alarm.
    """
    out: list[dict] = []

    for hb in report["heartbeats"]:
        if hb["status"] == "ok":
            continue
        if hb["status"] == "missing":
            out.append({"severity": "critical", "check": "heartbeat",
                        "detail": f"{hb['job']}: no log at {hb['log']} — "
                                  "job has never run here"})
        else:
            out.append({"severity": "critical", "check": "heartbeat",
                        "detail": f"{hb['job']}: last ran "
                                  f"{hb['age_s'] / 60:.0f} min ago, expected "
                                  f"every {hb['period_s'] / 60:.0f} min"})

    sw = report["sweeps"]
    if not sw["recent"]:
        # Only an outage if the job is also not running. With a live heartbeat
        # this means the very first sweep is still in flight — true right after
        # a fresh deploy, and not something to page anyone about.
        alive = any(h["job"] == "scrape_all" and h["status"] == "ok"
                    for h in report["heartbeats"])
        out.append({"severity": "warn" if alive else "critical",
                    "check": "sweep",
                    "detail": "no completed sweep found in the log"
                              + (" (one may be in flight)" if alive else "")})
    else:
        last = sw["recent"][-1]
        if last["failed"]:
            out.append({"severity": "warn", "check": "sweep",
                        "detail": f"last sweep reported {last['failed']} "
                                  "uncaught source failures"})
        # One deadline overrun is ordinary; every recent sweep overrunning
        # means the workload no longer fits the window it is given.
        if sw["skipping"] == len(sw["recent"]) and len(sw["recent"]) > 2:
            out.append({"severity": "warn", "check": "sweep",
                        "detail": f"deadline hit on all {len(sw['recent'])} "
                                  f"recent sweeps ({last['skipped']} sources "
                                  "unstarted last pass)"})

    # Findings are meant to be read top to bottom and acted on. A catalog this
    # size always has a long tail of hosts throwing 20-odd errors a day, and
    # listing every one of them turns the section into a second copy of the
    # table above. Name the worst few; count the rest.
    culprits = report["failures"]["culprit_hosts"]
    for host in culprits[:FINDING_HOSTS]:
        top = ", ".join(f"{k} x{v}" for k, v in host["reasons"].items())
        plural = "" if host["sources"] == 1 else "s"
        out.append({"severity": "warn", "check": "failures",
                    "detail": f"{host['host']}: {host['errors']} errors across "
                              f"{host['sources']} source{plural} ({top})"})
    if len(culprits) > FINDING_HOSTS:
        rest = culprits[FINDING_HOSTS:]
        out.append({"severity": "warn", "check": "failures",
                    "detail": f"{len(rest)} further hosts over {HOST_CALLOUT} "
                              f"errors ({sum(h['errors'] for h in rest)} total)"})

    sil = report["silence"]
    if sil["never_written"]:
        hosts = ", ".join(f"{h} x{n}"
                          for h, n in sil["never_written_hosts"].items())
        out.append({"severity": "warn", "check": "silence",
                    "detail": f"{sil['never_written']} live sources have never "
                              f"written a file ({hosts})"})
    if sil["not_writing"]:
        out.append({"severity": "warn", "check": "silence",
                    "detail": f"{sil['not_writing']} live sources have not been "
                              "written past their polling period"})

    under = report["undersampled"]
    if under["count"]:
        names = ", ".join(s["id"] for s in under["sources"][:3])
        out.append({"severity": "warn", "check": "undersampled",
                    "detail": f"{under['count']} snapshot sources are outside "
                              f"the every-minute band and so record at the "
                              f"15-min sweep's rate ({names}"
                              + (", ..." if under["count"] > 3 else "") + ")"})

    order = {"critical": 0, "warn": 1}
    out.sort(key=lambda f: order[f["severity"]])
    return out


def summarize(report: dict) -> str:
    """One grep-friendly line for a rolling log."""
    sil, fail = report["silence"], report["failures"]
    late = sum(1 for h in report["heartbeats"] if h["status"] != "ok")
    last = report["sweeps"]["recent"][-1] if report["sweeps"]["recent"] else None
    return (
        f"monitor {report['at']}: {report['status'].upper()} — "
        f"{len(JOBS) - late}/{len(JOBS)} jobs live, "
        f"last sweep {last['ok'] if last else '-'} ok/"
        f"{last['skipped'] if last else '-'} skipped, "
        f"{fail['total']} errors on {fail['hosts_affected']} hosts in "
        f"{fail['window_hours']:.0f}h, "
        f"{sil['writing']}/{sil['live']} sources writing"
    )


def render(report: dict) -> str:
    """Human-readable report for a terminal."""
    lines = [summarize(report), ""]

    lines.append("jobs")
    for hb in report["heartbeats"]:
        age = "-" if hb["age_s"] is None else f"{hb['age_s'] / 60:.1f}m ago"
        mark = "ok " if hb["status"] == "ok" else "!! "
        lines.append(f"  {mark}{hb['job']:<14} {age:>12}   "
                     f"every {hb['period_s'] / 60:.0f}m")

    lines += ["", "sweeps (most recent last)"]
    for row in report["sweeps"]["recent"]:
        lines.append(f"  {row['at']}  {row['ok']:>5} ok  "
                     f"{row['failed']:>3} failed  {row['skipped']:>4} skipped  "
                     f"{row['seconds']:.0f}s")

    fail = report["failures"]
    lines += ["", f"failures (last {fail['window_hours']:.0f}h): {fail['total']} "
                  f"across {fail['sources_affected']} sources"]
    for host in fail["culprit_hosts"]:
        top = ", ".join(f"{k} x{v}" for k, v in host["reasons"].items())
        lines.append(f"  {host['host']:<40} {host['errors']:>6}  {top}")
    for reason, n in fail["by_reason"].items():
        lines.append(f"    {reason:<38} {n:>6}")

    sil = report["silence"]
    lines += ["", f"writing {sil['writing']}/{sil['live']} live sources  "
                  f"(never written {sil['never_written']}, "
                  f"behind {sil['not_writing']}, "
                  f"undersampled {report['undersampled']['count']})"]
    for row in sil["not_writing_worst"][:10]:
        lines.append(f"  {row['days']:>3}d  {row['id']:<44} {row['host']}")

    if report["findings"]:
        lines += ["", "findings"]
        for f in report["findings"]:
            lines.append(f"  [{f['severity']}] {f['check']}: {f['detail']}")
    else:
        lines += ["", "no findings — pipeline healthy"]
    return "\n".join(lines)
