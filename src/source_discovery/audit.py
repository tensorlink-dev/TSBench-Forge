"""audit.py — catalog freshness audit: is each live source still moving?

Wire-time verification (wire.py) proves a source worked *once*. Upstreams then
die quietly — an expired reporting mandate, a stalled statistical migration, a
retired dataset — while the scraper keeps "succeeding" by re-reading the same
frozen rows. Every such case in this catalog's history was found by comparing
the newest *observation* timestamp on disk against the source's declared
cadence. This module makes that check a first-class, schedulable step.

For each non-disabled source: read the newest parquet's ``timestamp`` column,
parse the max observation time (the stored strings span a dozen formats), and
compare its age against a cadence-scaled threshold. Slow feeds get generous
slack — a monthly index that lags 3 months is normal; one that lags 8 is dead.

Statuses: ``ok`` | ``stale`` | ``nodata`` (no parquet yet) | ``unparsed``
(timestamps in a format this audit cannot read — an audit gap, not necessarily
a source failure).

``apply_disables`` is deliberately opt-in and conservative: it only touches
sources stale past ``hard_factor`` × threshold, edits by *inserting* two lines
after the entry's ``- id:`` line (never rewrites the file wholesale), and
validates the catalog reparses with the same entry count afterwards.
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path
from typing import Optional

import yaml

UTC = dt.timezone.utc

# Explicit strptime formats tried in order (naive results are assumed UTC).
_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M%z",
    "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
    "%Y-%m-%d", "%Y-%m",
    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d",
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
    "%m.%Y", "%d.%m.%Y",
    "%d%b%Y",  # CPC ENSO weekly: "15JUL2026" (strptime %b is case-insensitive)
)

# Digit-only strings are ambiguous under greedy strptime ("20260726" parses as
# %Y%m%d%H = 2026-07-02T06); dispatch them by LENGTH instead. 10 digits is
# epoch-seconds when inside the scraper's epoch range, else YYYYMMDDHH.
_DIGIT_FMT_BY_LEN = {14: "%Y%m%d%H%M%S", 10: "%Y%m%d%H", 8: "%Y%m%d",
                     6: "%Y%m", 4: "%Y"}

_QUARTER_RE = re.compile(r"^(\d{4})-?[QK](\d)$")
_SLASH_DMY_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")


def parse_ts(raw: object) -> Optional[dt.datetime]:
    """Parse one stored timestamp string to an aware UTC datetime, or None."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # epoch seconds / millis (stored numerically or as digit strings)
    if re.fullmatch(r"\d{10}\.\d+", s):
        return dt.datetime.fromtimestamp(float(s), UTC)
    if re.fullmatch(r"\d{13}", s):
        return dt.datetime.fromtimestamp(int(s) / 1000, UTC)
    if s.isdigit():
        if len(s) == 10 and 10**8 <= int(s) < 2 * 10**9:
            return dt.datetime.fromtimestamp(int(s), UTC)
        fmt = _DIGIT_FMT_BY_LEN.get(len(s))
        if not fmt:
            return None
        try:
            return dt.datetime.strptime(s, fmt).replace(tzinfo=UTC)
        except ValueError:
            return None
    m = _QUARTER_RE.match(s)
    if m:
        return dt.datetime(int(m.group(1)), 3 * int(m.group(2)) - 2, 1, tzinfo=UTC)
    m = _SLASH_DMY_RE.match(s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # DD/MM vs MM/DD is ambiguous; a component >12 settles it. When both
        # readings are valid, take the LATER one that isn't in the future —
        # for staleness detection a false "fresh" (bounded by the day/month
        # swap) beats a false alarm (bcb's 01/06/2026 read month-first looked
        # 6 months stale).
        horizon = dt.datetime.now(UTC) + dt.timedelta(days=31)
        cands = []
        for month, day in {(a, b), (b, a)}:
            try:
                got = dt.datetime(y, month, day, tzinfo=UTC)
            except ValueError:
                continue
            cands.append(got)
        past = [c for c in cands if c <= horizon]
        return max(past) if past else (min(cands) if cands else None)
    zless = s[:-1] if s.endswith("Z") else s
    for fmt in _FORMATS:
        try:
            got = dt.datetime.strptime(zless, fmt)
        except ValueError:
            continue
        return got if got.tzinfo else got.replace(tzinfo=UTC)
    return None


def _period_seconds(freq: str) -> Optional[int]:
    sources_dir = str(Path(__file__).resolve().parent.parent / "sources")
    if sources_dir not in sys.path:
        sys.path.insert(0, sources_dir)
    import scraper  # noqa: PLC0415
    return scraper._period_seconds(freq)


_FREQ_ALIASES = {"P1Q": "P3M"}  # ISO-ish quarter shorthand the scraper can't parse


def staleness_threshold(freq: str) -> dt.timedelta:
    """How old the newest observation may be before the source counts stale."""
    period = _period_seconds(_FREQ_ALIASES.get(freq, freq or ""))
    if period is None:                      # irregular / unparsable cadence
        return dt.timedelta(days=45)
    if period >= 7_000_000:                 # quarterly / yearly
        return dt.timedelta(days=550)
    if period >= 2_000_000:                 # monthly
        return dt.timedelta(days=100)
    if period >= 500_000:                   # weekly-ish
        return dt.timedelta(days=35)
    return max(dt.timedelta(seconds=period * 8), dt.timedelta(hours=72))


def newest_observation(data_dir: str | Path, sid: str) -> tuple[Optional[dt.datetime], int]:
    """(max parsed timestamp in the newest parquet, rows scanned).

    Minute-cadence day-files run to 10^5+ rows; per-value strptime over those
    made a full-catalog audit take ~20 min. Fast path: vectorised ISO8601 parse
    over the *unique* values (pandas), falling back to ``parse_ts`` only for
    whatever pandas could not read (the NYISO/AEMO/label formats — always few)."""
    import pandas as pd            # noqa: PLC0415 — heavy imports, keep lazy
    import pyarrow.compute as pc   # noqa: PLC0415
    import pyarrow.parquet as pq   # noqa: PLC0415

    files = sorted(Path(data_dir, sid).glob("*.parquet"))
    if not files:
        return None, 0
    try:
        col = pq.read_table(files[-1], columns=["timestamp"]).column("timestamp")
    except Exception:  # noqa: BLE001 — corrupt file == no readable data
        return None, 0
    nrows = len(col)
    uniq = pd.Series(pc.unique(col.combine_chunks()).to_pylist(), dtype=object)
    parsed = pd.to_datetime(uniq, format="ISO8601", utc=True, errors="coerce")
    newest = None
    if parsed.notna().any():
        newest = parsed.max().to_pydatetime()
    for v in uniq[parsed.isna()]:
        got = parse_ts(v)
        if got and (newest is None or got > newest):
            newest = got
    return newest, nrows


def audit_catalog(catalog_path: str | Path, data_dir: str | Path,
                  now: Optional[dt.datetime] = None) -> list[dict]:
    """Audit every non-disabled source; returns one finding dict per source."""
    now = now or dt.datetime.now(UTC)
    sources = yaml.safe_load(Path(catalog_path).read_text())
    findings: list[dict] = []
    for src in sources:
        sid, freq = src["id"], src.get("frequency", "")
        if src.get("disabled"):
            findings.append({"id": sid, "frequency": freq, "status": "disabled"})
            continue
        newest, nrows = newest_observation(data_dir, sid)
        if nrows == 0:
            findings.append({"id": sid, "frequency": freq, "status": "nodata"})
            continue
        if newest is None:
            findings.append({"id": sid, "frequency": freq, "status": "unparsed",
                             "rows_scanned": nrows})
            continue
        age = now - newest
        # Per-source override for known publication lags (openFDA's quarterly
        # refresh, WHO's multi-year estimate cycles): set `audit_slack_days` on
        # the catalog entry, with a note saying why, to silence the audit.
        slack = src.get("audit_slack_days")
        limit = (dt.timedelta(days=float(slack)) if slack
                 else staleness_threshold(freq))
        findings.append({
            "id": sid, "frequency": freq,
            "status": "stale" if age > limit else "ok",
            "newest": newest.isoformat(),
            "age_days": round(age.total_seconds() / 86400, 1),
            "limit_days": round(limit.total_seconds() / 86400, 1),
        })
    return findings


def summarize(findings: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for f in findings:
        counts[f["status"]] = counts.get(f["status"], 0) + 1
    return {
        "counts": counts,
        "stale": [f for f in findings if f["status"] == "stale"],
        "nodata": [f["id"] for f in findings if f["status"] == "nodata"],
        "unparsed": [f["id"] for f in findings if f["status"] == "unparsed"],
    }


_ID_LINE_RE = "- id: {sid}"


def apply_disables(catalog_path: str | Path, findings: list[dict],
                   hard_factor: float = 3.0,
                   today: Optional[str] = None) -> list[str]:
    """Disable sources stale past ``hard_factor`` × their threshold.

    Inserts ``disabled: true`` + ``disabled_reason`` immediately after the
    entry's ``- id:`` line — a two-line insertion, never a rewrite — then
    validates the catalog reparses with an unchanged entry count.
    """
    catalog_path = Path(catalog_path)
    today = today or dt.date.today().isoformat()
    victims = [f for f in findings
               if f["status"] == "stale"
               and f.get("age_days", 0) > hard_factor * f.get("limit_days", 1e9)]
    if not victims:
        return []

    text = catalog_path.read_text()
    before = yaml.safe_load(text)
    lines = text.splitlines(keepends=True)
    disabled: list[str] = []
    for f in victims:
        sid = f["id"]
        needle = _ID_LINE_RE.format(sid=sid)
        for i, line in enumerate(lines):
            if line.strip() == needle:
                reason = (f"{today}: auto-disabled by freshness audit — newest "
                          f"observation {f['newest']} is {f['age_days']}d old vs a "
                          f"{f['limit_days']}d limit for {f['frequency']}. Re-enable "
                          f"if the upstream resumes.")
                indent = " " * (len(line) - len(line.lstrip()) + 2)
                lines[i] = (line + f"{indent}disabled: true\n"
                            + f"{indent}disabled_reason: '{reason}'\n")
                disabled.append(sid)
                break

    new_text = "".join(lines)
    after = yaml.safe_load(new_text)
    if len(after) != len(before):
        raise RuntimeError("catalog entry count changed during disable edit — aborted")
    by_id = {s["id"]: s for s in after}
    for sid in disabled:
        if not by_id.get(sid, {}).get("disabled"):
            raise RuntimeError(f"disable edit did not take for {sid} — aborted")

    import os
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=catalog_path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        fh.write(new_text)
    os.replace(tmp, catalog_path)
    return disabled
