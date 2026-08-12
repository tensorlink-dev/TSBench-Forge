"""Daily composition of the EVAL POOL — not of the catalog.

``coverage.py`` answers "what is in ``sources.yaml``". This answers "what would
the eval actually draw from today", which is a materially different question:

* a live source contributes nothing until it has ``min_series_length``
  observations on disk, and most of the catalog was wired recently
* panel expansion turns one GTFS feed into hundreds of series, so a domain's
  share of *series* looks nothing like its share of *sources*
* ``ScrapedLiveSource._sample_indices_equal_weight`` then re-levels the draw
  across domain × dgp_class × cadence, so the served mix looks like neither

Concretely, on the day this was written: web_cloudops was 39% of the catalog
but one seventh of the sampled pool, and transport was 11% of the catalog but
72% of eligible series. None of those three numbers is guessable from the
other two, which is why this runs daily and keeps a dated history.

The eligibility frontier moves on its own as the cron accumulates history —
sources cross ``min_series_length`` without anyone touching the catalog — so
the interesting signal is the day-over-day delta, not any single snapshot.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict
from pathlib import Path

from . import coverage

# Fixed so the sampled mix is comparable across days: a moving seed would make
# ordinary sampling noise look like drift.
SAMPLE_SEED = 0
SAMPLE_N = 700


def _sources_dir_on_path(catalog_path: str | Path) -> None:
    """Make ``src/`` importable — mirrors wire.py's approach."""
    import sys

    src_dir = str(Path(catalog_path).resolve().parents[1])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def build(catalog_path: str | Path, data_dir: str | Path,
          sample_n: int = SAMPLE_N) -> dict:
    """Index the pool and describe its composition.

    Returns a JSON-ready dict. Reads every source's parquet metadata, so it is
    minutes-scale; it is a daily job, not something to call in a loop.
    """
    _sources_dir_on_path(catalog_path)
    import numpy as np
    import yaml

    from scraped_source import ScrapedLiveSource

    catalog = yaml.safe_load(Path(catalog_path).read_text()) or []
    live = [s for s in catalog if not s.get("disabled")]
    live_by_id = {s["id"]: s for s in live}

    src = ScrapedLiveSource(catalog_path, data_dir)
    series = src._catalog()

    eligible_ids = {s["source_id"] for s in series}
    # A live source with no eligible series is short of min_series_length. That
    # is the depth backlog, and it drains by itself as the cron accumulates.
    ineligible = sorted(set(live_by_id) - eligible_ids)

    by_domain_series: Counter[str] = Counter(s["domain"] for s in series)
    by_domain_sources: Counter[str] = Counter()
    for sid in eligible_ids:
        entry = live_by_id.get(sid)
        if entry:
            by_domain_sources[entry.get("domain", "?")] += 1

    # domain x cadence-band, over ELIGIBLE SOURCES (not series): series counts
    # are dominated by whichever domain happens to panel-expand widest, which
    # tells you about panel shape rather than about coverage.
    grid: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sid in eligible_ids:
        entry = live_by_id.get(sid)
        if not entry:
            continue
        band = coverage.band_for(str(entry.get("frequency") or ""))
        grid[entry.get("domain", "?")][band] += 1

    ineligible_by_domain = Counter(
        live_by_id[sid].get("domain", "?") for sid in ineligible)

    sampled: Counter[str] = Counter()
    sampled_bands: Counter[str] = Counter()
    sample_error = None
    if series:
        try:
            picks = src._sample_indices_equal_weight(
                sample_n, np.random.default_rng(SAMPLE_SEED))
            sampled = Counter(p["domain"] for p in picks)
            sampled_bands = Counter(p["cadence"] for p in picks)
        except Exception as exc:  # noqa: BLE001 — a failed draw must not lose the counts
            sample_error = f"{type(exc).__name__}: {exc}"

    return {
        "date": dt.datetime.now(dt.timezone.utc).date().isoformat(),
        "min_series_length": src.min_series_length,
        "sample_seed": SAMPLE_SEED,
        "totals": {
            "catalog_entries": len(catalog),
            "live_sources": len(live),
            "eligible_sources": len(eligible_ids),
            "ineligible_on_depth": len(ineligible),
            "eligible_series": len(series),
            "cells": len({(s["domain"], s["dgp_class"], s["cadence"])
                          for s in series}),
        },
        "eligible_sources_by_domain": dict(by_domain_sources.most_common()),
        "eligible_series_by_domain": dict(by_domain_series.most_common()),
        "ineligible_on_depth_by_domain": dict(ineligible_by_domain.most_common()),
        "eligible_sources_by_domain_cadence": {
            d: dict(sorted(bands.items())) for d, bands in sorted(grid.items())
        },
        "sampled_by_domain": dict(sampled.most_common()),
        "sampled_by_cadence": dict(sampled_bands.most_common()),
        "sample_n": sample_n,
        "sample_error": sample_error,
    }


def summarize(report: dict) -> str:
    """One grep-friendly line for the rolling log."""
    t = report["totals"]
    dom = report["eligible_sources_by_domain"]
    top = max(dom.items(), key=lambda kv: kv[1]) if dom else ("-", 0)
    share = (100.0 * top[1] / t["eligible_sources"]) if t["eligible_sources"] else 0.0
    return (
        f"pool {report['date']}: {t['eligible_sources']}/{t['live_sources']} "
        f"sources eligible, {t['eligible_series']} series, {t['cells']} cells, "
        f"{t['ineligible_on_depth']} short of {report['min_series_length']} obs; "
        f"top domain {top[0]} {share:.0f}%"
    )


def render_table(report: dict) -> str:
    """Human-readable domain × cadence grid over eligible sources."""
    grid = report["eligible_sources_by_domain_cadence"]
    bands = sorted({b for row in grid.values() for b in row})
    if not bands:
        return "(no eligible sources)"
    width = max([len(d) for d in grid] + [6])
    head = "domain".ljust(width) + "".join(b.rjust(11) for b in bands) + "total".rjust(8)
    lines = [head]
    for domain in sorted(grid):
        row = grid[domain]
        lines.append(
            domain.ljust(width)
            + "".join(str(row.get(b, 0)).rjust(11) for b in bands)
            + str(sum(row.values())).rjust(8)
        )
    totals = [sum(grid[d].get(b, 0) for d in grid) for b in bands]
    lines.append("TOTAL".ljust(width) + "".join(str(n).rjust(11) for n in totals)
                 + str(sum(totals)).rjust(8))
    return "\n".join(lines)
