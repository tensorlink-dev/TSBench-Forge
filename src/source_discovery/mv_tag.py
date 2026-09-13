"""A2 tagging pass: write curated ``mv_channels`` into ``sources.yaml``.

Consumes the JSON report emitted by :mod:`mv_audit` (the A1 cross-
predictiveness gate) and applies the curation layer the ridge test cannot
provide: a statistically "coupled" channel can still be an ID that dodged the
junk regex (a parking feed's ``sourceelementkey`` produced the run's top
"gain"), a static registry attribute (fire-hydrant MAKE/MODEL), a calendar
column (``weekday``, ``dayofyear``), or a sibling *derived* from another
channel (``last_score``, a cumulative ``aggregate``, confidence limits, a
published ``vorhersage``). Those earn real held-out gains — the information
is genuine — but they are not coupled *signals*, and tagging them would hand
cascade multivariate windows whose extra channels are keys, constants-in-
disguise, or leaks of the target itself.

Rules are deterministic and reviewable below: source-level blocks for feeds
that are event/static registries rather than aligned time series, and
channel-level blocks by name pattern. Deliberately kept: hierarchical sets
(total alongside its components — male/female/total, demand alongside
generation-by-fuel) and published forecast-vs-actual pairs, both genuine
multivariate structures.

The write is wire.py-style: textual in-place insert (``mv_channels: [...]``
directly under each entry's ``id:`` line), then reparse-validate (entry count
unchanged, ids unique, every tag readable), atomic replace, roll back on any
mismatch. Idempotent — entries that already carry ``mv_channels`` are left
alone.

    .venv/bin/python -m source_discovery.mv_tag --report mv_audit_report.json
    .venv/bin/python -m source_discovery.mv_tag --report mv_audit_report.json --apply
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

# Feeds whose rows are events or registry snapshots, not aligned time series —
# their channels are attributes of the row (permit value + square feet, quake
# magnitude + depth), so a cross-sectional relation masquerades as coupling.
BLOCK_SOURCES = re.compile(
    r"^(agis_|energystar_|hicscdata_|data_akf_data$|data_food_service_inspection"
    r"|ct_energy_storage|opendatasoft_encampment|vancouver_issued_building_permits$"
    r"|irishrail_current_train_positions$|digitraffic_finland_ais_vessel_speeds$"
    r"|datahub_real_time_road_conditions$|data_mta_metro_north_delay"
    r"|ch_gasverbrauch|swiss_weekly_number_of_deaths|hexpm_package_downloads$"
    r"|noaa_gml_co2_weekly_mauna_loa$|hadcrut5_|ncedc_norcal_earthquakes$"
    r"|liege_shop_drive|opendata_referentiel_parkings_saemes$"
    r"|cos_data_paid_parking|data_freshwater_swim_beach|data_crash_reporting"
    r"|kraken_futures_btc_funding$|elia_ods147)"
)

# Channel names that are not signals: identifiers the audit's junk regex
# missed, calendar features, coordinates/angles, static dimensions, and
# columns derived from a sibling (rolling/cumulative/smoothed/CI/forecast-of-
# self variants — a derived sibling "predicts" its parent by construction).
BLOCK_CHANNELS = re.compile(
    r"key$|number$|assetid|geocodingid|projectid|permitnum|sourceelementkey"
    r"|computed_region|locid|^(cx|cy|box)$"
    r"|\b(weekday|dayofyear)\b|^(year|month|week|day|hour|anno|settimana)$"
    r"|latitud|longitud|direction|^cog$"
    r"|width|depth_inches|height|hauteur|niveaux|nombre_de_places|bedrooms"
    r"|yearbuilt|address|^(make|model|lof)$|inv_by|diameter"
    r"|^last_|_normalisiert$|vorhersage|aggregate$|increase since|^trend$"
    r"|confidence limit|days_since|_ratio$|^aqi|_4hr$|^recent$|battery",
    re.IGNORECASE,
)

MIN_CHANNELS = 2


def curate(report: dict) -> tuple[list[dict], list[dict]]:
    """Split admitted audit results into (tag, curated_out) lists."""
    tags, dropped = [], []
    for r in report["results"]:
        if r["status"] != "admitted":
            continue
        if BLOCK_SOURCES.search(r["id"]):
            dropped.append({**r, "curation": "blocked_source"})
            continue
        keep = [c for c in r["mv_channels"] if not BLOCK_CHANNELS.search(c)]
        if len(keep) < MIN_CHANNELS:
            dropped.append({**r, "curation": "channels_blocked", "kept": keep})
            continue
        tags.append({
            "id": r["id"],
            "domain": r["domain"],
            "frequency": r["frequency"],
            "best_gain": r["best_gain"],
            "mv_channels": keep,
            "trimmed": len(keep) < len(r["mv_channels"]),
        })
    return tags, dropped


def apply_tags(catalog_path: Path, tags: list[dict]) -> dict:
    text = catalog_path.read_text()
    lines = text.splitlines(keepends=True)
    id_line = {  # "- id: <sid>" -> line index
        m.group(1): i
        for i, ln in enumerate(lines)
        if (m := re.match(r"- id: (\S+)\s*$", ln))
    }
    already = set()
    import yaml

    for s in yaml.safe_load(text):
        if "mv_channels" in s:
            already.add(s["id"])

    inserts = {}
    skipped = []
    for t in tags:
        if t["id"] in already:
            skipped.append(t["id"])
            continue
        if t["id"] not in id_line:
            skipped.append(t["id"])
            continue
        # JSON flow list is valid YAML; non-ASCII channel names (the JP grid
        # feeds) stay readable.
        inserts[id_line[t["id"]]] = (
            "  mv_channels: " + json.dumps(t["mv_channels"], ensure_ascii=False) + "\n"
        )

    out = []
    for i, ln in enumerate(lines):
        out.append(ln)
        if i in inserts:
            out.append(inserts[i])
    new_text = "".join(out)

    # Validate before touching the file: same entry count, unique ids, every
    # intended tag parseable as a >=2 list.
    reparsed = yaml.safe_load(new_text)
    old = yaml.safe_load(text)
    ids = [s["id"] for s in reparsed]
    tagged = {s["id"]: s.get("mv_channels") for s in reparsed if "mv_channels" in s}
    want = {t["id"] for t in tags if t["id"] not in set(skipped)}
    bad = [i for i in want if not (isinstance(tagged.get(i), list) and len(tagged[i]) >= 2)]
    if len(reparsed) != len(old) or len(ids) != len(set(ids)) or bad:
        raise RuntimeError(
            f"post-edit validation failed (entries {len(old)}->{len(reparsed)}, "
            f"dup_ids={len(ids) - len(set(ids))}, unreadable_tags={bad[:5]}) — not writing"
        )

    fd, tmp = tempfile.mkstemp(dir=catalog_path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(new_text)
    os.replace(tmp, catalog_path)
    return {"tagged": len(inserts), "skipped": skipped}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--catalog", default="src/sources/sources.yaml", type=Path)
    ap.add_argument("--apply", action="store_true", help="write tags into the catalog")
    args = ap.parse_args(argv)

    report = json.loads(args.report.read_text())
    tags, dropped = curate(report)

    from collections import Counter

    print(f"admitted {sum(1 for r in report['results'] if r['status'] == 'admitted')} "
          f"-> tag {len(tags)} (curated out {len(dropped)}: "
          f"{dict(Counter(d['curation'] for d in dropped))})")
    print("tag by domain:", dict(Counter(t["domain"] for t in tags)))
    for t in sorted(tags, key=lambda t: (t["domain"], -t["best_gain"])):
        mark = " (trimmed)" if t["trimmed"] else ""
        print(f"  {t['domain'][:4]:4s} {t['best_gain']:+.2f} {t['id'][:52]:52s} "
              f"{t['mv_channels']}{mark}")
    for d in dropped:
        print(f"  OUT [{d['curation']}] {d['id']}", file=sys.stderr)

    if args.apply:
        res = apply_tags(args.catalog, tags)
        print(f"applied: {res['tagged']} tagged, skipped {res['skipped'] or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
