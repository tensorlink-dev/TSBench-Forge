"""Tag every catalog source with its statistical archetype (``stat_archetype``).

The archetype is the third balance axis next to domain and cadence (see
:mod:`stat_clusters`). Cascade reads catalog entry keys directly, the same way
it reads ``mv_channels``, so a top-level ``stat_archetype: A07`` line is the
whole contract: its balanced selection can rotate over
(domain, freq, archetype), or weight the draw by ``draw_weight`` from the
frozen model file.

The model (centroids + scaling + reference shares + draw weights) is frozen in
``data/archetypes.json`` so ids are stable across runs. Re-fitting changes
cluster ids; do it deliberately with ``--refit`` and bump ``version``.

A source's archetype is the majority over its profiled series (up to 3 panel
rows). Sources without enough fresh data are left untagged, not guessed.

Unlike mv_tag, re-running REPLACES an existing tag: a series' statistical
character can drift (a gauge freezes, a market regime changes), and a stale
archetype would mis-weight it.

    .venv/bin/python -m source_discovery.stat_tag            # dry run, prints counts
    .venv/bin/python -m source_discovery.stat_tag --apply
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

from source_discovery.stat_clusters import Archetypes
from source_discovery.tsfeatures import FEATURES, features

MODEL = Path(__file__).parent / "data" / "archetypes.json"
SEASON = {"PT1S": 3600, "PT10S": 360, "PT15S": 240, "PT30S": 120, "PT1M": 60,
          "PT2M30S": 576, "PT5M": 288, "PT6M": 240, "PT10M": 144, "PT15M": 96,
          "PT30M": 48, "PT1H": 24, "P1D": 7}
SLOW = re.compile(r"^P\d+(W|M|Y)$|^P([7-9]|[1-9]\d+)D$")
DATED = re.compile(r"\d{4}-\d{2}-\d{2}$")


def load_model(path: Path = MODEL) -> tuple[Archetypes, dict]:
    d = json.loads(path.read_text())
    return Archetypes.from_dict(d), d


def source_features(entry: dict, data_dir: Path, max_files: int = 12,
                    max_age_days: int = 4) -> list[dict[str, float]]:
    """Feature dicts for a source's freshest data (<=3 panel rows)."""
    import pandas as pd

    if entry.get("disabled") or SLOW.match(str(entry.get("frequency", "")).upper()):
        return []
    d = data_dir / entry["id"]
    fs = sorted(p for p in d.glob("*.parquet") if DATED.match(p.stem)) if d.is_dir() else []
    if not fs or (dt.date.today() - dt.date.fromisoformat(fs[-1].stem)).days > max_age_days:
        return []
    try:
        df = pd.concat([pd.read_parquet(p) for p in fs[-max_files:]], ignore_index=True)
    except Exception:
        return []
    pc = [c for c in df.columns if c.startswith("_panel_")]
    vc = [c for c in df.columns if c != "timestamp" and not c.startswith("_panel_")]
    if not vc:
        return []
    season = SEASON.get(str(entry.get("frequency")), 0)
    out = []
    for _, sub in (list(df.groupby(pc))[:3] if pc else [(None, df)]):
        v = pd.to_numeric(sub[vc[0]], errors="coerce").to_numpy()[-4096:]
        if np.isfinite(v).sum() >= 256:
            out.append(features(v, season))
    return out


def assign_sources(catalog: list[dict], data_dir: Path, arch: Archetypes,
                   names: list[str]) -> dict[str, str]:
    tags = {}
    for e in catalog:
        fl = source_features(e, data_dir)
        if not fl:
            continue
        lab = arch.assign(np.array([[f[k] for k in FEATURES] for f in fl]))
        tags[e["id"]] = names[collections.Counter(lab.tolist()).most_common(1)[0][0]]
    return tags


def apply_tags(catalog_path: Path, tags: dict[str, str]) -> dict:
    import yaml

    text = catalog_path.read_text()
    old = yaml.safe_load(text)
    lines = text.splitlines(keepends=True)
    # drop existing tags first (replace semantics), then insert under id lines
    kept = [ln for ln in lines if not re.match(r"  stat_archetype: \S+\s*$", ln)]
    out = []
    n = 0
    for ln in kept:
        out.append(ln)
        m = re.match(r"- id: (\S+)\s*$", ln)
        if m and m.group(1) in tags:
            out.append(f"  stat_archetype: {tags[m.group(1)]}\n")
            n += 1
    new_text = "".join(out)
    reparsed = yaml.safe_load(new_text)
    ids = [s["id"] for s in reparsed]
    got = {s["id"]: s.get("stat_archetype") for s in reparsed}
    bad = [i for i, t in tags.items() if got.get(i) != t]
    if len(reparsed) != len(old) or len(ids) != len(set(ids)) or bad:
        raise RuntimeError(
            f"post-edit validation failed (entries {len(old)}->{len(reparsed)}, "
            f"dup_ids={len(ids) - len(set(ids))}, mismatched={bad[:5]}) — not writing")
    fd, tmp = tempfile.mkstemp(dir=catalog_path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(new_text)
    os.replace(tmp, catalog_path)
    return {"tagged": n}


def main(argv: list[str] | None = None) -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalog", default="src/sources/sources.yaml", type=Path)
    ap.add_argument("--data", default="src/sources/data", type=Path)
    ap.add_argument("--model", default=MODEL, type=Path)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    arch, meta = load_model(a.model)
    catalog = yaml.safe_load(a.catalog.read_text())
    tags = assign_sources(catalog, a.data, arch, meta["names"])
    c = collections.Counter(tags.values())
    print(f"{len(tags)} of {len(catalog)} sources assigned an archetype")
    for name in meta["names"]:
        i = meta["names"].index(name)
        print(f"  {name}  {c.get(name, 0):5d}  w={meta['draw_weight'][i]:.2f}  {meta['describe'][i]}")
    if a.apply:
        print(apply_tags(a.catalog, tags))
    return 0


if __name__ == "__main__":
    sys.exit(main())
