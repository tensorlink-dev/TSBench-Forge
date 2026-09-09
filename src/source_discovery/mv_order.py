"""Reorder each ``mv_channels`` tag so channel 0 is the column served today.

Cascade's PR #250 builder changes what an UNARMED tagged source serves: instead
of the densest numeric column (today's behaviour), it projects to
``mv_channels[0]``. If our first-listed channel differs from the densest
column, the live univariate series for that source silently changes the moment
the PR merges — a regime break for miners with no data reason behind it. This
pass makes the projection a no-op: for every tagged source, find the densest
numeric column in the newest parquet (the builder's own pick, approximated at
file level) and move it to the front of the tag, keeping the rest in order.

Sources whose densest column is NOT in the tag at all (it was curated out — an
ID, a calendar column) are reported, not rewritten: their served series changes
at merge whichever order we pick, and the right first channel is a judgement
call already encoded by the curation order.

    .venv/bin/python -m source_discovery.mv_order --data-dir <live data> [--apply]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path


def _densest_column(data_dir: Path, sid: str) -> str | None:
    import pandas as pd
    import pyarrow.parquet as pq

    d = data_dir / sid
    files = sorted(d.glob("*.parquet")) if d.is_dir() else []
    if not files:
        return None
    df = pq.read_table(files[-1]).to_pandas()
    best, best_n = None, -1
    for c in df.columns:
        if c == "timestamp" or c.startswith("_panel_"):
            continue
        n = int(pd.to_numeric(df[c], errors="coerce").notna().sum())
        if n > best_n:
            best, best_n = c, n
    return best if best_n > 0 else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--catalog", default="src/sources/sources.yaml", type=Path)
    ap.add_argument("--data-dir", default="src/sources/data", type=Path)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    import yaml

    text = args.catalog.read_text()
    cat = yaml.safe_load(text)
    tagged = [(s["id"], s["mv_channels"]) for s in cat if "mv_channels" in s]

    reorder: dict[str, list[str]] = {}
    mismatched, nodata, ok = [], [], []
    for sid, chans in tagged:
        densest = _densest_column(args.data_dir, sid)
        if densest is None:
            nodata.append(sid)
        elif densest not in chans:
            mismatched.append((sid, densest, chans[0]))
        elif chans[0] != densest:
            reorder[sid] = [densest] + [c for c in chans if c != densest]
        else:
            ok.append(sid)

    print(f"tagged {len(tagged)}: first-channel already densest {len(ok)}, "
          f"reorder {len(reorder)}, densest-not-in-tag {len(mismatched)}, no data {len(nodata)}")
    for sid, new in sorted(reorder.items()):
        print(f"  REORDER {sid}: {new}")
    for sid, densest, first in sorted(mismatched):
        print(f"  MISMATCH {sid}: densest '{densest}' curated out; serving flips to '{first}' at merge")

    if args.apply and reorder:
        lines = text.splitlines(keepends=True)
        cur = None
        for i, ln in enumerate(lines):
            m = re.match(r"- id: (\S+)\s*$", ln)
            if m:
                cur = m.group(1)
            elif cur in reorder and re.match(r"  mv_channels: \[", ln):
                lines[i] = "  mv_channels: " + json.dumps(reorder[cur], ensure_ascii=False) + "\n"
        new_text = "".join(lines)
        new = yaml.safe_load(new_text)
        assert len(new) == len(cat)
        got = {s["id"]: s["mv_channels"] for s in new if "mv_channels" in s}
        for sid, want in reorder.items():
            assert got[sid] == want, sid
            assert sorted(want) == sorted(dict(tagged)[sid]), sid  # same set, new order
        fd, tmp = tempfile.mkstemp(dir=args.catalog.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            f.write(new_text)
        os.replace(tmp, args.catalog)
        print(f"applied: {len(reorder)} tags reordered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
