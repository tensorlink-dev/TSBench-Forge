"""Fetch and profile the reference benchmarks (GIFT-Eval, BOOM, TIME) for
archetype alignment (DEC-TB-0007). Data only: arrow files are read with pyarrow,
nothing downloaded is executed.

    python -m source_discovery.stat_refs fetch   --out REFDIR
    python -m source_discovery.stat_refs profile --refdir REFDIR --out ref_features.csv

What is fetched (public HF datasets, no token):
  GIFT  Salesforce/GiftEval — every config's data-*.arrow (cache-*.arrow are
        duplicates and are skipped); ~1.2 GB.
  BOOM  Datadog/BOOM — 250 of ~1,000 configs, sampled per cadence suffix with a
        fixed seed so two runs pick the same configs; ~180 MB.
  TIME  Real-TSF/TIME — all 50 configs; ~210 MB.
Profiling takes up to 40 series per config and up to 3 channels per
multivariate row, the last 4,096 points each.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random
import re
import sys
import urllib.request

import numpy as np

from source_discovery.tsfeatures import FEATURES, features

HF = "https://huggingface.co"
BOOM_SAMPLE = 250
SEED = 0


def _j(url: str):
    return json.load(urllib.request.urlopen(url, timeout=120))


def _get(repo: str, path: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    urllib.request.urlretrieve(f"{HF}/datasets/{repo}/resolve/main/{path}", dst + ".part")
    os.replace(dst + ".part", dst)


def fetch(out: str) -> None:
    data_re = re.compile(r"data-\d+-of-\d+\.arrow$")
    g = _j(f"{HF}/api/datasets/Salesforce/GiftEval/tree/main?recursive=true")
    for x in g:
        if x["type"] == "file" and data_re.search(x["path"]) and x["size"] < 320e6:
            _get("Salesforce/GiftEval", x["path"], f"{out}/gift/{x['path']}")
    dirs = [x["path"] for x in _j(f"{HF}/api/datasets/Datadog/BOOM/tree/main") if x["type"] == "directory"]
    by = collections.defaultdict(list)
    for d in sorted(dirs):
        by[d.rsplit("-", 1)[-1]].append(d)
    rng = random.Random(SEED)
    pick = []
    for f in sorted(by):
        pick += rng.sample(by[f], max(1, round(BOOM_SAMPLE * len(by[f]) / len(dirs))))
    for d in pick:
        for x in _j(f"{HF}/api/datasets/Datadog/BOOM/tree/main/{d}"):
            if data_re.search(x["path"]):
                _get("Datadog/BOOM", x["path"], f"{out}/boom/{x['path']}")
    for s in _j(f"{HF}/api/datasets/Real-TSF/TIME")["siblings"]:
        if s["rfilename"].endswith(".arrow"):
            _get("Real-TSF/TIME", s["rfilename"], f"{out}/time/{s['rfilename']}")


_SEC = {"S": 1, "T": 60, "H": 3600, "D": 86400, "B": 86400, "W": 604800,
        "M": 2629800, "Q": 7889400, "A": 31557600, "Y": 31557600}


def freq_info(f: str) -> tuple[float | None, int]:
    """(seconds per step, seasonal lag) for a pandas-style frequency string."""
    m = re.match(r"^(\d*)([A-Z]+)", str(f).upper().replace("MIN", "T"))
    if not m or m.group(2)[0] not in _SEC:
        return None, 0
    s = int(m.group(1) or 1) * _SEC[m.group(2)[0]]
    if s < 3600:
        season = int(86400 / s) if s >= 60 else int(3600 / s)
    elif s < 86400:
        season = int(86400 / s)
    elif s == 86400:
        season = 7
    elif s <= 604800:
        season = 52
    elif s <= 2629800:
        season = 12
    elif s <= 7889400:
        season = 4
    else:
        season = 0
    return s, season


def profile(refdir: str, out: str, per_config: int = 40) -> int:
    import pyarrow as pa

    rows = []
    for bench in ("gift", "boom", "time"):
        for f in sorted(glob.glob(f"{refdir}/{bench}/**/data-*.arrow", recursive=True)):
            cfg = f.split(f"/{bench}/")[1].rsplit("/", 1)[0]
            try:
                rd = pa.ipc.open_stream(open(f, "rb"))
            except Exception:
                continue
            got = 0
            for b in rd:
                fr = b.column("freq")[0].as_py() if "freq" in b.schema.names else ""
                sec, season = freq_info(fr)
                for i in range(b.num_rows):
                    if got >= per_config:
                        break
                    t = b.column("target")[i].as_py()
                    chans = t[:3] if (t and isinstance(t[0], list)) else [t]
                    for ch in chans:
                        x = np.array([np.nan if v is None else v for v in ch], dtype=float)[-4096:]
                        if np.isfinite(x).sum() < 128:
                            continue
                        rows.append({"bench": bench, "config": cfg, "freq": fr, "sec": sec or "",
                                     **features(x, season)})
                        got += 1
                if got >= per_config:
                    break
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["bench", "config", "freq", "sec", *FEATURES])
        w.writeheader()
        w.writerows(rows)
    print(f"profiled {len(rows)} reference series -> {out}")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--out", required=True)
    p = sub.add_parser("profile")
    p.add_argument("--refdir", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        fetch(a.out)
    else:
        profile(a.refdir, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
