"""Fit the archetype model on pool + reference benchmarks, report alignment,
and freeze the model to ``data/archetypes.json`` for :mod:`stat_tag`.

Inputs are two CSVs of :mod:`tsfeatures` rows:
  pool  — columns id, domain, freq, <FEATURES>   (one row per pool series)
  ref   — columns bench, config, freq, sec, <FEATURES>  (GIFT / BOOM / TIME)

Reference weighting: each benchmark counts 1/3, and within a benchmark each
dataset (config) counts equally, so one large dataset cannot set the target.
Draw weights target the DAILY-OR-FASTER reference mix, because cascade's pool
cannot hold slower series.

Refitting renumbers clusters, so ``--freeze`` bumps ``version``; re-run
``stat_tag --apply`` afterwards.

    python -m source_discovery.stat_fit --pool pool.csv --ref ref.csv           # report
    python -m source_discovery.stat_fit --pool pool.csv --ref ref.csv --freeze
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

from source_discovery.stat_clusters import (alignment, fit, importance_weights, shares,
                                            sourcing_targets)
from source_discovery.tsfeatures import FEATURES

MODEL = Path(__file__).parent / "data" / "archetypes.json"
BENCHES = ("gift", "boom", "time")


def ref_weights(ref) -> np.ndarray:
    per_cfg = ref.groupby(["bench", "config"])["config"].transform("size")
    n_cfg = ref.groupby("bench")["config"].transform("nunique")
    return (1.0 / n_cfg / per_cfg / ref["bench"].nunique()).to_numpy()


def run(pool, ref, k: int = 12, seed: int = 0, out=print) -> dict:
    ref = ref.copy()
    ref["w"] = ref_weights(ref)
    P, R = pool[FEATURES].to_numpy(float), ref[FEATURES].to_numpy(float)
    for kk in sorted({8, k, 16}):
        a = fit(np.vstack([P, R]), k=kk, seed=seed)
        al = alignment(a.assign(P), a.assign(R), kk, ref["w"].to_numpy())
        out(f"k={kk:2d}  overlap {al['overlap']:.2f}  JS {al['js_divergence']:.3f}  "
            f"uncovered {al['uncovered_ref_mass']:.1%}")
    arch = fit(np.vstack([P, R]), k=k, seed=seed)
    pc, rc = arch.assign(P), arch.assign(R)
    names = [f"A{i:02d}" for i in range(k)]
    desc = arch.describe(3)
    fast = (ref["sec"].astype(float) <= 86400).to_numpy()
    al = alignment(pc, rc, k, ref["w"].to_numpy())
    alf = alignment(pc, rc[fast], k, ref["w"].to_numpy()[fast])
    out(f"\nall benchmarks        overlap {al['overlap']:.2f}  uncovered {al['uncovered_ref_mass']:.1%}")
    out(f"daily-or-faster only  overlap {alf['overlap']:.2f}  uncovered {alf['uncovered_ref_mass']:.1%}")
    by = {}
    for b in BENCHES:
        m = (ref["bench"] == b).to_numpy()
        if m.any():
            x = alignment(pc, rc[m], k, ref["w"].to_numpy()[m])
            by[b] = shares(rc[m], k, ref["w"].to_numpy()[m])
            out(f"   vs {b:5s} overlap {x['overlap']:.2f}  uncovered {x['uncovered_ref_mass']:.1%}")
    w = importance_weights(alf)
    eff = shares(pc, k) * w
    eff /= eff.sum()
    after = float(np.minimum(eff, alf["ref_share"]).sum())
    out(f"\n{'id':4s} {'pool':>6s} {'ref':>6s} " + " ".join(f"{b:>6s}" for b in by)
        + f" {'weight':>7s}  archetype")
    for c in np.argsort(-alf["ratio"]):
        out(f"{names[c]:4s} {alf['pool_share'][c]:6.1%} {alf['ref_share'][c]:6.1%} "
            + " ".join(f"{by[b][c]:6.1%}" for b in by) + f" {w[c]:7.2f}  {desc[c]}")
    out(f"\nreweighting the draw lifts daily-or-faster overlap {alf['overlap']:.2f} -> {after:.2f}")
    out("sourcing targets for a 4,200-series pool:")
    for c, n, _ in sourcing_targets(alf, 4200)[:6]:
        out(f"   {names[c]} +{n:4d}  {desc[c]}")
    model = arch.to_dict()
    model.update(k=k, names=names, describe=desc, draw_weight=w.round(3).tolist(),
                 weight_clip=[0.25, 4.0], pool_share=alf["pool_share"].round(4).tolist(),
                 ref_share_daily_or_faster=alf["ref_share"].round(4).tolist(),
                 ref_share_by_bench={b: v.round(4).tolist() for b, v in by.items()},
                 overlap={"all": round(al["overlap"], 3), "daily_or_faster": round(alf["overlap"], 3),
                          "after_reweighting": round(after, 3)},
                 n_pool_series=int(len(pool)), n_ref_series=int(len(ref)))
    return model


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--provenance", default="")
    a = ap.parse_args(argv)
    model = run(pd.read_csv(a.pool), pd.read_csv(a.ref), k=a.k)
    if a.freeze:
        prev = json.loads(MODEL.read_text()).get("version", 0) if MODEL.exists() else 0
        model.update(version=prev + 1, fitted=dt.date.today().isoformat(),
                     provenance=a.provenance)
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        MODEL.write_text(json.dumps(model, indent=1))
        print(f"\nfroze model v{model['version']} -> {MODEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
