#!/usr/bin/env python3
"""Merge per-group TSFM results.json files into one unified significance report.

Each ``scripts/run_tsfm_comparison_lium.py --group X`` run writes
``notebooks/results/group_X/results.json`` with a ``per_challenge`` block. Because
every group scores the *same* seeded challenge set, this script unions the model
score arrays and re-runs the full leaderboard + Friedman + paired tests across the
combined roster.

    python scripts/merge_tsfm_results.py notebooks/results/group_*/results.json
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import model_comparison as mc  # noqa: E402


def main(argv: list[str]) -> int:
    paths = []
    for a in argv or ["notebooks/results/group_*/results.json"]:
        paths.extend(sorted(glob.glob(a)))
    paths = [p for p in paths if Path(p).stat().st_size > 0]
    if not paths:
        print("no non-empty results.json found", file=sys.stderr)
        return 1

    blocks, loaded = [], {}
    for p in paths:
        d = json.load(open(p))
        blocks.append(d["per_challenge"])
        for r in d.get("load_report", []):
            loaded.setdefault(r["name"], r["loaded"])
        print(f"  {p}: {sum(1 for r in d.get('load_report',[]) if r['loaded'])} loaded")

    scores = mc.merge_group_scores(blocks)
    tsfms = [n for n in scores if loaded.get(n) and n not in
             ("strong", "ewma", "ar1", "drift", "seasonal_naive", "context_parrot")]
    print(f"\nMerged roster: {len(scores)} models "
          f"({len(tsfms)} TSFMs: {tsfms})")
    print(mc.source_clustered_note(scores), "\n")

    board = mc.leaderboard_from_scores(scores)
    print("=== UNIFIED LEADERBOARD (seasonal-naive-relative, lower=better) ===")
    print(f"  {'rank':>4} {'model':<16} {'crps_rel':>9} {'mase_rel':>9} {'crps':>7} {'mase':>7}")
    for r in board:
        print(f"  {r['rank']:>4} {r['model']:<16} {r['crps_rel']:>9.3f} {r['mase_rel']:>9.3f} "
              f"{r['crps']:>7.3f} {r['mase']:>7.3f}")

    for metric in ("crps", "mase"):
        fr = mc.friedman_omnibus(scores, metric=metric)
        print(f"\nFriedman [{metric}]: chi2={fr['statistic']:.1f} p={fr['p_value']:.3g}")
    print("\n=== vs seasonal_naive (CRPS, Holm) ===")
    for r in mc.compare_to_baseline(scores, metric="crps"):
        print(f"  {r['model']:16s} med={r['median_diff']:+.3f} win={r['win_rate']:.2f} "
              f"holm_p={r['wilcoxon_p_holm']:.3g} sig={r['significant']}")

    pw = mc.pairwise_significance(scores, metric="crps")
    out = {
        "models": list(scores),
        "leaderboard": board,
        "friedman_crps": mc.friedman_omnibus(scores, metric="crps"),
        "vs_baseline_crps": mc.compare_to_baseline(scores, metric="crps"),
        "pairwise_crps": {"names": pw["names"], "p": pw["p"].tolist(), "win": pw["win"].tolist()},
    }
    dest = Path("notebooks/results/merged.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=lambda o: getattr(o, "tolist", lambda: str(o))()))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
