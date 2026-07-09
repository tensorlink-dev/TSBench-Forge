#!/usr/bin/env python3
"""Render the TSFM leaderboard + pairwise-significance figure from a merged run.

Reads ``notebooks/results/merged.json`` (produced by merge_tsfm_results.py) and
writes ``notebooks/figures/tsfm_leaderboard.png``:
  * left  — seasonal-naive-relative CRPS bar (TSFMs vs classical panel)
  * right — pairwise BH-adjusted Wilcoxon p heatmap (* = pair differs at alpha)

    python scripts/plot_leaderboard.py [merged.json] [out.png]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

CLASSICAL = {"strong", "ewma", "ar1", "drift", "seasonal_naive", "context_parrot"}
TEAL, GREY, NAVY = "#1b9e77", "#9aa0a6", "#34495e"


def main(argv: list[str]) -> int:
    src = argv[0] if argv else "notebooks/results/merged.json"
    out = argv[1] if len(argv) > 1 else "notebooks/figures/tsfm_leaderboard.png"
    if not os.path.exists(src):
        print(f"no merged results at {src} — run merge_tsfm_results.py first", file=sys.stderr)
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    d = json.load(open(src))
    fr = d["friedman_crps"]
    rows = sorted(d["leaderboard"], key=lambda r: r["crps_rel"])
    names = [r["model"] for r in rows]
    vals = [r["crps_rel"] for r in rows]
    colors = [NAVY if n == "seasonal_naive" else (GREY if n in CLASSICAL else TEAL) for n in names]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw={"width_ratios": [1.15, 1]})

    yy = np.arange(len(names))[::-1]
    axA.barh(yy, vals, color=colors, edgecolor="white", height=0.72)
    axA.axvline(1.0, color="#c0392b", lw=1.4, ls="--", zorder=0)
    axA.text(1.005, len(names) - 0.4, "seasonal-naive", color="#c0392b", fontsize=8, va="top")
    for y, v in zip(yy, vals):
        axA.text(v + 0.012, y, f"{v:.3f}", va="center", fontsize=8.5, color="#222")
    axA.set_yticks(yy); axA.set_yticklabels(names, fontsize=10)
    axA.set_xlabel("CRPS relative to seasonal-naive  (lower = better)", fontsize=10)
    axA.set_xlim(0, max(vals) * 1.12)
    axA.set_title("Foundation-model leaderboard on live TSBench-Forge data", fontsize=11, weight="bold")
    axA.legend(handles=[Patch(color=TEAL, label="foundation model (TSFM)"),
                        Patch(color=GREY, label="classical baseline"),
                        Patch(color=NAVY, label="seasonal-naive ref")],
               loc="lower right", fontsize=8.5, frameon=True)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)

    pw = d["pairwise_crps"]
    keep = [n for n in pw["names"] if n not in ("drift", "ar1")]
    idx = [pw["names"].index(n) for n in keep]
    P = np.array(pw["p"])[np.ix_(idx, idx)]
    alpha = 0.05
    im = axB.imshow(np.clip(P, 0, 0.1), cmap="RdYlGn_r", vmin=0, vmax=alpha * 2)
    axB.set_xticks(range(len(keep))); axB.set_xticklabels(keep, rotation=90, fontsize=8)
    axB.set_yticks(range(len(keep))); axB.set_yticklabels(keep, fontsize=8)
    for i in range(len(keep)):
        for j in range(len(keep)):
            if i == j:
                axB.text(j, i, "—", ha="center", va="center", color="#888", fontsize=8)
            elif P[i, j] < alpha:
                axB.text(j, i, "*", ha="center", va="center", color="white", fontsize=11, weight="bold")
    axB.set_title(f"Pairwise significance (BH-adj Wilcoxon p on CRPS)\n"
                  f"* = pair differs at α={alpha}   ·   Friedman p={fr['p_value']:.1e}",
                  fontsize=11, weight="bold")
    plt.colorbar(im, ax=axB, fraction=0.046, pad=0.04).set_label("adjusted p-value", fontsize=8)

    fig.suptitle("TSBench-Forge — pretrained TSFMs, paired significance",
                 fontsize=12.5, weight="bold", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
