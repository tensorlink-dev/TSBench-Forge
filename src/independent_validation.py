"""Prove an independently-validated anchor, end to end — runnable now.

This is the script the "is the anchor actually good?" question deserves. It:

1. builds a **held-out, real, external** validation benchmark (gold, taxi, births
   — disjoint from the live feed),
2. resolves the strongest anchor this environment can run — a real **TSFM**
   (Chronos/TimesFM) if installed, else a literature-validated **statsforecast**
   model, else the numpy placeholder — and reports which,
3. **independently validates** it: it must beat every classical baseline on the
   held-out set,
4. assembles the live benchmark, promotes the validated anchor to ``strong``, and
   confirms it also leads on the live reveal (``validate_panel``),
5. reports the static-vs-live leakage gap.

    python independent_validation.py

To validate a *real TSFM*: ``pip install -e ".[chronos]"`` and stage weights, then
rerun — step 2 will resolve to Chronos automatically. Without it, the script still
runs and validates the best available real model.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from challenges import build_live_challenges
from evaluate import leaderboard, probabilistic_panel
from independent_eval import (
    is_independently_validated,
    leakage_gap,
    live_buffer,
    resolve_anchor,
    validated_panel,
    validation_benchmark,
)
from score import validate_panel
from seed import rng_for

BLOCK_HASH = "0xindependent-validation"


def run() -> None:
    print("=" * 72)
    print("Independent validation of a forecasting anchor (TSFM if available)")
    print("=" * 72)

    # 1. Held-out, real, external benchmark -------------------------------
    print("\n[1] Held-out external validation benchmark (disjoint from the live feed):")
    try:
        bench = validation_benchmark(n_per_domain=24)
    except Exception as exc:
        print(f"    could not build the real held-out set ({type(exc).__name__}: {exc}).")
        print("    This step needs network on first run; aborting.")
        return
    doms = Counter(c.meta["domain"] for c in bench)
    for dom, k in doms.items():
        print(f"    {dom:<20} {k:>3} tasks")

    # 2. Resolve the strongest available real anchor ----------------------
    anchor = resolve_anchor()
    print(f"\n[2] Anchor under validation: [{anchor.kind}] {anchor.detail}")
    if anchor.kind == "numpy":
        print("    (No TSFM/statsforecast available — install '.[chronos]' for a real TSFM.)")

    # 3. Independent validation: beat every classical baseline ------------
    print("\n[3] Independent validation on the held-out set (zero-shot, not fit to the live feed):")
    res = is_independently_validated(
        anchor.forecaster, name=f"anchor[{anchor.kind}]", benchmark=bench
    )
    print(f"    {'rank':>4}  {'model':<22}{'MASE':>8}{'WQL':>8}{'CRPS':>8}")
    for r in res.board:
        mark = "  <- anchor" if r["model"].startswith("anchor[") else ""
        print(
            f"    {r['rank']:>4}  {r['model']:<22}{r['mase']:>8.3f}"
            f"{r['wql']:>8.3f}{r['crps']:>8.3f}{mark}"
        )
    verdict = "VALIDATED" if res.validated else "NOT validated"
    print(
        f"    -> {verdict}: leads best baseline by {res.margin:+.3f} MASE"
        + (f" (beaten by {res.beaten_by})" if res.beaten_by else "")
    )

    # 4. Assemble the live benchmark, promote the anchor, re-check validity ---
    print("\n[4] Assemble the live benchmark, promote the validated anchor to 'strong':")
    buffer = live_buffer(pool_size=96, motif_len=384)
    buffer.refresh(np.random.default_rng(0xC0FFEE))
    reveal = build_live_challenges(buffer, rng_for(BLOCK_HASH, 0, "reveal"), 96)
    print(f"    revealed {len(reveal)} challenges from the live benchmark feed")

    panel, _ = validated_panel(anchor)
    vp = validate_panel(reveal, panel)
    print(
        f"    validate_panel on the live reveal: '{vp['runner_up']}' is runner-up, "
        f"anchor leads by {vp['margin']:+.3f} -> valid={vp['valid']}"
    )
    if not vp["valid"]:
        print("    (anchor does not lead on the live distribution — needs a stronger TSFM anchor)")

    # 5. Where the anchor lands on the leaderboard + leakage gap ----------
    print("\n[5] Leaderboard on the live reveal (anchor vs the reference rungs):")
    models = {f"anchor[{anchor.kind}]": anchor.forecaster, **probabilistic_panel()}
    board = leaderboard(models, reveal)
    print(f"    {'rank':>4}  {'model':<22}{'MASE':>8}{'WQL':>8}")
    for r in board:
        print(f"    {r['rank']:>4}  {r['model']:<22}{r['mase']:>8.3f}{r['wql']:>8.3f}")

    gap = leakage_gap(anchor.forecaster, reveal, benchmark=bench)
    print(
        f"\n[6] Leakage detector (track over time): held-out MASE={gap['static_mase']:.3f}  "
        f"live MASE={gap['live_mase']:.3f}  gap={gap['gap']:+.3f}"
    )
    print("    A live-reveal score that races ahead of the held-out score over epochs == leakage.")

    print("\n" + "=" * 72)
    print(f"done — anchor [{anchor.kind}] was independently validated: {res.validated}")


if __name__ == "__main__":
    run()
