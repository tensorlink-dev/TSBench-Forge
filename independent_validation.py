"""Prove an independently-validated anchor, end to end — runnable now.

This is the script the "is the anchor actually good?" question deserves. It:

1. builds a **held-out, real, external** validation benchmark (gold, taxi, births
   — disjoint from the forge feed),
2. resolves the strongest anchor this environment can run — a real **TSFM**
   (Chronos/TimesFM) if installed, else a literature-validated **statsforecast**
   model, else the numpy placeholder — and reports which,
3. **independently validates** it: it must beat every classical baseline on the
   held-out set,
4. hardens a forge benchmark, promotes the validated anchor to ``strong``, and
   confirms it also leads on the forge reveal (``validate_panel``),
5. reports the static-vs-forge leakage gap.

    python independent_validation.py

To validate a *real TSFM*: ``pip install -e ".[chronos]"`` and stage weights, then
rerun — step 2 will resolve to Chronos automatically. Without it, the script still
runs and validates the best available real model.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from config import WEAK_STATE
from evaluate import leaderboard, probabilistic_panel
from forge_loop import manifest_for, run_forge
from generate import build_challenges
from independent_eval import (
    forge_buffer,
    is_independently_validated,
    leakage_gap,
    resolve_anchor,
    validated_panel,
    validation_benchmark,
)
from score import validate_panel
from seed import rng_for

BLOCK_HASH = "0xindependent-validation"
EPOCHS = 12


def run() -> None:
    print("=" * 72)
    print("Independent validation of a forecasting anchor (TSFM if available)")
    print("=" * 72)

    # 1. Held-out, real, external benchmark -------------------------------
    print("\n[1] Held-out external validation benchmark (disjoint from the forge feed):")
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
    print("\n[3] Independent validation on the held-out set (zero-shot, not fit to the forge):")
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

    # 4. Harden a forge benchmark, promote the anchor, re-check validity ---
    print("\n[4] Forge a hard-to-game benchmark, promote the validated anchor to 'strong':")
    buffer = forge_buffer(pool_size=96, motif_len=384)
    buffer.refresh(np.random.default_rng(0xC0FFEE))
    final_state, log = run_forge(buffer, EPOCHS, BLOCK_HASH, WEAK_STATE)  # cheap numpy panel
    mhash = manifest_for(final_state)
    reveal = build_challenges(final_state, buffer, rng_for(BLOCK_HASH, EPOCHS, mhash), 96)
    print(
        f"    forge fitness {log[0].fitness:.3f} -> {log[-1].fitness:.3f}; "
        f"revealed {len(reveal)} challenges"
    )

    panel, _ = validated_panel(anchor)
    vp = validate_panel(reveal, panel)
    print(
        f"    validate_panel on the forge reveal: '{vp['runner_up']}' is runner-up, "
        f"anchor leads by {vp['margin']:+.3f} -> valid={vp['valid']}"
    )
    if not vp["valid"]:
        print("    (anchor does not lead on the forge distribution — needs a stronger TSFM anchor)")

    # 5. Where the anchor lands on the leaderboard + leakage gap ----------
    print("\n[5] Leaderboard on the forge reveal (anchor vs the reference rungs):")
    models = {f"anchor[{anchor.kind}]": anchor.forecaster, **probabilistic_panel()}
    board = leaderboard(models, reveal)
    print(f"    {'rank':>4}  {'model':<22}{'MASE':>8}{'WQL':>8}")
    for r in board:
        print(f"    {r['rank']:>4}  {r['model']:<22}{r['mase']:>8.3f}{r['wql']:>8.3f}")

    gap = leakage_gap(anchor.forecaster, reveal, benchmark=bench)
    print(
        f"\n[6] Leakage detector (track over time): held-out MASE={gap['static_mase']:.3f}  "
        f"forge MASE={gap['forge_mase']:.3f}  gap={gap['gap']:+.3f}"
    )
    print("    A forge score that races ahead of the held-out score over epochs == leakage.")

    print("\n" + "=" * 72)
    print(f"done — anchor [{anchor.kind}] was independently validated: {res.validated}")


if __name__ == "__main__":
    run()
