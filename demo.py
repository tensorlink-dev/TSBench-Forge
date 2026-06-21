"""End-to-end demo: run the forge, show it climb, and lint a submission.

Runs with **numpy only** (no network, no optional deps). It prints:

1. the forge fitness trajectory from a deliberately weak generator state,
2. the final blend ratios and difficulty priors the forge settled on,
3. a commit-reveal reproducibility check, and
4. static-analysis output for a clean and a cheating submission.

    python demo.py
"""

from __future__ import annotations

import numpy as np

from config import CONTEXT_LEN, HORIZON, N_CHALLENGES, WEAK_STATE
from forge_loop import committed_seed, manifest_for, run_forge
from generate import build_challenges
from ingest import FreshBuffer, SyntheticLiveSource
from score import default_panel, panel_fitness
from seed import rng_for
from static_analysis import scan_submission

BLOCK_HASH = "0xforge-demo-block-hash"
EPOCHS = 20


def _bar(value: float, width: int = 28, vmax: float = 0.8) -> str:
    fill = int(np.clip(value / vmax, 0, 1) * width)
    return "#" * fill + "-" * (width - fill)


def run_demo() -> None:
    print("=" * 72)
    print("TSBench-Forge demo  --  a self-improving, hard-to-game TSFM benchmark")
    print("=" * 72)
    print(
        f"context={CONTEXT_LEN}  horizon={HORIZON}  challenges/epoch={N_CHALLENGES}  "
        f"epochs={EPOCHS}\n"
    )

    # A frozen live pool stands in for a real, post-commit data feed.
    buffer = FreshBuffer(SyntheticLiveSource(), pool_size=96, motif_len=768)
    buffer.refresh(np.random.default_rng(0xC0FFEE))

    print("Starting from a deliberately WEAK generator state:")
    print(f"  blend (synth/spliced/aug_live) = {WEAK_STATE.blend_weights()}")
    print(
        f"  difficulty: changepoint={WEAK_STATE.changepoint_prob} "
        f"regime={WEAK_STATE.regime_switch_prob} aug={WEAK_STATE.aug_severity} "
        f"phi={WEAK_STATE.noise_ar_phi}\n"
    )

    final_state, log = run_forge(buffer, EPOCHS, BLOCK_HASH, WEAK_STATE)

    print("Forge fitness trajectory (KEEP if a one-knob mutation improved fitness):")
    cols = ("epoch", "fitness", "spread", "order", "gate", "decision")
    print("  " + "  ".join(f"{c:>7}" for c in cols) + "  knob")
    for step in log:
        knob = step.knob or "-"
        print(
            f"  {step.epoch:>5}  {step.fitness:>7.3f}  {step.spread:>6.2f}  "
            f"{step.ordering:>+6.2f}  {step.gate:>5.2f}  {step.decision:<8}  {knob}  "
            f"{_bar(step.fitness)}"
        )

    f0, ff = log[0].fitness, log[-1].fitness
    keeps = sum(1 for s in log if s.decision == "keep")
    print(f"\n  fitness {f0:.3f} -> {ff:.3f}  (+{ff - f0:.3f}, {keeps} KEEPs)")

    nf = final_state.normalized()
    print("\nFinal generator state the forge settled on:")
    print(
        f"  blend  synth={nf.w_synth:.2f}  spliced={nf.w_spliced:.2f}  "
        f"aug_live={nf.w_aug_live:.2f}   (started 0.85 / 0.10 / 0.05)"
    )
    print(
        f"  diff   changepoint={nf.changepoint_prob:.2f}  regime={nf.regime_switch_prob:.2f}  "
        f"aug={nf.aug_severity:.2f}  phi={nf.noise_ar_phi:.2f}"
    )
    print("  -> the forge rebalanced away from pre-fittable synthetic-heavy data,")
    print("     which lifted the generator-fitting validity gate.")

    # ----- commit-reveal reproducibility -----------------------------------
    print("\nCommit-reveal: challenges are a pure function of the revealed seed.")
    epoch = EPOCHS
    mhash = manifest_for(final_state)
    seed = committed_seed(BLOCK_HASH, epoch, final_state)
    a = build_challenges(final_state, buffer, rng_for(BLOCK_HASH, epoch, mhash), 8)
    b = build_challenges(final_state, buffer, rng_for(BLOCK_HASH, epoch, mhash), 8)
    identical = all(
        np.array_equal(x.context, y.context) and np.array_equal(x.truth, y.truth)
        for x, y in zip(a, b, strict=True)
    )
    print(f"  manifest_hash={mhash[:16]}...  seed={seed}")
    print(f"  two replays byte-identical: {identical}")

    panel = default_panel()
    reveal = build_challenges(final_state, buffer, rng_for(BLOCK_HASH, epoch, mhash), 64)
    errs = panel_fitness(reveal, panel)["errors"]
    ranked = ", ".join(f"{m}={errs[m]:.2f}" for m in sorted(errs, key=errs.get))
    print("  reference panel (mean normalised error, best->worst): " + ranked)

    # ----- submission static analysis --------------------------------------
    print("\nSubmission static analysis (the anti-hardcoding gate):")
    clean = (
        "import numpy as np\n"
        "def forecast(context):\n"
        "    level = np.mean(context[-12:])\n"
        "    slope = (context[-1] - context[0]) / len(context)\n"
        "    return level + slope * np.arange(1, 49)\n"
    )
    table = ", ".join(str(round(0.1 * i, 2)) for i in range(22))
    cheat = (
        "import numpy as np\n"
        "import requests\n"
        f"ANSWERS = [{table}]\n"
        "def forecast(context):\n"
        "    return np.array(eval('ANSWERS'))\n"
    )
    for name, code in (("clean numpy forecaster", clean), ("hardcoded/cheating", cheat)):
        findings = scan_submission(code)
        print(f"  {name}: {len(findings)} finding(s)")
        for f in findings:
            print(f"      - {f}")

    print("\n" + "=" * 72)
    print("done.")


if __name__ == "__main__":
    run_demo()
