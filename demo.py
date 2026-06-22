"""End-to-end demo: run the forge, show it climb, and lint a submission.

Runs with **numpy only** (no network, no optional deps). It prints:

1. the forge fitness trajectory from a deliberately weak generator state,
2. the final blend ratios and difficulty priors the forge settled on,
3. a commit-reveal reproducibility check,
4. foundational breadth: coverage across data-generating processes, and
5. static-analysis output for a clean and a cheating submission.

    python demo.py
"""

from __future__ import annotations

import numpy as np

from config import CONTEXT_LEN, HORIZON, N_CHALLENGES, WEAK_STATE
from domains import LorenzSource, default_live_source
from evaluate import (
    benchmark_has_headroom,
    headroom,
    leaderboard,
    probabilistic_panel,
)
from forge_llm import OpenRouterConfig, make_openrouter_proposer
from forge_loop import committed_seed, manifest_for, run_forge
from generate import build_challenges
from ingest import FreshBuffer
from sandbox import run_submission
from score import (
    DEFAULT_COVERAGE_TARGET,
    default_panel,
    domain_coverage,
    foundational_fitness,
    panel_fitness,
    stratified_fitness,
    validate_panel,
)
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

    # A frozen, multi-domain live pool stands in for a real, post-commit feed:
    # a zoo of distinct data-generating processes (random walk + Lorenz, Rössler,
    # Hopf, Hénon, logistic, OU, jump-diffusion) so the benchmark is broad enough
    # to certify a *foundation* model, not just one process.
    buffer = FreshBuffer(default_live_source(), pool_size=96, motif_len=768)
    buffer.refresh(np.random.default_rng(0xC0FFEE))

    print("Starting from a deliberately WEAK generator state:")
    print(f"  blend (synth/spliced/aug_live) = {WEAK_STATE.blend_weights()}")
    print(
        f"  difficulty: changepoint={WEAK_STATE.changepoint_prob} "
        f"regime={WEAK_STATE.regime_switch_prob} aug={WEAK_STATE.aug_severity} "
        f"phi={WEAK_STATE.noise_ar_phi}\n"
    )

    # The forge proposer is the LLM boundary. With OPENROUTER_API_KEY set, a real
    # model (Claude Opus 4.8 by default) proposes each one-knob move via
    # OpenRouter; otherwise -- and on any API error -- it falls back to the
    # deterministic heuristic, so the demo runs identically offline.
    cfg = OpenRouterConfig.from_env()
    if cfg.enabled:
        print(f"Forge proposer: OpenRouter LLM ({cfg.model}) with heuristic fallback.\n")
        proposer = make_openrouter_proposer(cfg, on_fallback=lambda r: None)
    else:
        print("Forge proposer: deterministic heuristic (set OPENROUTER_API_KEY for the LLM).\n")
        proposer = None

    final_state, log = run_forge(buffer, EPOCHS, BLOCK_HASH, WEAK_STATE, proposer=proposer)

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
    reveal = build_challenges(final_state, buffer, rng_for(BLOCK_HASH, epoch, mhash), 128)
    errs = panel_fitness(reveal, panel)["errors"]
    ranked = ", ".join(f"{m}={errs[m]:.2f}" for m in sorted(errs, key=errs.get))
    print("  reference panel (mean normalised error, best->worst): " + ranked)

    # The anchor is the load-bearing assumption; validate it loudly rather than
    # trusting it. A production validator runs this at startup with require=True.
    vp = validate_panel(reveal, panel)
    print(
        f"  anchor validation: strong leads '{vp['runner_up']}' by "
        f"{vp['margin']:.3f} -> valid={vp['valid']}"
    )

    # ----- evaluate models under test: the leaderboard ---------------------
    print("\nModel evaluation (this is what scores an actual TSFM):")
    board = leaderboard(probabilistic_panel(), reveal)
    print(f"  {'rank':>4}  {'model':<16} {'MASE':>7} {'WQL':>7} {'CRPS':>7}")
    for r in board:
        print(f"  {r['rank']:>4}  {r['model']:<16} {r['mase']:>7.3f} "
              f"{r['wql']:>7.3f} {r['crps']:>7.3f}")
    print("  -> to score a real TSFM: `leaderboard({'chronos': "
          "load_tsfm('chronos'), **probabilistic_panel()}, reveal)`")

    # Headroom (a setup-time go/no-go): can a model strictly BETTER than the
    # classical anchor actually win here? If not, the benchmark cannot certify
    # TSFM quality, whatever its leaderboard says. We inject a deliberately-strong
    # truth-informed probe (the true future plus small noise -- better than any
    # real model, but not perfect) and confirm the benchmark rewards it. A real
    # TSFM's *real* standing comes from the leaderboard above.
    from evaluate import ProbForecast

    probe_rng = np.random.default_rng(7)
    truth_by_id = {id(ch.context): np.asarray(ch.truth, dtype=float) for ch in reveal}

    def _superior_probe(context, meta=None):
        tgt = truth_by_id[id(context)]
        noisy = tgt + probe_rng.normal(0.0, 0.05 * (np.std(tgt) + 1e-8), size=tgt.shape)
        return ProbForecast(mean=noisy, quantiles={q: noisy for q in
                            (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)})

    hr = headroom(_superior_probe, reveal)
    has = benchmark_has_headroom(_superior_probe, reveal)
    print(
        f"  headroom check: a deliberately-superior probe beats the anchor by "
        f"{hr['mase_margin']:+.3f} MASE / {hr['wql_margin']:+.3f} WQL "
        f"-> has_headroom={has}"
    )
    print("     (the benchmark can separate a better-than-classical model; a real "
          "TSFM's standing is the leaderboard.)")

    # ----- foundational breadth: coverage across DGPs ----------------------
    print("\nFoundational breadth: a high score must mean 'good across worlds'.")
    cov = domain_coverage(reveal)
    fnd = foundational_fitness(reveal, panel)
    print(
        f"  the reveal spans {cov['n_domains']} domains; "
        f"effective domains (exp-entropy) = {cov['effective_domains']:.2f} "
        f"-> coverage_gate = {fnd['coverage_gate']:.2f} (target {DEFAULT_COVERAGE_TARGET:.0f})"
    )
    print(
        f"  fitness={fnd['fitness']:.3f}  x  coverage_gate={fnd['coverage_gate']:.2f}  "
        f"=  foundational_fitness={fnd['foundational_fitness']:.3f}"
    )
    print("\n  per-domain validity & discrimination (n, strong-err, spread, order, gate):")
    for dom, m in sorted(stratified_fitness(reveal, panel).items(), key=lambda kv: -kv[1]["n"]):
        print(
            f"    {dom:<14} n={m['n']:>3}  strong={m['difficulty']:>5.2f}  "
            f"spread={m['spread']:>4.2f}  order={m['ordering']:>+4.2f}  gate={m['gate']:>4.2f}"
        )

    # A single-domain feed is sharp but NARROW: same generator, but the coverage
    # gate collapses, so its breadth-aware (foundational) score is penalised.
    narrow_buf = FreshBuffer(LorenzSource(), pool_size=96, motif_len=768)
    narrow = build_challenges(final_state, narrow_buf, rng_for(BLOCK_HASH, epoch, mhash), 128)
    nf = foundational_fitness(narrow, panel)
    print(
        f"\n  narrow (Lorenz-only) feed: effective domains = "
        f"{nf['coverage']['effective_domains']:.2f}  -> coverage_gate "
        f"{fnd['coverage_gate']:.2f} (broad) vs {nf['coverage_gate']:.2f} (narrow)"
    )
    print("  -> breadth is measured, not assumed: the coverage gate rewards many DGPs.")

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

    # The static scan is only a pre-filter; the real boundary is sandboxed
    # execution. Run both submissions through it against a real context.
    print("\nSandboxed execution (the real boundary: isolated, resource-limited):")
    ctx = reveal[0].context
    for name, code in (("clean numpy forecaster", clean), ("hardcoded/cheating", cheat)):
        res = run_submission(code, ctx)
        detail = f"shape={res.prediction.shape}" if res.ok else (res.error or res.findings)
        print(f"  {name}: status={res.status}  ({detail})")

    print("\n" + "=" * 72)
    print("done.")


if __name__ == "__main__":
    run_demo()
