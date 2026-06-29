"""End-to-end demo on **real public data** — the live-feed analogue of demo.py.

Where ``demo.py`` runs the whole pipeline on the synthetic multi-domain zoo, this
script wires in genuinely real series (climate, solar, atmospheric CO2, finance,
weather) via ``live_feeds.py`` and runs the same forge -> commit-reveal ->
panel -> leaderboard -> headroom flow on them.

    python live_demo.py                 # pulls + caches the public feeds, then runs

Network: it fetches a handful of CSVs the first time and caches them under
``~/.cache/tsbench-forge/feeds`` (override with ``TSBENCH_FEED_CACHE``), so reruns
are offline. If the feeds are unreachable it falls back to the synthetic zoo so
the script still demonstrates the pipeline.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from config import CONTEXT_LEN, HORIZON, WEAK_STATE
from evaluate import (
    ProbForecast,
    benchmark_has_headroom,
    headroom,
    leaderboard,
    probabilistic_panel,
)
from forge_loop import manifest_for, run_forge
from generate import build_challenges
from ingest import FreshBuffer
from score import (
    default_panel,
    domain_coverage,
    panel_fitness,
    stratified_fitness,
    validate_panel,
)
from seed import rng_for

BLOCK_HASH = "0xlive-demo-block-hash"
EPOCHS = 18
MOTIF_LEN = 384
POOL_SIZE = 96


def _build_source():
    """Build the real multi-domain feed, falling back to the synthetic zoo offline."""
    from live_feeds import REGISTRY, build_real_live_source

    try:
        source = build_real_live_source()
        # Force a fetch now so we fail fast (and fall back) if the network is closed.
        source.pull(1, MOTIF_LEN, np.random.default_rng(0))
        print(f"Live feed: {len(REGISTRY)} real public series -> {sorted(REGISTRY)}")
        return source
    except Exception as exc:  # network closed, host blocked, schema drift, ...
        from domains import default_live_source

        print(f"Live feeds unreachable ({type(exc).__name__}: {exc}).")
        print("Falling back to the synthetic multi-domain zoo so the demo still runs.\n")
        return default_live_source()


def run() -> None:
    print("=" * 72)
    print("TSBench-Forge  --  end-to-end on REAL public time-series feeds")
    print("=" * 72)
    print(f"context={CONTEXT_LEN}  horizon={HORIZON}  epochs={EPOCHS}\n")

    source = _build_source()
    buffer = FreshBuffer(source, pool_size=POOL_SIZE, motif_len=MOTIF_LEN)
    buffer.refresh(np.random.default_rng(0xC0FFEE))
    print("Pool composition (real data-generating processes):")
    for dom, k in sorted(Counter(buffer.pool_domains).items(), key=lambda kv: -kv[1]):
        print(f"  {dom:<18} {k:>3} motifs")
    print()

    # ----- forge on real data ---------------------------------------------
    print("Running the forge on the real feed (weak -> hardened):")
    final_state, log = run_forge(buffer, EPOCHS, BLOCK_HASH, WEAK_STATE)
    keeps = sum(1 for s in log if s.decision == "keep")
    print(
        f"  fitness {log[0].fitness:.3f} -> {log[-1].fitness:.3f}  "
        f"({keeps} KEEPs over {EPOCHS} epochs)"
    )
    nf = final_state.normalized()
    print(
        f"  final blend  synth={nf.w_synth:.2f}  spliced={nf.w_spliced:.2f}  "
        f"aug_live={nf.w_aug_live:.2f}   (started 0.85 / 0.10 / 0.05)\n"
    )

    # ----- commit-reveal ---------------------------------------------------
    mhash = manifest_for(final_state)
    reveal = build_challenges(final_state, buffer, rng_for(BLOCK_HASH, EPOCHS, mhash), 128)
    a = build_challenges(final_state, buffer, rng_for(BLOCK_HASH, EPOCHS, mhash), 8)
    b = build_challenges(final_state, buffer, rng_for(BLOCK_HASH, EPOCHS, mhash), 8)
    identical = all(
        np.array_equal(x.context, y.context) and np.array_equal(x.truth, y.truth)
        for x, y in zip(a, b, strict=True)
    )
    print(f"Commit-reveal: {len(reveal)} challenges; two replays byte-identical: {identical}")

    # ----- panel validity --------------------------------------------------
    panel = default_panel()
    errs = panel_fitness(reveal, panel)["errors"]
    ranked = ", ".join(f"{m}={errs[m]:.2f}" for m in sorted(errs, key=errs.get))
    vp = validate_panel(reveal, panel)
    print(f"Reference panel (best->worst): {ranked}")
    print(
        f"Anchor validation: strong leads '{vp['runner_up']}' by "
        f"{vp['margin']:.3f} -> valid={vp['valid']}\n"
    )

    # ----- leaderboard -----------------------------------------------------
    print("Leaderboard on real data (MASE / WQL / CRPS, lower = better):")
    board = leaderboard(probabilistic_panel(), reveal)
    print(f"  {'rank':>4}  {'model':<16} {'MASE':>7} {'WQL':>7} {'CRPS':>7}")
    for r in board:
        print(
            f"  {r['rank']:>4}  {r['model']:<16} {r['mase']:>7.3f} "
            f"{r['wql']:>7.3f} {r['crps']:>7.3f}"
        )
    print(
        "  -> to score a real TSFM: "
        "leaderboard({'chronos': load_tsfm('chronos'), **probabilistic_panel()}, reveal)\n"
    )

    # ----- headroom --------------------------------------------------------
    probe_rng = np.random.default_rng(7)
    truth_by_id = {id(ch.context): np.asarray(ch.truth, dtype=float) for ch in reveal}

    def superior_probe(context, meta=None):
        tgt = truth_by_id[id(context)]
        noisy = tgt + probe_rng.normal(0.0, 0.05 * (np.std(tgt) + 1e-8), size=tgt.shape)
        deciles = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        return ProbForecast(mean=noisy, quantiles={q: noisy for q in deciles})

    hr = headroom(superior_probe, reveal)
    has_hr = benchmark_has_headroom(superior_probe, reveal)
    print(
        f"Headroom: a superior probe beats the anchor by {hr['mase_margin']:+.3f} MASE / "
        f"{hr['wql_margin']:+.3f} WQL -> has_headroom={has_hr}\n"
    )

    # ----- breadth on real domains ----------------------------------------
    cov = domain_coverage(reveal)
    print(
        f"Foundational breadth: {cov['n_domains']} real domains; "
        f"effective (exp-entropy) = {cov['effective_domains']:.2f}"
    )
    print(f"  {'domain':<18}{'n':>4}{'spread':>8}{'order':>8}{'gate':>7}")
    for dom, m in sorted(stratified_fitness(reveal, panel).items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {dom:<18}{m['n']:>4}{m['spread']:>8.2f}{m['ordering']:>+8.2f}{m['gate']:>7.2f}")

    print("\n" + "=" * 72)
    print("done — the full pipeline ran on real public data.")


if __name__ == "__main__":
    run()
