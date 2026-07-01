"""End-to-end demo on **real public data** — the whole pipeline, no forge.

The benchmark forecasts real series drawn from the live catalog (``src/sources/``).
This runs the full flow on them: build a fresh, breadth-balanced pool -> assemble
challenges (commit-reveal deterministic) -> panel validity -> MASE/WQL/CRPS
leaderboard -> headroom -> foundational breadth.

    python src/demo.py

It reads locally-scraped parquet (``src/sources/data``) if present, else the
committed trimmed fixture (``tests/fixtures/sources_data``), so it runs fully
offline — no network, no LLM, no synthetic generator.
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np

from challenges import build_live_challenges
from config import CONTEXT_LEN, HORIZON, N_CHALLENGES
from evaluate import (
    ProbForecast,
    benchmark_has_headroom,
    headroom,
    leaderboard,
    probabilistic_panel,
)
from ingest import FreshBuffer
from score import (
    default_panel,
    domain_coverage,
    foundational_fitness,
    panel_fitness,
    stratified_fitness,
    validate_panel,
)
from scraped_source import ScrapedLiveSource
from seed import rng_for

BLOCK_HASH = "0xlive-demo-block-hash"
MOTIF_LEN = 384
POOL_SIZE = 96
N_REVEAL = 128
CATALOG = os.path.join(os.path.dirname(__file__), "sources", "sources.yaml")


def _data_dir() -> str:
    """Prefer freshly-scraped data; fall back to the committed trimmed fixture."""
    here = os.path.dirname(__file__)
    live = os.path.join(here, "sources", "data")
    fixture = os.path.join(here, os.pardir, "tests", "fixtures", "sources_data")
    if os.path.isdir(live) and any(
        f.endswith(".parquet") for _, _, fs in os.walk(live) for f in fs
    ):
        return live
    return os.path.abspath(fixture)


def run() -> None:
    print("=" * 72)
    print("TSBench-Forge  --  a hard-to-game TSFM benchmark on REAL public data")
    print("=" * 72)
    print(f"context={CONTEXT_LEN}  horizon={HORIZON}  challenges={N_REVEAL}\n")

    data_dir = _data_dir()
    print(f"Live catalog: {data_dir}")
    source = ScrapedLiveSource(CATALOG, data_dir, min_series_length=MOTIF_LEN)
    buffer = FreshBuffer(source, pool_size=POOL_SIZE, motif_len=MOTIF_LEN)
    buffer.refresh(np.random.default_rng(0xC0FFEE))

    print("\nPool composition (breadth-balanced across the real catalog):")
    print(f"  domains     : {dict(Counter(buffer.pool_domains))}")
    print(f"  dgp_classes : {len(set(buffer.pool_dgp_classes))} classes")
    print(f"  cadences    : {dict(Counter(buffer.pool_cadences))}\n")

    # ----- commit-reveal ---------------------------------------------------
    reveal = build_live_challenges(buffer, rng_for(BLOCK_HASH, 0, "reveal"), N_REVEAL)
    a = build_live_challenges(buffer, rng_for(BLOCK_HASH, 0, "chk"), 8)
    b = build_live_challenges(buffer, rng_for(BLOCK_HASH, 0, "chk"), 8)
    identical = all(
        np.array_equal(x.context, y.context) and np.array_equal(x.truth, y.truth)
        for x, y in zip(a, b, strict=True)
    )
    print(f"Commit-reveal: {len(reveal)} challenges; two replays byte-identical: {identical}\n")

    # ----- panel validity --------------------------------------------------
    panel = default_panel()
    errs = panel_fitness(reveal, panel)["errors"]
    ranked = ", ".join(f"{m}={errs[m]:.2f}" for m in sorted(errs, key=errs.get))
    vp = validate_panel(reveal, panel)
    print(f"Reference panel (best->worst): {ranked}")
    print(
        f"Anchor validation: strong leads '{vp['runner_up']}' by "
        f"{vp['margin']:+.3f} -> valid={vp['valid']}"
    )
    if not vp["valid"]:
        print(
            "  NOTE: the numpy classical anchor does not lead on raw real data — expected.\n"
            "  The fix is an independently-validated zero-shot TSFM via\n"
            "  default_panel(strong_model=...), not to ignore the gate. See independent_eval.py.\n"
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

    # ----- foundational breadth -------------------------------------------
    cov = domain_coverage(reveal)
    fnd = foundational_fitness(
        reveal, panel,
        dgp_classes=buffer.pool_dgp_classes,
        cadences=buffer.pool_cadences,
    )
    print(
        f"Foundational breadth: {cov['n_domains']} real domains; "
        f"effective (exp-entropy) = {cov['effective_domains']:.2f}  "
        f"-> coverage_gate={fnd['coverage_gate']:.2f}"
    )
    print(
        f"  breadth gates: dgp_class={fnd['dgp_class_breadth_gate']:.0f}  "
        f"cadence={fnd['cadence_breadth_gate']:.0f}  parrot={fnd['parrot_gate']:.2f}"
    )
    print(f"  {'domain':<14}{'n':>4}{'spread':>8}{'order':>8}{'strong':>8}")
    for dom, m in sorted(stratified_fitness(reveal, panel).items(), key=lambda kv: -kv[1]["n"]):
        print(
            f"  {dom:<14}{m['n']:>4}{m['spread']:>8.2f}{m['ordering']:>+8.2f}"
            f"{m['difficulty']:>8.2f}"
        )

    print("\n" + "=" * 72)
    print("done — the full pipeline ran on real public data, no forge / LLM / synthetic.")


if __name__ == "__main__":
    run()
