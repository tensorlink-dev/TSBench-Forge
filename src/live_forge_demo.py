"""End-to-end demo: LLM forge running against the scraped catalog.

This is the "everything wired up" flow that the earlier ``demo.py`` couldn't
run — because there was no `LiveSource` implementation that read the scraper's
parquet output. Now there is (`ScrapedLiveSource`), and the forge sees real
motifs from ~40+ working scraped sources across 35 DGP classes.

Prereq: at least one full scraper run (``cd src/sources && python scraper.py --all``)
so ``src/sources/data/<source_id>/<date>.parquet`` files exist for the adapter
to walk. In an offline environment, the fixture parquet tree in
``tests/test_scraped_source.py`` demonstrates the same wiring on synthetic data.

    python src/live_forge_demo.py                       # deterministic heuristic proposer
    OPENROUTER_API_KEY=... python src/live_forge_demo.py  # LLM proposer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_INIT_STATE, N_CHALLENGES
from forge_loop import run_forge, FOUNDATIONAL_OBJECTIVE
from ingest import FreshBuffer
from scraped_source import ScrapedLiveSource
from score import default_panel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="src/sources/sources.yaml")
    ap.add_argument("--data-dir", default="src/sources/data")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--pool-size", type=int, default=64)
    ap.add_argument("--motif-len", type=int, default=384)
    ap.add_argument("--n-challenges", type=int, default=N_CHALLENGES)
    ap.add_argument("--block-hash", default="demo-block-000")
    ap.add_argument("--min-share", type=float, default=0.02,
                    help="Breadth-gate floor per DGP class / cadence band.")
    args = ap.parse_args()

    print(f"[live_forge_demo] loading catalog: {args.catalog}")
    print(f"[live_forge_demo] data dir:        {args.data_dir}")
    src = ScrapedLiveSource(
        catalog_path=args.catalog,
        data_dir=args.data_dir,
        min_series_length=max(args.motif_len, 128),
    )
    n_available = len(src._catalog())
    print(f"[live_forge_demo] {n_available} panel-expanded series indexed")
    if n_available == 0:
        print("[live_forge_demo] No parquet found. Run the scraper first:")
        print("    cd src/sources && python scraper.py --all")
        return 1

    buffer = FreshBuffer(src, pool_size=args.pool_size, motif_len=args.motif_len)
    buffer.refresh(np.random.default_rng(0xC0FFEE))

    from collections import Counter
    dom = Counter(buffer.pool_domains)
    dgp = Counter(c for c in buffer.pool_dgp_classes if c)
    cad = Counter(c for c in buffer.pool_cadences if c)
    print(f"[live_forge_demo] pool composition:")
    print(f"    domains ({len(dom)}):     {dict(dom)}")
    print(f"    DGP classes ({len(dgp)}): {dict(dgp)}")
    print(f"    cadences ({len(cad)}):    {dict(cad)}")

    panel = default_panel()
    print(f"[live_forge_demo] running forge for {args.epochs} epochs against foundational_fitness")
    final_state, log = run_forge(
        buffer=buffer,
        epochs=args.epochs,
        block_hash=args.block_hash,
        init_state=DEFAULT_INIT_STATE,
        n_challenges=args.n_challenges,
        panel=panel,
        objective=FOUNDATIONAL_OBJECTIVE,
    )
    print(f"[live_forge_demo] done. Final state:")
    for k, v in vars(final_state).items():
        print(f"    {k:<20} {v}")
    print(f"[live_forge_demo] epoch log:")
    for step in log:
        print(f"    epoch {step.epoch:>2}  {step.decision:<6}  "
              f"knob={step.knob:<20}  fitness={step.metrics.get('foundational_fitness', 0):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
