#!/usr/bin/env python3
"""Export one leaderboard round's challenges, exactly as a paracast round builds them.

Used to score candidate ensembles offline on the real Weir eval set, on
hardware that must never hold the mirror credentials: the round is built here
(where the secrets live) and only the challenges -- contexts, truths, tags --
leave as an artifact.

Same builder and defaults as run_paracast_round.py (seed, motif length,
challenges per draw, K jittered draws), so the export is the round today's
cron would score.

Layout (npz): context_flat + context_off (N+1,), truth (N, H) with NaN past a
challenge's own horizon, horizon (N,), and meta_json: a JSON list of each
challenge's meta dict, in round order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import tsfm_comparison  # noqa: E402
from config import K_DRAWS  # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=str(REPO / "src/sources/data"))
    ap.add_argument("--catalog", default=str(REPO / "src/sources/sources.yaml"))
    ap.add_argument("--seed", default="tsfm-significance-v1")
    ap.add_argument("--motif-len", type=int, default=304)
    ap.add_argument("--n-challenges", type=int, default=256)
    ap.add_argument("--k-draws", type=int, default=K_DRAWS)
    ap.add_argument("--out", type=Path, default=Path("round_challenges.npz"))
    args = ap.parse_args(argv)

    challenges = tsfm_comparison.build_challenges(
        args.data_dir, catalog=args.catalog, motif_len=args.motif_len,
        n_challenges=args.n_challenges, seed=args.seed, k_draws=args.k_draws,
    )
    n = len(challenges)
    horizon = max(len(ch.truth) for ch in challenges)
    truth = np.full((n, horizon), np.nan, dtype=np.float64)
    for i, ch in enumerate(challenges):
        truth[i, : len(ch.truth)] = ch.truth
    ctx = [np.asarray(ch.context, dtype=np.float64) for ch in challenges]
    meta = [{k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v))
             for k, v in (ch.meta or {}).items()} for ch in challenges]
    np.savez_compressed(
        args.out,
        context_flat=np.concatenate(ctx),
        context_off=np.concatenate([[0], np.cumsum([c.size for c in ctx])]).astype(np.int64),
        truth=truth,
        horizon=np.array([len(ch.truth) for ch in challenges]),
        meta_json=np.array(json.dumps(meta)),
    )
    sources = {m.get("source_id") for m in meta}
    print(f"exported {n} challenges from {len(sources)} sources, horizon {horizon} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
