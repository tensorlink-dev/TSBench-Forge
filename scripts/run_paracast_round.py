#!/usr/bin/env python3
"""Run one leaderboard round through the paracast endpoint and publish it.

The GPU-pod path (scripts/run_tsfm_comparison_lium.py → merge → publish) needs
a human to rent hardware, so rounds happened when someone had an afternoon.
This runner replaces the pod with HTTP: paracast already keeps the open-weights
panel loaded behind `POST /forecast`, so a round becomes a CPU-only loop that a
scheduled workflow can run (.github/workflows/paracast-round.yml) — which is
what makes the public leaderboard *live*.

    PARACAST_URL=https://…chutes.ai python scripts/run_paracast_round.py \
        --data-dir src/sources/data --publish --feedback all

End to end:

1. build the deterministic live challenge set (same seeding as the pod path);
2. fetch forecasts from paracast — one explicit row per enabled panel model,
   plus the `paracast-ensemble` and `paracast-router` pseudo-models, which
   compete as first-class leaderboard rows;
3. POST the realized truths back to `/feedback`, so paracast's EWMA ensemble
   weights and router log learn from the round (inference comes from paracast,
   evaluation flows back — see src/paracast_client.py for the division);
4. score everything CLIENT-SIDE with the standard evaluate/model_comparison
   stack (per-source seasonal-naive-relative shifted gmean + paired tests) —
   identical arithmetic to pod rounds, so the histories concatenate honestly;
5. write the usual results.json / merged-shape artifacts and, with --publish,
   the docs/data round + index + history the tracker serves.

Round identity stays the UTC date: publishing twice a day overwrites the
day's round (the contract says a round is the latest result for its date).
Pass --round-id to backfill or to publish sub-daily with a suffix.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import model_comparison as mc  # noqa: E402
import tsfm_comparison  # noqa: E402
from evaluate import probabilistic_panel  # noqa: E402
from paracast_client import (  # noqa: E402
    ModelSpec,
    ParacastClient,
    ParacastError,
    prefetch_forecasts,
    send_feedback,
)


def round_composition(challenges: list, k_draws: int) -> dict:
    """The realised domain mix of one round — the DEC-TB-0003 manifest.

    Computed from the challenge metas actually served (pooled and per draw),
    so the published feed carries proof that the jitter rotates the mix and
    that the pooled round stays broad. Post-hoc by construction: each round's
    Dirichlet draw is independent, so this predicts nothing about tomorrow.
    """
    from collections import Counter

    import numpy as np

    def _counts(chs: list, key: str) -> dict:
        return dict(sorted(Counter(
            str((getattr(ch, "meta", None) or {}).get(key)) for ch in chs
        ).items()))

    pooled = _counts(challenges, "domain")
    total = sum(pooled.values())
    p = np.array([v / total for v in pooled.values() if v], dtype=float)
    eff = float(np.exp(-(p * np.log(p)).sum())) if p.size else 0.0
    per_draw = max(1, len(challenges) // max(k_draws, 1))
    draws = [challenges[i:i + per_draw]
             for i in range(0, len(challenges), per_draw)]
    return {
        "domains": pooled,
        "domains_per_draw": [_counts(d, "domain") for d in draws],
        "effective_domains": round(eff, 2),
        "cadences": _counts(challenges, "cadence"),
        "n_dgp_classes": len({(getattr(ch, "meta", None) or {}).get("dgp_class")
                              for ch in challenges}),
    }


def _load_publish_round():
    spec = importlib.util.spec_from_file_location(
        "publish_round", REPO / "scripts/publish_round.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_specs(client: ParacastClient, args) -> list[ModelSpec]:
    """The rows to fetch: the live panel (or --models subset) + pseudo-models."""
    panel = client.panel()
    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        missing = [m for m in wanted if m not in panel]
        if missing:
            print(f"warning: not in the enabled+healthy panel, skipping: {missing}")
        panel = [m for m in wanted if m in panel]
    specs = [ModelSpec.explicit(name) for name in panel]
    if not args.no_ensemble:
        specs.append(ModelSpec.ensemble())
    if not args.no_router:
        specs.append(ModelSpec.route())
    return specs


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default=os.environ.get("PARACAST_URL"),
                    help="paracast endpoint (default: $PARACAST_URL)")
    ap.add_argument("--api-key", default=os.environ.get("PARACAST_API_KEY"),
                    help="bearer token if the gateway needs one (default: $PARACAST_API_KEY)")
    ap.add_argument("--data-dir", default=str(REPO / "src/sources/data"),
                    help="scraped parquet root (src/sources/data)")
    ap.add_argument("--catalog", default=str(REPO / "src/sources/sources.yaml"))
    ap.add_argument("--seed", default="tsfm-significance-v1",
                    help="challenge seed — keep the default so paracast rounds pair "
                         "with pod rounds on identical challenge sets")
    ap.add_argument("--motif-len", type=int, default=304)
    ap.add_argument("--n-challenges", type=int, default=256,
                    help="challenges PER DRAW; the round scores k-draws × this many")
    ap.add_argument("--k-draws", type=int, default=None,
                    help="independent jittered draws pooled into the round verdict "
                         "(default: config.K_DRAWS). DEC-TB-0003 jitters each "
                         "draw's mix; pooling K draws is what keeps the verdict's "
                         "seed-error margin tight. Evals here are cheap HTTP "
                         "against paracast's already-loaded panel.")
    ap.add_argument("--models", default=None,
                    help="comma-separated subset of the panel (default: all enabled+healthy)")
    ap.add_argument("--no-ensemble", action="store_true",
                    help="skip the paracast-ensemble pseudo-model row")
    ap.add_argument("--no-router", action="store_true",
                    help="skip the paracast-router pseudo-model row")
    ap.add_argument("--feedback", choices=("all", "ensemble", "off"), default="all",
                    help="which served requests get their truths POSTed back to /feedback "
                         "(default all: every model's EWMA skill sees every round)")
    ap.add_argument("--chunk", type=int, default=64, help="series per /forecast request")
    ap.add_argument("--out-dir", default=str(REPO / "notebooks/results/group_paracast"))
    ap.add_argument("--publish", action="store_true",
                    help="also write docs/data/rounds/<round-id>.json + index + history")
    ap.add_argument("--docs-data", default=str(REPO / "docs/data"),
                    help="feed root for --publish (docs/data; point elsewhere to rehearse)")
    ap.add_argument("--round-id", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    ap.add_argument("--note", default="scored via paracast",
                    help="provenance note shown in the round log")
    args = ap.parse_args(argv)

    if not args.base_url:
        print("error: --base-url or $PARACAST_URL is required", file=sys.stderr)
        return 2

    client = ParacastClient(args.base_url, api_key=args.api_key)
    try:
        health = client.health()
    except ParacastError as e:
        print(f"error: endpoint unreachable: {e}", file=sys.stderr)
        return 1
    print(f"paracast {args.base_url}: {health.get('status')} "
          f"({len(health.get('healthy', []))} healthy workers)")

    from config import K_DRAWS
    k_draws = args.k_draws if args.k_draws is not None else K_DRAWS
    print(f"building challenges ({k_draws} jittered draws)…")
    challenges = tsfm_comparison.build_challenges(
        args.data_dir, catalog=args.catalog, motif_len=args.motif_len,
        n_challenges=args.n_challenges, seed=args.seed, k_draws=k_draws,
    )
    n_series = len({str((getattr(ch, "meta", None) or {}).get("source_id"))
                    for ch in challenges})
    composition = round_composition(challenges, k_draws)
    print(f"  {len(challenges)} challenges from {n_series} source series, "
          f"effective domains {composition['effective_domains']}")

    specs = build_specs(client, args)
    if not specs:
        print("error: nothing to fetch — empty panel and pseudo-models disabled",
              file=sys.stderr)
        return 1
    print(f"fetching forecasts for {len(specs)} rows…")
    prefetch = prefetch_forecasts(client, challenges, specs, chunk=args.chunk)
    if not prefetch.forecasts:
        print("error: every model failed — not publishing an empty round", file=sys.stderr)
        return 1

    if args.feedback != "off":
        modes = ("explicit", "route", "ensemble") if args.feedback == "all" else ("ensemble",)
        fb = send_feedback(client, challenges, prefetch.requests, modes=modes)
        print(f"feedback: {fb['sent']} sent, {fb['failed']} failed, {fb['skipped']} skipped")

    # Client-side scoring: paracast rows replay their prefetched forecasts, the
    # classical panel runs in-process — one paired sample across all of them.
    models = {mid: prefetch.forecaster(mid) for mid in prefetch.forecasts}
    for name, fc in probabilistic_panel().items():
        models.setdefault(name, fc)
    print(f"scoring {len(models)} models client-side…")
    scores = mc.score_models(models, challenges)
    board = mc.leaderboard_from_scores(scores)
    note_breadth = mc.source_clustered_note(scores)

    load_report = (
        [{"name": s.model_id, "loaded": s.model_id in prefetch.forecasts,
          "error": prefetch.errors.get(s.model_id)} for s in specs]
        + [{"name": n, "loaded": True, "error": None} for n in models
           if not any(s.model_id == n for s in specs)]
    )

    _first = next(iter(scores.values()))
    per_challenge = {
        "source_ids": _first.source_ids.tolist(),
        "models": {
            name: {"mase": s.mase.tolist(), "crps": s.crps.tolist(),
                   "weight": s.weight.tolist()}
            for name, s in scores.items()
        },
    }
    results = {
        "config": {
            "data_dir": str(args.data_dir),
            "motif_len": args.motif_len,
            "n_challenges": len(challenges),
            "k_draws": k_draws,
            "seed": args.seed,
            "roster": [s.model_id for s in specs],
            "endpoint": "paracast",
        },
        "composition": composition,
        "load_report": load_report,
        "note": note_breadth,
        "leaderboard": board,
        "per_challenge": per_challenge,
        "by_metric": {},
    }
    pairwise = {}
    for metric in ("crps", "mase"):
        fr = mc.friedman_omnibus(scores, metric=metric)
        vb = mc.compare_to_baseline(scores, metric=metric)
        pw = mc.pairwise_significance(scores, metric=metric)
        pairwise[metric] = pw
        results["by_metric"][metric] = {
            "friedman": fr,
            "vs_baseline": vb,
            "pairwise": {"names": pw["names"], "p": pw["p"].tolist(),
                         "win": pw["win"].tolist(), "better": pw["better"].tolist()},
        }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=_json_default) + "\n")
    print(f"wrote {out_dir / 'results.json'}")

    print(f"\n=== ROUND {args.round_id} (seasonal-naive-relative, lower=better) ===")
    for r in board:
        print(f"  {r['rank']:>4} {r['model']:<18} crps_rel={r['crps_rel']:.3f} "
              f"mase_rel={r['mase_rel']:.3f}")
    print(f"\n{note_breadth}")

    # merged.json shape (what publish_round consumes) — written beside the
    # group results so a paracast round never clobbers a pod merge in flight.
    merged = {
        "models": list(scores),
        "leaderboard": board,
        "composition": composition,
        "friedman_crps": results["by_metric"]["crps"]["friedman"],
        "vs_baseline_crps": results["by_metric"]["crps"]["vs_baseline"],
        "pairwise_crps": {
            "names": pairwise["crps"]["names"],
            "p": pairwise["crps"]["p"].tolist(),
            "win": pairwise["crps"]["win"].tolist(),
        },
    }
    merged_path = out_dir / "merged.json"
    merged_path.write_text(json.dumps(merged, indent=2, default=_json_default) + "\n")
    print(f"wrote {merged_path}")

    if args.publish:
        publish_round = _load_publish_round()
        rounds_dir = Path(args.docs_data) / "rounds"
        rounds_dir.mkdir(parents=True, exist_ok=True)
        config = {"seed": args.seed, "motif_len": args.motif_len,
                  "n_challenges": len(challenges), "n_series": n_series,
                  "k_draws": k_draws}
        doc = publish_round.build_round(merged, args.round_id, config=config, note=args.note)
        dest = rounds_dir / f"{args.round_id}.json"
        dest.write_text(json.dumps(doc, indent=2) + "\n")
        publish_round.rebuild_index(rounds_dir)
        publish_round.rebuild_history(rounds_dir)
        print(f"published {dest} + index.json + history.json")

    return 0


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
