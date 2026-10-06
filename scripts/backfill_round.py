#!/usr/bin/env python3
"""Add new models to an already-published round, without re-running it.

The round workflow uploads each round's per-challenge scores (``results.json``)
and its exact challenges (``challenges.pkl.gz``) as the run's artifact, kept 30
days, so a model added later is scored on the very same challenges and merged
into the round: nothing is re-queried from Ephemeris and no existing row
changes. Rounds from before the challenges were saved are rebuilt from the
run's mirror cache (``hippius-mirror-<run id>``) instead, which only works
while GitHub still holds it: the 10 GB repo limit keeps the latest three.

Two steps, run by .github/workflows/backfill-round.yml:

    # 1. rebuild the round's challenges, prove they are the same, export contexts
    python scripts/backfill_round.py export --results results.json \
        --round-doc docs/data/rounds/2026-10-06.json --contexts ref/contexts.npz

    # 2. (forecast ref/contexts.npz on a GPU) score, merge and republish
    python scripts/backfill_round.py publish --results results.json \
        --round-doc docs/data/rounds/2026-10-06.json --reference-forecasts ref/out

Either way the challenges must prove themselves. Rebuilt ones use the round's
own seed and settings with the truth cutoff pinned to the round's date; saved
or rebuilt, the in-process seasonal-naive baseline must reproduce the round's
published per-challenge scores exactly. Any drift refuses the backfill.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def _challenges(args, results: dict, round_doc: dict) -> list:
    """The round's challenges: saved with the run if it saved them, else rebuilt."""
    saved = Path(args.results).with_name("challenges.pkl.gz")
    if saved.is_file():
        import gzip
        import pickle

        with gzip.open(saved, "rb") as fh:  # our own run artifact
            challenges = pickle.load(fh)
        print(f"loaded {len(challenges)} challenges saved with the round")
        return challenges
    return _rebuild(results, round_doc, args.data_dir, args.catalog)


def _rebuild(results: dict, round_doc: dict, data_dir: str, catalog: str) -> list:
    """The round's challenges, rebuilt under the round's own cutoff date."""
    import challenges as ch_mod
    import tsfm_comparison

    # The only clock read on the round path is the daily truth cutoff (the
    # freshness window is unset). Challenges are built moments after the round
    # id is stamped, so the round id's date is the build date even when the
    # publish crossed midnight. Patching the one function, rather than faking
    # datetime globally, keeps pyarrow and pandas happy (freezegun segfaults them).
    rid = str(round_doc["round_id"])
    day = rid if len(rid) == 10 else str(round_doc["generated_at"])[:10]
    cfg = results["config"]
    k = int(cfg["k_draws"])
    real = ch_mod.default_cutoff
    ch_mod.default_cutoff = lambda: np.datetime64(day)
    try:
        return tsfm_comparison.build_challenges(
            data_dir, catalog=catalog, motif_len=int(cfg["motif_len"]),
            n_challenges=int(cfg["n_challenges"]) // k, seed=cfg["seed"], k_draws=k,
        )
    finally:
        ch_mod.default_cutoff = real


def _stored_scores(results: dict) -> dict:
    from model_comparison import ModelScores

    pc = results["per_challenge"]
    sids = np.asarray(pc["source_ids"])
    return {
        name: ModelScores(name=name, mase=np.asarray(s["mase"], dtype=float),
                          crps=np.asarray(s["crps"], dtype=float),
                          weight=np.asarray(s["weight"], dtype=float), source_ids=sids)
        for name, s in pc["models"].items()
    }


def _verify(challenges: list, stored: dict) -> None:
    """Refuse unless the rebuilt challenges reproduce the round exactly."""
    import model_comparison as mc

    base = stored["seasonal_naive"]
    if len(challenges) != len(base.crps):
        raise SystemExit(f"refused: rebuilt {len(challenges)} challenges, the round had {len(base.crps)}")
    redo = mc.score_models({}, challenges)["seasonal_naive"]
    same_sources = [str(s) for s in redo.source_ids] == [str(s) for s in base.source_ids]
    if not same_sources:
        raise SystemExit("refused: rebuilt challenges come from different series than the round's")
    for metric in ("crps", "mase", "weight"):
        a, b = getattr(redo, metric), getattr(base, metric)
        if not np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
            worst = float(np.nanmax(np.abs(a - b)))
            raise SystemExit(f"refused: seasonal-naive {metric} differs from the round's (max |diff| {worst:.3g})")
    print(f"verified: {len(challenges)} rebuilt challenges reproduce the round's seasonal-naive scores exactly")


def cmd_export(args, results, round_doc) -> int:
    from reference_io import export_contexts

    challenges = _challenges(args, results, round_doc)
    _verify(challenges, _stored_scores(results))
    fp = export_contexts(challenges, Path(args.contexts))
    print(f"exported {len(challenges)} contexts to {args.contexts} (fingerprint {fp[:16]})")
    return 0


def cmd_publish(args, results, round_doc) -> int:
    import model_comparison as mc
    from reference_io import load_reference_forecasters

    import publish_round

    challenges = _challenges(args, results, round_doc)
    stored = _stored_scores(results)
    _verify(challenges, stored)

    refs, errors = load_reference_forecasters(Path(args.reference_forecasts), challenges)
    for name, err in errors.items():
        print(f"warning: {name} not scored: {err}", file=sys.stderr)
    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        refs = {n: f for n, f in refs.items() if n in wanted}
    if not refs:
        print("error: no new forecasts to add", file=sys.stderr)
        return 1
    new = mc.score_models(refs, challenges, include_seasonal_naive=False)
    replaced = sorted(set(new) & set(stored))
    scores = {**stored, **new}
    print(f"adding {sorted(new)}" + (f" (replacing {replaced})" if replaced else ""))

    board = mc.leaderboard_from_scores(scores)
    vb = mc.compare_to_baseline(scores, metric="crps")
    pw = mc.pairwise_significance(scores, metric="crps")
    merged = {
        "models": list(scores),
        "leaderboard": board,
        "composition": round_doc.get("composition") or results.get("composition"),
        "friedman_crps": mc.friedman_omnibus(scores, metric="crps"),
        "vs_baseline_crps": vb,
        "pairwise_crps": {"names": pw["names"], "p": pw["p"].tolist(), "win": pw["win"].tolist()},
    }
    added = ", ".join(sorted(new))
    note = round_doc.get("note") or ""
    note = f"{note}; backfilled {added}" if note else f"backfilled {added}"
    doc = publish_round.build_round(merged, round_doc["round_id"], config=round_doc.get("config"), note=note)
    doc["generated_at"] = round_doc["generated_at"]  # the round's time, not the backfill's

    rounds_dir = Path(args.docs_data) / "rounds"
    dest = rounds_dir / f"{round_doc['round_id']}.json"
    dest.write_text(json.dumps(doc, indent=2, default=_json_default) + "\n")
    publish_round.rebuild_index(rounds_dir)
    publish_round.rebuild_history(rounds_dir)

    # The extended per-challenge record, uploaded as the backfill's artifact.
    results["per_challenge"]["models"].update({
        n: {"mase": s.mase.tolist(), "crps": s.crps.tolist(), "weight": s.weight.tolist()}
        for n, s in new.items()
    })
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, default=_json_default) + "\n")

    print(f"\n=== ROUND {round_doc['round_id']} with {added} ===")
    for r in board:
        mark = "  +" if r["model"] in new else "   "
        print(f"{mark}{r['rank']:>3} {r['model']:<20} crps_rel={r['crps_rel']:.3f} mase_rel={r['mase_rel']:.3f}")
    print(f"published {dest} + index.json + history.json")
    return 0


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("export", "publish"))
    ap.add_argument("--results", required=True, help="the round's results.json (run artifact)")
    ap.add_argument("--round-doc", required=True, help="docs/data/rounds/<round-id>.json")
    ap.add_argument("--data-dir", default=str(REPO / "src/sources/data"))
    ap.add_argument("--catalog", default=str(REPO / "src/sources/sources.yaml"))
    ap.add_argument("--contexts", default="ref/contexts.npz")
    ap.add_argument("--reference-forecasts", default="ref/out")
    ap.add_argument("--models", default=None, help="only add these (default: every forecast file)")
    ap.add_argument("--docs-data", default=str(REPO / "docs/data"))
    ap.add_argument("--out-dir", default=str(REPO / "notebooks/results/group_paracast"))
    args = ap.parse_args(argv)

    results = json.loads(Path(args.results).read_text())
    round_doc = json.loads(Path(args.round_doc).read_text())
    return (cmd_export if args.step == "export" else cmd_publish)(args, results, round_doc)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
