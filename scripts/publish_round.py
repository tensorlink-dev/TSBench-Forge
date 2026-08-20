#!/usr/bin/env python3
"""Publish a leaderboard round to ``docs/data/rounds/`` for the public tracker.

Each merged GPU run (``scripts/merge_tsfm_results.py`` → ``notebooks/results/
merged.json``) becomes one dated *round* file under ``docs/data/rounds/``, and
``docs/data/rounds/index.json`` is rebuilt from whatever round files exist.
GitHub Pages serves ``docs/`` as-is, so the rounds become a public, CORS-open
JSON feed that the cascade-frontend leaderboard tracker fetches:

    https://tensorlink-dev.github.io/TSBench-Forge/data/rounds/index.json

Typical use, right after a merge::

    python scripts/publish_round.py \
        --config-from notebooks/results/group_self/results.json \
        --n-series 43 --note "post-autoresearch pool"

Round identity is the UTC date (``--round-id`` overrides, e.g. to backfill).
Re-publishing the same round id overwrites that round file — a round is the
*latest* merged result for its date.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CLASSICAL = {"strong", "ewma", "ar1", "drift", "seasonal_naive", "context_parrot"}

SCHEMA_VERSION = 1

# Keys copied per-model from merged.json's vs_baseline_crps block (the full
# block also carries raw/unadjusted p-values; the tracker only needs these).
_VS_BASELINE_KEYS = ("model", "median_diff", "win_rate", "wilcoxon_p_holm", "significant")


def build_round(merged: dict, round_id: str, config: dict | None = None,
                note: str | None = None) -> dict:
    """Shape one merged result into the published round envelope."""
    board = [
        {k: row.get(k) for k in ("rank", "model", "crps_rel", "mase_rel", "crps", "mase")}
        for row in merged["leaderboard"]
    ]
    out: dict = {
        "schema_version": SCHEMA_VERSION,
        "round_id": round_id,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config or {},
        "leaderboard": board,
    }
    if note:
        out["note"] = note
    if "composition" in merged:
        # DEC-TB-0003 transparency: the realised domain mix of THIS round —
        # pooled and per draw — so anyone can verify the jitter rotates and
        # averages back to equal-weight. Post-hoc only: it reveals nothing
        # about the next round's draw.
        out["composition"] = merged["composition"]
    if "friedman_crps" in merged:
        out["friedman_crps"] = merged["friedman_crps"]
    if "vs_baseline_crps" in merged:
        out["vs_baseline_crps"] = [
            {k: row.get(k) for k in _VS_BASELINE_KEYS} for row in merged["vs_baseline_crps"]
        ]
    return out


def round_summary(round_doc: dict) -> dict:
    """One index.json entry for a round file."""
    board = round_doc.get("leaderboard", [])
    ordered = sorted(board, key=lambda r: r.get("rank") or 1 << 30)
    out = {
        "round_id": round_doc["round_id"],
        "generated_at": round_doc.get("generated_at"),
        "n_models": len(board),
        "n_tsfms": sum(1 for r in board if r["model"] not in CLASSICAL),
        "top": [r["model"] for r in ordered[:3]],
        "has_metrics": any(r.get("crps_rel") is not None for r in board),
    }
    cfg = round_doc.get("config") or {}
    picked = {k: cfg[k] for k in ("seed", "n_challenges", "n_series", "k_draws")
              if k in cfg}
    if picked:
        out["config"] = picked
    eff = (round_doc.get("composition") or {}).get("effective_domains")
    if eff is not None:
        out["effective_domains"] = eff
    if round_doc.get("note"):
        out["note"] = round_doc["note"]
    return out


_DERIVED = {"index.json", "history.json"}


def _round_docs(rounds_dir: Path) -> list[dict]:
    docs = [
        json.loads(p.read_text())
        for p in sorted(rounds_dir.glob("*.json"))
        if p.name not in _DERIVED
    ]
    docs.sort(key=lambda d: d["round_id"])
    return docs


def rebuild_index(rounds_dir: Path) -> dict:
    """Rebuild index.json from every round file present in ``rounds_dir``."""
    entries = [round_summary(d) for d in _round_docs(rounds_dir)]
    index = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rounds": entries,
    }
    (rounds_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def rebuild_history(rounds_dir: Path) -> dict:
    """Rebuild history.json: the whole rank/CRPS history in one compact file.

    The tracker charts hundreds of rounds; fetching one file per round does not
    scale, so this is their feed. Per round, ``ranks``, ``crps_rel`` and
    ``mase_rel`` are arrays aligned with the top-level ``models`` list (null
    where a model did not run). Round files stay the detailed per-round record.
    """
    docs = _round_docs(rounds_dir)
    models: list[str] = []
    for d in docs:
        for row in d["leaderboard"]:
            if row["model"] not in models:
                models.append(row["model"])
    rounds = []
    for d in docs:
        by_model = {r["model"]: r for r in d["leaderboard"]}
        rounds.append({
            "round_id": d["round_id"],
            "ranks": [by_model.get(m, {}).get("rank") for m in models],
            "crps_rel": [by_model.get(m, {}).get("crps_rel") for m in models],
            "mase_rel": [by_model.get(m, {}).get("mase_rel") for m in models],
        })
    history = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "models": models,
        "rounds": rounds,
    }
    (rounds_dir / "history.json").write_text(json.dumps(history) + "\n")
    return history


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--merged", default=str(REPO / "notebooks/results/merged.json"),
                    help="merged.json from merge_tsfm_results.py")
    ap.add_argument("--round-id", default=datetime.now(UTC).strftime("%Y-%m-%d"),
                    help="round identity, default today (UTC)")
    ap.add_argument("--config-from", default=None,
                    help="a group results.json whose config block (seed/motif_len/"
                         "n_challenges) describes this run")
    ap.add_argument("--n-series", type=int, default=None,
                    help="number of distinct source series the challenges drew from")
    ap.add_argument("--note", default=None, help="free-text provenance note")
    ap.add_argument("--docs-data", default=str(REPO / "docs/data"),
                    help="output data dir (docs/data)")
    ap.add_argument("--index-only", action="store_true",
                    help="skip publishing, just rebuild index.json from existing rounds")
    args = ap.parse_args(argv)

    rounds_dir = Path(args.docs_data) / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)

    if not args.index_only:
        merged = json.loads(Path(args.merged).read_text())
        config: dict = {}
        if args.config_from:
            src = json.loads(Path(args.config_from).read_text()).get("config", {})
            config = {k: src[k] for k in ("seed", "motif_len", "n_challenges") if k in src}
        if args.n_series is not None:
            config["n_series"] = args.n_series
        doc = build_round(merged, args.round_id, config=config, note=args.note)
        dest = rounds_dir / f"{args.round_id}.json"
        dest.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {dest} ({len(doc['leaderboard'])} models)")

    index = rebuild_index(rounds_dir)
    history = rebuild_history(rounds_dir)
    print(f"wrote {rounds_dir / 'index.json'} ({len(index['rounds'])} rounds)")
    print(f"wrote {rounds_dir / 'history.json'} ({len(history['models'])} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
