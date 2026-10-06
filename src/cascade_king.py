"""The Cascade subnet's current king model, as a leaderboard forecaster.

Cascade (netuid 91) trains a fixed Toto2-4M backbone on each miner's synthetic
data generator; the king is the generator whose trained model won its duels.
The trained checkpoint is published with every round, so this module turns the
latest king checkpoint into an :data:`evaluate.Forecaster` scored in-process on
CPU (it is 4M parameters), alongside the classical panel.

Resolution is read-only and needs no credentials:

1. the public receipts index (``cascade-manifests/receipts/index.json``) names
   the latest scored round and its full receipt;
2. that receipt's signed manifest lists each trained checkpoint
   (``trained_pointer``); the king's ``toto2-4m`` entry is the one used;
3. the checkpoint is pulled anonymously from the Hippius Hub registry
   (digest-pinned, so the bytes are self-verifying) and loaded with cascade's
   own ``load_forecaster``, whose ingest guard requires the checkpoint's model
   code to be byte-identical to the installed cascade release.

Requires the ``cascade`` package with its ``train`` and ``hippius`` extras.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import HORIZON
from evaluate import DEFAULT_QUANTILES, Forecaster, ProbForecast

MODEL_ID = "cascade-toto2-4m"
RECEIPTS_BASE = "https://s3.hippius.com/cascade-manifests/"
POINTER_PREFIX = "metro-v1:trained:hippius:"
NUM_SAMPLES = 256


@dataclass(frozen=True)
class KingCheckpoint:
    ref: str          # repo@sha256:digest on the Hippius Hub
    round_id: str
    king_uid: int | None
    king_hotkey: str | None
    published_at: str | None


def _get_json(url: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def resolve_king(size: str = "toto2-4m") -> KingCheckpoint:
    """The king's trained checkpoint from the latest scored round's receipt."""
    index = _get_json(RECEIPTS_BASE + "receipts/index.json")
    rounds = [r for r in index.get("rounds", []) if r.get("status") == "scored" and r.get("receipt_key")]
    if not rounds:
        raise RuntimeError("no scored cascade round in the receipts index")
    latest = max(rounds, key=lambda r: (r.get("published_at") or "", r.get("epoch_start_block") or 0))
    receipt = _get_json(RECEIPTS_BASE + latest["receipt_key"])
    for entry in (receipt.get("manifest") or {}).get("entries", []):
        pointer = entry.get("trained_pointer") or ""
        if "-king-" in pointer and size in pointer and pointer.startswith(POINTER_PREFIX):
            return KingCheckpoint(
                ref=pointer[len(POINTER_PREFIX):],
                round_id=str(latest.get("round_id")),
                king_uid=latest.get("post_round_king_uid", latest.get("king_uid")),
                king_hotkey=latest.get("post_round_king_hotkey", latest.get("king_hotkey")),
                published_at=latest.get("published_at"),
            )
    raise RuntimeError(f"round {latest.get('round_id')} has no {size} king checkpoint in its manifest")


def load_king_forecaster(cache_dir: Path, *, num_samples: int = NUM_SAMPLES,
                         quantiles: tuple[float, ...] = DEFAULT_QUANTILES) -> tuple[Forecaster, KingCheckpoint]:
    """Fetch the king checkpoint and wrap it as a leaderboard forecaster."""
    from cascade.shared.hippius import fetch_from_hub
    from cascade.validator.evaluator import load_forecaster

    king = resolve_king()
    dest = Path(cache_dir) / king.ref.split("@")[-1].replace(":", "-")
    if not (dest / "forecast_wrapper.py").is_file():
        fetch_from_hub(king.ref, dest)
    joint = load_forecaster(dest, device="cpu")
    levels = np.asarray(quantiles, dtype=float)

    def forecaster(context: np.ndarray, meta: dict | None = None) -> ProbForecast:
        horizon = int(meta["horizon"]) if isinstance(meta, dict) and meta.get("horizon") else HORIZON
        history = np.asarray(context, dtype=float)
        history = history[None, :] if history.ndim == 1 else history
        samples = np.asarray(joint(history, horizon, num_samples), dtype=float)[0]  # (m, H)
        qs = np.quantile(samples, levels, axis=0)
        median = np.quantile(samples, 0.5, axis=0)
        return ProbForecast(mean=median, quantiles={float(q): qs[i] for i, q in enumerate(levels)})

    return forecaster, king
