"""Hand a round's challenges to an off-runner GPU job and read its forecasts back.

Reference models (TimesFM-3, Toto-2 2.5B) need a GPU the round workflow does
not have, so the round exports only what a forecaster may see -- each
challenge's context and horizon, never the truth -- runs them on a rented pod
(scripts/reference_forecasts.py), and replays the returned quantiles here as
ordinary leaderboard forecasters, scored on the same paired sample as every
other row.

A fingerprint of the contexts and horizons travels both ways: forecasts made
for any other challenge set are refused rather than silently misaligned.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from evaluate import DEFAULT_QUANTILES, Forecaster, ProbForecast


def _horizon(ch) -> int:
    return int(len(np.asarray(ch.truth)))


def fingerprint(challenges: list) -> str:
    h = hashlib.sha256()
    for ch in challenges:
        ctx = np.ascontiguousarray(np.asarray(ch.context, dtype=np.float64))
        h.update(ctx.tobytes())
        h.update(str(ctx.shape).encode())
        h.update(str(_horizon(ch)).encode())
    return h.hexdigest()


def export_contexts(challenges: list, path: Path) -> str:
    """Write contexts + horizons (no truth) as one npz; return the fingerprint."""
    ctxs = [np.asarray(ch.context, dtype=np.float64).reshape(-1) for ch in challenges]
    offsets = np.concatenate([[0], np.cumsum([len(c) for c in ctxs])]).astype(np.int64)
    fp = fingerprint(challenges)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        context_flat=np.concatenate(ctxs) if ctxs else np.zeros(0),
        context_offsets=offsets,
        horizons=np.asarray([_horizon(ch) for ch in challenges], dtype=np.int64),
        levels=np.asarray(DEFAULT_QUANTILES, dtype=np.float64),
        fingerprint=np.asarray(fp),
    )
    return fp


def load_reference_forecasters(directory: Path, challenges: list) -> tuple[dict[str, Forecaster], dict[str, str]]:
    """Replay every ``<model>.npz`` in ``directory`` for these challenges.

    Each file holds ``quantiles`` (N, len(levels), max_H) padded with NaN,
    ``levels`` and ``fingerprint``. Returns (forecasters, errors).
    """
    fp = fingerprint(challenges)
    by_id = {id(ch.context): i for i, ch in enumerate(challenges)}
    forecasters: dict[str, Forecaster] = {}
    errors: dict[str, str] = {}
    for f in sorted(Path(directory).glob("*.npz")):
        model = f.stem
        try:
            z = np.load(f, allow_pickle=False)
            if str(z["fingerprint"]) != fp:
                raise ValueError("forecasts were made for a different challenge set")
            q = np.asarray(z["quantiles"], dtype=np.float64)
            levels = [float(x) for x in z["levels"]]
            if q.shape[0] != len(challenges) or q.shape[1] != len(levels):
                raise ValueError(f"shape {q.shape} does not match {len(challenges)} challenges x {len(levels)} levels")
            med = levels.index(0.5)
            horizons = [_horizon(ch) for ch in challenges]
            table = {}
            for i, H in enumerate(horizons):
                block = q[i, :, :H]
                if not np.isfinite(block).all():
                    raise ValueError(f"non-finite forecast for challenge {i}")
                table[i] = ProbForecast(mean=block[med].copy(), quantiles={lv: block[k].copy() for k, lv in enumerate(levels)})

            def replay(context, meta=None, _table=table):
                return _table[by_id[id(context)]]

            forecasters[model] = replay
        except Exception as e:  # noqa: BLE001
            errors[model] = f"{type(e).__name__}: {e}"
    return forecasters, errors
