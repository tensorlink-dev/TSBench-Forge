#!/usr/bin/env python3
"""Forecast a round's exported challenges with reference models, on a GPU pod.

Input: the npz written by ``run_paracast_round.py --export-contexts`` (contexts
and horizons only -- this process never sees the truth). Output: one
``<model>.npz`` per model in ``--out`` with ``quantiles`` (N, levels, max_H,
NaN-padded), ``levels`` and the round's ``fingerprint``, which
``run_paracast_round.py --reference-forecasts`` replays and scores.

Models (each needs its own package; run each in its own venv):

  timesfm3     google/timesfm-3.0-pytorch   (pip: timesfm[torch] >= 3)
  toto2-2.5b   Datadog/Toto-2.0-2.5B        (pip: toto-models)

Both are scored zero-shot with their published defaults: TimesFM-3 through
``TimesFM3Forecaster.predict_batch``; Toto-2 with the same context budget,
patch padding and masking the served Toto-2 313M uses in paracast.

    python scripts/reference_forecasts.py --contexts contexts.npz --out ref/ --models timesfm3
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_contexts(path: Path):
    z = np.load(path, allow_pickle=False)
    flat, off = z["context_flat"], z["context_offsets"]
    contexts = [flat[off[i]:off[i + 1]].astype(np.float64) for i in range(len(off) - 1)]
    return contexts, z["horizons"].astype(int), [float(x) for x in z["levels"]], str(z["fingerprint"])


def to_levels(values: np.ndarray, native: list[float], wanted: list[float]) -> np.ndarray:
    """(n, len(native), H) quantile values -> (n, len(wanted), H) by linear interpolation in level."""
    native = np.asarray(native, dtype=float)
    order = np.argsort(native)
    native, values = native[order], values[:, order, :]
    if np.allclose(native, wanted):
        return values
    out = np.empty((values.shape[0], len(wanted), values.shape[2]))
    for k, q in enumerate(wanted):
        j = int(np.clip(np.searchsorted(native, q), 1, len(native) - 1))
        w = (q - native[j - 1]) / (native[j] - native[j - 1])
        out[:, k, :] = (1 - w) * values[:, j - 1, :] + w * values[:, j, :]
    return out


class TimesFM3:
    name = "timesfm3"

    def __init__(self, device: str):
        import timesfm

        self.model = timesfm.TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device=device)
        cfg = getattr(self.model, "config", None)
        self.levels = [float(q) for q in getattr(cfg, "quantiles", [0.1 * i for i in range(1, 10)])]

    def predict(self, contexts: list[np.ndarray], horizon: int) -> np.ndarray:
        outs = list(self.model.predict_batch(contexts, horizon, return_quantiles=True))
        q = np.stack([np.asarray(o.quantiles, dtype=float) for o in outs])  # (n, H, L)
        return np.transpose(q, (0, 2, 1))  # (n, L, H)


class Toto25B:
    name = "toto2-2.5b"
    budget, patch, decode_block = 4096, 32, 768

    def __init__(self, device: str):
        import torch
        from toto2 import Toto2Model

        self.torch = torch
        self.device = torch.device(device)
        self.model = Toto2Model.from_pretrained("Datadog/Toto-2.0-2.5B").to(self.device).eval()
        knots = getattr(getattr(self.model, "output_head", None), "knots", None)
        self.levels = [float(k) for k in knots] if knots else [0.1 * i for i in range(1, 10)]

    def predict(self, contexts: list[np.ndarray], horizon: int) -> np.ndarray:
        torch = self.torch
        # Context + decoded horizon share Toto-2's positional budget.
        cap = max(self.patch, self.budget - (math.ceil(horizon / self.patch) - 1) * self.patch)
        ctx = [c[-cap:] for c in contexts]
        width = max(len(c) for c in ctx)
        width += (-width) % self.patch
        values = np.zeros((len(ctx), 1, width))
        observed = np.zeros((len(ctx), 1, width), dtype=bool)
        for i, c in enumerate(ctx):
            ok = np.isfinite(c)
            values[i, 0, width - len(c):] = np.where(ok, c, 0.0)
            observed[i, 0, width - len(c):] = ok
        with torch.no_grad():
            out = self.model.forecast(
                {
                    "target": torch.as_tensor(values, dtype=torch.float32, device=self.device),
                    "target_mask": torch.as_tensor(observed, dtype=torch.bool, device=self.device),
                    "series_ids": torch.zeros((len(ctx), 1), dtype=torch.long, device=self.device),
                },
                horizon=horizon,
                decode_block_size=self.decode_block,
                has_missing_values=bool(not observed.all()),
            )
        q = out.detach().float().cpu().numpy()  # (Q, B, V, H)
        return np.transpose(q[:, :, 0, :], (1, 0, 2))  # (B, Q, H)


MODELS = {"timesfm3": TimesFM3, "toto2-2.5b": Toto25B}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--contexts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--models", required=True, help=f"comma list of {sorted(MODELS)}")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    contexts, horizons, levels, fp = load_contexts(args.contexts)
    args.out.mkdir(parents=True, exist_ok=True)
    by_h = defaultdict(list)
    for i, h in enumerate(horizons):
        by_h[int(h)].append(i)
    failed = 0
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        t0 = time.time()
        try:
            model = MODELS[name](args.device)
            q = np.full((len(contexts), len(levels), int(horizons.max())), np.nan)
            for h, idx in by_h.items():
                for s in range(0, len(idx), args.batch):
                    chunk = idx[s:s + args.batch]
                    pred = model.predict([contexts[i] for i in chunk], h)
                    q[chunk, :, :h] = to_levels(pred, model.levels, levels)
            for i, h in enumerate(horizons):
                if not np.isfinite(q[i, :, :h]).all():
                    raise ValueError(f"non-finite forecast for challenge {i}")
            np.savez_compressed(args.out / f"{name}.npz", quantiles=q,
                                levels=np.asarray(levels), fingerprint=np.asarray(fp))
            print(f"{name}: {len(contexts)} challenges in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"{name}: FAILED {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        finally:
            model = None
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
