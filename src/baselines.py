"""Floor baselines: the trivial models a *hard* benchmark must defeat.

Two baselines belong on the floor of any serious TSFM benchmark, and clearing
*both* is the minimum evidence of genuine skill:

* **seasonal-naive** -- the classical floor (already in :mod:`score`). The M4 /
  forecast-evaluation literature (Hewamalage et al. 2023) makes beating it
  mandatory.
* **context parroting** -- the floor this module adds. Zhang & Gilpin
  ("Context Parroting", arXiv:2505.11349) show a trivial nearest-neighbour
  *copy-the-most-similar-context-window* baseline beats Chronos, TimesFM,
  Moirai, Time-MoE -- and even the dynamical-systems model DynaMix -- on chaotic
  systems, at ~six orders of magnitude less compute. Much of what looks like
  "foundation-model skill" is induction-head copying; a benchmark that does not
  measure against parroting cannot tell the two apart.

The role here mirrors the ``overfit`` detector in :mod:`score`: parroting is not
a legitimate competitor a submission should lose to. A challenge set on which
parroting matches or beats the ``strong`` anchor is *not discriminating genuine
forecasting skill* -- it is rewarding repetition -- exactly as a set where a
naive baseline beats ``strong`` is rewarding an artifact. :func:`parrot_gate`
turns that into an independent, report-only validity multiplier (see
``score.parrot_gate``).
"""

from __future__ import annotations

import numpy as np

from config import CONTEXT_LEN, HORIZON

_EPS = 1e-8


def _znorm(w: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-norm window so matching is offset/scale invariant."""
    c = w - w.mean()
    nrm = float(np.linalg.norm(c))
    return c / nrm if nrm > _EPS else c


def context_parrot(
    context: np.ndarray,
    meta: dict | None = None,
    *,
    query_len: int | None = None,
    horizon: int = HORIZON,
) -> np.ndarray:
    """Nearest-neighbour "copy the future that followed the most similar past".

    Take the most recent ``query_len`` window as the query, slide it over the
    earlier context, find the historical window whose *shape* best matches
    (z-normalised correlation -- offset/scale invariant), and return the
    ``horizon`` values that followed that match, re-levelled to continue from the
    last observed point. Falls back to a last-value (persistence) forecast when
    the context is too short to hold a query plus a horizon.

    This is deliberately the cheapest non-trivial forecaster imaginable; a model
    that cannot beat it has not learned dynamics, only repetition.
    """
    x = np.asarray(context, dtype=float).reshape(-1)
    n = x.size
    q = query_len or max(8, min(2 * horizon, n // 4))
    # Need at least one candidate window [j:j+q] with a following horizon.
    if n < q + horizon + 1:
        return np.full(horizon, float(x[-1]) if n else 0.0)

    query = _znorm(x[n - q :])
    best_j, best_sim = -1, -np.inf
    # Candidate match windows must leave a full horizon of continuation, and we
    # exclude the query window itself (the trivial perfect match).
    for j in range(0, n - q - horizon):
        sim = float(np.dot(_znorm(x[j : j + q]), query))
        if sim > best_sim:
            best_sim, best_j = sim, j
    if best_j < 0:
        return np.full(horizon, float(x[-1]))

    cont = x[best_j + q : best_j + q + horizon]
    # Re-level so the copied continuation starts from the current value.
    offset = float(x[-1]) - float(x[best_j + q - 1])
    return np.asarray(cont, dtype=float) + offset


def context_parrot_for(horizon: int):
    """A ``context_parrot`` bound to a specific horizon (for non-default grids)."""

    def _parrot(context: np.ndarray, meta: dict | None = None) -> np.ndarray:
        return context_parrot(context, meta, horizon=horizon)

    return _parrot


__all__ = ["context_parrot", "context_parrot_for", "CONTEXT_LEN"]
