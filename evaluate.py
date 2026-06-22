"""Model-under-test evaluation: the half that actually scores a TSFM.

The forge (``forge_loop``/``forge_llm``) keeps the benchmark hard-to-game and the
panel (``score``) decides whether a challenge set is *valid*. Neither one scores a
**submitted model**. This module closes that gap: given a forecaster and the
revealed challenges, it computes the metrics the TSFM literature actually uses and
ranks the model on a leaderboard against the reference panel.

What a "proper TSFM" needs that point-MAE could not give
--------------------------------------------------------
Real TSFM benchmarks (GIFT-Eval, the Chronos / Moirai / TimesFM papers) score
*probabilistic* forecasts, not single points. A model emits quantiles and is
judged on both:

* **MASE** -- mean absolute scaled error (point accuracy), scaled by the in-sample
  naive error so it is comparable across series of different scales.
* **WQL** -- weighted quantile loss (a.k.a. weighted pinball / normalized CRPS),
  the probabilistic headline metric: does the model's *uncertainty* match reality?
* **CRPS** -- continuous ranked probability score, approximated on a dense quantile
  grid, reported alongside WQL.

A :class:`ProbForecast` (mean + quantiles) is the model contract. Point-only
forecasters (the classical panel) are lifted to probabilistic via
:func:`probabilistic` so every model is scored on the same probabilistic footing.

Headroom
--------
A benchmark only *certifies* TSFMs if a genuinely-better model scores
measurably better than the classical anchor on it. :func:`headroom` measures
exactly that margin, and :func:`benchmark_has_headroom` is the go/no-go check: if
a model known to be superior cannot beat the anchor here, the benchmark has no
discriminating power in the range that matters and its scores are not meaningful.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from score import default_panel

# The decile grid GIFT-Eval / Chronos report WQL on; CRPS uses a denser grid.
DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_CRPS_QUANTILES: tuple[float, ...] = tuple(round(q, 3) for q in np.linspace(0.05, 0.95, 19))
_EPS = 1e-8


@dataclass(frozen=True)
class ProbForecast:
    """A probabilistic forecast: a point ``mean`` plus per-level ``quantiles``.

    ``mean`` and every quantile array are HORIZON-length. A real TSFM fills
    ``quantiles`` from its predictive distribution; a point model is lifted by
    :func:`probabilistic`.
    """

    mean: np.ndarray
    quantiles: dict[float, np.ndarray]


# A forecaster maps (context, meta) -> ProbForecast. Real TSFMs ignore ``meta``.
Forecaster = Callable[[np.ndarray, "dict | None"], ProbForecast]


# --------------------------------------------------------------------------- #
# Probit: standard-normal quantile (Acklam), so point models get sane spreads
# without depending on scipy.
# --------------------------------------------------------------------------- #


def _probit(p: float) -> float:
    """Inverse standard-normal CDF via Acklam's rational approximation."""
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
         -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
         3.754408661907416e0]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _naive_scale(context: np.ndarray, m: int = 1) -> float:
    """In-sample seasonal-naive MAE used to scale MASE (m=1 == non-seasonal)."""
    x = np.asarray(context, dtype=float)
    if x.size <= m:
        return _EPS
    s = float(np.mean(np.abs(x[m:] - x[:-m])))
    return s if s > _EPS else float(np.std(x) + _EPS)


# --------------------------------------------------------------------------- #
# Lifting point forecasters to probabilistic
# --------------------------------------------------------------------------- #


def probabilistic(
    point_fn: Callable[[np.ndarray, dict | None], np.ndarray],
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> Forecaster:
    """Wrap a point forecaster into a :class:`Forecaster`.

    The spread is estimated from the context's one-step volatility and widened
    with the forecast horizon (a random-walk ``sqrt(h)`` law), giving a defensible
    Gaussian predictive band. This is deliberately *generic*: a real TSFM with a
    genuinely sharper, better-calibrated distribution should beat these bands on
    WQL/CRPS -- which is exactly the signal the benchmark needs to reward.
    """

    def forecaster(context: np.ndarray, meta: dict | None = None) -> ProbForecast:
        point = np.asarray(point_fn(context, meta), dtype=float)
        sigma0 = _naive_scale(context)
        widen = np.sqrt(np.arange(1, len(point) + 1))
        sigma_h = sigma0 * widen
        qs = {q: point + _probit(q) * sigma_h for q in quantiles}
        return ProbForecast(mean=point, quantiles=qs)

    return forecaster


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _pinball(truth: np.ndarray, pred_q: np.ndarray, q: float) -> np.ndarray:
    """Elementwise pinball (quantile) loss at level ``q``."""
    diff = truth - pred_q
    return np.maximum(q * diff, (q - 1) * diff)


@dataclass
class _Accum:
    abs_err: float = 0.0
    scaled_n: float = 0.0  # sum of per-challenge MASE
    wql_num: float = 0.0
    wql_den: float = 0.0
    crps_sum: float = 0.0
    crps_n: int = 0
    n: int = 0


def evaluate_forecaster(
    forecaster: Forecaster,
    challenges: list,
    *,
    seasonality: int = 1,
) -> dict[str, float]:
    """Score one forecaster over the challenge set.

    Returns ``mase`` (lower better), ``wql`` and ``crps`` (probabilistic, lower
    better), ``mae``, and ``n``. WQL is pooled (sum numerators / sum
    denominators) the way GIFT-Eval aggregates, so large-magnitude series do not
    dominate by scale.
    """
    acc = _Accum()
    for ch in challenges:
        truth = np.asarray(ch.truth, dtype=float)
        meta = getattr(ch, "meta", None)
        fc = forecaster(ch.context, meta)
        mean = np.asarray(fc.mean, dtype=float)

        acc.abs_err += float(np.mean(np.abs(truth - mean)))
        acc.scaled_n += float(np.mean(np.abs(truth - mean))) / _naive_scale(ch.context, seasonality)

        denom = float(np.sum(np.abs(truth)))
        acc.wql_den += denom
        for q in DEFAULT_QUANTILES:
            acc.wql_num += 2.0 * float(np.sum(_pinball(truth, fc.quantiles[q], q)))

        # CRPS ~ 2 * mean pinball over a dense quantile grid. The model only emits
        # a coarse decile grid, so we interpolate ITS quantiles onto the dense grid
        # (per horizon step) -- using the model's real distribution, not a
        # synthesized band, so a perfectly-calibrated model scores ~0.
        levels = sorted(fc.quantiles)
        q_stack = np.stack([np.asarray(fc.quantiles[lv], dtype=float) for lv in levels])
        for q in _CRPS_QUANTILES:
            pq = np.array([np.interp(q, levels, q_stack[:, h]) for h in range(q_stack.shape[1])])
            acc.crps_sum += 2.0 * float(np.mean(_pinball(truth, pq, q)))
            acc.crps_n += 1
        acc.n += 1

    n = max(acc.n, 1)
    return {
        "mase": acc.scaled_n / n,
        "wql": acc.wql_num / (len(DEFAULT_QUANTILES) * max(acc.wql_den, _EPS)),
        "crps": acc.crps_sum / max(acc.crps_n, 1),
        "mae": acc.abs_err / n,
        "n": float(acc.n),
    }


# --------------------------------------------------------------------------- #
# Leaderboard + headroom
# --------------------------------------------------------------------------- #


def probabilistic_panel() -> dict[str, Forecaster]:
    """The reference panel as probabilistic forecasters (rungs on the leaderboard).

    Excludes the ``overfit`` detector (it is anti-gaming machinery, not a model a
    real submission competes against).
    """
    return {
        name: probabilistic(fn)
        for name, fn in default_panel().items()
        if name != "overfit"
    }


def leaderboard(
    forecasters: dict[str, Forecaster],
    challenges: list,
    *,
    primary: str = "mase",
    seasonality: int = 1,
) -> list[dict[str, object]]:
    """Score every named forecaster and return rows ranked by ``primary`` (asc).

    Pass the models under test merged with :func:`probabilistic_panel` to see
    where a submission lands relative to the known-quality reference rungs.
    """
    rows = []
    for name, fc in forecasters.items():
        m = evaluate_forecaster(fc, challenges, seasonality=seasonality)
        rows.append({"model": name, **m})
    rows.sort(key=lambda r: r[primary])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def headroom(
    candidate: Forecaster,
    challenges: list,
    *,
    anchor: Forecaster | None = None,
    seasonality: int = 1,
) -> dict[str, object]:
    """Margin of a candidate model over the classical anchor.

    Positive margins mean the candidate is *better* than the anchor (lower error),
    i.e. the benchmark has room above the classical baseline for this model to
    demonstrate skill. ``anchor`` defaults to the probabilistic ``strong`` model.
    """
    anchor = anchor or probabilistic(default_panel()["strong"])
    cand = evaluate_forecaster(candidate, challenges, seasonality=seasonality)
    base = evaluate_forecaster(anchor, challenges, seasonality=seasonality)
    return {
        "candidate": cand,
        "anchor": base,
        "mase_margin": base["mase"] - cand["mase"],
        "wql_margin": base["wql"] - cand["wql"],
        "crps_margin": base["crps"] - cand["crps"],
        "beats_anchor": bool(cand["mase"] < base["mase"] and cand["wql"] <= base["wql"]),
    }


def benchmark_has_headroom(
    known_better: Forecaster,
    challenges: list,
    *,
    min_margin: float = 0.0,
    seasonality: int = 1,
) -> bool:
    """Go/no-go: can a model known to be superior actually beat the anchor here?

    If this is False, the benchmark cannot separate a better-than-classical model
    from the anchor and its scores do not certify TSFM quality. Run it at setup
    with a deliberately-strong probe before trusting any leaderboard.
    """
    h = headroom(known_better, challenges, seasonality=seasonality)
    return bool(h["mase_margin"] > min_margin)
