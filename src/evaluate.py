"""Model-under-test evaluation: the half that actually scores a TSFM.

The panel (``score``) decides whether a challenge set is *valid*, but it does not
score a **submitted model**. This module closes that gap: given a forecaster and
the revealed challenges, it computes the metrics the TSFM literature actually uses
and ranks the model on a leaderboard against the reference panel.

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

import math
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
        # Season longer than the context: degrade to the non-seasonal scale
        # rather than returning epsilon (which would explode MASE).
        m = 1
        if x.size <= 1:
            return _EPS
    s = float(np.mean(np.abs(x[m:] - x[:-m])))
    return s if s > _EPS else float(np.std(x) + _EPS)


# Season length per ISO-8601 sampling interval, the gluonts/GIFT-Eval convention:
# one natural cycle in steps (daily cycle for sub-daily data; the calendar cycle
# above that). Frequencies absent here score with m=1 (non-seasonal MASE).
FREQ_SEASONALITY: dict[str, int] = {
    "PT30S": 120,   # 1 hour
    "PT1M": 1440,   # 1 day
    "PT2M30S": 576,
    "PT5M": 288,
    "PT6M": 240,
    "PT10M": 144,
    "PT15M": 96,
    "PT30M": 48,
    "PT1H": 24,
    "PT8H": 3,
    "P1D": 1,
    "P1W": 1,
    "P1M": 12,
    "P1Q": 4,
    "P1Y": 1,
}


def season_length(freq: str | None, context_len: int) -> int:
    """Season length ``m`` for MASE scaling, derived from the sampling interval.

    Falls back to 1 (non-seasonal) when the frequency is unknown or the full
    cycle does not fit inside the context — the same degradation gluonts applies
    — so high-frequency series with a daily cycle longer than the context are
    scored non-seasonally rather than against an inestimable season.
    """
    m = FREQ_SEASONALITY.get(freq or "", 1)
    return m if 1 < m < context_len else 1


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


# Central prediction intervals (from the symmetric decile pairs) used by WIS, and
# the matching alpha = 1 - nominal-coverage of each interval.
_WIS_PAIRS: tuple[tuple[float, float], ...] = ((0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6))
_WIS_ALPHAS: tuple[float, ...] = tuple(round(1.0 - (hi - lo), 3) for lo, hi in _WIS_PAIRS)


def _interval_score(truth: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> np.ndarray:
    """Elementwise interval score (Gneiting & Raftery) for a central interval."""
    return (hi - lo) + (2.0 / alpha) * (lo - truth) * (truth < lo) \
        + (2.0 / alpha) * (truth - hi) * (truth > hi)


@dataclass
class _Accum:
    abs_err: float = 0.0
    scaled_n: float = 0.0  # sum of per-challenge MASE
    wql_num: float = 0.0
    wql_den: float = 0.0
    crps_sum: float = 0.0
    crps_n: float = 0.0
    wis_sum: float = 0.0  # sum of per-challenge mean WIS
    cov80_sum: float = 0.0  # sum of per-challenge [q10,q90] coverage fraction
    cov_hits: dict | None = None  # per-quantile-level count of truth <= q (for PCE)
    cov_total: float = 0.0  # total (challenge x horizon) points seen
    n: int = 0
    wsum: float = 0.0  # sum of per-challenge unseen weights
    unseen_sum: float = 0.0  # sum of per-challenge unseen_frac (diagnostic)


def evaluate_forecaster(
    forecaster: Forecaster,
    challenges: list,
    *,
    seasonality: int | None = None,
) -> dict[str, float]:
    """Score one forecaster over the challenge set.

    Returns ``mase`` (lower better), ``wql`` and ``crps`` (probabilistic, lower
    better), ``mae``, and ``n``. WQL is pooled (sum numerators / sum
    denominators) the way GIFT-Eval aggregates, so large-magnitude series do not
    dominate by scale.

    ``seasonality`` fixes one MASE season length for every challenge; the default
    (``None``) derives it per challenge from ``meta["freq"]`` via
    :func:`season_length`, so a mixed-frequency set scales each series by its own
    seasonal-naive error (m=1 when the frequency is untagged or unknown).

    **Unseen weighting** — every challenge contributes with weight
    ``UNSEEN_WEIGHT_FLOOR + (1 - UNSEEN_WEIGHT_FLOOR) * meta["unseen_frac"]``:
    truth that postdates the daily cutoff (which no pretrained model can have
    memorised) carries full weight, fully-historical truth carries the floor.
    Challenge sets without the tag weigh uniformly, so results are unchanged
    for legacy/synthetic sets.
    """
    from config import UNSEEN_WEIGHT_FLOOR

    acc = _Accum()
    acc.cov_hits = {q: 0.0 for q in DEFAULT_QUANTILES}
    for ch in challenges:
        truth = np.asarray(ch.truth, dtype=float)
        meta = getattr(ch, "meta", None)
        fc = forecaster(ch.context, meta)
        mean = np.asarray(fc.mean, dtype=float)

        unseen = float(meta.get("unseen_frac", 0.0)) if isinstance(meta, dict) else 0.0
        w = UNSEEN_WEIGHT_FLOOR + (1.0 - UNSEEN_WEIGHT_FLOOR) * unseen
        acc.wsum += w
        acc.unseen_sum += unseen

        if seasonality is None:
            freq = meta.get("freq") if isinstance(meta, dict) else None
            m = season_length(freq, len(ch.context))
        else:
            m = seasonality
        acc.abs_err += w * float(np.mean(np.abs(truth - mean)))
        acc.scaled_n += w * float(np.mean(np.abs(truth - mean))) / _naive_scale(ch.context, m)

        denom = float(np.sum(np.abs(truth)))
        acc.wql_den += w * denom
        for q in DEFAULT_QUANTILES:
            acc.wql_num += w * 2.0 * float(np.sum(_pinball(truth, fc.quantiles[q], q)))
            # Calibration bookkeeping: how often the truth falls at/below level q.
            acc.cov_hits[q] += w * float(np.sum(truth <= fc.quantiles[q]))
        acc.cov_total += w * truth.size

        # Interval coverage of the nominal-80% band and the Weighted Interval Score.
        lo10, hi90 = fc.quantiles[0.1], fc.quantiles[0.9]
        acc.cov80_sum += w * float(np.mean((truth >= lo10) & (truth <= hi90)))
        wis = 0.5 * np.abs(truth - fc.quantiles[0.5])
        for (lo_q, hi_q), alpha in zip(_WIS_PAIRS, _WIS_ALPHAS, strict=True):
            wis = wis + (alpha / 2.0) * _interval_score(
                truth, fc.quantiles[lo_q], fc.quantiles[hi_q], alpha
            )
        acc.wis_sum += w * float(np.mean(wis / (len(_WIS_PAIRS) + 0.5)))

        # CRPS ~ 2 * mean pinball over a dense quantile grid. The model only emits
        # a coarse decile grid, so we interpolate ITS quantiles onto the dense grid
        # (per horizon step) -- using the model's real distribution, not a
        # synthesized band, so a perfectly-calibrated model scores ~0.
        levels = sorted(fc.quantiles)
        q_stack = np.stack([np.asarray(fc.quantiles[lv], dtype=float) for lv in levels])
        for q in _CRPS_QUANTILES:
            pq = np.array([np.interp(q, levels, q_stack[:, h]) for h in range(q_stack.shape[1])])
            acc.crps_sum += w * 2.0 * float(np.mean(_pinball(truth, pq, q)))
            acc.crps_n += w
        acc.n += 1

    n = max(acc.wsum, _EPS)
    # Probabilistic Calibration Error: mean over quantile levels of the gap
    # between nominal level q and the empirically observed coverage at q. CRPS/WQL
    # conflate calibration and sharpness (Adler et al., ICLR 2026), so PCE is
    # reported separately as a dedicated calibration axis.
    total_pts = max(acc.cov_total, _EPS)
    pce = float(
        np.mean([abs((acc.cov_hits[q] / total_pts) - q) for q in DEFAULT_QUANTILES])
    )
    return {
        "mase": acc.scaled_n / n,
        "wql": acc.wql_num / (len(DEFAULT_QUANTILES) * max(acc.wql_den, _EPS)),
        "crps": acc.crps_sum / max(acc.crps_n, _EPS),
        "wis": acc.wis_sum / n,
        "pce": pce,
        "coverage_80": acc.cov80_sum / n,
        "mae": acc.abs_err / n,
        "n": float(acc.n),
        "unseen_frac": acc.unseen_sum / max(acc.n, 1),
    }


# --------------------------------------------------------------------------- #
# Leaderboard + headroom
# --------------------------------------------------------------------------- #


def probabilistic_panel() -> dict[str, Forecaster]:
    """The reference panel as probabilistic forecasters (rungs on the leaderboard).

    Adds the **context-parroting** floor baseline (``baselines.context_parrot``):
    a submission that cannot beat trivial nearest-neighbour copying has not
    demonstrated forecasting skill, so parrot is a rung every real model must
    clear (Zhang & Gilpin, arXiv:2505.11349).
    """
    from baselines import context_parrot

    panel = {name: probabilistic(fn) for name, fn in default_panel().items()}
    panel["context_parrot"] = probabilistic(context_parrot)
    return panel


# The two mandatory floor baselines: a real submission must beat BOTH to count as
# demonstrating skill (seasonal-naive = classical floor; context_parrot =
# repetition floor).
FLOOR_BASELINES: tuple[str, ...] = ("seasonal_naive", "context_parrot")


def clears_floor(
    candidate: Forecaster,
    challenges: list,
    *,
    primary: str = "mase",
    seasonality: int | None = None,
) -> dict[str, object]:
    """Check a model beats both floor baselines (seasonal-naive AND parrot).

    Returns ``{clears, candidate_score, floors, worst_floor}``. ``clears`` is
    True only if the candidate's ``primary`` metric is strictly below *every*
    floor baseline's -- the minimum bar for a result to mean anything.
    """
    from baselines import context_parrot
    from score import seasonal_naive

    floor_fns = {"seasonal_naive": seasonal_naive, "context_parrot": context_parrot}
    cand = evaluate_forecaster(candidate, challenges, seasonality=seasonality)[primary]
    floors = {
        name: evaluate_forecaster(probabilistic(fn), challenges, seasonality=seasonality)[primary]
        for name, fn in floor_fns.items()
    }
    worst = max(floors, key=floors.get)  # the easiest floor to beat is the lowest
    clears = all(cand < f for f in floors.values())
    return {
        "clears": bool(clears),
        "candidate_score": float(cand),
        "floors": {k: float(v) for k, v in floors.items()},
        "worst_floor": worst,
    }


def _per_challenge_scores(
    forecaster: Forecaster, challenges: list, seasonality: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-challenge (mase, crps, unseen-weight) arrays for robust aggregation."""
    from config import UNSEEN_WEIGHT_FLOOR

    mases, crpss, ws = [], [], []
    for ch in challenges:
        truth = np.asarray(ch.truth, dtype=float)
        meta = getattr(ch, "meta", None)
        fc = forecaster(ch.context, meta)
        if seasonality is None:
            freq = meta.get("freq") if isinstance(meta, dict) else None
            m = season_length(freq, len(ch.context))
        else:
            m = seasonality
        mases.append(float(np.mean(np.abs(truth - np.asarray(fc.mean, dtype=float))))
                     / _naive_scale(ch.context, m))
        levels = sorted(fc.quantiles)
        q_stack = np.stack([np.asarray(fc.quantiles[lv], dtype=float) for lv in levels])
        crps = np.mean([
            2.0 * float(np.mean(_pinball(
                truth,
                np.array([np.interp(q, levels, q_stack[:, h]) for h in range(q_stack.shape[1])]),
                q,
            )))
            for q in _CRPS_QUANTILES
        ])
        crpss.append(float(crps))
        unseen = float(meta.get("unseen_frac", 0.0)) if isinstance(meta, dict) else 0.0
        ws.append(UNSEEN_WEIGHT_FLOOR + (1.0 - UNSEEN_WEIGHT_FLOOR) * unseen)
    return np.array(mases), np.array(crpss), np.array(ws)


def leaderboard(
    forecasters: dict[str, Forecaster],
    challenges: list,
    *,
    primary: str = "mase",
    seasonality: int | None = None,
) -> list[dict[str, object]]:
    """Score every named forecaster and return rows ranked by ``primary`` (asc).

    Ranking uses the **seasonal-naive-relative shifted geometric mean** of the
    primary metric (``<primary>_rel`` columns), normalised at the *source*
    level — the GIFT-Eval / BOOM per-dataset convention: challenges are grouped
    by ``source_id``, each model's unseen-weighted mean score per source is
    divided by the seasonal-naive forecaster's, and the ratios are aggregated
    by weighted shifted gmean. Per-source (not per-challenge) normalisation
    matters: on near-flat challenges seasonal-naive's error approaches zero and
    per-challenge ratios explode, inverting the ranking. The gmean is robust to
    the heavy-tailed per-challenge scores that let a handful of blow-ups set an
    arithmetic-mean rank (observed live: Chronos-Bolt won 63% of challenges
    against ewma yet ranked below it on mean MASE). The arithmetic-mean metric
    suite is still reported per row.

    Pass the models under test merged with :func:`probabilistic_panel` to see
    where a submission lands relative to the known-quality reference rungs.
    """
    from score import seasonal_naive

    naive_mase, naive_crps, w = _per_challenge_scores(
        probabilistic(seasonal_naive), challenges, seasonality
    )
    sources = np.array([
        str((getattr(ch, "meta", None) or {}).get("source_id")) for ch in challenges
    ])
    groups = {s: np.flatnonzero(sources == s) for s in dict.fromkeys(sources)}

    def _per_source_rel(model_scores: np.ndarray, naive_scores: np.ndarray) -> float:
        ratios, weights = [], []
        for idx in groups.values():
            wg = w[idx]
            naive_mean = float(np.average(naive_scores[idx], weights=wg))
            model_mean = float(np.average(model_scores[idx], weights=wg))
            ratios.append(model_mean / max(naive_mean, _EPS))
            weights.append(float(wg.sum()))
        return shifted_gmean(ratios, weights=weights)

    rows = []
    for name, fc in forecasters.items():
        # Cache forecasts by context identity so each model runs inference
        # exactly once per challenge across the two scoring passes.
        cache: dict[int, ProbForecast] = {}

        def replay(context, meta=None, _fc=fc, _cache=cache):
            key = id(context)
            if key not in _cache:
                _cache[key] = _fc(context, meta)
            return _cache[key]

        m = evaluate_forecaster(replay, challenges, seasonality=seasonality)
        ms, cs, _ = _per_challenge_scores(replay, challenges, seasonality)
        m["mase_rel"] = _per_source_rel(ms, naive_mase)
        m["crps_rel"] = _per_source_rel(cs, naive_crps)
        rows.append({"model": name, **m})
    rank_key = f"{primary}_rel" if f"{primary}_rel" in rows[0] else primary
    rows.sort(key=lambda r: r[rank_key])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def headroom(
    candidate: Forecaster,
    challenges: list,
    *,
    anchor: Forecaster | None = None,
    seasonality: int | None = None,
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
    seasonality: int | None = None,
) -> bool:
    """Go/no-go: can a model known to be superior actually beat the anchor here?

    If this is False, the benchmark cannot separate a better-than-classical model
    from the anchor and its scores do not certify TSFM quality. Run it at setup
    with a deliberately-strong probe before trusting any leaderboard.
    """
    h = headroom(known_better, challenges, seasonality=seasonality)
    return bool(h["mase_margin"] > min_margin)


# --------------------------------------------------------------------------- #
# Robust aggregation + statistical significance
# --------------------------------------------------------------------------- #


def shifted_gmean(
    values: list[float] | np.ndarray,
    shift: float = 1e-5,
    weights: list[float] | np.ndarray | None = None,
) -> float:
    """Shifted geometric mean ``exp(mean(log(x + shift))) - shift``.

    The aggregation BOOM / Toto use across heterogeneous datasets: robust to the
    near-zero, heavy-tailed, zero-inflated scores that wreck an arithmetic mean,
    while still defined when some values are exactly zero. NaNs are dropped.
    ``weights`` (e.g. the unseen-fraction challenge weights) turn it into a
    weighted geometric mean; they are aligned with ``values`` before NaN-drop.
    """
    x = np.asarray(values, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    keep = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[keep], w[keep]
    if x.size == 0:
        return float("nan")
    x = np.clip(x, 0.0, None)
    return float(np.exp(np.average(np.log(x + shift), weights=w)) - shift)


def normalized_leaderboard(
    forecasters: dict[str, Forecaster],
    challenge_sets: dict[str, list],
    *,
    metric: str = "mase",
    baseline: str = "seasonal_naive",
    seasonality: int | None = None,
) -> list[dict[str, object]]:
    """Seasonal-naive-normalised, rank-aggregated leaderboard across datasets.

    The GIFT-Eval / BOOM convention: for each dataset score every model, divide by
    the ``baseline`` model's score on that dataset (so heterogeneous scales become
    comparable), then aggregate each model across datasets two ways -- the
    **shifted geometric mean** of its normalised scores and its **average rank**.
    Rows are returned sorted by average rank (the more robust headline).

    ``challenge_sets`` maps a dataset name to its challenge list. The ``baseline``
    forecaster must be present in ``forecasters``.
    """
    from score import seasonal_naive

    fns = dict(forecasters)
    if baseline not in fns:
        fns[baseline] = probabilistic(seasonal_naive)

    # raw[dataset][model] = metric value
    raw: dict[str, dict[str, float]] = {}
    for ds, chs in challenge_sets.items():
        raw[ds] = {
            name: evaluate_forecaster(fc, chs, seasonality=seasonality)[metric]
            for name, fc in fns.items()
        }

    normed: dict[str, list[float]] = {m: [] for m in fns}
    ranks: dict[str, list[int]] = {m: [] for m in fns}
    for scores in raw.values():
        base = scores.get(baseline, float("nan"))
        order = sorted(scores, key=lambda m: scores[m])
        rank_of = {m: i + 1 for i, m in enumerate(order)}
        for m in fns:
            denom = base if (np.isfinite(base) and abs(base) > _EPS) else _EPS
            normed[m].append(scores[m] / denom)
            ranks[m].append(rank_of[m])

    rows = [
        {
            "model": m,
            f"{metric}_norm_gmean": shifted_gmean(normed[m]),
            "avg_rank": float(np.mean(ranks[m])),
            "n_datasets": len(challenge_sets),
        }
        for m in fns
    ]
    rows.sort(key=lambda r: r["avg_rank"])
    return rows


def evaluate_multiseed(
    forecaster: Forecaster,
    challenge_sets: list[list],
    *,
    seasonality: int | None = None,
) -> dict[str, object]:
    """Score a model over several independent challenge sets; report mean +/- std.

    A single sampled set carries real sampling noise, so a credible result reports
    spread across seeds (Hewamalage et al. 2023). ``challenge_sets`` is one
    challenge list per seed/origin. Returns per-metric ``{mean, std}`` plus the
    raw per-seed values.
    """
    per_metric: dict[str, list[float]] = {}
    for chs in challenge_sets:
        res = evaluate_forecaster(forecaster, chs, seasonality=seasonality)
        for k, v in res.items():
            if k == "n":
                continue
            per_metric.setdefault(k, []).append(float(v))
    return {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
        for k, v in per_metric.items()
    }


def friedman_test(score_matrix: list[list[float]] | np.ndarray) -> dict[str, float]:
    """Friedman test for differences among k models over n paired blocks.

    ``score_matrix`` is ``n_blocks x k_models`` of an error metric (lower better).
    Returns the chi-square ``statistic``, degrees of freedom ``df = k-1``, and a
    ``p_value`` from the chi-square survival function (Wilson-Hilferty
    approximation, so no scipy dependency). Use it to check whether a leaderboard's
    differences are real before reading into them.
    """
    a = np.asarray(score_matrix, dtype=float)
    n, k = a.shape
    if n < 2 or k < 2:
        return {"statistic": 0.0, "df": float(max(k - 1, 0)), "p_value": 1.0}
    # Rank within each block (average ranks for ties), lower error -> rank 1.
    ranks = np.empty_like(a)
    for i in range(n):
        order = np.argsort(a[i])
        r = np.empty(k)
        r[order] = np.arange(1, k + 1)
        # average ties
        row = a[i]
        for v in np.unique(row):
            mask = row == v
            if mask.sum() > 1:
                r[mask] = r[mask].mean()
        ranks[i] = r
    rj = ranks.sum(axis=0)
    stat = (12.0 / (n * k * (k + 1))) * float(np.sum(rj**2)) - 3.0 * n * (k + 1)
    df = k - 1
    return {"statistic": float(stat), "df": float(df), "p_value": _chi2_sf(stat, df)}


def _chi2_sf(x: float, df: int) -> float:
    """Upper-tail chi-square probability via the Wilson-Hilferty approximation."""
    if x <= 0 or df <= 0:
        return 1.0
    t = (x / df) ** (1.0 / 3.0)
    mean = 1.0 - 2.0 / (9.0 * df)
    sd = np.sqrt(2.0 / (9.0 * df))
    z = (t - mean) / sd
    # standard-normal upper tail
    return float(0.5 * math.erfc(z / np.sqrt(2.0)))
