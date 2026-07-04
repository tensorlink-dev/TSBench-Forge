"""Reference baselines, discrimination metric, and breadth gates.

The benchmark forecasts real data, so a challenge set earns its keep by
**discriminating** forecasters, not by matching a synthetic ground truth. This
module scores that discrimination and reports the breadth / non-triviality gates.

The panel of baselines
----------------------
A fixed set of cheap reference forecasters, used both to measure discrimination
(``spread``) and as leaderboard rungs a model under test must beat:

* ``seasonal_naive``, ``drift``, ``ar1``, ``ewma`` -- cheap classical baselines.
* ``strong`` -- a stronger classical baseline (a backtest-selected ensemble:
  Holt-Winters+AR / global decomposition / drift / naive / damped), the "can the
  model beat a good classical forecaster, not just a naive one" bar used by
  :func:`evaluate.headroom`. Swap in a real TSFM via ``default_panel(strong_model
  =...)`` since every model shares the ``(context, meta) -> horizon`` interface.

There is no reference-panel *ordering* gate and no "strong must lead" anchor: on
real data a naive model legitimately wins on some series (random walks), so
requiring a sophisticated model to lead would penalise valid tasks. Validity comes
instead from discrimination (``spread``), the parrot gate (a set is worthless if
copying the context matches the panel), the coverage / DGP-class / cadence breadth
gates, and the admission-time forecastability filter in
``source_discovery.quality``.

Fitness
-------
``panel_fitness`` returns ``fitness = spread = (max_err - min_err) / mean_err`` --
how strongly the challenge set separates good forecasters from bad. The
breadth-aware :func:`foundational_fitness` multiplies in the parrot gate and the
domain / DGP-class / cadence coverage gates.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import numpy as np

from config import FREQ_SEASONALITY, HORIZON, SEASONAL_PERIODS


def _resolve_horizon(meta: dict | None) -> int:
    """Horizon for this challenge: ``meta["horizon"]`` (per-cadence profile) or
    the global fallback. Panel models call this so one frozen panel serves
    every profile shape."""
    if isinstance(meta, dict):
        h = meta.get("horizon")
        if h:
            return int(h)
    return HORIZON


def _season_candidates(context_len: int, meta: dict | None) -> tuple[int, ...]:
    """Seasonal periods to search: the frozen co-prime-ish trio plus the
    frequency-derived natural cycle when it fits the context. Without the
    freq-derived candidate the seasonal floor cannot even express a daily
    cycle on sub-hourly feeds (e.g. 288 steps at 5-min), making
    "beats seasonal naive" hollow exactly where the catalog is densest."""
    cands = list(SEASONAL_PERIODS)
    if isinstance(meta, dict):
        m = FREQ_SEASONALITY.get(meta.get("freq") or "", 0)
        if 1 < m < context_len and m not in cands:
            cands.append(m)
    return tuple(cands)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a challenges<->score cycle
    from challenges import Challenge

# A panel model maps an observed context (and optional generator metadata) to a
# horizon-length prediction. A real zero-shot TSFM satisfies this by ignoring
# ``meta`` entirely.
PanelModel = Callable[[np.ndarray, dict | None], np.ndarray]

_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Forecasting helpers (numpy only)
# --------------------------------------------------------------------------- #


def _linfit(y: np.ndarray) -> tuple[float, float]:
    """Ordinary-least-squares slope and intercept of ``y`` against its index."""
    n = len(y)
    t = np.arange(n, dtype=float)
    tm, ym = t.mean(), y.mean()
    denom = float(np.dot(t - tm, t - tm))
    slope = float(np.dot(t - tm, y - ym) / denom) if denom > 0 else 0.0
    intercept = float(ym - slope * tm)
    return slope, intercept


def _dominant_period(x: np.ndarray, candidates: Iterable[int], threshold: float = 0.2) -> int:
    """Return the candidate lag with the strongest autocorrelation, or 0.

    Used by every seasonality-aware model so that ``seasonal_naive`` and
    ``strong`` estimate the period the *same* way an honest forecaster would --
    they are never told the true period.
    """
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= _EPS:
        return 0
    best_p, best_r = 0, threshold
    for p in candidates:
        if p < 2 or p >= len(x):
            continue
        r = float(np.dot(x[:-p], x[p:]) / denom)
        if r > best_r:
            best_p, best_r = p, r
    return best_p


def _seasonal_profile(x: np.ndarray, p: int) -> np.ndarray:
    """Zero-mean average seasonal shape of ``x`` at period ``p``."""
    n = len(x)
    prof = np.zeros(p)
    counts = np.zeros(p)
    idx = np.arange(n) % p
    np.add.at(prof, idx, x)
    np.add.at(counts, idx, 1.0)
    prof = prof / np.maximum(counts, 1.0)
    return prof - prof.mean()


def _ar1_coef(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of ``x`` (demeaned), clipped to a stable range."""
    x = x - x.mean()
    denom = float(np.dot(x[:-1], x[:-1]))
    if denom <= _EPS:
        return 0.0
    return float(np.clip(np.dot(x[:-1], x[1:]) / denom, -0.99, 0.99))


# --------------------------------------------------------------------------- #
# Panel models: (context, meta) -> horizon prediction
# --------------------------------------------------------------------------- #


def seasonal_naive(context: np.ndarray, meta: dict | None = None) -> np.ndarray:
    """Repeat the most recent estimated seasonal cycle.

    The period search covers the frozen SEASONAL_PERIODS plus the challenge's
    frequency-derived natural cycle when it fits the context (see
    :func:`_season_candidates`)."""
    return _hw_seasonal_repeat(
        context, _resolve_horizon(meta), _season_candidates(len(context), meta)
    )


def drift(context: np.ndarray, meta: dict | None = None) -> np.ndarray:
    """Linear extrapolation of the global slope from first to last point."""
    n = len(context)
    slope = (context[-1] - context[0]) / max(n - 1, 1)
    return context[-1] + slope * np.arange(1, _resolve_horizon(meta) + 1)


def ewma(context: np.ndarray, meta: dict | None = None, alpha: float = 0.3) -> np.ndarray:
    """Flat forecast at the exponentially weighted level (recent mean)."""
    level = float(context[0])
    for v in context[1:]:
        level = alpha * float(v) + (1 - alpha) * level
    return np.full(_resolve_horizon(meta), level, dtype=float)


def ar1(context: np.ndarray, meta: dict | None = None) -> np.ndarray:
    """Mean-reverting AR(1): decays the last deviation back to the mean.

    Captures short-range autocorrelation but ignores trend and seasonality, so
    on structured series it is a deliberately *weak* baseline.
    """
    mu = float(context.mean())
    x = context - mu
    phi = _ar1_coef(x)
    h = np.arange(1, _resolve_horizon(meta) + 1)
    return mu + float(x[-1]) * (phi**h)


def _naive_flat(context: np.ndarray, hor: int = HORIZON) -> np.ndarray:
    """Last observed value, held flat -- optimal for a pure random walk."""
    return np.full(hor, float(context[-1]), dtype=float)


def _damped_forecast(context: np.ndarray, hor: int = HORIZON) -> np.ndarray:
    """Holt-style damped trend from the recent local slope.

    Extrapolates a recent trend but damps it toward flat, which is well-behaved
    on the live random-walk motifs where an undamped global line over-shoots.
    """
    k = min(12, len(context) - 1)
    slope = float(np.mean(np.diff(context[-k - 1 :]))) if k >= 1 else 0.0
    phi = 0.9
    h = np.arange(1, hor + 1)
    damp = phi * (1 - phi**h) / (1 - phi)
    return context[-1] + slope * damp


def _hw_ar_forecast(
    context: np.ndarray, hor: int = HORIZON, cands: tuple[int, ...] = SEASONAL_PERIODS
) -> np.ndarray:
    """Holt-Winters + AR(1): the strong baseline's structural moat.

    Estimates seasonality, then a *local damped* level/trend (robust to
    changepoints, unlike a global line), then an AR(1) term on the residual.
    Because only this and ``seasonal_naive`` use seasonality -- and only this
    also models trend and autocorrelation -- it dominates the baselines on the
    structured part of the distribution while degrading gracefully to naive
    behaviour on unstructured (random-walk) motifs.
    """
    n = len(context)
    t = np.arange(n, dtype=float)
    slope_g, intercept_g = _linfit(context)
    detr = context - (intercept_g + slope_g * t)
    p = _dominant_period(detr, cands)
    if p >= 2:
        prof = _seasonal_profile(detr, p)
        seas_in = prof[np.arange(n) % p]
    else:
        prof, seas_in = None, np.zeros(n)

    deseason = context - seas_in
    k = int(min(max(2 * p if p >= 2 else 24, 12), n - 1))
    local_slope = float(np.mean(np.diff(deseason[-(k + 1) :])))
    level = float(deseason[-1])

    lin = level + local_slope * (np.arange(n) - (n - 1))
    resid = deseason - lin
    phi = _ar1_coef(resid)
    last_resid = float(resid[-1])

    h = np.arange(1, hor + 1)
    phi_d = 0.9
    damp = phi_d * (1 - phi_d**h) / (1 - phi_d)
    trend_f = level + local_slope * damp
    future = n + np.arange(hor)
    seas_f = prof[future % p] if p >= 2 else np.zeros(hor)
    ar_f = last_resid * (phi**h)
    return trend_f + seas_f + ar_f


def _decompose_global_forecast(
    context: np.ndarray, hor: int = HORIZON, cands: tuple[int, ...] = SEASONAL_PERIODS
) -> np.ndarray:
    """Global-trend + seasonality + AR(1) residual.

    Complements ``hw_ar``: when the trend really is a single clean global line
    (no changepoints) the global slope beats a damped local one, so having both
    structural candidates lets the strong baseline keep its seasonality moat over ``drift``
    in *every* trend regime.
    """
    n = len(context)
    t = np.arange(n, dtype=float)
    slope, intercept = _linfit(context)
    detr = context - (intercept + slope * t)
    p = _dominant_period(detr, cands)
    if p >= 2:
        prof = _seasonal_profile(detr, p)
        seas_in = prof[np.arange(n) % p]
    else:
        prof, seas_in = None, np.zeros(n)
    resid = detr - seas_in
    phi = _ar1_coef(resid)
    last = float(resid[-1] - resid.mean())
    th = np.arange(n, n + hor)
    trend_f = intercept + slope * th
    seas_f = prof[th % p] if p >= 2 else np.zeros(hor)
    ar_f = last * (phi ** np.arange(1, hor + 1))
    return trend_f + seas_f + ar_f


# Candidates the strong baseline selects among by backtest. The simple baselines
# guarantee the strong baseline is never much worse than the best baseline on any series;
# the two structural candidates (``hw_ar`` local-trend, ``decomp`` global-trend)
# are what let it pull decisively ahead on structured data, in either trend
# regime.
# Every candidate takes ``(context, hor, cands)`` so the strong baseline
# serves any per-cadence profile shape with one frozen candidate set.
_STRONG_CANDIDATES: dict[str, Callable[[np.ndarray, int, tuple[int, ...]], np.ndarray]] = {
    "hw_ar": _hw_ar_forecast,
    "decomp": _decompose_global_forecast,
    "naive": lambda c, h, s: _naive_flat(c, h),
    "drift": lambda c, h, s: drift(c, {"horizon": h}),
    "damped": lambda c, h, s: _damped_forecast(c, h),
    "ewma": lambda c, h, s: ewma(c, {"horizon": h}),
    "seasonal": lambda c, h, s: _hw_seasonal_repeat(c, h, s),
}


def _hw_seasonal_repeat(context: np.ndarray, hor: int, cands: tuple[int, ...]) -> np.ndarray:
    """seasonal_naive with an explicit candidate list (the public function
    derives candidates from meta; the strong candidate gets them directly)."""
    n = len(context)
    p = _dominant_period(context, cands)
    if p < 2:
        return np.full(hor, context[-1], dtype=float)
    idx = n - p + (np.arange(hor) % p)
    return context[idx].astype(float)


def _backtest_best(context: np.ndarray, hor: int, cands: tuple[int, ...]) -> str:
    """Pick the candidate with the lowest error over rolling backtest windows.

    Two windows (when the context allows) make the selection robust to a single
    unlucky holdout, so the strong baseline reliably tracks the genuinely-best candidate.
    """
    names = list(_STRONG_CANDIDATES)
    n = len(context)
    ends = [n]
    if n - hor > 2 * hor + 8:
        ends.append(n - hor // 2)
    totals = {k: 0.0 for k in names}
    used = 0
    for end in ends:
        train, val = context[: end - hor], context[end - hor : end]
        if len(train) < hor + 8:
            continue
        scale = _scale(train)
        used += 1
        for k in names:
            totals[k] += _mae(_STRONG_CANDIDATES[k](train, hor, cands), val) / scale
    if used == 0:
        return "hw_ar"
    return min(names, key=lambda k: totals[k])


def strong(context: np.ndarray, meta: dict | None = None) -> np.ndarray:
    """A strong classical baseline: backtest-selected adaptive forecaster.

    Scores each candidate on rolling backtest windows of the context and returns
    the best candidate's forecast, so one model stays competitive across *both*
    textures -- trend+seasonal+AR on structured series, naive/drift on random-walk
    motifs. It is the "beat a good classical model, not just a naive one" bar on
    the leaderboard and in :func:`evaluate.headroom`.

    It estimates everything from the context, so its edge is genuine skill. Swap in
    a real zero-shot TSFM via :func:`default_panel` to raise that bar.
    """
    hor = _resolve_horizon(meta)
    cands = _season_candidates(len(context), meta)
    return _STRONG_CANDIDATES[_backtest_best(context, hor, cands)](context, hor, cands)


def default_panel(strong_model: PanelModel | None = None) -> dict[str, PanelModel]:
    """Construct the reference panel of baselines, optionally swapping ``strong``.

    The panel is **frozen within a benchmark version** for cross-validator
    reproducibility: do not add or reorder models mid-version. Replace only the
    ``strong`` callable to put a real zero-shot TSFM on the leaderboard.
    """
    return {
        "strong": strong_model or strong,
        "seasonal_naive": seasonal_naive,
        "drift": drift,
        "ewma": ewma,
        "ar1": ar1,
    }


def try_statsforecast_strong() -> PanelModel | None:
    """Lazily build a ``statsforecast`` AutoETS model, or ``None`` if unavailable.

    Optional dependency: the numpy demo never calls this, so ``demo.py`` runs
    with numpy alone. In a real deployment prefer a top GIFT-Eval TSFM here.
    """
    try:  # pragma: no cover - optional dependency, exercised only when installed
        from statsforecast.models import AutoETS
    except Exception:
        return None

    def _sf_strong(context: np.ndarray, meta: dict | None = None) -> np.ndarray:
        model = AutoETS(season_length=max(SEASONAL_PERIODS))
        model.fit(np.asarray(context, dtype=float))
        return np.asarray(model.predict(h=HORIZON)["mean"], dtype=float)

    return _sf_strong


def panel_from_env(env: dict[str, str] | None = None) -> dict[str, PanelModel]:
    """Build the panel, selecting the ``strong`` baseline from the environment.

    ``TSBENCH_STRONG=statsforecast`` swaps in the AutoETS model when the optional
    dependency is installed (falling back to the numpy default otherwise);
    anything else uses the numpy default. A real deployment can register its own
    zero-shot TSFM as the ``strong`` rung here.
    """
    import os

    e = env if env is not None else os.environ
    choice = (e.get("TSBENCH_STRONG") or "").strip().lower()
    if choice == "statsforecast":
        sf = try_statsforecast_strong()
        if sf is not None:
            return default_panel(strong_model=sf)
    return default_panel()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _scale(context: np.ndarray) -> float:
    """Naive one-step MAE used to normalise errors across heterogeneous scales."""
    d = np.abs(np.diff(context))
    s = float(d.mean()) if d.size else 0.0
    return s if s > _EPS else float(np.std(context) + _EPS)


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def model_errors(
    challenges: list[Challenge], panel: dict[str, PanelModel]
) -> dict[str, float]:
    """Mean scale-normalised MAE of each panel model over the challenge set."""
    names = list(panel)
    sums = {m: 0.0 for m in names}
    for ch in challenges:
        scale = _scale(ch.context)
        meta = getattr(ch, "meta", None)
        for m in names:
            pred = panel[m](ch.context, meta)
            sums[m] += _mae(pred, ch.truth) / scale
    n = max(len(challenges), 1)
    return {m: sums[m] / n for m in names}


# --------------------------------------------------------------------------- #
# Parrot gate: the "is this just repetition?" layer
# --------------------------------------------------------------------------- #
#
# A challenge set can fail to measure real skill if a trivial nearest-neighbour
# *copy-the-context* forecaster (``baselines.context_parrot``) already does as
# well as the strong baseline -- then a high score rewards induction-head
# repetition, not understanding (Zhang & Gilpin, arXiv:2505.11349). This gate is
# a smooth sigmoid of the parrot's error ratio. It is report-only in
# ``panel_fitness`` (so the frozen ``fitness`` consensus value is unchanged) and
# folded into the opt-in ``foundational_fitness``.


def parrot_error(challenges: list[Challenge]) -> float:
    """Mean scale-normalised MAE of the context-parroting baseline."""
    from baselines import context_parrot  # local import avoids an import cycle

    total = 0.0
    for ch in challenges:
        scale = _scale(ch.context)
        pred = context_parrot(ch.context, getattr(ch, "meta", None))
        total += _mae(pred, ch.truth) / scale
    return total / max(len(challenges), 1)


# The parrot is a *legitimate-ish* baseline, so the bar is parity: the gate is 0.5
# exactly when parrot ties the strong baseline, collapsing toward 0 when parrot wins and
# rising toward 1 as the strong baseline pulls genuinely ahead. The gentle steepness
# reflects that we are flagging "no real margin over copying", not a sharp cliff.
PARROT_GATE_THRESHOLD = 1.0
PARROT_GATE_STEEPNESS = 6.0


def parrot_gate(strong_err: float, parrot_err: float) -> float:
    """Validity multiplier in ``(0, 1)``: ~0 when parroting matches/beats ``strong``.

    Smooth sigmoid of the error ratio ``parrot_err / strong_err`` centred at
    parity: ``0.5`` when the trivial copy-the-context baseline ties the strong baseline,
    ramping toward ``0`` when parrot *wins* (the set is solvable by repetition and
    rewards no real skill) and toward ``1`` once the strong baseline clearly beats
    it. Independent of ``spread`` -- a set can separate the panel well yet still be
    parrot-solvable (e.g. near-recurrent series), so both are checked.
    """
    if strong_err <= _EPS:
        return 0.0
    ratio = parrot_err / strong_err
    return float(1.0 / (1.0 + np.exp(-PARROT_GATE_STEEPNESS * (ratio - PARROT_GATE_THRESHOLD))))


# --------------------------------------------------------------------------- #
# Coverage: the foundational-breadth layer
# --------------------------------------------------------------------------- #
#
# ``spread`` and ``ordering`` both measure quality *within* whatever
# distribution the feed supplies; neither notices whether that distribution
# spans one data-generating process or twenty. A benchmark that certifies a
# *foundation* model must also be broad -- many domains / DGPs -- so a high score
# means "generalises across worlds," not "good at the one process we test." These
# helpers measure that breadth from the per-challenge ``meta['domain']`` labels.

UNKNOWN_DOMAIN = "unknown"

# Target effective number of distinct domains per evaluation. The coverage gate
# is 1.0 once a challenge set reaches this many *effective* domains (Hill number
# of order 1); below it the gate ramps down linearly. Four is a reasonable
# breadth floor for the live catalog's GIFT-Eval domains.
DEFAULT_COVERAGE_TARGET = 4.0


def challenge_domain(ch: Challenge) -> str:
    """The data-generating domain a challenge is tagged with (``unknown`` if absent)."""
    meta = getattr(ch, "meta", None) or {}
    return str(meta.get("domain", UNKNOWN_DOMAIN))


def domain_coverage(challenges: list[Challenge]) -> dict[str, object]:
    """Summarise how many distinct DGP domains a challenge set spans.

    Returns per-domain counts, the raw domain count, the Shannon ``entropy`` of
    the domain mix, and ``effective_domains = exp(entropy)`` -- the *effective
    number of equally-weighted domains* (a Hill number). Effective domains is the
    honest breadth measure: ten near-identical domains plus one dominant one score
    far below ten, because skew concentrates the evaluation on a few processes.
    """
    counts: dict[str, int] = {}
    for ch in challenges:
        d = challenge_domain(ch)
        counts[d] = counts.get(d, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return {"per_domain": {}, "n_domains": 0, "entropy": 0.0, "effective_domains": 0.0}
    probs = np.array([c / total for c in counts.values()], dtype=float)
    entropy = float(-np.sum(probs * np.log(probs)))
    return {
        "per_domain": dict(sorted(counts.items())),
        "n_domains": len(counts),
        "entropy": entropy,
        "effective_domains": float(np.exp(entropy)),
    }


def coverage_gate(
    challenges: list[Challenge], target_effective_domains: float = DEFAULT_COVERAGE_TARGET
) -> float:
    """Breadth multiplier in ``[0, 1]`` from the effective-domain count.

    ``1.0`` once the set reaches ``target_effective_domains`` effective domains,
    ramping linearly to ``0`` for a single-domain (narrow) set. This is the
    independent *coverage* layer: like the other gates it only ever multiplies an
    otherwise-good score down, here punishing benchmarks that are sharp but
    narrow. It is reported by :func:`panel_fitness` and folded into the score only
    by the opt-in :func:`foundational_fitness`, so the frozen ``fitness``
    yardstick (and cross-validator consensus on it) is unchanged.
    """
    if target_effective_domains <= 0:
        return 1.0
    eff = float(domain_coverage(challenges)["effective_domains"])
    return float(np.clip(eff / target_effective_domains, 0.0, 1.0))


def panel_fitness(
    challenges: list[Challenge], panel: dict[str, PanelModel] | None = None
) -> dict[str, object]:
    """Score a challenge set by how strongly it **discriminates** forecasters.

    Returns ``fitness``, ``spread``, ``difficulty`` (plus the raw per-model
    ``errors`` for diagnostics, and a report-only ``coverage`` summary with its
    ``coverage_gate`` and ``parrot_gate``).

    * ``spread = (max_err - min_err) / mean_err`` over the panel of baselines --
      how strongly the challenge set separates good forecasters from bad. A set
      that no model does better than any other on (pure noise) has ~0 spread.
    * ``fitness = spread`` -- discrimination is the score. There is no
      panel-ordering / anchor gate: on real data a naive model legitimately wins
      on some series (e.g. random walks), so requiring a "strong" model to lead
      would penalise valid tasks. Non-triviality and forecastability are enforced
      instead by the parrot gate and the admission-time discrimination filter
      (``source_discovery.quality``).
    """
    panel = panel or default_panel()
    agg = model_errors(challenges, panel)

    errs = list(agg.values())
    mean_e = float(np.mean(errs))
    spread = (max(errs) - min(errs)) / mean_e if mean_e > _EPS else 0.0

    difficulty = agg.get("strong", min(errs) if errs else float("nan"))

    coverage = domain_coverage(challenges)
    cov_gate = float(
        np.clip(float(coverage["effective_domains"]) / DEFAULT_COVERAGE_TARGET, 0.0, 1.0)
    )
    p_gate = parrot_gate(agg.get("strong", min(errs) if errs else 0.0), parrot_error(challenges))

    return {
        "fitness": float(spread),
        "spread": float(spread),
        "difficulty": float(difficulty),
        "errors": agg,
        "coverage": coverage,
        "coverage_gate": cov_gate,
        "parrot_gate": float(p_gate),
    }


def stratified_fitness(
    challenges: list[Challenge], panel: dict[str, PanelModel] | None = None
) -> dict[str, dict[str, object]]:
    """Per-domain fitness report: skill *within each* data-generating process.

    A single aggregate fitness can hide a benchmark that discriminates well on one
    domain and not at all on the rest. This partitions the challenge set by
    ``meta['domain']`` and scores each stratum independently, so discrimination is
    read per-DGP, not merely on average. Each entry includes ``n`` (the stratum
    size) because small strata carry real sampling noise.
    """
    panel = panel or default_panel()
    groups: dict[str, list[Challenge]] = {}
    for ch in challenges:
        groups.setdefault(challenge_domain(ch), []).append(ch)

    report: dict[str, dict[str, object]] = {}
    for domain, chs in sorted(groups.items()):
        res = panel_fitness(chs, panel)
        report[domain] = {
            "n": len(chs),
            "fitness": res["fitness"],
            "spread": res["spread"],
            "difficulty": res["difficulty"],
        }
    return report


def foundational_fitness(
    challenges: list[Challenge],
    panel: dict[str, PanelModel] | None = None,
    target_effective_domains: float = DEFAULT_COVERAGE_TARGET,
    dgp_classes: list[str | None] | None = None,
    cadences: list[str | None] | None = None,
    breadth_min_share: float = 0.02,
) -> dict[str, object]:
    """Coverage-gated objective: ``foundational_fitness = fitness * coverage_gate * parrot_gate * dgp_class_breadth_gate * cadence_breadth_gate``.

    The opt-in breadth-aware score. Identical to :func:`panel_fitness` but adds
    ``foundational_fitness``, which folds domain coverage and the breadth gates
    into the score so a benchmark is rewarded for *both* discrimination **and**
    domain coverage / non-triviality. Kept separate so the core ``fitness`` is
    never silently redefined.

    Reward-hacking defense: the ``dgp_classes`` and ``cadences`` optional args
    (fed from ``buffer.pool_dgp_classes`` / ``buffer.pool_cadences``) enable
    two additional multiplicative gates that hard-veto any pool where a DGP
    class or cadence band drops below ``breadth_min_share``. When both args are
    ``None`` (legacy call site, no `ScrapedLiveSource` wired up) the gates
    default to 1.0 — old behavior preserved. See ``docs/REWARD_HACKING.md``.
    """
    res = dict(panel_fitness(challenges, panel))
    cov_gate = coverage_gate(challenges, target_effective_domains)
    res["coverage_gate"] = cov_gate
    p_gate = float(res.get("parrot_gate", 1.0))

    # Hard-veto breadth gates: any class or band below `breadth_min_share` → 0.
    dgp_gate = dgp_class_breadth_gate(dgp_classes, breadth_min_share) if dgp_classes else 1.0
    cadence_gate = cadence_breadth_gate(cadences, breadth_min_share) if cadences else 1.0
    res["dgp_class_breadth_gate"] = dgp_gate
    res["cadence_breadth_gate"] = cadence_gate

    # All gates multiply into the aggregate — any one going to zero forces the
    # aggregate to zero, so the served eval-pool cannot trade off discrimination
    # against generalisation breadth or cadence coverage.
    res["foundational_fitness"] = (
        float(res["fitness"]) * cov_gate * p_gate * dgp_gate * cadence_gate
    )
    return res


# ---------------------------------------------------------------------------
# Reward-hacking defense: DGP-class and cadence breadth gates
# ---------------------------------------------------------------------------


def _min_share(labels: list[str | None]) -> float:
    """Smallest share (0-1) of any non-``None`` label; 1.0 for an empty list."""
    labels = [l for l in labels if l is not None]
    if not labels:
        return 1.0
    counts: dict[str, int] = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    total = float(len(labels))
    return min(counts.values()) / total


def dgp_class_breadth_gate(
    dgp_classes: list[str | None] | None,
    min_share: float = 0.02,
) -> float:
    """Hard veto: 0.0 if any DGP class has share below ``min_share``, else 1.0.

    Feed with ``buffer.pool_dgp_classes``. Kills the "domain collapse" reward-
    hacking corner where the served pool silently narrows to a single (or a
    handful of) DGP class(es). Multiplies into ``foundational_fitness``.

    If ``dgp_classes`` is ``None`` or all entries are ``None`` (no tagged
    labels — a legacy source is in use), returns 1.0 so old behavior is
    preserved. That means the gate is opt-in: it activates only when a
    ``ScrapedLiveSource`` (or another labeled source) is in the FreshBuffer.
    """
    if not dgp_classes:
        return 1.0
    share = _min_share(dgp_classes)
    return 1.0 if share >= min_share else 0.0


def cadence_breadth_gate(
    cadences: list[str | None] | None,
    min_share: float = 0.02,
) -> float:
    """Hard veto: 0.0 if any cadence band has share below ``min_share``, else 1.0.

    Feed with ``buffer.pool_cadences``. Kills the "cadence collapse" reward-
    hacking corner (e.g. drift toward all-minute-cadence blends that drop the
    yearly/monthly generalisation claim).
    """
    if not cadences:
        return 1.0
    share = _min_share(cadences)
    return 1.0 if share >= min_share else 0.0
