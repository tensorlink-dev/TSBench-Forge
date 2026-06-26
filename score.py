"""Reference panel, validity gate, and fitness metrics.

This is the foundation of the whole benchmark. **Validity does not come from
trusting the data; it comes from a frozen reference panel.** A challenge only
"counts" if a fixed panel of independently-known-quality forecasters ranks in
its expected order. If a naive baseline beats the ``strong`` anchor, the
challenge is measuring an artifact rather than forecasting skill, and the forge
that produced it is penalised.

The panel
---------
* ``seasonal_naive``, ``drift``, ``ar1``, ``ewma`` -- cheap classical baselines.
* ``strong`` -- the **validity anchor**. Its quality must be established
  *independently of this benchmark* (e.g. a top, non-leaking GIFT-Eval model run
  zero-shot, or a strong classical ensemble). The numpy default is a
  backtest-selected classical ensemble (Holt-Winters+AR / global decomposition /
  drift / naive / damped), which adapts per series so it stays best across both
  synthetic and live textures; a real TSFM can be dropped in unchanged because
  every model shares the ``(context, meta) -> horizon`` interface.
* ``overfit`` -- a **gaming detector**, not a real forecaster. It is handed the
  synthetic generator's own continuation (including its deterministic
  "fingerprint"), so it is near-oracle on pure synthetic data but badly wrong on
  live motifs. It therefore beats the anchor exactly when the generator leans too
  hard on pre-fittable synthetic structure, and that is caught two ways: it sits
  last in ``PANEL_QUALITY_ORDER`` (so its rise lowers ``ordering``) and it drives
  an explicit ``gate`` that multiplies the fitness toward zero.

Fitness
-------
``panel_fitness`` returns ``spread``, ``ordering``, ``difficulty``, ``gate`` and
the doubly-gated ``fitness = spread * max(0, ordering) * gate``:

* ``spread`` measures how *discriminating* the challenge set is (computed over
  the legitimate models only, so a blown-up detector can't inflate it).
* ``ordering`` is the Kendall-tau agreement between the achieved ranking and the
  established quality order -- the panel-validity gate. If ``strong`` is beaten by
  a naive baseline, ``ordering`` goes negative and ``max(0, ordering)`` zeroes it.
* ``gate`` is the generator-fitting gate: if the ``overfit`` detector matches or
  beats the anchor, the data is pre-fittable and ``gate -> 0``.

The two gates are independent layers -- one catches *invalid* challenge sets, the
other catches *pre-fittable* ones -- so the forge can only score by producing
challenges that are valid, non-fittable, and discriminating all at once.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import numpy as np

from config import HORIZON, PANEL_QUALITY_ORDER, SEASONAL_PERIODS

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a generate<->score cycle
    from generate import Challenge

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
    """Repeat the most recent estimated seasonal cycle."""
    n = len(context)
    p = _dominant_period(context, SEASONAL_PERIODS)
    if p < 2:
        return np.full(HORIZON, context[-1], dtype=float)
    idx = n - p + (np.arange(HORIZON) % p)
    return context[idx].astype(float)


def drift(context: np.ndarray, meta: dict | None = None) -> np.ndarray:
    """Linear extrapolation of the global slope from first to last point."""
    n = len(context)
    slope = (context[-1] - context[0]) / max(n - 1, 1)
    return context[-1] + slope * np.arange(1, HORIZON + 1)


def ewma(context: np.ndarray, meta: dict | None = None, alpha: float = 0.3) -> np.ndarray:
    """Flat forecast at the exponentially weighted level (recent mean)."""
    level = float(context[0])
    for v in context[1:]:
        level = alpha * float(v) + (1 - alpha) * level
    return np.full(HORIZON, level, dtype=float)


def ar1(context: np.ndarray, meta: dict | None = None) -> np.ndarray:
    """Mean-reverting AR(1): decays the last deviation back to the mean.

    Captures short-range autocorrelation but ignores trend and seasonality, so
    on structured series it is a deliberately *weak* baseline.
    """
    mu = float(context.mean())
    x = context - mu
    phi = _ar1_coef(x)
    h = np.arange(1, HORIZON + 1)
    return mu + float(x[-1]) * (phi**h)


def _naive_flat(context: np.ndarray) -> np.ndarray:
    """Last observed value, held flat -- optimal for a pure random walk."""
    return np.full(HORIZON, float(context[-1]), dtype=float)


def _damped_forecast(context: np.ndarray) -> np.ndarray:
    """Holt-style damped trend from the recent local slope.

    Extrapolates a recent trend but damps it toward flat, which is well-behaved
    on the live random-walk motifs where an undamped global line over-shoots.
    """
    k = min(12, len(context) - 1)
    slope = float(np.mean(np.diff(context[-k - 1 :]))) if k >= 1 else 0.0
    phi = 0.9
    h = np.arange(1, HORIZON + 1)
    damp = phi * (1 - phi**h) / (1 - phi)
    return context[-1] + slope * damp


def _hw_ar_forecast(context: np.ndarray) -> np.ndarray:
    """Holt-Winters + AR(1): the anchor's structural moat.

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
    p = _dominant_period(detr, SEASONAL_PERIODS)
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

    h = np.arange(1, HORIZON + 1)
    phi_d = 0.9
    damp = phi_d * (1 - phi_d**h) / (1 - phi_d)
    trend_f = level + local_slope * damp
    future = n + np.arange(HORIZON)
    seas_f = prof[future % p] if p >= 2 else np.zeros(HORIZON)
    ar_f = last_resid * (phi**h)
    return trend_f + seas_f + ar_f


def _decompose_global_forecast(context: np.ndarray) -> np.ndarray:
    """Global-trend + seasonality + AR(1) residual.

    Complements ``hw_ar``: when the trend really is a single clean global line
    (no changepoints) the global slope beats a damped local one, so having both
    structural candidates lets the anchor keep its seasonality moat over ``drift``
    in *every* trend regime.
    """
    n = len(context)
    t = np.arange(n, dtype=float)
    slope, intercept = _linfit(context)
    detr = context - (intercept + slope * t)
    p = _dominant_period(detr, SEASONAL_PERIODS)
    if p >= 2:
        prof = _seasonal_profile(detr, p)
        seas_in = prof[np.arange(n) % p]
    else:
        prof, seas_in = None, np.zeros(n)
    resid = detr - seas_in
    phi = _ar1_coef(resid)
    last = float(resid[-1] - resid.mean())
    th = np.arange(n, n + HORIZON)
    trend_f = intercept + slope * th
    seas_f = prof[th % p] if p >= 2 else np.zeros(HORIZON)
    ar_f = last * (phi ** np.arange(1, HORIZON + 1))
    return trend_f + seas_f + ar_f


# Candidates the strong anchor selects among by backtest. The simple baselines
# guarantee the anchor is never much worse than the best baseline on any series;
# the two structural candidates (``hw_ar`` local-trend, ``decomp`` global-trend)
# are what let it pull decisively ahead on structured data, in either trend
# regime.
_STRONG_CANDIDATES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "hw_ar": _hw_ar_forecast,
    "decomp": _decompose_global_forecast,
    "naive": _naive_flat,
    "drift": lambda c: drift(c),
    "damped": _damped_forecast,
    "ewma": lambda c: ewma(c),
    "seasonal": lambda c: seasonal_naive(c),
}


def _backtest_best(context: np.ndarray) -> str:
    """Pick the candidate with the lowest error over rolling backtest windows.

    Two windows (when the context allows) make the selection robust to a single
    unlucky holdout, so the anchor reliably tracks the genuinely-best candidate.
    """
    names = list(_STRONG_CANDIDATES)
    n = len(context)
    ends = [n]
    if n - HORIZON > 2 * HORIZON + 8:
        ends.append(n - HORIZON // 2)
    totals = {k: 0.0 for k in names}
    used = 0
    for end in ends:
        train, val = context[: end - HORIZON], context[end - HORIZON : end]
        if len(train) < HORIZON + 8:
            continue
        scale = _scale(train)
        used += 1
        for k in names:
            totals[k] += _mae(_STRONG_CANDIDATES[k](train), val) / scale
    if used == 0:
        return "hw_ar"
    return min(names, key=lambda k: totals[k])


def strong(context: np.ndarray, meta: dict | None = None) -> np.ndarray:
    """Default **validity anchor**: backtest-selected adaptive forecaster.

    The anchor scores each candidate on rolling backtest windows of the context
    and returns the best candidate's forecast. This lets one model be genuinely
    best across *both* textures -- trend+seasonal+AR on structured synthetic
    series, naive/drift on random-walk live motifs -- which is what keeps the
    validity gate meaningful as the blend shifts toward live data.

    It estimates everything from the context (never the generator's parameters),
    so its edge is genuine skill. Replace it with a real, independently-validated
    zero-shot TSFM via :func:`default_panel` to raise the validity bar further.
    """
    return _STRONG_CANDIDATES[_backtest_best(context)](context)


def overfit(context: np.ndarray, meta: dict | None = None) -> np.ndarray:
    """Generator-fitting *detector* (NOT a legitimate forecaster).

    When the challenge carries the synthetic generator's own noise-free
    continuation (``meta['oracle']``), this model returns it verbatim -- i.e. it
    behaves like a miner who has perfectly reverse-engineered the synthetic
    process. That makes it near-oracle on pure-synthetic challenges but useless
    on spliced/live ones, where no such fingerprint exists and it falls back to a
    (poor) stationary mean-reversion prior.

    Because it sits *last* in ``PANEL_QUALITY_ORDER``, any epoch where it climbs
    the ranking -- i.e. where the generator leans too hard on pre-fittable
    synthetic structure -- collapses the ``ordering`` term and the fitness.
    """
    if meta is not None and meta.get("oracle") is not None:
        return np.asarray(meta["oracle"], dtype=float)
    return np.full(HORIZON, float(context.mean()), dtype=float)


def default_panel(strong_model: PanelModel | None = None) -> dict[str, PanelModel]:
    """Construct the reference panel, optionally swapping in a real ``strong``.

    The panel is **frozen within a benchmark version** for consensus: do not add
    or reorder models mid-version. To raise the validity bar, replace only the
    ``strong`` callable with an independently-validated zero-shot TSFM.
    """
    return {
        "strong": strong_model or strong,
        "seasonal_naive": seasonal_naive,
        "drift": drift,
        "ewma": ewma,
        "ar1": ar1,
        "overfit": overfit,
    }


def try_statsforecast_strong() -> PanelModel | None:
    """Lazily build a ``statsforecast`` AutoETS anchor, or ``None`` if unavailable.

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
    """Build the panel, selecting the ``strong`` anchor from the environment.

    ``TSBENCH_STRONG=statsforecast`` swaps in the AutoETS anchor when the optional
    dependency is installed (falling back to the numpy anchor otherwise);
    anything else uses the numpy default. A real deployment registers its own
    independently-validated TSFM here. Whatever is chosen, callers should gate it
    through :func:`validate_panel` before trusting a run.
    """
    import os

    e = env if env is not None else os.environ
    choice = (e.get("TSBENCH_STRONG") or "").strip().lower()
    if choice == "statsforecast":
        sf = try_statsforecast_strong()
        if sf is not None:
            return default_panel(strong_model=sf)
    return default_panel()


class AnchorValidationError(RuntimeError):
    """Raised when the ``strong`` anchor is not good enough to be a validity anchor."""


def validate_panel(
    challenges: list[Challenge],
    panel: dict[str, PanelModel] | None = None,
    *,
    min_lead: float = 0.02,
    require: bool = False,
) -> dict[str, object]:
    """Check the ``strong`` anchor is genuinely the best *legitimate* forecaster.

    The whole benchmark rests on ``strong`` being independently good: if a naive
    baseline matches or beats it on a calibration set, the validity gate is hollow
    and every downstream score is suspect. This runs the panel on ``challenges``
    and verifies ``strong`` leads every other legitimate model (excluding the
    ``overfit`` detector) by at least ``min_lead`` in scale-normalised error.

    Returns a report ``{valid, strong_error, runner_up, margin, errors}``. With
    ``require=True`` an invalid anchor raises :class:`AnchorValidationError`
    instead -- the recommended posture for a production validator at startup.
    """
    panel = panel or default_panel()
    errs = model_errors(challenges, panel)
    strong_err = errs.get("strong", float("inf"))
    competitors = {m: e for m, e in errs.items() if m not in ("strong", "overfit")}
    runner_up = min(competitors, key=competitors.get) if competitors else None
    runner_err = competitors[runner_up] if runner_up is not None else float("inf")
    margin = runner_err - strong_err
    valid = bool(runner_up is not None and margin >= min_lead)

    report: dict[str, object] = {
        "valid": valid,
        "strong_error": float(strong_err),
        "runner_up": runner_up,
        "runner_up_error": float(runner_err),
        "margin": float(margin),
        "errors": errs,
    }
    if require and not valid:
        raise AnchorValidationError(
            f"strong anchor not validated: leads {runner_up} by {margin:.4f} "
            f"(< required {min_lead}); the validity gate would be hollow"
        )
    return report


def validate_generalization(
    challenges: list[Challenge],
    heldout_models: dict[str, PanelModel],
    panel: dict[str, PanelModel] | None = None,
    *,
    min_lead: float = 0.0,
) -> dict[str, object]:
    """Check the anchor's lead **generalises to models the forge never optimised against**.

    The forge climbs by making the *frozen* reference panel rank in its expected
    order, which risks overfitting the challenge distribution to that specific
    panel: challenges that happen to order these six models correctly without
    rewarding genuine skill. The defence is a held-out set -- extra forecasters
    (a real TSFM, other classical baselines, the parrot) that were not part of the
    forge's objective. If ``strong`` still beats every held-out model on the same
    challenges, the lead reflects real skill, not panel-overfitting.

    Returns ``{generalizes, strong_error, beaten_by, errors}``. ``beaten_by``
    lists any held-out model that matches/beats ``strong`` within ``min_lead``.
    """
    panel = panel or default_panel()
    merged = {"strong": panel["strong"]}
    merged.update({k: v for k, v in heldout_models.items() if k != "strong"})
    errs = model_errors(challenges, merged)
    strong_err = errs["strong"]
    beaten_by = [m for m, e in errs.items() if m != "strong" and e <= strong_err + min_lead]
    return {
        "generalizes": len(beaten_by) == 0,
        "strong_error": float(strong_err),
        "beaten_by": beaten_by,
        "errors": errs,
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    """Kendall's tau between two orderings of the *same* set of names.

    ``+1`` identical, ``-1`` reversed, ``0`` uncorrelated. This is the validity
    gate's core: ``order_a`` = achieved (best->worst by error), ``order_b`` =
    the established quality order.
    """
    items = list(order_a)
    if set(items) != set(order_b):
        raise ValueError("kendall_tau requires the two orderings to cover the same set")
    rank_a = {m: i for i, m in enumerate(items)}
    rank_b = {m: i for i, m in enumerate(order_b)}
    names = items
    n = len(names)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = rank_a[names[i]] - rank_a[names[j]]
            b = rank_b[names[i]] - rank_b[names[j]]
            s = np.sign(a) * np.sign(b)
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    return float((concordant - discordant) / denom) if denom else 0.0


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


# The generator-fitter is expected to trail the anchor by ~20% (ratio 1.2) for
# the benchmark to count as fully valid. The gate is a smooth sigmoid of the
# error ratio so there is always a gradient leading *away* from pre-fittable
# regions -- no flat dead zone for the forge to get stuck on.
GATE_THRESHOLD = 1.2
GATE_STEEPNESS = 8.0


def overfit_gate(strong_err: float, overfit_err: float) -> float:
    """Validity multiplier in ``(0, 1)`` from the generator-fitting detector.

    Near ``0`` when the detector (a miner who has reverse-engineered the
    generator) matches or beats the independently-strong anchor -- i.e. the
    challenges are pre-fittable artifacts -- and ramping smoothly to near ``1``
    once the detector clearly trails the anchor. This independent layer stops the
    forge from drifting toward a synthetic-heavy, pre-fittable benchmark even when
    that benchmark looks superficially discriminating (high ``spread``).
    """
    if strong_err <= _EPS:
        return 0.0
    ratio = overfit_err / strong_err
    return float(1.0 / (1.0 + np.exp(-GATE_STEEPNESS * (ratio - GATE_THRESHOLD))))


# --------------------------------------------------------------------------- #
# Parrot gate: the "is this just repetition?" layer
# --------------------------------------------------------------------------- #
#
# ``overfit_gate`` catches *pre-fittable synthetic* structure. A different,
# orthogonal way a challenge set can fail to measure real skill is if a trivial
# nearest-neighbour *copy-the-context* forecaster (``baselines.context_parrot``)
# already does as well as the strong anchor -- then a high score rewards
# induction-head repetition, not understanding (Zhang & Gilpin, arXiv:2505.11349).
# This gate mirrors ``overfit_gate`` exactly but uses the parrot's error. It is
# report-only in ``panel_fitness`` (so the frozen ``fitness`` consensus value is
# unchanged) and folded into the opt-in ``foundational_fitness``.


def parrot_error(challenges: list[Challenge]) -> float:
    """Mean scale-normalised MAE of the context-parroting baseline."""
    from baselines import context_parrot  # local import avoids an import cycle

    total = 0.0
    for ch in challenges:
        scale = _scale(ch.context)
        pred = context_parrot(ch.context)
        total += _mae(pred, ch.truth) / scale
    return total / max(len(challenges), 1)


# Unlike the generator-fitter (a near-oracle expected to trail by ~20%), the
# parrot is a *legitimate-ish* baseline, so the bar is parity: the gate is 0.5
# exactly when parrot ties the anchor, collapsing toward 0 when parrot wins and
# rising toward 1 as the anchor pulls genuinely ahead. The gentler steepness
# reflects that we are flagging "no real margin over copying", not a sharp cliff.
PARROT_GATE_THRESHOLD = 1.0
PARROT_GATE_STEEPNESS = 6.0


def parrot_gate(strong_err: float, parrot_err: float) -> float:
    """Validity multiplier in ``(0, 1)``: ~0 when parroting matches/beats ``strong``.

    Smooth sigmoid of the error ratio ``parrot_err / strong_err`` centred at
    parity: ``0.5`` when the trivial copy-the-context baseline ties the anchor,
    ramping toward ``0`` when parrot *wins* (the set is solvable by repetition and
    rewards no real skill) and toward ``1`` once the anchor clearly beats it. An
    independent layer from :func:`overfit_gate` -- a set can be non-pre-fittable
    yet still parrot-solvable (e.g. near-recurrent chaos), and vice versa.
    """
    if strong_err <= _EPS:
        return 0.0
    ratio = parrot_err / strong_err
    return float(1.0 / (1.0 + np.exp(-PARROT_GATE_STEEPNESS * (ratio - PARROT_GATE_THRESHOLD))))


# --------------------------------------------------------------------------- #
# Coverage: the foundational-breadth layer
# --------------------------------------------------------------------------- #
#
# ``spread``/``ordering``/``gate`` all measure quality *within* whatever
# distribution the feed supplies; none of them notices whether that distribution
# spans one data-generating process or twenty. A benchmark that certifies a
# *foundation* model must also be broad -- many domains / DGPs -- so a high score
# means "generalises across worlds," not "good at the one process we test." These
# helpers measure that breadth from the per-challenge ``meta['domain']`` labels.

UNKNOWN_DOMAIN = "unknown"

# Target effective number of distinct domains per evaluation. The coverage gate
# is 1.0 once a challenge set reaches this many *effective* domains (Hill number
# of order 1); below it the gate ramps down linearly. Eight is the breadth of the
# default multi-domain feed (``domains.default_live_source``) including ``synth``.
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
    """Score a challenge set: the forge's objective and the validity gates.

    Returns ``fitness``, ``spread``, ``ordering``, ``difficulty``, ``gate`` (plus
    the raw per-model ``errors`` for diagnostics, and a report-only ``coverage``
    summary with its ``coverage_gate``).

    * ``spread = (max_err - min_err) / mean_err`` over the *legitimate* models
      (the detector is excluded so it can't inflate discrimination).
    * ``ordering = kendall_tau(achieved_order, PANEL_QUALITY_ORDER)`` over all
      models -- the panel-validity gate. If a naive model beats ``strong``, the
      anchor falls in the ranking and ``ordering`` goes negative.
    * ``gate`` -- the generator-fitting gate (see :func:`overfit_gate`).
    * ``fitness = spread * max(0, ordering) * gate`` -- discrimination, counted
      only once *both* independent validity conditions hold. Either a naive model
      beating ``strong`` (``ordering <= 0``) or the generator-fitter beating
      ``strong`` (``gate == 0``) drives the fitness to zero.
    """
    panel = panel or default_panel()
    agg = model_errors(challenges, panel)

    legit = [m for m in agg if m != "overfit"]
    legit_errs = [agg[m] for m in legit]
    mean_e = float(np.mean(legit_errs))
    spread = (max(legit_errs) - min(legit_errs)) / mean_e if mean_e > _EPS else 0.0

    ref = [m for m in PANEL_QUALITY_ORDER if m in agg]
    common = set(ref)
    achieved = [m for m in sorted(agg, key=lambda m: agg[m]) if m in common]
    ordering = kendall_tau(achieved, ref)

    gate = overfit_gate(agg.get("strong", 0.0), agg.get("overfit", 0.0))
    fitness = spread * max(0.0, ordering) * gate
    difficulty = agg.get("strong", float("nan"))

    coverage = domain_coverage(challenges)
    cov_gate = float(
        np.clip(float(coverage["effective_domains"]) / DEFAULT_COVERAGE_TARGET, 0.0, 1.0)
    )
    p_gate = parrot_gate(agg.get("strong", 0.0), parrot_error(challenges))

    return {
        "fitness": float(fitness),
        "spread": float(spread),
        "ordering": float(ordering),
        "difficulty": float(difficulty),
        "gate": float(gate),
        "errors": agg,
        "coverage": coverage,
        "coverage_gate": cov_gate,
        "parrot_gate": float(p_gate),
    }


def stratified_fitness(
    challenges: list[Challenge], panel: dict[str, PanelModel] | None = None
) -> dict[str, dict[str, object]]:
    """Per-domain fitness report: skill *within each* data-generating process.

    A single aggregate fitness can hide a benchmark that is excellent on one
    domain and meaningless on the rest. This partitions the challenge set by
    ``meta['domain']`` and scores each stratum independently, so "foundational"
    can be read as "valid and discriminating across strata," not merely on
    average. Each entry includes ``n`` (the stratum size) because small strata
    carry real sampling noise and should be read with that in mind.
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
            "ordering": res["ordering"],
            "gate": res["gate"],
            "difficulty": res["difficulty"],
        }
    return report


def foundational_fitness(
    challenges: list[Challenge],
    panel: dict[str, PanelModel] | None = None,
    target_effective_domains: float = DEFAULT_COVERAGE_TARGET,
) -> dict[str, object]:
    """Coverage-gated objective: ``foundational_fitness = fitness * coverage_gate``.

    The opt-in breadth-aware score. Identical to :func:`panel_fitness` but adds
    ``foundational_fitness``, which a forge can target instead of bare ``fitness``
    when it must climb toward *both* discrimination/validity **and** domain
    coverage. Kept separate so the core ``fitness`` (frozen within a benchmark
    version for consensus) is never silently redefined.
    """
    res = dict(panel_fitness(challenges, panel))
    cov_gate = coverage_gate(challenges, target_effective_domains)
    res["coverage_gate"] = cov_gate
    # Fold in the parrot gate too: a foundational benchmark must be broad AND not
    # solvable by trivial repetition. Both are independent multipliers on the
    # core fitness, so foundational_fitness only rises when discrimination,
    # validity, breadth, and non-parrotability all hold at once.
    p_gate = float(res.get("parrot_gate", 1.0))
    res["foundational_fitness"] = float(res["fitness"]) * cov_gate * p_gate
    return res
