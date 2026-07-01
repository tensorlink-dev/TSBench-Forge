"""Independent validation of a forecasting anchor — the missing credibility step.

The whole benchmark rests on the ``strong`` anchor being genuinely good. But
checking that on the benchmark's *own* challenges is circular: a hollow anchor can
look fine on a distribution it helped shape. **Independent validation** breaks the
circle — establish the anchor's quality on a *held-out, real benchmark disjoint
from the live feed*, exactly as TSFM papers validate zero-shot on external data
(e.g. GIFT-Eval), and only then promote it to the anchor.

This module is that step, end to end:

1. :func:`validation_benchmark` builds a static set of forecasting tasks from real
   public series that are **disjoint from the live feed** (commodity prices,
   transport demand, demography) — the external yardstick.
2. :func:`independent_leaderboard` scores candidate anchors on it (MASE/WQL/CRPS),
   so "this model is good" is a measured claim, not an assumption.
3. :func:`is_independently_validated` is the go/no-go: the candidate must beat
   every classical baseline on the held-out set by a margin.
4. :func:`resolve_anchor` picks the best *available* real anchor — a real TSFM
   (Chronos/TimesFM) if the optional deps and weights are present, else a
   literature-validated classical model (statsforecast), else the numpy default —
   and reports which one was used, so a run is never silently on the placeholder.
5. :func:`leakage_gap` contrasts the anchor's held-out (independent) score with its
   score on the live benchmark reveal: a large, persistent gap is the leakage
   signal the README calls for.

Nothing here needs torch: the TSFM path is optional and lazily imported, so the
harness runs offline with classical anchors and lights up the moment a real TSFM
is installed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from config import CONTEXT_LEN, HORIZON
from evaluate import (
    Forecaster,
    ProbForecast,
    _naive_scale,
    evaluate_forecaster,
    leaderboard,
    probabilistic,
)
from challenges import Challenge
from ingest import FreshBuffer
from live_feeds import CsvFeed, Fetcher, cached_fetch
from score import default_panel

# Real public series that are DISJOINT from the live feed (``live_feeds.REGISTRY``).
# Distinct domains so "good here" means broad skill, and external so validation is
# not circular. Each is long enough for a CONTEXT_LEN+HORIZON window.
VALIDATION_REGISTRY: dict[str, dict] = {
    "commodity_gold": {
        "url": "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv",
        "value_column": "Price",
        "note": "Monthly gold price, 1833-present.",
    },
    "transport_taxi": {
        "url": "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv",
        "value_column": "value",
        "note": "NYC taxi demand, half-hourly 2014-2015.",
    },
    "demography_births": {
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv",
        "value_column": "Births",
        "note": "Daily female births, California 1959.",
    },
}


def _challenge_from_window(window: np.ndarray, domain: str) -> Challenge:
    """Split a real window into (context, truth) and wrap it as a Challenge."""
    ctx = np.asarray(window[:CONTEXT_LEN], dtype=float)
    truth = np.asarray(window[CONTEXT_LEN : CONTEXT_LEN + HORIZON], dtype=float)
    return Challenge(context=ctx, truth=truth, mode="real_holdout", meta={"domain": domain})


def _is_degenerate(window: np.ndarray, min_scale: float = 1e-3) -> bool:
    """A window whose context is near-constant — trivial to forecast and a blow-up
    risk for the MASE denominator (``_naive_scale`` -> ~0). Standard TSFM benchmarks
    drop such series; so do we, so the held-out scores stay well-conditioned.
    """
    return _naive_scale(np.asarray(window[:CONTEXT_LEN], dtype=float)) < min_scale


def validation_benchmark(
    n_per_domain: int = 40,
    *,
    fetch: Fetcher | None = None,
    names: list[str] | None = None,
    seed: int = 20240619,
    max_oversample: int = 12,
) -> list[Challenge]:
    """Build the held-out, real, *external* validation set (disjoint from the live feed).

    Pulls held-out real series, slices fixed-geometry (context, truth) windows, and
    returns them as plain :class:`Challenge` objects so the same metrics score them.
    Purely zero-shot: no model is fit on these, so any anchor's score here is an
    independent measure of skill. Degenerate (near-constant) windows are rejected
    and refilled so MASE stays well-conditioned on smooth real series.
    """
    f = fetch or cached_fetch()
    rng = np.random.default_rng(seed)
    chosen = names or list(VALIDATION_REGISTRY)
    out: list[Challenge] = []
    for name in chosen:
        spec = VALIDATION_REGISTRY[name]
        feed = CsvFeed(url=spec["url"], domain=name, value_column=spec["value_column"], fetch=f)
        kept: list[np.ndarray] = []
        for _ in range(max_oversample):
            if len(kept) >= n_per_domain:
                break
            for w in feed.pull(n_per_domain, CONTEXT_LEN + HORIZON, rng):
                if not _is_degenerate(w):
                    kept.append(w)
                if len(kept) >= n_per_domain:
                    break
        out.extend(_challenge_from_window(w, name) for w in kept[:n_per_domain])
    return out


def independent_leaderboard(
    candidates: dict[str, Forecaster],
    benchmark: list[Challenge] | None = None,
    **bench_kwargs,
) -> list[dict[str, object]]:
    """Rank candidate anchors on the held-out external set (MASE/WQL/CRPS)."""
    bench = benchmark if benchmark is not None else validation_benchmark(**bench_kwargs)
    return leaderboard(candidates, bench)


# The classical baselines a real anchor must beat to earn the title. These are the
# legitimate reference models (everything but ``strong`` itself), lifted to probabilistic.
def _baselines() -> dict[str, Forecaster]:
    return {
        name: probabilistic(fn)
        for name, fn in default_panel().items()
        if name != "strong"
    }


@dataclass
class ValidationResult:
    """Outcome of validating one candidate anchor on the held-out benchmark."""

    name: str
    validated: bool
    mase: float
    beaten_by: list[str] = field(default_factory=list)  # baselines that beat it ([] == clean win)
    margin: float = 0.0  # MASE gap to the best baseline (positive == candidate leads)
    board: list[dict[str, object]] = field(default_factory=list)


def is_independently_validated(
    candidate: Forecaster,
    *,
    name: str = "candidate",
    benchmark: list[Challenge] | None = None,
    min_margin: float = 0.0,
    **bench_kwargs,
) -> ValidationResult:
    """Go/no-go: does ``candidate`` beat every classical baseline on the held-out set?

    Returns a :class:`ValidationResult` with the margin to the best baseline and the
    full leaderboard. ``validated`` is True only when the candidate's MASE leads the
    best baseline by at least ``min_margin`` — the same bar the README sets for an
    anchor that may certify TSFM quality.
    """
    bench = benchmark if benchmark is not None else validation_benchmark(**bench_kwargs)
    board = leaderboard({name: candidate, **_baselines()}, bench)
    by_name = {r["model"]: r for r in board}
    cand_mase = float(by_name[name]["mase"])
    baseline_mases = {m: float(by_name[m]["mase"]) for m in by_name if m != name}
    best_baseline = min(baseline_mases, key=baseline_mases.get)
    margin = baseline_mases[best_baseline] - cand_mase
    beaten_by = [m for m, v in baseline_mases.items() if v <= cand_mase]
    return ValidationResult(
        name=name,
        validated=bool(margin >= min_margin and not beaten_by),
        mase=cand_mase,
        beaten_by=beaten_by,
        margin=margin,
        board=board,
    )


# --------------------------------------------------------------------------- #
# Resolving the best AVAILABLE real anchor (TSFM -> classical -> numpy)
# --------------------------------------------------------------------------- #


def _point_from_tsfm(tsfm: Callable[[np.ndarray, dict | None], ProbForecast]):
    """Adapt a ProbForecast-emitting TSFM to the point ``PanelModel`` the panel needs."""

    def point(context: np.ndarray, meta: dict | None = None) -> np.ndarray:
        return np.asarray(tsfm(context, meta).mean, dtype=float)

    return point


@dataclass
class ResolvedAnchor:
    """A chosen anchor plus provenance: which kind it is and why."""

    kind: str  # "tsfm" | "statsforecast" | "numpy"
    detail: str
    point_model: Callable[[np.ndarray, dict | None], np.ndarray]  # for score.default_panel
    forecaster: Forecaster  # probabilistic, for the leaderboard


def resolve_anchor(prefer_tsfm: str | None = "chronos") -> ResolvedAnchor:
    """Pick the best anchor that is actually runnable here, with provenance.

    Order of preference:

    1. A real **TSFM** (``prefer_tsfm`` via ``tsfm_adapters.load_tsfm``) — needs the
       optional deps and staged weights; the genuine "independently-validated TSFM".
    2. A **statsforecast** classical model (AutoETS) — M-competition-validated, a
       legitimate real anchor when no TSFM is installed.
    3. The **numpy** default ``strong`` — always available, but only a placeholder.

    Each tier degrades silently to the next on ImportError / load failure, so this
    returns the strongest anchor the current environment supports.
    """
    if prefer_tsfm:
        try:  # pragma: no cover - exercised only where torch + weights exist
            from tsfm_adapters import load_tsfm

            tsfm = load_tsfm(prefer_tsfm)
            tsfm(np.zeros(CONTEXT_LEN))  # force lazy load now so failure falls through
            return ResolvedAnchor(
                kind="tsfm",
                detail=f"real TSFM via tsfm_adapters.load_tsfm({prefer_tsfm!r})",
                point_model=_point_from_tsfm(tsfm),
                forecaster=tsfm,
            )
        except Exception:
            pass

    from score import try_statsforecast_strong

    sf = try_statsforecast_strong()
    if sf is not None:
        return ResolvedAnchor(
            kind="statsforecast",
            detail="statsforecast AutoETS (M-competition-validated classical model)",
            point_model=sf,
            forecaster=probabilistic(sf),
        )

    strong = default_panel()["strong"]
    return ResolvedAnchor(
        kind="numpy",
        detail="numpy backtest-selected ensemble (placeholder — not independently validated)",
        point_model=strong,
        forecaster=probabilistic(strong),
    )


def leakage_gap(
    anchor: Forecaster,
    reveal: list[Challenge],
    *,
    benchmark: list[Challenge] | None = None,
    **bench_kwargs,
) -> dict[str, float]:
    """Independent (held-out) score vs live-reveal score for the same anchor.

    The README's leakage detector: track the gap over time. If the live-reveal
    score races ahead of the static one, the live distribution is leaking. Returns
    both MASE scores and their difference (reveal minus held-out).
    """
    bench = benchmark if benchmark is not None else validation_benchmark(**bench_kwargs)
    static = evaluate_forecaster(anchor, bench)["mase"]
    live = evaluate_forecaster(anchor, reveal)["mase"]
    return {"static_mase": static, "live_mase": live, "gap": live - static}


def validated_panel(
    anchor: ResolvedAnchor | None = None,
) -> tuple[dict, ResolvedAnchor]:
    """Build the reference panel with the resolved real anchor swapped in for ``strong``."""
    a = anchor or resolve_anchor()
    return default_panel(strong_model=a.point_model), a


# Re-export so callers can build the live benchmark feed without importing both.
def live_buffer(pool_size: int = 96, motif_len: int = 384, *, fetch: Fetcher | None = None):
    """A FreshBuffer over the real benchmark feed (live_feeds), for convenience."""
    from live_feeds import build_real_live_source

    source = build_real_live_source(fetch=fetch)
    return FreshBuffer(source, pool_size=pool_size, motif_len=motif_len)
