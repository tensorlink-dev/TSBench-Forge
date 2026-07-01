"""Calibration metrics, robust aggregation, multi-seed and significance.

Acceptance: calibration error is low for a well-calibrated forecaster and high
for an overconfident one; shifted-gmean is robust to zeros; the normalized
leaderboard ranks a better model above the baseline; Friedman flags a real
difference and not a null one.
"""

from __future__ import annotations

import numpy as np

from config import CONTEXT_LEN, HORIZON
from evaluate import (
    ProbForecast,
    evaluate_forecaster,
    evaluate_multiseed,
    friedman_test,
    normalized_leaderboard,
    probabilistic,
    shifted_gmean,
)
from challenges import Challenge


def _gaussian_truth_challenges(n: int, sigma: float, seed: int) -> list[Challenge]:
    """Challenges whose truth is Gaussian noise of known sigma around a flat level."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ctx = rng.normal(0.0, sigma, size=CONTEXT_LEN)
        truth = rng.normal(0.0, sigma, size=HORIZON)
        out.append(Challenge(context=ctx, truth=truth, mode="g", meta={}))
    return out


def _well_calibrated(sigma: float):
    """A forecaster that emits the TRUE predictive distribution N(0, sigma)."""
    from evaluate import _probit

    qs_levels = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    def fc(context, meta=None):
        mean = np.zeros(HORIZON)
        quantiles = {q: mean + _probit(q) * sigma for q in qs_levels}
        return ProbForecast(mean=mean, quantiles=quantiles)

    return fc


def _overconfident(sigma: float):
    """Same mean, but a far-too-narrow band (under-dispersed -> miscalibrated)."""
    from evaluate import _probit

    qs_levels = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    def fc(context, meta=None):
        mean = np.zeros(HORIZON)
        quantiles = {q: mean + _probit(q) * (sigma * 0.2) for q in qs_levels}
        return ProbForecast(mean=mean, quantiles=quantiles)

    return fc


def test_calibration_error_separates_good_from_overconfident() -> None:
    chs = _gaussian_truth_challenges(60, sigma=1.0, seed=1)
    good = evaluate_forecaster(_well_calibrated(1.0), chs)
    bad = evaluate_forecaster(_overconfident(1.0), chs)
    assert good["pce"] < bad["pce"]
    # A well-calibrated 80% band should cover close to 80% of points.
    assert abs(good["coverage_80"] - 0.8) < 0.1
    # The overconfident model's narrow band covers far less.
    assert bad["coverage_80"] < good["coverage_80"]
    # WIS (sharpness + calibration) also prefers the calibrated model here.
    assert good["wis"] < bad["wis"]


def test_metric_keys_present() -> None:
    chs = _gaussian_truth_challenges(8, sigma=1.0, seed=2)
    res = evaluate_forecaster(_well_calibrated(1.0), chs)
    for k in ("mase", "wql", "crps", "wis", "pce", "coverage_80", "mae", "n"):
        assert k in res


def test_shifted_gmean_robust_to_zero() -> None:
    assert shifted_gmean([1.0, 1.0, 1.0]) == 1.0 or abs(shifted_gmean([1.0, 1.0, 1.0]) - 1.0) < 1e-4
    # A zero does not annihilate the whole aggregate (unlike a plain gmean).
    val = shifted_gmean([0.0, 1.0, 2.0])
    assert np.isfinite(val) and val > 0.0
    assert np.isnan(shifted_gmean([]))


def test_normalized_leaderboard_ranks_better_model_first() -> None:
    from score import seasonal_naive, strong

    # Build challenge sets with real structure so 'strong' beats seasonal_naive.
    structured = {}
    for i in range(3):
        rng = np.random.default_rng(100 + i)
        chs = []
        for _ in range(20):
            t = np.arange(CONTEXT_LEN + HORIZON)
            s = 0.05 * t + 2.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, t.size)
            chs.append(Challenge(context=s[:CONTEXT_LEN], truth=s[CONTEXT_LEN:], mode="s", meta={}))
        structured[f"ds{i}"] = chs
    board = normalized_leaderboard(
        {"strong": probabilistic(strong), "seasonal_naive": probabilistic(seasonal_naive)},
        structured,
    )
    ranks = {r["model"]: r["avg_rank"] for r in board}
    assert ranks["strong"] < ranks["seasonal_naive"]


def test_friedman_detects_and_ignores() -> None:
    # Model 0 always best, model 2 always worst -> significant.
    sig = np.array([[1.0, 2.0, 3.0]] * 10)
    res = friedman_test(sig)
    assert res["p_value"] < 0.05
    # All tied -> not significant.
    null = np.array([[1.0, 1.0, 1.0]] * 10)
    assert friedman_test(null)["p_value"] > 0.5


def test_evaluate_multiseed_reports_spread() -> None:
    sets = [_gaussian_truth_challenges(15, sigma=1.0, seed=s) for s in range(4)]
    rep = evaluate_multiseed(_well_calibrated(1.0), sets)
    assert "mase" in rep and "mean" in rep["mase"] and "std" in rep["mase"]
    assert len(rep["mase"]["values"]) == 4
