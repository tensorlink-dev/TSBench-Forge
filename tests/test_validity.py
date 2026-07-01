"""Validity gate: a challenge set where a naive model beats ``strong`` is junk.

The acceptance condition: a deliberately broken challenge set (a naive baseline
out-predicts the anchor) yields low/negative ``ordering`` and near-zero
``fitness``. A well-formed set yields positive, gated fitness.
"""

from __future__ import annotations

import numpy as np

from challenges import Challenge
from conftest import structured_challenges
from config import CONTEXT_LEN, HORIZON, SEASONAL_PERIODS
from score import kendall_tau, panel_fitness


def _broken_challenges(n: int = 24) -> list[Challenge]:
    """Strongly trending+seasonal context, but a FLAT (mean-level) truth.

    Trend/seasonal extrapolators (``strong``, ``drift``, ``seasonal_naive``)
    over-shoot badly; the mean-reverting baselines win. The anchor is beaten, so
    the set is measuring an artifact.
    """
    rng = np.random.default_rng(0)
    out: list[Challenge] = []
    for _ in range(n):
        t = np.arange(CONTEXT_LEN + HORIZON)
        series = 0.2 * t + 2.0 * np.sin(2 * np.pi * t / SEASONAL_PERIODS[1])
        series = series + rng.normal(0.0, 0.3, size=t.size)
        ctx = series[:CONTEXT_LEN]
        truth = np.full(HORIZON, float(ctx.mean()))
        out.append(Challenge(context=ctx, truth=truth, mode="broken", meta={}))
    return out


def test_kendall_tau_extremes() -> None:
    order = ["a", "b", "c", "d"]
    assert kendall_tau(order, order) == 1.0
    assert kendall_tau(order, list(reversed(order))) == -1.0


def test_broken_set_has_low_ordering_and_zero_fitness() -> None:
    res = panel_fitness(_broken_challenges())
    errs = res["errors"]
    achieved = sorted(errs, key=errs.get)
    # The anchor must NOT be best on a broken set.
    assert achieved[0] != "strong"
    assert res["ordering"] <= 0.0
    assert res["fitness"] < 1e-6


def test_wellformed_set_is_valid_and_discriminating() -> None:
    # A well-formed set: trend+seasonal+AR series the strong anchor is built for.
    # This tests the validity-gate *logic* (does a genuinely-ordered panel score
    # positive fitness), independent of the numpy anchor's quality on raw real
    # data (which is exactly what validate_panel flags when it is insufficient).
    res = panel_fitness(structured_challenges(96, seed=3))
    errs = res["errors"]
    assert min(errs, key=errs.get) == "strong"  # anchor leads by construction
    assert res["ordering"] > 0.0
    assert res["fitness"] > 0.0
