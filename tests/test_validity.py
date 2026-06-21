"""Validity gate: a challenge set where a naive model beats ``strong`` is junk.

The acceptance condition: a deliberately broken challenge set (a naive baseline
out-predicts the anchor) yields low/negative ``ordering`` and near-zero
``fitness``. A well-formed set yields positive, gated fitness.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from config import CONTEXT_LEN, HORIZON, SEASONAL_PERIODS, WEAK_STATE
from generate import Challenge, build_challenges
from ingest import FreshBuffer, SyntheticLiveSource
from score import kendall_tau, panel_fitness
from seed import rng_for


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
    state = replace(
        WEAK_STATE, w_synth=0.45, w_spliced=0.35, w_aug_live=0.20, aug_severity=0.3
    )
    buffer = FreshBuffer(SyntheticLiveSource(), pool_size=64, motif_len=768)
    buffer.refresh(np.random.default_rng(7))

    # Average over several beacons (as the forge does) so the comparison reflects
    # the state, not single-draw sampling noise.
    metrics = {"fitness": 0.0, "ordering": 0.0, "gate": 0.0}
    errs_sum: dict[str, float] = {}
    n_seeds = 5
    for s in range(n_seeds):
        res = panel_fitness(build_challenges(state, buffer, rng_for("valid", s, "m"), 64))
        for k in metrics:
            metrics[k] += float(res[k]) / n_seeds
        for m, e in res["errors"].items():
            errs_sum[m] = errs_sum.get(m, 0.0) + e

    assert min(errs_sum, key=errs_sum.get) == "strong"  # anchor leads on average
    assert metrics["ordering"] > 0.0
    assert metrics["gate"] > 0.0
    assert metrics["fitness"] > 0.0
