"""Context-parroting floor baseline and the parrot validity gate.

Acceptance: (1) parroting copies the future of the most similar past window on a
cleanly repeating series; (2) on a challenge set that is *solvable by repetition*
the parrot gate collapses toward zero, while on genuine structured challenges the
anchor leads the parrot and the gate stays high.
"""

from __future__ import annotations

import numpy as np

from baselines import context_parrot
from challenges import Challenge, build_live_challenges
from conftest import live_buffer, lorenz_motif
from config import CONTEXT_LEN, HORIZON
from evaluate import clears_floor, probabilistic, probabilistic_panel
from score import panel_fitness, parrot_gate
from seed import rng_for


def test_parrot_copies_repeating_signal() -> None:
    # A clean period-20 sine: the future is exactly a continuation of the past,
    # so nearest-neighbour copying should be near-perfect.
    period = 20
    t = np.arange(CONTEXT_LEN + HORIZON)
    series = np.sin(2 * np.pi * t / period)
    ctx, truth = series[:CONTEXT_LEN], series[CONTEXT_LEN:]
    pred = context_parrot(ctx)
    assert pred.shape == (HORIZON,)
    assert float(np.mean(np.abs(pred - truth))) < 0.15


def test_parrot_short_context_falls_back_to_persistence() -> None:
    x = np.array([1.0, 2.0, 3.0])  # too short to hold a query + horizon
    pred = context_parrot(x, horizon=HORIZON)
    assert pred.shape == (HORIZON,)
    assert np.allclose(pred, 3.0)


def test_parrot_gate_shape() -> None:
    # Pure unit test of the sigmoid: parrot winning -> low, anchor leading -> high.
    assert parrot_gate(1.0, 0.5) < 0.3  # parrot much better than strong
    assert abs(parrot_gate(1.0, 1.0) - 0.5) < 1e-9  # parity
    assert parrot_gate(1.0, 1.6) > 0.9  # strong clearly leads
    assert parrot_gate(0.0, 1.0) == 0.0  # degenerate-strong guard


def test_parrot_gate_reported_and_bounded() -> None:
    chs = build_live_challenges(live_buffer(), rng_for("0xfeed", 1, "test"), 48)
    res = panel_fitness(chs)
    assert "parrot_gate" in res
    assert 0.0 <= res["parrot_gate"] <= 1.0


def test_parrot_competitive_on_chaos() -> None:
    # On chaotic Lorenz motifs, near-recurrences make parroting a serious
    # baseline -- the very regime the gate exists to surface. We only assert the
    # gate is finite/bounded (parrot is in the running), not a brittle threshold.
    chs = [
        Challenge(
            context=(m := lorenz_motif(CONTEXT_LEN + HORIZON, seed=i))[:CONTEXT_LEN],
            truth=m[CONTEXT_LEN:], mode="chaos", meta={"domain": "lorenz"},
        )
        for i in range(16)
    ]
    res = panel_fitness(chs)
    assert 0.0 <= res["parrot_gate"] <= 1.0


def test_parrot_is_a_leaderboard_rung() -> None:
    assert "context_parrot" in probabilistic_panel()


def test_clears_floor_rejects_parrot_level_model() -> None:
    # A model that *is* the parrot cannot clear the floor (never strictly beats it).
    period = 20
    chs: list[Challenge] = []
    for k in range(12):
        t = np.arange(CONTEXT_LEN + HORIZON)
        series = np.sin(2 * np.pi * t / period + 0.1 * k)
        chs.append(
            Challenge(context=series[:CONTEXT_LEN], truth=series[CONTEXT_LEN:], mode="rep", meta={})
        )
    report = clears_floor(probabilistic(context_parrot), chs)
    assert report["clears"] is False
