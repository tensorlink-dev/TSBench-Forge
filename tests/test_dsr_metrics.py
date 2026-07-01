"""Long-horizon DSR metrics: geometric (D_stsp), temporal (D_H), VPT, Lyapunov.

Acceptance, using real Lorenz-63 trajectories from ``domains`` plus simple
analytic signals:
  * D_stsp and D_H are ~0 between two trajectories of the SAME system and large
    between a chaotic attractor and an unrelated (periodic) signal;
  * a parroting-style cyclic collapse is exposed by D_H (peaked vs broadband);
  * VPT is full-length for an identical forecast and small for a diverging one;
  * the Lyapunov estimate is clearly positive for chaos and ~0 for a limit cycle.
"""

from __future__ import annotations

import numpy as np

from conftest import lorenz_motif
from dsr_metrics import (
    d_h,
    d_stsp,
    delay_embed,
    free_run,
    max_lyapunov,
    valid_prediction_time,
)


def _lorenz(length: int, seed: int) -> np.ndarray:
    return lorenz_motif(length, seed)


def test_delay_embed_shape() -> None:
    x = np.sin(np.linspace(0, 40 * np.pi, 1000))
    emb = delay_embed(x, m=3, tau=5)
    assert emb.shape[1] == 3
    assert emb.shape[0] == 1000 - 2 * 5


def test_dstsp_same_system_small_vs_unrelated_large() -> None:
    a = _lorenz(4000, 1)
    b = _lorenz(4000, 2)  # same attractor, different initial condition
    t = np.arange(4000)
    sine = np.sin(2 * np.pi * t / 50)  # unrelated periodic signal
    same = d_stsp(b, a)
    diff = d_stsp(sine, a)
    assert same < diff
    assert same < diff * 0.6  # clearly separated, not a coin flip


def test_dh_same_system_small_vs_unrelated_large() -> None:
    a = _lorenz(4000, 3)
    b = _lorenz(4000, 4)
    t = np.arange(4000)
    sine = np.sin(2 * np.pi * t / 50)
    assert d_h(b, a) < d_h(sine, a)


def test_dh_exposes_periodic_collapse() -> None:
    # A chaotic truth vs a model that collapsed to a single clean oscillation
    # (the context-parroting failure mode) -> large temporal misalignment.
    a = _lorenz(4000, 5)
    t = np.arange(4000)
    collapsed = np.sin(2 * np.pi * t / 12)
    assert d_h(collapsed, a) > 0.3


def test_dh_bounded() -> None:
    a = _lorenz(2000, 6)
    assert 0.0 <= d_h(a, a) < 1e-6
    assert 0.0 <= d_h(np.arange(2000.0), a) <= 1.0


def test_vpt_identical_vs_diverging() -> None:
    true = _lorenz(200, 7)
    horizon = true[100:160]
    identical = valid_prediction_time(horizon, horizon, threshold=0.4)
    assert identical == len(horizon)
    diverging = valid_prediction_time(horizon, horizon[::-1], threshold=0.4)
    assert diverging < len(horizon)


def test_vpt_in_lyapunov_units() -> None:
    true = _lorenz(200, 8)
    h = true[100:160]
    steps = valid_prediction_time(h, h, threshold=0.4)
    in_lyap = valid_prediction_time(h, h, threshold=0.4, lyapunov_time=10.0)
    assert abs(in_lyap - steps / 10.0) < 1e-9


def test_lyapunov_positive_for_chaos_zero_for_cycle() -> None:
    chaos = _lorenz(4000, 9)
    t = np.arange(4000)
    limit_cycle = np.sin(2 * np.pi * t / 40)  # periodic -> ~0 exponent
    lam_chaos = max_lyapunov(chaos)
    lam_cycle = max_lyapunov(limit_cycle)
    assert lam_chaos > lam_cycle
    assert lam_chaos > 0.0


def test_free_run_reaches_length() -> None:
    from score import seasonal_naive

    ctx = _lorenz(300, 10)[:256]
    gen = free_run(seasonal_naive, ctx, total=500)
    assert gen.shape == (500,)
