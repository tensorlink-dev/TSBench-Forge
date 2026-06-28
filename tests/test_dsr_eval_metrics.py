"""DSR metrics against known values.

The acceptance criteria from the task:
  * D_stsp ~ 0 for identical distributions (binning AND GMM estimators);
  * the GMM/Monte-Carlo estimator separates same-system from unrelated;
  * the Lyapunov spectrum of Lorenz-63 has lambda_max ~ 0.9 and D_KY ~ 2.06, and
    Kaplan-Yorke from a known spectrum reproduces the textbook dimensions;
  * D_H is bounded, ~0 for identical, and averages across dimensions;
  * VPT in Lyapunov units scales correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from dsr_eval.metrics import (
    d_stsp,
    d_stsp_gmm,
    kaplan_yorke_dimension,
    lyapunov_spectrum,
    max_lyapunov_rate,
    power_spectrum_hellinger,
    state_space_divergence,
    valid_prediction_time,
    vpt_lyapunov,
)
from dsr_eval.systems import get_system, integrate


def _lorenz(length: int, seed: int) -> np.ndarray:
    return integrate(get_system("lorenz"), length, rng=np.random.default_rng(seed), backend="numpy")


# --------------------------------------------------------------------------- #
# Kaplan-Yorke dimension from a known spectrum
# --------------------------------------------------------------------------- #


def test_kaplan_yorke_known_systems() -> None:
    # Textbook spectra -> textbook fractal dimensions.
    assert kaplan_yorke_dimension((0.9056, 0.0, -14.5723)) == pytest.approx(2.062, abs=1e-3)
    assert kaplan_yorke_dimension((0.0714, 0.0, -5.3943)) == pytest.approx(2.013, abs=1e-3)


def test_kaplan_yorke_edge_cases() -> None:
    assert kaplan_yorke_dimension((-0.1, -0.2, -0.3)) == 0.0  # all negative -> fixed point
    assert kaplan_yorke_dimension((0.5, 0.5)) == 2.0  # whole spectrum non-negative -> full dim


# --------------------------------------------------------------------------- #
# Lyapunov spectrum of a real system (Benettin)
# --------------------------------------------------------------------------- #


def test_lyapunov_spectrum_lorenz() -> None:
    spec = lyapunov_spectrum(get_system("lorenz"), n_steps=20000, rng=np.random.default_rng(0))
    assert spec[0] == pytest.approx(0.9056, abs=0.06)  # lambda_max ~ 0.9
    assert spec[1] == pytest.approx(0.0, abs=0.05)  # the zero exponent of a flow
    assert spec[2] < -10.0  # strongly contracting third exponent
    assert kaplan_yorke_dimension(spec) == pytest.approx(2.062, abs=0.05)


def test_lyapunov_spectrum_rossler() -> None:
    spec = lyapunov_spectrum(get_system("rossler"), n_steps=20000, rng=np.random.default_rng(0))
    assert spec[0] == pytest.approx(0.0714, abs=0.03)
    assert kaplan_yorke_dimension(spec) == pytest.approx(2.013, abs=0.05)


# --------------------------------------------------------------------------- #
# D_stsp: identical -> 0, and discrimination (binning + GMM)
# --------------------------------------------------------------------------- #


def test_d_stsp_zero_for_identical() -> None:
    a = _lorenz(4000, 1)
    assert d_stsp(a, a) == pytest.approx(0.0, abs=1e-9)  # binning, exact
    assert d_stsp_gmm(a, a, seed=0) == 0.0  # identical GMM fits cancel exactly


def test_d_stsp_gmm_discriminates() -> None:
    a = _lorenz(4000, 1)
    b = _lorenz(4000, 2)  # same attractor, different orbit
    t = np.arange(4000)
    sine = np.sin(2 * np.pi * t / 50)  # unrelated periodic signal
    same = d_stsp_gmm(b, a, seed=0)
    diff = d_stsp_gmm(sine, a, seed=0)
    assert same < diff
    assert same < 0.5 * diff


def test_state_space_divergence_auto_uses_gmm_in_high_dim() -> None:
    a = _lorenz(4000, 3)
    b = _lorenz(4000, 4)
    sine = np.sin(2 * np.pi * np.arange(4000) / 50)
    # m=4 -> auto switches to the GMM/Monte-Carlo estimator.
    same = state_space_divergence(b, a, m=4)
    diff = state_space_divergence(sine, a, m=4)
    assert same < diff


# --------------------------------------------------------------------------- #
# D_H: bounded, ~0 for identical, multivariate averaging
# --------------------------------------------------------------------------- #


def test_power_spectrum_hellinger_identical_and_bounded() -> None:
    a = _lorenz(2000, 5)
    assert power_spectrum_hellinger(a, a) == pytest.approx(0.0, abs=1e-6)
    val = power_spectrum_hellinger(np.arange(2000.0), a)
    assert 0.0 <= val <= 1.0


def test_power_spectrum_hellinger_averages_dimensions() -> None:
    full = integrate(get_system("lorenz"), 4000, rng=np.random.default_rng(6), full_state=True)
    # A 2-column trajectory vs itself is 0; vs an unrelated signal is positive.
    assert power_spectrum_hellinger(full[:, :2], full[:, :2]) == pytest.approx(0.0, abs=1e-6)
    noise = np.random.default_rng(0).standard_normal((4000, 2))
    assert power_spectrum_hellinger(noise, full[:, :2]) > 0.1


# --------------------------------------------------------------------------- #
# Data-driven lambda_max and VPT in Lyapunov units
# --------------------------------------------------------------------------- #


def test_max_lyapunov_rate_positive_for_chaos_zero_for_cycle() -> None:
    chaos = _lorenz(4000, 7)
    cycle = np.sin(2 * np.pi * np.arange(4000) / 40)
    lam_chaos = max_lyapunov_rate(chaos, dt=0.01)
    lam_cycle = max_lyapunov_rate(cycle, dt=0.01)
    assert lam_chaos > 0.0
    assert lam_chaos > lam_cycle


def test_vpt_lyapunov_units() -> None:
    true = _lorenz(400, 8)[100:200]
    steps = valid_prediction_time(true, true, threshold=0.4)
    in_lyap = vpt_lyapunov(true, true, dt=0.01, lyapunov_time=1.0 / 0.9056, threshold=0.4)
    lt_steps = (1.0 / 0.9056) / 0.01
    assert in_lyap == pytest.approx(steps / lt_steps, rel=1e-9)
    # A clearly diverging forecast (large constant offset) loses validity sooner.
    diverged = true + 10.0
    assert vpt_lyapunov(true, diverged, dt=0.01, lyapunov_time=1.0 / 0.9056) < in_lyap
