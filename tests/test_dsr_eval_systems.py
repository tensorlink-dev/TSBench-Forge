"""Chaotic-system generators: integration backends, determinism, reference invariants.

Checks that the built-in systems integrate to the right shapes on either backend, that
the numpy fallback is deterministic per seed, and that the canonical reference
invariants (Lyapunov time, D_KY) are wired up correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from dsr_eval.systems import available_systems, from_dysts, get_system, integrate


def test_available_systems() -> None:
    assert {"lorenz", "rossler"} <= set(available_systems())


def test_integrate_shapes_observable_and_full_state() -> None:
    s = get_system("lorenz")
    obs = integrate(s, 1000, rng=np.random.default_rng(0), backend="numpy")
    full = integrate(s, 1000, rng=np.random.default_rng(0), backend="numpy", full_state=True)
    assert obs.shape == (1000,)
    assert full.shape == (1000, 3)
    # The observable is exactly coordinate `system.observable` of the full state.
    assert np.allclose(obs, full[:, s.observable])
    assert np.all(np.isfinite(full))


def test_numpy_backend_deterministic_per_seed() -> None:
    s = get_system("rossler")
    a = integrate(s, 500, rng=np.random.default_rng(7), backend="numpy")
    b = integrate(s, 500, rng=np.random.default_rng(7), backend="numpy")
    c = integrate(s, 500, rng=np.random.default_rng(8), backend="numpy")
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_scipy_backend_lands_on_same_attractor() -> None:
    scipy = pytest.importorskip("scipy")  # noqa: F841
    s = get_system("lorenz")
    rng = np.random.default_rng(3)
    obs_np = integrate(s, 4000, rng=np.random.default_rng(3), backend="numpy", full_state=True)
    obs_sp = integrate(s, 4000, rng=rng, backend="scipy", full_state=True)
    # Trajectories diverge (chaos) but the attractor's extent matches closely.
    assert np.allclose(obs_np.min(0), obs_sp.min(0), atol=3.0)
    assert np.allclose(obs_np.max(0), obs_sp.max(0), atol=3.0)


def test_auto_backend_runs() -> None:
    s = get_system("lorenz")
    obs = integrate(s, 500, rng=np.random.default_rng(0), backend="auto")
    assert obs.shape == (500,) and np.all(np.isfinite(obs))


def test_reference_invariants() -> None:
    lorenz = get_system("lorenz")
    assert lorenz.lyap_max_ref == pytest.approx(0.9056, abs=1e-3)
    assert lorenz.lyapunov_time == pytest.approx(1.0 / 0.9056, rel=1e-6)
    assert lorenz.lyapunov_time_steps == pytest.approx(lorenz.lyapunov_time / lorenz.dt, rel=1e-6)
    assert lorenz.d_ky_ref == pytest.approx(2.062, abs=1e-2)
    rossler = get_system("rossler")
    assert rossler.d_ky_ref == pytest.approx(2.013, abs=1e-2)


def test_param_and_field_overrides() -> None:
    s = get_system("lorenz", params={"rho": 99.0}, dt=0.02, burn_in=10)
    assert s.params["rho"] == 99.0
    assert s.params["sigma"] == 10.0  # untouched canonical value
    assert s.dt == 0.02 and s.burn_in == 10


def test_dysts_backend_requires_optional_dep() -> None:
    try:
        import dysts  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="dysts"):
            from_dysts("Lorenz")
    else:  # pragma: no cover - only when the optional extra is installed
        s = from_dysts("Lorenz")
        assert s.name == "dysts:Lorenz" and s.jac is None
