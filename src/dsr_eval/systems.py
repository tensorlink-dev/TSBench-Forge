"""Chaotic-system trajectory generators with known ground truth.

This is the *dataset* layer of the DSR (dynamical-systems-reconstruction) eval. A
TSFM benchmark that only scores point error (MASE) cannot tell whether a model has
learned the *dynamics* of a chaotic system -- its attractor geometry, its temporal
structure, its chaos signature -- because on a chaotic flow any forecast decorrelates
from the truth after ~one Lyapunov time (Durstewitz et al., arXiv:2602.16864). To
score the dynamics we need long, controlled trajectories from systems whose invariants
are known analytically.

Each :class:`ChaoticSystem` bundles a right-hand side, an analytic Jacobian (used by
:mod:`dsr_eval.metrics` to compute the true Lyapunov spectrum and Kaplan-Yorke
dimension), canonical parameters, and the *reference* invariants from the literature so
the unit tests have known values to check against.

Integration backends
---------------------
:func:`integrate` integrates a system to a trajectory. Three backends:

* ``"scipy"`` -- ``scipy.integrate.solve_ivp`` with the RK45 adaptive integrator
  (the protocol the paper uses). Imported lazily; enabled by ``pip install -e ".[dsr]"``.
* ``"numpy"`` -- the repo's dependency-free fixed-step RK4 (``domains._rk4_rollout``),
  so the core path stays numpy-only and consensus-safe.
* ``"dysts"`` -- pull a system from Gilpin's ``dysts`` chaotic-systems library
  (``pip install -e ".[dysts]"``) via :func:`from_dysts`; integrated with the same
  RK45/RK4 machinery.

``"auto"`` (the default) prefers scipy/RK45 and silently falls back to numpy/RK4 when
scipy is not installed, so nothing in the eval *requires* scipy.

Adding a system is a one-liner: write an ``rhs`` (and, for the Lyapunov spectrum, a
``jac``), then register a factory in :data:`_REGISTRY`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Reuse the repo's pure-numpy fixed-step RK4 so the fallback path adds no dependency.
from domains import _rk4_rollout

# Right-hand side / Jacobian signatures: f(state, params) -> derivative / matrix.
_Rhs = Callable[[np.ndarray, "dict[str, float]"], np.ndarray]
_Jac = Callable[[np.ndarray, "dict[str, float]"], np.ndarray]


# --------------------------------------------------------------------------- #
# System container
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChaoticSystem:
    """A continuous-time dynamical system plus its known reference invariants.

    Attributes:
        name: short identifier (``"lorenz"``, ``"rossler"``, ...).
        dim: state-space dimension.
        params: canonical parameter values.
        dt: sampling interval (physical time per emitted step). The Lyapunov time
            in *steps* is ``lyapunov_time / dt``.
        x0: default initial condition (perturbed per-seed by :func:`integrate`).
        observable: index of the coordinate a univariate forecaster actually sees.
        burn_in: steps discarded so the trajectory settles onto the attractor.
        rhs: ``f(state, params) -> d(state)/dt``.
        jac: analytic Jacobian ``J(state, params)`` (``None`` if unavailable, e.g.
            for ``dysts`` systems) -- needed only for the true Lyapunov spectrum.
        lyap_spectrum_ref: literature Lyapunov spectrum (descending), for tests.
    """

    name: str
    dim: int
    params: dict[str, float]
    dt: float
    x0: tuple[float, ...]
    rhs: _Rhs
    observable: int = 0
    burn_in: int = 2000
    jac: _Jac | None = None
    lyap_spectrum_ref: tuple[float, ...] | None = None

    def field(self) -> Callable[[np.ndarray], np.ndarray]:
        """Autonomous vector field ``f(x)`` with parameters bound (for integrators)."""
        return lambda x: self.rhs(x, self.params)

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Analytic Jacobian at ``state``. Raises if the system has no ``jac``."""
        if self.jac is None:
            raise ValueError(f"system {self.name!r} has no analytic Jacobian")
        return self.jac(np.asarray(state, dtype=float), self.params)

    @property
    def lyap_max_ref(self) -> float | None:
        """Reference maximal Lyapunov exponent (per unit *time*), if known."""
        return max(self.lyap_spectrum_ref) if self.lyap_spectrum_ref else None

    @property
    def lyapunov_time(self) -> float | None:
        """Reference Lyapunov time ``1 / lambda_max`` (physical time units)."""
        lam = self.lyap_max_ref
        return 1.0 / lam if lam and lam > 0 else None

    @property
    def lyapunov_time_steps(self) -> float | None:
        """Reference Lyapunov time expressed in integration *steps* (``1/lam/dt``)."""
        lt = self.lyapunov_time
        return lt / self.dt if lt else None

    @property
    def d_ky_ref(self) -> float | None:
        """Reference Kaplan-Yorke (Lyapunov) dimension from the known spectrum."""
        if not self.lyap_spectrum_ref:
            return None
        from dsr_eval.metrics import kaplan_yorke_dimension

        return kaplan_yorke_dimension(self.lyap_spectrum_ref)


# --------------------------------------------------------------------------- #
# Built-in systems: right-hand sides + analytic Jacobians
# --------------------------------------------------------------------------- #


def _lorenz_rhs(x: np.ndarray, p: dict[str, float]) -> np.ndarray:
    s, r, b = p["sigma"], p["rho"], p["beta"]
    return np.array([s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]])


def _lorenz_jac(x: np.ndarray, p: dict[str, float]) -> np.ndarray:
    s, r, b = p["sigma"], p["rho"], p["beta"]
    return np.array(
        [[-s, s, 0.0], [r - x[2], -1.0, -x[0]], [x[1], x[0], -b]]
    )


def _rossler_rhs(x: np.ndarray, p: dict[str, float]) -> np.ndarray:
    a, b, c = p["a"], p["b"], p["c"]
    return np.array([-x[1] - x[2], x[0] + a * x[1], b + x[2] * (x[0] - c)])


def _rossler_jac(x: np.ndarray, p: dict[str, float]) -> np.ndarray:
    a, _b, c = p["a"], p["b"], p["c"]
    return np.array([[0.0, -1.0, -1.0], [1.0, a, 0.0], [x[2], 0.0, x[0] - c]])


# Canonical parameterisations and literature invariants. The Lyapunov spectra are
# the widely-cited values (e.g. Sprott, "Chaos and Time-Series Analysis"): Lorenz-63
# lambda ~ (0.906, 0, -14.572) -> D_KY ~ 2.062; Rossler lambda ~ (0.0714, 0, -5.39)
# -> D_KY ~ 2.013.
def _lorenz_system() -> ChaoticSystem:
    return ChaoticSystem(
        name="lorenz",
        dim=3,
        params={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        dt=0.01,
        x0=(1.0, 1.0, 20.0),
        rhs=_lorenz_rhs,
        jac=_lorenz_jac,
        observable=0,
        burn_in=2000,
        lyap_spectrum_ref=(0.9056, 0.0, -14.5723),
    )


def _rossler_system() -> ChaoticSystem:
    return ChaoticSystem(
        name="rossler",
        dim=3,
        params={"a": 0.2, "b": 0.2, "c": 5.7},
        dt=0.05,
        x0=(1.0, 1.0, 0.0),
        rhs=_rossler_rhs,
        jac=_rossler_jac,
        observable=0,
        burn_in=2000,
        lyap_spectrum_ref=(0.0714, 0.0, -5.3943),
    )


_REGISTRY: dict[str, Callable[[], ChaoticSystem]] = {
    "lorenz": _lorenz_system,
    "rossler": _rossler_system,
}


def available_systems() -> list[str]:
    """Names of the built-in systems (``dysts`` systems are reached via name string)."""
    return sorted(_REGISTRY)


def get_system(name: str, *, params: dict[str, float] | None = None, **overrides) -> ChaoticSystem:
    """Construct a registered :class:`ChaoticSystem` by name.

    ``params`` overrides individual canonical parameters; any other keyword
    (``dt``, ``burn_in``, ``observable``, ...) overrides the corresponding field.
    Unknown names that start with ``dysts:`` are routed to :func:`from_dysts`.
    """
    if name.startswith("dysts:"):
        return from_dysts(name.split(":", 1)[1], **overrides)
    if name not in _REGISTRY:
        raise KeyError(f"unknown system {name!r}; known: {available_systems()} (or 'dysts:<Name>')")
    sys = _REGISTRY[name]()
    if params:
        merged = {**sys.params, **params}
        sys = _replace_system(sys, params=merged)
    if overrides:
        sys = _replace_system(sys, **overrides)
    return sys


def _replace_system(sys: ChaoticSystem, **changes) -> ChaoticSystem:
    from dataclasses import replace

    return replace(sys, **changes)


# --------------------------------------------------------------------------- #
# dysts backend (optional, lazy)
# --------------------------------------------------------------------------- #


def from_dysts(name: str, **overrides) -> ChaoticSystem:
    """Wrap a system from Gilpin's ``dysts`` library as a :class:`ChaoticSystem`.

    The ``dysts`` model supplies the right-hand side and a canonical ``dt``; no
    analytic Jacobian is exposed, so the true Lyapunov spectrum / ``D_KY`` are not
    available for these systems (the data-driven metrics still apply). Requires the
    optional dependency: ``pip install -e ".[dysts]"``.
    """
    try:
        import dysts.flows as flows
    except ImportError as exc:  # pragma: no cover - exercised only with the extra
        raise ImportError(
            "the 'dysts' backend needs the optional dependency: pip install -e '.[dysts]'"
        ) from exc

    model = getattr(flows, name)()

    def rhs(x: np.ndarray, _p: dict[str, float]) -> np.ndarray:
        return np.asarray(model.rhs(list(x), 0.0), dtype=float)

    dim = len(np.atleast_1d(model.ic))
    dt = float(getattr(model, "dt", 0.01)) or 0.01
    sys = ChaoticSystem(
        name=f"dysts:{name}",
        dim=dim,
        params=dict(getattr(model, "params", {}) or {}),
        dt=dt,
        x0=tuple(float(v) for v in np.atleast_1d(model.ic)),
        rhs=rhs,
        observable=0,
        burn_in=2000,
        jac=None,
        lyap_spectrum_ref=None,
    )
    return _replace_system(sys, **overrides) if overrides else sys


# --------------------------------------------------------------------------- #
# Integration
# --------------------------------------------------------------------------- #


def _scipy_available() -> bool:
    try:
        import scipy.integrate  # noqa: F401

        return True
    except ImportError:
        return False


def _integrate_scipy(
    system: ChaoticSystem, x0: np.ndarray, total: int, dt: float, *, rtol: float, atol: float
) -> np.ndarray:
    from scipy.integrate import solve_ivp

    field = system.field()
    t_eval = np.arange(total) * dt
    sol = solve_ivp(
        lambda _t, y: field(y),
        (0.0, float(t_eval[-1])),
        np.asarray(x0, dtype=float),
        method="RK45",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=dt,
    )
    if not sol.success:  # pragma: no cover - solver only fails on pathological params
        raise RuntimeError(f"solve_ivp failed for {system.name!r}: {sol.message}")
    return sol.y.T  # (total, dim)


def integrate(
    system: ChaoticSystem,
    n_steps: int,
    *,
    dt: float | None = None,
    burn_in: int | None = None,
    backend: str = "auto",
    rng: np.random.Generator | None = None,
    perturb: float = 1.0,
    rtol: float = 1e-9,
    atol: float = 1e-9,
    full_state: bool = False,
) -> np.ndarray:
    """Integrate ``system`` to ``n_steps`` samples after discarding the burn-in.

    Returns the 1-D observable (coordinate ``system.observable``) by default, or the
    full ``(n_steps, dim)`` state when ``full_state=True``. ``rng`` perturbs the
    initial condition (scaled by ``perturb``) so different seeds give different
    trajectories on the same attractor. ``backend`` is ``"auto"`` (RK45 if scipy is
    present, else RK4), ``"scipy"``, or ``"numpy"``.
    """
    dt = system.dt if dt is None else dt
    burn_in = system.burn_in if burn_in is None else burn_in
    total = burn_in + n_steps

    x0 = np.asarray(system.x0, dtype=float).copy()
    if rng is not None and perturb:
        x0 = x0 + perturb * rng.standard_normal(x0.size)

    if backend == "auto":
        backend = "scipy" if _scipy_available() else "numpy"
    if backend == "scipy":
        states = _integrate_scipy(system, x0, total, dt, rtol=rtol, atol=atol)
    elif backend == "numpy":
        states = _rk4_rollout(system.field(), x0, total - 1, dt)
    else:
        raise ValueError(f"unknown backend {backend!r}; use 'auto', 'scipy', or 'numpy'")

    states = states[burn_in:total]
    if full_state:
        return np.asarray(states, dtype=float)
    return np.asarray(states[:, system.observable], dtype=float)


__all__ = [
    "ChaoticSystem",
    "available_systems",
    "from_dysts",
    "get_system",
    "integrate",
]
