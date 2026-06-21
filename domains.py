"""DGP zoo: a broad, multi-domain live feed of distinct data-generating processes.

Foundational-breadth role
-------------------------
``LiveSource`` (see ``ingest.py``) is the benchmark's diversity intake. A single
source means a single data-generating process, which makes the benchmark
*narrow*: a high score then certifies "good at one process," not "good across
worlds" -- the opposite of what a **foundation-model** benchmark should measure.
A narrow feed is also invisible to every other defence here: the validity gate,
the generator-fitting gate and the forge all operate *within* whatever
distribution the feed supplies, so none of them notices if that distribution
covers one domain or twenty.

This module is the investment the README's "recipe-space size" / "feed freshness"
robustness conditions call for. It supplies a *zoo* of genuinely different
dynamical systems -- chaotic flows, chaotic maps, a limit cycle, and stochastic
processes -- each tagged with its ``domain``, plus a :class:`MixtureLiveSource`
that blends them so every refresh spans many DGPs at once.

Relationship to dynaprior
--------------------------
The systems are pure-numpy reimplementations of dynaprior's generating families
(F5 *named*: Lorenz-63 / Rössler / Hénon / logistic / Hopf normal form;
F3 *stochastic*: Ornstein-Uhlenbeck / jump-diffusion), using the same canonical
parameterisations. They are deliberately **self-contained** -- no dynaprior
import, no new dependency, no network -- so the core path stays numpy-only and
consensus-safe: every motif is a pure function of the passed-in ``rng``. A literal
dynaprior adapter (importing its keyed-RNG generators) remains the documented
extension point; this zoo is the offline, dependency-free stand-in that proves
the wiring.

Each source emits a *one-dimensional observable* (one coordinate of a possibly
multi-dimensional system) -- what a forecaster actually sees. Motifs are made
finite and non-degenerate by construction (bounded attractors, stable ``dt``,
state clipping) and z-scored by :func:`_finalize`, so heterogeneous systems land
on a comparable scale and ``FreshBuffer`` never holds a constant or non-finite
window.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ingest import LiveSource, SyntheticLiveSource

# Burn-in steps discarded before an observable window so flows/maps settle onto
# their attractor (or a stochastic process reaches its stationary regime).
_BURN_ODE = 600
_BURN_MAP = 100
_BURN_SDE = 200
_CLIP = 1e6


# --------------------------------------------------------------------------- #
# Numpy integrators (pure, deterministic)
# --------------------------------------------------------------------------- #


def _rk4_rollout(
    field: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    n_steps: int,
    dt: float,
    clip: float = _CLIP,
) -> np.ndarray:
    """Fixed-step RK4 rollout of an autonomous vector field; ``(n_steps+1, dim)``."""
    x = np.asarray(x0, dtype=float).copy()
    out = np.empty((n_steps + 1, x.size))
    out[0] = x
    for t in range(n_steps):
        k1 = field(x)
        k2 = field(x + 0.5 * dt * k1)
        k3 = field(x + 0.5 * dt * k2)
        k4 = field(x + dt * k3)
        x = np.clip(x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), -clip, clip)
        out[t + 1] = x
    return out


def _map_rollout(
    step: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, n_steps: int
) -> np.ndarray:
    """Iterate a discrete map ``x_{t+1} = step(x_t)``; ``(n_steps+1, dim)``."""
    x = np.asarray(x0, dtype=float).copy()
    out = np.empty((n_steps + 1, x.size))
    out[0] = x
    for t in range(n_steps):
        x = step(x)
        out[t + 1] = x
    return out


def _finalize(series: np.ndarray) -> np.ndarray:
    """Make a raw observable finite, non-constant, and unit-scale (z-scored).

    Heterogeneous systems live on wildly different scales (Lorenz ``x ~ ±15``, a
    logistic orbit in ``[0, 1]``); z-scoring puts them on a common footing so no
    single domain dominates a splice and the per-challenge error normalisation in
    ``score`` stays well-conditioned. A degenerate (constant) window -- which
    would make the naive one-step scale collapse -- falls back to a deterministic
    ramp rather than a flat line.
    """
    x = np.nan_to_num(np.asarray(series, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean()
    s = float(np.std(x))
    if s < 1e-8:
        x = np.linspace(-1.0, 1.0, x.size)
        x = x - x.mean()
        s = float(np.std(x)) or 1.0
    return x / s


# --------------------------------------------------------------------------- #
# Vector fields / maps (module-scope factories -- no loop-variable capture)
# --------------------------------------------------------------------------- #


def _lorenz_field(sigma: float, rho: float, beta: float) -> Callable[[np.ndarray], np.ndarray]:
    def f(x: np.ndarray) -> np.ndarray:
        return np.array(
            [sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]]
        )

    return f


def _rossler_field(a: float, b: float, c: float) -> Callable[[np.ndarray], np.ndarray]:
    def f(x: np.ndarray) -> np.ndarray:
        return np.array([-x[1] - x[2], x[0] + a * x[1], b + x[2] * (x[0] - c)])

    return f


def _hopf_field(a: float, omega: float) -> Callable[[np.ndarray], np.ndarray]:
    """Super-critical Hopf normal form (``a > 0`` => stable limit cycle, radius √a)."""

    def f(x: np.ndarray) -> np.ndarray:
        r2 = x[0] * x[0] + x[1] * x[1]
        return np.array(
            [a * x[0] - omega * x[1] - r2 * x[0], omega * x[0] + a * x[1] - r2 * x[1]]
        )

    return f


def _henon_map(a: float, b: float) -> Callable[[np.ndarray], np.ndarray]:
    def g(x: np.ndarray) -> np.ndarray:
        return np.array([1.0 - a * x[0] * x[0] + x[1], b * x[0]])

    return g


def _logistic_map(r: float) -> Callable[[np.ndarray], np.ndarray]:
    def g(x: np.ndarray) -> np.ndarray:
        return np.clip(r * x * (1.0 - x), 0.0, 1.0)

    return g


# --------------------------------------------------------------------------- #
# F5 named -- chaotic / limit-cycle systems with analytic structure
# --------------------------------------------------------------------------- #


class LorenzSource(LiveSource):
    """Lorenz-63 chaotic attractor (canonical ``σ=10, ρ=28, β=8/3``), x-coordinate."""

    domain = "lorenz"

    def pull(self, n: int, length: int, rng: np.random.Generator) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for _ in range(n):
            j = 1.0 + 0.03 * rng.standard_normal(3)
            field = _lorenz_field(10.0 * j[0], 28.0 * j[1], (8.0 / 3.0) * j[2])
            x0 = np.array([1.0, 1.0, 20.0]) + rng.normal(0.0, 2.0, 3)
            states = _rk4_rollout(field, x0, _BURN_ODE + length - 1, 0.01)
            out.append(_finalize(states[_BURN_ODE : _BURN_ODE + length, 0]))
        return out


class RosslerSource(LiveSource):
    """Rössler chaotic attractor (``a=0.2, b=0.2, c=5.7``), x-coordinate."""

    domain = "rossler"

    def pull(self, n: int, length: int, rng: np.random.Generator) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for _ in range(n):
            j = 1.0 + 0.03 * rng.standard_normal(3)
            field = _rossler_field(0.2 * j[0], 0.2 * j[1], 5.7 * j[2])
            x0 = np.array([1.0, 1.0, 0.0]) + rng.normal(0.0, 0.5, 3)
            states = _rk4_rollout(field, x0, _BURN_ODE + length - 1, 0.04)
            out.append(_finalize(states[_BURN_ODE : _BURN_ODE + length, 0]))
        return out


class HopfSource(LiveSource):
    """Super-critical Hopf normal form: a clean nonlinear limit cycle."""

    domain = "limit_cycle"

    def pull(self, n: int, length: int, rng: np.random.Generator) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for _ in range(n):
            a = float(rng.uniform(0.3, 1.0))
            omega = float(rng.uniform(0.5, 2.5))
            field = _hopf_field(a, omega)
            x0 = rng.normal(0.0, 0.1, 2)
            states = _rk4_rollout(field, x0, _BURN_ODE + length - 1, 0.05)
            out.append(_finalize(states[_BURN_ODE : _BURN_ODE + length, 0]))
        return out


class HenonSource(LiveSource):
    """Hénon map (canonical ``a=1.4, b=0.3``), x-coordinate -- chaotic discrete time."""

    domain = "henon"

    def pull(self, n: int, length: int, rng: np.random.Generator) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for _ in range(n):
            # Jitter stays inside the attractor's existence window (a >~ 1.42 escapes).
            a = 1.4 if rng.random() < 0.3 else float(rng.uniform(1.2, 1.41))
            b = 0.3 if rng.random() < 0.3 else float(rng.uniform(0.25, 0.31))
            step = _henon_map(a, b)
            x0 = np.array([0.1 * rng.random(), 0.1 * rng.random()])
            states = _map_rollout(step, x0, _BURN_MAP + length - 1)
            out.append(_finalize(states[_BURN_MAP : _BURN_MAP + length, 0]))
        return out


class LogisticSource(LiveSource):
    """Logistic map in the chaotic band (``r ∈ [3.7, 4.0]``)."""

    domain = "logistic"

    def pull(self, n: int, length: int, rng: np.random.Generator) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for _ in range(n):
            r = 4.0 if rng.random() < 0.3 else float(rng.uniform(3.7, 4.0))
            step = _logistic_map(r)
            x0 = np.array([0.2 + 0.6 * rng.random()])
            states = _map_rollout(step, x0, _BURN_MAP + length - 1)
            out.append(_finalize(states[_BURN_MAP : _BURN_MAP + length, 0]))
        return out


# --------------------------------------------------------------------------- #
# F3 stochastic -- noise-driven processes
# --------------------------------------------------------------------------- #


class OrnsteinUhlenbeckSource(LiveSource):
    """Mean-reverting Ornstein-Uhlenbeck process (Euler-Maruyama, unit step)."""

    domain = "mean_reverting"

    def pull(self, n: int, length: int, rng: np.random.Generator) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for _ in range(n):
            theta = float(rng.uniform(0.05, 0.3))  # reversion rate (stable for dt=1)
            sigma = float(rng.uniform(0.5, 1.5))
            noise = rng.normal(0.0, 1.0, size=_BURN_SDE + length)
            x = 0.0
            series = np.empty(_BURN_SDE + length)
            for t in range(_BURN_SDE + length):
                x = x + theta * (0.0 - x) + sigma * noise[t]
                series[t] = x
            out.append(_finalize(series[_BURN_SDE:]))
        return out


class JumpDiffusionSource(LiveSource):
    """Merton-style jump-diffusion: Brownian drift + sparse Poisson jumps."""

    domain = "jump_diffusion"

    def pull(self, n: int, length: int, rng: np.random.Generator) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for _ in range(n):
            mu = float(rng.normal(0.0, 0.01))
            sigma = float(rng.uniform(0.05, 0.2))
            jump_prob = float(rng.uniform(0.01, 0.04))
            jump_scale = float(rng.uniform(1.0, 3.0))
            steps = mu + sigma * rng.normal(0.0, 1.0, size=length)
            jumps = (rng.random(length) < jump_prob) * rng.normal(0.0, jump_scale, size=length)
            out.append(_finalize(np.cumsum(steps + jumps)))
        return out


# --------------------------------------------------------------------------- #
# Mixture: the multi-domain feed
# --------------------------------------------------------------------------- #


class MixtureLiveSource(LiveSource):
    """Blend several :class:`LiveSource` components into one multi-domain feed.

    Each motif is drawn from a component chosen by the (normalised) component
    weights, and carries that component's ``domain`` label via
    :meth:`pull_labeled`. The draw and every downstream integration flow through
    the passed-in ``rng``, so the whole feed stays a pure function of the beacon
    seed -- the property the determinism / consensus tests rely on.
    """

    domain = "mixture"

    def __init__(self, components: list[tuple[LiveSource, float]]) -> None:
        if not components:
            raise ValueError("MixtureLiveSource needs at least one component")
        weights = np.array([w for _, w in components], dtype=float)
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("component weights must be non-negative and sum to > 0")
        self.sources = [s for s, _ in components]
        self.weights = weights / weights.sum()

    def pull_labeled(
        self, n: int, length: int, rng: np.random.Generator
    ) -> list[tuple[np.ndarray, str]]:
        idx = rng.choice(len(self.sources), size=n, p=self.weights)
        out: list[tuple[np.ndarray, str]] = []
        for i in idx:
            src = self.sources[int(i)]
            out.append((src.pull(1, length, rng)[0], src.domain))
        return out

    def pull(self, n: int, length: int, rng: np.random.Generator) -> list[np.ndarray]:
        return [m for m, _ in self.pull_labeled(n, length, rng)]


# The default broad feed: random-walk "market" texture plus the dynamical-system
# zoo. Weights favour the stochastic/random-walk textures the classical anchor
# handles gracefully while still guaranteeing heavy chaotic/limit-cycle coverage,
# so the feed is simultaneously broad AND keeps the validity gate meaningful.
def default_live_source() -> MixtureLiveSource:
    """A ready-made multi-domain feed spanning eight distinct data-generating processes."""
    return MixtureLiveSource(
        [
            (SyntheticLiveSource(), 0.25),  # random_walk
            (LorenzSource(), 0.15),
            (RosslerSource(), 0.12),
            (HopfSource(), 0.12),
            (HenonSource(), 0.10),
            (LogisticSource(), 0.10),
            (OrnsteinUhlenbeckSource(), 0.08),
            (JumpDiffusionSource(), 0.08),
        ]
    )


# Every concrete zoo source (excluding the random-walk stand-in already in
# ``ingest``), handy for tests and for building custom mixtures.
ZOO_SOURCES: tuple[type[LiveSource], ...] = (
    LorenzSource,
    RosslerSource,
    HopfSource,
    HenonSource,
    LogisticSource,
    OrnsteinUhlenbeckSource,
    JumpDiffusionSource,
)
