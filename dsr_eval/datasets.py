"""DSR datasets: context window + held-out continuation + long ground-truth rollout.

Each :class:`DSRSample` is one evaluation instance for one ``(system, seed)``: the
forecaster is shown :attr:`DSRSample.context` and must, zero-shot, reproduce the
system's future. Two ground-truths come out of the same trajectory:

* :attr:`DSRSample.continuation` -- a short held-out window for point scoring (MASE,
  valid-prediction-time), where pointwise error still carries signal.
* :attr:`DSRSample.truth` -- a long (``T >= 10^4`` steps, configurable) rollout for the
  invariant / long-term metrics (``D_stsp``, ``D_H``, ``lambda``), which only converge
  when the trajectory has explored the whole attractor.

The continuation is simply the prefix of the long truth, so both come from one
contiguous integration and are mutually consistent. :attr:`DSRSample.transient` is the
number of leading steps a caller should discard from *both* the generated and the true
long rollouts before computing invariant metrics (default 10%).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from config import HORIZON
from dsr_eval.systems import ChaoticSystem, get_system, integrate

# A generous default context (a few characteristic cycles); chaotic systems need more
# history than the repo's CONTEXT_LEN=256 to pin down which orbit they are on.
DEFAULT_CONTEXT_LEN: int = 512
# Long-term metrics need the ergodic limit; 10k steps is the documented floor.
DEFAULT_LONG_LEN: int = 10000
DEFAULT_TRANSIENT_FRAC: float = 0.1


@dataclass(frozen=True)
class DSRSample:
    """One ``(system, seed)`` evaluation instance with known ground truth."""

    system: str
    seed: int
    context: np.ndarray  # (context_len,) observable shown to the forecaster
    truth: np.ndarray  # (long_len,) long ground-truth continuation
    horizon: int  # length of the short held-out window
    transient: int  # leading steps to discard before invariant metrics
    dt: float
    lyapunov_time: float | None  # 1/lambda_max in physical time, if known
    lyap_max_ref: float | None  # reference maximal Lyapunov exponent, if known
    d_ky_ref: float | None  # reference Kaplan-Yorke dimension, if known

    @property
    def continuation(self) -> np.ndarray:
        """The short held-out window for point scoring (prefix of :attr:`truth`)."""
        return self.truth[: self.horizon]


def build_sample(
    system: ChaoticSystem,
    seed: int,
    *,
    context_len: int = DEFAULT_CONTEXT_LEN,
    horizon: int = HORIZON,
    long_len: int = DEFAULT_LONG_LEN,
    transient_frac: float = DEFAULT_TRANSIENT_FRAC,
    backend: str = "auto",
) -> DSRSample:
    """Integrate one trajectory for ``system`` and split it into a :class:`DSRSample`.

    The burn-in is discarded inside :func:`~dsr_eval.systems.integrate`; ``transient``
    is the *additional* fraction of the long rollout a caller cuts before invariant
    metrics. ``seed`` perturbs the initial condition so seeds give distinct orbits.
    """
    if long_len < horizon:
        raise ValueError(f"long_len ({long_len}) must be >= horizon ({horizon})")
    rng = np.random.default_rng(seed)
    obs = integrate(system, context_len + long_len, rng=rng, backend=backend)
    context = np.asarray(obs[:context_len], dtype=float)
    truth = np.asarray(obs[context_len : context_len + long_len], dtype=float)
    return DSRSample(
        system=system.name,
        seed=int(seed),
        context=context,
        truth=truth,
        horizon=int(horizon),
        transient=int(transient_frac * long_len),
        dt=system.dt,
        lyapunov_time=system.lyapunov_time,
        lyap_max_ref=system.lyap_max_ref,
        d_ky_ref=system.d_ky_ref,
    )


def build_dataset(
    systems: Sequence[str],
    seeds: Sequence[int],
    *,
    context_len: int = DEFAULT_CONTEXT_LEN,
    horizon: int = HORIZON,
    long_len: int = DEFAULT_LONG_LEN,
    transient_frac: float = DEFAULT_TRANSIENT_FRAC,
    backend: str = "auto",
) -> dict[str, list[DSRSample]]:
    """Build samples for every ``system x seed``, grouped by system name.

    Returns ``{system_name: [DSRSample, ...]}`` (one per seed). System names may be
    built-ins (``"lorenz"``, ``"rossler"``) or ``"dysts:<Name>"``.
    """
    out: dict[str, list[DSRSample]] = {}
    for name in systems:
        system = get_system(name)
        out[name] = [
            build_sample(
                system,
                seed,
                context_len=context_len,
                horizon=horizon,
                long_len=long_len,
                transient_frac=transient_frac,
                backend=backend,
            )
            for seed in seeds
        ]
    return out


__all__ = [
    "DEFAULT_CONTEXT_LEN",
    "DEFAULT_LONG_LEN",
    "DEFAULT_TRANSIENT_FRAC",
    "DSRSample",
    "build_dataset",
    "build_sample",
]
