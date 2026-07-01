"""Shared test helpers.

The production code has no synthetic generator anymore — the benchmark forecasts
real data only. But some *unit* tests exercise infrastructure (feed dedup, as-of
gating, the parrot gate, the DSR chaos metrics) that needs a cheap, deterministic
source without reading parquet. Those helpers live here, in the test tree, so
they never ship in ``src``:

* :class:`RandomWalkSource` — a tiny stochastic ``LiveSource`` stand-in (the old
  offline source), for feed / leakage / dedup unit tests.
* :func:`live_buffer` — a ``FreshBuffer`` over the committed real-data fixture
  (``tests/fixtures/sources_data``), for tests that need genuine catalog breadth.
* :func:`structured_challenges` — constructed trend+seasonal+AR challenges where
  the classical ``strong`` anchor genuinely leads, for testing the validity-gate
  *logic* independently of anchor quality on raw real data.
* :func:`lorenz_motif` — a Lorenz-63 observable for the chaos-metric tests.
"""

from __future__ import annotations

import os

import numpy as np

from challenges import Challenge
from config import CONTEXT_LEN, HORIZON, SEASONAL_PERIODS
from ingest import FreshBuffer, LiveSource
from scraped_source import ScrapedLiveSource

_HERE = os.path.dirname(__file__)
CATALOG = os.path.join(_HERE, os.pardir, "src", "sources", "sources.yaml")
FIXTURE_DATA = os.path.join(_HERE, "fixtures", "sources_data")


class RandomWalkSource(LiveSource):
    """Random walk + Poisson-like jumps + multiplicative noise (test-only).

    A cheap, deterministic ``LiveSource`` for unit-testing feed machinery that
    must not depend on parquet fixtures. Every draw is a pure function of ``rng``.
    """

    domain = "random_walk"

    def __init__(
        self,
        vol: float = 0.1,
        jump_prob: float = 0.03,
        jump_scale: float = 1.5,
        drift_scale: float = 0.01,
        mult_noise: float = 0.02,
        base: float = 10.0,
    ) -> None:
        self.vol = vol
        self.jump_prob = jump_prob
        self.jump_scale = jump_scale
        self.drift_scale = drift_scale
        self.mult_noise = mult_noise
        self.base = base

    def pull(self, n: int, length: int, rng: np.random.Generator) -> list[np.ndarray]:
        motifs: list[np.ndarray] = []
        for _ in range(n):
            drift = rng.normal(0.0, self.drift_scale)
            steps = rng.normal(0.0, self.vol, size=length)
            jumps = (rng.random(length) < self.jump_prob) * rng.normal(
                0.0, self.jump_scale, size=length
            )
            walk = np.cumsum(drift + steps + jumps)
            mult = np.exp(rng.normal(0.0, self.mult_noise, size=length))
            motifs.append((self.base + walk) * mult)
        return motifs


def live_buffer(pool_size: int = 48, motif_len: int = 384, seed: int = 0xC0FFEE) -> FreshBuffer:
    """A ``FreshBuffer`` over the committed real-data fixture, refreshed and ready."""
    source = ScrapedLiveSource(CATALOG, FIXTURE_DATA, min_series_length=motif_len)
    buf = FreshBuffer(source, pool_size=pool_size, motif_len=motif_len)
    buf.refresh(np.random.default_rng(seed))
    return buf


def structured_challenges(n: int = 64, seed: int = 0) -> list[Challenge]:
    """Constructed trend+seasonal+AR(1) challenges where ``strong`` genuinely leads.

    Used to test the validity-gate *logic* (does the panel order correctly, does a
    hollow anchor get flagged) without depending on the numpy anchor's quality on
    raw real data. The series are z-scored by context so they match the live path.
    """
    rng = np.random.default_rng(seed)
    p = SEASONAL_PERIODS[1]
    out: list[Challenge] = []
    for _ in range(n):
        t = np.arange(CONTEXT_LEN + HORIZON)
        base = float(rng.normal(0.0, 3.0))
        slope = float(rng.uniform(0.02, 0.06)) * (1 if rng.random() < 0.5 else -1)
        amp = float(rng.uniform(1.0, 2.0))
        phase = float(rng.uniform(0, 2 * np.pi))
        series = base + slope * t + amp * np.sin(2 * np.pi * t / p + phase)
        # mild AR(1) noise
        noise = np.zeros(t.size)
        for k in range(1, t.size):
            noise[k] = 0.5 * noise[k - 1] + rng.normal(0.0, 0.3)
        series = series + noise
        ctx = series[:CONTEXT_LEN]
        mu, sd = float(ctx.mean()), float(ctx.std()) or 1.0
        series = (series - mu) / sd
        out.append(
            Challenge(
                context=series[:CONTEXT_LEN],
                truth=series[CONTEXT_LEN:],
                mode="structured",
                meta={"domain": "structured", "oracle": None},
            )
        )
    return out


def lorenz_motif(length: int, seed: int = 0) -> np.ndarray:
    """A z-scored Lorenz-63 x-coordinate observable (test-only chaos generator)."""
    rng = np.random.default_rng(seed)
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    dt, burn = 0.01, 600
    x = np.array([1.0, 1.0, 20.0]) + rng.normal(0.0, 2.0, 3)

    def field(s: np.ndarray) -> np.ndarray:
        return np.array(
            [sigma * (s[1] - s[0]), s[0] * (rho - s[2]) - s[1], s[0] * s[1] - beta * s[2]]
        )

    obs = np.empty(burn + length)
    for i in range(burn + length):
        k1 = field(x)
        k2 = field(x + 0.5 * dt * k1)
        k3 = field(x + 0.5 * dt * k2)
        k4 = field(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        obs[i] = x[0]
    w = obs[burn:]
    return (w - w.mean()) / (w.std() or 1.0)
