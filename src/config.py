"""Configuration: the benchmark's global constants.

The benchmark forecasts real series drawn from the live catalog. The concrete
challenges are a pure function of ``(pool, revealed seed)`` (see ``challenges.py``
and ``seed.py``), so every validator that agrees on the pool snapshot and the
revealed seed reconstructs byte-identical challenges. There is no hidden mutable
configuration a miner could probe or a validator could diverge on.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Global constants
# --------------------------------------------------------------------------- #

CONTEXT_LEN: int = 256
"""Fallback context length (used when a challenge's cadence has no profile,
and by fixed-shape test/synthetic sets)."""

HORIZON: int = 48
"""Fallback horizon (same role as CONTEXT_LEN)."""

# Per-cadence challenge shapes, keyed by the FREQ_BAND cadence label.
# Grounded in the 2026-07-03 rank-stability experiment (verification_log.md):
# ctx 512 cuts hourly-band error ~35% vs 256 and keeps the context-parrot floor
# honest (parrot only dominates at 1024); daily horizons beyond ~2 weeks are
# noise-dominated (parrot ranked #1 in 8/9 daily cells at h>=24). Horizons track
# the operational loop: daily cutoff, forecast O(12-24h) ahead at native cadence.
PROFILES: dict[str, tuple[int, int]] = {
    "sub-min": (512, 48),
    "few-min": (512, 48),
    "half-hour": (512, 48),
    "hourly": (512, 24),
    "daily": (256, 14),
    "weekly": (128, 8),
    "monthly": (128, 12),
    "quarterly": (64, 8),
    "yearly": (64, 8),
}

# Season length per ISO-8601 sampling interval, the gluonts/GIFT-Eval convention:
# one natural cycle in steps (daily cycle for sub-daily data; the calendar cycle
# above that). Frequencies absent here score with m=1 (non-seasonal MASE).
# Shared by the evaluator (MASE scaling) and the panel's seasonality search.
FREQ_SEASONALITY: dict[str, int] = {
    "PT30S": 120,   # 1 hour
    "PT1M": 1440,   # 1 day
    "PT2M30S": 576,
    "PT5M": 288,
    "PT6M": 240,
    "PT10M": 144,
    "PT15M": 96,
    "PT30M": 48,
    "PT1H": 24,
    "PT8H": 3,
    "P1D": 1,
    "P1W": 1,
    "P1M": 12,
    "P1Q": 4,
    "P1Y": 1,
}

N_CHALLENGES: int = 64
"""Default number of challenges assembled per epoch / evaluation."""

# Seasonal periods the panel models search over when estimating seasonality on a
# context. They are deliberately co-prime-ish and not equal to HORIZON so that
# ``seasonal_naive`` cannot win by the horizon happening to align with a period.
SEASONAL_PERIODS: tuple[int, ...] = (12, 24, 36)

UNSEEN_WEIGHT_FLOOR: float = 0.25
"""Scoring weight of a fully-historical challenge (its truth predates the daily
cutoff, so a pretrained model could in principle have memorised it). A challenge
whose truth is entirely post-cutoff weighs 1.0; in between the weight scales
linearly with ``unseen_frac``. The floor keeps historical challenges contributing
breadth (a partial hold) without letting memorisable data dominate the score."""
