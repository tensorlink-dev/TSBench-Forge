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
"""Length of the observed context window handed to every forecaster."""

HORIZON: int = 48
"""Number of steps each forecaster must predict."""

N_CHALLENGES: int = 64
"""Default number of challenges assembled per epoch / evaluation."""

# Seasonal periods the panel models search over when estimating seasonality on a
# context. They are deliberately co-prime-ish and not equal to HORIZON so that
# ``seasonal_naive`` cannot win by the horizon happening to align with a period.
SEASONAL_PERIODS: tuple[int, ...] = (12, 24, 36)

# Best -> worst quality order of the reference panel on real series. ``strong`` is
# the independently-established validity anchor and MUST sit first. A challenge
# set "counts" only when the achieved ranking matches this order (see
# ``score.panel_fitness``): if a naive baseline beats ``strong``, the set is
# measuring an artifact and its validity term goes negative. The ordering of the
# middle models is the empirical order that emerges on well-formed structured
# series; it is the *established* quality order, not an arbitrary choice.
PANEL_QUALITY_ORDER: tuple[str, ...] = (
    "strong",
    "drift",
    "ewma",
    "seasonal_naive",
    "ar1",
)
