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
