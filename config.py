"""Configuration: the forge's optimization target and global constants.

The ``GeneratorState`` is the single object the forge loop mutates. Everything
that controls *what the benchmark looks like this epoch* lives here, which is
what lets the forge treat benchmark design as an optimization problem:

    state  --(forge mutation)-->  challenges  --(panel)-->  fitness

Anti-gaming role
----------------
Centralising the knobs in one frozen dataclass is itself a defensive choice:
the concrete challenges are a pure function of ``(GeneratorState, seed)``, so
every validator that agrees on the state and the revealed seed reconstructs
byte-identical challenges (see ``seed.py``). There is no hidden mutable
configuration a miner could probe or a validator could diverge on.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# --------------------------------------------------------------------------- #
# Global constants
# --------------------------------------------------------------------------- #

CONTEXT_LEN: int = 256
"""Length of the observed context window handed to every forecaster."""

HORIZON: int = 48
"""Number of steps each forecaster must predict."""

N_CHALLENGES: int = 64
"""Default number of challenges assembled per epoch / evaluation."""

# Seasonal periods the synthetic generator may draw from. They are deliberately
# co-prime-ish and not equal to HORIZON so that ``seasonal_naive`` cannot win by
# the horizon happening to align with a single period.
SEASONAL_PERIODS: tuple[int, ...] = (12, 24, 36)

# Best -> worst quality order of the reference panel on *this benchmark's*
# distribution. ``strong`` is the independently-established validity anchor and
# MUST sit first; ``overfit`` is the generator-fitting detector and sits last
# because on genuine (non-fingerprinted) data it is a poor general forecaster.
# The forge is rewarded for producing challenges whose achieved order matches
# this list (see ``score.panel_fitness``). The ordering of the middle models is
# the empirical order that emerges on well-formed structured series; it is the
# *established* quality order for this distribution, not an arbitrary choice.
PANEL_QUALITY_ORDER: tuple[str, ...] = (
    "strong",
    "drift",
    "ewma",
    "seasonal_naive",
    "ar1",
    "overfit",
)

# Inclusive bounds the forge must respect when nudging knobs. Keeping these here
# (rather than in the loop) means the validity envelope is part of the frozen
# configuration and therefore part of consensus.
PROB_BOUNDS: tuple[float, float] = (0.0, 0.6)
SEVERITY_BOUNDS: tuple[float, float] = (0.0, 1.0)
PHI_BOUNDS: tuple[float, float] = (0.0, 0.95)


# --------------------------------------------------------------------------- #
# GeneratorState
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GeneratorState:
    """The forge's optimization target.

    Two groups of knobs:

    * **Blend ratios** (``w_synth``, ``w_spliced``, ``w_aug_live``) decide how
      often each challenge *mode* is drawn. They are renormalised to sum to 1 by
      :meth:`normalized`. Keeping all three strictly positive is a load-bearing
      anti-gaming property: pure synthetic data is pre-fittable, pure live data
      is thin and memorisable, so the benchmark must always mix.

    * **Difficulty priors** shape how hard / structured the series are:
        - ``changepoint_prob``  -- probability of a trend break (defeats models
          that assume a single global slope, e.g. ``drift``).
        - ``regime_switch_prob`` -- probability the noise process switches
          regime (defeats stationarity assumptions).
        - ``aug_severity`` -- strength of augmentations applied to live motifs
          (defeats exact-match memorisation).
        - ``noise_ar_phi`` -- persistence of the AR noise (rewards models that
          actually model autocorrelation; this is also the structure the
          ``overfit`` detector exploits).

    The dataclass is frozen: mutations always produce a *new* state via
    :func:`dataclasses.replace`, never in-place edits. This keeps consensus
    intact -- a state, once hashed into a manifest, can never silently change.
    """

    # Blend ratios (renormalised on use).
    w_synth: float = 0.34
    w_spliced: float = 0.33
    w_aug_live: float = 0.33

    # Difficulty priors.
    changepoint_prob: float = 0.15
    regime_switch_prob: float = 0.15
    aug_severity: float = 0.3
    noise_ar_phi: float = 0.4

    def normalized(self) -> GeneratorState:
        """Return a copy whose blend ratios are non-negative and sum to 1.

        If all three weights are non-positive we fall back to an equal blend so
        the generator can never collapse to "no challenges".
        """
        w = [max(0.0, self.w_synth), max(0.0, self.w_spliced), max(0.0, self.w_aug_live)]
        total = sum(w)
        if total <= 0.0:
            w = [1 / 3, 1 / 3, 1 / 3]
            total = 1.0
        return replace(
            self,
            w_synth=w[0] / total,
            w_spliced=w[1] / total,
            w_aug_live=w[2] / total,
        )

    def blend_weights(self) -> tuple[float, float, float]:
        """``(w_synth, w_spliced, w_aug_live)`` after normalisation."""
        n = self.normalized()
        return (n.w_synth, n.w_spliced, n.w_aug_live)

    def clamped(self) -> GeneratorState:
        """Project the difficulty priors back into their valid envelope.

        The forge proposes free-form nudges; this guarantees the result is still
        a legal, consensus-safe state regardless of what the (LLM) proposer did.
        """

        def clip(x: float, lo: float, hi: float) -> float:
            return float(min(hi, max(lo, x)))

        return replace(
            self,
            changepoint_prob=clip(self.changepoint_prob, *PROB_BOUNDS),
            regime_switch_prob=clip(self.regime_switch_prob, *PROB_BOUNDS),
            aug_severity=clip(self.aug_severity, *SEVERITY_BOUNDS),
            noise_ar_phi=clip(self.noise_ar_phi, *PHI_BOUNDS),
        )


# A deliberately WEAK starting point used by the demo and tests: almost no
# structure for the strong model to exploit, so the benchmark barely
# discriminates. The forge should measurably climb away from here.
WEAK_STATE: GeneratorState = GeneratorState(
    w_synth=0.85,
    w_spliced=0.10,
    w_aug_live=0.05,
    changepoint_prob=0.02,
    regime_switch_prob=0.02,
    aug_severity=0.05,
    noise_ar_phi=0.05,
)
