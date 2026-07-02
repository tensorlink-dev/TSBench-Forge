"""Discrimination fitness: a good challenge set separates forecasters.

``panel_fitness`` scores a set by ``spread`` — how far apart the panel's errors
are. A structured set (models genuinely differ) scores high; a set where every
model does about equally well or equally badly (e.g. pure noise) scores low.
There is no panel-ordering / anchor gate anymore.
"""

from __future__ import annotations

import numpy as np

from challenges import Challenge
from conftest import structured_challenges
from config import CONTEXT_LEN, HORIZON
from score import panel_fitness


def _noise_challenges(n: int = 48) -> list[Challenge]:
    """Pure white noise: nothing to forecast, so all models cluster together."""
    rng = np.random.default_rng(0)
    out: list[Challenge] = []
    for _ in range(n):
        s = rng.normal(0.0, 1.0, size=CONTEXT_LEN + HORIZON)
        out.append(Challenge(context=s[:CONTEXT_LEN], truth=s[CONTEXT_LEN:], mode="noise", meta={}))
    return out


def test_fitness_is_spread_and_has_no_ordering_term() -> None:
    res = panel_fitness(structured_challenges(48, seed=1))
    assert res["fitness"] == res["spread"]          # fitness == discrimination
    assert "ordering" not in res                    # the anchor/ordering gate is gone


def test_discriminating_set_scores_positive() -> None:
    res = panel_fitness(structured_challenges(96, seed=3))
    assert res["spread"] > 0.0 and res["fitness"] > 0.0


def test_structured_separates_more_than_noise() -> None:
    # A structured set separates the panel more strongly than pure noise, where no
    # model can do better than another — exactly what "discrimination" should mean.
    structured = panel_fitness(structured_challenges(96, seed=5))["spread"]
    noise = panel_fitness(_noise_challenges(96))["spread"]
    assert structured > noise
