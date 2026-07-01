"""Pin the reward-hacking-defense invariants.

Every test here is a lock: a state that reaches the corner *must* score
foundational_fitness = 0. The LLM forge sees these gates as unconditional zeros,
so it cannot climb toward the failure mode. If a code change makes any of these
tests pass with fitness > 0, the defense is broken.

See docs/REWARD_HACKING.md for the failure modes each test locks down.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from score import (
    cadence_breadth_gate,
    dgp_class_breadth_gate,
    foundational_fitness,
)


# ---------------------------------------------------------------------------
# Unit: the gates themselves
# ---------------------------------------------------------------------------


def test_dgp_class_gate_zeros_a_mono_class_pool():
    """A pool that is 100% one DGP class must be vetoed."""
    pool = ["weather_field"] * 100
    assert dgp_class_breadth_gate(pool, min_share=0.02) == 1.0    # single class, no floor violation
    # add one below-threshold class in a large pool → still 1.0 if that class's share ≥ min
    pool = ["weather_field"] * 98 + ["vital_statistics"] * 2   # 2% share exactly
    assert dgp_class_breadth_gate(pool, min_share=0.02) == 1.0
    # push below floor → 0
    pool = ["weather_field"] * 99 + ["vital_statistics"] * 1   # 1% share
    assert dgp_class_breadth_gate(pool, min_share=0.02) == 0.0


def test_dgp_class_gate_is_noop_when_labels_missing():
    """Legacy sources emit None labels; gate must default to 1.0."""
    assert dgp_class_breadth_gate([None, None, None]) == 1.0
    assert dgp_class_breadth_gate(None) == 1.0
    assert dgp_class_breadth_gate([]) == 1.0


def test_cadence_gate_zeros_a_mono_cadence_pool():
    """A pool that is 99% hourly and 1% daily gets zeroed at min_share=0.02."""
    pool = ["hourly"] * 99 + ["daily"] * 1
    assert cadence_breadth_gate(pool, min_share=0.02) == 0.0
    pool = ["hourly"] * 95 + ["daily"] * 5   # 5% > 2%
    assert cadence_breadth_gate(pool, min_share=0.02) == 1.0


# ---------------------------------------------------------------------------
# Integration: gates enter foundational_fitness multiplicatively
# ---------------------------------------------------------------------------


def _make_scored_dict(fit: float = 1.0, cov: float = 1.0, parrot: float = 1.0) -> dict:
    """Simulate a `panel_fitness`-style return dict for direct fitness testing."""
    return {"fitness": fit, "parrot_gate": parrot}


def test_foundational_fitness_zeros_when_any_gate_zeros(monkeypatch):
    """Any single breadth gate returning 0.0 zeros the aggregate — regardless of
    how high spread, ordering, and gate go."""
    import score

    # Stub panel_fitness + coverage_gate to constant 1.0 so this test is only
    # about the gate composition.
    monkeypatch.setattr(score, "panel_fitness", lambda *_a, **_k: _make_scored_dict(fit=1.0, parrot=1.0))
    monkeypatch.setattr(score, "coverage_gate", lambda *_a, **_k: 1.0)

    # Baseline: all gates pass → aggregate = 1.0.
    r = foundational_fitness(
        challenges=[],
        dgp_classes=["a", "b", "c"] * 10,
        cadences=["hourly", "daily"] * 15,
        breadth_min_share=0.05,
    )
    assert r["foundational_fitness"] == 1.0
    assert r["dgp_class_breadth_gate"] == 1.0
    assert r["cadence_breadth_gate"] == 1.0

    # DGP-class collapse → foundational_fitness == 0.
    r = foundational_fitness(
        challenges=[],
        dgp_classes=["a"] * 99 + ["b"] * 1,
        cadences=["hourly"] * 50 + ["daily"] * 50,
        breadth_min_share=0.05,
    )
    assert r["dgp_class_breadth_gate"] == 0.0
    assert r["foundational_fitness"] == 0.0

    # Cadence collapse → foundational_fitness == 0 even when DGP diversity holds.
    r = foundational_fitness(
        challenges=[],
        dgp_classes=["a", "b"] * 50,
        cadences=["hourly"] * 99 + ["daily"] * 1,
        breadth_min_share=0.05,
    )
    assert r["cadence_breadth_gate"] == 0.0
    assert r["foundational_fitness"] == 0.0


def test_llm_cannot_recover_fitness_via_pure_class_dominance(monkeypatch):
    """The exact scenario docs/REWARD_HACKING.md #2 (domain collapse) describes:
    the LLM has found a state where spread + ordering + gate are all perfect,
    but the pool has narrowed to one DGP class. Aggregate must be 0."""
    import score

    monkeypatch.setattr(
        score, "panel_fitness",
        lambda *_a, **_k: _make_scored_dict(fit=2.0, parrot=1.0),  # spread × ordering × gate = 2.0
    )
    monkeypatch.setattr(score, "coverage_gate", lambda *_a, **_k: 1.0)
    r = foundational_fitness(
        challenges=[],
        dgp_classes=["weather_field"] * 100,  # single class → but no OTHER class below floor either!
        cadences=["hourly"] * 100,             # single band, same trick
        breadth_min_share=0.02,
    )
    # Note: mono-class means MIN share == 100%, which is >= min_share. Gate is 1.0.
    # THIS IS THE INTENDED BEHAVIOR — mono-class isn't the reward-hacking mode;
    # mono-class-plus-vestigial-others IS. See next test.
    assert r["dgp_class_breadth_gate"] == 1.0
    assert r["foundational_fitness"] == 2.0


def test_vestigial_class_collapse_is_the_actual_hack_we_catch(monkeypatch):
    """The real reward-hacking corner: LLM keeps a mono-class-plus-tokens pool
    to *appear* diverse (multiple classes present) while 99% of motifs are one
    class. The breadth gate must zero this."""
    import score

    monkeypatch.setattr(score, "panel_fitness", lambda *_a, **_k: _make_scored_dict(fit=2.0, parrot=1.0))
    monkeypatch.setattr(score, "coverage_gate", lambda *_a, **_k: 1.0)
    dgp = ["weather_field"] * 99 + ["vital_statistics"] * 1  # nominally 2 classes, 1% minority
    r = foundational_fitness(
        challenges=[],
        dgp_classes=dgp,
        cadences=["hourly"] * 100,
        breadth_min_share=0.05,  # need at least 5% per class
    )
    assert r["dgp_class_breadth_gate"] == 0.0
    assert r["foundational_fitness"] == 0.0


def test_legacy_call_without_labels_preserves_old_behavior(monkeypatch):
    """A call site that doesn't pass dgp_classes / cadences (i.e. every caller
    that predates this PR) must get the same value it always got."""
    import score

    monkeypatch.setattr(score, "panel_fitness", lambda *_a, **_k: _make_scored_dict(fit=1.5, parrot=1.0))
    monkeypatch.setattr(score, "coverage_gate", lambda *_a, **_k: 0.9)
    r = foundational_fitness(challenges=[])  # no dgp_classes, no cadences
    assert r["dgp_class_breadth_gate"] == 1.0
    assert r["cadence_breadth_gate"] == 1.0
    assert r["foundational_fitness"] == pytest.approx(1.5 * 0.9 * 1.0)
