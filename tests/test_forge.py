"""Forge loop: from a weak state, fitness measurably increases with KEEPs."""

from __future__ import annotations

import numpy as np

from config import PROB_BOUNDS, WEAK_STATE
from forge_loop import ForgeStep, propose_mutation, run_forge
from ingest import FreshBuffer, SyntheticLiveSource
from seed import rng_for


def _weak_step() -> ForgeStep:
    return ForgeStep(
        epoch=0,
        knob=None,
        decision="init",
        fitness=0.05,
        spread=1.0,
        ordering=0.3,
        difficulty=2.0,
        gate=0.15,
        state=WEAK_STATE,
    )


def test_forge_climbs_from_weak_state() -> None:
    buffer = FreshBuffer(SyntheticLiveSource(), pool_size=64, motif_len=768)
    buffer.refresh(np.random.default_rng(1))

    final, log = run_forge(
        buffer,
        epochs=16,
        block_hash="forge-test",
        init_state=WEAK_STATE,
        n_challenges=48,
        n_seeds=3,
    )

    init_fit, final_fit = log[0].fitness, log[-1].fitness
    keeps = sum(1 for s in log if s.decision == "keep")

    assert keeps >= 1
    assert final_fit > init_fit
    # The climb here comes from lifting the generator-fitting validity gate by
    # rebalancing away from synthetic-heavy data.
    assert log[-1].gate > log[0].gate
    assert final.normalized().w_synth < WEAK_STATE.w_synth


def test_forge_log_fitness_is_monotone_nondecreasing() -> None:
    buffer = FreshBuffer(SyntheticLiveSource(), pool_size=48, motif_len=768)
    buffer.refresh(np.random.default_rng(2))
    _, log = run_forge(
        buffer, epochs=10, block_hash="mono", init_state=WEAK_STATE, n_challenges=32, n_seeds=3
    )
    fits = [s.fitness for s in log]
    assert all(b >= a - 1e-9 for a, b in zip(fits[:-1], fits[1:], strict=True))


def test_propose_mutation_returns_valid_one_knob_move() -> None:
    new_state, knob = propose_mutation(WEAK_STATE, [_weak_step()], rng_for("p", 1, "m"))
    assert knob in {
        "w_synth",
        "w_spliced",
        "w_aug_live",
        "changepoint_prob",
        "regime_switch_prob",
        "aug_severity",
        "noise_ar_phi",
    }
    # Result is always a legal, consensus-safe state.
    weights = new_state.blend_weights()
    assert abs(sum(weights) - 1.0) < 1e-9
    assert all(w >= 0 for w in weights)
    assert PROB_BOUNDS[0] <= new_state.changepoint_prob <= PROB_BOUNDS[1]


def test_weak_state_gate_is_low() -> None:
    # Sanity: the weak (synthetic-heavy) state is exactly what the gate should
    # punish, so it must not already look valid.
    from generate import build_challenges
    from score import panel_fitness

    buffer = FreshBuffer(SyntheticLiveSource(), pool_size=48, motif_len=768)
    res = panel_fitness(build_challenges(WEAK_STATE, buffer, rng_for("w", 0, "m"), 48))
    assert res["gate"] < 0.6
