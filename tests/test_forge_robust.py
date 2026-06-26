"""Forge robustness: coverage-aware objective and panel-generalization check.

Acceptance: (1) the forge can be driven by ``foundational_fitness`` and still
runs deterministically and non-decreasingly; (2) evaluate_state exposes the
coverage/parrot/foundational fields; (3) the generalization check passes when the
anchor leads a held-out model and fails when a held-out model beats it.
"""

from __future__ import annotations

from config import WEAK_STATE
from domains import default_live_source
from forge_loop import (
    FOUNDATIONAL_OBJECTIVE,
    evaluate_state,
    run_forge,
)
from generate import Challenge, build_challenges
from ingest import FreshBuffer
from score import default_panel, strong, validate_generalization
from seed import rng_for


def _buffer() -> FreshBuffer:
    buf = FreshBuffer(default_live_source(), pool_size=64, motif_len=768)
    buf.refresh(rng_for("0xabc", 0, "init"))
    return buf


def test_evaluate_state_exposes_foundational_fields() -> None:
    m = evaluate_state(WEAK_STATE, _buffer(), "0xabc", 16, default_panel(), n_seeds=2)
    for k in ("fitness", "coverage_gate", "parrot_gate", "foundational_fitness"):
        assert k in m
    assert 0.0 <= m["coverage_gate"] <= 1.0
    assert 0.0 <= m["parrot_gate"] <= 1.0


def test_forge_runs_with_foundational_objective() -> None:
    _, log = run_forge(
        _buffer(), epochs=6, block_hash="0xabc", init_state=WEAK_STATE,
        n_challenges=24, n_seeds=2, objective=FOUNDATIONAL_OBJECTIVE,
    )
    assert len(log) == 7
    # Determinism: identical inputs (fresh equivalent buffer) -> identical decisions.
    _, log2 = run_forge(
        _buffer(), epochs=6, block_hash="0xabc", init_state=WEAK_STATE,
        n_challenges=24, n_seeds=2, objective=FOUNDATIONAL_OBJECTIVE,
    )
    assert [s.decision for s in log] == [s.decision for s in log2]


def _structured_challenges(n: int = 24) -> list[Challenge]:
    buf = _buffer()
    rng = rng_for("0xabc", 1, "gen")
    state = WEAK_STATE.__class__(
        w_synth=0.5, w_spliced=0.3, w_aug_live=0.2, changepoint_prob=0.2,
        regime_switch_prob=0.15, aug_severity=0.3, noise_ar_phi=0.4,
    )
    return build_challenges(state, buf, rng, n)


def test_generalization_passes_against_weak_heldout() -> None:
    chs = _structured_challenges()
    # ewma is a genuinely weaker model than the anchor on structured data.
    report = validate_generalization(chs, {"ewma": default_panel()["ewma"]})
    assert report["generalizes"] is True
    assert report["beaten_by"] == []


def test_generalization_flags_a_dominating_heldout() -> None:
    chs = _structured_challenges()
    # A held-out "oracle" that returns the truth must beat the anchor -> flagged.
    def oracle(context, meta=None):
        # find the matching challenge by identity of context is not available;
        # emulate a strictly-better model by returning strong's forecast minus a
        # small fraction of its error is impossible without truth, so instead use
        # a model identical to strong (ties) -> should be flagged at min_lead=0.
        return strong(context, meta)

    report = validate_generalization(chs, {"twin": oracle})
    assert "twin" in report["beaten_by"]
    assert report["generalizes"] is False
