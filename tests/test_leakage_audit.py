"""Contamination-resistance: default buffer, t_now barrier, behavioural audit.

Acceptance: the production buffer dedups (and gates by as-of when asked); the
global barrier is the max cutoff and rejects pre-cutoff points; the memorization
probe flags a model that is much better on seen data and clears an honest one;
feed novelty is low for re-served data and high for genuinely new data.
"""

from __future__ import annotations

import numpy as np

from config import CONTEXT_LEN, HORIZON
from feeds import DatedLiveSource, DedupFreshBuffer
from generate import Challenge
from ingest import SyntheticLiveSource
from leakage_audit import (
    assert_post_cutoff,
    default_fresh_buffer,
    feed_novelty,
    global_t_now,
    memorization_probe,
)


class _DatedSyntheticSource(DatedLiveSource):
    """Synthetic motifs stamped with timestamps straddling a cutoff at t=100."""

    domain = "dated"

    def pull_dated(self, n, length, rng):
        base = SyntheticLiveSource()
        motifs = base.pull(n, length, rng)
        # Alternate timestamps below/above 100 so the as-of gate has work to do.
        return [(m, self.domain, float(50 + 100 * (i % 2))) for i, m in enumerate(motifs)]


def test_default_fresh_buffer_is_dedup() -> None:
    buf = default_fresh_buffer(SyntheticLiveSource(), pool_size=32, motif_len=512)
    assert isinstance(buf, DedupFreshBuffer)


def test_default_fresh_buffer_requires_dated_for_commit_time() -> None:
    try:
        default_fresh_buffer(SyntheticLiveSource(), commit_time=100.0)
    except TypeError:
        return
    raise AssertionError("expected TypeError for as-of barrier on undated source")


def test_as_of_barrier_only_admits_post_cutoff() -> None:
    src = _DatedSyntheticSource()
    gated = default_fresh_buffer(src, commit_time=100.0, pool_size=8, motif_len=256)
    # The wrapped source must only yield motifs stamped after the cutoff.
    dated = gated.source.pull_dated(8, 256, np.random.default_rng(0))
    assert dated  # some survive
    assert all(ts > 100.0 for _, _, ts in dated)


def test_global_t_now_and_assert() -> None:
    assert global_t_now({"a": 10.0, "b": 30.0, "c": 5.0}) == 30.0
    ok = assert_post_cutoff([31.0, 40.0], t_now=30.0)
    assert ok["ok"] and ok["n_stale"] == 0
    bad = assert_post_cutoff([29.0, 40.0], t_now=30.0)
    assert not bad["ok"] and bad["n_stale"] == 1


def _challenges_with_leak(seed: int, leak: bool) -> list[Challenge]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(16):
        series = np.cumsum(rng.normal(0, 1, CONTEXT_LEN + HORIZON)) + 10.0
        ctx, truth = series[:CONTEXT_LEN], series[CONTEXT_LEN:]
        meta = {"leak": truth.copy()} if leak else {}
        out.append(Challenge(context=ctx, truth=truth, mode="x", meta=meta))
    return out


def test_memorization_probe_flags_memoriser_clears_honest() -> None:
    def honest(context, meta=None):
        return np.full(HORIZON, float(context[-1]))  # persistence, ignores meta

    def memoriser(context, meta=None):
        if meta and meta.get("leak") is not None:
            return np.asarray(meta["leak"], dtype=float)  # returns the answer it "saw"
        return np.full(HORIZON, float(context[-1]))

    fresh = _challenges_with_leak(1, leak=False)
    repeated = _challenges_with_leak(1, leak=True)  # same series, but "seen" (leak present)

    honest_rep = memorization_probe(honest, fresh, repeated)
    assert abs(honest_rep.memorization_gap) < 0.1
    assert honest_rep.suspicious is False

    mem_rep = memorization_probe(memoriser, fresh, repeated)
    assert mem_rep.memorization_gap > 0.5
    assert mem_rep.suspicious is True


def test_feed_novelty_low_for_reserved_high_for_new() -> None:
    buf = default_fresh_buffer(SyntheticLiveSource(), pool_size=24, motif_len=512)
    buf.refresh(np.random.default_rng(0))
    reserved = [m for m, _ in buf._pool]  # exactly what was just admitted
    low = feed_novelty(buf, reserved)
    assert low["novelty_rate"] < 0.2
    # Distinct deterministic shapes the random-walk feed never produced.
    t = np.linspace(0, 1, 512)
    new = [np.sin(2 * np.pi * (k + 2) * t) for k in range(10)]
    high = feed_novelty(buf, new)
    assert high["novelty_rate"] > 0.6
