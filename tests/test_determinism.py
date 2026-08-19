"""Determinism / consensus: identical beacon inputs -> identical challenges.

This is the property every validator relies on to replay the same challenge set
from the revealed seed.
"""

from __future__ import annotations

import numpy as np

from challenges import build_live_challenges
from conftest import live_buffer
from seed import beacon_seed, manifest_hash, rng_for


def _buffer():
    return live_buffer(pool_size=32, motif_len=384)


def test_beacon_seed_is_pure_and_distinct() -> None:
    assert beacon_seed("blk", 3, "m") == beacon_seed("blk", 3, "m")
    assert beacon_seed("blk", 3, "m") != beacon_seed("blk", 4, "m")
    assert beacon_seed("blk", 3, "m") != beacon_seed("blk", 3, "m2")
    # The field separator prevents (block, epoch) ambiguity collisions.
    assert beacon_seed("a", 11, "m") != beacon_seed("a1", 1, "m")
    assert isinstance(beacon_seed("blk", 0, "m"), int)


def test_rng_for_is_reproducible_and_independent() -> None:
    a = rng_for("blk", 1, "m").standard_normal(16)
    b = rng_for("blk", 1, "m").standard_normal(16)
    assert np.array_equal(a, b)
    c = rng_for("blk", 2, "m").standard_normal(16)
    assert not np.array_equal(a, c)


def test_challenges_byte_identical_across_runs() -> None:
    block, epoch, man = "0xabc", 7, manifest_hash("payload")
    ch1 = build_live_challenges(_buffer(), rng_for(block, epoch, man), 16)
    ch2 = build_live_challenges(_buffer(), rng_for(block, epoch, man), 16)

    assert len(ch1) == len(ch2) == 16
    for a, b in zip(ch1, ch2, strict=True):
        assert a.mode == b.mode
        assert np.array_equal(a.context, b.context)
        assert np.array_equal(a.truth, b.truth)
        assert a.meta.get("domain") == b.meta.get("domain")


def test_different_beacons_yield_different_challenges() -> None:
    man = manifest_hash("p")
    ch1 = build_live_challenges(_buffer(), rng_for("blkA", 1, man), 8)
    ch2 = build_live_challenges(_buffer(), rng_for("blkB", 1, man), 8)
    differs = any(
        not np.array_equal(a.context, b.context) for a, b in zip(ch1, ch2, strict=True)
    )
    assert differs


# ── flat contexts are redrawn, and the redraw stays deterministic ───────────


def _flat_and_live_buffer(n_flat: int, n_live: int, motif_len: int = 96):
    """A FreshBuffer whose pool is part constant, part real.

    Built directly rather than through ScrapedLiveSource: the point here is what
    build_live_challenges does with a flat window that reaches it, not how one
    gets into the pool.
    """
    from ingest import FreshBuffer, MotifMeta

    rng = np.random.default_rng(0)
    pool = [MotifMeta(motif=np.full(motif_len, 5.0), domain="nature",
                      dgp_class="x", cadence="hourly", source_id="flat")
            for _ in range(n_flat)]
    pool += [MotifMeta(motif=rng.normal(size=motif_len).cumsum(), domain="nature",
                       dgp_class="x", cadence="hourly", source_id="live")
             for _ in range(n_live)]

    class _Pool:
        def __init__(self, p): self._p = p
        def ensure(self, rng): pass
        def refresh(self, rng): pass
        @property
        def motif_len(self): return motif_len
        def sample_meta(self, k, length, rng):
            return [self._p[int(rng.integers(0, len(self._p)))] for _ in range(k)]
    return _Pool(pool)


def test_a_flat_context_is_redrawn() -> None:
    """The source refuses a wholly-constant motif, but the profile cut takes
    only the fresh tail and that tail can be flat where the motif was not.
    Measured 2026-08-18: 4 of 64 challenges, three with a constant truth too."""
    buf = _flat_and_live_buffer(n_flat=8, n_live=8)
    ch = build_live_challenges(buf, np.random.default_rng(1), 40, augment=False)
    flat = [c for c in ch if float(np.std(c.context)) == 0.0]
    assert not flat, f"{len(flat)}/{len(ch)} challenges have a constant context"
    assert {c.meta["source_id"] for c in ch} == {"live"}


def test_the_redraw_is_still_byte_reproducible() -> None:
    """Redrawing consumes more of the child stream, but it is the SAME child
    stream — so replay from the revealed seed still matches, which is the
    property every validator depends on."""
    a = build_live_challenges(_flat_and_live_buffer(6, 6),
                              np.random.default_rng(9), 24, augment=False)
    b = build_live_challenges(_flat_and_live_buffer(6, 6),
                              np.random.default_rng(9), 24, augment=False)
    assert all(np.array_equal(x.context, y.context)
               and np.array_equal(x.truth, y.truth) for x, y in zip(a, b))


def test_an_all_flat_pool_still_returns_the_requested_count() -> None:
    """n_challenges is a contract. If every candidate is flat the bound is hit
    and the last draw is kept, rather than returning short or looping."""
    ch = build_live_challenges(_flat_and_live_buffer(8, 0),
                               np.random.default_rng(3), 12, augment=False)
    assert len(ch) == 12
