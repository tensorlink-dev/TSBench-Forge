"""Determinism / consensus: identical beacon inputs -> identical challenges.

This is the property every validator relies on to replay the same challenge set
from the revealed seed.
"""

from __future__ import annotations

import numpy as np

from config import WEAK_STATE
from generate import build_challenges
from ingest import FreshBuffer, SyntheticLiveSource
from seed import beacon_seed, manifest_hash, rng_for


def _buffer() -> FreshBuffer:
    return FreshBuffer(SyntheticLiveSource(), pool_size=32, motif_len=512)


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
    ch1 = build_challenges(WEAK_STATE, _buffer(), rng_for(block, epoch, man), 16)
    ch2 = build_challenges(WEAK_STATE, _buffer(), rng_for(block, epoch, man), 16)

    assert len(ch1) == len(ch2) == 16
    for a, b in zip(ch1, ch2, strict=True):
        assert a.mode == b.mode
        assert np.array_equal(a.context, b.context)
        assert np.array_equal(a.truth, b.truth)
        # The synthetic oracle (consumed by the detector) must also match.
        oa, ob = a.meta.get("oracle"), b.meta.get("oracle")
        if oa is None or ob is None:
            assert oa is ob
        else:
            assert np.array_equal(oa, ob)


def test_different_beacons_yield_different_challenges() -> None:
    man = manifest_hash("p")
    ch1 = build_challenges(WEAK_STATE, _buffer(), rng_for("blkA", 1, man), 8)
    ch2 = build_challenges(WEAK_STATE, _buffer(), rng_for("blkB", 1, man), 8)
    differs = any(
        not np.array_equal(a.context, b.context) for a, b in zip(ch1, ch2, strict=True)
    )
    assert differs
