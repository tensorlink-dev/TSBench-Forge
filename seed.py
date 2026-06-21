"""Commit-reveal deterministic seeding.

Anti-gaming role
----------------
The forge LLM is non-deterministic, so it must not run live per validator --
validators would diverge and consensus would break. Instead the forge runs
*once* per epoch and commits a hashed manifest (its chosen ``GeneratorState``
plus metadata). The concrete challenges are then derived deterministically from

    seed = H(block_hash || epoch || manifest_hash)

revealed only *after* miners have committed their submissions. This closes two
gaming vectors at once:

* **Precomputation** -- miners cannot derive the seed (and therefore the exact
  challenges) before they commit, because ``block_hash`` is a future,
  unpredictable beacon value.
* **Validator divergence** -- every validator replays the identical seed and
  therefore identical challenge arrays, so a single honest reference panel run
  is reproducible by all.

Everything here is pure and deterministic: no global RNG, no wall-clock, no
network. ``rng_for`` returns a fresh ``numpy`` generator so callers never share
mutable random state.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _digest(block_hash: str, epoch: int, manifest_hash: str) -> bytes:
    """SHA-256 of the canonical ``block_hash|epoch|manifest_hash`` encoding.

    The field separator (``|``) makes the encoding unambiguous so that e.g.
    ``("ab", 1, ...)`` and ``("a", 11, ...)`` cannot collide.
    """
    payload = f"{block_hash}|{int(epoch)}|{manifest_hash}".encode()
    return hashlib.sha256(payload).digest()


def beacon_seed(block_hash: str, epoch: int, manifest_hash: str) -> int:
    """Derive a 64-bit integer seed from the commit-reveal beacon.

    Pure function of its inputs -- the cornerstone of validator consensus.
    """
    return int.from_bytes(_digest(block_hash, epoch, manifest_hash)[:8], "big")


def rng_for(block_hash: str, epoch: int, manifest_hash: str) -> np.random.Generator:
    """Return a fresh, independent ``numpy`` RNG seeded from the beacon.

    A new generator is returned on every call; no module-level mutable RNG state
    exists, so two callers with the same beacon inputs get statistically
    identical -- and reproducible -- streams without interfering with each other.
    """
    return np.random.default_rng(beacon_seed(block_hash, epoch, manifest_hash))


def manifest_hash(payload: str) -> str:
    """Hash a manifest payload to a hex string for the commit step.

    The forge commits ``manifest_hash(json.dumps(state, sorted))`` so the state
    is frozen before the seed (and thus the challenges) is revealed.
    """
    return hashlib.sha256(payload.encode()).hexdigest()
