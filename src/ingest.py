"""Live data ingestion: the freshness layer.

Anti-gaming role
----------------
Memorisation is the easiest way to "game" a forecasting benchmark: if the test
series existed at commit time, a miner can simply look up the answer. The live
layer closes that vector by sourcing real motifs that **did not exist when the
miner committed**. The benchmark only ever uses data timestamped *after* the
commit point.

``LiveSource`` is an abstraction with a synthetic stand-in so the demo runs
offline; real adapters (a market feed, a sensor stream, an energy load API, ...)
are a documented extension point, and ``domains.py`` ships a dependency-free
*multi-domain* feed (a zoo of dynamical systems) for breadth. Two properties are
load-bearing:

* **Different texture.** ``SyntheticLiveSource`` produces random-walk + jump +
  multiplicative-noise series -- deliberately *unlike* the trend/seasonal/AR
  process in ``generate.py``. If the "live" motifs shared the synthetic texture,
  a miner who fit the synthetic process would also fit the motifs, and the
  splice defence would be hollow.
* **Seeded sampling.** ``FreshBuffer.sample_motifs`` draws from a fixed pool
  using the beacon-derived RNG, so every validator splices the *same* motifs and
  consensus holds.

Production note
---------------
A real adapter MUST use as-of / vintage snapshots. Many vendors silently revise
history; training or scoring on revised values leaks future information back into
the context window. Pull only points timestamped after the commit beacon, and
**quarantine + dedup** every pull (drop near-duplicates of anything previously
served) so a finite feed cannot be memorised across epochs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotifMeta:
    """Rich metadata for one motif in the freshness pool.

    ``domain`` is the coarse tag every source has emitted since the beginning
    (kept for backward compatibility with the ``(motif, domain)`` tuples that
    ``generate.py`` unpacks). ``dgp_class`` and ``cadence`` are new fields added
    for the reward-hacking-defense breadth gates in ``score.py`` — a
    ``ScrapedLiveSource`` fills them from ``sources.yaml``; legacy sources
    default them to ``None``, in which case the breadth gates degrade to a
    coverage-by-domain fallback.
    """

    motif: np.ndarray
    domain: str
    dgp_class: str | None = None
    cadence: str | None = None
    source_id: str | None = None


class LiveSource(ABC):
    """Abstract source of real time-series motifs.

    A concrete adapter wraps an external feed. The only required contract is
    :meth:`pull`; everything downstream (buffering, splicing) is texture-agnostic.

    Each source advertises a ``domain`` label naming the data-generating process
    (or real-world feed) it represents. The label rides along with every motif
    (see :meth:`pull_labeled` and ``FreshBuffer``) so the scorer can measure and
    stratify by *coverage* -- the property a foundation-model benchmark needs but
    a single-source feed cannot provide. ``domains.MixtureLiveSource`` overrides
    :meth:`pull_labeled` to vary the label per motif.
    """

    domain: str = "live"

    @abstractmethod
    def pull(self, n: int, length: int, rng: np.random.Generator) -> list[np.ndarray]:
        """Return ``n`` real-valued motifs, each of the given ``length``.

        ``rng`` makes the pull reproducible for the demo; a real adapter would
        instead return the genuinely-fresh window and ignore ``rng`` for content
        (still using it only for any sampling choices).
        """
        raise NotImplementedError

    def pull_labeled(
        self, n: int, length: int, rng: np.random.Generator
    ) -> list[tuple[np.ndarray, str]]:
        """Return ``n`` ``(motif, domain)`` pairs.

        The default tags every motif with this source's :attr:`domain`; a mixture
        feed overrides this to draw a per-motif domain. Defining it in terms of
        :meth:`pull` keeps single-domain adapters a one-method implementation.
        """
        return [(motif, self.domain) for motif in self.pull(n, length, rng)]

    def pull_meta(
        self, n: int, length: int, rng: np.random.Generator
    ) -> list[MotifMeta]:
        """Rich labeled pull returning ``MotifMeta`` records.

        The default wraps :meth:`pull_labeled` and leaves ``dgp_class`` /
        ``cadence`` as ``None``. A concrete adapter (e.g. ``ScrapedLiveSource``)
        that has access to the source catalog should override this to fill
        those fields — the reward-hacking-defense breadth gates in ``score.py``
        need them to be non-``None`` to enforce the DGP-class-share and
        cadence-band-share floors.
        """
        return [
            MotifMeta(motif=motif, domain=domain)
            for motif, domain in self.pull_labeled(n, length, rng)
        ]


class SyntheticLiveSource(LiveSource):
    """Offline stand-in whose texture is *distinct* from the synthetic generator.

    Random walk + Poisson-like jumps + multiplicative noise. This looks nothing
    like the additive trend/seasonal/AR series the generator builds, which is the
    whole point: "real motifs" must not be reproducible by fitting the synthetic
    process.
    """

    domain = "random_walk"

    def __init__(
        self,
        vol: float = 0.1,
        jump_prob: float = 0.03,
        jump_scale: float = 1.5,
        drift_scale: float = 0.01,
        mult_noise: float = 0.02,
        base: float = 10.0,
    ) -> None:
        self.vol = vol
        self.jump_prob = jump_prob
        self.jump_scale = jump_scale
        self.drift_scale = drift_scale
        self.mult_noise = mult_noise
        self.base = base

    def pull(self, n: int, length: int, rng: np.random.Generator) -> list[np.ndarray]:
        motifs: list[np.ndarray] = []
        for _ in range(n):
            drift = rng.normal(0.0, self.drift_scale)
            steps = rng.normal(0.0, self.vol, size=length)
            jumps = (rng.random(length) < self.jump_prob) * rng.normal(
                0.0, self.jump_scale, size=length
            )
            walk = np.cumsum(drift + steps + jumps)
            mult = np.exp(rng.normal(0.0, self.mult_noise, size=length))
            motifs.append((self.base + walk) * mult)
        return motifs


class FreshBuffer:
    """A fixed pool of live motifs with seeded, consensus-safe sampling.

    In the demo the pool is generated from a seeded source; in production the
    pool *is* the freshly ingested, post-commit, quarantined data snapshot --
    identical across validators because they share the same feed snapshot.
    """

    def __init__(self, source: LiveSource, pool_size: int = 128, motif_len: int = 768) -> None:
        self.source = source
        self.pool_size = pool_size
        self.motif_len = motif_len
        self._pool: list[tuple[np.ndarray, str]] = []
        self._pool_meta: list[MotifMeta] = []

    def refresh(self, rng: np.random.Generator) -> None:
        """Repopulate the pool. Seeded so all validators hold the same pool.

        Stores each motif with its ``domain`` label so sampled windows inherit
        the data-generating process they came from.
        """
        meta = self.source.pull_meta(self.pool_size, self.motif_len, rng)
        self._pool_meta = meta
        # Legacy (motif, domain) tuple pool, kept for backward compat with
        # generate.py's `motif, domain = buffer.sample_labeled(...)` destructuring.
        self._pool = [(m.motif, m.domain) for m in meta]

    def ensure(self, rng: np.random.Generator) -> None:
        """Populate the pool once if it is empty."""
        if not self._pool:
            self.refresh(rng)

    @property
    def pool_domains(self) -> list[str]:
        """The domain label of every motif currently pooled (diagnostics/tests)."""
        return [domain for _, domain in self._pool]

    @property
    def pool_dgp_classes(self) -> list[str | None]:
        """The dgp_class of every motif currently pooled (fills ``None`` for
        legacy sources that don't tag it). Used by the reward-hacking-defense
        breadth gates in ``score.py``."""
        return [m.dgp_class for m in self._pool_meta]

    @property
    def pool_cadences(self) -> list[str | None]:
        """The cadence band of every motif currently pooled (fills ``None`` for
        legacy sources that don't tag it). Used by the cadence-breadth reward-
        hacking-defense gate."""
        return [m.cadence for m in self._pool_meta]

    def sample_labeled(
        self, k: int, length: int, rng: np.random.Generator
    ) -> list[tuple[np.ndarray, str]]:
        """Draw ``k`` contiguous ``(window, domain)`` pairs from the pool.

        Seeded entirely by ``rng`` so the draw is identical for every validator
        replaying the same beacon. Raises if the pool is empty -- callers must
        :meth:`ensure`/:meth:`refresh` first.
        """
        if not self._pool:
            raise RuntimeError("FreshBuffer is empty; call refresh()/ensure() first")
        if length > self.motif_len:
            raise ValueError(f"requested length {length} exceeds motif_len {self.motif_len}")
        out: list[tuple[np.ndarray, str]] = []
        for _ in range(k):
            series, domain = self._pool[int(rng.integers(0, len(self._pool)))]
            start = int(rng.integers(0, len(series) - length + 1))
            out.append((np.asarray(series[start : start + length], dtype=float), domain))
        return out

    def sample_motifs(self, k: int, length: int, rng: np.random.Generator) -> list[np.ndarray]:
        """Draw ``k`` contiguous windows (domain labels dropped) -- back-compat shim."""
        return [motif for motif, _ in self.sample_labeled(k, length, rng)]
