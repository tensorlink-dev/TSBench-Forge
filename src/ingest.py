"""Live data ingestion: the freshness layer.

Anti-gaming role
----------------
Memorisation is the easiest way to "game" a forecasting benchmark: if the test
series existed at commit time, a miner can simply look up the answer. The live
layer closes that vector by sourcing real motifs that **did not exist when the
miner committed**. The benchmark only ever uses data timestamped *after* the
commit point.

``LiveSource`` is the abstraction every real feed implements;
``scraped_source.ScrapedLiveSource`` is the production adapter over the scraped
catalog (``src/sources/``), and ``live_feeds`` / ``daily_feeds`` provide curated
public-data adapters. :class:`MixtureLiveSource` blends several sources into one
multi-domain feed. One property is load-bearing:

* **Seeded sampling.** ``FreshBuffer.sample_meta`` draws from a fixed pool using
  the beacon-derived RNG, so every validator samples the *same* windows and
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


def _finalize(series: np.ndarray) -> np.ndarray:
    """Make a raw observable finite, non-constant, and unit-scale (z-scored).

    Heterogeneous feeds live on wildly different scales; z-scoring puts them on a
    common footing so no single source dominates the pool and the per-challenge
    error normalisation in ``score`` stays well-conditioned. A degenerate
    (constant) window -- which would make the naive one-step scale collapse --
    falls back to a deterministic ramp rather than a flat line. Shared by the
    real-data adapters (``live_feeds`` / ``daily_feeds``).
    """
    x = np.nan_to_num(np.asarray(series, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean()
    s = float(np.std(x))
    if s < 1e-8:
        x = np.linspace(-1.0, 1.0, x.size)
        x = x - x.mean()
        s = float(np.std(x)) or 1.0
    return x / s


@dataclass(frozen=True)
class MotifMeta:
    """Rich metadata for one motif in the freshness pool.

    ``domain`` is the coarse tag every source has emitted since the beginning
    (kept for the ``(motif, domain)`` tuple pool that ``FreshBuffer`` and its
    ``sample_labeled`` consumers unpack). ``dgp_class`` and ``cadence`` are new fields added
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
    freq: str | None = None
    """Exact sampling interval as an ISO-8601 duration (``PT1H``, ``P1D``, …) from
    ``sources.yaml`` — finer than the coarse ``cadence`` band. The evaluator maps
    it to a season length for MASE's seasonal-naive scaling."""
    ts: np.ndarray | None = None
    """UTC-naive ``datetime64[ns]`` timestamps aligned with ``motif``, or ``None``
    when the feed's timestamps don't parse. The challenge builder compares the
    truth window's timestamps against the daily cutoff to compute the challenge's
    ``unseen_frac`` (the share of the horizon a pretrained model cannot have seen)."""


class LiveSource(ABC):
    """Abstract source of real time-series motifs.

    A concrete adapter wraps an external feed. The only required contract is
    :meth:`pull`; everything downstream (buffering, splicing) is texture-agnostic.

    Each source advertises a ``domain`` label naming the data-generating process
    (or real-world feed) it represents. The label rides along with every motif
    (see :meth:`pull_labeled` and ``FreshBuffer``) so the scorer can measure and
    stratify by *coverage* -- the property a foundation-model benchmark needs but
    a single-source feed cannot provide. :class:`MixtureLiveSource` overrides
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


class MixtureLiveSource(LiveSource):
    """Blend several :class:`LiveSource` components into one multi-domain feed.

    Each motif is drawn from a component chosen by the (normalised) component
    weights, and carries that component's ``domain`` label via
    :meth:`pull_labeled`. The draw and every downstream pull flow through the
    passed-in ``rng``, so the whole feed stays a pure function of the beacon seed
    -- the property the determinism / consensus tests rely on. Used by
    ``live_feeds`` / ``daily_feeds`` to blend curated real public sources.
    """

    domain = "mixture"

    def __init__(self, components: list[tuple[LiveSource, float]]) -> None:
        if not components:
            raise ValueError("MixtureLiveSource needs at least one component")
        weights = np.array([w for _, w in components], dtype=float)
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("component weights must be non-negative and sum to > 0")
        self.sources = [s for s, _ in components]
        self.weights = weights / weights.sum()

    def pull_labeled(
        self, n: int, length: int, rng: np.random.Generator
    ) -> list[tuple[np.ndarray, str]]:
        idx = rng.choice(len(self.sources), size=n, p=self.weights)
        out: list[tuple[np.ndarray, str]] = []
        for i in idx:
            src = self.sources[int(i)]
            out.append((src.pull(1, length, rng)[0], src.domain))
        return out

    def pull(self, n: int, length: int, rng: np.random.Generator) -> list[np.ndarray]:
        return [m for m, _ in self.pull_labeled(n, length, rng)]


class FreshBuffer:
    """A fixed pool of live motifs with seeded, consensus-safe sampling.

    In the demo the pool is generated from a seeded source; in production the
    pool *is* the freshly ingested, post-commit, quarantined data snapshot --
    identical across validators because they share the same feed snapshot.
    """

    def __init__(
        self,
        source: LiveSource,
        pool_size: int = 128,
        motif_len: int = 768,
        tail_frac: float = 0.5,
    ) -> None:
        self.source = source
        self.pool_size = pool_size
        self.motif_len = motif_len
        self.tail_frac = float(tail_frac)
        self._pool: list[tuple[np.ndarray, str]] = []
        self._pool_meta: list[MotifMeta] = []

    def refresh(self, rng: np.random.Generator) -> None:
        """Repopulate the pool. Seeded so all validators hold the same pool.

        Stores each motif with its ``domain`` label so sampled windows inherit
        the data-generating process they came from.
        """
        meta = self.source.pull_meta(self.pool_size, self.motif_len, rng)
        self._pool_meta = meta
        # (motif, domain) tuple pool, kept for the ``sample_labeled`` consumers
        # that destructure `motif, domain = buffer.sample_labeled(...)`.
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

    def sample_meta(
        self, k: int, length: int, rng: np.random.Generator
    ) -> list[MotifMeta]:
        """Draw ``k`` contiguous windows as ``MotifMeta``, carrying every label.

        Unlike :meth:`sample_labeled` (which keeps only ``domain``), this preserves
        ``dgp_class`` / ``cadence`` / ``source_id`` on each sampled window, so the
        pure-live challenge builder can stamp them onto every ``Challenge`` for
        stratified scoring and the breadth gates. Seeded entirely by ``rng``.
        """
        if not self._pool_meta:
            raise RuntimeError("FreshBuffer is empty; call refresh()/ensure() first")
        if length > self.motif_len:
            raise ValueError(f"requested length {length} exceeds motif_len {self.motif_len}")
        out: list[MotifMeta] = []
        for _ in range(k):
            m = self._pool_meta[int(rng.integers(0, len(self._pool_meta)))]
            series = m.motif
            # Tail-anchor bias mirrors ScrapedLiveSource: half the sub-windows
            # pin to the motif's fresh end so pooled post-cutoff steps survive
            # the second slice instead of being cut off by a uniform start.
            if float(rng.random()) < self.tail_frac:
                start = len(series) - length
            else:
                start = int(rng.integers(0, len(series) - length + 1))
            window = np.asarray(series[start : start + length], dtype=float)
            ts_win = None
            if m.ts is not None and len(m.ts) == len(series):
                ts_win = m.ts[start : start + length]
            out.append(
                MotifMeta(
                    motif=window,
                    domain=m.domain,
                    dgp_class=m.dgp_class,
                    cadence=m.cadence,
                    source_id=m.source_id,
                    freq=m.freq,
                    ts=ts_win,
                )
            )
        return out
