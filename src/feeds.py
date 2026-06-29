"""Production live-feed adapters and the freshness discipline they need.

``ingest.py`` defines the ``LiveSource`` contract and a synthetic stand-in so the
demo runs offline. This module supplies the two things a *real* deployment must
add on top, the things the ``ingest`` docstring calls load-bearing but leaves as
an extension point:

1. **As-of / vintage gating** (:class:`AsOfLiveSource`). Many feeds silently
   revise history; scoring on a revised value leaks the future into the context.
   An as-of source carries a timestamp per motif and admits only points stamped
   *after* the commit beacon time -- so a miner cannot have seen them at commit.

2. **Quarantine + dedup across epochs** (:class:`DedupFreshBuffer`). A finite or
   slowly-refreshing feed is memorisable: serve the same motif twice and the
   second serving is a lookup. The dedup buffer fingerprints every motif it has
   ever served and rejects near-duplicates on later refreshes, so the effective
   feed keeps shrinking toward "only genuinely new data counts."

Plus one concrete real adapter, :class:`HttpCsvLiveSource`, that pulls a numeric
column over HTTP -- enough to wire a real vendor feed, with the network fetch
injected so it is testable offline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ingest import FreshBuffer, LiveSource

# A dated motif: the series, its domain label, and the time it became available.
# Timestamps are plain floats (epoch seconds / block heights) so this stays
# dependency-free; any monotonic "availability clock" works.
DatedMotif = tuple[np.ndarray, str, float]


class DatedLiveSource(LiveSource):
    """A live source that also stamps each motif with an availability time.

    Subclasses implement :meth:`pull_dated`; :meth:`pull`/:meth:`pull_labeled`
    are derived so a dated source is still a drop-in plain ``LiveSource``.
    """

    def pull_dated(self, n: int, length: int, rng: np.random.Generator) -> list[DatedMotif]:
        raise NotImplementedError

    def pull(self, n: int, length: int, rng: np.random.Generator) -> list[np.ndarray]:
        return [m for m, _, _ in self.pull_dated(n, length, rng)]

    def pull_labeled(self, n, length, rng):  # type: ignore[override]
        return [(m, d) for m, d, _ in self.pull_dated(n, length, rng)]


@dataclass
class AsOfLiveSource(DatedLiveSource):
    """Wrap a dated source and admit only motifs available after ``commit_time``.

    This is the vintage gate: at scoring time ``commit_time`` is the beacon/commit
    point, and only data that became available strictly after it can be served.
    To return ``n`` motifs it pulls in batches (over-pulling by ``overscan``) and
    keeps fetching until enough fresh motifs pass the gate or ``max_batches`` is
    hit -- so a feed that is mostly stale still yields a full, fresh challenge set
    when possible rather than silently shrinking it.
    """

    inner: DatedLiveSource
    commit_time: float
    overscan: int = 4
    max_batches: int = 8

    @property
    def domain(self) -> str:  # type: ignore[override]
        return self.inner.domain

    def pull_dated(self, n: int, length: int, rng: np.random.Generator) -> list[DatedMotif]:
        kept: list[DatedMotif] = []
        for _ in range(self.max_batches):
            batch = self.inner.pull_dated(n * self.overscan, length, rng)
            kept.extend(m for m in batch if m[2] > self.commit_time)
            if len(kept) >= n:
                break
        return kept[:n]


def _signature(series: np.ndarray, dims: int = 32) -> np.ndarray:
    """A scale/level-invariant fingerprint of a motif for dedup.

    Resample to ``dims`` points and z-score, so two windows that differ only by
    offset, gain, or minor resampling collapse to (near) the same unit vector --
    exactly the cases a memorising miner would exploit.
    """
    x = np.asarray(series, dtype=float).reshape(-1)
    if x.size == 0:
        return np.zeros(dims)
    idx = np.linspace(0, x.size - 1, dims)
    resampled = np.interp(idx, np.arange(x.size), x)
    centered = resampled - resampled.mean()
    norm = float(np.linalg.norm(centered))
    return centered / norm if norm > 1e-12 else np.zeros(dims)


class DedupFreshBuffer(FreshBuffer):
    """A :class:`~ingest.FreshBuffer` that quarantines near-duplicate motifs.

    Every motif admitted to the pool is fingerprinted (:func:`_signature`) and
    remembered across refreshes. On each refresh, a freshly pulled motif is
    dropped if its fingerprint is within ``similarity_threshold`` cosine
    similarity of anything served before (or already admitted this round). It
    over-pulls to refill the pool after rejections.

    The result: re-serving memorised data does not count, so the benchmark's
    freshness does not silently decay as a finite feed is reused.
    """

    def __init__(
        self,
        source: LiveSource,
        pool_size: int = 128,
        motif_len: int = 768,
        similarity_threshold: float = 0.985,
        refill_factor: int = 3,
    ) -> None:
        super().__init__(source, pool_size, motif_len)
        self.similarity_threshold = similarity_threshold
        self.refill_factor = refill_factor
        self._seen: list[np.ndarray] = []

    def _is_duplicate(self, sig: np.ndarray, against: list[np.ndarray]) -> bool:
        if not against or float(np.linalg.norm(sig)) <= 1e-12:
            return False
        sims = np.array([float(np.dot(sig, s)) for s in against])
        return bool(np.any(sims >= self.similarity_threshold))

    def refresh(self, rng: np.random.Generator) -> None:
        """Repopulate the pool with motifs that are novel vs. all prior servings."""
        admitted: list[tuple[np.ndarray, str]] = []
        admitted_sigs: list[np.ndarray] = []
        for _ in range(self.refill_factor):
            if len(admitted) >= self.pool_size:
                break
            batch = self.source.pull_labeled(
                self.pool_size * self.refill_factor, self.motif_len, rng
            )
            for motif, domain in batch:
                if len(admitted) >= self.pool_size:
                    break
                sig = _signature(motif)
                if self._is_duplicate(sig, self._seen) or self._is_duplicate(sig, admitted_sigs):
                    continue
                admitted.append((np.asarray(motif, dtype=float), domain))
                admitted_sigs.append(sig)
        self._pool = admitted
        self._seen.extend(admitted_sigs)

    @property
    def n_quarantined_total(self) -> int:
        """How many distinct motifs have ever been admitted (diagnostics)."""
        return len(self._seen)


# A fetcher maps a URL to its raw text body. Injected so HttpCsvLiveSource is
# testable without network (and so the network policy stays in the caller's hands).
Fetcher = Callable[[str], str]


def _urllib_fetch(url: str, timeout: float = 30.0) -> str:  # pragma: no cover - network
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


@dataclass
class HttpCsvLiveSource(LiveSource):
    """Pull a numeric series from a CSV endpoint -- a minimal real feed adapter.

    Reads ``value_column`` from CSV text fetched at ``url``, forms one long motif,
    and serves random windows of it. Real deployments would point this at an
    as-of vendor endpoint and wrap it in :class:`AsOfLiveSource`; the ``fetch``
    callable is injected so the network policy and offline testing stay external.
    """

    url: str
    domain: str = "http_csv"  # type: ignore[assignment]
    value_column: int = -1
    has_header: bool = True
    fetch: Fetcher = _urllib_fetch

    def _load(self) -> np.ndarray:
        text = self.fetch(self.url)
        rows = [r for r in text.splitlines() if r.strip()]
        if self.has_header and rows:
            rows = rows[1:]
        vals: list[float] = []
        for row in rows:
            cells = row.split(",")
            try:
                vals.append(float(cells[self.value_column]))
            except (ValueError, IndexError):
                continue
        return np.asarray(vals, dtype=float)

    def pull(self, n: int, length: int, rng: np.random.Generator) -> list[np.ndarray]:
        series = self._load()
        if series.size < length:
            raise ValueError(f"feed has {series.size} points, need >= {length}")
        out: list[np.ndarray] = []
        for _ in range(n):
            start = int(rng.integers(0, series.size - length + 1))
            out.append(series[start : start + length].copy())
        return out
