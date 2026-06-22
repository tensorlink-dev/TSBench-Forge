"""Feed discipline: as-of gating, cross-epoch dedup, and the CSV adapter."""

from __future__ import annotations

import numpy as np

from feeds import (
    AsOfLiveSource,
    DatedLiveSource,
    DedupFreshBuffer,
    HttpCsvLiveSource,
)
from ingest import SyntheticLiveSource


class _DatedConstant(DatedLiveSource):
    """Emits motifs whose availability time is an incrementing counter."""

    domain = "dated"

    def __init__(self) -> None:
        self.t = 0.0

    def pull_dated(self, n, length, rng):
        out = []
        for _ in range(n):
            self.t += 1.0
            out.append((np.ones(length) * self.t, self.domain, self.t))
        return out


def test_as_of_admits_only_post_commit_motifs() -> None:
    src = AsOfLiveSource(inner=_DatedConstant(), commit_time=3.0)
    motifs = src.pull_dated(4, 8, np.random.default_rng(0))
    assert len(motifs) == 4
    assert all(ts > 3.0 for _, _, ts in motifs)


def test_dedup_buffer_rejects_repeated_motifs_across_refresh() -> None:
    # A source that always returns the SAME shape: after the first refresh, every
    # later motif is a near-duplicate and must be quarantined.
    class _Repeat(SyntheticLiveSource):
        def pull(self, n, length, rng):
            base = np.sin(np.linspace(0, 6.28, length))
            return [base + 0.0 for _ in range(n)]

    buf = DedupFreshBuffer(_Repeat(), pool_size=8, motif_len=64)
    buf.refresh(np.random.default_rng(1))
    first = buf.n_quarantined_total
    assert first >= 1  # at least one admitted
    buf.refresh(np.random.default_rng(2))
    # Second refresh adds (almost) nothing new because everything is a duplicate.
    assert buf.n_quarantined_total - first <= 1


def test_dedup_buffer_keeps_distinct_motifs() -> None:
    buf = DedupFreshBuffer(SyntheticLiveSource(), pool_size=16, motif_len=128)
    buf.refresh(np.random.default_rng(3))
    # Genuinely varied source: the pool fills with distinct motifs.
    assert len(buf.pool_domains) >= 8


def test_dedup_buffer_is_a_drop_in_freshbuffer() -> None:
    buf = DedupFreshBuffer(SyntheticLiveSource(), pool_size=16, motif_len=128)
    buf.ensure(np.random.default_rng(4))
    windows = buf.sample_motifs(5, 32, np.random.default_rng(5))
    assert len(windows) == 5
    assert all(w.shape == (32,) for w in windows)


def test_http_csv_source_parses_injected_feed() -> None:
    csv = "date,value\n" + "\n".join(f"2026-01-{i:02d},{i * 1.5}" for i in range(1, 31))
    src = HttpCsvLiveSource(url="https://example/feed.csv", fetch=lambda _u: csv)
    motifs = src.pull(3, 10, np.random.default_rng(6))
    assert len(motifs) == 3
    assert all(m.shape == (10,) for m in motifs)
    # Values came from the last column (1.5 * row index).
    assert motifs[0].max() <= 30 * 1.5
