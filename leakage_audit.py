"""Contamination resistance: the production default buffer and a leakage audit.

The benchmark's whole anti-gaming thesis assumes the data stream is genuinely
fresh and unmemorisable. Three concrete defences make that real, on top of the
``feeds`` primitives:

1. :func:`default_fresh_buffer` -- the *production default*: dedup-across-epochs
   (``DedupFreshBuffer``) plus, when a commit time is supplied, the as-of /
   ``t_now`` temporal barrier (``AsOfLiveSource``). The bare ``FreshBuffer`` is
   for the offline demo; real deployments should use this.

2. :func:`global_t_now` / :func:`assert_post_cutoff` -- the *global post-training
   barrier* (Meyer et al., arXiv:2510.13654, R2): every test point must be dated
   after the **latest pretraining cutoff across all compared models**, not a
   fixed date. ``t_now = max(cutoffs)``.

3. :func:`memorization_probe` / :func:`feed_novelty` -- the *audit*. Static loss
   is an unreliable contamination signal for time series (TSFMAudit,
   arXiv:2605.26161); instead we measure *behaviour*: a model that memorised the
   feed does conspicuously better on series it has seen before (``repeated``) than
   on genuinely fresh ones, and a feed that has stopped producing novel data is
   memorisable. Both are reported as flags a validator can act on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from feeds import AsOfLiveSource, DatedLiveSource, DedupFreshBuffer, _signature
from ingest import LiveSource

_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Production default buffer
# --------------------------------------------------------------------------- #


def default_fresh_buffer(
    source: LiveSource,
    *,
    commit_time: float | None = None,
    pool_size: int = 128,
    motif_len: int = 768,
    similarity_threshold: float = 0.985,
) -> DedupFreshBuffer:
    """Build the contamination-resistant buffer a real deployment should use.

    Always dedups motifs across epochs. If ``commit_time`` is given, the source is
    first wrapped in an :class:`~feeds.AsOfLiveSource` so only motifs that became
    available *after* the commit beacon are admitted -- the ``t_now`` barrier. That
    requires a :class:`~feeds.DatedLiveSource`; passing ``commit_time`` with an
    undated source is a programming error and raised as such.
    """
    if commit_time is not None:
        if not isinstance(source, DatedLiveSource):
            raise TypeError(
                "commit_time/as-of barrier requires a DatedLiveSource; "
                f"got {type(source).__name__}"
            )
        source = AsOfLiveSource(source, commit_time=commit_time)
    return DedupFreshBuffer(
        source,
        pool_size=pool_size,
        motif_len=motif_len,
        similarity_threshold=similarity_threshold,
    )


# --------------------------------------------------------------------------- #
# Global post-training temporal barrier
# --------------------------------------------------------------------------- #


def global_t_now(model_cutoffs: dict[str, float] | list[float]) -> float:
    """The global barrier ``t_now = max(pretraining cutoffs)`` across all models.

    Every test observation must be dated strictly after this for a leakage-free
    comparison: a point before some model's cutoff could be in that model's
    pretraining corpus. Pass the per-model cutoffs (epoch seconds / block heights).
    """
    vals = list(model_cutoffs.values()) if isinstance(model_cutoffs, dict) else list(model_cutoffs)
    if not vals:
        raise ValueError("need at least one model cutoff to define t_now")
    return float(max(vals))


def assert_post_cutoff(timestamps: list[float] | np.ndarray, t_now: float) -> dict[str, object]:
    """Report whether every test timestamp is strictly after ``t_now``.

    Returns ``{ok, n_stale, n_total}``. ``ok`` is False if any point predates the
    barrier -- a hard stop a production validator should enforce before scoring.
    """
    ts = np.asarray(list(timestamps), dtype=float)
    stale = int(np.sum(ts <= t_now))
    return {"ok": stale == 0, "n_stale": stale, "n_total": int(ts.size)}


# --------------------------------------------------------------------------- #
# Behavioural leakage audit
# --------------------------------------------------------------------------- #

# A point model: (context, meta) -> horizon prediction (the panel/TSFM contract).
PointModel = Callable[[np.ndarray, "dict | None"], np.ndarray]


def _scaled_error(model: PointModel, challenges: list) -> float:
    """Mean scale-normalised MAE of a point model over a challenge list."""
    from score import _mae, _scale

    total = 0.0
    for ch in challenges:
        scale = _scale(ch.context)
        pred = model(ch.context, getattr(ch, "meta", None))
        total += _mae(pred, ch.truth) / scale
    return total / max(len(challenges), 1)


@dataclass
class MemorizationReport:
    err_fresh: float
    err_repeated: float
    memorization_gap: float  # (fresh - repeated) / fresh; >0 means better on seen data
    suspicious: bool


def memorization_probe(
    model: PointModel,
    fresh: list,
    repeated: list,
    *,
    suspicious_gap: float = 0.25,
) -> MemorizationReport:
    """Behavioural contamination check: is the model better on *seen* data?

    ``repeated`` are challenges the model may have been exposed to (e.g. re-served
    motifs); ``fresh`` are genuinely new. An honest forecaster scores about the
    same on both (gap ~ 0); a memoriser scores much better on ``repeated``, so a
    large positive ``memorization_gap`` flags likely contamination. This is a
    relative, behavioural signal -- robust where static loss is not.
    """
    ef = _scaled_error(model, fresh)
    er = _scaled_error(model, repeated)
    gap = (ef - er) / ef if ef > _EPS else 0.0
    return MemorizationReport(
        err_fresh=float(ef),
        err_repeated=float(er),
        memorization_gap=float(gap),
        suspicious=bool(gap > suspicious_gap),
    )


def feed_novelty(
    buffer: DedupFreshBuffer,
    motifs: list[np.ndarray],
    *,
    threshold: float | None = None,
) -> dict[str, object]:
    """Fraction of a candidate batch that is novel vs. everything ever served.

    A healthy live feed keeps producing data unlike anything in the dedup
    buffer's memory; a stale/finite feed's novelty collapses toward zero, meaning
    re-serving it would be a memorisation lookup. Returns ``{novelty_rate,
    n_novel, n_total}``. ``threshold`` defaults to the buffer's own dedup cutoff.
    """
    thr = threshold if threshold is not None else buffer.similarity_threshold
    seen = buffer._seen  # signatures of everything admitted so far
    novel = 0
    for m in motifs:
        sig = _signature(m)
        if float(np.linalg.norm(sig)) <= 1e-12:
            continue
        sims = [float(np.dot(sig, s)) for s in seen] if seen else [0.0]
        if max(sims) < thr:
            novel += 1
    n = max(len(motifs), 1)
    return {"novelty_rate": novel / n, "n_novel": novel, "n_total": len(motifs)}


__all__ = [
    "MemorizationReport",
    "assert_post_cutoff",
    "default_fresh_buffer",
    "feed_novelty",
    "global_t_now",
    "memorization_probe",
]
