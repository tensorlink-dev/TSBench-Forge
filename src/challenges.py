"""Challenge construction from real live data — the only generation path.

The benchmark forecasts **real series**. A challenge is a window of a genuinely
real motif (climate, energy, markets, transport, …) drawn from the live catalog,
split into an observed ``context`` and a held-out ``truth`` horizon. The
distribution the benchmark tests is exactly the distribution of the ingested feeds
— there is no synthetic generation.

Anti-gaming role
----------------
Two independent layers protect this path; neither invents structure, they only
guard the real data:

* **Freshness / as-of gating** (``feeds.py`` + ``leakage_audit.py``) is the
  primary defence against memorisation: the pool only ever holds motifs
  timestamped *after* the miner's commit, so the answer could not have been
  looked up. This is load-bearing — the benchmark is exactly as memorisation-safe
  as the feed's vintage discipline.
* **Light augmentation** (``jitter`` / ``magnitude_warp`` / ``time_warp`` /
  ``history_cutout``) perturbs each served window so that even a *repeat* of the
  same underlying motif is not byte-identical to anything seen before — a
  belt-and-suspenders layer for finite or predictable feeds. Crucially the
  augmentations are **truth-preserving**: each produces a self-consistent series
  whose ``truth`` is still the genuine continuation of its ``context`` (magnitude
  and time warps are invertible reparametrisations; ``history_cutout`` only ever
  blanks *context*), so a good forecaster can still track them — they defeat
  exact-match lookup, not skill.

Determinism
-----------
Every draw flows through the passed-in ``rng`` (beacon-derived) via ``rng.spawn``,
so a given ``(pool, seed)`` yields a byte-identical challenge set across all
validators — the cornerstone of commit-reveal consensus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import CONTEXT_LEN, HORIZON
from ingest import FreshBuffer

SERIES_LEN = CONTEXT_LEN + HORIZON
_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Challenge container
# --------------------------------------------------------------------------- #


@dataclass
class Challenge:
    """One forecasting task handed to the panel and the model under test.

    ``meta`` carries provenance the scorer aggregates by:

    * ``domain``    — the GIFT-Eval domain of the source feed (``nature``,
      ``econ_fin``, …); the scorer stratifies and measures coverage by it.
    * ``dgp_class`` — the finer data-generating-process class (``sources/
      DGP_TAXONOMY.md``), read by the breadth gates.
    * ``cadence``   — the coarse frequency band, read by the cadence-breadth gate.
    * ``source_id`` — the ``sources.yaml`` id, for audit / dedup.
    * ``oracle``    — always ``None`` on the live path. The field is retained so
      the scorer's ``(context, meta) -> horizon`` model interface is unchanged;
      real forecasters ignore ``meta`` entirely.

    ``mode`` is always ``"live"`` (there is no synthetic mode anymore); it is kept
    for backward compatibility with code and reports that group by challenge mode.
    """

    context: np.ndarray
    truth: np.ndarray
    mode: str
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Augmentations (truth-preserving perturbations of a real motif)
# --------------------------------------------------------------------------- #


def jitter(x: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise scaled to the series' own variability."""
    sd = severity * (float(np.std(x)) + _EPS)
    return x + rng.normal(0.0, sd, size=x.shape)


def magnitude_warp(
    x: np.ndarray, severity: float, rng: np.random.Generator, knots: int = 4
) -> np.ndarray:
    """Multiply by a smooth random curve (slowly-varying gain)."""
    n = len(x)
    kx = np.linspace(0, n - 1, knots)
    ky = 1.0 + rng.normal(0.0, severity, size=knots)
    curve = np.interp(np.arange(n), kx, ky)
    return x * curve


def time_warp(
    x: np.ndarray, severity: float, rng: np.random.Generator, knots: int = 4
) -> np.ndarray:
    """Smoothly speed up / slow down local time (phase distortion)."""
    n = len(x)
    kx = np.linspace(0, n - 1, knots)
    speed = np.clip(1.0 + rng.normal(0.0, severity, size=knots), 0.2, None)
    fine = np.interp(np.arange(n), kx, speed)
    cum = np.cumsum(fine)
    cum = cum / cum[-1] * (n - 1)
    return np.interp(np.arange(n), cum, x)


def history_cutout(x: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Blank a contiguous chunk of *history* (context only), holding last value.

    Forces robustness to missing context; never touches the horizon, so the truth
    stays a faithful continuation.
    """
    span = int(severity * CONTEXT_LEN * 0.5)
    if span < 1:
        return x
    start = int(rng.integers(0, max(1, CONTEXT_LEN - span)))
    out = x.copy()
    out[start : start + span] = out[start]
    return out


# Structure-preserving augmentations are listed first; ``jitter`` (which adds
# irreducible noise to the horizon and so inflates every model's error uniformly)
# is included but used sparingly at low severity.
_AUGMENTATIONS = {
    "magnitude_warp": magnitude_warp,
    "time_warp": time_warp,
    "history_cutout": history_cutout,
    "jitter": jitter,
}


def _normalize_by_context(series: np.ndarray) -> np.ndarray:
    """Z-score a full ``context+horizon`` series by its **context** statistics.

    The real catalog spans wildly different scales (treasury debt in dollars, a UV
    index in single digits); without normalisation the scale-sensitive metrics
    (CRPS) and the pooled error average would be dominated by the largest-magnitude
    feeds. Normalising by context mean/std alone is **leak-free** — the horizon's
    own statistics never enter — and self-consistent: context and truth are
    divided by the same context-derived constant, so ``truth`` stays the genuine
    continuation of ``context``. MASE is already scale-free (it normalises by the
    context's naive one-step error), so this only *adds* comparability, it does not
    change the point-accuracy ranking.
    """
    ctx = series[:CONTEXT_LEN]
    mu = float(ctx.mean())
    sd = float(ctx.std())
    if sd < _EPS:
        sd = float(np.std(series)) or 1.0
    return (series - mu) / sd


def _apply_light_augmentation(
    series: np.ndarray, severity: float, rng: np.random.Generator
) -> np.ndarray:
    """Apply one randomly-chosen truth-preserving augmentation at ``severity``.

    Exactly one op keeps the perturbation *light* — enough to break exact-match
    memorisation of a repeated motif without distorting the forecasting task. The
    op is drawn from ``_AUGMENTATIONS`` so no single fingerprint is predictable.
    """
    if severity <= 0.0:
        return series
    op = str(rng.choice(list(_AUGMENTATIONS)))
    return _AUGMENTATIONS[op](series, severity, rng)


# --------------------------------------------------------------------------- #
# Challenge assembly
# --------------------------------------------------------------------------- #


def build_live_challenges(
    buffer: FreshBuffer,
    rng: np.random.Generator,
    n: int,
    *,
    augment: bool = True,
    aug_severity: float = 0.3,
) -> list[Challenge]:
    """Deterministically assemble ``n`` challenges from real motifs.

    Each challenge draws one real window from ``buffer`` (which samples
    equal-weight across domain × dgp_class × cadence when backed by a
    ``ScrapedLiveSource``), optionally applies one light truth-preserving
    augmentation, and splits it into ``context`` / ``truth``. Every challenge is
    tagged with its source domain / dgp_class / cadence / source_id.

    Each challenge gets an independent child stream via ``rng.spawn`` so the set
    is order-stable and byte-reproducible — the basis of cross-validator consensus
    and the determinism test. ``children[0]`` is reserved for a one-time pool
    refresh so the per-challenge streams (``children[1:]``) are identical whether
    or not the buffer was already populated.
    """
    children = rng.spawn(n + 1)
    buffer.ensure(children[0])
    challenges: list[Challenge] = []
    for child in children[1:]:
        m = buffer.sample_meta(1, SERIES_LEN, child)[0]
        series = _normalize_by_context(np.asarray(m.motif, dtype=float))
        if augment:
            series = _apply_light_augmentation(series, aug_severity, child)
        context, truth = series[:CONTEXT_LEN], series[CONTEXT_LEN:]
        challenges.append(
            Challenge(
                context=np.asarray(context, dtype=float),
                truth=np.asarray(truth, dtype=float),
                mode="live",
                meta={
                    "mode": "live",
                    "oracle": None,
                    "domain": m.domain,
                    "dgp_class": m.dgp_class,
                    "cadence": m.cadence,
                    "source_id": m.source_id,
                },
            )
        )
    return challenges


__all__ = [
    "Challenge",
    "SERIES_LEN",
    "build_live_challenges",
    "jitter",
    "magnitude_warp",
    "time_warp",
    "history_cutout",
]
