"""Long-horizon dynamical-systems metrics: the hard tier's real yardstick.

Point metrics (MASE/WQL/CRPS) saturate to a noise ceiling after ~one Lyapunov
time: on a chaotic system *any* forecast -- a perfect model included --
decorrelates from the truth, so short-horizon error stops telling skill from
luck. What stays invariant under the dynamics, and is therefore the right thing
to score long-term, is the **invariant measure** (where the trajectory lives in
state space) and its **temporal/spectral structure**. This module implements the
dynamical-systems-reconstruction (DSR) metrics used by the Durstewitz group
(Koppe et al. 2019; Mikhaeil et al. 2022; Hess et al. 2023) and DynaMix:

* :func:`d_stsp` -- **geometric misalignment**: KL divergence between the
  state-space histograms (invariant measures) of generated vs. true trajectories.
  Catches attractor collapse, mode-dropping, mean-reversion. Blind to time order.
* :func:`d_h` -- **temporal misalignment**: Hellinger distance between power
  spectra. Catches wrong frequencies / autocorrelation, and -- crucially -- the
  *context-parroting* failure where a model collapses chaos into a sharply-peaked
  cyclic repeat instead of a broadband spectrum. Blind to geometry.
* :func:`valid_prediction_time` -- the short->long bridge: how long the pointwise
  forecast stays valid, optionally in Lyapunov-time units.
* :func:`max_lyapunov` -- a Rosenstein estimate of the largest Lyapunov exponent;
  ``Delta lambda`` between generated and true is the single sharpest
  anti-parroting scalar (a periodic collapse has lambda ~ 0).

The two misalignments are *orthogonal*: each alone is gameable, together they are
hard to fool. A univariate observable is delay-embedded (Takens) before the
geometric metric so "state space" is meaningful for a 1-D series.

Everything is numpy-only and operates on standardised trajectories.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    mu, sd = float(x.mean()), float(x.std())
    return (x - mu) / (sd if sd > _EPS else 1.0)


def auto_tau(x: np.ndarray, max_lag: int = 50) -> int:
    """Delay for embedding: first lag where autocorrelation drops below 1/e."""
    x = _standardize(x)
    n = x.size
    denom = float(np.dot(x, x)) or 1.0
    for lag in range(1, min(max_lag, n - 1)):
        if float(np.dot(x[:-lag], x[lag:])) / denom < np.exp(-1.0):
            return lag
    return 1


def delay_embed(x: np.ndarray, m: int = 3, tau: int | None = None) -> np.ndarray:
    """Takens delay embedding of a 1-D series into ``m`` dimensions.

    Returns an ``(N - (m-1)*tau, m)`` array. ``tau`` defaults to :func:`auto_tau`.
    A 2-D input is returned standardised per-column (already a state space).
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 2 and arr.shape[1] > 1:
        cols = [_standardize(arr[:, j]) for j in range(arr.shape[1])]
        return np.column_stack(cols)
    x = _standardize(arr)
    tau = tau or auto_tau(x)
    rows = x.size - (m - 1) * tau
    if rows <= 1:
        return x.reshape(-1, 1)
    return np.column_stack([x[i * tau : i * tau + rows] for i in range(m)])


# --------------------------------------------------------------------------- #
# Geometric misalignment: D_stsp
# --------------------------------------------------------------------------- #


def d_stsp(
    generated: np.ndarray,
    true: np.ndarray,
    *,
    m: int = 3,
    tau: int | None = None,
    bins: int = 20,
    laplace: float = 1e-6,
) -> float:
    """KL divergence between the state-space histograms (invariant measures).

    Both series are delay-embedded into ``m`` dimensions on a *shared* bin grid
    (spanned by the true trajectory, widened slightly), histogrammed, Laplace-
    smoothed, and compared by ``KL(p_true || p_gen)``. Lower is better; ~0 means
    the generated trajectory visits the same regions of state space with the same
    frequency as the truth. Asymmetric on purpose: it punishes a model that fails
    to put mass where the true system spends time.

    Note: histogram estimation is reliable for low embedding dimensions (m<=3);
    for higher-D attractors switch to a GMM/k-NN KL estimator (left as an
    extension) -- the API is unchanged.
    """
    g = delay_embed(generated, m=m, tau=tau)
    t = delay_embed(true, m=m, tau=tau)
    d = min(g.shape[1], t.shape[1])
    g, t = g[:, :d], t[:, :d]

    lo = t.min(axis=0)
    hi = t.max(axis=0)
    span = np.where(hi - lo > _EPS, hi - lo, 1.0)
    lo, hi = lo - 0.05 * span, hi + 0.05 * span
    edges = [np.linspace(lo[j], hi[j], bins + 1) for j in range(d)]

    p_true, _ = np.histogramdd(t, bins=edges)
    p_gen, _ = np.histogramdd(g, bins=edges)
    p_true = p_true.ravel() + laplace
    p_gen = p_gen.ravel() + laplace
    p_true /= p_true.sum()
    p_gen /= p_gen.sum()
    return float(np.sum(p_true * np.log(p_true / p_gen)))


# --------------------------------------------------------------------------- #
# Temporal misalignment: D_H
# --------------------------------------------------------------------------- #


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    """Gaussian smoothing via a normalised kernel (no scipy)."""
    if width <= 1:
        return x
    radius = int(3 * width)
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / width) ** 2)
    k /= k.sum()
    return np.convolve(x, k, mode="same")


def power_spectrum(x: np.ndarray, smoothing: int = 20) -> np.ndarray:
    """Normalised (probability-distribution) smoothed power spectrum of a series."""
    x = _standardize(x)
    ps = np.abs(np.fft.rfft(x)) ** 2
    ps = _smooth(ps, smoothing)
    total = float(ps.sum())
    return ps / total if total > _EPS else ps


def d_h(generated: np.ndarray, true: np.ndarray, *, smoothing: int = 20) -> float:
    """Hellinger distance in ``[0, 1]`` between generated and true power spectra.

    ~0 means matching frequency content (autocorrelation structure); ~1 means
    disjoint spectra. Symmetric and bounded, so it aggregates cleanly. This is the
    metric that exposes a parroting model: cyclic repetition gives a sharply
    peaked spectrum where true chaos is broadband.
    """
    pg = power_spectrum(generated, smoothing)
    pt = power_spectrum(true, smoothing)
    n = min(pg.size, pt.size)
    bc = float(np.sum(np.sqrt(pg[:n] * pt[:n])))
    return float(np.sqrt(max(0.0, 1.0 - bc)))


# --------------------------------------------------------------------------- #
# Valid prediction time + Lyapunov
# --------------------------------------------------------------------------- #


def valid_prediction_time(
    true: np.ndarray,
    pred: np.ndarray,
    *,
    threshold: float = 0.4,
    lyapunov_time: float | None = None,
) -> float:
    """First step at which the normalised forecast error exceeds ``threshold``.

    Error is scaled by the standard deviation of the true horizon so it is
    dimensionless and comparable across systems. If ``lyapunov_time`` (in steps)
    is given, the result is divided by it, returning VPT in *Lyapunov times* --
    the hardness-normalised horizon. Returns the full length when never exceeded.
    """
    true = np.asarray(true, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    n = min(true.size, pred.size)
    scale = float(true[:n].std()) or 1.0
    err = np.abs(pred[:n] - true[:n]) / scale
    over = np.argmax(err > threshold) if np.any(err > threshold) else n
    steps = float(over)
    return steps / lyapunov_time if lyapunov_time else steps


def max_lyapunov(
    x: np.ndarray,
    *,
    m: int = 3,
    tau: int | None = None,
    horizon: int = 20,
    theiler: int = 10,
) -> float:
    """Largest Lyapunov exponent via a simplified Rosenstein estimate.

    Delay-embeds the series, finds each point's nearest neighbour (excluding a
    Theiler window of temporally-close points), and fits the slope of the mean
    log divergence of neighbour pairs over ``horizon`` steps. Positive => chaotic;
    ~0 => periodic/limit-cycle. Approximate (intended for *relative* comparison,
    e.g. ``|lambda_gen - lambda_true|``), not a high-precision spectrum.
    """
    emb = delay_embed(x, m=m, tau=tau)
    npts = emb.shape[0]
    if npts < horizon + theiler + 2:
        return 0.0
    usable = npts - horizon
    div = np.zeros(horizon)
    counts = np.zeros(horizon)
    for i in range(usable):
        d = np.linalg.norm(emb[:usable] - emb[i], axis=1)
        d[max(0, i - theiler) : i + theiler + 1] = np.inf
        j = int(np.argmin(d))
        if not np.isfinite(d[j]):
            continue
        for h in range(horizon):
            dist = float(np.linalg.norm(emb[i + h] - emb[j + h]))
            if dist > _EPS:
                div[h] += np.log(dist)
                counts[h] += 1
    valid = counts > 0
    if valid.sum() < 2:
        return 0.0
    hs = np.arange(horizon)[valid]
    curve = div[valid] / counts[valid]
    slope = float(np.polyfit(hs, curve, 1)[0])
    return slope  # per-step exponent


# --------------------------------------------------------------------------- #
# Free-running rollout + report
# --------------------------------------------------------------------------- #


def free_run(
    point_fn: Callable[[np.ndarray, dict | None], np.ndarray],
    context: np.ndarray,
    total: int,
    *,
    block: int | None = None,
) -> np.ndarray:
    """Autoregressively roll a fixed-horizon forecaster out to ``total`` steps.

    Calls ``point_fn(history)`` for a block of predictions, appends them to the
    history, and repeats until ``total`` generated steps exist. This is the
    generation mode the DSR metrics need (thousands of steps), distinct from the
    benchmark's fixed-horizon scoring. ``block`` defaults to the model's natural
    horizon as inferred from one call.
    """
    hist = list(np.asarray(context, dtype=float).reshape(-1))
    gen: list[float] = []
    first = np.asarray(point_fn(np.asarray(hist), None), dtype=float).reshape(-1)
    step = block or max(1, first.size)
    while len(gen) < total:
        pred = np.asarray(point_fn(np.asarray(hist), None), dtype=float).reshape(-1)[:step]
        if pred.size == 0:
            break
        gen.extend(pred.tolist())
        hist.extend(pred.tolist())
    return np.asarray(gen[:total], dtype=float)


def dsr_report(
    generated: np.ndarray,
    true: np.ndarray,
    *,
    m: int = 3,
    tau: int | None = None,
    bins: int = 20,
    smoothing: int = 20,
) -> dict[str, float]:
    """Bundle the long-horizon DSR metrics on two free-running trajectories.

    Returns ``d_stsp`` (geometric), ``d_h`` (temporal), ``lyap_true`` /
    ``lyap_gen`` and their absolute difference ``lyap_gap``. Compute on long
    rollouts (>= a few thousand steps) for the invariant measure to be well
    sampled; burn-in should already be discarded by the caller.
    """
    lam_t = max_lyapunov(true, m=m, tau=tau)
    lam_g = max_lyapunov(generated, m=m, tau=tau)
    return {
        "d_stsp": d_stsp(generated, true, m=m, tau=tau, bins=bins),
        "d_h": d_h(generated, true, smoothing=smoothing),
        "lyap_true": lam_t,
        "lyap_gen": lam_g,
        "lyap_gap": abs(lam_t - lam_g),
    }


__all__ = [
    "auto_tau",
    "d_h",
    "d_stsp",
    "delay_embed",
    "dsr_report",
    "free_run",
    "max_lyapunov",
    "power_spectrum",
    "valid_prediction_time",
]
