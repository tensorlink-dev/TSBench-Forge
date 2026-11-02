"""Statistical-character features for a series, and PCA over the catalog.

Balance by domain and cadence is a PROXY. Two hourly nature series can be a
smooth diurnal curve and an intermittent spike train — the same cell, opposite
forecasting problems. BOOM characterises its corpus by statistical features
instead; this does the same for ours, so "what is the eval missing" can be
answered in feature space rather than by label.

Deliberately numpy/scipy only and ~15 cheap features: this runs over thousands
of series on a 2-core box, so anything needing a model fit per series is out.
"""
from __future__ import annotations

import numpy as np

FEATURES = [
    "log_cv", "skew", "log_kurtosis", "zero_frac", "flat_frac",
    "acf1", "acf_seasonal", "trend_strength", "spectral_entropy",
    "spikiness", "lumpiness", "stability", "crossing_rate",
    "level_shift", "hurst",
]


def _acf(x: np.ndarray, lag: int) -> float:
    if x.size <= lag + 2:
        return 0.0
    a, b = x[:-lag], x[lag:]
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def _tiles(x: np.ndarray, k: int = 10) -> list[np.ndarray]:
    n = x.size // k
    return [x[i * n:(i + 1) * n] for i in range(k)] if n >= 3 else []


def features(x: np.ndarray, season: int = 0) -> dict[str, float]:
    """~15 shape descriptors for one series. NaNs are dropped, not imputed."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    out = {f: 0.0 for f in FEATURES}
    if x.size < 64:
        return out
    mu, sd = float(x.mean()), float(x.std())
    out["log_cv"] = float(np.log1p(sd / max(abs(mu), 1e-9)))
    if sd > 0:
        z = (x - mu) / sd
        out["skew"] = float(np.clip((z ** 3).mean(), -10, 10))
        out["log_kurtosis"] = float(np.log1p(max((z ** 4).mean() - 3.0, 0.0)))
    out["zero_frac"] = float((x == 0).mean())
    runs = np.diff(np.flatnonzero(np.r_[True, x[1:] != x[:-1], True]))
    out["flat_frac"] = float(runs.max() / x.size)
    out["acf1"] = _acf(x, 1)
    if season and x.size > season + 2:
        out["acf_seasonal"] = _acf(x, season)
    # trend strength: 1 - var(resid of a linear fit) / var(x)
    t = np.arange(x.size, dtype=float)
    p = np.polyfit(t, x, 1)
    resid = x - np.polyval(p, t)
    out["trend_strength"] = float(1.0 - resid.var() / max(x.var(), 1e-12)) if x.var() > 0 else 0.0
    # spectral entropy, normalised to [0,1]: 1 = white noise, 0 = one pure tone
    sp = np.abs(np.fft.rfft(x - mu)) ** 2
    sp = sp[1:]
    if sp.sum() > 0:
        pr = sp / sp.sum()
        pr = pr[pr > 0]
        out["spectral_entropy"] = float(-(pr * np.log(pr)).sum() / np.log(pr.size))
    # spikiness: variance of the leave-one-out variances of the residual
    if resid.size > 10:
        out["spikiness"] = float(np.log1p(np.var([np.var(np.delete(resid, i))
                                                  for i in np.argsort(-np.abs(resid))[:20]])))
    tl = _tiles(x)
    if tl:
        means = np.array([t_.mean() for t_ in tl])
        varis = np.array([t_.var() for t_ in tl])
        out["lumpiness"] = float(np.log1p(varis.var()))
        out["stability"] = float(np.log1p(means.var()))
    out["crossing_rate"] = float((np.diff(np.sign(x - np.median(x))) != 0).mean())
    # level shift: biggest jump between adjacent tile means, in sd units
    if tl and sd > 0:
        out["level_shift"] = float(np.abs(np.diff([t_.mean() for t_ in tl])).max() / sd)
    # Hurst exponent by rescaled-range analysis, on the LEVELS.
    #
    # Fixed 2026-11-02: the first version differenced x first, which estimates
    # the persistence of the INCREMENTS, not H. Validated against known
    # processes: on levels, white noise -> 0.57, random walk -> 1.00, smoothed
    # noise -> 1.01 (textbook 0.5 / 1.0 / high). Differencing first gave
    # 0.15 / 0.59 / 0.60 — so the old feature ranked a random walk as
    # ANTI-persistent and made real long-memory series look absent from the
    # corpus. Larger windows too: k<=64 is too short to fit the scaling slope.
    if x.size > 128 and sd > 0:
        ns, rs = [], []
        for k in (16, 32, 64, 128, 256):
            if x.size < k * 2:
                break
            ch = x[:(x.size // k) * k].reshape(-1, k)
            z = ch - ch.mean(axis=1, keepdims=True)
            r = np.ptp(z.cumsum(axis=1), axis=1)
            s = ch.std(axis=1)
            ok = s > 0
            if ok.any():
                ns.append(np.log(k))
                rs.append(np.log((r[ok] / s[ok]).mean()))
        if len(ns) >= 2:
            out["hurst"] = float(np.clip(np.polyfit(ns, rs, 1)[0], 0, 2))
    return {k: (v if np.isfinite(v) else 0.0) for k, v in out.items()}


def pca(mat: np.ndarray, k: int = 2):
    """Standardise, then project onto the top-k principal components.

    numpy SVD rather than sklearn — sklearn is not a dependency of this repo
    and one eigendecomposition does not justify adding it.
    """
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0)
    sd[sd == 0] = 1.0
    z = (mat - mu) / sd
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    scores = u[:, :k] * s[:k]
    var = (s ** 2) / (s ** 2).sum()
    return scores, vt[:k], var[:k]
