"""DSR metrics: the core yardstick for *dynamics*, not point error.

This extends the repo's :mod:`dsr_metrics` (binning ``d_stsp``, Hellinger ``d_h``,
Rosenstein ``max_lyapunov``, ``valid_prediction_time``, ``free_run``) with the pieces
the DSR-eval protocol needs and that the literature (Mikhaeil et al. 2022; Hess et al.
2023; Durstewitz et al. arXiv:2602.16864) specifies:

* :func:`state_space_divergence` -- ``D_stsp`` with a *binning* estimator in low
  dimensions and a **GMM / Monte-Carlo** KL estimator in higher dimensions where a
  histogram has too many empty cells. ``estimator="auto"`` picks by embedding dim.
* :func:`lyapunov_spectrum` -- the **full** Lyapunov spectrum of a known system via
  the Benettin QR algorithm on the analytic variational equations, from which
* :func:`kaplan_yorke_dimension` -- the ``D_KY`` fractal dimension follows.
* :func:`max_lyapunov_rate` -- the *data-driven* maximal exponent (Rosenstein) of a
  generated series, expressed per unit *time* (divided by ``dt``).
* :func:`power_spectrum_hellinger` -- ``D_H`` with a configurable Gaussian smoothing
  ``sigma``, averaged across the dimensions of a (possibly multivariate) trajectory.
* :func:`valid_prediction_time` / :func:`vpt_lyapunov` -- short-horizon validity in
  Lyapunov-time units.

Caveats (read before comparing across runs)
-------------------------------------------
The invariant metrics (``D_stsp``, ``D_H``, ``lambda``, ``D_KY``) only converge in the
**long-time ergodic limit**: they need long rollouts (T >= ~10^4 steps) with the
initial transient discarded, or they measure sampling noise rather than the attractor.
``D_stsp`` is sensitive to ``bins`` / ``n_components`` and ``D_H`` to ``sigma`` -- so a
cross-run comparison is only meaningful when these are held fixed. The Monte-Carlo
``D_stsp`` is additionally seed-dependent; pass a fixed ``seed``.
"""

from __future__ import annotations

import numpy as np

# Re-export the existing primitives so callers have one DSR-metrics surface.
from dsr_metrics import (
    auto_tau,
    d_h,
    d_stsp,
    delay_embed,
    free_run,
    max_lyapunov,
    power_spectrum,
    valid_prediction_time,
)

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# A small diagonal-covariance Gaussian mixture (numpy-only, no sklearn)
# --------------------------------------------------------------------------- #


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable log-sum-exp along ``axis`` (avoids a scipy dependency)."""
    amax = np.max(a, axis=axis, keepdims=True)
    amax = np.where(np.isfinite(amax), amax, 0.0)
    out = np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True)) + amax
    return np.squeeze(out, axis=axis)


def gmm_fit(
    x: np.ndarray,
    n_components: int,
    *,
    n_iter: int = 100,
    tol: float = 1e-4,
    reg: float = 1e-6,
    seed: int = 0,
) -> dict:
    """Fit a diagonal-covariance Gaussian mixture by EM. Deterministic given ``seed``.

    Returns ``{"means", "vars", "weights"}``. Determinism matters: two identical
    inputs fit byte-identical mixtures, so :func:`d_stsp_gmm` of a series with itself
    is exactly ``0``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[0] == 1:  # a single row is a 1xD point; treat columns as the sample
        x = x.reshape(-1, 1)
    n, d = x.shape
    rng = np.random.default_rng(seed)
    k = max(1, min(n_components, n))

    idx = rng.choice(n, size=k, replace=n < k)
    means = x[idx].copy()
    base_var = x.var(axis=0) + reg
    variances = np.tile(base_var, (k, 1))
    weights = np.full(k, 1.0 / k)

    prev_ll = -np.inf
    for _ in range(n_iter):
        log_prob = _gmm_component_logprob(x, means, variances, weights)  # (n, k)
        ll_per = _logsumexp(log_prob, axis=1)  # (n,)
        ll = float(np.sum(ll_per))
        resp = np.exp(log_prob - ll_per[:, None])  # (n, k)

        nk = resp.sum(axis=0) + _EPS
        weights = nk / n
        means = (resp.T @ x) / nk[:, None]
        for j in range(k):
            diff = x - means[j]
            variances[j] = (resp[:, j] @ (diff * diff)) / nk[j] + reg
        if abs(ll - prev_ll) <= tol * (abs(ll) + _EPS):
            break
        prev_ll = ll
    return {"means": means, "vars": variances, "weights": weights}


def _gmm_component_logprob(
    x: np.ndarray, means: np.ndarray, variances: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """``log( w_k * N(x | mu_k, diag var_k) )`` for every point/component; ``(n, k)``."""
    d = x.shape[1]
    out = np.empty((x.shape[0], means.shape[0]))
    for j in range(means.shape[0]):
        var = variances[j]
        diff = x - means[j]
        log_det = np.sum(np.log(var))
        quad = np.sum((diff * diff) / var, axis=1)
        out[:, j] = np.log(weights[j] + _EPS) - 0.5 * (d * np.log(2 * np.pi) + log_det + quad)
    return out


def gmm_logpdf(x: np.ndarray, gmm: dict) -> np.ndarray:
    """Log density of points ``x`` under a fitted mixture ``gmm``; shape ``(n,)``."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    log_prob = _gmm_component_logprob(x, gmm["means"], gmm["vars"], gmm["weights"])
    return _logsumexp(log_prob, axis=1)


# --------------------------------------------------------------------------- #
# D_stsp: binning (low-D) + GMM/Monte-Carlo (higher-D)
# --------------------------------------------------------------------------- #


def d_stsp_gmm(
    generated: np.ndarray,
    true: np.ndarray,
    *,
    m: int = 3,
    tau: int | None = None,
    n_components: int = 8,
    n_mc: int = 10000,
    seed: int = 0,
) -> float:
    """Monte-Carlo estimate of ``KL(p_true || p_gen)`` via Gaussian mixtures.

    Fits a GMM to each delay-embedded trajectory and estimates the KL divergence as
    ``E_{x~true}[ log p_true(x) - log p_gen(x) ]`` using the true embedded points as
    Monte-Carlo draws. Reliable where a histogram is too sparse (embedding dim > 3).
    Returns ``0`` for a series compared with itself (the two fits are identical) and a
    positive value for a misaligned invariant measure; clamped at 0 to absorb MC noise.

    The delay ``tau`` is resolved from the *true* series and applied to both
    embeddings, so generated and true live in the same reconstructed state space even
    when a collapsed model has a very different autocorrelation.
    """
    if tau is None:
        tau = auto_tau(true)
    g = delay_embed(generated, m=m, tau=tau)
    t = delay_embed(true, m=m, tau=tau)
    d = min(g.shape[1], t.shape[1])
    g, t = g[:, :d], t[:, :d]

    gmm_t = gmm_fit(t, n_components, seed=seed)
    gmm_g = gmm_fit(g, n_components, seed=seed)

    if n_mc < t.shape[0]:
        sample = t[np.random.default_rng(seed).choice(t.shape[0], size=n_mc, replace=False)]
    else:
        sample = t
    kl = float(np.mean(gmm_logpdf(sample, gmm_t) - gmm_logpdf(sample, gmm_g)))
    return max(0.0, kl)


def state_space_divergence(
    generated: np.ndarray,
    true: np.ndarray,
    *,
    m: int = 3,
    tau: int | None = None,
    estimator: str = "auto",
    bins: int = 20,
    n_components: int = 8,
    n_mc: int = 10000,
    seed: int = 0,
) -> float:
    """``D_stsp`` with the estimator chosen for the embedding dimension.

    ``estimator="auto"`` uses the binning KL (``dsr_metrics.d_stsp``) for ``m <= 3``
    and the GMM/Monte-Carlo KL (:func:`d_stsp_gmm`) for higher ``m`` where a histogram
    becomes too sparse. Force one with ``estimator="binning"`` / ``"gmm"``.

    When ``tau`` is ``None`` it is resolved once from the *true* series and applied to
    both embeddings, so a comparison is always made in a single, shared state space
    (``dsr_metrics.d_stsp`` would otherwise pick a separate delay per argument).
    """
    if tau is None:
        tau = auto_tau(true)
    if estimator == "auto":
        estimator = "binning" if m <= 3 else "gmm"
    if estimator == "binning":
        return d_stsp(generated, true, m=m, tau=tau, bins=bins)
    if estimator == "gmm":
        return d_stsp_gmm(
            generated, true, m=m, tau=tau, n_components=n_components, n_mc=n_mc, seed=seed
        )
    raise ValueError(f"unknown estimator {estimator!r}; use 'auto', 'binning', or 'gmm'")


# --------------------------------------------------------------------------- #
# Lyapunov spectrum (Benettin) + Kaplan-Yorke dimension
# --------------------------------------------------------------------------- #


def kaplan_yorke_dimension(spectrum) -> float:
    """Kaplan-Yorke (Lyapunov) dimension ``D_KY`` from a Lyapunov spectrum.

    ``D_KY = j + (sum_{i<=j} lambda_i) / |lambda_{j+1}|`` where ``j`` is the largest
    index whose partial sum of the descending exponents is still non-negative. For
    Lorenz-63 (``lambda ~ 0.906, 0, -14.57``) this is ``~2.062``; for Rossler
    (``0.0714, 0, -5.39``) ``~2.013``.
    """
    lam = np.sort(np.asarray(spectrum, dtype=float))[::-1]  # descending
    csum = np.cumsum(lam)
    nonneg = np.where(csum >= 0.0)[0]
    if nonneg.size == 0:  # even the largest exponent is negative -> a fixed point
        return 0.0
    j = int(nonneg[-1]) + 1  # number of exponents in the non-negative running sum
    if j >= lam.size:  # whole spectrum sums non-negative -> full-dimensional
        return float(lam.size)
    return float(j + csum[j - 1] / abs(lam[j]))


def _rk4_advance(field, x: np.ndarray, n: int, dt: float) -> np.ndarray:
    """Advance ``n`` RK4 steps and return the final state (burn-in helper)."""
    x = np.asarray(x, dtype=float).copy()
    for _ in range(n):
        k1 = field(x)
        k2 = field(x + 0.5 * dt * k1)
        k3 = field(x + 0.5 * dt * k2)
        k4 = field(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def lyapunov_spectrum(
    system,
    *,
    n_steps: int = 30000,
    dt: float | None = None,
    burn_in: int | None = None,
    k: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Full Lyapunov spectrum of a :class:`~dsr_eval.systems.ChaoticSystem` (Benettin QR).

    Integrates the state together with ``k`` orthonormal tangent vectors via the
    analytic variational equations ``dQ/dt = J(x) Q``, reorthonormalising every step
    and accumulating ``log|diag(R)|``; the time-averaged logs are the exponents
    (descending). Needs the system's analytic Jacobian. ``k`` defaults to the full
    dimension. The maximal entry is the system's ``lambda_max``; feed the whole
    spectrum to :func:`kaplan_yorke_dimension` for ``D_KY``.
    """
    dt = system.dt if dt is None else dt
    burn_in = (system.burn_in if burn_in is None else burn_in)
    d = system.dim
    k = d if k is None else k
    field = system.field()

    x = np.asarray(system.x0, dtype=float)
    if rng is not None:
        x = x + rng.standard_normal(d)
    x = _rk4_advance(field, x, burn_in, dt)

    q = np.linalg.qr(np.eye(d)[:, :k])[0]
    log_r = np.zeros(k)
    for _ in range(n_steps):
        x, y = _variational_rk4_step(system, x, q, dt)
        q, r = np.linalg.qr(y)
        log_r += np.log(np.abs(np.diag(r)) + _EPS)
    return np.sort(log_r / (n_steps * dt))[::-1]


def _variational_rk4_step(system, x: np.ndarray, q: np.ndarray, dt: float):
    """One RK4 step of the coupled ``(state, tangent-frame)`` variational system."""
    field = system.field()

    def deriv(xx, qq):
        return field(xx), system.jacobian(xx) @ qq

    k1x, k1q = deriv(x, q)
    k2x, k2q = deriv(x + 0.5 * dt * k1x, q + 0.5 * dt * k1q)
    k3x, k3q = deriv(x + 0.5 * dt * k2x, q + 0.5 * dt * k2q)
    k4x, k4q = deriv(x + dt * k3x, q + dt * k3q)
    x_next = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    q_next = q + (dt / 6.0) * (k1q + 2 * k2q + 2 * k3q + k4q)
    return x_next, q_next


# --------------------------------------------------------------------------- #
# Data-driven lambda_max, multivariate D_H, VPT in Lyapunov units
# --------------------------------------------------------------------------- #


def max_lyapunov_rate(
    x: np.ndarray,
    dt: float,
    *,
    m: int = 3,
    tau: int | None = None,
    horizon: int = 50,
    theiler: int = 10,
) -> float:
    """Data-driven maximal Lyapunov exponent of a series, per unit *time*.

    Wraps the Rosenstein per-step estimate (:func:`dsr_metrics.max_lyapunov`) and
    divides by ``dt`` so the result is in the same units as
    :func:`lyapunov_spectrum`. Positive => chaotic; ~0 => periodic. Approximate
    (intended for ``|lambda_gen - lambda_true|``-style comparison on a model's rollout).

    ``horizon`` is the number of steps the divergence slope is fit over and should span
    roughly one Lyapunov time; far-too-short windows badly overestimate the exponent
    (e.g. ``horizon=20`` triples Lorenz's lambda at ``dt=0.01``). The runner derives it
    from the system's Lyapunov time; the default suits ``dt`` of order ``0.01-0.05``.
    """
    slope = max_lyapunov(x, m=m, tau=tau, horizon=horizon, theiler=theiler)
    return slope / dt if dt else slope


def power_spectrum_hellinger(
    generated: np.ndarray, true: np.ndarray, *, sigma: int = 20
) -> float:
    """``D_H`` averaged across dimensions, with Gaussian smoothing width ``sigma``.

    For a univariate observable this is exactly :func:`dsr_metrics.d_h`; for a
    multivariate ``(steps, dim)`` trajectory it FFTs each dimension, Gaussian-smooths
    (width ``sigma``), normalises, takes the per-dimension Hellinger distance, and
    averages. Bounded in ``[0, 1]``.
    """
    g = np.asarray(generated, dtype=float)
    t = np.asarray(true, dtype=float)
    if g.ndim == 1:
        g = g[:, None]
    if t.ndim == 1:
        t = t[:, None]
    d = min(g.shape[1], t.shape[1])
    return float(np.mean([d_h(g[:, j], t[:, j], smoothing=sigma) for j in range(d)]))


def vpt_lyapunov(
    true: np.ndarray,
    pred: np.ndarray,
    *,
    dt: float,
    lyapunov_time: float,
    threshold: float = 0.4,
) -> float:
    """Valid prediction time in Lyapunov-time units (NRMSE threshold ``epsilon``).

    ``lyapunov_time`` is the system's ``1/lambda_max`` in *physical* time; it is
    converted to steps with ``dt``. Returns how many Lyapunov times the pointwise
    forecast stays within ``threshold`` of the truth.
    """
    lt_steps = lyapunov_time / dt if dt else lyapunov_time
    return valid_prediction_time(true, pred, threshold=threshold, lyapunov_time=lt_steps)


__all__ = [
    "auto_tau",
    "d_h",
    "d_stsp",
    "d_stsp_gmm",
    "delay_embed",
    "free_run",
    "gmm_fit",
    "gmm_logpdf",
    "kaplan_yorke_dimension",
    "lyapunov_spectrum",
    "max_lyapunov",
    "max_lyapunov_rate",
    "power_spectrum",
    "power_spectrum_hellinger",
    "state_space_divergence",
    "valid_prediction_time",
    "vpt_lyapunov",
]
