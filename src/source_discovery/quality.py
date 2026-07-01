"""Automatic data-quality vetting — the deterministic gate a fetched source clears.

The system prompt's "discrimination filter + leakage check" made concrete for the
*data itself*. The discovery agent proposes; ``vet.py`` checks the proposal's
metadata; **this** runs on the actual fetched series and decides — with no human
and no LLM — whether a source is fit to enter rotation.

Two layers, because they answer different questions:

1. **Intrinsic quality** (:func:`assess_series`) — unambiguous, per-series
   auto-rejects: non-finite fraction, too short, constant / near-constant,
   degenerate value-cardinality (a stuck or quantized sensor), long flatlines, and
   single-spike variance domination. These are hard fails.

2. **Discrimination** (:func:`discrimination`) — the "appropriate signal" gate,
   done right. A raw SNR/spectral-flatness cutoff *cannot* separate genuinely-noisy
   real data from pure noise (they overlap), so those are reported as diagnostics,
   not hard gates. The real test is behavioural: run the reference panel on the
   source's windows and require (a) the set **discriminates** (spread over the panel
   above a floor — some models do better than others) and (b) it is **not
   trivially solved** by a naive copy (which flags pure periodicity *or* pure
   noise, where naive ties the field). This is the same machinery the benchmark
   scores with, so "admittable" means the same thing as "useful challenge".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import CONTEXT_LEN, HORIZON

_EPS = 1e-8
SERIES_LEN = CONTEXT_LEN + HORIZON


# --------------------------------------------------------------------------- #
# Intrinsic per-series quality
# --------------------------------------------------------------------------- #


@dataclass
class SeriesQuality:
    ok: bool
    metrics: dict[str, float]
    failures: list[str] = field(default_factory=list)   # hard fails -> not admittable
    warnings: list[str] = field(default_factory=list)   # soft flags -> diagnostic only


def _spectral_flatness(x: np.ndarray) -> float:
    """Wiener entropy of the detrended series in (0, 1].

    ~0 = a single dominant tone (trivially periodic); ~1 = white noise. Reported as
    a diagnostic — NOT a hard gate, since real noisy feeds sit near white noise too.
    """
    n = len(x)
    if n < 8:
        return float("nan")
    y = x - np.polyval(np.polyfit(np.arange(n), x, 1), np.arange(n))
    y = y * np.hanning(n)
    psd = np.abs(np.fft.rfft(y)) ** 2 + 1e-20
    psd = psd[1:]  # drop DC
    return float(np.exp(np.mean(np.log(psd))) / np.mean(psd))


def _snr_db(x: np.ndarray) -> float:
    """Crude SNR in dB: series std over a 2nd-difference white-noise estimate.

    Diagnostic only (see module docstring): real noisy feeds and pure noise overlap
    here, so it is reported and soft-warned on, never used to hard-reject.
    """
    if len(x) < 3:
        return float("nan")
    noise = float(np.std(np.diff(x, 2))) / np.sqrt(6.0)
    sig = float(np.std(x))
    if noise <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(sig / noise)) if sig > 0 else float("-inf")


def _max_flatline_frac(x: np.ndarray) -> float:
    """Longest run of identical consecutive values, as a fraction of length."""
    n = len(x)
    if n <= 1:
        return 1.0
    change = np.where(np.diff(x) != 0)[0]
    bounds = np.concatenate(([-1], change, [n - 1]))
    return float(np.diff(bounds).max() / n)


def _spike_dominance(x: np.ndarray) -> float:
    """Fraction of total squared deviation from the single largest-deviation point."""
    dev = (x - np.median(x)) ** 2
    tot = float(dev.sum())
    return float(dev.max() / tot) if tot > 0 else 0.0


def assess_series(
    x: np.ndarray,
    *,
    min_len: int = SERIES_LEN,
    min_finite_frac: float = 0.85,
    min_unique_ratio: float = 0.05,
    max_flatline_frac: float = 0.5,
    max_spike_dominance: float = 0.5,
    rel_std_floor: float = 1e-6,
    snr_warn_below_db: float = -20.0,
    flatness_warn_above: float = 0.7,
) -> SeriesQuality:
    """Assess one raw series. Hard fails go in ``failures``; ``ok`` is ``not failures``."""
    x0 = np.asarray(x, dtype=float)
    n0 = x0.size
    finite = np.isfinite(x0)
    finite_frac = float(finite.mean()) if n0 else 0.0
    x = x0[finite]
    n = x.size

    failures: list[str] = []
    warnings: list[str] = []

    if finite_frac < min_finite_frac:
        failures.append(f"finite_frac {finite_frac:.2f} < {min_finite_frac}")
    if n < min_len:
        failures.append(f"length {n} < {min_len}")

    # Everything below needs a usable series.
    if n >= 8:
        mean = float(np.mean(x))
        std = float(np.std(x))
        # Relative variance floor: constant / near-constant series.
        if std <= rel_std_floor * (abs(mean) + _EPS) or std == 0.0:
            failures.append(f"near-constant (std {std:.3g}, mean {mean:.3g})")
        unique_ratio = float(len(np.unique(np.round(x, 10))) / n)
        if unique_ratio < min_unique_ratio:
            failures.append(f"degenerate value cardinality (unique_ratio {unique_ratio:.3f})")
        flat = _max_flatline_frac(x)
        if flat > max_flatline_frac:
            failures.append(f"flatline run {flat:.2f} > {max_flatline_frac}")
        spike = _spike_dominance(x)
        if spike > max_spike_dominance:
            failures.append(f"single-spike variance domination {spike:.2f} > {max_spike_dominance}")
        snr = _snr_db(x)
        flatness = _spectral_flatness(x)
        if np.isfinite(snr) and snr < snr_warn_below_db:
            warnings.append(f"very low SNR ({snr:.1f} dB) — may be near-unforecastable")
        if np.isfinite(flatness) and flatness > flatness_warn_above:
            warnings.append(f"near-white spectrum (flatness {flatness:.2f}) — check discrimination")
        metrics = {
            "n": float(n), "finite_frac": finite_frac, "mean": mean, "std": std,
            "unique_ratio": unique_ratio, "max_flatline_frac": flat,
            "spike_dominance": spike, "snr_db": snr, "spectral_flatness": flatness,
        }
    else:
        metrics = {"n": float(n), "finite_frac": finite_frac}

    return SeriesQuality(ok=not failures, metrics=metrics, failures=failures, warnings=warnings)


# --------------------------------------------------------------------------- #
# Discrimination filter (the behavioural "appropriate signal" gate)
# --------------------------------------------------------------------------- #


@dataclass
class Discrimination:
    ok: bool
    predictability: float   # max autocorrelation over lags 1..K; ~0 => pure noise
    naive_error: float      # seasonal-naive normalised error; ~0 => trivially periodic
    spread: float           # panel spread (reported; anchor-dependent, not a hard gate)
    n_windows: int
    reasons: list[str] = field(default_factory=list)


def _windows(series_list: list[np.ndarray], stride: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for s in series_list:
        s = np.asarray(s, dtype=float)
        s = s[np.isfinite(s)]
        for start in range(0, len(s) - SERIES_LEN + 1, max(1, stride)):
            out.append(s[start : start + SERIES_LEN])
    return out


def _predictability(x: np.ndarray, lags: tuple[int, ...]) -> float:
    """Largest |autocorrelation| over a FIXED set of meaningful lags.

    ~0 for white noise (unforecastable); high for anything with short-range
    persistence (lags 1-3) or seasonality (the seasonal-period lags). Using a
    fixed, small lag set — rather than scanning many lags — keeps the statistic
    sample-size-independent: scanning dozens of lags inflates the max on pure noise
    (a spurious peak is likely), which fixed lags avoid. Anchor-independent.
    """
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= _EPS:
        return 0.0
    best = 0.0
    for lag in lags:
        if 1 <= lag < len(x):
            r = abs(float(np.dot(x[:-lag], x[lag:]) / denom))
            best = max(best, r)
    return best


def discrimination(
    series_list: list[np.ndarray],
    *,
    min_predictability: float = 0.15,
    min_naive_error: float = 0.10,
    stride: int = HORIZON,
    max_windows: int = 128,
) -> Discrimination:
    """Behavioural forecastability gate over a source's windows.

    Admits the source only if it is BOTH:

    * **forecastable** — ``predictability >= min_predictability`` (some
      autocorrelation exists; pure noise has ~none and is rejected), and
    * **non-trivial** — ``naive_error >= min_naive_error`` (a seasonal-naive copy
      does NOT already nail it; a clean sine, which naive forecasts perfectly, is
      rejected as it cannot separate models).

    Both signals are anchor-independent by design. The panel ``spread`` is computed
    and reported as a diagnostic but is NOT a hard gate, because it depends on the
    ``strong`` anchor's quality (which on raw real data is exactly what
    ``validate_panel`` flags separately).
    """
    from challenges import Challenge, _normalize_by_context  # local import
    from config import SEASONAL_PERIODS
    from score import _mae, _scale, default_panel, model_errors, seasonal_naive

    lags = (1, 2, 3, *SEASONAL_PERIODS)  # persistence + the seasonal periods

    wins = _windows(series_list, stride)[:max_windows]
    if len(wins) < 4:
        return Discrimination(False, 0.0, 0.0, 0.0, len(wins),
                              [f"too few windows ({len(wins)}) to judge discrimination"])

    challenges, naive_tot, pred_sum = [], 0.0, 0.0
    for w in wins:
        s = _normalize_by_context(w)
        ch = Challenge(context=s[:CONTEXT_LEN], truth=s[CONTEXT_LEN:],
                       mode="admit", meta={"domain": "candidate"})
        challenges.append(ch)
        naive_tot += _mae(seasonal_naive(ch.context), ch.truth) / _scale(ch.context)
        pred_sum += _predictability(ch.context, lags)
    naive_error = naive_tot / len(challenges)
    # Mean over windows (not max): a single window of pure noise will occasionally
    # show a spurious peak, so the max inflates; the mean is the honest signal.
    pred = pred_sum / len(challenges)

    errs = model_errors(challenges, default_panel())
    vals = list(errs.values())
    mean_e = float(np.mean(vals))
    spread = (max(vals) - min(vals)) / mean_e if mean_e > _EPS else 0.0

    reasons: list[str] = []
    if pred < min_predictability:
        reasons.append(f"predictability {pred:.3f} < {min_predictability} "
                       f"(no autocorrelation — likely pure noise, unforecastable)")
    if naive_error < min_naive_error:
        reasons.append(f"seasonal-naive error {naive_error:.3f} < {min_naive_error} "
                       f"(a trivial copy already solves it — non-discriminating)")
    return Discrimination(not reasons, pred, naive_error, spread, len(wins), reasons)


# --------------------------------------------------------------------------- #
# Source-level verdict
# --------------------------------------------------------------------------- #


@dataclass
class SourceQuality:
    ok: bool
    n_series: int
    n_series_ok: int
    per_series: list[SeriesQuality]
    discrimination: Discrimination | None
    reasons: list[str] = field(default_factory=list)


def assess_source(
    series_list: list[np.ndarray],
    *,
    min_series_pass_frac: float = 0.6,
    run_discrimination: bool = True,
    **series_kwargs,
) -> SourceQuality:
    """Automatic admission verdict for a fetched source (no human, no LLM).

    A source is admitted iff a sufficient fraction of its series clear the intrinsic
    battery AND (optionally) the pooled series clear the discrimination filter.
    """
    per = [assess_series(s, **series_kwargs) for s in series_list]
    n_ok = sum(p.ok for p in per)
    reasons: list[str] = []
    frac = n_ok / len(per) if per else 0.0
    if not per:
        reasons.append("no series to assess")
    elif frac < min_series_pass_frac:
        reasons.append(f"only {n_ok}/{len(per)} series pass intrinsic checks "
                       f"(< {min_series_pass_frac:.0%})")

    disc = None
    if run_discrimination and frac >= min_series_pass_frac and per:
        good = [s for s, p in zip(series_list, per) if p.ok]
        disc = discrimination(good)
        if not disc.ok:
            reasons.extend(disc.reasons)

    return SourceQuality(ok=not reasons, n_series=len(per), n_series_ok=n_ok,
                         per_series=per, discrimination=disc, reasons=reasons)


def assess_scraped_source(
    catalog_path: str,
    data_dir: str,
    source_id: str,
    *,
    n_series: int = 12,
    motif_len: int = 768,
    **kwargs,
) -> SourceQuality:
    """Assess a *scraped* source by id, using ``ScrapedLiveSource`` to extract series.

    Goes through the real adapter, so panel series are separated and the densest
    numeric column is chosen — the same series the benchmark would actually serve,
    which is what makes this a faithful admission test (and avoids the pitfalls of
    reading raw parquet columns by hand).
    """
    import numpy as _np

    from scraped_source import ScrapedLiveSource

    src = ScrapedLiveSource(catalog_path, data_dir, min_series_length=SERIES_LEN)
    specs = [s for s in src._catalog() if s.get("source_id") == source_id]
    if not specs:
        return SourceQuality(False, 0, 0, [], None,
                             [f"source '{source_id}' has no usable series under {data_dir}"])
    rng = _np.random.default_rng(0)
    series = [src._extract_motif(spec, motif_len, rng) for spec in specs[:n_series]]
    return assess_source(series, **kwargs)
