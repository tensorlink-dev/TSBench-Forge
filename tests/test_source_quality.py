"""Automatic data-quality admission gate (source_discovery.quality).

Covers the intrinsic per-series checks (nan/inf, variance, cardinality, flatline,
spike) and the behavioural discrimination filter (reject pure noise and trivially
periodic; admit genuine structure), on constructed series with known verdicts —
plus a smoke test through the real scraped adapter.
"""

from __future__ import annotations

import os

import numpy as np

from source_discovery import quality

CATALOG = os.path.join(os.path.dirname(__file__), os.pardir, "src", "sources", "sources.yaml")
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sources_data")


# --------------------------------------------------------------------------- #
# Intrinsic per-series checks
# --------------------------------------------------------------------------- #


def test_good_series_passes_intrinsic() -> None:
    rng = np.random.default_rng(0)
    x = 0.02 * np.arange(600) + 2 * np.sin(np.arange(600) * 2 * np.pi / 24) + np.cumsum(
        rng.normal(0, 0.3, 600)
    )
    q = quality.assess_series(x)
    assert q.ok, q.failures


def test_constant_and_near_constant_rejected() -> None:
    assert not quality.assess_series(np.ones(400) * 5).ok
    near = np.ones(400) * 5 + np.random.default_rng(0).normal(0, 1e-9, 400)
    r = quality.assess_series(near)
    assert not r.ok and any("constant" in f for f in r.failures)


def test_nan_inf_fraction_rejected() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 400)
    x[rng.random(400) < 0.8] = np.nan
    x[5] = np.inf
    r = quality.assess_series(x)
    assert not r.ok and any("finite_frac" in f for f in r.failures)


def test_degenerate_cardinality_rejected() -> None:
    # A clean sine has very few distinct rounded values relative to length.
    sine = np.sin(np.arange(400) * 2 * np.pi / 24)
    r = quality.assess_series(sine)
    assert not r.ok and any("cardinality" in f for f in r.failures)


def test_single_spike_domination_rejected() -> None:
    x = np.concatenate([np.ones(399) * 5.0 + np.random.default_rng(0).normal(0, 0.1, 399),
                        [1e6]])
    r = quality.assess_series(x)
    assert not r.ok and any("spike" in f for f in r.failures)


def test_too_short_rejected() -> None:
    r = quality.assess_series(np.random.default_rng(0).normal(0, 1, 100))
    assert not r.ok and any("length" in f for f in r.failures)


def test_snr_and_flatness_are_diagnostics_not_hard_gates() -> None:
    # Genuinely noisy but finite/varied series must NOT be hard-rejected on SNR.
    rng = np.random.default_rng(2)
    noisy = np.cumsum(rng.normal(0, 1, 600)) + rng.normal(0, 3, 600)
    q = quality.assess_series(noisy)
    assert q.ok, q.failures
    assert "snr_db" in q.metrics and "spectral_flatness" in q.metrics


# --------------------------------------------------------------------------- #
# Discrimination filter
# --------------------------------------------------------------------------- #


def _long(kind: str, n: int = 1200) -> np.ndarray:
    rng = np.random.default_rng(3)
    t = np.arange(n)
    if kind == "good":
        return 0.02 * t + 2 * np.sin(t * 2 * np.pi / 24) + np.cumsum(rng.normal(0, 0.3, n))
    if kind == "random_walk":
        return np.cumsum(rng.normal(0, 1, n))
    if kind == "noise":
        return rng.normal(0, 1, n)
    if kind == "sine":
        return np.sin(t * 2 * np.pi / 24)
    raise ValueError(kind)


def test_discrimination_admits_structure() -> None:
    for kind in ("good", "random_walk"):
        d = quality.discrimination([_long(kind)])
        assert d.ok, (kind, d.reasons)


def test_discrimination_rejects_pure_noise() -> None:
    d = quality.discrimination([_long("noise")])
    assert not d.ok and any("predictability" in r for r in d.reasons)


def test_discrimination_rejects_trivially_periodic() -> None:
    d = quality.discrimination([_long("sine")])
    assert not d.ok and any("trivial" in r or "naive" in r for r in d.reasons)


def test_discrimination_needs_enough_windows() -> None:
    d = quality.discrimination([np.random.default_rng(0).normal(0, 1, quality.SERIES_LEN)])
    assert not d.ok and any("too few windows" in r for r in d.reasons)


# --------------------------------------------------------------------------- #
# Source-level verdict + real-adapter smoke
# --------------------------------------------------------------------------- #


def test_assess_source_aggregates() -> None:
    good = [_long("good"), _long("random_walk")]
    q = quality.assess_source(good)
    assert q.ok and q.n_series_ok == 2

    bad = [np.ones(400) * 3, np.sin(np.arange(400) * 2 * np.pi / 24)]
    qb = quality.assess_source(bad)
    assert not qb.ok


def test_assess_scraped_source_admits_a_good_real_source() -> None:
    # aemo grid demand is a strong, structured real feed — must be admitted.
    q = quality.assess_scraped_source(CATALOG, FIXTURE, "aemo_nem_5min", n_series=4)
    assert q.ok, q.reasons
    assert q.discrimination is not None and q.discrimination.predictability > 0.3
