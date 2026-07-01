"""Behaviour tests for the equal-weight sampler.

The sampler's whole point is that a domain with 30 sources of 1 DGP class doesn't
get 30x the weight of a domain with 1 source of 1 class. These tests pin that.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from pool_sampler import (
    SeriesMeta, sample_pool, generalisation_gap, equal_domain_aggregate,
    lodo_score, coverage_report, _distribute_slots,
)


def _make(n: int, domain: str, dgp: str, freq: str = "PT1H") -> list[SeriesMeta]:
    return [
        SeriesMeta(
            series_id=f"{domain}_{dgp}_{i}", domain=domain, dgp_class=dgp,
            freq=freq, source_id=f"src_{domain}_{dgp}", length=1000,
        )
        for i in range(n)
    ]


def test_slot_distribution_totals():
    assert _distribute_slots(10, 3) == [4, 3, 3]
    assert sum(_distribute_slots(2000, 7)) == 2000
    assert _distribute_slots(0, 5) == [0, 0, 0, 0, 0]
    assert _distribute_slots(100, 0) == []


def test_equal_weight_per_domain_defies_source_count_skew():
    """A domain with 30 sources but 1 DGP class must get the same pool share as
    a domain with 1 source. This is the whole point."""
    catalog = _make(30, "nature", "weather_field") + _make(1, "healthcare", "vital_statistics")
    rng = np.random.default_rng(0)
    picks = sample_pool(catalog, n_windows=1000, rng=rng)
    dom_counts = Counter(p.domain for p in picks)
    # Should be ~500/500, not ~967/33 (which is what source-count-weighted would give).
    assert 400 <= dom_counts["nature"] <= 600
    assert 400 <= dom_counts["healthcare"] <= 600
    assert abs(dom_counts["nature"] - dom_counts["healthcare"]) <= 2  # deterministic split


def test_equal_weight_per_dgp_class_within_domain():
    """Within `nature`, two DGP classes should get equal share, not one dominating
    just because it has more sources."""
    catalog = _make(20, "nature", "weather_field") + _make(2, "nature", "seismic_event")
    rng = np.random.default_rng(42)
    picks = sample_pool(catalog, n_windows=1000, rng=rng)
    cls_counts = Counter(p.dgp_class for p in picks)
    assert abs(cls_counts["weather_field"] - cls_counts["seismic_event"]) <= 2


def test_empty_dgp_class_slots_redistribute_within_domain():
    """If a domain has one DGP class with entries and one without, all its slots
    go to the class that has entries — never a division-by-zero."""
    catalog = _make(5, "nature", "weather_field")  # only one class present
    picks = sample_pool(catalog, n_windows=500, rng=np.random.default_rng(0))
    assert len(picks) == 500
    assert all(p.dgp_class == "weather_field" for p in picks)


def test_lodo_ablation_drops_held_out_domain_from_pool():
    catalog = (
        _make(5, "nature", "weather_field")
        + _make(5, "healthcare", "vital_statistics")
        + _make(5, "energy", "grid_demand")
    )
    picks = sample_pool(
        catalog, n_windows=300, rng=np.random.default_rng(0),
        domains=["nature", "energy"],   # hold out healthcare
    )
    assert all(p.domain in {"nature", "energy"} for p in picks)


def test_deterministic_under_seed():
    catalog = _make(3, "nature", "x") + _make(3, "energy", "y")
    a = sample_pool(catalog, 100, rng=np.random.default_rng(123))
    b = sample_pool(catalog, 100, rng=np.random.default_rng(123))
    assert [x.series_id for x in a] == [x.series_id for x in b]


def test_cadence_balance_when_enabled():
    """With `enforce_cadence_balance=True`, a domain that has both hourly and
    daily series should give equal share to each band."""
    catalog = (
        _make(10, "nature", "weather_field", freq="PT1H")   # hourly
        + _make(1, "nature", "weather_field", freq="P1D")   # daily
    )
    picks = sample_pool(
        catalog, n_windows=200, rng=np.random.default_rng(0),
        enforce_cadence_balance=True,
    )
    band_counts = Counter(p.cadence_band for p in picks)
    # 200 slots -> 100 each band under equal weight, despite 10:1 source ratio.
    assert abs(band_counts["hourly"] - band_counts["daily"]) <= 2


def test_generalisation_gap():
    scores = {"nature": 0.7, "energy": 0.72, "healthcare": 0.68}
    assert generalisation_gap(scores) == pytest.approx(0.04, abs=1e-9)
    scores2 = {"nature": 0.5, "energy": 0.9, "healthcare": 1.4}
    assert generalisation_gap(scores2) == pytest.approx(0.9, abs=1e-9)


def test_equal_domain_aggregate_ignores_series_count():
    """The aggregate must ignore how many series each domain contributed —
    otherwise the scoring reverses the whole point of the sampler."""
    scores = {"nature": 1.0, "healthcare": 3.0}
    # If we row-weighted by (say) 30:1 series, we'd get ~1.06. Equal-weight
    # gives 2.0.
    assert equal_domain_aggregate(scores) == pytest.approx(2.0)


def test_lodo_reports_gap_between_held_out_and_in_distribution():
    scores = {"nature": 0.5, "energy": 0.55, "healthcare": 1.4}
    ho, id_mean = lodo_score(scores, held_out="healthcare")
    assert ho == 1.4
    assert id_mean == pytest.approx((0.5 + 0.55) / 2)


def test_coverage_report_captures_expected_dimensions():
    catalog = (
        _make(3, "nature", "weather_field")
        + _make(2, "nature", "seismic_event")
        + _make(4, "energy", "grid_demand")
    )
    rep = coverage_report(catalog)
    assert rep["n_series"] == 9
    assert rep["n_domains"] == 2
    assert rep["n_dgp_classes"] == 3
    assert rep["series_per_dgp_class"] == {
        "weather_field": 3, "seismic_event": 2, "grid_demand": 4,
    }
