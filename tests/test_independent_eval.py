"""Tests for the independent-validation harness — offline via injected fetch.

The real anchor resolution (statsforecast / TSFM) is environment-dependent, so
those tests assert the *contract* (a runnable anchor with provenance), not a
specific kind. Everything else is hermetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import CONTEXT_LEN, HORIZON
from evaluate import ProbForecast, probabilistic
from generate import Challenge
from independent_eval import (
    VALIDATION_REGISTRY,
    ResolvedAnchor,
    _is_degenerate,
    _point_from_tsfm,
    is_independently_validated,
    leakage_gap,
    resolve_anchor,
    validated_panel,
    validation_benchmark,
)


def _fetch_for_validation(url: str) -> str:
    """A long, non-degenerate sinusoid+noise CSV matching whichever spec asked."""
    spec = next(s for s in VALIDATION_REGISTRY.values() if s["url"] == url)
    rng = np.random.default_rng(0)
    n = 2000
    t = np.arange(n)
    series = 50.0 + 10.0 * np.sin(t / 12.0) + np.cumsum(rng.normal(0, 0.3, n))
    rows = "\n".join(f"2020-01-01,{v:.4f}" for v in series)
    return f"Date,{spec['value_column']}\n{rows}"


def _make_challenges(n: int, seed: int = 0) -> list[Challenge]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        full = 20.0 + 5.0 * np.sin(np.arange(CONTEXT_LEN + HORIZON) / 9.0) + np.cumsum(
            rng.normal(0, 0.4, CONTEXT_LEN + HORIZON)
        )
        out.append(
            Challenge(
                context=full[:CONTEXT_LEN],
                truth=full[CONTEXT_LEN:],
                mode="real_holdout",
                meta={"domain": "test"},
            )
        )
    return out


def test_is_degenerate_flags_constant_context():
    # A constant context collapses the MASE denominator regardless of the truth tail.
    assert _is_degenerate(np.full(CONTEXT_LEN + HORIZON, 5.0))
    flat_ctx = np.concatenate([np.full(CONTEXT_LEN, 5.0), np.arange(HORIZON, dtype=float)])
    assert _is_degenerate(flat_ctx)
    assert not _is_degenerate(
        np.sin(np.arange(CONTEXT_LEN + HORIZON) / 5.0) + np.arange(CONTEXT_LEN + HORIZON)
    )


def test_validation_benchmark_offline_geometry():
    bench = validation_benchmark(n_per_domain=10, fetch=_fetch_for_validation)
    assert len(bench) == 10 * len(VALIDATION_REGISTRY)
    domains = {c.meta["domain"] for c in bench}
    assert domains == set(VALIDATION_REGISTRY)
    for ch in bench:
        assert ch.context.shape == (CONTEXT_LEN,)
        assert ch.truth.shape == (HORIZON,)
        assert np.isfinite(ch.context).all() and np.isfinite(ch.truth).all()


def test_validation_benchmark_returns_only_well_conditioned_windows():
    # Invariant that protects MASE: no challenge in the set has a degenerate context.
    bench = validation_benchmark(n_per_domain=10, fetch=_fetch_for_validation)
    assert bench
    for ch in bench:
        full = np.concatenate([ch.context, ch.truth])
        assert not _is_degenerate(full)


def test_independent_validation_true_for_a_good_model():
    bench = _make_challenges(40, seed=1)
    truth_by_id = {id(c.context): np.asarray(c.truth, float) for c in bench}

    def near_oracle(context, meta=None):
        tgt = truth_by_id[id(context)]
        noisy = tgt + np.random.default_rng(0).normal(0, 0.02 * (tgt.std() + 1e-8), tgt.shape)
        d = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        return ProbForecast(mean=noisy, quantiles={q: noisy for q in d})

    res = is_independently_validated(near_oracle, name="oracle", benchmark=bench)
    assert res.validated is True
    assert res.beaten_by == []
    assert res.margin > 0


def test_independent_validation_false_for_a_bad_model():
    bench = _make_challenges(40, seed=2)

    def constant_zero(context, meta=None):
        z = np.zeros(HORIZON)
        d = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        return ProbForecast(mean=z, quantiles={q: z for q in d})

    res = is_independently_validated(constant_zero, name="bad", benchmark=bench)
    assert res.validated is False
    assert res.beaten_by  # at least one baseline beats it


def test_point_from_tsfm_extracts_mean():
    def fake_tsfm(context, meta=None):
        mean = np.arange(HORIZON, dtype=float)
        return ProbForecast(mean=mean, quantiles={0.5: mean})

    point = _point_from_tsfm(fake_tsfm)
    out = point(np.zeros(CONTEXT_LEN))
    assert np.array_equal(out, np.arange(HORIZON, dtype=float))


def test_resolve_anchor_returns_runnable_anchor_with_provenance():
    a = resolve_anchor()
    assert isinstance(a, ResolvedAnchor)
    assert a.kind in {"tsfm", "statsforecast", "numpy"}
    out = a.point_model(np.linspace(0, 1, CONTEXT_LEN))
    assert np.asarray(out).shape == (HORIZON,)


def test_resolve_anchor_falls_back_when_no_tsfm():
    # An unknown preferred TSFM must degrade to a non-tsfm anchor, never raise.
    a = resolve_anchor(prefer_tsfm="definitely-not-a-real-model")
    assert a.kind in {"statsforecast", "numpy"}


def test_validated_panel_swaps_strong():
    a = ResolvedAnchor(
        kind="numpy",
        detail="test",
        point_model=lambda c, m=None: np.zeros(HORIZON),
        forecaster=probabilistic(lambda c, m=None: np.zeros(HORIZON)),
    )
    panel, used = validated_panel(a)
    assert used is a
    assert panel["strong"] is a.point_model
    assert "overfit" in panel  # the rest of the panel is intact


def test_leakage_gap_shape():
    bench = _make_challenges(12, seed=3)
    reveal = _make_challenges(12, seed=4)
    anchor = probabilistic(lambda c, m=None: np.full(HORIZON, float(np.mean(c[-12:]))))
    gap = leakage_gap(anchor, reveal, benchmark=bench)
    assert set(gap) == {"static_mase", "forge_mase", "gap"}
    assert gap["gap"] == pytest.approx(gap["forge_mase"] - gap["static_mase"])
