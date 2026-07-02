"""Model evaluation: metrics, leaderboard ranking, and headroom detection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from config import HORIZON
from evaluate import (
    ProbForecast,
    _naive_scale,
    _probit,
    benchmark_has_headroom,
    evaluate_forecaster,
    headroom,
    leaderboard,
    probabilistic,
    probabilistic_panel,
    season_length,
)
from score import default_panel


def _challenges(n: int = 40, seed: int = 0):
    """Structured series (trend + seasonal + noise) with known continuation.

    Each carries its own future in ``meta['truth_for_test']`` so a 'perfect'
    candidate can be expressed as a forecaster, used to prove headroom.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        total = 256 + HORIZON
        t = np.arange(total)
        series = 0.05 * t + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.4, total)
        context, truth = series[:256], series[256:]
        out.append(SimpleNamespace(context=context, truth=truth,
                                   meta={"truth_for_test": truth}))
    return out


def _perfect(context, meta=None):
    truth = meta["truth_for_test"]
    return ProbForecast(mean=truth, quantiles={q: truth.copy() for q in
                        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)})


def test_probit_matches_known_values() -> None:
    assert abs(_probit(0.5)) < 1e-6
    assert abs(_probit(0.975) - 1.959964) < 1e-3
    assert abs(_probit(0.025) + 1.959964) < 1e-3


def test_perfect_forecaster_scores_near_zero() -> None:
    m = evaluate_forecaster(_perfect, _challenges())
    assert m["mase"] < 1e-6
    assert m["wql"] < 1e-6
    assert m["crps"] < 1e-6


def test_metrics_are_positive_for_a_real_model() -> None:
    anchor = probabilistic(default_panel()["strong"])
    m = evaluate_forecaster(anchor, _challenges())
    assert m["mase"] > 0 and m["wql"] > 0 and m["n"] == 40


def test_probabilistic_wrap_emits_ordered_quantiles() -> None:
    fc = probabilistic(default_panel()["drift"])
    out = fc(_challenges(1)[0].context, None)
    assert out.mean.shape == (HORIZON,)
    lo, hi = out.quantiles[0.1], out.quantiles[0.9]
    assert np.all(hi >= lo)  # higher quantile is never below a lower one


def test_leaderboard_ranks_better_model_first() -> None:
    chs = _challenges()
    board = leaderboard({"perfect": _perfect, **probabilistic_panel()}, chs)
    assert board[0]["model"] == "perfect"
    assert board[0]["rank"] == 1
    # The strong baseline should outrank the naive baselines among the rest.
    ranks = {r["model"]: r["rank"] for r in board}
    assert ranks["strong"] < ranks["ar1"]


def test_headroom_is_positive_for_a_superior_model() -> None:
    chs = _challenges()
    h = headroom(_perfect, chs)
    assert h["mase_margin"] > 0
    assert h["wql_margin"] > 0
    assert h["beats_anchor"] is True


def test_headroom_is_nonpositive_for_an_inferior_model() -> None:
    chs = _challenges()

    def flat_bad(context, meta=None):
        const = float(context[0])  # ignore everything: a poor forecast
        arr = np.full(HORIZON, const)
        return ProbForecast(mean=arr, quantiles={q: arr.copy() for q in
                            (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)})

    h = headroom(flat_bad, chs)
    assert h["mase_margin"] <= 0
    assert h["beats_anchor"] is False


def test_benchmark_has_headroom_gate() -> None:
    chs = _challenges()
    assert benchmark_has_headroom(_perfect, chs) is True


def test_season_length_follows_frequency_and_caps_to_context() -> None:
    assert season_length("PT1H", 256) == 24
    assert season_length("PT30M", 256) == 48
    assert season_length("P1M", 256) == 12
    # Cycle longer than the context degrades to non-seasonal, as does unknown.
    assert season_length("PT1M", 256) == 1
    assert season_length("P1D", 256) == 1
    assert season_length(None, 256) == 1
    assert season_length("PT7H", 256) == 1


def test_naive_scale_degrades_when_season_exceeds_context() -> None:
    x = np.arange(10, dtype=float)
    assert _naive_scale(x, 50) == _naive_scale(x, 1)  # not an epsilon blow-up


def test_mase_uses_per_challenge_seasonality_from_freq() -> None:
    """On a strongly periodic series, seasonal-naive scaling (from freq=PT1H,
    period 24) yields a larger MASE than non-seasonal scaling, because the
    in-sample seasonal-naive error (the denominator) is much smaller."""
    rng = np.random.default_rng(3)
    total = 256 + HORIZON
    t = np.arange(total)
    series = 5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.1, total)
    ch = SimpleNamespace(context=series[:256], truth=series[256:],
                         meta={"freq": "PT1H"})
    fc = probabilistic(default_panel()["drift"])
    seasonal = evaluate_forecaster(fc, [ch])["mase"]           # m from freq
    plain = evaluate_forecaster(fc, [ch], seasonality=1)["mase"]
    assert seasonal > plain
    # An untagged challenge falls back to the non-seasonal scale.
    ch_untagged = SimpleNamespace(context=ch.context, truth=ch.truth, meta={})
    assert evaluate_forecaster(fc, [ch_untagged])["mase"] == plain
