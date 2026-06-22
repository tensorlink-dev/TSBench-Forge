"""TSFM adapter wiring: lazy construction, registry, and the Forecaster contract.

The real models (torch/chronos/timesfm) are not installed in CI; these tests
verify the adapter plumbing without them, and exercise the ProbForecast contract
with a TSFM-shaped fake so the end-to-end scoring path is proven model-agnostic.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from config import HORIZON
from evaluate import DEFAULT_QUANTILES, ProbForecast, evaluate_forecaster, leaderboard
from tsfm_adapters import ChronosForecaster, TimesFMForecaster, load_tsfm


def test_load_tsfm_returns_adapters_by_name() -> None:
    assert isinstance(load_tsfm("chronos"), ChronosForecaster)
    assert isinstance(load_tsfm("chronos-bolt"), ChronosForecaster)
    assert isinstance(load_tsfm("timesfm"), TimesFMForecaster)


def test_load_tsfm_unknown_raises() -> None:
    with pytest.raises(KeyError):
        load_tsfm("not-a-model")


def test_construction_is_lazy_no_torch_import() -> None:
    # Building the adapter must not import torch/chronos (heavy, optional).
    load_tsfm("chronos")
    load_tsfm("timesfm")
    assert "torch" not in sys.modules
    assert "chronos" not in sys.modules


def test_scoring_path_is_model_agnostic_with_a_tsfm_shaped_fake() -> None:
    # A fake that emits a mean + quantiles exactly like a real TSFM adapter would,
    # proving the leaderboard/metrics work for any quantile-emitting model.
    class FakeTSFM:
        def __call__(self, context, meta=None):
            last = float(context[-1])
            mean = np.full(HORIZON, last)
            spread = np.linspace(0.1, 1.0, HORIZON)
            return ProbForecast(
                mean=mean,
                quantiles={q: mean + (q - 0.5) * spread for q in DEFAULT_QUANTILES},
            )

    chs = [
        SimpleNamespace(context=np.linspace(0, 10, 256),
                        truth=np.linspace(10, 12, HORIZON), meta=None)
        for _ in range(8)
    ]
    m = evaluate_forecaster(FakeTSFM(), chs)
    assert m["mase"] > 0 and m["wql"] > 0 and m["n"] == 8

    board = leaderboard({"fake_tsfm": FakeTSFM()}, chs)
    assert board[0]["model"] == "fake_tsfm" and board[0]["rank"] == 1
