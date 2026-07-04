"""Adapters that turn a real time-series foundation model into a Forecaster.

These are what make the benchmark *actually test a TSFM*. Each adapter wraps a
published zero-shot model (Chronos, TimesFM, ...) behind the
:class:`evaluate.ProbForecast` contract, so it drops straight onto the leaderboard
next to the classical panel and is scored on the same MASE / WQL / CRPS footing.

Dependencies are imported lazily, so the numpy-only core, the demo, and the tests
never need torch. Install the extras and provide model weights to enable them:

    pip install -e ".[chronos]"     # torch + chronos-forecasting
    pip install -e ".[timesfm]"     # timesfm

Offline / air-gapped note
-------------------------
The sandbox blocks network for *submissions*; the model under test is loaded by
the **validator**, not the submission, so weights are staged once (e.g.
``HF_HOME`` pointed at a pre-downloaded cache) and loaded from disk. Loading a
TSFM is a trusted, validator-side step -- keep it outside the submission sandbox.
"""

from __future__ import annotations

import numpy as np

from config import HORIZON
from evaluate import DEFAULT_QUANTILES, ProbForecast


def _meta_horizon(meta: dict | None, fallback: int) -> int:
    """Per-cadence profile horizon from challenge meta, else the fallback."""
    if isinstance(meta, dict) and meta.get("horizon"):
        return int(meta["horizon"])
    return fallback


class ChronosForecaster:
    """Amazon Chronos / Chronos-Bolt as a probabilistic :class:`evaluate.Forecaster`.

    Produces quantiles directly from the model's predictive samples/quantiles, so
    WQL and CRPS reflect the model's *real* uncertainty rather than a Gaussian
    stand-in. Lazily constructs the pipeline on first call.
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-bolt-base",
        *,
        device: str = "cpu",
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        horizon: int = HORIZON,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.quantiles = quantiles
        self.horizon = horizon
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is None:
            from chronos import BaseChronosPipeline  # lazy: needs torch

            self._pipeline = BaseChronosPipeline.from_pretrained(
                self.model_name, device_map=self.device
            )
        return self._pipeline

    def __call__(self, context: np.ndarray, meta: dict | None = None) -> ProbForecast:
        import torch

        pipeline = self._ensure_pipeline()
        ctx = torch.tensor(np.asarray(context, dtype=float))
        levels = list(self.quantiles)
        hor = _meta_horizon(meta, self.horizon)
        q_tensor, mean_tensor = pipeline.predict_quantiles(
            ctx, prediction_length=hor, quantile_levels=levels
        )
        q_np = q_tensor[0].cpu().numpy()  # (horizon, n_levels)
        mean = np.asarray(mean_tensor[0].cpu().numpy(), dtype=float)
        quantiles = {lvl: q_np[:, i].astype(float) for i, lvl in enumerate(levels)}
        return ProbForecast(mean=mean, quantiles=quantiles)


class TimesFMForecaster:
    """Google TimesFM as a probabilistic :class:`evaluate.Forecaster`.

    TimesFM returns a point forecast plus experimental quantile heads; we use the
    quantile output when available and otherwise widen the point forecast.
    """

    def __init__(
        self,
        *,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        horizon: int = HORIZON,
        **model_kwargs,
    ) -> None:
        self.quantiles = quantiles
        self.horizon = horizon
        self.model_kwargs = model_kwargs
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            import timesfm  # lazy

            self._model = timesfm.TimesFm(**self.model_kwargs)
        return self._model

    def __call__(self, context: np.ndarray, meta: dict | None = None) -> ProbForecast:
        model = self._ensure_model()
        point_fc, quantile_fc = model.forecast(
            [np.asarray(context, dtype=float)], freq=[0]
        )
        mean = np.asarray(point_fc[0][: self.horizon], dtype=float)
        # TimesFM's experimental quantile output is (horizon, n_quantiles) when on;
        # fall back to a widened band if it is not configured.
        try:
            q_arr = np.asarray(quantile_fc[0], dtype=float)[: self.horizon]
            quantiles = {q: q_arr[:, i] for i, q in enumerate(self.quantiles)}
        except (IndexError, TypeError, ValueError):
            from evaluate import _naive_scale, _probit

            sigma_h = _naive_scale(context) * np.sqrt(np.arange(1, self.horizon + 1))
            quantiles = {q: mean + _probit(q) * sigma_h for q in self.quantiles}
        return ProbForecast(mean=mean, quantiles=quantiles)


class Toto2Forecaster:
    """Datadog Toto-2.0 as a probabilistic :class:`evaluate.Forecaster`.

    ``model.forecast`` returns quantiles of shape (9, batch, n_variates, horizon)
    at exactly the benchmark's decile levels. Median serves as the point forecast.
    """

    def __init__(
        self,
        model_name: str = "Datadog/Toto-2.0-313m",
        *,
        device: str = "cuda",
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        horizon: int = HORIZON,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.quantiles = quantiles
        self.horizon = horizon
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            import torch
            from toto2 import Toto2Model  # lazy: needs torch

            device = self.device if torch.cuda.is_available() else "cpu"
            self._model = Toto2Model.from_pretrained(self.model_name).to(device).eval()
            self._device = device
        return self._model

    def __call__(self, context: np.ndarray, meta: dict | None = None) -> ProbForecast:
        import torch

        model = self._ensure_model()
        hor = _meta_horizon(meta, self.horizon)
        ctx = np.asarray(context, dtype=np.float32)
        target = torch.tensor(ctx, device=self._device).reshape(1, 1, -1)
        batch = {
            "target": target,
            "target_mask": torch.ones_like(target, dtype=torch.bool),
            "series_ids": torch.zeros(1, 1, dtype=torch.long, device=self._device),
        }
        with torch.no_grad():
            q = model.forecast(batch, horizon=hor, decode_block_size=768,
                               has_missing_values=False)
        q_np = np.asarray(q.detach().cpu().numpy(), dtype=float)[:, 0, 0, :]  # (9, horizon)
        quantiles = {lvl: q_np[i] for i, lvl in enumerate(self.quantiles)}
        return ProbForecast(mean=q_np[4], quantiles=quantiles)


class TiRexForecaster:
    """NX-AI TiRex as a probabilistic :class:`evaluate.Forecaster`.

    ``model.forecast`` returns (quantiles, mean); the quantile grid defaults to
    the benchmark's nine deciles.
    """

    def __init__(
        self,
        model_name: str = "NX-AI/TiRex",
        *,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        horizon: int = HORIZON,
    ) -> None:
        self.model_name = model_name
        self.quantiles = quantiles
        self.horizon = horizon
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from tirex import load_model  # lazy: needs torch

            self._model = load_model(self.model_name)
        return self._model

    def __call__(self, context: np.ndarray, meta: dict | None = None) -> ProbForecast:
        import torch

        model = self._ensure_model()
        hor = _meta_horizon(meta, self.horizon)
        ctx = torch.tensor(np.asarray(context, dtype=np.float32)).reshape(1, -1)
        q, mean = model.forecast(context=ctx, prediction_length=hor)
        q_np = np.asarray(torch.as_tensor(q).detach().cpu().numpy(), dtype=float)[0]
        if q_np.shape[0] == len(self.quantiles):        # (9, horizon)
            pass
        elif q_np.shape[-1] == len(self.quantiles):     # (horizon, 9)
            q_np = q_np.T
        else:
            raise ValueError(f"unexpected TiRex quantile shape {q_np.shape}")
        quantiles = {lvl: q_np[i] for i, lvl in enumerate(self.quantiles)}
        mean_np = np.asarray(torch.as_tensor(mean).detach().cpu().numpy(), dtype=float)[0]
        return ProbForecast(mean=mean_np, quantiles=quantiles)


_REGISTRY = {
    "chronos": ChronosForecaster,
    "chronos-bolt": ChronosForecaster,
    "timesfm": TimesFMForecaster,
    "toto2": Toto2Forecaster,
    "tirex": TiRexForecaster,
}


def load_tsfm(name: str, **kwargs):
    """Construct a TSFM adapter by short name (``chronos`` / ``timesfm``).

    The adapter is returned without touching torch; the heavy import happens on
    first forecast. Raises ``KeyError`` for an unknown name.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown TSFM {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)
