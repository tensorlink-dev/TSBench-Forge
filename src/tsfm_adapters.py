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


class Chronos2Forecaster:
    """Amazon Chronos-2 (``amazon/chronos-2``) as a probabilistic Forecaster.

    Chronos-2 natively emits quantiles (trained on a 21-quantile grid), so the
    deciles come straight from ``predict_quantiles`` with no sampling. The
    ``mean`` return of that API is the **median** — we surface it as the point
    forecast, consistent with the other quantile-native adapters here.
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-2",
        *,
        device: str = "cuda",
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        horizon: int = HORIZON,
        max_context: int = 8192,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.quantiles = quantiles
        self.horizon = horizon
        self.max_context = max_context
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            from chronos import Chronos2Pipeline  # lazy: chronos-forecasting>=2.0

            self._pipe = Chronos2Pipeline.from_pretrained(
                self.model_name, device_map=self.device
            )
        return self._pipe

    def __call__(self, context: np.ndarray, meta: dict | None = None) -> ProbForecast:
        pipe = self._ensure()
        hor = _meta_horizon(meta, self.horizon)
        ctx = np.asarray(context, dtype=np.float32)[-self.max_context :]
        q, m = pipe.predict_quantiles(
            [ctx], prediction_length=hor, quantile_levels=list(self.quantiles)
        )
        q0 = np.asarray(q[0].float().cpu().numpy(), dtype=float)  # (n_var, H, 9)
        if q0.ndim == 3:
            q0 = q0[0]                                            # (H, 9)
        mean = np.asarray(m[0].float().cpu().numpy(), dtype=float)
        mean = mean[0] if mean.ndim == 2 else mean               # (H,)
        quantiles = {lvl: q0[:, i].astype(float) for i, lvl in enumerate(self.quantiles)}
        return ProbForecast(mean=mean, quantiles=quantiles)


class TimesFM25Forecaster:
    """Google TimesFM-2.5 (``google/timesfm-2.5-200m-pytorch``) as a Forecaster.

    2.5 dropped the ``freq`` argument and gained a continuous-quantile head. The
    model must be ``compile``d once with a horizon/context ceiling; we compile
    lazily on the first call using a ceiling derived from that call's horizon.
    """

    def __init__(
        self,
        model_name: str = "google/timesfm-2.5-200m-pytorch",
        *,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        horizon: int = HORIZON,
        max_context: int = 2048,
        max_horizon: int = 256,
    ) -> None:
        self.model_name = model_name
        self.quantiles = quantiles
        self.horizon = horizon
        self.max_context = max_context
        self.max_horizon = max_horizon
        self._model = None

    def _ensure(self):
        if self._model is None:
            import timesfm  # lazy

            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_name)
            model.compile(
                timesfm.ForecastConfig(
                    max_context=self.max_context,
                    max_horizon=self.max_horizon,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=False,
                    fix_quantile_crossing=True,
                )
            )
            self._model = model
        return self._model

    def __call__(self, context: np.ndarray, meta: dict | None = None) -> ProbForecast:
        model = self._ensure()
        hor = min(_meta_horizon(meta, self.horizon), self.max_horizon)
        ctx = np.asarray(context, dtype=float)[-self.max_context :]
        point, quantiles = model.forecast(horizon=hor, inputs=[ctx])
        point = np.asarray(point, dtype=float)[0]                 # (H,)
        q_arr = np.asarray(quantiles, dtype=float)[0]             # (H, 10): idx0=mean, 1..9=deciles
        qs = {lvl: q_arr[:, i + 1] for i, lvl in enumerate(self.quantiles)}
        return ProbForecast(mean=point, quantiles=qs)


class Moirai2Forecaster:
    """Salesforce Moirai-2.0 (``Salesforce/moirai-2.0-R-small``) as a Forecaster.

    Moirai-2 is quantile-based (deterministic decile heads, no sampling). The
    pretrained *module* is cached; the thin ``Moirai2Forecast`` wrapper is rebuilt
    per call because context/prediction length are constructor arguments.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/moirai-2.0-R-small",
        *,
        device: str = "cuda",
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        horizon: int = HORIZON,
        max_context: int = 8000,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.quantiles = quantiles
        self.horizon = horizon
        self.max_context = max_context
        self._module = None

    def _ensure(self):
        if self._module is None:
            import torch
            from uni2ts.model.moirai2 import Moirai2Module  # lazy

            self._module = Moirai2Module.from_pretrained(self.model_name)
            self._device = self.device if torch.cuda.is_available() else "cpu"
        return self._module

    def __call__(self, context: np.ndarray, meta: dict | None = None) -> ProbForecast:
        import torch
        from uni2ts.model.moirai2 import Moirai2Forecast

        module = self._ensure()
        hor = _meta_horizon(meta, self.horizon)
        ctx = np.asarray(context, dtype=np.float32)[-self.max_context :]
        model = (
            Moirai2Forecast(
                module=module,
                prediction_length=hor,
                context_length=len(ctx),
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            .to(self._device)
            .eval()
        )
        t = torch.tensor(ctx, device=self._device)[None, :, None]     # [1, C, 1]
        obs = torch.ones_like(t, dtype=torch.bool)
        pad = torch.zeros(1, t.shape[1], dtype=torch.bool, device=self._device)
        with torch.no_grad():
            out = model(past_target=t, past_observed_target=obs, past_is_pad=pad)
        q = np.asarray(out[0].float().cpu().numpy(), dtype=float)      # (9, H, 1)
        q = q[..., 0] if q.ndim == 3 else q                           # (9, H)
        quantiles = {lvl: q[i] for i, lvl in enumerate(self.quantiles)}
        return ProbForecast(mean=q[len(self.quantiles) // 2], quantiles=quantiles)


class FlowStateForecaster:
    """IBM Granite FlowState (``ibm-granite/granite-timeseries-flowstate-r1``).

    A small (9M-param) SSM/functional-basis model with native decile heads. The
    output layout is normalised defensively (quantile axis found by matching the
    decile count) because the published example does not pin the axis order.
    """

    def __init__(
        self,
        model_name: str = "ibm-granite/granite-timeseries-flowstate-r1",
        *,
        revision: str = "r1.1",
        device: str = "cuda",
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        horizon: int = HORIZON,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.quantiles = quantiles
        self.horizon = horizon
        self._model = None

    def _ensure(self):
        if self._model is None:
            import torch
            from tsfm_public import FlowStateForPrediction  # lazy: granite-tsfm

            device = self.device if torch.cuda.is_available() else "cpu"
            self._model = (
                FlowStateForPrediction.from_pretrained(self.model_name, revision=self.revision)
                .to(device)
                .eval()
            )
            self._device = device
        return self._model

    def __call__(self, context: np.ndarray, meta: dict | None = None) -> ProbForecast:
        import torch

        model = self._ensure()
        hor = _meta_horizon(meta, self.horizon)
        ctx = np.asarray(context, dtype=np.float32)
        x = torch.tensor(ctx, device=self._device).reshape(-1, 1, 1)   # (T, B=1, C=1)
        with torch.no_grad():
            out = model(x, scale_factor=1.0, prediction_length=hor, batch_first=False)
        arr = np.asarray(torch.as_tensor(out).float().cpu().numpy(), dtype=float)
        arr = np.squeeze(arr)                                          # drop size-1 dims
        nq = len(self.quantiles)
        # Orient so axis 0 is the quantile (decile) axis of length nq.
        if arr.ndim == 1:
            arr = np.repeat(arr[None, :], nq, axis=0)                  # point-only fallback
        elif arr.shape[0] != nq:
            qaxes = [ax for ax, s in enumerate(arr.shape) if s == nq]
            if qaxes:
                arr = np.moveaxis(arr, qaxes[0], 0)
        arr = arr.reshape(nq, -1)[:, :hor]                            # (nq, H)
        quantiles = {lvl: arr[i] for i, lvl in enumerate(self.quantiles)}
        return ProbForecast(mean=arr[nq // 2], quantiles=quantiles)


class SundialForecaster:
    """Tsinghua Sundial (``thuml/sundial-base-128m``) as a Forecaster.

    Generative / flow-matching: draws ``num_samples`` trajectories and reduces
    them to deciles. NOTE Sundial pins ``transformers==4.40.1`` — install it in a
    dedicated environment; it does not coexist with the newer-transformers models.
    """

    def __init__(
        self,
        model_name: str = "thuml/sundial-base-128m",
        *,
        device: str = "cuda",
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        horizon: int = HORIZON,
        num_samples: int = 100,
        seed: int = 0,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.quantiles = quantiles
        self.horizon = horizon
        self.num_samples = num_samples
        self.seed = seed
        self._model = None

    def _ensure(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM  # lazy

            device = self.device if torch.cuda.is_available() else "cpu"
            self._model = (
                AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
                .to(device)
                .eval()
            )
            self._device = device
        return self._model

    def __call__(self, context: np.ndarray, meta: dict | None = None) -> ProbForecast:
        import torch

        model = self._ensure()
        torch.manual_seed(self.seed)
        hor = _meta_horizon(meta, self.horizon)
        ctx = np.asarray(context, dtype=np.float32)
        x = torch.tensor(ctx, device=self._device).unsqueeze(0)       # (1, T)
        with torch.no_grad():
            out = model.generate(x, max_new_tokens=hor, num_samples=self.num_samples)
        s = np.asarray(torch.as_tensor(out).float().cpu().numpy(), dtype=float)
        s = np.squeeze(s)                                             # (H, N) or (N, H)
        # sample axis = the one whose length is num_samples (fall back to axis 0)
        sample_ax = next((ax for ax, n in enumerate(s.shape) if n == self.num_samples), 0)
        hor_ax = 1 - sample_ax
        s = np.moveaxis(s, hor_ax, 0)[:hor]                           # (H, N)
        quantiles = {lvl: np.quantile(s, lvl, axis=1) for lvl in self.quantiles}
        return ProbForecast(mean=s.mean(axis=1), quantiles=quantiles)


class TabPFNTSForecaster:
    """PriorLabs TabPFN-TS (``tabpfn-time-series``) as a Forecaster.

    TabPFN-TS wants a one-item ``TimeSeriesDataFrame`` and emits decile columns
    directly. Requires ``tabpfn>=8.0.0`` and a one-time license acceptance at
    ux.priorlabs.ai. Timestamps are synthesised as a plain range — TabPFN-TS adds
    its own calendar features, and our motifs carry no absolute time anyway.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        horizon: int = HORIZON,
    ) -> None:
        self.device = device
        self.quantiles = quantiles
        self.horizon = horizon
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline  # lazy

            self._pipe = TabPFNTSPipeline(
                tabpfn_mode=TabPFNMode.LOCAL, tabpfn_output_selection="median"
            )
        return self._pipe

    def __call__(self, context: np.ndarray, meta: dict | None = None) -> ProbForecast:
        import pandas as pd

        pipe = self._ensure()
        hor = _meta_horizon(meta, self.horizon)
        ctx = np.asarray(context, dtype=float)
        ts = pd.date_range("2000-01-01", periods=len(ctx), freq="h")
        ctx_df = pd.DataFrame({"item_id": 0, "timestamp": ts, "target": ctx})
        pred = pipe.predict_df(ctx_df, prediction_length=hor)
        quantiles = {lvl: np.asarray(pred[str(lvl)], dtype=float)[:hor] for lvl in self.quantiles}
        mean = np.asarray(pred["0.5"], dtype=float)[:hor]
        return ProbForecast(mean=mean, quantiles=quantiles)


_REGISTRY = {
    "chronos": ChronosForecaster,
    "chronos-bolt": ChronosForecaster,
    "chronos2": Chronos2Forecaster,
    "chronos-2": Chronos2Forecaster,
    "timesfm": TimesFMForecaster,
    "timesfm25": TimesFM25Forecaster,
    "timesfm-2.5": TimesFM25Forecaster,
    "toto2": Toto2Forecaster,
    "tirex": TiRexForecaster,
    "moirai2": Moirai2Forecaster,
    "moirai-2": Moirai2Forecaster,
    "flowstate": FlowStateForecaster,
    "sundial": SundialForecaster,
    "tabpfn-ts": TabPFNTSForecaster,
    "tabpfn": TabPFNTSForecaster,
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
