"""DSR evaluation runner: the protocol that turns a TSFM into a Table-2 row.

Protocol (per model x system x seed), following Durstewitz et al. (arXiv:2602.16864):

1. Show the forecaster the context window.
2. Free-run it autoregressively (``dsr_metrics.free_run``) to the length of the long
   ground-truth rollout -- a *zero-shot* generation, no fine-tuning.
3. Discard the initial transient (default 10%) from both the generated and the true
   long rollouts.
4. Score: ``D_stsp`` (state-space divergence), ``D_H`` (power-spectrum Hellinger),
   ``lambda_max`` of the generated series, ``VPT`` (valid prediction time in Lyapunov
   units), and ``MASE`` -- kept deliberately, to show how a point metric saturates
   while the dynamics metrics keep discriminating.

Models stay behind the repo's existing :class:`evaluate.Forecaster` abstraction, so any
registered TSFM (``tsfm_adapters.load_tsfm``) or classical panel model works with no
per-model code. Run programmatically via :func:`run_dsr_eval` or from the CLI
(``python -m dsr_eval --models seasonal_naive --systems lorenz rossler``).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from dsr_eval.datasets import (
    DEFAULT_CONTEXT_LEN,
    DEFAULT_LONG_LEN,
    DEFAULT_TRANSIENT_FRAC,
    DSRSample,
    build_sample,
)
from dsr_eval.metrics import (
    free_run,
    max_lyapunov_rate,
    power_spectrum_hellinger,
    state_space_divergence,
    valid_prediction_time,
    vpt_lyapunov,
)
from dsr_eval.systems import get_system
from evaluate import Forecaster, _naive_scale, probabilistic


@dataclass(frozen=True)
class MetricConfig:
    """Knobs that must be held FIXED across runs for the metrics to be comparable.

    ``D_stsp`` depends on ``embed_dim`` / ``bins`` / ``n_components`` and ``D_H`` on
    ``sigma``; see the module caveats. ``ssp_seed`` fixes the Monte-Carlo draw of the
    GMM estimator so the divergence is reproducible.
    """

    embed_dim: int = 3
    tau: int | None = None
    bins: int = 20
    estimator: str = "auto"
    n_components: int = 8
    n_mc: int = 10000
    sigma: int = 20
    vpt_threshold: float = 0.4
    ssp_seed: int = 0
    # Rosenstein fit window for lambda_max(gen). None -> derive ~1 Lyapunov time from
    # the system (a fixed short window badly overestimates the exponent).
    lyap_horizon: int | None = None


# --------------------------------------------------------------------------- #
# Model resolution -- stay behind the existing Forecaster abstraction
# --------------------------------------------------------------------------- #


def resolve_forecaster(spec: str | Forecaster) -> Forecaster:
    """Resolve a model spec to a :class:`evaluate.Forecaster`.

    Accepts a callable forecaster as-is, a registered TSFM name
    (``tsfm_adapters.load_tsfm``: ``chronos`` / ``timesfm`` / ...), or a classical
    reference-panel / floor-baseline name (``seasonal_naive``, ``drift``, ``ewma``,
    ``ar1``, ``strong``, ``context_parrot``) lifted to a probabilistic forecaster.
    """
    if callable(spec) and not isinstance(spec, str):
        return spec

    name = str(spec)
    from tsfm_adapters import _REGISTRY as _TSFM

    if name.lower() in _TSFM:
        from tsfm_adapters import load_tsfm

        return load_tsfm(name)

    from score import default_panel

    panel = default_panel()
    if name in panel:
        return probabilistic(panel[name])
    if name == "context_parrot":
        from baselines import context_parrot

        return probabilistic(context_parrot)
    raise KeyError(
        f"unknown model {name!r}; use a TSFM ({sorted(_TSFM)}), a panel model "
        f"({sorted(panel)}), or 'context_parrot'"
    )


# --------------------------------------------------------------------------- #
# Per-sample evaluation
# --------------------------------------------------------------------------- #


def _point_fn(forecaster: Forecaster) -> Callable[[np.ndarray, dict | None], np.ndarray]:
    """Expose a forecaster's mean as the point function ``free_run`` rolls out."""

    def fn(context: np.ndarray, meta: dict | None) -> np.ndarray:
        return np.asarray(forecaster(context, meta).mean, dtype=float).reshape(-1)

    return fn


def evaluate_sample(
    forecaster: Forecaster, sample: DSRSample, *, cfg: MetricConfig | None = None
) -> dict[str, float]:
    """Run the protocol on one sample and return the metric row.

    Keys: ``d_stsp``, ``d_h``, ``vpt``, ``lyap_gen``, ``lyap_true``, ``lyap_gap``,
    ``d_ky_ref``, ``mase``, and ``n_gen`` (generated length actually scored).
    """
    cfg = cfg or MetricConfig()
    point_fn = _point_fn(forecaster)
    gen = free_run(point_fn, sample.context, total=int(sample.truth.size))

    # Short-horizon point validity on the un-cut early rollout.
    if sample.lyapunov_time:
        vpt = vpt_lyapunov(
            sample.truth, gen, dt=sample.dt, lyapunov_time=sample.lyapunov_time,
            threshold=cfg.vpt_threshold,
        )
    else:
        vpt = valid_prediction_time(sample.truth, gen, threshold=cfg.vpt_threshold)

    # Invariant metrics on the transient-cut long rollouts.
    tr = min(sample.transient, max(0, gen.size - 1))
    g_cut, t_cut = gen[tr:], sample.truth[tr:]
    d_stsp = state_space_divergence(
        g_cut, t_cut, m=cfg.embed_dim, tau=cfg.tau, estimator=cfg.estimator,
        bins=cfg.bins, n_components=cfg.n_components, n_mc=cfg.n_mc, seed=cfg.ssp_seed,
    )
    d_h_val = power_spectrum_hellinger(g_cut, t_cut, sigma=cfg.sigma)
    # Fit the Rosenstein slope over ~1 Lyapunov time (clamped); a fixed short window
    # overestimates lambda badly. Fall back to the function default when unknown.
    if cfg.lyap_horizon is not None:
        lyap_h = cfg.lyap_horizon
    elif sample.lyapunov_time and sample.dt:
        lyap_h = int(min(200, max(20, round(sample.lyapunov_time / sample.dt))))
    else:
        lyap_h = 50
    lyap_gen = max_lyapunov_rate(g_cut, sample.dt, m=cfg.embed_dim, tau=cfg.tau, horizon=lyap_h)
    lyap_true = sample.lyap_max_ref

    # MASE on the short held-out window, using the model's own forecast length.
    fc = forecaster(sample.context, None)
    h = min(int(np.asarray(fc.mean).size), sample.truth.size)
    truth_h = sample.truth[:h]
    mae = float(np.mean(np.abs(truth_h - np.asarray(fc.mean, dtype=float)[:h])))
    mase = mae / _naive_scale(sample.context, 1)

    return {
        "d_stsp": float(d_stsp),
        "d_h": float(d_h_val),
        "vpt": float(vpt),
        "lyap_gen": float(lyap_gen),
        "lyap_true": float(lyap_true) if lyap_true is not None else float("nan"),
        "lyap_gap": (
            abs(float(lyap_gen) - float(lyap_true)) if lyap_true is not None else float("nan")
        ),
        "d_ky_ref": float(sample.d_ky_ref) if sample.d_ky_ref is not None else float("nan"),
        "mase": float(mase),
        "n_gen": float(g_cut.size),
    }


# --------------------------------------------------------------------------- #
# Top-level run
# --------------------------------------------------------------------------- #


def run_dsr_eval(
    models: Sequence[str | Forecaster] | dict[str, Forecaster],
    systems: Sequence[str],
    *,
    seeds: Sequence[int] = tuple(range(5)),
    context_len: int = DEFAULT_CONTEXT_LEN,
    horizon: int | None = None,
    long_len: int = DEFAULT_LONG_LEN,
    transient_frac: float = DEFAULT_TRANSIENT_FRAC,
    backend: str = "auto",
    cfg: MetricConfig | None = None,
) -> list[dict[str, object]]:
    """Run the full eval and return one flat row per ``(model, system, seed)``.

    ``models`` is a list of specs (resolved by :func:`resolve_forecaster`) or a
    ``{name: forecaster}`` mapping. Each row carries ``model``, ``system``, ``seed``
    and every metric from :func:`evaluate_sample`. Aggregate with
    :mod:`dsr_eval.report`.
    """
    cfg = cfg or MetricConfig()
    from config import HORIZON

    horizon = HORIZON if horizon is None else horizon

    if isinstance(models, dict):
        resolved = dict(models)
    else:
        resolved = {str(m): resolve_forecaster(m) for m in models}

    rows: list[dict[str, object]] = []
    for sysname in systems:
        system = get_system(sysname)
        samples = [
            build_sample(
                system, seed, context_len=context_len, horizon=horizon,
                long_len=long_len, transient_frac=transient_frac, backend=backend,
            )
            for seed in seeds
        ]
        for mname, fc in resolved.items():
            for sample in samples:
                metrics = evaluate_sample(fc, sample, cfg=cfg)
                rows.append({"model": mname, "system": sysname, "seed": sample.seed, **metrics})
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m dsr_eval",
        description="Dynamical-systems (DSR) evaluation of time-series foundation models.",
    )
    p.add_argument(
        "--models", nargs="+", default=["seasonal_naive", "context_parrot"],
        help="model specs: TSFM names (chronos, timesfm) and/or panel baselines.",
    )
    p.add_argument(
        "--systems", nargs="+", default=["lorenz", "rossler"],
        help="chaotic systems: built-ins (lorenz, rossler) or dysts:<Name>.",
    )
    p.add_argument("--seeds", type=int, default=5, help="number of seeds (mean +/- std).")
    p.add_argument("--context-len", type=int, default=DEFAULT_CONTEXT_LEN)
    p.add_argument("--long-len", type=int, default=DEFAULT_LONG_LEN)
    p.add_argument("--transient-frac", type=float, default=DEFAULT_TRANSIENT_FRAC)
    p.add_argument("--backend", default="auto", choices=["auto", "scipy", "numpy"])
    p.add_argument("--bins", type=int, default=20, help="D_stsp histogram bins (fix across runs).")
    p.add_argument("--sigma", type=int, default=20, help="D_H Gaussian smoothing (fix per run).")
    p.add_argument("--embed-dim", type=int, default=3, help="delay-embedding dim for D_stsp.")
    p.add_argument("--out-dir", default="dsr_results", help="dir for CSV + markdown output.")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    from dsr_eval.report import write_reports

    cfg = MetricConfig(embed_dim=args.embed_dim, bins=args.bins, sigma=args.sigma)
    rows = run_dsr_eval(
        args.models,
        args.systems,
        seeds=tuple(range(args.seeds)),
        context_len=args.context_len,
        long_len=args.long_len,
        transient_frac=args.transient_frac,
        backend=args.backend,
        cfg=cfg,
    )
    csv_path, md_path = write_reports(rows, args.out_dir, cfg=cfg)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "MetricConfig",
    "evaluate_sample",
    "main",
    "resolve_forecaster",
    "run_dsr_eval",
]
