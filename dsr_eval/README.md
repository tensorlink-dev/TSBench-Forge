# `dsr_eval` — dynamical-systems evaluation for TSFMs

A benchmark that scores a time-series foundation model on whether it reconstructs the
**dynamics** of a chaotic system — attractor geometry, long-term temporal structure,
and chaos signature — not just its short-horizon point error. It follows Durstewitz et
al., *"Why a Dynamical Systems Perspective is Needed to Advance Time Series Modeling"*
(arXiv:2602.16864), and reuses this repo's existing `dsr_metrics`, `domains`, and
`tsfm_adapters` rather than duplicating them.

## Why

On a chaotic system, point-forecast error (MASE/WQL/CRPS) saturates to a noise ceiling
after ~one Lyapunov time: *any* forecast — a perfect model included — decorrelates from
the truth, so short-horizon error stops telling skill from luck. What stays invariant
under the dynamics, and is therefore the right thing to score long-term, is the
**invariant measure** (where the trajectory lives in state space) and its **temporal /
spectral structure**. This module scores those.

## Protocol

For each `model × system × seed`:

1. Generate a context window, a short held-out continuation, and a long ground-truth
   rollout (`T ≥ 10,000` steps, configurable) from a chaotic system with known
   invariants (`dsr_eval.datasets`).
2. Feed the context to the model and **free-run** it autoregressively (zero-shot) to
   the length of the long ground-truth (`dsr_metrics.free_run`).
3. Discard the initial transient (default 10%) from both the generated and the true
   long rollouts.
4. Score the metrics below; report `mean ± std` over `N` seeds (default 5).

Models stay behind the repo's existing `evaluate.Forecaster` abstraction, so any
registered TSFM (`tsfm_adapters.load_tsfm`: `chronos`, `timesfm`, …) or classical panel
baseline works with no per-model code.

## Metrics

| Metric | What it catches | Implementation |
|---|---|---|
| **`D_stsp`** (state-space divergence) | attractor collapse, mode-dropping, mean-reversion | KL between delay-embedded state distributions — **binning** in low dim (`m ≤ 3`), **GMM/Monte-Carlo** KL in higher dim |
| **`D_H`** (power-spectrum Hellinger) | wrong frequencies; the *context-parroting* collapse to a clean cycle | FFT per dim, Gaussian-smooth (σ), normalize, Hellinger distance, average across dims |
| **`λ_max`** (maximal Lyapunov exponent) | loss of the chaos signature (a periodic collapse has λ ≈ 0) | Rosenstein estimate on the generated series (per unit time); the **true** spectrum + Kaplan-Yorke **`D_KY`** come from the analytic Jacobian (Benettin QR) |
| **`VPT`** (valid prediction time) | short-horizon validity, hardness-normalized | first step where NRMSE exceeds ε (default 0.4), in Lyapunov-time units |
| **`MASE`** | kept deliberately, to show the contrast | mean absolute scaled error on the held-out continuation |

## Usage

```sh
# CLI (defaults to the classical panel baselines so it runs with no torch):
python -m dsr_eval --models seasonal_naive context_parrot --systems lorenz rossler \
    --seeds 5 --long-len 10000 --out-dir dsr_results
# put a real TSFM under test (install the extra + stage weights):
python -m dsr_eval --models chronos --systems lorenz rossler
```

```python
from dsr_eval import run_dsr_eval, MetricConfig
from dsr_eval.report import write_reports

rows = run_dsr_eval(["chronos", "seasonal_naive"], ["lorenz", "rossler"], seeds=range(5))
write_reports(rows, "dsr_results", cfg=MetricConfig())  # -> results.csv + summary.md
```

The runner writes a flat per-seed `results.csv` and a `summary.md` mirroring Table 2 of
the paper (systems as sections, models as rows, metrics as columns, each cell
`mean ± std`).

## Datasets

`Lorenz-63` (σ=10, ρ=28, β=8/3) and `Rössler` (a=0.2, b=0.2, c=5.7) are built in, with
canonical parameters, analytic Jacobians, and literature reference invariants. Adding a
system is a one-liner — write an `rhs` (and a `jac` for the Lyapunov spectrum) and
register a factory in `dsr_eval.systems._REGISTRY`.

**Integration backends.** `scipy.integrate.solve_ivp` (RK45, the paper's protocol) is
the default *when scipy is installed* (`pip install -e ".[dsr]"`); otherwise the eval
silently falls back to the repo's dependency-free fixed-step RK4, so nothing *requires*
scipy. Gilpin's [`dysts`](https://github.com/williamgilpin/dysts) chaotic-systems
library is an optional backend (`pip install -e ".[dysts]"`); reference a `dysts` system
as `"dysts:<Name>"` (e.g. `"dysts:Lorenz"`). `dysts` systems have no analytic Jacobian,
so the true Lyapunov spectrum / `D_KY` are unavailable for them (the data-driven metrics
still apply).

## ⚠️ Caveats — read before comparing across runs

These are **invariant** metrics: they only converge in the **long-time ergodic limit**.

- **Long rollouts, transients cut.** `D_stsp`, `D_H`, `λ`, and `D_KY` need a trajectory
  that has explored the whole attractor — use `T ≥ ~10⁴` steps and discard the initial
  transient (default 10%). On short rollouts they measure sampling noise, not the
  attractor.
- **`D_stsp` / `D_H` are estimator-sensitive.** `D_stsp` depends on the bin count
  (`bins`) and embedding dimension, or on `n_components` for the GMM estimator; `D_H`
  depends on the smoothing width `σ`. These are **not** absolute distances —
  cross-run/cross-model comparisons are only valid when `bins`, `n_components`, `σ`, and
  the embedding (`m`, `τ`) are **held fixed**. `MetricConfig` carries them as one frozen
  bundle, and the `summary.md` header records them so a report is self-documenting.
- **Monte-Carlo `D_stsp` is seed-dependent.** The GMM/MC estimator draws samples; fix
  `ssp_seed` (it is fixed by default) for reproducibility.
- **`λ_max(gen)` is approximate.** The data-driven (Rosenstein) exponent of a generated
  series is for *relative* comparison (`|λ_gen − λ_true|`), not high-precision
  measurement; the precise reference value comes from the analytic Lyapunov spectrum.
- **MASE on chaos is expected to be uninformative** past ~one Lyapunov time — that is
  the whole point of reporting it alongside the dynamics metrics.
