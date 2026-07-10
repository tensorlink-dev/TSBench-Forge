# TSFM significance on live TSBench-Forge data (GPU run, 2026-07-09)

All 9 target foundation models scored against the live benchmark and tested with paired
Wilcoxon / Friedman. Run via `scripts/run_tsfm_comparison_lium.py --group all`
(one GPU pod — RTX 4090 or, when the 4090 spot pool is churny, A100, a fresh venv per model group so conflicting libraries can't
collide), merged with `scripts/merge_tsfm_results.py`.

**Setup.** 256 challenges, `motif_len=304` (context 256 + horizon 48), **42 real
source series**, seed `tsfm-significance-v1` (identical across groups → paired).

## Unified leaderboard — 9 TSFMs vs classical panel

Seasonal-naive-relative shifted-gmean (lower = better). All nine foundation models
**sweep the top on CRPS**, each significantly beating seasonal-naive. The edge is
probabilistic (CRPS); on MASE they're mixed (TabPFN-TS leads point accuracy).

| rank | model | crps_rel | mase_rel | CRPS | MASE |
|---:|---|--:|--:|--:|--:|
| 1 | **toto2** | 0.465 | 0.831 | 0.339 | 1.571 |
| 2 | **tabpfn-ts** | 0.470 | 0.683 | 0.377 | 1.711 |
| 3 | **timesfm25** | 0.492 | 0.718 | 0.398 | 1.851 |
| 4 | **chronos2** | 0.511 | 0.951 | 0.364 | 1.820 |
| 5 | **tirex** | 0.563 | 1.051 | 0.379 | 1.837 |
| 6 | **chronos-bolt** | 0.668 | 1.268 | 0.391 | 2.256 |
| 7 | **sundial** | 0.755 | 1.387 | 0.459 | 2.173 |
| 8 | **moirai2** | 0.759 | 1.326 | 0.385 | 2.950 |
| 9 | **flowstate** | 0.858 | 1.592 | 0.422 | 12.376 |
| 10 | ewma | 0.874 | 0.761 | 0.562 | 1.903 |
| 11 | strong | 0.929 | 0.801 | 0.599 | 2.065 |
| 12 | seasonal_naive | 1.000 | 0.914 | 0.648 | 2.183 |
| 13 | ar1 | 1.036 | 1.047 | 0.636 | 2.658 |
| 14 | drift | 1.073 | 1.022 | 0.708 | 2.906 |
| 15 | context_parrot | 1.173 | 1.182 | 0.724 | 2.790 |

## Significance

- **Friedman χ² = 1400, p = 2e-185** (CRPS) — the models genuinely differ.
- **Every TSFM beats seasonal-naive on CRPS** (Holm-adjusted paired Wilcoxon):

  | model | median CRPS Δ vs naive | win-rate | Holm p | sig |
  |---|--:|--:|--:|:--:|
  | chronos-bolt | -0.227 | 0.84 | 8e-32 | Y |
  | toto2 | -0.224 | 0.86 | 7e-28 | Y |
  | tirex | -0.218 | 0.86 | 1e-31 | Y |
  | moirai2 | -0.212 | 0.79 | 5e-25 | Y |
  | chronos2 | -0.210 | 0.86 | 4e-32 | Y |
  | tabpfn-ts | -0.204 | 0.86 | 6e-26 | Y |
  | flowstate | -0.191 | 0.81 | 1e-25 | Y |
  | timesfm25 | -0.190 | 0.86 | 6e-32 | Y |
  | sundial | -0.150 | 0.74 | 7e-20 | Y |
  | ewma / strong | ~0 | ~0.5 | (sig but tiny) | Y |
  | ar1 / drift / context_parrot | >=0 | <=0.50 | ns | N |


## Caveats

- **Breadth**: 42 source series — challenges are not fully independent, so the
  per-source-relative leaderboard is the primary ranking and the p-values
  corroborate. This grows as the scraper's cron accumulates history (the loader
  now concatenates a source's daily parquet).
- **Full roster of 9 loaded** after per-group venv recipes (own torch each) + adapter
  fixes (FlowState: unwrap `out.quantile_outputs`; Toto-2: trim context to a multiple of
  patch_size 32; TabPFN-TS: float quantile-column keys + `target` point column, plus a
  `TABPFN_TOKEN` in `.env` and a one-time license accept at ux.priorlabs.ai).
- **FlowState MASE=12.4 is an outlier** (its CRPS 0.858 is fine): its *point* forecast
  is miscalibrated in scale on some series — likely the adapter's `scale_factor=1.0`
  / median extraction. Its probabilistic (CRPS) forecast ranks normally. Point-metric
  use of FlowState should be treated with caution pending a scale-handling fix.

## Reproduce

```bash
python scripts/run_tsfm_comparison_lium.py --gpu RTX4090 --group all --ttl 90m --yes
python scripts/merge_tsfm_results.py
```

Total GPU spend to this result: ≈ $16 across development iterations; a clean
single-pod `--group all` run is ~$0.50 (one RTX 4090, ~45 min).
