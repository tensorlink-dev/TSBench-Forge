# TSFM significance on live TSBench-Forge data (GPU run, 2026-07-09)

Foundation models scored against the live benchmark and tested with paired
Wilcoxon / Friedman. Run via `scripts/run_tsfm_comparison_lium.py --group all`
(one GPU pod — RTX 4090 or, when the 4090 spot pool is churny, A100, a fresh venv per model group so conflicting libraries can't
collide), merged with `scripts/merge_tsfm_results.py`.

**Setup.** 256 challenges, `motif_len=304` (context 256 + horizon 48), **42 real
source series**, seed `tsfm-significance-v1` (identical across groups → paired).

## Unified leaderboard — 8 TSFMs vs classical panel

Seasonal-naive-relative shifted-gmean (lower = better). All eight foundation models
**sweep the top on CRPS**, each significantly beating seasonal-naive. The edge is
probabilistic (CRPS); on MASE they're mixed.

| rank | model | crps_rel | mase_rel | CRPS | MASE |
|---:|---|--:|--:|--:|--:|
| 1 | **toto2** | 0.465 | 0.831 | 0.339 | 1.571 |
| 2 | **timesfm25** | 0.492 | 0.718 | 0.398 | 1.851 |
| 3 | **chronos2** | 0.511 | 0.951 | 0.364 | 1.820 |
| 4 | **tirex** | 0.563 | 1.051 | 0.379 | 1.837 |
| 5 | **chronos-bolt** | 0.668 | 1.268 | 0.391 | 2.256 |
| 6 | **sundial** | 0.755 | 1.387 | 0.459 | 2.173 |
| 7 | **moirai2** | 0.759 | 1.326 | 0.385 | 2.950 |
| 8 | **flowstate** | 0.858 | 1.592 | 0.422 | 12.376 |
| 9 | ewma | 0.874 | 0.761 | 0.562 | 1.903 |
| 10 | strong | 0.929 | 0.801 | 0.599 | 2.065 |
| 11 | seasonal_naive | 1.000 | 0.914 | 0.648 | 2.183 |
| 12 | ar1 | 1.036 | 1.047 | 0.636 | 2.658 |
| 13 | drift | 1.073 | 1.022 | 0.708 | 2.906 |
| 14 | context_parrot | 1.173 | 1.182 | 0.724 | 2.790 |

## Significance

- **Friedman χ² = 1310, p = 1e-173** (CRPS) — the models genuinely differ.
- **Every TSFM beats seasonal-naive on CRPS** (Holm-adjusted paired Wilcoxon):

  | model | median CRPS Δ vs naive | win-rate | Holm p | sig |
  |---|--:|--:|--:|:--:|
  | chronos-bolt | -0.227 | 0.84 | 7e-32 | Y |
  | toto2 | -0.224 | 0.86 | 7e-28 | Y |
  | tirex | -0.218 | 0.86 | 9e-32 | Y |
  | moirai2 | -0.212 | 0.79 | 5e-25 | Y |
  | chronos2 | -0.210 | 0.86 | 4e-32 | Y |
  | flowstate | -0.191 | 0.81 | 1e-25 | Y |
  | timesfm25 | -0.190 | 0.86 | 6e-32 | Y |
  | sundial | -0.150 | 0.74 | 7e-20 | Y |
  | ewma | -0.009 | 0.65 | 2e-11 | Y |
  | strong | -0.000 | 0.52 | 1e-04 | Y |
  | ar1 / drift / context_parrot | >=0 | <=0.50 | ns | N |


## Caveats

- **Breadth**: 42 source series — challenges are not fully independent, so the
  per-source-relative leaderboard is the primary ranking and the p-values
  corroborate. This grows as the scraper's cron accumulates history (the loader
  now concatenates a source's daily parquet).
- **Full roster of 8 loaded** after per-group venv recipes (own torch each) + two
  adapter fixes (FlowState: unwrap `out.quantile_outputs`; Toto-2: trim context to a
  multiple of patch_size 32). Only TabPFN-TS remains out — it needs a `TABPFN_TOKEN`
  (priorlabs API key) in `.env`.
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
