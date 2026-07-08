# TSFM significance on live TSBench-Forge data (GPU run, 2026-07-09)

Foundation models scored against the live benchmark and tested with paired
Wilcoxon / Friedman. Run via `scripts/run_tsfm_comparison_lium.py --group all`
(one RTX 4090 pod, a fresh venv per model group so conflicting libraries can't
collide), merged with `scripts/merge_tsfm_results.py`.

**Setup.** 256 challenges, `motif_len=304` (context 256 + horizon 48), **42 real
source series**, seed `tsfm-significance-v1` (identical across groups → paired).

## Unified leaderboard — 6 TSFMs vs classical panel

Seasonal-naive-relative shifted-gmean (lower = better). All six foundation models
**sweep the top on CRPS**, each significantly beating seasonal-naive; the edge is
probabilistic (CRPS), not point (MASE), where they're mixed.

| rank | model | crps_rel | mase_rel | CRPS | MASE |
|---:|---|--:|--:|--:|--:|
| 1 | **timesfm25** | 0.492 | 0.718 | 0.398 | 1.851 |
| 2 | **chronos2** | 0.511 | 0.951 | 0.364 | 1.820 |
| 3 | **tirex** | 0.563 | 1.051 | 0.379 | 1.837 |
| 4 | **chronos-bolt** | 0.668 | 1.268 | 0.391 | 2.256 |
| 5 | **sundial** | 0.755 | 1.387 | 0.459 | 2.173 |
| 6 | **moirai2** | 0.759 | 1.326 | 0.385 | 2.950 |
| 7 | ewma | 0.874 | 0.761 | 0.562 | 1.903 |
| 8 | strong | 0.929 | 0.801 | 0.599 | 2.065 |
| 9 | seasonal_naive | 1.000 | 0.914 | 0.648 | 2.183 |
| 10 | ar1 | 1.036 | 1.047 | 0.636 | 2.658 |
| 11 | drift | 1.073 | 1.022 | 0.708 | 2.906 |
| 12 | context_parrot | 1.173 | 1.182 | 0.724 | 2.790 |

## Significance

- **Friedman χ² = 1116, p = 2e-148** (CRPS) — the models genuinely differ.
- **Every TSFM beats seasonal-naive on CRPS** (Holm-adjusted paired Wilcoxon):

  | model | median CRPS Δ vs naive | win-rate | Holm p | sig |
  |---|--:|--:|--:|:--:|
  | chronos-bolt | −0.227 | 0.84 | 6e-32 | ✅ |
  | tirex | −0.218 | 0.86 | 7e-32 | ✅ |
  | moirai2 | −0.212 | 0.79 | 5e-25 | ✅ |
  | chronos2 | −0.210 | 0.86 | 3e-32 | ✅ |
  | timesfm25 | −0.190 | 0.86 | 5e-32 | ✅ |
  | sundial | −0.150 | 0.74 | 7e-20 | ✅ |
  | ewma | −0.009 | 0.65 | 2e-11 | ✅ |
  | strong | −0.000 | 0.52 | 1e-04 | ✅ |
  | ar1 / drift / context_parrot | ≥0 | ≤0.50 | ns | ❌ |

- `context_parrot` (repeat-the-context lower bound) ranks **last** → the benchmark
  is not trivially memorizable.

## Caveats

- **Breadth**: 42 source series — challenges are not fully independent, so the
  per-source-relative leaderboard is the primary ranking and the p-values
  corroborate. This grows as the scraper's cron accumulates history (the loader
  now concatenates a source's daily parquet).
- **2 models not yet in the table.** Per-group venv recipes fixed the dependency
  issues (Moirai-2 now loads via pinned `jax`/`jaxlib`); the last two now *import*
  cleanly but hit **adapter-call bugs**, not dep issues: FlowState returns a
  `FlowStateForPredictionOutput` object the adapter must unwrap (not tensor-ify),
  and Toto-2 raises an einops shape error in its forecast call. Both are small
  adapter fixes. TabPFN-TS needs a `TABPFN_TOKEN` (priorlabs API key) in `.env`.

## Reproduce

```bash
python scripts/run_tsfm_comparison_lium.py --gpu RTX4090 --group all --ttl 90m --yes
python scripts/merge_tsfm_results.py
```

Total GPU spend to this result: ≈ $16 across development iterations; a clean
single-pod `--group all` run is ~$0.50 (one RTX 4090, ~45 min).
