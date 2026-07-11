# TSFM significance on live TSBench-Forge data (GPU run, 2026-07-11)

All 9 target foundation models scored against the live benchmark and tested with paired
Wilcoxon / Friedman. Run via `scripts/run_tsfm_comparison_lium.py --group all`
(one GPU pod — RTX 4090 or, when the 4090 spot pool is churny, A100, a fresh venv per model group so conflicting libraries can't
collide), merged with `scripts/merge_tsfm_results.py`.

**Setup.** 256 challenges, `motif_len=304` (context 256 + horizon 48), **43 real
source series**, seed `tsfm-significance-v1` (identical across groups → paired).
First run on the **expanded post-autoresearch pool**: 62 eligible sources including
the 2026-07-10 batch (Gemini trades, SG taxi, CRW reef SST, RIPE Atlas RTT,
Datalakes lakes, USACE reservoirs) plus the repaired NDBC/GLERL parquets —
28/256 challenges (11%) draw from the new sources.

## Unified leaderboard — 9 TSFMs vs classical panel

Seasonal-naive-relative shifted-gmean (lower = better). All nine foundation models
**sweep the top on CRPS**, each significantly beating seasonal-naive. The edge is
probabilistic (CRPS); on MASE Toto-2 and TabPFN-TS lead point accuracy too.

| rank | model | crps_rel | mase_rel | CRPS | MASE |
|---:|---|--:|--:|--:|--:|
| 1 | **toto2** | 0.427 | 0.671 | 0.353 | 2.143 |
| 2 | **tabpfn-ts** | 0.441 | 0.621 | 0.384 | 2.245 |
| 3 | **timesfm25** | 0.540 | 0.730 | 0.517 | 2.938 |
| 4 | **chronos2** | 0.559 | 0.851 | 0.503 | 3.075 |
| 5 | **moirai2** | 0.562 | 0.877 | 0.379 | 2.811 |
| 6 | **tirex** | 0.579 | 0.883 | 0.490 | 2.994 |
| 7 | **chronos-bolt** | 0.624 | 0.972 | 0.475 | 2.878 |
| 8 | **flowstate** | 0.749 | 1.124 | 0.563 | 8.715 |
| 9 | **sundial** | 0.759 | 1.111 | 0.668 | 3.868 |
| 10 | strong | 0.920 | 0.823 | 0.748 | 2.999 |
| 11 | ewma | 0.935 | 0.918 | 0.750 | 3.211 |
| 12 | drift | 0.966 | 0.959 | 0.777 | 3.290 |
| 13 | seasonal_naive | 1.000 | 0.960 | 0.812 | 3.508 |
| 14 | context_parrot | 1.017 | 1.034 | 0.776 | 3.523 |
| 15 | ar1 | 1.020 | 1.030 | 0.799 | 3.810 |

## Significance

- **Friedman χ² = 1462.5, p = 1.8e-192** (CRPS) — the models genuinely differ.
- **Every TSFM beats seasonal-naive on CRPS** (Holm-adjusted paired Wilcoxon):

  | model | median CRPS Δ vs naive | win-rate | Holm p | sig |
  |---|--:|--:|--:|:--:|
  | toto2 | -0.254 | 0.89 | 7e-33 | Y |
  | chronos-bolt | -0.244 | 0.88 | 3e-34 | Y |
  | moirai2 | -0.228 | 0.84 | 6e-32 | Y |
  | tabpfn-ts | -0.228 | 0.88 | 2e-32 | Y |
  | chronos2 | -0.226 | 0.91 | 3e-35 | Y |
  | tirex | -0.220 | 0.91 | 3e-35 | Y |
  | timesfm25 | -0.207 | 0.91 | 1e-36 | Y |
  | flowstate | -0.177 | 0.84 | 2e-27 | Y |
  | sundial | -0.147 | 0.75 | 1e-19 | Y |
  | ewma | -0.009 | 0.64 | 4e-06 | Y |
  | strong | -0.003 | 0.61 | 5e-08 | Y |
  | ar1 | -0.000 | 0.51 | 0.37 | N |
  | drift | +0.000 | 0.41 | 1 | N |
  | context_parrot | +0.010 | 0.40 | 1 | N |

## Movement vs the 2026-07-09 run (42-series pool, pre-autoresearch)

The pool expansion reshuffled the mid-table while the top two held:

- **Toto-2 and TabPFN-TS stay 1–2** (gap narrowed: 0.427 vs 0.441 crps_rel).
- **Moirai-2 jumped 8th → 5th** and **chronos-bolt slid 6th → 7th**; TiRex and
  chronos2 swapped. Sundial dropped 7th → 9th.
- Absolute CRPS rose across the board (0.648 → 0.812 for seasonal-naive):
  the new sources (irregular trade ticks, RTT with -1 sentinels, reservoir
  guide-curves) are genuinely harder targets, which is what they were added for.
- The classical panel ordering is stable; ar1/drift/context_parrot remain
  indistinguishable from seasonal-naive — the discrimination filter is doing
  its job.

Figure: `notebooks/figures/tsfm_leaderboard.png`. Raw per-group results under
`notebooks/results/group_*/`; unified stats in `notebooks/results/merged.json`.

Run note: the first pod (gentle-orbit-6d) was evicted mid-sundial after 6 of 7
groups completed; sundial re-ran on a fresh pod (brave-lion-d7) against the
identical seeded challenge set, so the merge stays paired.
