# TSFM significance — first live GPU run (2026-07-08)

First real run of `notebooks/tsfm_significance.ipynb` / `scripts/run_tsfm_comparison_lium.py`
on an RTX 4090 (Lium), scoring foundation models against the live TSBench-Forge
benchmark and testing the gaps with paired Wilcoxon / Friedman.

**Setup.** 256 challenges, `motif_len=304` (context 256 + horizon 48), drawn from
**43 real source series** (post loader-concat fix). Seed `tsfm-significance-v1`.

## Result — TimesFM-2.5 and TiRex are significantly better on CRPS

Two foundation models loaded cleanly (see coexistence note below); both **beat every
classical baseline on probabilistic accuracy (CRPS) by a wide, highly-significant margin**,
while being only middling on point accuracy (MASE). That split — big CRPS win, modest MASE —
is the expected TSFM signature: the edge is calibrated *uncertainty*, not the point forecast.

| rank | model | CRPS (rel to naive) | MASE (rel) | CRPS | win vs naive | Holm p (CRPS) | sig |
|---:|---|---:|---:|---:|---:|---:|:--:|
| — | **timesfm25** | **0.547** | 0.814 | 0.500 | 0.87 | 3e-31 | ✅ |
| — | **tirex** | **0.581** | 0.935 | 0.493 | 0.87 | 4e-32 | ✅ |
| 1 | strong (classical) | 0.925 | 0.778 | 0.708 | 0.52 | 8e-05 | ✅ |
| 2 | ewma | 0.895 | 0.784 | 0.690 | 0.64 | 5e-08 | ✅ |
| — | seasonal_naive | 1.000 | 0.888 | 0.753 | — | — | — |
| — | drift | 1.003 | 0.893 | 0.783 | 0.34 | 1.0 | ❌ |
| — | context_parrot | 1.188 | 1.363 | 0.849 | 0.32 | 1.0 | ❌ |

- **Friedman χ² = 640, p = 5e-88** — the models genuinely differ.
- `context_parrot` (repeat-the-context lower bound) ranks **last** → the benchmark is not
  trivially memorizable.
- Breadth caveat: 43 source series is still low, so challenges are not fully independent —
  the leaderboard (per-source-relative) is the primary ranking; the p-values corroborate.
  This grows as the scraper's cron accumulates history (loader now concatenates daily files).

## Dependency-coexistence note (why only 2/8 this run)

The other six were **skipped by the load-probe, not by the benchmark** — all dependency
conflicts from co-installing many TSFM libraries in one environment:

| model | skip reason | class |
|---|---|---|
| toto2, moirai2 | `operator torchvision::nms does not exist` | a later model lib re-installs a torch that mismatches torchvision |
| chronos2, chronos-bolt, flowstate | `Could not import module 'PreTrainedModel'` | a model lib pins an incompatible `transformers` |
| tabpfn-ts | `TabPFNLicenseError` | LOCAL mode needs a `TABPFN_TOKEN` (priorlabs API key), not just a license accept |

TimesFM-2.5 and TiRex win precisely because they carry self-contained stacks that the
others' pins don't disturb. Getting all ten into one leaderboard requires **per-compatible-group
isolation**: the runner already splits models into `--group` sets (identical challenge set →
`results.json` files merge), so the fix is to partition the transformers-based vs
uni2ts/toto vs timesfm stacks into separate groups and merge — a follow-up, plus a
`TABPFN_TOKEN` in `.env` for TabPFN-TS.

Total GPU spend to reach this result: ≈ $2 (RTX 4090 @ $0.33/hr, several short iterations).
