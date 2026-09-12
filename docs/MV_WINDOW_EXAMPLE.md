# What a multivariate eval window looks like (miner-facing example)

Real data, real window: the series below is **in today's live pool** as a
univariate window, and becomes the 4-channel window shown after the MV gate
(block 9068400, Mon 2026-09-14 20:30 UTC) once cascade publishes an
`--mv-pack` pool build. Full-precision arrays: `docs/mv_window_example.json`.

## Today (univariate) — one pathogen, one state

    series_id : cdc_nssp_ed_covid19_daily_by_state__geography_California
    freq      : P1D   domain: healthcare
    context   : (256,)  2025-10-14 … 2026-06-26
                [0.36, 0.35, 0.3, …, 0.08, 0.08, 0.11]
    horizon   : 64      2026-06-27 … 2026-08-29

    forecast(context, horizon=64, num_samples=S) -> (1, S, 64)

Scored per window (MASE/CRPS against the held-out truth). The other three
pathogens for the same state are *separate* univariate series today.

## After the gate (multivariate) — the same state, all four pathogens as ONE window

The forge catalog tags the coupled columns (`mv_channels`, EVAL_POOL.md
contract: ordered, <= 8, columns of one (source, panel_row), shared timestamp
index). Cascade's `--mv-pack` build packs them into one aligned (C, L) window:

    series_id  : cdc_nssp_ed_pathogen_mix_daily_by_state__geography_California
    mv_channels: ['covid', 'influenza', 'rsv', 'ari']    (C = 4)
    context    : (4, 256)  rows in mv_channels order, same dates as above
        covid     [0.36, 0.35, 0.3, …, 0.08, 0.08, 0.11]
        influenza [0.14, 0.13, 0.13, …, 0.15, 0.2, 0.14]
        rsv       [0.03, 0.03, 0.06, …, 0.02, 0.02, 0.01]
        ari       [9.93, 10.16, 9.85, …, 7.68, 7.91, 8.01]
    horizon    : 64 steps x 4 channels — truth is (4, 64)
        covid     [0.1, 0.09, 0.1, …, 0.83, 0.69, 0.8]
        influenza [0.17, 0.17, 0.15, …, 0.21, 0.22, 0.21]
        rsv       [0.02, 0.0, 0.01, …, 0.03, 0.01, 0.01]
        ari       [8.03, 8.43, 8.0, …, 8.2, 7.49, 7.48]

    forecast_joint(context, horizon=64, num_samples=S) -> (4, S, 64)

Scoring: per variate (each channel scored like a univariate window against
its own truth), combined arithmetically within the window. A checkpoint
without `forecast_joint` is lifted channel-by-channel through the adapter —
valid, but it cannot use cross-channel structure; joint decoding that
conditions channels on each other is where the MV reward lives (these four
share epidemic and reporting dynamics — that is *why* the group is tagged).

## The accounting (why this costs miners nothing extra)

- One (C, L) window = **one series** against every cap: one slot in the pool,
  one forecast call, one bootstrap cluster (clusters key on `source`).
- C = 1 joint decoding is bit-identical to the univariate path (enforced by
  cascade's `test_forecast_joint`), so univariate scoring never moves.
- Tags are admitted on measured cross-predictiveness (a sibling's *lags* must
  improve a held-out forecast beyond own-lags, placebo-adjusted), never on
  correlation — see `decisions/DEC-TB-0004-mv-channels-coupled-supply.md`.
