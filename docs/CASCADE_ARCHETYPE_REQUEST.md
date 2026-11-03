# Request to cascade: archetype-aware pool selection

**From:** TSBench-Forge · **Date:** 2026-11-03 · **Decision record:** DEC-TB-0007

## The problem

Miners who win cascade rounds are losing ground on GIFT-Eval, BOOM and TIME.
The pool is balanced by domain and cadence, but those labels say where a series
came from, not what kind of forecasting problem it is. The pool's statistical
mix differs from the benchmarks', so the KOTH verdict rewards behaviour the
benchmarks don't.

## What we measured

4,936 pool series and 2,510 benchmark series (all of GIFT and TIME, 250 BOOM
configs stratified by cadence) were profiled on the same 15 scale-invariant
statistical features and clustered together into 12 archetypes. Overlap of the
archetype mixes (1.0 = identical): **0.81** against the daily-or-faster
benchmark series; GIFT 0.70, BOOM 0.62, TIME 0.67. The result is stable for
k = 8, 12 and 16.

| archetype | pool | benchmarks | note |
|---|---|---|---|
| A09 slow trend + level shifts | 6.7% | 17.4% | largest gap; M4, taxi demand, coastal T/S |
| A04 flat stretches, lumpy, skewed | 2.0% | 4.9% | BOOM 10.7% |
| A07 heavy-tailed, lumpy | 6.5% | 8.8% | BOOM 18.7% |
| A08 intermittent skewed counts | 14.9% | 3.7% | over-held; cheapest to scrape |

## What forge has done

- Every catalog entry now carries `stat_archetype: A00..A11`, a top-level key
  read the same way as `mv_channels`. The frozen model, with centroids,
  scaling, reference shares and a clipped `draw_weight` per archetype, is in
  `src/source_discovery/data/archetypes.json`.
- New sources in the over-held archetype are rejected at wire time.
- Sourcing now targets the under-held archetypes (coastal temperature and
  salinity, groundwater levels and similar).

## What we're asking

Pick one of these:

1. **Selection (preferred):** make `--selection balanced` round-robin over
   `(domain, freq, stat_archetype)` instead of `(domain, freq)`. This needs no
   new parameters. The archetype mix then follows supply, and our sourcing
   moves supply toward the benchmarks.
2. **Draw weighting:** weight the stage-3 window draw by `draw_weight[archetype]`
   (already clipped to 0.25–4). Simulated on our side, this lifts overlap from
   0.81 to 1.00 in expectation, at a cost: Kish effective sample drops to 75%.
   That cost falls as our sourcing fills the under-held archetypes.

Untagged entries (feeds too young to profile) should be treated as their own
"unknown" archetype, not dropped.

## Compatibility

- Additive key; nothing existing changes meaning.
- Option 1 is a pool-build change (no validator release). Option 2 changes
  what validators score, so it needs release-then-activate.
- Archetype ids are stable within a model version. We will announce any refit,
  since it renumbers them, and bump `version` in the model file.
