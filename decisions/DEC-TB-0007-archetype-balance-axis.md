---
id: DEC-TB-0007
type: decision
title: "Statistical archetypes become a third balance axis, so optimising for the pool stops costing GIFT/BOOM/TIME"
status: active
date: 2026-11-03
tags: [eval-pool, balance, benchmark-alignment, cascade, sourcing]
revisit_when: "raw archetype overlap with the daily-or-faster benchmark mix exceeds 0.90 (reweighting no longer needed); cascade declines archetype-aware selection; or a new external benchmark should join the reference mix (refit with stat_fit --freeze, then stat_tag --apply)"
relations: {builds-on: "DEC-TB-0003 (jittered round mix)", complements: "DEC-TB-0004 (mv_channels)"}
---

**Problem.** Miners who win cascade rounds have been losing ground on GIFT-Eval,
BOOM and TIME. Domain and cadence balance cannot see why: they are labels about
where a series came from, not what forecasting problem it poses.

**Measurement (2026-11-03).** 4,936 pool series and 2,510 benchmark series
(all 56 GIFT data files, 250 BOOM configs stratified by cadence, all 50 TIME
configs) were profiled on the same scale-invariant features (`tsfeatures` v2)
and clustered together with k-means, k=12 (`stat_clusters`, `stat_fit`).
Fitting on the union matters: a region the pool lacks still gets its own
cluster. Overlap of the archetype mixes (1.0 = identical):

| vs | overlap |
|---|---|
| all benchmarks | 0.79 |
| daily-or-faster benchmark series | 0.81 |
| GIFT / BOOM / TIME | 0.70 / 0.62 / 0.67 |

The conclusion holds across k = 8 / 12 / 16 (overlap 0.82 / 0.79 / 0.74). No
benchmark region is absent from the pool. The mix is off:

- **A09, slow trend with level shifts**: 6.7% of the pool against 17.4% of the
  benchmarks (GIFT 23%, BOOM 15%, TIME 32%). The largest gap, consistent with
  the earlier absolute-threshold finding that strong-trend series are 2.4% of
  the pool. In the benchmarks it comes from M4, Shenzhen taxi demand, coastal
  temperature and salinity, commodity production and port activity.
- **A04** (flat stretches, lumpy, skewed) and **A07** (heavy-tailed, lumpy):
  BOOM-style telemetry, 10.7% and 18.7% of BOOM.
- **Over-held: A08**, intermittent skewed counts (municipal event counts, edit
  and request counts), 14.9% of the pool against 3.7% of the benchmarks. These
  are the cheapest series to scrape.

**Trap recorded.** The first fit (v0, discarded) used unit-dependent features
(log CV, zero fraction, raw-variance spikiness/lumpiness/stability). BOOM
publishes its series standardised (67% are exactly mean 0 / sd 1, 84%
zero-centred; GIFT and TIME 0%), so v0 split BOOM from everything else by
storage format. It reported a BOOM-only archetype at 0.4% of the pool and
overlap 0.40 with BOOM. Both were artefacts. v2 standardises every series
before computing features, and a test pins `features(a·x+b) == features(x)`.

**Decision.**

1. Every catalog entry carries `stat_archetype: A00..A11`
   (`source_discovery.stat_tag`): the majority archetype of the source's fresh
   series. It is re-assigned on every run because a series' character drifts.
   The model is frozen in `src/source_discovery/data/archetypes.json` (v1), so
   ids are stable. Refitting renumbers the clusters, and `stat_fit --freeze`
   bumps the version.
2. `wire.verify_entry` rejects a candidate whose archetype has draw weight
   below 0.5 (A08 today). Opt out per entry with
   `allow_overcovered_archetype: true`. A candidate with fewer than 256 points
   cannot be judged; it passes and is tagged later.
3. Sourcing targets archetypes with draw weight above 1 (A09, A11, A04, A07,
   A01, A02), classified at wire time. Volume is not the target.

**Asked of cascade** (their decision): make `--selection balanced` round-robin
over (domain, freq, stat_archetype), or weight the stage-3 draw by
`draw_weight` from the model file. Reweighting alone lifts daily-or-faster
overlap from 0.81 to 1.00 in expectation. It costs effective sample: Kish ESS
is 3,690 of 4,936 (75%), because A08 series count 0.25× and A09 series 2.6×.
Item 3 exists to shrink that cost: the more of the under-held archetypes the
pool holds, the closer the weights get to 1.

**Why not just add more data.** More of the cheap series (municipal and
platform event counts) lands in A08 and lowers benchmark overlap. Volume and
alignment pull in opposite directions until both sourcing and selection are
archetype-aware.
