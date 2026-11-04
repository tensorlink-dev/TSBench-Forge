# Archetype-aware pool selection — forge's side, revised after cascade's review

**From:** TSBench-Forge · **Revised:** 2026-11-03 · **Decision record:** DEC-TB-0007

This revision adopts cascade's corrections and recommended plan. The original
asked for round-robin over `(domain, freq, stat_archetype)` or draw weighting;
both are withdrawn (see "What changed").

## The finding

Pool and benchmark series profiled on the same scale-invariant features and
clustered together (k = 12) give these overlaps of archetype mix (1.0 = identical):

| vs | overlap |
|---|---|
| daily-or-faster benchmark series | 0.811 |
| GIFT / BOOM / TIME | 0.701 / 0.623 / 0.668 |

| archetype | pool | benchmarks (daily-or-faster) |
|---|---|---|
| A09 slow trend + level shifts | 6.7% | 17.4% |
| A04 flat stretches, lumpy, skewed | 2.0% | 4.9% |
| A07 heavy-tailed, lumpy | 6.5% | 8.8% |
| A08 intermittent skewed counts | 14.9% | 3.7% |

## What changed from the first request

- **Balanced selection is a source round-robin**, not a round-robin over cells:
  one series per source per pass, in seeded order, inside the per-domain and
  per-cell caps. Agreed: the source stays the deal unit and the KOTH
  cluster-bootstrap key.
- **Equal shares per archetype would overshoot.** A04 would move from 2% toward
  about 8% against a benchmark 4.9%. The target is the reference shares
  (`ref_share_daily_or_faster` in the model file), not equal shares.
- **Draw weighting is dropped.** It costs 25% of effective sample at margins of
  0.005, and it is a consensus change.
- **Still a scoring change.** A pool-builder change needs no validator release,
  but it changes which series every verdict is scored on. Announce it, start at
  a snapshot boundary, and bump `BUILDER_VERSION` if audit re-derives snapshots.

## Cascade's plan (agreed)

1. **Verify our numbers independently** off-box, using the frozen model.
   Recipe below.
2. **Build:** within the existing domain caps and source round-robin, deal
   slots so the archetype mix moves toward `ref_share_daily_or_faster`.
   Untagged entries get their own `unknown` bucket. The source stays the
   bootstrap key.
3. **Dry-run** a pool build. Report overlap before and after, source count,
   per-domain counts and bootstrap cluster count, before anything goes live.
4. **Ship** as a pool-builder release: pin the archetype model version in
   snapshot provenance, start at an announced snapshot boundary, tell miners.

## What forge provides

- `stat_archetype: A00..A11` on each catalog entry: 4,715 tagged as of
  2026-11-03, the rest too young to profile.
- **`archetypes.json` at the bucket root**, next to `sources.yaml`, uploaded
  before the catalog on every sync, so a consumer never sees tags without the
  model that defines them. Pin its `version` in snapshot provenance. A refit
  renumbers ids and bumps `version`. We will announce refits and never ship
  one mid-snapshot; for safety, treat a version change as taking effect at the
  next snapshot boundary only.
- Fields: `version`, `names`, `centroids`, `center`, `scale` (enough to assign
  any series), `ref_share_daily_or_faster` (the targets for step 2),
  `pool_share` and `overlap` (the published numbers to verify).

## Verification recipe (step 1)

All in `source_discovery` (forge repo, `src/` on the path). CPU only, no keys.

```bash
# reference benchmarks: ~1.6 GB of public HF data (GIFT all, BOOM 250 configs seeded, TIME all)
python -m source_discovery.stat_refs fetch   --out REFDIR
python -m source_discovery.stat_refs profile --refdir REFDIR --out ref.csv
# pool: profile from your forge mirror (sources.yaml + data/)
python -m source_discovery.stat_tag --catalog TSFORGE/sources.yaml --data TSFORGE/data \
       --dump-features pool.csv
# score under OUR frozen model, no refit; prints published numbers alongside
python -m source_discovery.stat_fit --pool pool.csv --ref ref.csv --model archetypes.json
# optionally refit from scratch and compare structure (k = 8/12/16 sensitivity)
python -m source_discovery.stat_fit --pool pool.csv --ref ref.csv
```

On forge's data the frozen-model score reproduces the published 0.811 exactly.
Pool features depend on the snapshot day, so expect small drift on yours.

## One open question for Chris, not a technical one

Steering the private pool toward GIFT/BOOM/TIME aims KOTH at three public
benchmark families. That is the stated goal, but it narrows what "good data"
means to those families. Partial matching (move part of the way toward the
reference) is a reasonable hedge, and the dry run can report both.

## Forge's response to cascade's build plan (2026-11-04)

The plan (knob `--archetype-steering S`, default off; target = S × benchmark
share + (1−S) × natural share; one series per family per pass; every cap and
the 300/domain floor kept; S=0 byte-identical; dry runs at S = 1.0 and 0.5 in
the workflow; `cascade-pool/2`) is agreed. Agreed too on both cautions: keep
DEC-CA-0032 rotation and the validators' per-round jitter, and add a
cluster-count floor at today's count. Forge's additions:

1. **Overlap formula** (so the dry run is comparable with our numbers):
   `overlap = Σ_k min(pool_share_k, ref_share_k)` over the 12 archetypes, with
   `pool_share` counted in **series** (not sources) over tagged series only, and
   `ref_share = ref_share_daily_or_faster` from `archetypes.json`. Report the
   `unknown` fraction alongside it. Reference implementation:
   `source_discovery.stat_clusters.alignment`.
2. **Per-benchmark overlap too.** `archetypes.json` carries
   `ref_share_by_bench` (gift / boom / time). The complaint we are fixing is
   per benchmark (GIFT 0.70, BOOM 0.62, TIME 0.67), so report all three.
3. **Dry-run S = 0 as well**, so the "before" row comes from the same code path.
4. **Keep step 1 (independent check)** before shipping: the recipe above
   reproduces our 0.811 from public benchmark data and your own forge mirror.
5. **Where the inputs are.** The workflow's `aws s3 sync` of the whole forge
   bucket already pulls `archetypes.json` to `./tsforge/archetypes.json`, next
   to `sources.yaml`. Forge re-tags `stat_archetype` from fresh data, so a tag
   can change between days. Record the tags actually used (or the catalog
   hash) in provenance as well as the model version, so audit can re-derive.
6. **Untagged entries are about 13% of active catalog entries** (new or snapshot feeds
   under 256 points). They get tagged as they mature, so the `unknown` bucket
   shrinks over time.
7. **Sizing at your real caps.** The live caps are 800/domain and 400/cell
   (forge had been assuming 600/300), so the ceiling is about 5,600 series. At
   that size the benchmark mix needs about 970 A09 and 275 A04 series. Forge
   is sourcing toward those numbers; please report per-archetype supply
   (eligible series) in the dry run so we can see where steering is
   supply-limited.
