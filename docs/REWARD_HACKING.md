# Reward-hacking analysis (pure-live benchmark)

The benchmark forecasts **real data only**. Challenges are windows of genuinely
real series drawn from a scraped live catalog (`ScrapedLiveSource` over
`src/sources/`) and assembled by `challenges.build_live_challenges`. The
distribution the benchmark tests is exactly the distribution of the ingested feeds
— there is no synthetic generation.

A challenge pool earns credit as a benchmark through

```
foundational_fitness = fitness
                     × coverage_gate
                     × parrot_gate
                     × dgp_class_breadth_gate
                     × cadence_breadth_gate
```

where `fitness = spread` (`score.panel_fitness` / `score.foundational_fitness`),
with `spread = (max_err − min_err) / mean_err` over a panel of baselines. Every
factor is multiplicative, so any *one* going to zero forces the aggregate to zero.

Because the data is real, the whole *generator-fitting* gaming vector is gone:
there is no synthetic DGP to reverse-engineer, so the old near-oracle `overfit`
detector and its `gate` factor have been deleted. What remains are the ways a
challenge pool can look discriminating while failing to measure genuine
forecasting skill — whether the pool degrades because a feed dries up, the
catalog narrows, or a submitter games the served set. This document catalogs
those failure modes and the shipped defenses.

## Failure modes

1. **Non-discriminating pool.** If no model in the panel meaningfully beats any
   other — the pool is pure noise, or trivially forecastable by every baseline —
   then `spread → 0` and `fitness → 0`. A benchmark on which every model scores
   the same measures no skill difference. (Note: we do *not* require a specific
   model to lead. On real data a naive model legitimately wins on some series —
   random walks — so demanding that a "strong" model beat naive would penalise
   valid tasks. Discrimination, not ordering, is what we enforce.)

2. **DGP-class collapse.** `spread`/`parrot` all measure quality
   *within* whatever distribution the feed supplies; none notices whether that
   distribution spans one data-generating process or thirty. If the catalog or a
   degrading feed narrows the served pool to a handful of DGP classes, the
   "foundation-model" claim silently shrinks without any single factor tripping.

3. **Cadence collapse.** High-frequency windows tend to give higher spread; a
   pool that drifts toward minute-cadence-heavy content quietly drops the
   yearly / monthly generalisation claim.

4. **Parrot-solvable pool.** A pool can show healthy spread yet still be
   near-trivially forecastable by a nearest-neighbour *copy-the-context* baseline
   (`baselines.context_parrot`; Zhang & Gilpin, arXiv:2505.11349). Then a high
   score rewards induction-head repetition, not understanding — and it measures no
   skill difference between competing TSFMs.

5. **Memorisation of a finite / slowly-refreshing feed.** Real feeds are finite.
   Serve the same real motif twice and the second serving is a lookup, not a
   forecast — the answer could have been memorised at commit time.

6. **Truth-destroying augmentation.** The served windows carry a *light*
   augmentation (§below). If augmentation were heavy or non-invertible it would
   stop being real data — the `truth` would no longer be the genuine continuation
   of the `context` — destroying the very property that makes real data valuable.

## Defenses shipped

| defense | catches | code |
|---|---|---|
| **Discrimination** — `fitness = spread` rewards pools where models genuinely differ; `spread → 0` on a pure-noise or trivially-forecastable pool. The admission-time filter (`source_discovery.quality`) rejects pure noise via low autocorrelation and trivially-periodic series via near-zero seasonal-naive error before they ever reach the pool. | non-discriminating pool (#1) | `score.panel_fitness`, `source_discovery.quality` |
| **DGP-class breadth hard veto** — if any DGP class in the pool has share below `min_share`, `dgp_class_breadth_gate = 0`, so `foundational_fitness → 0`. Reads `buffer.pool_dgp_classes`. | DGP-class collapse (#2) | `score.dgp_class_breadth_gate` |
| **Cadence-band breadth hard veto** — if any cadence band has share below `min_share`, `cadence_breadth_gate = 0`. Reads `buffer.pool_cadences`. | cadence collapse (#3) | `score.cadence_breadth_gate` |
| **Coverage gate** — ramps to 0 as the effective-domain count (Hill number of the `meta['domain']` mix) falls below the breadth target. | narrow-but-sharp pools (#2) | `score.coverage_gate` |
| **Parrot gate** — smooth sigmoid of `parrot_err / strong_err`, ~0 when copy-the-context matches/beats the panel. | parrot-solvable pool (#4) | `score.parrot_gate` |
| **Freshness / as-of + `t_now` + cross-epoch dedup** — the **primary** anti-memorisation defense: only admit motifs stamped *after* the commit beacon, past the global pretraining barrier, and never re-serve a near-duplicate. | memorisation (#5) | `feeds.py`, `leakage_audit.py` |
| **Light truth-preserving augmentation** — one invertible reparametrisation (`magnitude_warp` / `time_warp` / `history_cutout`, sparse low-severity `jitter`) per window, defeating exact-match lookup without distorting the task. Belt-and-suspenders on top of freshness. | memorisation (#5), truth-destroying augmentation (#6) | `challenges.py` |
| **Sandboxed submission execution + static analysis** — submissions run isolated with their source scanned, so a model cannot reach outside its `(context, meta) → horizon` interface to look up the answer. | submission-side gaming | `sandbox.py`, `static_analysis.py` |

## Anti-memorisation rests on freshness, not augmentation

The load-bearing anti-memorisation guarantee is the **freshness / as-of vintage
gating** in `feeds.py` + `leakage_audit.py`: the pool only ever holds motifs that
became available *after* the miner's commit and after the global pretraining
cutoff (`t_now = max(model cutoffs)`), with cross-epoch dedup rejecting near-repeats.
The benchmark is exactly as memorisation-safe as the feed's vintage discipline.
The light augmentation in `challenges.py` is a secondary, belt-and-suspenders
layer for finite or predictable feeds — it defeats byte-identical lookup, not
memorisation of the underlying vintage.

## Composition guarantee

Multiplication means any single factor going to zero forces the aggregate to
zero. A pool cannot trade off "great spread + terrible domain breadth" — the
breadth veto zeroes it unconditionally; likewise a parrot-solvable pool scores
zero regardless of spread, and a non-discriminating pool scores zero because
spread itself collapses. The breadth gates are
opt-in by construction: when a `ScrapedLiveSource` is wired up they read real
`dgp_class` / `cadence` labels off the buffer; with a legacy unlabeled source
they default to `1.0`, preserving old behaviour.

## Not a replacement for judgment

These gates raise the bar; they don't eliminate the risk. They catch the failure
modes we can enumerate on the served pool. For a benchmark that is a load-bearing
input to a live subnet, plan for periodic manual review of feed freshness, catalog
breadth, and a canary zero-shot TSFM regression run alongside every version bump.
