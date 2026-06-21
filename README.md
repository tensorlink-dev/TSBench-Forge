# tsbench-forge

A continuous, **hard-to-game benchmark for time-series foundation models
(TSFMs)**, driven by a self-improving *forge* loop on the validator side.

A TSFM benchmark is fit for purpose only if a high score is achievable **solely
through genuine forecasting capability**. `tsbench-forge` gets there with
*defense in depth*: each known way to game a forecasting benchmark is closed by
an **independent** layer, so beating the benchmark requires defeating all of them
at once — which collapses into "actually be a good forecaster."

```
python demo.py        # end-to-end, numpy only: forge climb + static analysis
pytest                # determinism / validity / forge / static-analysis tests
ruff check .
```

## Defense in depth

| Layer | Module | Gaming vector it closes |
|---|---|---|
| Live ingestion | `ingest.py` | memorization — data didn't exist at commit time |
| Augmentation | `generate.py` | exact-match memorization |
| Synthetic composition | `generate.py` | coverage gaps live data misses |
| Real-motif splice | `generate.py` | synthetic-only fingerprints |
| Blend controller | `generate.py` | knowing the test composition in advance |
| Commit-reveal seed | `seed.py` | precomputing the concrete challenges |
| Reference-panel fitness | `score.py` | invalid / saturated challenges |
| Overfit detector | `score.py` | generator-fitting (exploiting synthetic quirks) |
| Submission static analysis | `static_analysis.py` | hardcoded answers regardless of data |
| Forge loop | `forge_loop.py` | benchmark saturation over time |

## Two load-bearing design commitments

1. **Validity comes from a frozen reference panel, not from trusting the data.**
   A challenge only counts if a fixed panel of *known-quality* forecasters ranks
   in its known order. If a naive baseline beats the strong model, the challenge
   is measuring an artifact and the forge is penalized. The panel's top tier
   (`strong`) is the validity anchor: its quality must be established
   *independently* (e.g. on GIFT-Eval), not by this benchmark. The panel is
   **frozen within a version** so all validators reach consensus.

2. **Consensus by determinism.** The LLM forge is non-deterministic, so it runs
   **once** per epoch and commits a hashed manifest; the concrete challenges are
   derived deterministically from `seed = H(block_hash || epoch ||
   manifest_hash)`, revealed only *after* miners commit. Every validator replays
   identical challenges. The LLM never runs live per-validator.

### How the two validity gates compose (see `score.panel_fitness`)

```
fitness = spread * max(0, ordering) * gate
```

- `spread` — how strongly the challenge set separates good from bad forecasters.
- `ordering` — Kendall-τ of the achieved panel ranking against the known-good
  order. A naive model beating `strong` sends it negative ⇒ **fitness 0**.
- `gate` — the generator-fitting detector: the `overfit` model is handed the
  synthetic generator's own continuation (a worst-case reverse-engineering
  miner). If it matches/beats `strong`, the benchmark is pre-fittable ⇒ `gate→0`
  ⇒ **fitness 0**.

The two gates are independent: one catches *invalid* challenges, the other
catches *pre-fittable* ones. The forge can only raise fitness by producing
challenges that are simultaneously valid, non-fittable, **and** discriminating.

## What the demo shows

Starting from a deliberately **weak** generator state (synthetic-heavy, low
augmentation — i.e. easy to memorize/pre-fit), the forge climbs by rebalancing
the blend toward real/spliced data, which lifts the generator-fitting gate, and
then by adding structural richness. Fitness rises from ~0.05 to ~0.6 over ~20
epochs via single-knob keep/revert moves.

## Robustness is conditional — invest in these three things

This benchmark is **not unconditionally robust.** "Robust" holds only to the
degree these are invested in:

1. **Panel quality.** The `strong` anchor must be genuinely good and of
   *independently established* quality — e.g. a top, non-data-leaking GIFT-Eval
   model run zero-shot, or a strong classical ensemble. A weak anchor makes the
   validity gate hollow (everything "beats" a bad anchor, so nothing is flagged).
   The numpy default here is a backtest-selected classical ensemble; swap in a
   real zero-shot TSFM via `score.default_panel(strong_model=...)` for production.

2. **Feed freshness.** The live pool must refresh from sources **outside miners'
   reach**, using as-of/vintage snapshots timestamped *after* the commit point. A
   finite or predictable feed degrades freshness back toward leakability. Always
   quarantine and dedup every pull.

3. **Recipe-space size.** The generator must *out-produce saturation*: the space
   of reachable challenges must be large enough that miners cannot cover it. Too
   small a space and the arms race tips toward the miners no matter how clever the
   forge is.

## The one accepted trade-off

An **evolving** benchmark sacrifices exact cross-epoch comparability. We recover
comparability *approximately* from the **frozen reference panel** as a fixed
yardstick. For exact longitudinal tracking, pair `tsbench-forge` with a **static
anchor** (a held-out real benchmark) and read the **gap** between the evolving
score and the static one as a *leakage detector*: if the static score climbs much
faster than the forged one, something is leaking.

## Repo layout

```
config.py           GeneratorState (the forge's optimization target) + constants
ingest.py           LiveSource ABC, SyntheticLiveSource stand-in, FreshBuffer
generate.py         primitives, augmentations, Recipe grammar, blend controller
score.py            reference panel (frozen strong anchor + overfit detector) + metrics
seed.py             commit-reveal deterministic seeding
static_analysis.py  AST/regex linter for miner submissions
forge_loop.py       the keep/revert autoresearch loop over GeneratorState
program.md          the forge LLM's instructions (what it may/may not change)
demo.py             runnable end-to-end demo
tests/              determinism, validity, forge, static-analysis
```

## Notes

- Python ≥ 3.11. The core path is **numpy only**; the classical `strong` anchor
  can optionally use `statsforecast` (lazily imported, see
  `score.try_statsforecast_strong`) without affecting the numpy demo.
- Everything is seeded and reproducible — no global mutable RNG, no network in
  the core path. Live ingestion is an abstraction with a synthetic stand-in; real
  adapters are a documented extension point.
