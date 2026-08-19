---
id: DEC-TB-0003
type: decision
title: "The per-round eval mix is jittered — composition is unpredictable, equal-weight only in expectation"
status: active
date: 2026-08-19
tags: [eval-pool, sampling, anti-overfitting, benchmark-integrity, consensus]
revisit_when: "the coverage gate target (4.0 effective domains) or the dgp-gate min share (2%) changes, which re-derives both mix_jitter_alpha and MIN_CLASS_SLOTS; or the pool grows past ~500 classes, at which point ~40 active classes per round may be too narrow a per-round slice and class_keep_frac should rise; or miners demonstrably shift to variance-gaming (throwing rounds whose realised mix disfavours them), which argues for narrowing the Dirichlet (higher alpha) rather than abandoning the jitter"
relations: {supersedes-part-of: "the fixed equal-weight split described in DEC-TB-0002's sampler"}
---

`ScrapedLiveSource._sample_indices_equal_weight` no longer produces the same
stratification every round. The *expected* mix is unchanged — equal weight per
domain, then per DGP class, the property that stops source-count-heavy domains
drowning out light ones — but any single round's realised mix is drawn through
the beacon-derived rng, so the exact composition of a round cannot be
predicted even by a miner holding the full published pool.

**The problem.** The old split was deterministic given the pool: ~18 slots per
domain of 128, divided equally across that domain's classes, every round,
forever. The eval pool is published daily (Hippius), so a miner could compute
tomorrow's composition to the slot — and worse, tiny cells were the most
predictable part: a 2-series class received its fixed quota every single day,
which made those specific series the highest-value memorisation targets in the
benchmark.

**Four changes, one knob each:**

1. **Dirichlet domain mix** (`mix_jitter_alpha=4.0`). The domain split is
   drawn from a symmetric Dirichlet around uniform, allocated by largest
   remainder in 3-slot blocks with a one-block floor per domain. Measured on
   the 2026-08-19 production pool (24,763 series, 7 domains, 209 classes), a
   domain swings between 6 and 56 slots of 128 across seeds while effective
   domains (the coverage gate's Hill number, target 4.0) stayed in 5.33–6.78;
   simulated P(effective < 4) = 0 over 20k rounds.
2. **Per-round class rotation** (`class_keep_frac=0.7`). Each domain activates
   a seeded subset of its classes per round; ~40 of 209 classes run in any
   given round, and no class has a guaranteed daily quota. Activation is
   weighted by how many real series a class can field (capped at one block),
   so a 1-series class still gets rounds, just 3× less often — this cut
   duplicate slots per round from ~35 to ~20.
3. **Per-round series bag** (`series_bag_frac=0.7`). Which series are eligible
   *at all* rotates per round, decided by a salted hash of the series key (not
   a positional draw, so it is independent of catalog enumeration order). An
   unlucky bag that empties the catalog falls back to the full pool.
4. **Without-replacement picks** (`_pick`). A cell's quota is filled with
   distinct series until the cell is exhausted; only then do duplicates fill
   the remainder.

**The gate-safety invariant that shaped all of this.** The dgp-class breadth
gate in `score.py` hard-vetoes (multiplies the score by 0) any class whose
share of the pool falls below 2% — 3 entries of a 128 pool clear it (2.34%),
2 do not (1.56%). The gate permits *absence* but vetoes *token presence*. So
every allocation happens in blocks of `MIN_CLASS_SLOTS = 3`: a class is either
out of the round or in it with a gate-clearing share. Verified over 50 seeded
rounds on the production pool: min class count is always exactly ≥ 3.

Incidentally, this fixes a latent bug: the **old** sampler served 108 classes
at ~1 slot each on today's pool — min share 0.78%, which the gate would veto
for everyone the moment it was fed real labels. The fixed split was
gate-illegal on the very pool it was designed for.

**Why the jitter lives in the pool draw, not the challenge draw.** Challenges
are drawn uniformly from the 128-motif pool (`FreshBuffer.sample_meta`), so
jittering the pool composition carries through to the challenge mix, and both
the challenge-set coverage gate and the pool-fed breadth gates see mutually
consistent distributions. Jittering only the challenge draw would leave the
published pool composition fixed and predictable.

**Consensus is untouched.** Every draw flows through the passed rng over
sorted keys — domains and classes are iterated in sorted order, the bag salt
is a single rng draw, and bag membership is a pure hash. A given (catalog,
seed) yields byte-identical picks on every validator; pinned by
`test_same_seed_gives_byte_identical_picks`.

**What deliberately did not change:**

- `pool_report` constructs its source with neutral knobs
  (`mix_jitter_alpha=None, series_bag_frac=1.0, class_keep_frac=1.0`) because
  it answers "what does the eval draw *on average*", not "what did one round
  look like".
- Rejection top-ups (`pull_meta`'s redraw loop) go through the same machinery;
  a top-up smaller than one block per domain concentrates on a seeded subset
  of domains in usable chunks instead of spraying token 1-slot counts.
- Cascade is unaffected for the same reason as DEC-TB-0002: it reads the
  published parquet through its own pool builder and never imports this
  sampler.

**The matching aggregation: K draws, one bootstrap.** Jitter alone raises
single-draw variance — that is its job — so the round *verdict* is aggregated
over `config.K_DRAWS = 5` independently jittered draws (user call,
2026-08-19): one model, K cheap evals, pooled into one bootstrap.

- `challenges.build_round_draws(buffer, rng, n, k_draws)` builds K sets from
  one beacon seed via `rng.spawn` — each draw refreshes the pool through its
  own child stream, so draws differ in both composition and windows, and the
  K-set structure is byte-reproducible across validators.
- `evaluate.evaluate_pooled(forecaster, sets)` scores each challenge exactly
  once (the forecaster is never re-called; `evaluate_forecaster` was split
  into `_accumulate`/`_finalize` so per-challenge contributions can be
  stored), pools all K×n contributions, and bootstraps mase/wql/crps over
  them — reporting `*_ci95` / `*_boot_std` so model comparisons are made
  against the stated seed-error margin, not against round-to-round mix luck.
- The live paracast round (`scripts/run_paracast_round.py`) builds
  `--k-draws` (default `K_DRAWS`) jittered draws flattened into one challenge
  list, so the existing paired tests / Friedman / relative-gmean machinery
  aggregates over draws with no further change. The GPU-pod path keeps
  `k_draws=1` — pod inference costs real money per forecast; paracast evals
  are cheap HTTP against an already-loaded panel.

Pooling K=5 draws shrinks the composition-noise component of the verdict by
~sqrt(5) ≈ 2.2×, which more than returns the variance the jitter added, while
each individual draw stays untunable.

**Costs, stated honestly.** ~20 of 128 slots per round are duplicate series
(distinct windows of the same series, further separated by augmentation) —
the price of letting tiny classes participate at a gate-legal share at all.
And a round now covers ~40 classes instead of 108-at-token-presence; full
class coverage is a property of the week, not the day. Both are the intended
trade: breadth incentives live in the expectation, unpredictability in the
realisation.
