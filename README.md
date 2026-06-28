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

For a guided, visual tour, open **`example.ipynb`** (`pip install -e ".[notebook]"`)
— it walks the whole pipeline with plots: the forge climbing, commit-reveal,
panel validity, the MASE/WQL/CRPS leaderboard, the headroom check, and sandboxed
submissions.

## Running on real data

The synthetic zoo proves the wiring offline; `live_feeds.py` runs the **same**
pipeline on genuinely real public series (climate, solar activity, atmospheric
CO₂, equities, weather):

```
python live_demo.py     # forge + leaderboard + headroom on real feeds (caches pulls)
```

```python
from live_feeds import build_real_live_source
from ingest import FreshBuffer

source = build_real_live_source()              # multi-domain feed over real data
buffer = FreshBuffer(source, pool_size=96, motif_len=384)
buffer.refresh(np.random.default_rng(0xC0FFEE))   # then run_forge / build_challenges as usual
```

Every adapter takes an injected `fetch` (defaulting to an on-disk cache over
`urllib`), so tests run fully offline and reruns never re-hit the network. The
`DatedCsvFeed` adapter reads the date column and stamps each window with its
availability time, so it drops straight into `feeds.AsOfLiveSource` for vintage
gating. The curated `live_feeds.REGISTRY` is the list of bundled feeds; point the
adapters at your own as-of vendor endpoints for production. **`experiments/live_feeds.ipynb`**
(`python experiments/build_live_feeds_notebook.py` to (re)build it) is the visual
real-data walkthrough.

## Independently validating the anchor (TSFM-ready)

The benchmark's validity rests on the `strong` anchor being good — but checking
that on the forge's *own* challenges is circular. `independent_eval.py` does the
non-circular thing: it establishes an anchor's quality on a **held-out, real,
external benchmark the forge never touches** (commodity / transport / demography),
then promotes the validated anchor and re-checks `validate_panel`.

```
python independent_validation.py    # resolves the best anchor available, validates it
```

`resolve_anchor()` returns the strongest anchor this environment can actually run —
a real **TSFM** (Chronos/TimesFM via `tsfm_adapters`, with `.[chronos]` + staged
weights), else a literature-validated **statsforecast** model (`.[strong]`), else
the numpy placeholder — and reports which, so a run is never silently on the
placeholder. `is_independently_validated(...)` is the go/no-go (the anchor must beat
every classical baseline on the held-out set) and `leakage_gap(...)` contrasts the
held-out score with the forge score as the README's leakage detector.
**`experiments/independent_validation.ipynb`** is the visual walkthrough; with
`.[chronos]` installed it is the independent validation of a *real neural TSFM*.

## Defense in depth

| Layer | Module | Gaming vector it closes |
|---|---|---|
| Live ingestion | `ingest.py` | memorization — data didn't exist at commit time |
| Domain coverage | `domains.py` + `score.py` | narrowness — a sharp but single-DGP, non-foundational benchmark |
| Augmentation | `generate.py` | exact-match memorization |
| Synthetic composition | `generate.py` | coverage gaps live data misses |
| Real-motif splice | `generate.py` | synthetic-only fingerprints |
| Blend controller | `generate.py` | knowing the test composition in advance |
| Commit-reveal seed | `seed.py` | precomputing the concrete challenges |
| Reference-panel fitness | `score.py` | invalid / saturated challenges |
| Overfit detector | `score.py` | generator-fitting (exploiting synthetic quirks) |
| Parrot floor | `baselines.py` + `score.parrot_gate` | repetition — a trivial copy-the-context baseline already matching the anchor |
| Submission static analysis | `static_analysis.py` | hardcoded answers (cheap **pre-filter**) |
| Sandboxed execution | `sandbox.py` | submissions that phone home, OOM, loop, or persist state — the **real** boundary the linter only pre-screens for |
| As-of / dedup feed | `feeds.py` | vintage-revision leakage and re-serving a finite feed (memorization) |
| Leakage audit | `leakage_audit.py` | cross-epoch memorization / a stale feed — behavioural probe, feed-novelty meter, global `t_now` barrier |
| Anchor validation | `score.validate_panel` / `validate_generalization` | a hollow `strong` anchor, or challenges overfit to the one frozen panel |
| Forge loop | `forge_loop.py` + `forge_llm.py` | benchmark saturation over time (LLM-driven via OpenRouter) |

## Running the forge with a real LLM (OpenRouter)

The forge's `propose_mutation` is the one non-deterministic boundary — where an
LLM reads `program.md` + the metric history and proposes the next one-knob move.
`forge_llm.py` implements that against **OpenRouter** (stdlib `urllib`, no extra
dependency). Enable it with an API key:

```bash
export OPENROUTER_API_KEY=sk-or-...
export OPENROUTER_MODEL=anthropic/claude-opus-4.8   # default; any OpenRouter model
python demo.py                                      # forge now climbs via the LLM
```

```python
from forge_llm import make_openrouter_proposer, OpenRouterConfig
from forge_loop import run_forge
run_forge(buffer, epochs, block_hash, init_state,
          proposer=make_openrouter_proposer(OpenRouterConfig.from_env()))
```

Other env knobs: `OPENROUTER_BASE_URL`, `OPENROUTER_TEMPERATURE`,
`OPENROUTER_MAX_TOKENS`, `OPENROUTER_TIMEOUT`, `OPENROUTER_MAX_RETRIES`. Three
properties keep the LLM consensus-safe and robust:

- **Runs once per epoch.** Only the committed manifest + revealed seed determine
  the challenges, so validators never call the LLM and never diverge.
- **One-knob by construction.** Only the single `(knob, value)` the model returns
  is applied, then `normalized().clamped()` forces it back into the legal
  envelope — the `program.md` invariants can't be violated even by a bad reply.
- **Fail-safe.** No key, a network error, malformed output, or an illegal knob
  all fall back to the deterministic heuristic; the forge degrades, never breaks.

## Evaluating an actual TSFM

The forge keeps the benchmark hard-to-game; `evaluate.py` is the half that
**scores a model under test**. A forecaster emits a `ProbForecast` (mean +
quantiles) and is judged on the metrics the TSFM literature uses — **MASE**
(point accuracy), **WQL** (weighted quantile loss) and **CRPS** (probabilistic),
plus **calibration** as a first-class axis (**PCE**, interval **coverage**,
**WIS**) because CRPS/WQL conflate calibration with sharpness — then ranked on a
leaderboard against the reference panel.

```python
from evaluate import leaderboard, probabilistic_panel
from tsfm_adapters import load_tsfm   # pip install -e ".[chronos]"

models = {"chronos": load_tsfm("chronos"), **probabilistic_panel()}
for row in leaderboard(models, reveal):   # reveal = the committed challenges
    print(row["rank"], row["model"], row["mase"], row["wql"])
```

`probabilistic_panel()` includes the **context-parroting** rung, and
`evaluate.clears_floor(model, reveal)` enforces the minimum bar: a submission must
beat **both** seasonal-naive and parrot (`FLOOR_BASELINES`) — a model that cannot
out-predict trivial copy-the-context has shown no skill (Zhang & Gilpin,
arXiv:2505.11349). For credible numbers across datasets, `normalized_leaderboard`
aggregates seasonal-naive-normalised scores by rank + shifted geometric mean
(GIFT-Eval/BOOM convention), `evaluate_multiseed` reports mean ± std over
seeds/origins, and `friedman_test` checks whether leaderboard gaps are significant.

`tsfm_adapters.py` ships real Chronos / TimesFM adapters (lazy torch import;
weights are staged validator-side, outside the submission sandbox). Point-only
models are lifted to probabilistic via `evaluate.probabilistic`, so a classical
baseline and a frontier TSFM are scored on the same footing.

### Long-horizon dynamics — the hard tier (`dsr_metrics.py`)

Point metrics saturate to a noise ceiling after ~one Lyapunov time: on a chaotic
system *every* forecast decorrelates from the truth, so short-horizon error can't
tell a model that learned the dynamics from one that memorised a plausible
squiggle. `dsr_metrics.py` scores what stays invariant under the dynamics instead
(Koppe et al. 2019; Mikhaeil et al. 2022; DynaMix):

- `d_stsp` — **geometric** misalignment (KL between delay-embedded state-space
  histograms / invariant measures): catches attractor collapse and mean-reversion.
- `d_h` — **temporal** misalignment (Hellinger between power spectra): catches
  wrong frequencies and the parroting cyclic-collapse (peaked vs broadband).
- `valid_prediction_time` (optionally in Lyapunov-time units) and `max_lyapunov`
  (`|λ_gen − λ_true|` is the sharpest anti-parroting scalar; a periodic collapse
  has λ ≈ 0). `free_run` rolls a fixed-horizon model out to the thousands of steps
  these invariant-measure metrics need.

The two misalignments are orthogonal — each alone is gameable, together they are
hard to fool, and parroting provably loses on them.

**Headroom — the go/no-go.** A benchmark only *certifies* TSFMs if a genuinely
better model scores measurably better than the classical anchor on it.
`evaluate.benchmark_has_headroom(probe, reveal)` injects a deliberately-superior
probe and confirms the benchmark rewards it; run it at setup, because a benchmark
with no room above its anchor cannot tell a great TSFM from a decent classical
model no matter what the leaderboard prints.

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
   real zero-shot TSFM via `score.default_panel(strong_model=...)` (or
   `score.panel_from_env`) for production. **Don't trust it blindly** —
   `score.validate_panel(challenges, panel, require=True)` fails loudly at startup
   if `strong` does not actually lead the baselines.

2. **Feed freshness.** The live pool must refresh from sources **outside miners'
   reach**, using as-of/vintage snapshots timestamped *after* the commit point. A
   finite or predictable feed degrades freshness back toward leakability. Always
   quarantine and dedup every pull. `feeds.py` provides this discipline:
   `AsOfLiveSource` admits only post-commit motifs, `DedupFreshBuffer`
   fingerprints every motif ever served and quarantines near-duplicates across
   epochs, and `HttpCsvLiveSource` is a real adapter to point at a vendor feed.
   `leakage_audit.default_fresh_buffer` wires dedup + the as-of/`t_now` barrier
   into the production default; `memorization_probe` and `feed_novelty` then audit
   whether a model is suspiciously better on seen data, or the feed has gone stale.

3. **Recipe-space size.** The generator must *out-produce saturation*: the space
   of reachable challenges must be large enough that miners cannot cover it. Too
   small a space and the arms race tips toward the miners no matter how clever the
   forge is.

## Foundational breadth (domain coverage)

A benchmark certifies a *foundation* model only if a high score means "generalises
across worlds," not "good at the one process we test." Breadth therefore has to be
**generated and measured**, not assumed:

- **A multi-domain feed.** `domains.py` ships a dependency-free zoo of distinct
  data-generating processes — random-walk plus dynaprior-inspired dynamical
  systems (Lorenz-63, Rössler, Hopf limit cycle, Hénon, logistic map,
  Ornstein–Uhlenbeck, jump-diffusion) — blended by `MixtureLiveSource`. Each motif
  carries its `domain`, which every `spliced` / `aug_live` challenge inherits. A
  literal `dynaprior` adapter is the documented extension point; this zoo is the
  offline, numpy-only stand-in that proves the wiring.
- **A coverage gate.** `score.domain_coverage` reports the *effective* number of
  domains (`exp(entropy)` of the domain mix — skew counts against you), and
  `coverage_gate` ramps to 1.0 once a set is broad enough. Like the other gates it
  only ever multiplies a sharp-but-narrow benchmark *down*.
- **Stratified reporting.** `score.stratified_fitness` scores each domain
  separately, so validity/discrimination is read per-DGP instead of hidden in one
  average — the demo's table shows, e.g., that the panel orders cleanly on
  `random_walk` but only weakly on some chaotic maps (the validity-gate ↔ exotic-DGP
  tension, made visible).

Coverage is **report-first**: `panel_fitness` reports `coverage` / `coverage_gate`
and `parrot_gate` but does not fold them into the frozen `fitness` (which stays the
consensus yardstick, comparable across validators within a version). A forge that
must climb toward breadth and non-parrotability *as well as* discrimination targets
the opt-in `score.foundational_fitness = fitness × coverage_gate × parrot_gate`
instead — pass `forge_loop.FOUNDATIONAL_OBJECTIVE` to `run_forge(objective=...)`.

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
ingest.py           LiveSource ABC (domain-tagged), SyntheticLiveSource stand-in, FreshBuffer
domains.py          multi-domain DGP zoo (dynaprior-inspired) + MixtureLiveSource
generate.py         primitives, augmentations, Recipe grammar, blend controller
score.py            reference panel (frozen strong anchor + overfit detector), metrics, coverage, parrot gate, validate_panel/generalization
baselines.py        context-parroting floor baseline (the repetition floor every model must clear)
seed.py             commit-reveal deterministic seeding
static_analysis.py  AST/regex linter for miner submissions (cheap pre-filter)
sandbox.py          isolated, resource-limited execution of submissions (the real boundary)
feeds.py            production feed discipline: as-of gating, cross-epoch dedup, HTTP/CSV adapter
leakage_audit.py    contamination-resistant default buffer, global t_now barrier, memorization probe, feed-novelty meter
live_feeds.py       real public-data adapters (CSV/dated), cached fetch, curated REGISTRY, real mixture
daily_feeds.py      daily-updated JSON-API adapters (path engine + DatedJsonFeed), curated DAILY_REGISTRY, daily mixture
live_demo.py        end-to-end demo on real public feeds (live analogue of demo.py)
independent_eval.py held-out external validation set, anchor resolution (TSFM/statsforecast/numpy), leakage gap
independent_validation.py  end-to-end proof: validate the best available anchor, then promote it
experiments/        runnable experiment notebooks (live_feeds, independent_validation) + their builders
forge_loop.py       the keep/revert autoresearch loop over GeneratorState (fitness or foundational objective)
forge_llm.py        OpenRouter-backed forge proposer (the LLM boundary) + fail-safe fallback
evaluate.py         model-under-test scoring: MASE/WQL/CRPS + calibration (PCE/coverage/WIS), floor check, robust aggregation, multi-seed, significance
dsr_metrics.py      long-horizon dynamics: D_stsp / D_H / valid-prediction-time / Lyapunov + free-running rollout
tsfm_adapters.py    real Chronos / TimesFM adapters (the actual TSFMs under test)
program.md          the forge LLM's instructions (what it may/may not change)
demo.py             runnable end-to-end demo
docs/               PRODUCTION_GRADE_ROADMAP.md — peer-benchmark synthesis + P0–P3 plan and status
tests/              determinism, validity, forge, static-analysis, domains/coverage, llm, sandbox, feeds, anchor, baselines, robust metrics, DSR, leakage audit
```

## Notes

- Python ≥ 3.11. The core path is **numpy only**; the classical `strong` anchor
  can optionally use `statsforecast` (lazily imported, see
  `score.try_statsforecast_strong`) without affecting the numpy demo.
- Everything is seeded and reproducible — no global mutable RNG, no network in
  the core path. Live ingestion is an abstraction with a synthetic stand-in; real
  adapters are a documented extension point.
