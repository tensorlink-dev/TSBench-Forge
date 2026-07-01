# tsbench-forge

A **hard-to-game benchmark for time-series foundation models (TSFMs)**, built on
**fresh, real-world data**.

A TSFM benchmark is fit for purpose only if a high score is achievable **solely
through genuine forecasting capability**. `tsbench-forge` gets there with two
commitments and *defense in depth*: it forecasts **real series** that a model
could not have memorized, and it only counts a challenge as valid when a **frozen
panel of known-quality forecasters ranks in its expected order**. Each known way
to game a forecasting benchmark is closed by an independent layer.

```
python src/demo.py    # end-to-end on real public data: pool -> challenges -> leaderboard
pytest                # determinism / validity / coverage / feeds / sandbox / DSR tests
ruff check .
```

`src/demo.py` runs the whole pipeline offline over locally-scraped parquet
(`src/sources/data`), falling back to a small committed fixture
(`tests/fixtures/sources_data`) so it always runs — no network, no LLM, no
synthetic generator.

## The data: a live catalog of real series

The benchmark forecasts windows of genuinely real public time series — climate,
solar activity, energy load, markets, transport, public health — pulled from a
**curated catalog of daily-or-faster-updating sources** (`src/sources/`, 55
verified feeds across the 7 GIFT-Eval domains). `sources/scraper.py` snapshots
each source to dated parquet; `scraped_source.ScrapedLiveSource` serves those
snapshots into a `FreshBuffer`, sampling **equal-weight across
domain × DGP-class × cadence** so no source-count-heavy domain dominates.

```python
from scraped_source import ScrapedLiveSource
from ingest import FreshBuffer
from challenges import build_live_challenges
from seed import rng_for
import numpy as np

source = ScrapedLiveSource("src/sources/sources.yaml", "src/sources/data", min_series_length=384)
buffer = FreshBuffer(source, pool_size=96, motif_len=384)
buffer.refresh(np.random.default_rng(0xC0FFEE))
reveal = build_live_challenges(buffer, rng_for("0xblock", 0, "reveal"), 128)
```

Each challenge is a real window split into an observed `context` and a held-out
`truth` horizon, z-scored by its context (leak-free) so heterogeneous scales are
comparable, and lightly, **truth-preservingly** augmented so an exact repeat of a
motif is never byte-identical to anything seen before. Every challenge carries its
`domain` / `dgp_class` / `cadence` / `source_id` for stratified scoring and the
breadth gates.

## Two load-bearing design commitments

1. **Validity comes from a frozen reference panel, not from trusting the data.**
   A challenge only counts if a fixed panel of *known-quality* forecasters ranks
   in its known order. If a naive baseline beats the strong model, the challenge
   is measuring an artifact and its validity term goes to zero. The panel's top
   tier (`strong`) is the validity anchor: its quality must be established
   *independently* (see below), not by this benchmark. The panel is **frozen
   within a version** so all validators reach consensus.

2. **Consensus by determinism.** The concrete challenges are a pure function of
   the fixed post-commit pool snapshot and the revealed beacon seed
   (`seed = H(block_hash || epoch || …)`, see `seed.py`). Every validator replays
   byte-identical challenges. There is no LLM and no per-validator search — nothing
   to diverge on.

### How validity and discrimination compose (`score.panel_fitness`)

```
fitness = spread * max(0, ordering)
```

- `spread = (max_err − min_err) / mean_err` — how strongly the challenge set
  separates good forecasters from bad.
- `ordering = kendall_tau(achieved_order, PANEL_QUALITY_ORDER)` — the
  **panel-validity gate**. If a naive model beats `strong`, `ordering` goes
  negative and `max(0, ordering)` zeroes the fitness.

The breadth-aware `score.foundational_fitness` additionally folds in the parrot
gate and the coverage / DGP-class / cadence gates (below), so a sharp-but-narrow
or repetition-solvable set is multiplied down.

## Independently validating the anchor (TSFM-ready) — the key lever

The benchmark's validity rests on the `strong` anchor actually being good — and
checking that on the benchmark's own challenges is circular. `independent_eval.py`
does the non-circular thing: it establishes an anchor's quality on a **held-out,
real, external benchmark** (commodity / transport / demography, disjoint from the
main feed), then promotes the validated anchor and re-checks `validate_panel`.

```
python src/independent_validation.py    # resolves the best anchor available, validates it
```

`resolve_anchor()` returns the strongest anchor this environment can run — a real
**TSFM** (Chronos/TimesFM via `tsfm_adapters`, with `.[chronos]` + staged
weights), else a literature-validated **statsforecast** model (`.[strong]`), else
the numpy placeholder — and reports which.

> **This is now the load-bearing investment.** On raw real data the numpy
> classical anchor is often *not* the best forecaster (drift/EWMA can beat it), so
> `validate_panel` correctly returns `valid=False` with the placeholder. That is
> the gate doing its job: the fix is an independently-validated zero-shot TSFM
> anchor via `score.default_panel(strong_model=...)`, **not** ignoring the gate.

## Defense in depth

| Layer | Module | Gaming vector it closes |
|---|---|---|
| Live ingestion | `ingest.py` / `scraped_source.py` | memorization — data didn't exist at commit time |
| As-of / dedup feed | `feeds.py` | vintage-revision leakage and re-serving a finite feed |
| Leakage audit | `leakage_audit.py` | cross-epoch memorization / a stale feed — behavioural probe, feed-novelty meter, global `t_now` barrier |
| Light augmentation | `challenges.py` | exact-match memorization of a repeated motif |
| Reference-panel validity | `score.panel_fitness` | invalid / saturated challenges (naive beats `strong`) |
| Parrot floor | `baselines.py` + `score.parrot_gate` | repetition — a trivial copy-the-context baseline matching the anchor |
| Domain coverage | `score.coverage_gate` | narrowness — a sharp but single-domain, non-foundational set |
| DGP-class / cadence breadth | `score.foundational_fitness` | pool collapse — some DGP class or cadence band drops below a min-share floor (hard veto) |
| Anchor validation | `score.validate_panel` + `independent_eval.py` | a hollow `strong` anchor, established on a held-out external set |
| Submission static analysis | `static_analysis.py` | hardcoded answers (cheap **pre-filter**) |
| Sandboxed execution | `sandbox.py` | submissions that phone home, OOM, loop, or persist state — the **real** boundary |

## Evaluating an actual TSFM

`evaluate.py` scores a model under test. A forecaster emits a `ProbForecast`
(mean + quantiles) and is judged on the metrics the TSFM literature uses —
**MASE** (point), **WQL** (weighted quantile loss) and **CRPS** (probabilistic) —
plus **calibration** as a first-class axis (**PCE**, interval **coverage**,
**WIS**), then ranked on a leaderboard against the reference panel.

```python
from evaluate import leaderboard, probabilistic_panel
from tsfm_adapters import load_tsfm   # pip install -e ".[chronos]"

models = {"chronos": load_tsfm("chronos"), **probabilistic_panel()}
for row in leaderboard(models, reveal):
    print(row["rank"], row["model"], row["mase"], row["wql"])
```

`probabilistic_panel()` includes the **context-parroting** rung, and
`evaluate.clears_floor(model, reveal)` enforces the minimum bar: a submission must
beat **both** seasonal-naive and parrot — a model that cannot out-predict trivial
copy-the-context has shown no skill (Zhang & Gilpin, arXiv:2505.11349).
`normalized_leaderboard`, `evaluate_multiseed`, and `friedman_test` give robust
aggregation, multi-seed spread, and significance (GIFT-Eval/BOOM conventions).

**Headroom — the go/no-go.** `evaluate.benchmark_has_headroom(probe, reveal)`
injects a deliberately-superior probe and confirms the benchmark rewards it; run
it at setup, because a benchmark with no room above its anchor cannot tell a great
TSFM from a decent classical model.

### Long-horizon dynamics — the hard tier (`dsr_metrics.py`, `dsr_eval/`)

Point metrics saturate to a noise ceiling after ~one Lyapunov time. `dsr_metrics.py`
scores what stays invariant under the dynamics instead (Koppe et al. 2019; Mikhaeil
et al. 2022): `d_stsp` (state-space / invariant-measure misalignment), `d_h` (power-
spectrum misalignment), `valid_prediction_time`, and `max_lyapunov`. The `dsr_eval/`
package is a standalone dynamical-systems reconstruction benchmark with its own
systems zoo, datasets, metrics, and runner.

## Robustness is conditional — invest in these three things

This benchmark is **not unconditionally robust.** "Robust" holds only to the degree
these are invested in:

1. **Anchor quality.** The `strong` anchor must be genuinely good and of
   *independently established* quality — a top, non-leaking zero-shot TSFM or a
   strong classical ensemble. On real data the numpy default is often insufficient;
   `score.validate_panel(..., require=True)` fails loudly at startup if `strong`
   does not lead. Swap in a real TSFM (`independent_eval.py` proves it on a held-out
   set first).

2. **Feed freshness.** The live pool must refresh from sources **outside miners'
   reach**, using as-of/vintage snapshots timestamped *after* the commit point.
   This is now the **primary** anti-memorization defense (augmentation is
   belt-and-suspenders). `feeds.AsOfLiveSource` admits only post-commit motifs,
   `feeds.DedupFreshBuffer` fingerprints and quarantines near-duplicates across
   epochs, and `leakage_audit.default_fresh_buffer` wires dedup + the `t_now`
   barrier into the production default.

3. **Catalog breadth & depth.** Coverage must span many domains, DGP classes, and
   cadences, and each source must accumulate enough history. The catalog grows as
   the cron scrapes daily; today ~29 sources have enough history for the live path,
   with the rest maturing over time. The **`source_discovery`** agent
   (`python -m source_discovery`) keeps this pool diverse and uncontaminated: it
   maps coverage gaps, then an LLM proposes concrete new sources that deterministic
   code vets (contamination denylist, dedup, schema) before a human adds them —
   an offline, human-vetted curation step, never in the scoring path.

## Foundational breadth (domain coverage)

A benchmark certifies a *foundation* model only if a high score means "generalises
across worlds." Breadth is **measured**, not assumed:

- `score.domain_coverage` reports the *effective* number of domains
  (`exp(entropy)` of the domain mix — skew counts against you), and `coverage_gate`
  ramps to 1.0 once a set is broad enough.
- `dgp_class_breadth_gate` / `cadence_breadth_gate` are **hard vetoes**: any DGP
  class or cadence band below a min-share floor sends `foundational_fitness` to 0,
  so the served pool can never silently narrow.
- `score.stratified_fitness` scores each domain separately, so validity and
  discrimination are read per-DGP instead of hidden in one average.

## What this benchmark deliberately does *not* have

Earlier versions drove the benchmark with a **self-improving "forge"**: an LLM ran
a keep/revert autosearch loop over a synthetic data generator to keep the benchmark
hard to game. That machinery — the LLM/OpenRouter proposer, the search loop, the
synthetic generator and its generator-fitting detector — has been **removed**. The
search space it explored was a handful of bounded scalars a classical optimizer
covers exactly, and the robustness that matters for a *general-purpose* TSFM
benchmark comes from the **real data + validity/breadth gates + freshness
discipline**, not from an evolving synthetic generator. Removing it makes the
benchmark simpler, fully deterministic, and free of a non-deterministic boundary —
at the cost of the "moving target over epochs" property, which real, continually
refreshed data provides on its own.

## Repo layout

```
src/
  config.py           global constants + the frozen PANEL_QUALITY_ORDER
  ingest.py           LiveSource ABC (domain-tagged), MixtureLiveSource, FreshBuffer, _finalize
  scraped_source.py   ScrapedLiveSource: the live catalog -> FreshBuffer adapter (the production feed)
  challenges.py       build_live_challenges: real windows -> Challenge, with light truth-preserving augmentation
  score.py            reference panel (frozen strong anchor), validity/parrot/coverage/breadth gates, metrics
  baselines.py        context-parroting floor baseline
  seed.py             commit-reveal deterministic seeding
  feeds.py            production feed discipline: as-of gating, cross-epoch dedup, HTTP/CSV adapter
  leakage_audit.py    contamination-resistant default buffer, t_now barrier, memorization probe, feed-novelty meter
  live_feeds.py       curated real public-data adapters (CSV/dated), cached fetch, real mixture
  daily_feeds.py      daily-updated public-source adapters (JSON path engine), curated registry
  pool_sampler.py     equal-weight-per-domain-per-DGP-class eval-pool sampler + generalization gap
  independent_eval.py held-out external validation set, anchor resolution (TSFM/statsforecast/numpy), leakage gap
  independent_validation.py  end-to-end proof: validate the best available anchor, then promote it
  evaluate.py         model-under-test scoring: MASE/WQL/CRPS + calibration, floor check, robust aggregation, significance
  dsr_metrics.py      long-horizon dynamics: D_stsp / D_H / valid-prediction-time / Lyapunov + free-running rollout
  tsfm_adapters.py    real Chronos / TimesFM adapters (the actual TSFMs under test)
  static_analysis.py  AST/regex linter for miner submissions (cheap pre-filter)
  sandbox.py          isolated, resource-limited execution of submissions (the real boundary)
  dsr_eval/           standalone dynamical-systems (DSR) eval package: systems zoo, datasets, metrics, runner, report
  source_discovery/   LLM catalog-curation agent: map coverage gaps -> propose sources -> deterministic vetting (offline, human-vetted)
  sources/            the live-data catalog: sources.yaml, scraper.py, DGP_TAXONOMY.md, samples/, data/
  demo.py             runnable end-to-end demo on real public data
notebooks/            example.ipynb — full guided walkthrough with plots
experiments/          runnable experiment notebooks (live_feeds, independent_validation) + their builders
docs/                 REWARD_HACKING.md, PRODUCTION_GRADE_ROADMAP.md
tests/                determinism, validity, coverage, feeds, anchor, baselines, robust metrics, DSR, leakage, sandbox
```

## Notes

- Python ≥ 3.11. The core path is **numpy only**; reading the scraped parquet
  catalog uses `pyarrow` + `pyyaml` + `pandas`. The classical `strong` anchor can
  optionally use `statsforecast` (lazily imported).
- Everything is seeded and reproducible — no global mutable RNG, no LLM, and no
  network in the core path (the scraped parquet is on disk; live-feed adapters
  cache their pulls).
