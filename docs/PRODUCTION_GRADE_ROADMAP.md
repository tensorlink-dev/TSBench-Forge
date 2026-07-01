# TSBench-Forge → Production-Grade, Academically-Credible TSFM Benchmark

*A synthesis of what BOOM, GIFT-Eval, TIME, TempusBench, DynaMix and our own
horizon-forge experiment teach us, and a concrete roadmap to make
`tsbench-forge` the hardest, fairest, hardest-to-game TSFM benchmark — and one
that reviewers at ICML/NeurIPS/ICLR will accept.*

---

## 0. TL;DR — where we are and the one-sentence thesis

`tsbench-forge` owns a thesis nobody else in this list owns: **validity is
enforced, not assumed.** The benchmark forecasts **real data only** — challenges
are windows of live scraped series (`ScrapedLiveSource` over `src/sources/`,
assembled by `challenges.build_live_challenges`) — and freshness / as-of vintage
gating (`feeds.py`, `leakage_audit.py`), commit-reveal seeding (`seed.py`), and a
stack of multiplicative validity gates in `score.py` (panel-validity + parrot +
coverage + DGP-class/cadence breadth) make it the only design here that treats
"a high score must mean genuine forecasting skill" as a *property to be enforced*,
not hoped for. Anti-gaming rests on the real-data freshness discipline plus the
validity / parrot / coverage / breadth gates.

Everyone else is a static (or lightly-refreshed) dataset suite; we are a
*refreshing, live* one. That is our moat. But a moat is not a paper. To be
production-grade **and** academically acceptable we must bolt onto that live core
the three things the static benchmarks already do well and we currently stub:

1. **Real, diverse, *provably-uncontaminated* data at scale** (the scraped
   catalog spans 7 GIFT-Eval domains and 35 DGP classes; GIFT-Eval has 144k
   series / 177M points, BOOM has 2,807 real observability series, TIME has 50
   fresh post-cutoff datasets).
2. **A capability taxonomy** that says *which property of "understanding time"*
   each challenge probes — and reports per-capability, not one scalar.
3. **The standard rigor checklist** reviewers now demand for TSFMs: leakage
   provably handled, mandatory seasonal-naive normalization, proper
   probabilistic + *calibration* metrics, multiple seeds + significance,
   documented protocol, licensing, public leaderboard.

The rest of this doc is the specific plan.

---

## 1. What each peer teaches us (distilled, actionable)

### GIFT-Eval — the credibility template (arXiv:2410.10393)
The reference for *how an academic TSFM benchmark is structured.* Steal wholesale:
- **Coverage matrix as a first-class object**: 7 domains × 10 frequencies ×
  {short=1×, medium=10×, long=15× horizon} × {uni, multivariate}. We currently
  hard-code `HORIZON=48`, `CONTEXT_LEN=256` (`config.py`). We need a *task grid*.
- **Non-leaking pretraining corpus shipped alongside the benchmark**
  (`Salesforce/GiftEvalPretrain`, ~230B points, eval datasets excluded). This is
  *the* move that makes zero-shot claims defensible. Our analogue is stronger and
  needs no shipped corpus: the live feed only ever serves *post-commit* vintages,
  so what we score is disjoint-by-construction from any pretraining set (§3.1).
- **MASE + CRPS, each normalized against Seasonal Naive, aggregated by average
  Rank** across configs. We already compute MASE/WQL/CRPS (`evaluate.py`) — adopt
  their *rank-aggregation* and *seasonal-naive normalization* exactly so numbers
  are comparable to the leaderboard people already trust.
- **Metadata-driven honesty**: every submission carries
  `model_type ∈ {statistical, DL, pretrained, fine-tuned, zero-shot, agentic}`,
  `testdata_leakage: Yes/No`, `replication_code_available`. Cheap, and it is most
  of what "academically credible" means in practice.
- **Reference implementations as notebooks + PR-based submission + git
  leaderboard.** Lower friction than a model registry.

### BOOM / Toto — real-world messiness + probabilistic eval done right (`/toto`)
BOOM (2,807 Datadog observability series) is the antidote to "benchmarks are too
clean" (the LTSF critique, §3). Steal:
- **Heavy-tailed, sparse, bursty, high-cardinality, non-stationary real data** as
  a domain we deliberately include — observability metrics, not just smooth
  econ/energy. The scraped catalog already leans real-world-messy (MTA, Binance,
  weather); the dynamical-systems tier (`dsr_eval/`, `dsr_metrics.py`) supplies
  the opposite, chaotic extreme — we want both ends.
- **Zero-inflation is a real failure mode**: ~9.8% of BOOM has near-zero
  variance, which wrecks MASE/MAPE. BOOM evaluates zero-inflated series
  *separately* and aggregates with a **shifted geometric mean** (eps=1e-5) — far
  more robust than arithmetic mean across heterogeneous scales. We should adopt
  shifted-gmean aggregation and a zero-inflated bucket.
- **`MaskedTimeseries` data format** (series + padding_mask + id_mask +
  timestamp + interval) cleanly unifies univariate / multivariate / missing /
  exogenous. Adopt as our challenge payload type so multivariate isn't a retrofit.
- **Quantile-head probabilistic forecasting + GluonTS metric reuse** (CRPS via
  `MeanWeightedSumQuantileLoss`, MSIS, etc.). Reuse GluonTS `evaluate_model`
  rather than re-implementing metrics — it's what every other benchmark reports,
  so our numbers stay comparable and audit-free.
- **Breakdown tables by domain/type/term**: report *where* a model wins, not just
  a scalar. This is the engineering form of our needed capability taxonomy.

### TIME — fresh data + feature-stratified analysis (arXiv:2602.12147 / ICML'26)
- **Fresh-by-construction datasets** (post-dating the audited TSFMs) → an
  external negative-control with ~100% non-leakage. Validates our live-feed
  ambition and gives us a citable partner/curation strategy.
- **88 STL/FFT features per series** (trend strength, seasonal strength, Hurst,
  entropy, spectral periods, stationarity/ADF, ARCH) → **stratify results by
  feature bucket.** This is the cheapest path to a capability taxonomy on *real*
  data: compute features, then report MASE-vs-(trend_strength, seasonality,
  stationarity). Lift `src/timebench/feature/features.py` directly.
- **Term-based horizons normalized by seasonality** (short=1× seasonal,
  medium=10×, long=15×). Generalizes our single `HORIZON`.
- **Quantile prediction interface where the model just emits quantiles and the
  framework computes everything** — matches our `ProbForecast`.

### TempusBench — execution rigor & proper scoring (`/TempusBench`)
- **Isolated per-model conda environments + subprocess execution + JSON
  serialization** — solves the dependency-hell of evaluating ~50 heterogeneous
  models. Complements (does not replace) our `sandbox.py`; sandbox is the
  *security* boundary, isolated envs are the *reproducibility* boundary.
- **Rolling-window eval with per-window hyperparameter selection on the validate
  split only** — textbook leakage-safe protocol; our eval is single-origin, we
  should add rolling origins.
- **Proper scoring rules incl. WIS (Weighted Interval Score)** with explicit
  sharpness + calibration decomposition. We have CRPS/WQL; add WIS and a
  dedicated calibration metric (§3).
- **Win-rate + skill-score aggregators** (geometric mean of clipped relative
  errors vs seasonal-naive). Another robust aggregation option alongside rank.
- Its **gap is exactly our opportunity**: synthetic tasks are *static CSVs with
  no difficulty knobs.* We generate procedurally with controllable difficulty —
  that's our differentiator over TempusBench.

### DynaMix — how to test whether a model *understands dynamics* (NeurIPS'25, `/DynaMix-python`)
This is the scientific heart of "truly understands time." Short-horizon MASE/CRPS
**cannot** tell you if a model learned the *dynamics* vs. parroting context. Steal:
- **Long-term invariant metrics** that survive after pointwise prediction fails:
  - **D_stsp** — KL divergence between the *attractor/state-space histograms* of
    generated vs. true trajectories (does the model reproduce the invariant
    measure / attractor geometry?).
  - **D_H** — Hellinger distance between *power spectra* (does it reproduce the
    temporal correlation structure / dominant frequencies?).
  These are observable-independent, scale-invariant after standardization, and
  *decouple geometry from temporal dynamics*. Implement both
  (`src/dynamix/metrics/metrics.py` is the reference).
- **Chaotic dynamical systems as provably-uncontaminated, difficulty-tunable
  tasks.** Difficulty has a *physical* axis: the **largest Lyapunov exponent λ**
  (Lyapunov time τ = 1/λ); Valid Prediction Time measured in Lyapunov times is
  the natural hardness-normalized horizon. `dsr_eval/systems.py` already carries
  Lorenz / Rössler (with reference λ, Lyapunov time, and Kaplan-Yorke dimension)
  plus a `dysts` bridge; `dsr_metrics.py` reports VPT.
- **Regime/parameter generalization tiers**: train-regime vs. bifurcated-regime
  vs. interpolate/extrapolate across the parameter (e.g., Lorenz ρ). A natural
  hard-tier difficulty ladder.

### horizon-forge — our own validated precursor (`/horizon-forge`)
Already proved out the generation machinery we should consolidate into
`tsbench-forge`:
- **Canonical-JSON recipe + SHA-256 hash + SQLite registry** (`tracker.py`) →
  O(1) duplicate rejection and a `sample_novel()` loop. This is how a *dynamic*
  benchmark guarantees it never re-serves an item — directly reusable.
- **Forecasting-coherent augmentation library** (`augmentations.py`, 14 ops, each
  with a literature citation and a fully-reconstructive recipe): spectral ops
  (frequency_mask/FrAug, frequency_mix), moving-block bootstrap, time/magnitude
  warp, window warp, TSMixup, plus `inject_anomaly` / `inject_regime_change` with
  tunable location & magnitude = ready-made difficulty knobs. **Lift verbatim.**
- **KernelSynth GP-prior synthesis** (`kernels.py`) — held-out-by-construction,
  same approach Chronos used for ~half its training data. Lift verbatim.
- **Verified live sources with full provenance, as-of gating, cross-epoch dedup,
  frequency-weighted sampling.** This is the real-data backbone — now **landed**:
  the `src/sources/` catalog + scraper are consolidated in-repo and wired to the
  live pool through `scraped_source.ScrapedLiveSource`, which is the production
  feed `challenges.build_live_challenges` samples from.
- Honest gaps it flagged: no difficulty stratification, 1-D only (no
  multivariate), single-shot (no adaptive/feedback generation). We fix these.

---

## 2. The capability taxonomy — what "truly understands time" means, operationally

Reviewers and users both ask "*what is this benchmark actually measuring?*" We
answer with an explicit taxonomy. Every challenge is tagged with the capabilities
it stresses, and the leaderboard reports **per-capability scores**, not one
number. Proposed axes (each maps to a slice of the live catalog, its DGP
taxonomy, or a metric we already have or can lift):

| Capability | Probes whether the model… | How we test it | Source to lift |
|---|---|---|---|
| **Trend / changepoint** | tracks local, breaking trends (not a global line) | real trending/changepoint DGP classes (`sources/DGP_TAXONOMY.md`), stratified | live catalog |
| **Seasonality / multi-period** | locks onto multiple seasonal periods, incl. unseen ones | seasonal DGP classes, TIME's seasonal-strength buckets | live catalog, TIME features |
| **Non-stationarity / regime shift** | adapts across distribution shifts & level/variance regimes | `inject_regime_change`, regime-switched AR noise | horizon-forge augs |
| **Long-horizon / long-context** | uses long history & forecasts far without collapse | term grid (1×/10×/15× seasonal), 30× stress | TIME / GIFT term design |
| **Chaotic dynamics (the hard tier)** | reconstructs the *attractor*, not just next steps | `dsr_eval/systems.py` + `dysts` systems, scored by D_stsp / D_H / VPT | DynaMix |
| **Uncertainty calibration** | emits honest predictive distributions | PCE/WIS + coverage on all of the above | §3 |
| **Multivariate / cross-series** | exploits genuine cross-channel dependence | coupled ODEs, BOOM high-cardinality (F-score-verified) | Toto/BOOM, DynaMix |
| **Robustness** | tolerates noise, missingness, outliers, sparsity/zero-inflation | jitter/cutout augs, BOOM zero-inflated bucket | horizon-forge, BOOM |
| **Anomaly sensitivity** | reacts correctly to injected shocks | `inject_anomaly` at known location/magnitude | horizon-forge |

The **chaotic-dynamics + invariant-metrics** row is what differentiates a "good
curve fitter" from a model that "understands time," and almost no production
benchmark scores it. It is our headline scientific contribution.

---

## 3. The academic-credibility gap analysis (with citations)

What recent literature says reviewers now require, and where we stand.

### 3.1 Leakage / contamination — the #1 threat to TSFM-benchmark credibility
- Meyer et al., *Rethinking Evaluation in the Era of TSFMs: (Un)known
  Information Leakage* (arXiv:2510.13654): tracing **22 TSFMs across 401
  datasets, only 6% of datasets have never appeared in any model's
  pretraining/fine-tuning corpus** — "one model's pre-training corpus is another
  model's test set." Two leakage types: **direct** (sample overlap) and
  **indirect/temporal** (correlated disjoint series sharing a driver, e.g. COVID;
  empirically ~43% MAE improvement with *no test point ever seen*). Requires (R1)
  test sets that *provably could not* be in any pretraining corpus, and (R2) a
  **global post-training temporal barrier `t_now`**: every test point dated after
  the latest pretraining cutoff across all compared models.
- GIFT-Eval's own ablation: ~0.1% leaked data improves MAPE ~8/15/29 pts at
  short/medium/long horizons, **larger models benefit more** (capacity amplifies
  contamination).
- TSFMAudit (arXiv:2605.26161): static loss/perplexity is *unreliable* for TS
  contamination detection; contamination shows up as *unusually efficient
  adaptation* under a fine-tuning probe. Concrete trap: Monash `Elecdemand` is a
  1/1000-rescaled subset of `Australian Electricity Demand` — exact-match misses it.

**Where we stand:** *This is our strongest hand and we should play it hard.*
Contamination resistance now rests entirely on the **real-data freshness
discipline**: commit-reveal seeding + as-of / vintage gating + cross-epoch dedup
(`feeds.py`, `leakage_audit.py`) mean only *post-commit* live data is ever scored,
and `leakage_audit.global_t_now` implements the R2 barrier
(`t_now = max(model cutoffs)`). This is a strong anti-contamination story: real
data that provably could not have been in a pretraining corpus. Action items: (a) point
`AsOfLiveSource` / `HttpCsvLiveSource` at real vendor-timestamped endpoints so the
barrier binds on live vintages; (b) keep the leakage/contamination *audit*
(`leakage_audit.memorization_probe` / `feed_novelty`, TSFMAudit-style) as a
reported column; (c) state the contamination-resistance argument as Theorem-ish
prose in the paper.

### 3.2 Strong simple baselines + the "benchmarks reward complexity that isn't there" critique
- Zeng et al., *Are Transformers Effective for Time Series Forecasting?*
  (AAAI'23, arXiv:2205.13504): a **one-layer linear model beats all LTSF
  Transformers on all 9 benchmarks by 20–50% MSE**; "repeat-last-value" beats them
  ~45% on Exchange. Self-attention is permutation-invariant → poor temporal order.
- M4 (Makridakis et al., 2020): pure-ML methods mostly failed to beat Naïve2/Comb;
  combinations of statistical methods won.
- Hewamalage et al., *Forecast Evaluation… Pitfalls and Best Practices* (DAMI'23):
  **always** compare against naive **and** seasonal-naive; pick metrics by data
  characteristics; MAPE invalid near zero.

**Where we stand:** Good — our panel *requires* `strong` to beat naive baselines
(`PANEL_QUALITY_ORDER`, the ordering gate), which is exactly the discipline above,
enforced automatically. Action: make seasonal-naive the **mandatory normalizer**
for every reported metric (GIFT-Eval/BOOM convention) and publish naive/seasonal
-naive/AutoARIMA/AutoETS/AutoTheta as permanent leaderboard rows.

### 3.2b Context parroting — the baseline that should terrify a *hard* benchmark
- Zhang & Gilpin, *Context Parroting: A Simple but Tough-to-Beat Baseline for
  Foundation Models in Scientific ML* (arXiv:2505.11349): a trivial
  **nearest-neighbor "copy-the-highest-correlating-context-window" baseline beats
  Chronos, Chronos-Bolt, TimesFM, Time-MoE, Moirai — and DynaMix —** on chaotic
  systems, turbulence, coupled oscillators and ECG, at ~6 orders of magnitude less
  compute. The lesson: much of what looks like "TSFM skill" is **induction-head
  copying**, with **mean-reversion** at long horizons (esp. MSE-trained models).
  "If a foundation model cannot beat context parroting, it arguably has failed to
  learn the underlying physics."

**Why this is load-bearing for us:** a *hard-tier* benchmark must (a) ship
**context-parroting as a second mandatory floor baseline** alongside seasonal-naive
— any model that doesn't clear *both* is not demonstrating understanding; and
(b) deliberately include tasks that **cannot be solved by repetition** (aperiodic
chaos beyond one Lyapunov time, regime changes after the copyable window, novel
parameter regimes). This is the single sharpest blade we can add, and it pairs
perfectly with the dynamics tier (§2, P1-6): parroting *looks* fine on
short-horizon MASE but fails D_stsp/D_H, so reporting both exposes it. Note: even
DynaMix — the DS-specialist — is beaten by parroting on some systems, so we cannot
assume any single model "solves" the hard tier; the benchmark's job is to *expose
the gap*, which parroting makes measurable.

### 3.3 Probabilistic metrics + *calibration* (not just CRPS)
- Adler et al., *Beyond Accuracy: Are TSFMs Well-Calibrated?* (ICLR'26,
  arXiv:2510.16060): CRPS/WQL/MSIS conflate **calibration and sharpness** (Chung
  et al. 2021) and can crown a poorly-calibrated model as best. Use a **dedicated
  calibration metric — PCE (Probabilistic Calibration Error)** + CCE for
  over/under-confidence direction. Findings: Gaussian heads under-confident;
  quantile/Student-t/mixture heads well-calibrated; **all autoregressive TSFMs
  drift overconfident at long horizons.**

**Where we stand:** We have MASE/WQL/CRPS and we lift point models to Gaussian
bands. Action: add **PCE + quantile-coverage + WIS**, and report calibration as a
first-class axis (it's a capability in §2). This is cheap and very on-trend —
"calibration of TSFMs" is a 2026 hot topic and a free differentiator.

### 3.4 Multiple seeds + significance testing
- Hewamalage et al. (DAMI'23): error numbers alone don't show whether
  differences are real; use **Diebold-Mariano**, **Friedman + Nemenyi/Holm**
  post-hoc, Wilcoxon. The leakage paper itself reports 10-seed mean±std.

**Where we stand:** The multi-seed + significance machinery is in place —
`evaluate.evaluate_multiseed` and `evaluate.friedman_test` — and challenge
assembly is fully seed-deterministic (`build_live_challenges` via `rng.spawn`), so
common-random-number variance reduction across models is free. Action: wire these
into the reported leaderboard (run each model over N seeds/origins, report
mean±std + **Friedman + post-hoc**). Multiple test windows (rolling origins, §1
TempusBench) give the samples for free.

### 3.5 Benchmark-gaming / cherry-picking / dataset-simplicity critiques
- Roque et al., *Cherry-Picking in TSF* (AAAI'25): with **4 cherry-picked test
  sets, 46% of models can be made to look SoTA**; DL models more susceptible.
- Abdelmalak et al. (arXiv:2502.09683): standard LTSF datasets have **very weak
  channel dependence** (Granger F<2), are "overly simplistic," favor
  channel-independent models; **fixed look-back L=96 can invert rankings.** Real
  cross-channel structure only appears in coupled-ODE data.
- Qiu et al., *TFB* (PVLDB'24, arXiv:2403.20150): the inherited Informer codebase
  **"Drop-Last" bug** makes reported metrics depend on batch size (PatchTST ETTh2
  MSE 0.414→0.348 just by enlarging batch). Disable drop-last in testing.

**Where we stand:** Anti-gaming is structural: cherry-picking is hard because the
benchmark commit-reveal *assembles and seeds the test set* from a fresh live
catalog (`build_live_challenges`), and the breadth/coverage gates (§4) veto any
narrow pool — the submitter never picks the series.
Action: (a) for the **multivariate** axis, *verify* cross-channel dependence
(Granger F-score) so we don't repeat the LTSF "no real coupling" mistake — use
coupled ODEs and BOOM high-cardinality, not independent channels; (b) treat
look-back/context as a reported variable, not a silent constant; (c) audit our
own metric code for drop-last-style artifacts.

### 3.6 The datasets everyone (rightly) distrusts, so we don't lean on them
- **Monash** (NeurIPS'21, arXiv:2105.06643): foundational, but GIFT-Eval notes it
  "lacks sufficient diversity… challenging to evaluate zero-shot," and it's a
  documented contamination vector (TimesFM traffic-hourly).
- **LTSF (ETT/Weather/Electricity/Traffic/ILI)**: small (ILI = 966 steps),
  distribution-shifted, weak channel coupling, contamination-prone, no
  probabilistic eval. Use sparingly and only with the caveats above.

**Where we stand:** We barely touch these (good). Keep them only as *optional
familiar reference points*, never as the core, and always with leakage flags.

---

## 4. Production-grade engineering gaps (from the TSBench-Forge audit)

Concrete, file-level. Status from the code review:

| Area | Current state | Gap → Action |
|---|---|---|
| **Live feeds** | scraped live catalog wired via `scraped_source.ScrapedLiveSource` (`src/sources/` + `feeds.py` as-of/dedup) | **Landed.** Remaining: point `AsOfLiveSource` / `HttpCsvLiveSource` at real vendor-timestamped endpoints so `t_now` binds on live vintages (§3.1). |
| **External validation** | held-out real series disjoint from the feed (`independent_eval.py`) | Wire to GIFT-Eval/TIME as the external negative-control; expand to a real curated held-out suite. |
| **Task geometry** | fixed `HORIZON=48`, `CONTEXT_LEN=256` (`config.py`) | Introduce a **task grid**: term-scaled horizons (1×/10×/15× seasonal), variable context as a reported axis. |
| **Multivariate** | challenges are 1-D | Adopt `MaskedTimeseries`; add coupled-ODE + BOOM-style multivariate w/ verified cross-channel dependence. |
| **Metrics** | MASE/WQL/CRPS, Gaussian lift; PCE/WIS/coverage + robust aggregation in `evaluate.py`; D_stsp/D_H/VPT in `dsr_metrics.py` | Reuse GluonTS `evaluate_model`; make seasonal-naive normalization the default everywhere; wire shifted-gmean + rank aggregation and a zero-inflated bucket into the reported leaderboard. |
| **Anchor quality** | numpy classical `strong` anchor; on raw real data it is often beaten by `drift`/`ewma`, so `validate_panel` returns `valid=False` (by design) | Promote an independently-validated zero-shot TSFM (`independent_eval.resolve_anchor` / `default_panel(strong_model=…)`) so the validity gate holds on real data. |
| **Recipe / catalog breadth** | equal-weight sampling across domain × dgp_class × cadence + cross-epoch dedup (`scraped_source.py`, `feeds.py`) | Grow the `src/sources/` catalog; keep the DGP-class/cadence breadth gates green as it grows (see `docs/REWARD_HACKING.md`). |
| **Reproducibility env** | single-process numpy | Add TempusBench-style isolated per-model envs/subprocess for heterogeneous real TSFMs (Chronos/TimesFM/Moirai/Toto) on top of `sandbox.py`. |
| **Consensus** | single validator | Out of scope for the academic artifact; document as subnet-deployment layer. |
| **Coverage objective** | `foundational_fitness = fitness × coverage_gate × parrot_gate × dgp_class_breadth_gate × cadence_breadth_gate` (`score.py`) | Report `foundational_fitness` (not bare `fitness`) as the default pool-quality yardstick once the task grid lands. |

---

## 5. Roadmap (prioritized)

### P0 — credibility floor (do these before any public/academic claim)
1. **Seasonal-naive-normalized MASE + CRPS, rank-aggregated**, GluonTS-backed, with
   naive/seasonal-naive/**context-parroting**/AutoARIMA/ETS/Theta as permanent
   baseline rows; require models to clear **both** floor baselines. *(GIFT-Eval/BOOM
   parity + the parroting blade, §3.2b.)*
2. **Leakage story made real**: point the as-of live feed at real
   vendor-timestamped endpoints so the `t_now` barrier (`leakage_audit.global_t_now`)
   binds on live vintages; document the contamination-resistance argument.
3. **Multiple seeds + significance** for model evaluation (mean±std, Friedman+post-hoc); rolling origins for the sample.
4. **Submission metadata + PR-based public leaderboard** (model_type, testdata_leakage, repro-code), HF Space.
5. Audit metrics for drop-last/aggregation artifacts; adopt **shifted-gmean** + **zero-inflated bucket**.

### P1 — the scientific differentiators (the paper's contribution)
6. **Dynamical-systems hard tier**: the `dsr_eval/` runner + `dsr_metrics.py` already implement **D_stsp, D_H, VPT** and carry reference λ / Lyapunov times (`dsr_eval/systems.py`); extend with a regime/parameter-generalization difficulty ladder and fold the tier into the reported leaderboard.
7. **Capability taxonomy + per-capability leaderboard**: tag every challenge; compute TIME-style STL/FFT features on the real series and stratify. Report per-axis, not one scalar.
8. **Calibration as a first-class axis**: PCE + WIS + coverage are in `evaluate.py`; surface them per-axis on the leaderboard. *(ICLR'26-topical, cheap, differentiating.)*
9. **Grow the live catalog**: expand `src/sources/` breadth (domains × DGP classes × cadences) while keeping the breadth gates green; the scraper + `ScrapedLiveSource` feed path is already the production generator.

### P2 — breadth & polish
10. **Multivariate with verified cross-channel dependence** (coupled ODEs + BOOM high-cardinality; Granger-check it). `MaskedTimeseries` payload.
11. **Isolated per-model execution envs** for real heterogeneous TSFMs; reference notebooks (Chronos/TimesFM/Moirai/Toto).
12. **Catalog curation toward weak capabilities**: prioritise adding real sources that stress capabilities where the field is weak (the live analogue of horizon-forge's "single-shot" gap).
13. **Governance**: documented protocol, model-type taxonomy, dataset licensing manifest, contribution guide (GIFT-Eval style).

---

## 6. Positioning for the paper

The one-line claim: **"A *live, contamination-resistant* TSFM benchmark that
enforces validity and scores genuine temporal understanding — including long-term
dynamical invariants — rather than memorization or short-horizon curve-fitting."**

Three defensible contributions, each grounded in the gaps above:
1. **Validity-enforced-by-construction** (real-data freshness / as-of gating +
   commit-reveal seeding + multiplicative validity/parrot/coverage/breadth gates) —
   answers the cherry-picking (Roque'25) and contamination (Meyer'25) crises
   *structurally*, not by patching.
2. **Dynamics-aware hard tier** with invariant metrics (D_stsp/D_H/VPT, Lyapunov-
   normalized) — answers "does it understand time?" beyond MASE (DynaMix /
   Durstewitz'26 critique that pointwise metrics are meaningless past a horizon),
   and built so **context parroting cannot win it** (arXiv:2505.11349) — the
   sharpest evidence that a high score reflects understanding, not copying.
3. **Capability-resolved, calibration-aware, leakage-audited reporting** — meets
   the GIFT-Eval/BOOM/TIME credibility bar and the 2026 calibration agenda.

Ship it next to GIFT-Eval/TIME as the *external, refreshing, hard* complement —
not a competitor to the static suites but the live, freshness-enforced stress-test
they lack.

---

---

## 7. Implementation status (this branch)

What has been built and tested in-repo (numpy-only, all green via `pytest`), and
what still needs network / model-weights / a decision and so is scaffolded only.

### Delivered (code + tests)
- **P1 — context-parroting floor** (`baselines.py`, `score.parrot_gate`,
  `evaluate.clears_floor` / `FLOOR_BASELINES`, `tests/test_baselines.py`). Parrot
  is a leaderboard rung and an independent report-only validity gate folded into
  `foundational_fitness`; submissions must beat seasonal-naive **and** parrot.
- **P1 — calibration + robust aggregation + significance** (`evaluate.py`:
  `pce`, `coverage_80`, `wis`; `shifted_gmean`; `normalized_leaderboard`;
  `evaluate_multiseed`; `friedman_test`; `tests/test_metrics_robust.py`).
- **Pure-live challenge assembly + breadth defenses** (`challenges.build_live_challenges`,
  `scraped_source.ScrapedLiveSource`, `score.foundational_fitness` folding in
  coverage / parrot / DGP-class / cadence breadth gates, `score.validate_generalization`;
  `tests/test_reward_hacking.py`, `tests/test_scraped_source.py`, `tests/test_coverage.py`).
  The benchmark forecasts real data only — there is no synthetic DGP to reverse-engineer.
- **P3 — DSR dynamics hard tier** (`dsr_metrics.py`: `d_stsp`, `d_h`,
  `valid_prediction_time`, `max_lyapunov`, `free_run`, `dsr_report`;
  `tests/test_dsr_metrics.py`).
- **P0 — contamination-resistance defaults + audit** (`leakage_audit.py`:
  `default_fresh_buffer` [dedup + as-of], `global_t_now` / `assert_post_cutoff`,
  `memorization_probe`, `feed_novelty`; `tests/test_leakage_audit.py`). The
  as-of/dedup machinery in `feeds.py` is now the documented default path.

### Scaffolded — needs resources/decisions not available in-container
- **Real vendor as-of feed**: the `src/sources/` catalog + scraper +
  `ScrapedLiveSource` are consolidated and wired through `HttpCsvLiveSource` /
  `AsOfLiveSource` / `default_fresh_buffer`; pointing at live *timestamped* vendor
  endpoints (so the `t_now` barrier binds on real vintages) is the remaining
  data-eng task (needs network + credentials).
- **Real TSFM anchor**: `tsfm_adapters.load_tsfm` + `default_panel(strong_model=)`
  accept a zero-shot Chronos/TimesFM/Toto unchanged; running weights and
  validating on external GIFT-Eval/TIME needs torch + checkpoints.
- **Task grid** (term-scaled horizons), **multivariate** (`MaskedTimeseries` +
  Granger-verified coupling), **isolated per-model envs**, and **CI** remain on
  the roadmap above; the metrics/baselines now in place are horizon-parameterised
  (`context_parrot_for`, `valid_prediction_time(..., lyapunov_time=)`) to make the
  grid a smaller step.

---

*Sources: repo audits of `/TSBench-Forge`, `/horizon-forge`, `/gift-eval`,
`/TIME`, `/TempusBench`, `/DynaMix-python`, `/toto` (BOOM); literature —
arXiv:2410.10393 (GIFT-Eval), 2505.14766 (Toto/BOOM), 2602.12147 (TIME),
2205.13504 (DLinear/LTSF critique), 2510.13654 (TSFM leakage), 2510.16060 (TSFM
calibration, ICLR'26), 2203.10716 (forecast-eval best practices), 2403.20150
(TFB / Drop-Last), 2502.09683 (dataset-simplicity bias), 2105.06643 (Monash),
2403.07815 (Chronos/KernelSynth), 2409.15771 (zero-shot chaos / dysts),
2505.11349 (context parroting), 2501.02945 (TabPFN-TS, synthetic-prior #1 on
GIFT-Eval), 2602.16864 (dynamical-systems critique), 2605.26161 (TSFMAudit),
AAAI'25 cherry-picking (Roque et al.).*
