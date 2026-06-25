# TSBench-Forge → Production-Grade, Academically-Credible TSFM Benchmark

*A synthesis of what BOOM, GIFT-Eval, TIME, TempusBench, DynaMix and our own
horizon-forge experiment teach us, and a concrete roadmap to make
`tsbench-forge` the hardest, fairest, hardest-to-game TSFM benchmark — and one
that reviewers at ICML/NeurIPS/ICLR will accept.*

---

## 0. TL;DR — where we are and the one-sentence thesis

`tsbench-forge` already owns a thesis nobody else in this list owns: **the
benchmark is an adversary, not a dataset.** The forge loop (`forge_loop.py`),
commit-reveal seeding (`seed.py`), 11 independent anti-gaming layers, and the
two load-bearing validity gates (panel-validity + generator-fitting detection in
`score.py`) make it the only design here that treats "a high score must mean
genuine forecasting skill" as a *property to be enforced*, not hoped for.

Everyone else is a static (or lightly-refreshed) dataset suite. That is our
moat. But a moat is not a paper. To be production-grade **and** academically
acceptable we must bolt onto that adversarial core the three things the static
benchmarks already do well and we currently fake or stub:

1. **Real, diverse, *provably-uncontaminated* data at scale** (we have ~8 DGPs +
   static GitHub CSVs; GIFT-Eval has 144k series / 177M points, BOOM has 2,807
   real observability series, TIME has 50 fresh post-cutoff datasets).
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
  *the* move that makes zero-shot claims defensible. We should ship a
  "forge-pretrain" split that is provably disjoint from what the forge serves.
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
  econ/energy. Our `domains.py` chaotic zoo is the *opposite* extreme; we want
  both ends.
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
  split only** — textbook leakage-safe protocol; our forge is single-origin, we
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
  the natural hardness-normalized horizon. Our `domains.py` already has Lorenz,
  Rössler, Hénon, logistic — annotate each with λ and report VPT.
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
- **76 verified live sources / 366 series with full provenance, as-of gating,
  cross-epoch dedup, frequency-weighted sampling.** This is the real-data backbone
  our current static-CSV `live_feeds.py` is faking. **Promote horizon-forge's
  `sources/` catalog + scraper into tsbench-forge's production feed.**
- Honest gaps it flagged: no difficulty stratification, 1-D only (no
  multivariate), single-shot (no adaptive/feedback generation). We fix these.

---

## 2. The capability taxonomy — what "truly understands time" means, operationally

Reviewers and users both ask "*what is this benchmark actually measuring?*" We
answer with an explicit taxonomy. Every challenge is tagged with the capabilities
it stresses, and the leaderboard reports **per-capability scores**, not one
number. Proposed axes (each maps to generators we already have or can lift):

| Capability | Probes whether the model… | How we test it | Source to lift |
|---|---|---|---|
| **Trend / changepoint** | tracks local, breaking trends (not a global line) | piecewise-linear trend w/ changepoints late in context | `generate.py` (have it) |
| **Seasonality / multi-period** | locks onto multiple seasonal periods, incl. unseen ones | sinusoid sums (12/24/36), TIME's seasonal-strength buckets | `generate.py`, TIME features |
| **Non-stationarity / regime shift** | adapts across distribution shifts & level/variance regimes | `inject_regime_change`, regime-switched AR noise | horizon-forge augs |
| **Long-horizon / long-context** | uses long history & forecasts far without collapse | term grid (1×/10×/15× seasonal), 30× stress | TIME / GIFT term design |
| **Chaotic dynamics (the hard tier)** | reconstructs the *attractor*, not just next steps | dysts/`domains.py` systems, scored by D_stsp / D_H / VPT | DynaMix |
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

**Where we stand:** *This is our strongest hand and we should play it hard.* The
forge's procedural generation + commit-reveal + as-of gating + cross-epoch dedup
is *already* a contamination-resistant design (held-by-construction synthetic, +
post-commit live data). Action items: (a) ship a provably-disjoint forge-pretrain
split; (b) implement an explicit `t_now` temporal barrier in the live feed
(`feeds.py` `AsOfLiveSource` is the hook — make it real, vendor-timestamped); (c)
add a leakage/contamination *audit* (TSFMAudit-style probe) as a reported column;
(d) state the contamination-resistance argument as Theorem-ish prose in the paper.

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

**Where we stand:** The forge already averages fitness over a **seed bank with
common random numbers** (`evaluate_state`, n_seeds=4) — good variance reduction.
Action: extend to *model evaluation*: run each model over N seeds/origins, report
mean±std and a **Friedman + post-hoc** significance test across the leaderboard.
Multiple test windows (rolling origins, §1 TempusBench) give the samples for free.

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

**Where we stand:** The forge's whole point is anti-gaming; cherry-picking is
structurally hard because *we* generate and seed the test set, not the submitter.
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
| **Live feeds** | static GitHub CSV snapshots (`live_feeds.py`) | Replace with real as-of vendor endpoints + horizon-forge's 76-source catalog/scraper; make `feeds.AsOfLiveSource` timestamp-real (enables `t_now`, §3.1). |
| **External validation** | 3 hand-picked held-out series (`independent_eval.py`) | Wire to GIFT-Eval/TIME as the external negative-control; expand to a real curated held-out suite. |
| **Task geometry** | fixed `HORIZON=48`, `CONTEXT_LEN=256` (`config.py`) | Introduce a **task grid**: term-scaled horizons (1×/10×/15× seasonal), variable context as a reported axis. |
| **Multivariate** | challenges are 1-D | Adopt `MaskedTimeseries`; add coupled-ODE + BOOM-style multivariate w/ verified cross-channel dependence. |
| **Metrics** | MASE/WQL/CRPS, Gaussian lift | Add WIS, **PCE/calibration**, D_stsp, D_H, VPT; reuse GluonTS `evaluate_model`; seasonal-naive normalization everywhere; shifted-gmean + rank aggregation; zero-inflated bucket. |
| **Forge proposer** | OpenRouter-only LLM + hand-coded heuristic fallback (`forge_llm.py`, `forge_loop.py`) | Provider-agnostic proposer; keep deterministic fallback; log proposals as part of the auditable recipe. |
| **Recipe/dedup** | per-challenge oracle, no cross-epoch registry | Lift horizon-forge `tracker.py` (canonical hash + SQLite + `sample_novel`) for guaranteed novelty across epochs. |
| **Reproducibility env** | single-process numpy | Add TempusBench-style isolated per-model envs/subprocess for heterogeneous real TSFMs (Chronos/TimesFM/Moirai/Toto) on top of `sandbox.py`. |
| **Consensus** | single validator | Out of scope for the academic artifact; document as subnet-deployment layer. |
| **Coverage objective** | `foundational_fitness = fitness × coverage_gate` exists but isn't default | Make breadth/coverage a default forge objective once the task grid lands. |

---

## 5. Roadmap (prioritized)

### P0 — credibility floor (do these before any public/academic claim)
1. **Seasonal-naive-normalized MASE + CRPS, rank-aggregated**, GluonTS-backed, with
   naive/seasonal-naive/**context-parroting**/AutoARIMA/ETS/Theta as permanent
   baseline rows; require models to clear **both** floor baselines. *(GIFT-Eval/BOOM
   parity + the parroting blade, §3.2b.)*
2. **Leakage story made real**: `t_now` temporal barrier in a real as-of live feed;
   ship a provably-disjoint forge-pretrain split; document the contamination-resistance argument.
3. **Multiple seeds + significance** for model evaluation (mean±std, Friedman+post-hoc); rolling origins for the sample.
4. **Submission metadata + PR-based public leaderboard** (model_type, testdata_leakage, repro-code), HF Space.
5. Audit metrics for drop-last/aggregation artifacts; adopt **shifted-gmean** + **zero-inflated bucket**.

### P1 — the scientific differentiators (the paper's contribution)
6. **Dynamical-systems hard tier**: annotate `domains.py` systems with λ; implement **D_stsp, D_H, VPT** (from DynaMix); regime/parameter-generalization difficulty ladder.
7. **Capability taxonomy + per-capability leaderboard**: tag every challenge; compute TIME-style STL/FFT features on real series and stratify. Report per-axis, not one scalar.
8. **Calibration as a first-class axis**: PCE + WIS + coverage. *(ICLR'26-topical, cheap, differentiating.)*
9. **Consolidate horizon-forge generation**: lift `augmentations.py`, `kernels.py`, `tracker.py`, and the `sources/` catalog into tsbench-forge as the production generator + feed.

### P2 — breadth & polish
10. **Multivariate with verified cross-channel dependence** (coupled ODEs + BOOM high-cardinality; Granger-check it). `MaskedTimeseries` payload.
11. **Isolated per-model execution envs** for real heterogeneous TSFMs; reference notebooks (Chronos/TimesFM/Moirai/Toto).
12. **Adaptive generation feedback loop**: route the forge toward capabilities where the field is weak (closes horizon-forge's "single-shot" gap).
13. **Governance**: documented protocol, model-type taxonomy, dataset licensing manifest, contribution guide (GIFT-Eval style).

---

## 6. Positioning for the paper

The one-line claim: **"The first *adversarial, contamination-resistant* TSFM
benchmark that scores genuine temporal understanding — including long-term
dynamical invariants — rather than memorization or short-horizon curve-fitting."**

Three defensible contributions, each grounded in the gaps above:
1. **Anti-gaming-by-construction** (forge loop + commit-reveal + validity gates) —
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
not a competitor to the static suites but the adversarial stress-test they lack.

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
