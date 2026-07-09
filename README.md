# tsbench-forge

A **live-data benchmark for time-series foundation models (TSFMs)**. It fetches
suites of real time series across domains and frequencies, scores forecasters on
**MASE / WQL / CRPS against a seasonal-naive floor**, and keeps itself expanding
with an autosearch agent that proposes and vets new sources. The repo does three
things:

1. **Fetch and manage live sources** — a curated catalog of real, fast-updating
   public time series, scraped on a cron into dated parquet.
2. **Evaluate TSFMs** — build forecasting challenges from the catalog and rank
   models on probabilistic metrics, normalised against seasonal naive.
3. **Discover new sources** — an autosearch-style LLM agent that maps coverage
   gaps, proposes concrete new sources, and vets them automatically before they
   enter rotation.

```bash
pip install httpx pyarrow pyyaml pandas numpy

python src/sources/scraper.py --list          # every source + cadence
python src/sources/scraper.py --all           # scrape everything once
python src/demo.py                            # full pipeline: catalog -> challenges -> leaderboard
python -m source_discovery --coverage         # map gaps in the catalog
```

## 1. The live catalog (`src/sources/`)

A curated catalog of **daily-or-faster-updating** real public time series —
climate, solar activity, energy load, markets, transport, public health — across
the 7 GIFT-Eval domains. Today: **92 verified feeds**, **36 DGP (archetype)
classes**, sub-minute to yearly cadence. Every entry is verified by a real HTTP
call before it lands in `sources.yaml`.

```
src/sources/
  sources.yaml         92 verified entries with full schema (domain, cadence, DGP class, novelty)
  scraper.py           config-driven unified scraper (one adapter, many sources)
  cron.yaml            poll cadence per source
  data/                scraper output: parquet per source per UTC date (~90 sources scraped)
  samples/             ~100-row trimmed payload per source (for review)
  candidates.md        initial brainstorm (pre-verification)
  verification_log.md  every URL attempted, kept|dropped, and why
  coverage_matrix.md   domain × archetype, domain × frequency, gaps
  DGP_TAXONOMY.md      the archetype/DGP-class taxonomy
```

```bash
python src/sources/scraper.py --id usgs_earthquakes_realtime   # one source
python src/sources/scraper.py --domain energy                  # a whole domain
python src/sources/scraper.py --id binance_btcusdt_1m --dry-run # fetch+parse, no write
```

### Publishing the mirror to a bucket

`scripts/publish_data_bucket.sh` syncs `data/` + `sources.yaml` to a
**private** S3-compatible bucket (AWS, Cloudflare R2, MinIO) so a downstream
consumer — e.g. cascade's held-out eval-pool publisher (`cascade-pool publish
--sources tsbench_forge`) — can mirror the catalog without access to this
host. Append it to the scrape cron:

```bash
0 * * * *  python src/sources/scraper.py --all && TSFORGE_BUCKET=tsforge-raw scripts/publish_data_bucket.sh
```

The dated, append-only parquet layout makes the sync idempotent, and the dated
bucket doubles as an audit trail: any consumer build pinned to an `as_of` can
be re-run bit-for-bit against the same objects.

### Serving and freshness (`src/`)

The snapshots are served through a small ingestion layer with the anti-staleness
discipline a live feed needs:

- `scraped_source.py` — `ScrapedLiveSource`: serves the dated parquet snapshots,
  sampling **equal-weight across domain × DGP-class × cadence** so no
  source-count-heavy domain dominates.
- `ingest.py` — `LiveSource` ABC (domain-tagged), `MixtureLiveSource`, `FreshBuffer`.
- `feeds.py` — production feed discipline: **as-of / vintage gating** (only serve
  motifs timestamped after a commit point) and **cross-epoch dedup** (fingerprint
  and quarantine near-duplicates).
- `leakage_audit.py` — the contamination-resistant default buffer: dedup + a global
  `t_now` barrier, a memorization probe, and a feed-novelty meter.
- `live_feeds.py` / `daily_feeds.py` — curated real public-data adapters (CSV/dated
  and JSON-path), with cached fetches.

## 2. The benchmark (`src/`)

The evaluation half: real series in, a probabilistic leaderboard out.

- `challenges.py` — builds challenge sets from the live catalog: each challenge
  is a real window split into `context` / `truth`, tagged with its source's
  domain / DGP class / cadence / frequency, with light truth-preserving
  augmentation against memorisation. Deterministic per `(pool, seed)`
  (`seed.py`), so replays are byte-identical.
- `evaluate.py` — scores a forecaster on **MASE** (seasonal-naive-scaled point
  error, season length derived per series from its sampling frequency), **WQL**
  and **CRPS** (probabilistic, on the model's own quantiles), plus WIS,
  calibration error (PCE), and interval coverage. `clears_floor` requires a
  model to beat **seasonal naive** and **context parrot**; `normalized_leaderboard`
  is the GIFT-Eval convention — per-dataset scores divided by seasonal naive's,
  aggregated by shifted geometric mean and average rank. `headroom` /
  `benchmark_has_headroom` check the benchmark can actually separate a superior
  model from the classical anchor.
- `score.py` — the classical reference panel (seasonal naive, drift, EWMA, AR(1),
  a backtest-selected `strong` anchor) and the challenge-set validity gates
  (discrimination spread, parrot gate, domain / DGP-class / cadence breadth).
- `baselines.py` — the context-parrot floor (nearest-neighbour copying), the
  skill bar every real model must clear.
- `tsfm_adapters.py` — wraps published zero-shot TSFMs (Chronos / Chronos-Bolt,
  TimesFM) behind the `ProbForecast` contract so they drop straight onto the
  leaderboard: `leaderboard({'chronos': load_tsfm('chronos'), **probabilistic_panel()}, reveal)`.
- `config.py` — context length, horizon, panel seasonality-search periods.
- `demo.py` — the end-to-end run on locally scraped data (offline, no LLM):
  pool → challenges → panel validity → MASE/WQL/CRPS leaderboard → headroom →
  breadth.

```bash
python src/demo.py                  # full pipeline on scraped (or fixture) data
pip install -e ".[chronos]"         # enable the Chronos adapter (torch)
```

## 3. The source-discovery agent (`src/source_discovery/`)

An autosearch-style LLM curation tool that keeps the catalog diverse and
uncontaminated. It maps coverage gaps, an LLM proposes concrete new sources, and
**two deterministic stages vet them automatically** — the model never makes a
decision and never touches any data path:

- `coverage.py` — deterministic gap analysis over `sources.yaml` (domain × cadence
  matrix + ranked gaps). No model, no network.
- `llm.py` — the only non-deterministic step: proposes candidates via the
  OpenRouter chat API (provider-agnostic, stdlib `urllib`, `OPENROUTER_MODEL`).
- `vet.py` — **metadata pre-filters** on each proposal *before* anything is
  fetched: schema, a **contamination denylist** (known pretraining datasets and
  repackagings), duplicate-of-existing, and a novelty sanity check.
- `quality.py` — the **data admission gate** that runs on the actually-fetched
  series: finite/variance/flatline/spike auto-rejects, plus a behavioural
  discrimination filter. No human, no LLM.
- `config.py` — the deterministic knobs: denylist, domains, cadence bands, targets.
- `runner.py` / `__main__.py` / `system_prompt.md` — glue, CLI, and the agent prompt.

A human is in the loop only for **licensing / legal sign-off** on paywalled or
contract sources.

```bash
python -m source_discovery --coverage                       # gaps only (deterministic)
python -m source_discovery --dry-run                        # print the agent prompt, no model call
python -m source_discovery --out src/sources/discovered     # full run (needs OPENROUTER_API_KEY)
python -m source_discovery --vet candidates.json --out ...  # vet a candidate list from elsewhere
python -m source_discovery --assess aemo_nem_5min           # data-quality gate on a scraped source
```

## Obsolete / pending removal

These modules are out of the benchmark's scope and slated for deletion:

- `src/sandbox.py`, `src/static_analysis.py`, `docs/REWARD_HACKING.md` — gating
  of *miner-submitted code* (a subnet concern, not part of scoring a TSFM).
- `src/dsr_metrics.py`, `src/dsr_eval/` — the dynamical-systems-reconstruction
  eval, a separate protocol from the MASE/WQL/CRPS benchmark.
- `notebooks/`, `experiments/`, `docs/PRODUCTION_GRADE_ROADMAP.md` — exploratory
  artifacts.

## Notes

- Python ≥ 3.11. Scraping and serving use `httpx` + `pyarrow` + `pyyaml` +
  `pandas`; the benchmark core is numpy-only (TSFM adapters lazily import torch
  behind extras); the source-discovery LLM step needs `OPENROUTER_API_KEY`.
- The catalog grows as the cron scrapes daily; sources accumulate history over
  time before they have enough depth to serve.
