# tsbench-forge

A **live-data catalog for time-series foundation models (TSFMs)** and the tooling
that keeps it fresh and growing. The repo does two things:

1. **Fetch and manage live sources** — a curated catalog of real, fast-updating
   public time series, scraped on a cron into dated parquet.
2. **Discover new sources** — an autosearch-style LLM agent that maps coverage
   gaps, proposes concrete new sources, and vets them automatically before they
   enter rotation.

Everything else the repo has grown (scoring panels, challenge generation, model
evaluation, DSR metrics, the sandbox) is **obsolete** — see
[Obsolete / pending removal](#obsolete--pending-removal).

```bash
pip install httpx pyarrow pyyaml pandas

python src/sources/scraper.py --list          # every source + cadence
python src/sources/scraper.py --all           # scrape everything once
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

## 2. The source-discovery agent (`src/source_discovery/`)

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

The repo was previously a full TSFM *benchmark*; that scope is retired. These
modules are **obsolete** and slated for deletion — they are not part of fetching
or discovering sources:

`src/score.py`, `src/baselines.py`, `src/challenges.py`, `src/seed.py`,
`src/evaluate.py`, `src/tsfm_adapters.py`, `src/dsr_metrics.py`, `src/dsr_eval/`,
`src/static_analysis.py`, `src/sandbox.py`, `src/demo.py`, `src/config.py`,
`notebooks/`, `experiments/`, `docs/REWARD_HACKING.md`,
`docs/PRODUCTION_GRADE_ROADMAP.md`, and their tests.

> One coupling to untangle first: `source_discovery/quality.py` and `vet.py`
> currently import `score.panel_fitness` for the discrimination filter, so
> `score.py` can't be removed until that check is inlined or dropped.

## Notes

- Python ≥ 3.11. Scraping and serving use `httpx` + `pyarrow` + `pyyaml` +
  `pandas`; the source-discovery LLM step needs `OPENROUTER_API_KEY`.
- The catalog grows as the cron scrapes daily; sources accumulate history over
  time before they have enough depth to serve.
