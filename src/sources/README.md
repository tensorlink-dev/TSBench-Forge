# Timeframe benchmark — verified live data sources

Curated catalog of **daily-or-faster updating** time-series data sources for a foundation-model benchmark. Every entry is verified by a real HTTP call.

## Layout

```
sources/
├── sources.yaml         # 55 verified entries with full schema
├── candidates.md        # initial brainstorm (pre-verification)
├── verification_log.md  # every URL attempted, kept|dropped, reason
├── coverage_matrix.md   # domain × archetype, domain × frequency, gaps
├── samples/             # ~100-row trimmed payload per source
├── data/                # scraper output (parquet per source per UTC date)
├── scraper.py           # config-driven unified scraper
├── cron.yaml            # poll cadence per source
└── README.md            # this file
```

## Quick start

```bash
# Install dependencies
pip install httpx pyarrow pyyaml pandas

# List all sources with their cadence
python scraper.py --list

# Run a single source
python scraper.py --id usgs_earthquakes_realtime

# Run an entire domain
python scraper.py --domain energy

# Run every source (smoke test)
python scraper.py --all

# Dry-run (fetch + parse, no parquet write)
python scraper.py --id binance_btcusdt_1m --dry-run
```

## Hard constraints (every kept source meets all)

1. **Cadence ≥ daily**, every day (markets/exchanges may have weekend gaps; flagged in `notes`).
2. **Programmatic, free, public** endpoint (no scraping; API keys flagged via `auth: env:VAR_NAME`).
3. **≥ 1 year of history** available.
4. **Stable timestamped schema**.
5. **Research-permissive license**.
6. **HTTP 200 verified** with parseable payload and a value within ~48 h of `now` (with documented exceptions for Wikimedia/npm publication lags).

## Coverage targets — all met

| target | required | actual |
|---|---|---|
| total verified                 | ≥40 | 55 |
| sources per GIFT-Eval domain   | ≥3  | 4–15 (all 7 domains) |
| sources per Tempus archetype   | ≥1  | 2–37 (all 10 archetypes) |
| sources per frequency band     | ≥2  | 9–19 (all 4 bands) |
| share `clean` or `partial`     | ≥70% | 91% (50 / 55) |

See `coverage_matrix.md` for the full table.

## Adding a new source

1. Append an entry to `sources.yaml` with the schema documented at the top of the file.
2. Save a verifying sample to `samples/<id>.{json,csv,xml,...}`.
3. Append a row to `verification_log.md`.
4. Run `python scraper.py --id <new_id>` to confirm end-to-end.

For sources that need per-instance iteration (per-package, per-station, per-article) define a `panel:` list of substitutions; each `{TOKEN}` in the URL is filled with the matching key.

## Known caveats

- `nyiso_realtime_lmp_zonal` uses a UTC-derived date in its URL; for the first ~5h of UTC each day the previous-day NY-local file is still the live one.
- `octopus_agile_tariff` product code rolls (currently `AGILE-24-10-01`); discover the latest via `/v1/products/?brand=OCTOPUS_ENERGY`.
- `aemo_nem_5min` filename rolls monthly; scraper.expand_url handles the `{YYYYMM}` token.
- `crates_io_downloads` and `reddit_top_daily` require a real User-Agent; configure via env (`CRATESIO_USER_AGENT`, `REDDIT_USER_AGENT`).
- Scraper dedupe is on the full row; for sources whose payload contains many records sharing one timestamp (multi-zone, multi-station), the parser currently emits one row per distinct (timestamp, value-tuple). Extend per-source schemas with additional `value_field` paths to widen the row.
