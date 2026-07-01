# Coverage matrix

Last updated 2026-07-01. **92 sources** in catalog, **80 working** post scraper extensions,
**35 distinct DGP classes** across 7 domains (see `DGP_TAXONOMY.md`).

## 2026-07-01 delta from 2026-05-11 baseline

| metric | 2026-05-11 | 2026-07-01 | delta |
|---|---:|---:|---:|
| catalog entries | 76 | **92** | +16 |
| working sources (single-fetch OK) | — | **80** | — |
| rows per full scrape | ~46k | **222,349** | +383% |
| distinct DGP classes | not tracked | **35** | new axis |
| yearly cadence coverage | 0 | 1 (WHO GHO) | new tier |
| weekly cadence coverage | 0 | 2 working + 3 in `sources-slow.yaml` | new tier |
| scraper endpoint types | rest_json, rest_csv, rss, s3 | + rest_xml, html_table, rest_xlsx | +3 |
| scraper token substitutions | 7 date tokens | + `{H}`, `{H-N}`, `{HH}`, `{YYYY-MM-DD-Nd}`, `{ISO_DATE}`, `{ISO_DATETIME}`, `{ENVVAR}` | +7 |
| scraper SDMX-JSON decoder | none | present (OECD/Eurostat/ECB/IMF ready) | new parser |
| generalization sampler | none | `scraped_source.ScrapedLiveSource` (equal-weight per DGP-class × domain × cadence) | in the live adapter |

### Domain × DGP-class breadth (from 2026-07-01 live scrape)

| domain | working sources | **distinct DGP classes** | class-diversity ratio |
|---|---:|---:|---:|
| nature | 20 | **9** | 0.45 |
| web_cloudops | 8 | **7** | 0.88 |
| econ_fin | 15 | **7** | 0.47 |
| healthcare | 8 | **5** | 0.62 |
| transport | 10 | **5** | 0.50 |
| energy | 10 | 3 | 0.30 |
| sales | 9 | 3 | 0.33 |

`web_cloudops` has the highest class-diversity ratio (7 classes / 8 sources = 0.88 — nearly
one distinct DGP per source). `energy` and `sales` are the lowest at 0.30 / 0.33 — many
sources of the same class (grid_demand, download_counts, attention_pageviews).

### The previous (2026-05-11) coverage tables below are kept for reference

70 / 70 (100%) of (domain × archetype) cells filled. 32 / 77 (42%) of (domain × frequency) cells filled.

> **PENDING (2026-07-01) — eval-pool gap-closers added to `sources.yaml`.** Eleven sources appended,
> then round-1 verification via WebFetch / WebSearch ran 2026-07-01 — see
> `verification_log.md` "Round-1 verification" section for per-entry details. Status now:
>
> | gap closed | source ids | activation status |
> |---|---|---|
> | healthcare floor | `cdc_nssp_ed_visits` (cadence corrected → P1W; **slow-catalog entry in spirit**), `openaq_global_air_quality` (v3 — panel discovery needed at activation) | ready (with operator actions) |
> | transport — daily | `eurocontrol_daily_traffic`, `tsa_checkpoint_daily` | **BLOCKED** on `html_table` scraper extension |
> | transport — hourly NYC | `nyc_mta_subway_hourly` (dataset id swapped `wujg-7c2s` → `5wq4-mkjj` after Round-1) | ready |
> | daily band re-balance | `treasury_daily_debt_to_penny`, `eia_natural_gas_spot_daily` (URL corrected from futures→spot) | ready (EIA needs URL-query-auth scraper fix) |
> | geographic diversity | `bcb_focus_market_expectations`, `jma_japan_forecast` (schema path corrected `timeSeries[0]→[2]`), `banxico_daily_rates`, `lta_singapore_traffic_speeds` | partial — Banxico + LTA need scraper fixes (URL-query-auth + header-auth) |
>
> Headline counts after gap-closers (assuming all unblock and verify):
>   - healthcare 4 → 6
>   - transport 6 → 9 (or 7 if HTML-table extension is deferred)
>   - new daily-band non-software-attention sources: +2
>   - new geographic-diversity sources: +4
>
> **The per-cadence cell tables below have NOT been updated** — re-run build_report after
> each entry passes activation verification and update the totals here.

## Series count after panel expansion

| frequency band | source endpoints | univariate series (after panel) |
|---|---:|---:|
| sub-min / 1-min | 26 | **46** |
| 5–30 min        | 18 | **151** |
| hourly          | 14 | **33** |
| daily           | 18 | **136** |
| **total**       | **76** | **366** |

The 5–30 min band — previously light — is now the densest after expanding panels on `octopus_agile_tariff` (14 UK regions), `metar_us_airports` (50 US airports, new source), `aemo_nem_5min` (5 NEM regions), `usgs_streamflow_iv` (30 US river gauges), `noaa_tides_coops` (25 US tide stations).

## Domain × Archetype cell counts

| domain | count_discrete | zero_inflated_sparse | binary_state | categorical | bounded | non_stationary_regime | noisy_clean_pair | smooth_periodic | chaotic_nonlinear | hierarchical |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| econ_fin     | 3 | 1 | 1 | 1 | 4 | 9 | 2 | 1 | 3 | 5 |
| energy       | 2 | 2 | 1 | 2 | 4 | 5 | 1 | 5 | 3 | 7 |
| healthcare   | 3 | 1 | 1 | 1 | 2 | 3 | 1 | 1 | 1 | 3 |
| nature       | 3 | 4 | 1 | 2 | 3 | 1 | 2 | 6 | 7 | 9 |
| sales        | 7 | 1 | 1 | 1 | 1 | 3 | 1 | 2 | 1 | 9 |
| transport    | 5 | 1 | 3 | 2 | 2 | 1 | 1 | 3 | 1 | 6 |
| web_cloudops | 6 | 1 | 1 | 2 | 1 | 2 | 1 | 3 | 1 | 3 |

**0 unfilled cells** ✓.

### Per-archetype source counts

| archetype | sources |
|---|---:|
| count_discrete | 29 |
| zero_inflated_sparse | 11 |
| binary_state | 9 |
| categorical | 11 |
| bounded | 17 |
| non_stationary_regime | 24 |
| noisy_clean_pair | 9 |
| smooth_periodic | 21 |
| chaotic_nonlinear | 17 |
| hierarchical | 42 |

## Domain × Frequency (per exact ISO-8601 cadence)

| domain | PT30S | PT1M | PT2M30S | PT5M | PT6M | PT10M | PT15M | PT30M | PT1H | PT8H | P1D | row total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| econ_fin     | 1 | 4 | — | 1 | — | 1 | 1 | 1 | 2 | 1 | 3 | **15** |
| energy       | — | — | — | 4 | — | — | 1 | 4 | 1 | — | 1 | **11** |
| healthcare   | — | — | — | — | — | — | — | — | 2 | — | 3 | **5** |
| nature       | — | 6 | 1 | — | 1 | — | 2 | — | 5 | — | 4 | **19** |
| sales        | — | 3 | — | — | — | — | — | — | 1 | — | 6 | **10** |
| transport    | — | 6 | — | — | — | — | — | — | — | — | — | **6** |
| web_cloudops | — | 5 | — | 1 | — | — | — | — | 1 | 1 | 1 | **9** |
| **column**   | **1** | **24** | **1** | **6** | **1** | **1** | **4** | **5** | **12** | **2** | **18** | **75** |

**31 / 77** (domain × cadence) cells filled, up from 27 / 77 before this round.

### Sources at each exact cadence

**PT30S** — `mempool_unconfirmed_count`
**PT1M (22)** — `kalshi_markets_snapshot`, `manifold_markets_snapshot`, `binance_btcusdt_1m`, `usgs_earthquakes_realtime`, `swpc_planetary_k`, `swpc_solar_wind_plasma`, `nws_active_alerts`, `ingv_seismicity`, `hn_top_stories`, `opensky_states_local`, `citibike_station_status`, `bart_realtime_etd`, `tfl_arrivals`, `tfl_bikepoints_status`, `hn_max_item`, `github_events_firehose`, `stackoverflow_questions`, `nws_alerts_count`, `polymarket_new_market_rate`, `swpc_solar_xray_flux`, `hn_ask_stories`, `tfl_line_status`
**PT2M30S** — `sensor_community_pm25`
**PT5M (6)** — `coinbase_btc_5min`, `aemo_nem_5min`, `caiso_oasis_lmp_5min`, `nyiso_realtime_lmp_zonal`, `aws_health_rss`, `bpa_balancing_load_wind`
**PT6M** — `noaa_tides_coops`
**PT10M** — `bitcoin_block_arrivals`
**PT15M (3)** — `elia_solar_15min`, `usgs_streamflow_iv`, `gdacs_natural_disasters`
**PT30M (4)** — `uk_carbon_intensity_actual`, `uk_carbon_intensity_genmix`, `octopus_agile_tariff`, `uk_neso_carbon_historic_30min`
**PT1H (10)** — `polymarket_btc_history`, `coingecko_btc_hourly`, `ndbc_buoy_realtime`, `open_meteo_forecast`, `open_meteo_air_quality`, `open_meteo_marine`, `reddit_top_daily`, `nyc_311_complaints`, `open_meteo_uv_index`, `gharchive_hourly_events`
**PT8H (2)** — `ripestat_announced_prefixes`, `binance_btcusdt_funding_rate`
**P1D (15)** — `defillama_chain_tvl_daily`, `frankfurter_fx_daily`, `treasury_yield_curve_daily`, `mauna_loa_co2_daily`, `usda_snotel_swe`, `wikimedia_top_articles_daily`, `wikimedia_per_article_pageviews`, `npm_downloads_per_pkg`, `pypi_downloads_per_pkg`, `crates_io_downloads`, `itunes_top_podcasts`, `rki_germany_cases`, `rki_germany_hospitalizations`, `wikimedia_health_pageviews`, `neso_stor_notifications`

### Coarse-band totals (kept for reference)

| band | total |
|---|---:|
| sub-min / 1-min       | 24 |
| 5–30 min              | 15 |
| hourly (PT1H, PT8H)   | 12 |
| daily (P1D)           | 15 |

All ≥2 ✓.

## Domain totals (target ≥3)

| domain | count | OK? |
|---|---:|---|
| econ_fin     | 13 | ✓ |
| energy       | 10 | ✓ |
| healthcare   | 4  | ✓ |
| nature       | 17 | ✓ |
| sales        | 9  | ✓ |
| transport    | 6  | ✓ |
| web_cloudops | 7  | ✓ |
| **total**    | **66** | |

## Pretraining novelty distribution

| novelty | count | share |
|---|---:|---:|
| clean        | 47 | 71% |
| partial      | 14 | 21% |
| contaminated | 5  | 8%  |

`clean+partial = 92%` ≥ 70% ✓.

## Newly-added sources (final-7 closing)

| id | domain | archetypes filled |
|---|---|---|
| `tfl_line_status` | transport | binary_state, categorical, hierarchical, zero_inflated_sparse |
| `bpa_balancing_load_wind` | energy | chaotic_nonlinear, smooth_periodic, hierarchical, count_discrete, non_stationary_regime |
| `neso_stor_notifications` | energy | zero_inflated_sparse, count_discrete, categorical, binary_state |
| `gharchive_hourly_events` | web_cloudops | count_discrete, smooth_periodic, noisy_clean_pair (with `github_events_firehose`) |

## Noisy-clean pairs (paired_with relationships)

| noisy | clean | rationale |
|---|---|---|
| `sensor_community_pm25` | `open_meteo_air_quality` | citizen sensor at lat 50.474, 7.616 vs CAMS regulatory model |
| `coinbase_btc_5min` | `binance_btcusdt_1m` | same BTC asset, two exchanges |
| `wikimedia_health_pageviews` | `rki_germany_cases` | Wikipedia "Influenza" attention vs official RKI case counts |
| `opensky_states_local` | `tfl_arrivals` | aircraft over NYC vs ground-truth London transit (different fidelity transit signals) |
| `github_events_firehose` | `gharchive_hourly_events` | live noisy 100-event sample vs hourly aggregate |
| `uk_carbon_intensity_actual` | (self) | actual + forecast columns of the same series form a paired clean/noisy view |

## Fragile sources (operational notes)

- `bart_realtime_etd` — uses BART's documented public test API key.
- `nyiso_realtime_lmp_zonal` — daily file URL uses UTC date; rolls at 00:00 ET.
- `octopus_agile_tariff` — product code rolls (currently `AGILE-24-10-01`).
- `aemo_nem_5min` — filename rolls monthly (`{YYYYMM}` token).
- `frankfurter_fx_daily`, `treasury_yield_curve_daily` — weekend / holiday gaps.
- `crates_io_downloads`, `reddit_top_daily` — require User-Agent header with contact info.
- `opensky_states_local` — anonymous quota 10 req/min.
- `stackoverflow_questions` — anonymous quota 300 req/day.
- `wikimedia_*_pageviews` — typical 24–48 h publication lag.
- `mauna_loa_co2_daily` — 1–2 day lag from NOAA GML.
- `caiso_oasis_lmp_5min` — returns ZIP; needs unzip step (handled in scraper).
- `aws_health_rss` — extreme zero-inflation (can go weeks empty); feature, not bug.
- `mempool_unconfirmed_count` — single-call snapshot only; must be polled.
- `swpc_solar_xray_flux` — sample mixes two energy bands; filter to `0.1-0.8nm`.
- `bpa_balancing_load_wind` — text file with `\t` separator and ~14 header lines.
- `neso_stor_notifications` — extremely sparse — most weeks are empty.
- `gharchive_hourly_events` — each hourly archive is 30–50 MB compressed; downsample server-side.
