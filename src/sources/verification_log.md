# Verification log

Append-only. Every source considered with attempted URL, kept|dropped, reason, verified-at.

## econ_fin

### polymarket_btc_history
- attempted: https://clob.polymarket.com/prices-history?market=13915689317269078219168496739008737517740566192006337297676041270492637394586&interval=1d&fidelity=60
- result: kept
- reason: HTTP 200, hourly ticks, latest t = 2026-05-11 04:04 UTC, bounded [0,1].
- verified_at: 2026-05-11T01:23:00Z

### kalshi_markets_snapshot
- attempted: https://api.elections.kalshi.com/trade-api/v2/markets?limit=10&status=open
- result: kept
- reason: HTTP 200, last_price_dollars present per market, real-time on trade.
- verified_at: 2026-05-11T01:23:00Z

### manifold_markets_snapshot
- attempted: https://api.manifold.markets/v0/markets?limit=10
- result: kept
- reason: HTTP 200, lastUpdatedTime ms, probability field bounded.
- verified_at: 2026-05-11T01:23:00Z

### coingecko_btc_hourly
- attempted: https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1&interval=hourly
- result: kept (calibration baseline)
- reason: HTTP 200, latest 2026-05-11 ~01:20 UTC.
- verified_at: 2026-05-11T01:22:00Z

### binance_btcusdt_1m
- attempted: https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=100
- result: kept
- reason: HTTP 200, latest open_time 1778456640000 = 2026-05-11 02:24 UTC.
- verified_at: 2026-05-11T01:23:00Z

### coinbase_btc_5min
- attempted: https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=300
- result: kept
- reason: HTTP 200, latest ts 1778462400 = 2026-05-11 04:00 UTC.
- verified_at: 2026-05-11T01:23:00Z

### bitfinex_tickers
- attempted: https://api-pub.bitfinex.com/v2/tickers?symbols=tBTCUSD,tETHUSD
- result: dropped
- reason: response v2 schema lacks native timestamp on tickers; relying on poll-time is brittle. Coinbase/Binance better.
- verified_at: 2026-05-11T01:23:00Z

### frankfurter_fx_daily
- attempted: https://api.frankfurter.dev/v1/latest?from=USD
- result: kept (calibration baseline)
- reason: HTTP 200, latest date 2026-05-08 (Mon poll, ECB Mon-Fri only — flagged).
- verified_at: 2026-05-11T01:23:00Z

### defillama_chain_tvl_daily
- attempted: https://api.llama.fi/v2/historicalChainTvl/Ethereum
- result: kept
- reason: HTTP 200, latest date 1778457600 = 2026-05-11.
- verified_at: 2026-05-11T01:23:00Z

### treasury_yield_curve_daily
- attempted: https://home.treasury.gov/.../daily-treasury-rates.csv/2026/all
- result: kept (calibration baseline)
- reason: HTTP 200, latest 05/08/2026 (business-day only — flagged).
- verified_at: 2026-05-11T01:23:00Z

### alphavantage_intraday
- attempted: https://www.alphavantage.co/query
- result: not attempted
- reason: Free tier limited to 25 req/day; insufficient for daily polling discipline.
- verified_at: 2026-05-11T11:00:00Z

## energy

### uk_carbon_intensity_actual
- attempted: https://api.carbonintensity.org.uk/intensity
- result: kept
- reason: HTTP 200, current 30-min slot 2026-05-11T00:30Z–01:00Z.
- verified_at: 2026-05-11T01:25:00Z

### uk_carbon_intensity_genmix
- attempted: https://api.carbonintensity.org.uk/generation
- result: kept
- reason: HTTP 200, 9 fuels, current slot 2026-05-11T00:30Z.
- verified_at: 2026-05-11T01:25:00Z

### octopus_agile_tariff
- attempted: https://api.octopus.energy/v1/products/AGILE-24-10-01/electricity-tariffs/E-1R-AGILE-24-10-01-A/standard-unit-rates/
- result: kept
- reason: HTTP 200, 140 half-hour slots, latest valid_to 2026-05-11T22:00Z.
- verified_at: 2026-05-11T01:27:00Z
- note: Initial AGILE-FLEX-22-11-25 product code is retired; use AGILE-24-10-01.

### aemo_nem_5min
- attempted: https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_202605_NSW1.csv
- result: kept
- reason: HTTP 200 (after redirect), 5-min cadence, latest SETTLEMENTDATE 2026/05/11 00:00:00.
- verified_at: 2026-05-11T01:27:00Z

### caiso_oasis_lmp_5min
- attempted: https://oasis.caiso.com/oasisapi/SingleZip?queryname=PRC_INTVL_LMP&...&node=TH_NP15_GEN-APND
- result: kept
- reason: HTTP 200, returns zipped XML with 5-min interval LMP.
- verified_at: 2026-05-11T01:28:00Z

### nyiso_realtime_lmp_zonal
- attempted: http://mis.nyiso.com/public/csv/realtime/20260510realtime_zone.csv
- result: kept
- reason: HTTP 200, daily file with 5-min cadence, 11 zones.
- verified_at: 2026-05-11T01:29:00Z

### elia_solar_15min
- attempted: https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods032/records?limit=20&order_by=datetime%20desc
- result: kept
- reason: HTTP 200, 15-min cadence, latest 2026-05-09T21:45+00:00 (~4h lag; daily-or-faster met).
- verified_at: 2026-05-11T01:28:00Z

### uk_neso_carbon_historic_30min
- attempted: https://api.neso.energy/api/3/action/datastore_search?resource_id=f93d1835-75bc-43e5-84ad-12472b180a98
- result: kept
- reason: HTTP 200, CKAN datastore, 30-min cadence.
- verified_at: 2026-05-11T01:30:00Z

### entsoe_load
- attempted: https://web-api.tp.entsoe.eu/api?securityToken=...
- result: dropped
- reason: HTTP 401 — requires registration token. Free but blocks automatic verification.
- verified_at: 2026-05-11T01:27:00Z

### eia_grid_demand
- attempted: https://api.eia.gov/v2/electricity/rto/region-data/data/?...&api_key=
- result: dropped
- reason: HTTP 403 / API_KEY_MISSING. Free key but blocks automatic verification.
- verified_at: 2026-05-11T01:27:00Z

### nordpool_day_ahead
- attempted: https://www.nordpoolgroup.com/api/marketdata/page/10
- result: dropped
- reason: HTTP 410 Gone — endpoint deprecated; current Nord Pool API gated.
- verified_at: 2026-05-11T01:27:00Z

### ercot_real_time
- attempted: https://www.ercot.com/content/cdr/html/real_time_lmp.html
- result: dropped
- reason: HTTP 403 — anti-bot block.
- verified_at: 2026-05-11T01:27:00Z

### elexon_imbalance
- attempted: https://data.elexon.co.uk/bmrs/api/v1/balancing/dynamic/all
- result: dropped
- reason: HTTP 404 on guessed paths; correct endpoint structure not derivable from a single discovery pass; deferred.
- verified_at: 2026-05-11T01:28:00Z

### ree_es / red electrica
- attempted: https://apidatos.ree.es/en/datos/balance/balance-electrico
- result: dropped
- reason: HTTP 400 internal error.
- verified_at: 2026-05-11T01:28:00Z

### miso / pjm
- attempted: https://api.misoenergy.org/... ; https://api.pjm.com/...
- result: dropped
- reason: empty body / 401 — production access requires onboarding.
- verified_at: 2026-05-11T01:28:00Z

## nature

### usgs_earthquakes_realtime
- attempted: https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson
- result: kept
- reason: HTTP 200, 590 features, latest event 1778462493780 ms = 2026-05-11 04:01 UTC.
- verified_at: 2026-05-11T01:25:00Z

### usgs_streamflow_iv
- attempted: https://waterservices.usgs.gov/nwis/iv/?format=json&sites=01646500&parameterCd=00060&period=PT2H
- result: kept
- reason: HTTP 200, 17 readings, latest 2026-05-10T20:50 EST = 2026-05-11 00:50 UTC.
- verified_at: 2026-05-11T01:25:00Z

### noaa_tides_coops
- attempted: https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?date=today&station=8454000&product=water_level&...
- result: kept
- reason: HTTP 200, 14 readings, latest 2026-05-11 01:18.
- verified_at: 2026-05-11T01:25:00Z

### ndbc_buoy_realtime
- attempted: https://www.ndbc.noaa.gov/data/realtime2/41008.txt
- result: kept
- reason: HTTP 200, latest row 2026 05 11 01:00.
- verified_at: 2026-05-11T01:25:00Z

### swpc_planetary_k
- attempted: https://services.swpc.noaa.gov/json/planetary_k_index_1m.json
- result: kept
- reason: HTTP 200, 358 rows, latest 2026-05-11T01:21.
- verified_at: 2026-05-11T01:25:00Z

### swpc_solar_wind_plasma
- attempted: https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json
- result: kept
- reason: HTTP 200, 1390 rows, latest 2026-05-11 01:22.
- verified_at: 2026-05-11T01:25:00Z

### open_meteo_forecast
- attempted: https://api.open-meteo.com/v1/forecast?...
- result: kept
- reason: HTTP 200, 72 hourly points covering 2026-05-09..11.
- verified_at: 2026-05-11T01:36:00Z

### open_meteo_air_quality
- attempted: https://air-quality-api.open-meteo.com/v1/air-quality?latitude=50.474&longitude=7.616&hourly=pm2_5,pm10,carbon_monoxide&...
- result: kept (paired with sensor_community_pm25)
- reason: HTTP 200, hourly PM2.5/PM10/CO from CAMS at sensor.community sensor 27357 lat/lon.
- verified_at: 2026-05-11T01:36:00Z

### open_meteo_marine
- attempted: https://marine-api.open-meteo.com/v1/marine?latitude=54.5&longitude=7.5&hourly=wave_height...
- result: kept
- reason: HTTP 200, hourly wave_height/wave_period.
- verified_at: 2026-05-11T01:36:00Z

### sensor_community_pm25
- attempted: https://data.sensor.community/airrohr/v1/sensor/27357/
- result: kept
- reason: HTTP 200, latest timestamp 2026-05-11 01:35 UTC; pairs with open_meteo_air_quality at same lat/lon.
- verified_at: 2026-05-11T01:25:00Z

### nasa_power_daily
- attempted: https://power.larc.nasa.gov/api/temporal/daily/point?...&start=20260401&end=20260510
- result: dropped
- reason: HTTP 200 but 3-day publication lag — last non-missing value 2026-05-07 (~96h old > 48h freshness rule).
- verified_at: 2026-05-11T01:25:00Z

### mauna_loa_co2_daily
- attempted: https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_trend_gl.txt
- result: kept (calibration baseline)
- reason: HTTP 200, latest 2026 05 09 (50h old; daily file with normal 1-2 day publication lag).
- verified_at: 2026-05-11T01:25:00Z

### nws_active_alerts
- attempted: https://api.weather.gov/alerts/active
- result: kept
- reason: HTTP 200, 358 alerts, updated 2026-05-11T01:24:45+00:00.
- verified_at: 2026-05-11T01:25:00Z

### ingv_seismicity
- attempted: https://webservices.ingv.it/fdsnws/event/1/query?starttime=2026-05-09&minmag=2.0&format=text
- result: kept
- reason: HTTP 200, latest event 2026-05-10T20:29 UTC.
- verified_at: 2026-05-11T01:25:00Z

### usda_snotel_swe
- attempted: https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data?stationTriplets=908:WA:SNTL&elements=WTEQ&duration=DAILY&beginDate=2026-04-01&endDate=2026-05-10
- result: kept
- reason: HTTP 200, daily, last value 2026-05-10.
- verified_at: 2026-05-11T01:25:00Z

### gdacs_natural_disasters
- attempted: https://www.gdacs.org/xml/rss.xml
- result: kept
- reason: HTTP 200, RSS, "Start 5/11/2026 3:25:01 AM".
- verified_at: 2026-05-11T01:25:00Z

### inmet_brazil
- attempted: https://apitempo.inmet.gov.br/estacao/2026-05-09/2026-05-10/A001
- result: dropped
- reason: TLS connection reset (curl exit 56).
- verified_at: 2026-05-11T01:25:00Z

### openaq_v3
- attempted: https://api.openaq.org/v3/locations?...
- result: dropped
- reason: HTTP 401 — requires X-API-Key. (OpenAQ S3 archive separately verified as public but lags 4-5 days, fails freshness.)
- verified_at: 2026-05-11T01:25:00Z

### airnow_aqi
- attempted: https://www.airnowapi.org/aq/observation/zipCode/current/?...
- result: dropped
- reason: HTTP 401 — requires API_KEY.
- verified_at: 2026-05-11T01:25:00Z

## sales / web_cloudops / transport

### wikimedia_top_articles_daily
- attempted: https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/2026/05/09
- result: kept
- reason: HTTP 200, 1000 articles, day 2026/05/09 (~50h lag, normal Wikimedia cadence).
- verified_at: 2026-05-11T01:29:00Z

### wikimedia_per_article_pageviews
- attempted: https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/Cleveland_Clinic/daily/20260401/20260510
- result: kept
- reason: HTTP 200, 39 days, latest 2026-05-09.
- verified_at: 2026-05-11T01:29:00Z

### hn_top_stories
- attempted: https://hacker-news.firebaseio.com/v0/topstories.json
- result: kept
- reason: HTTP 200, 500 ranked ids.
- verified_at: 2026-05-11T01:29:00Z

### hn_max_item
- attempted: https://hacker-news.firebaseio.com/v0/maxitem.json
- result: kept
- reason: HTTP 200, monotonic int (48090055 on poll); diff = arrivals/min.
- verified_at: 2026-05-11T01:29:00Z

### npm_downloads_per_pkg
- attempted: https://api.npmjs.org/downloads/range/last-month/lodash
- result: kept
- reason: HTTP 200, 30-day daily series, latest 2026-05-08 (npm has 2-3 day publication lag).
- verified_at: 2026-05-11T01:30:00Z

### pypi_downloads_per_pkg
- attempted: https://pypistats.org/api/packages/numpy/overall?mirrors=true
- result: kept
- reason: HTTP 200, 181 days, latest 2026-05-10.
- verified_at: 2026-05-11T01:30:00Z

### crates_io_downloads
- attempted: https://crates.io/api/v1/crates/serde/downloads (with User-Agent)
- result: kept
- reason: HTTP 200 after setting UA, latest 2026-05-11.
- verified_at: 2026-05-11T01:30:00Z

### itunes_top_podcasts
- attempted: https://rss.applemarketingtools.com/api/v2/us/podcasts/top/100/podcasts.json
- result: kept
- reason: HTTP 200, feed.updated = Mon 2026-05-11 01:30:00 +0000.
- verified_at: 2026-05-11T01:30:00Z

### reddit_top_daily
- attempted: https://www.reddit.com/r/all/top.json?t=day&limit=25 (with UA)
- result: kept
- reason: HTTP 200, 25 posts, first created 2026-05-10.
- verified_at: 2026-05-11T01:30:00Z

### spotify_charts
- attempted: https://spotifycharts.com/regional/global/daily/2026-05-09/download
- result: dropped
- reason: empty body — endpoint discontinued; charts.spotify.com requires login.
- verified_at: 2026-05-11T01:30:00Z

### opensky_states_local
- attempted: https://opensky-network.org/api/states/all?lamin=40.5&lomin=-74.5&lamax=41.0&lomax=-73.5
- result: kept
- reason: HTTP 200, time 1778462939 = 2026-05-11 04:09 UTC, 84 aircraft.
- verified_at: 2026-05-11T01:29:00Z

### citibike_station_status
- attempted: https://gbfs.citibikenyc.com/gbfs/en/station_status.json
- result: kept
- reason: HTTP 200, 2406 stations, last_updated 2026-05-11 04:08 UTC.
- verified_at: 2026-05-11T01:29:00Z

### bart_realtime_etd
- attempted: https://api.bart.gov/api/etd.aspx?cmd=etd&orig=ALL&key=MW9S-E7SL-26DU-VV8V&json=y
- result: kept
- reason: HTTP 200, time 06:28:48 PM PDT 2026-05-10 = 2026-05-11 01:28 UTC.
- verified_at: 2026-05-11T01:29:00Z

### tfl_arrivals
- attempted: https://api.tfl.gov.uk/Line/victoria/Arrivals
- result: kept
- reason: HTTP 200, real-time predictions, timestamp 2026-05-11T01:28:17Z.
- verified_at: 2026-05-11T01:29:00Z

### tfl_bikepoints_status
- attempted: https://api.tfl.gov.uk/BikePoint/
- result: kept
- reason: HTTP 200, 798 BikePoints, snapshot poll.
- verified_at: 2026-05-11T01:36:00Z

### github_events_firehose
- attempted: https://api.github.com/events
- result: kept
- reason: HTTP 200, latest created_at 2026-05-11T01:24:07Z.
- verified_at: 2026-05-11T01:29:00Z

### stackoverflow_questions
- attempted: https://api.stackexchange.com/2.3/questions?order=desc&sort=creation&site=stackoverflow&pagesize=100
- result: kept
- reason: HTTP 200, latest creation_date 1778462565.
- verified_at: 2026-05-11T01:29:00Z

### aws_health_rss
- attempted: https://status.aws.amazon.com/rss/all.rss
- result: kept
- reason: HTTP 200, lastBuildDate Sun 10 May 2026 18:29:19 PDT.
- verified_at: 2026-05-11T01:30:00Z

### nws_alerts_count
- attempted: https://api.weather.gov/alerts/active/count
- result: kept
- reason: HTTP 200, total: 360 with regional/area breakdowns.
- verified_at: 2026-05-11T01:29:00Z

### ripestat_announced_prefixes
- attempted: https://stat.ripe.net/data/announced-prefixes/data.json?resource=AS3333
- result: kept
- reason: HTTP 200, latest available 2026-05-10 16:00 UTC.
- verified_at: 2026-05-11T01:30:00Z

### cloudflare_radar_*
- attempted: https://api.cloudflare.com/client/v4/radar/...
- result: dropped
- reason: HTTP 400 / 9106 missing auth headers — requires Cloudflare API token.
- verified_at: 2026-05-11T01:30:00Z

### dns_stats_icann
- attempted: https://stats.dns.icann.org/hedgehog/cgi-bin/hedgehog.cgi
- result: dropped
- reason: HTML dashboard, not API.
- verified_at: 2026-05-11T01:30:00Z

## healthcare

### rki_germany_cases
- attempted: https://api.corona-zahlen.org/germany/history/cases/30
- result: kept
- reason: HTTP 200, daily, latest 2026-05-09.
- verified_at: 2026-05-11T01:33:00Z

### rki_germany_hospitalizations
- attempted: https://api.corona-zahlen.org/germany/history/hospitalization/30
- result: kept
- reason: HTTP 200, latest 2026-05-08, 7-day rolling rate.
- verified_at: 2026-05-11T01:33:00Z

### wikimedia_health_pageviews (cluster)
- attempted: per-article (Influenza, Cancer, Cleveland_Clinic) wikimedia REST API
- result: kept
- reason: HTTP 200 for each tested article, latest 2026-05-09.
- verified_at: 2026-05-11T01:34:00Z

### nyc_311_complaints
- attempted: https://data.cityofnewyork.us/resource/erm2-nwe9.json?$limit=100&$order=created_date DESC
- result: kept
- reason: HTTP 200, latest created_date 2026-05-09T02:32:59. Filter to health complaint types for healthcare-only panel.
- verified_at: 2026-05-11T01:33:00Z

### delphi_covidcast
- attempted: https://api.delphi.cmu.edu/epidata/covidcast/?data_source=hhs&signals=confirmed_admissions_covid_1d&...
- result: dropped
- reason: HTTP 200 but signals stale (HHS COVID admissions reporting deprecated; current signal returned no results post-2024).
- verified_at: 2026-05-11T01:33:00Z

### cdc_wastewater_nwss
- attempted: https://data.cdc.gov/resource/2ew6-ywp6.json?$order=date_end+desc
- result: dropped
- reason: HTTP 200 but latest date_end 2025-09-07 — dataset stopped updating ~8 months ago.
- verified_at: 2026-05-11T01:33:00Z

### healthdata_hosp
- attempted: https://healthdata.gov/resource/anag-cw7u.json?$order=collection_week+desc
- result: dropped
- reason: HTTP 200 but latest collection_week 2024-04-21 — HHS hospital reporting ended after PHE.
- verified_at: 2026-05-11T01:33:00Z

### openfda_drug_enforcement
- attempted: https://api.fda.gov/drug/enforcement.json?search=report_date:[20260301+TO+20260511]&count=report_date
- result: dropped
- reason: HTTP 200 but meta.last_updated 2026-04-29 — 12-day data refresh lag, fails 48h freshness.
- verified_at: 2026-05-11T01:35:00Z

### disease_sh_covid
- attempted: https://disease.sh/v3/covid-19/historical/usa?lastdays=30
- result: dropped
- reason: HTTP 200 but timeline ends 2/23/23 — JHU upstream stopped 2023.
- verified_at: 2026-05-11T01:33:00Z

### who_covid_global_data
- attempted: https://covid19.who.int/WHO-COVID-19-global-data.csv
- result: dropped
- reason: Latest date_reported 2026-04-19 — WHO COVID is now WEEKLY, not daily.
- verified_at: 2026-05-11T01:33:00Z

### italy_dpc_covid
- attempted: https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-json/dpc-covid19-ita-andamento-nazionale-latest.json
- result: dropped
- reason: HTTP 200 but latest data 2025-01-08 — dataset frozen since Jan 2025.
- verified_at: 2026-05-11T01:33:00Z

### covid_act_now
- attempted: https://api.covidactnow.org/v2/country/US.timeseries.json
- result: dropped
- reason: HTTP 403 + body confirms "API has been permanently shut down".
- verified_at: 2026-05-11T01:35:00Z

### cdc_respiratory_virus_dashboards (multiple resource ids tried)
- attempted: data.cdc.gov resources eb7t-r2ne, c526-7nnc, k4tw-hktw, qag6-hwu5
- result: dropped
- reason: HTTP 404 on each — dataset ids retired/moved; current data.cdc.gov layout for respiratory dashboards is fragmented and not derivable in a single discovery pass.
- verified_at: 2026-05-11T01:35:00Z

### promed_mail_rss
- attempted: https://promedmail.org/promed-posts/feed/
- result: dropped
- reason: HTTP 308 redirect; body unstructured prose, not parseable as time series.
- verified_at: 2026-05-11T01:35:00Z

### uk_hsa_flu_surveillance
- attempted: https://www.gov.uk/government/statistics/national-flu-and-covid-19-surveillance-reports-...
- result: dropped
- reason: HTML page; reports are weekly PDFs, not daily API.
- verified_at: 2026-05-11T01:36:00Z

### ecdc_respiratory
- attempted: https://opendata.ecdc.europa.eu/respiratory/respiratory_dashboards/data
- result: dropped
- reason: HTTP 404; ECDC respiratory dashboard does not currently expose a public CSV endpoint.
- verified_at: 2026-05-11T01:35:00Z

### uk_phe_coronavirus
- attempted: https://api.coronavirus.data.gov.uk/v1/data?...
- result: dropped
- reason: DNS resolution failure — UK Coronavirus Dashboard API decommissioned.
- verified_at: 2026-05-11T01:33:00Z

## Gap-filler additions (verified 2026-05-11)

### mempool_unconfirmed_count
- attempted: https://mempool.space/api/mempool
- result: kept
- reason: HTTP 200, count=90364 unconfirmed txs (snapshot — polled 8× over 4 min for series).
- verified_at: 2026-05-11T03:03:00Z

### binance_btcusdt_funding_rate
- attempted: https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1000
- result: kept
- reason: HTTP 200, 1000 historical 8-hour rates with markPrice.
- verified_at: 2026-05-11T03:03:00Z

### bitcoin_block_arrivals
- attempted: https://blockchain.info/blocks/{ts_ms}?format=json
- result: kept
- reason: HTTP 200, 163 blocks in past 24h with timestamps.
- verified_at: 2026-05-11T03:03:00Z

### polymarket_new_market_rate
- attempted: https://gamma-api.polymarket.com/markets?limit=200&order=createdAt&ascending=false
- result: kept
- reason: HTTP 200, 200 markets with createdAt timestamps spanning recent ~30 minutes.
- verified_at: 2026-05-11T03:03:00Z

### swpc_solar_xray_flux
- attempted: https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json
- result: kept
- reason: HTTP 200, 2876 1-min readings (two energy bands; we filter 0.1-0.8nm).
- verified_at: 2026-05-11T03:03:00Z

### open_meteo_uv_index
- attempted: https://api.open-meteo.com/v1/forecast?latitude=40.71&longitude=-74.0&hourly=uv_index&past_days=14&forecast_days=2
- result: kept
- reason: HTTP 200, 384 hourly UV-index values, bounded 0-12, clean diurnal cycle.
- verified_at: 2026-05-11T03:03:00Z

### hn_ask_stories
- attempted: https://hacker-news.firebaseio.com/v0/askstories.json
- result: kept
- reason: HTTP 200, 200 Ask-HN ids; for-each fetched item.time/score gives a timeseries.
- verified_at: 2026-05-11T03:04:00Z

### eaglei_oregonstate
- attempted: https://eaglei.geo.oregonstate.edu/api/v1/outage_data
- result: dropped
- reason: DNS resolution failure (curl exit 6).
- verified_at: 2026-05-11T03:03:00Z

### caiso_systemstatus
- attempted: https://www.caiso.com/api/getSystemConditions
- result: dropped
- reason: HTTP 404; current CAISO Flex Alert URL not derivable.
- verified_at: 2026-05-11T03:03:00Z

### who_disease_outbreak_news
- attempted: https://www.who.int/feeds/entity/csr/don/en/rss.xml ; https://www.emro.who.int/...feed
- result: dropped
- reason: HTTP 404 — both legacy WHO RSS endpoints returned 404 / View-not-found.
- verified_at: 2026-05-11T03:03:00Z

### usda_aphis_hpai
- attempted: https://www.aphis.usda.gov/livestock-poultry-disease/avian/avian-influenza/hpai-detections/...
- result: dropped
- reason: HTML page, no public JSON/CSV API; would require scraping.
- verified_at: 2026-05-11T03:03:00Z

### cdc_hpai_dairy_cattle
- attempted: https://data.cdc.gov/resource/wjku-fcwh.json (Socrata)
- result: dropped
- reason: HTTP 404 — dataset id retired.
- verified_at: 2026-05-11T03:03:00Z

### healthmap_alerts
- attempted: https://www.healthmap.org/feeds/wp.json
- result: dropped
- reason: HTTP 404; HealthMap public feed endpoint moved.
- verified_at: 2026-05-11T03:03:00Z

### cloudflare_radar (no-auth)
- attempted: https://api.cloudflare.com/client/v4/radar/...
- result: dropped (re-confirmed)
- reason: HTTP 400 / 9106 — Radar API needs Cloudflare API token.
- verified_at: 2026-05-11T03:03:00Z

## Final-7 cell closing (verified 2026-05-11)

### tfl_line_status
- attempted: https://api.tfl.gov.uk/Line/Mode/tube/Status
- result: kept
- reason: HTTP 200, 11 lines × statusSeverity (10=Good, lower=disrupted). Polled 6× over 3 min for time-series sample.
- verified_at: 2026-05-11T03:37:00Z

### bpa_balancing_load_wind
- attempted: https://transmission.bpa.gov/business/operations/Wind/baltwg.txt
- result: kept
- reason: HTTP 200, ~2000 5-min rows from BPA SCADA, columns: Load, VER (wind/solar), Hydro, Fossil/Biomass, Nuclear. last update 2026-05-10 20:31 PT (~5h lag).
- verified_at: 2026-05-11T03:37:00Z

### neso_stor_notifications
- attempted: https://api.neso.energy/dataset/.../stor-iin.csv
- result: kept
- reason: HTTP 200, sparse notification log (11 entries from 2024–2025). Most weeks are empty — perfect for zero_inflated_sparse archetype.
- verified_at: 2026-05-11T03:37:00Z

### gharchive_hourly_events
- attempted: https://data.gharchive.org/2026-05-10-12.json.gz
- result: kept
- reason: HTTP 200, 34 MB compressed for one hour. Downsampled to per-minute event counts (60 points, ~2500 events/min). Pairs with github_events_firehose.
- verified_at: 2026-05-11T03:33:00Z

### Candidates dropped during this round
- `mainsfrequency.com` & `netzfrequenzmessung.de` — HTTP 404; live grid frequency dashboards no longer expose JSON
- `aemo.com.au/rss/marketnotices` — Cloudflare bot challenge (HTML JS)
- `aishub.net` (guest API) — returns HTML registration page
- `ntsb.gov` — SharePoint-rendered HTML, no API
- `rte-france.com/api/eco2mix/getDataRealTime` — HTTP 404
- `apidatos.ree.es` — HTTP 400 on demand-evolution path; works only with auth
- `mta-api/all-alerts` — GTFS-RT protobuf binary; would require gtfs-realtime-bindings library
- `mainsfrequency` / `gridradar.net` — both 404
- `oct_wholesale` (Octopus wholesale) — 404
- `neso_freq` (UK system frequency CSV) — works but latest CSV is for September 2025; >7 month lag fails 48h freshness rule

## PENDING — eval-pool gap-closers (added 2026-07-01)

Eleven entries appended to `sources.yaml` to close gaps identified for the cascade eval-pool
use case (see `coverage_matrix.md` PENDING note). URL patterns are taken from each agency's
public API docs but have **not** been HTTP-verified — each row below needs:

  1. one live `python scraper.py --id <id> --dry-run` (or equivalent curl) → confirm HTTP 200
  2. save the response under `samples/<id>.{json,csv}`
  3. confirm `latest_timestamp` falls within freshness window for the source's cadence
  4. confirm parser handles the schema (timestamp_field + value_field) cleanly
  5. flip `last_verified: PENDING` to the real ISO timestamp; remove from this section

### Round-1 verification (2026-07-01 — endpoints checked via WebFetch / WebSearch)

| id | status | finding |
|---|---|---|
| `cdc_nssp_ed_visits` | ✅ dataset id correct — **cadence corrected** | `rdmq-nq56` confirmed at data.cdc.gov ("NSSP ED Visit Trajectories..."). **Publication is weekly Fridays**, not daily. YAML `frequency` updated `P1D → P1W`. Belongs in `sources-slow.yaml` once that exists. |
| `nyc_mta_subway_hourly` | ⚠ dataset id wrong — **swapped** | `wujg-7c2s` is the **2020-2024 historical** dataset, frozen 2025-01-10. Live refreshing dataset for 2025+ is **`5wq4-mkjj`** ("MTA Subway Hourly Ridership: Beginning 2025"), same schema, weekly publication. YAML updated. |
| `eurocontrol_daily_traffic` | ❌ **BLOCKED** | No clean CSV exists; data only via dashboard HTML pages (`eurocontrol.int/Economics/DailyTrafficVariation-States.html`). Filename pattern I guessed does not exist. YAML marked BLOCKED — needs `html_table` scraper extension OR a confirmed programmatic endpoint at ansperformance.eu. |
| `tsa_checkpoint_daily` | ❌ **BLOCKED** | Current URL is `tsa.gov/travel/passenger-volumes` (not the old `/coronavirus/...`). HTML-table only; WebFetch returned HTTP 403 (TSA bot-blocks). YAML marked BLOCKED — same `html_table` extension dependency. Reference impl: `github.com/hunj/tsa-passenger-throughput`. |
| `eia_natural_gas_spot_daily` | ⚠ URL was for futures — **corrected** | My `pri/fut` URL returns RNGC1 (futures) not spot. Corrected to `pri/sum` + `facets[series][]=RNGWHHD` for Henry Hub Spot daily. EIA v2 + DEMO_KEY confirmed functional. |
| `treasury_daily_debt_to_penny` | ✅ verified | endpoint returns JSON, `data[].record_date` + `data[].tot_pub_debt_out_amt` confirmed, no auth, fresh through 2026-06. |
| `bcb_focus_market_expectations` | ✅ verified | Olinda OData endpoint returns JSON, fields `Data` / `Mediana` / `Indicador` confirmed, no auth required. |
| `jma_japan_forecast` | ⚠ schema path wrong — **corrected** | Endpoint + area code 130000 confirmed. JMA forecasts have FOUR `timeSeries` entries: [0] weather/wind/wave, [1] precipitation `pops`, **[2] temperature `temps`**, [3] 7-day outlook. My path pointed at [0] (weather codes) — corrected to [2]. |
| `openaq_global_air_quality` | ⚠ v2 IDs invalid — **panel replaced with discovery sentinel** | v3 auth-required confirmed (HTTP 401 without key). **OpenAQ v1/v2 retired 2025-01-31** — the v2 location IDs I inlined cannot be assumed to work. Panel replaced with `TBD-FROM-V3-DISCOVERY` sentinel; activation requires one-time `GET /v3/locations?parameters_id=2` discovery call. |
| `lta_singapore_traffic_speeds` | ◌ pending | could not fully verify without real LTA_ACCOUNT_KEY; endpoint shape documented at datamall.lta.gov.sg. Blocker is the header-auth scraper extension below. |
| `banxico_daily_rates` | ◌ pending | could not fully verify with `token=demo` (HTTP 400). Path matches Banxico SIE REST docs; needs real BANXICO_TOKEN to confirm response shape. |

Three entries are now BLOCKED on scraper extensions and four are activation-ready
(`treasury_daily_debt_to_penny`, `bcb_focus_market_expectations`, `jma_japan_forecast`,
`cdc_nssp_ed_visits` — the last as a P1W entry pending slow catalog). The remaining four
have known corrections applied; verification call still needed by the operator.

### Scraper extension points needed (confirmed by reading scraper.py)

Read of `scraper.py::_resolve_auth` (lines 133-152) confirmed: it emits **headers only**,
choosing the header name by env-var-name suffix (`*_USER_AGENT` → User-Agent,
`*_API_KEY` → X-API-Key, `*_TOKEN` → Authorization: Bearer, fallback → Bearer).
The new entries hit three real gaps:

1. **URL-query-string auth.** EIA (`?api_key=...`) and Banxico (`?token=...`) put the
   key in the URL, not a header. Currently the literal `{EIA_API_KEY}` / `{BANXICO_TOKEN}`
   in the URL is left un-substituted and the auth header is sent uselessly. (The existing
   `eia_nyis_hourly_demand` entry has the same latent bug — it hard-codes `DEMO_KEY` in the
   URL because the env-var substitution doesn't exist.) **Fix:** extend `_expand_panel` (or
   a new `_expand_env` step) to substitute `{ENVVAR}` placeholders from `os.environ` after
   panel expansion. ~5 LOC.

2. **Non-Bearer-non-X-API-Key header auth.** Singapore LTA DataMall uses the literal header
   name `AccountKey: <value>`. Today, env var `LTA_ACCOUNT_KEY` would hit the
   `*_API_KEY` branch and emit `X-API-Key`, which LTA ignores. **Fix:** extend the
   `auth:` field syntax to allow explicit header naming, e.g. `auth: env:LTA_ACCOUNT_KEY:AccountKey`.
   ~10 LOC.

3. **HTML table fetcher (`endpoint.type: html_table`).** EUROCONTROL and TSA publish only
   via HTML tables. Add a fetcher that calls httpx, parses with BeautifulSoup,
   extracts the table identified by a CSS selector (`endpoint.table_selector`), and
   yields rows the same shape as `_records_from_json`. ~40 LOC + bs4 dep.

4. **OData URL-encoding sanity check** (Socrata `$where`, OData `$filter`). Confirm the
   scraper's URL builder doesn't double-encode `'` or `$`. Looked safe on read but
   needs the real verification fetch to confirm.

5. **DD/MM/YYYY timestamp parsing** (Banxico). The parquet writer needs to parse this
   format and emit UTC. Pandas usually handles it but the schema's `timestamp_field`
   resolution path may need a small parser hook.

Items 1 and 2 unblock the four "URL-query-auth" + "header-auth" entries
(`eia_natural_gas_spot_daily`, `banxico_daily_rates`, `lta_singapore_traffic_speeds`,
and retroactively fixes `eia_nyis_hourly_demand`). Item 3 unblocks the two HTML-only
sources. Total scraper extension work: ~60 LOC + one dep (`beautifulsoup4`).

## 2026-07-01 — fixes for the 7 zero-data sources

The seven sources whose `data/<id>/` directories contained only an `_errors.log` after the
2026-05-11 / 2026-05-15 scrape runs. Per-source resolution:

| source | original error | resolution applied |
|---|---|---|
| `polymarket_btc_history` | `[Errno 113] No route to host` (transient); also pinned market id resolved (June-2026 expiry) | URL changed to `{MARKET_ID}` panel placeholder + operator-action note (enumerate via gamma-api at activation). Time-bounded markets can't be pinned by id long-term. |
| `polymarket_new_market_rate` | `[Errno 113] No route to host` (transient) | **No YAML change.** Re-run on the cron'd host should succeed; endpoint pattern still valid per gamma-api docs. |
| `ndbc_buoy_realtime` | HTTP 404 on `BUOY=44017` | Removed buoy 44017 from panel (NOAA station page confirms recovered 2023-02, never re-deployed). 44025 (Long Island, verified active through 2026-06-30) covers the same region. Panel now 19 buoys. |
| `bitcoin_block_arrivals` | HTTP 404 on `blockchain.info/blocks/{ts_ms}` | Swapped to **Blockstream Esplora** (`blockstream.info/api/blocks`) — confirmed live, returns 10 most-recent blocks with `height` + `timestamp`. blockchain.info migrated to blockchain.com and dropped the `/blocks/{ts_ms}` path. |
| `gharchive_hourly_events` | scraper error: `unresolved placeholders ['H']` | Two-part fix: (a) `scraper.py::expand_url` extended to substitute `{H}` (current UTC hour), `{HH}` (zero-padded), `{H-N}` (hour minus N). (b) URL changed from `{H}` to `{H-1}` because gharchive publishes ~30-60 min after the hour ends — current-hour fetches usually 404. |
| `ingv_seismicity` | `[Errno -3] DNS` (transient); also hardcoded `starttime=2026-05-09` was stale | (a) `scraper.py::expand_url` extended to substitute `{YYYY-MM-DD-Nd}` (today minus N days). (b) URL changed `starttime=2026-05-09T00:00:00` → `starttime={YYYY-MM-DD-30d}T00:00:00` for a rolling 30-day window. |
| `nyc_311_complaints` | `[Errno -3] DNS` (transient) | **No YAML change.** Endpoint verified live (returned current 2026-06-29 records). Re-run on the cron'd host should succeed. |
| `usda_snotel_swe` | `[Errno -3] DNS` (transient); also hardcoded `beginDate=2026-04-01&endDate=2026-05-10` was stale | Used the new `{YYYY-MM-DD-30d}` + `{YYYY-MM-DD}` tokens for a rolling 30-day window. AWDB endpoint verified live + JSON shape (`data[].values[].date|value`) confirmed. |

### Scraper changes that landed

`scraper.py` patched in this round:

- `_OFFSET_DATE_RE = re.compile(r"\{YYYY-MM-DD-(\d+)d\}")` — substitutes `{YYYY-MM-DD-Nd}`
  with today minus N days, formatted YYYY-MM-DD.
- `_OFFSET_HOUR_RE = re.compile(r"\{H-(\d+)\}")` — substitutes `{H-N}` with the UTC
  hour N hours ago, wrapping at midnight.
- Static tokens `{H}` and `{HH}` added for the current UTC hour (unpadded / zero-padded).
- All applied **inside** `expand_url`, before the existing `_TODAY_PATTERNS` loop, so
  offset tokens take precedence over the literal `{H}` / `{YYYY-MM-DD}` substitutions.

### What this means operationally

Of the 7 zero-data sources:
- **5 should produce data on the next scraper run** as soon as a cron is installed
  (`bitcoin_block_arrivals`, `gharchive_hourly_events`, `ingv_seismicity`, `nyc_311_complaints`,
  `polymarket_new_market_rate`, `usda_snotel_swe`, `ndbc_buoy_realtime` — that's 6 actually
  the first time I run the list. **6 of 7.**)
- **1 needs operator action before activation**: `polymarket_btc_history` requires a
  current `{MARKET_ID}` panel populated from gamma-api (one-time, then auto-rolls until
  the chosen markets resolve). This is inherent to time-bounded prediction markets,
  not a fixable YAML defect.

---

## 2026-07-03 — first full live run with real API keys (all-source scrape + fix pass)

Full `--all` scrape: 80/92 sources wrote parquet. The 12 failures triaged below;
all endpoints re-probed live with the real keys loaded.

### eia_natural_gas_spot_daily
- attempted: https://api.eia.gov/v2/natural-gas/pri/fut/data/?frequency=daily&facets[series][]=RNGWHHD (was pri/sum)
- result: kept — URL fixed pri/sum → pri/fut
- reason: pri/sum now rejects frequency=daily (monthly/annual only). RNGWHHD under
  pri/fut is process PS0 "Spot Price", daily, fresh to previous business day.
- verified_at: 2026-07-03T14:40:00Z

### swpc_solar_wind_plasma
- attempted: https://services.swpc.noaa.gov/products/geospace/propagated-solar-wind-1-hour.json
- result: kept — URL replaced
- reason: products/solar-wind/plasma-*.json all 404 now (whole directory gone).
  Geospace propagated product has identical array-of-arrays shape; header row
  [time_tag, speed, density, temperature, ...] keeps schema indices [][1..3] valid.
- verified_at: 2026-07-03T14:45:00Z

### openaq_global_air_quality
- attempted: https://api.openaq.org/v3/sensors/{SENSOR_ID}/measurements?datetime_from={YYYY-MM-DD-14d}&limit=1000
- result: kept — activated with 6-sensor panel (Delhi, London, Mexico City, LA, Seoul, Warsaw)
- reason: v3 has no top-level /measurements (404). /hours rollups empty for many live
  sensors; raw sensor /measurements with a 14-day window verified fresh for all 6.
  Caution: location-level datetimeLast is unreliable — verify sensor-level datetimeLast.
- verified_at: 2026-07-03T15:00:00Z

### ndbc_buoy_realtime (panel)
- attempted: realtime2/{44004,44005}.txt → 404 (retired); {41048,44027}.txt → 200
- result: kept — panel swap 44004→41048 (West Bermuda), 44005→44027 (Jonesport, ME)
- verified_at: 2026-07-03T14:45:00Z

### metar_us_airports (panel)
- attempted: aviationweather.gov/api/data/metar?ids=PANC → 200 with data
- result: kept — panel fix KANC→PANC (Alaska ICAO prefix is PA-; KANC returns 204 empty)
- verified_at: 2026-07-03T14:45:00Z

### wikimedia_per_article_pageviews (panel)
- attempted: per-article/.../Tesla,_Inc./daily/... → 200
- result: kept — YAML fix: quoted "Tesla,_Inc." (bare comma split the flow mapping into
  {ARTICLE: Tesla, _Inc.: null}). Cleveland_Clinic/NATO 404s were transient; both 200 now.
- verified_at: 2026-07-03T14:45:00Z

### gharchive_hourly_events
- attempted: data.gharchive.org/{YYYY-MM-DD}-{H-1}.json.gz
- result: kept — new scraper endpoint type `ndjson_minute_counts`
- reason: hourly files are gzipped NDJSON, not a JSON document; json.loads failed with
  "Extra data". New parser regex-extracts created_at, buckets per-minute counts.
- verified_at: 2026-07-03T15:00:00Z

### pypi_downloads_per_pkg
- result: kept — scraper now honours Retry-After on 429/503 (2 retries, capped 60s);
  8/30 packages had been dropped by pypistats burst rate-limiting.
- verified_at: 2026-07-03T15:00:00Z

### reddit_top_daily
- attempted: www/old/api.reddit.com JSON endpoints, with contact User-Agent
- result: DISABLED (`disabled: true` in sources.yaml)
- reason: 403 / redirect-to-login from datacenter IPs regardless of UA. Needs a Reddit
  OAuth app + OAuth-aware fetch to revive.
- verified_at: 2026-07-03T14:50:00Z

### lta_singapore_traffic_speeds
- attempted: /ltaodataservice/TrafficSpeedBands and /v3/TrafficSpeedBands with AccountKey header
- result: DISABLED
- reason: HTTP 401 on both — the configured LTA_ACCOUNT_KEY is rejected. Operator: check
  DataMall account activation or regenerate key, then re-enable.
- verified_at: 2026-07-03T14:40:00Z

### polymarket_btc_history / polymarket_new_market_rate
- attempted: clob.polymarket.com, gamma-api.polymarket.com
- result: DISABLED
- reason: "no route to host" from the scrape host — connection-level block, not HTTP.
  btc_history additionally still has the TBD-FROM-GAMMA-API panel sentinel. Re-enable
  after network access is restored (+ panel activation for btc_history).
- verified_at: 2026-07-03T14:50:00Z

### Environment-only failures (no catalog change)
- open_meteo_forecast / _uv_index / _climate_daily: SSL handshake timeout from this host
  (api.open-meteo.com unreachable at connection level; endpoints known-good historically).
- nyc_mta_subway_hourly: DNS resolution failure for data.ny.gov from this host.
- Left enabled: both look host-network-specific, not endpoint regressions.

### Scraper changes (2026-07-03)
- fetch_payload: 429/503 → honour Retry-After (capped 60 s), 2 retries.
- parse_payload: new endpoint type `ndjson_minute_counts` (GH Archive).
- run_one/--list: `disabled: true` + `disabled_reason` per-source skip support.

### lta_singapore_traffic_speeds (update, same day)
- attempted: https://datamall2.mytransport.sg/ltaodataservice/v3/TrafficSpeedBands with AccountKey header
- result: RE-ENABLED — key fixed by operator; URL moved to https v3 (unversioned path 404s);
  auth changed to explicit `env:LTA_ACCOUNT_KEY:AccountKey` (plain `env:VAR` infers
  `Authorization: Bearer`, which DataMall rejects). Fresh payload (lastUpdatedTime within
  5 min), ~10k segments; snapshot semantics — one raw row per poll.
- verified_at: 2026-07-03T16:26:00Z

### open_meteo_* / nyc_mta_subway_hourly (update, same day)
- attempted: re-ran all four previously failing sources after network diagnostics
- result: kept, all writing (open_meteo_forecast 192, _uv_index 384, _climate_daily 2192,
  nyc_mta_subway_hourly 6515 rows). The earlier failures were transient: WSL2
  resolver-proxy DNS blip (data.ny.gov) and a temporary SSL-handshake stall window
  (api.open-meteo.com, Hetzner). DNS/TLS/TCP all verified clean on retest.
- scraper change: fetch_payload now also retries httpx.TransportError (3 attempts,
  5s/10s backoff) alongside the 429/503 Retry-After handling, so cron runs don't
  lose a source to a network blip.
- verified_at: 2026-07-03T16:48:00Z

## 2026-07-03 (later) — historical backfill pass: deepen shallow daily/weekly series

Lookbacks widened (all verified writing): banxico_daily_rates → `/datos/2016-01-01/{YYYY-MM-DD}`
(2,641 rows/series; was latest-datum-only), wikimedia_per_article + _health → start 20150701
(~4,019 rows/article; _health also had a hardcoded stale end 20260509), npm_* → 540-day range
(541 rows; API max ~18 months/query), usgs_streamflow_daily → rolling 10y (3,650 rows; had
hardcoded Apr–May 2026 window), usda_snotel_swe → rolling 10y (3,649 rows),
treasury_yield_curve_daily → {YEAR} panel 2016–2026 with new `panel_concat: true`
(2,625 rows, one continuous series). API-capped, cannot deepen: pypistats (180d), crates.io (90d).
wikimedia panel note: 'Cryptocurrency' 404s with the 2015 start — 29/30 articles fine.

### cdc_nssp_ed_visits — dataset was wrong, replaced
- rdmq-nq56 has categorical `ed_trends_*` per county and **no percent_visits/pathogen columns**;
  the schema never matched (2-row writes). Repointed to vutn-jzwm (NSSP percent of ED visits),
  national geography, panel over pathogen {COVID-19, Influenza, RSV} — 142 weekly rows each,
  fresh to 2026-06-20.

### Scraper/loader fixes surfaced by the backfill
- **Loader chronological sort (correctness bug):** ScrapedLiveSource._extract_motif sliced
  windows in file order; newest-first feeds (treasury CSV, EIA sort=desc) were being served
  TIME-REVERSED to forecasters. Loader now sorts by parsed timestamp when >90% parse and the
  series isn't already ascending. Verified: treasury serves 2016-01-04 → 2026-07-02.
- **_apply/_walk flattening:** multi-level paths (`a[].b[].c`) returned nested lists and
  collapsed to 1 record (SNOTEL). Iterate steps now flatten one level.
- **panel_concat:** `panel_concat: true` skips `_panel_` tagging for panels that paginate one
  series over time (treasury years) instead of enumerating distinct series.

### Bin eligibility after backfill (series with enough rows in the latest snapshot)
- daily/weekly (need 286): 7 → 84 of 150
- monthly+ (need 140): 0 → 1
- hourly-ish (need 560): 22 of 45; sub-hourly (need 1120): 7 — accumulates via cron.

## 2026-07-03 (evening) — unseen-fraction weighting + serving-validity fixes

Benchmark change (Chris's design): challenges are now weighted by how much of their
truth window postdates a daily cutoff (start of current UTC day), rather than
hard-anchoring truth to the live tail. `w = 0.25 + 0.75 * unseen_frac`
(config.UNSEEN_WEIGHT_FLOOR): fully-historical truth contributes at the floor
(partial hold for breadth), post-cutoff truth at full weight. Plumbing: MotifMeta
gains `ts` (aligned datetime64 timestamps); ScrapedLiveSource/FreshBuffer take
`tail_frac=0.5` (share of windows pinned to the fresh end); build_live_challenges
takes `cutoff` and stamps meta["unseen_frac"]; evaluate_forecaster weights every
metric accumulation (legacy sets weigh uniformly — results unchanged, suite green).

Serving-validity fixes shaken out by the first weighted leaderboard run
(pooled MASE was ~430 from a handful of poisoned challenges; now median 1.2, mean ~3,
panel rungs order correctly with parrot last):
- **Gap segmentation:** windows can no longer straddle sampling gaps (>8× median
  interval). The loader samples inside the longest contiguous segment, tile-padding
  if none is long enough. gharchive's scattered hourly chunks were serving fake
  level-shifts with MASE in the thousands.
- **gharchive_hourly_events relabeled PT1H → PT1M** (parser emits per-minute counts;
  the PT1H label pointed MASE at the wrong season). Contiguous only once cron polls hourly.
- **cdc_nwss_wastewater pinned to 3 wwtp sites via key_plot_id** (unfiltered, thousands
  of sites interleave at the same week → noise). NOTE: public dataset frozen at
  2025-09-07; keeps floor weight only.
- **Timestamp parse fallbacks** for wikimedia (YYYYMMDDHH) and NDBC ("YY MM DD hh mm")
  so those series carry ts and can earn unseen weight (pool ts-None 23→12 of 96).

## 2026-07-04 — v2: per-cadence challenge profiles + freq-aware seasonal floor

Grounded in the 2026-07-03 rank-stability experiment (3x3 context/horizon grid x
3 cadence bands x 3 seeds, frozen panel, freq-aware MASE):
- hourly band: ctx 512 cuts panel MASE ~35% vs 256 and flips the ranking — shape
  changes conclusions, so one global shape was hiding real differences.
- daily band: context_parrot ranked #1 in 8/9 cells at h>=24 days — long daily
  horizons reward retrieval, not forecasting. Horizon shortened to 14.
- sub-hourly at ctx 1024: parrot dominates outright (0.62 vs 0.87 runner-up);
  ctx 512 keeps the parrot floor honest. 1024 revisit deferred.
- H1 refuted as stated: seasonal_naive did NOT improve with context because its
  period search was hardcoded to (12,24,36) — it could never express a 288-step
  (5-min) or 1440-step (1-min) daily cycle. Fixed: the search now adds the
  frequency-derived natural cycle when it fits the context.

Changes (benchmark-version bump — panel behaviour changes on profiled sets):
- config.PROFILES: (context, horizon) per cadence band; FREQ_SEASONALITY moved
  to config (shared by evaluator MASE scaling + panel seasonality search).
- challenges.build_live_challenges cuts each profile shape from the fresh end
  of the pooled motif; meta gains context_len/horizon; context shrinks to fit
  short pools (fixtures), horizon never silently changes.
- score.py panel + strong candidates are horizon- and freq-aware via meta;
  identical behaviour for meta-less calls (fixed-shape tests unchanged).
- Verified live: shape mix (256,14)x60 (512,24)x40 (512,48)x17 (128,8)x5 (64,8)x6;
  leaderboard strong>ewma>...>parrot with best MASE 1.25 (was 2.7 at one global
  shape); replay byte-identical; demo end-to-end green; 176 tests pass.
