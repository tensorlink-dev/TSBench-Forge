# Candidate sources for verification

Each row: `id | domain | archetype | freq | endpoint hint | novelty guess`

## econ_fin
- polymarket_market_prices | econ_fin | bounded, non_stationary_regime | minute | https://gamma-api.polymarket.com/events | clean
- kalshi_markets | econ_fin | bounded | minute | https://api.elections.kalshi.com/trade-api/v2/markets | clean
- manifold_markets | econ_fin | bounded | minute | https://api.manifold.markets/v0/markets | clean
- coingecko_btc_ohlc | econ_fin | non_stationary_regime | hourly | https://api.coingecko.com/api/v3/coins/bitcoin/market_chart | partial
- binance_klines | econ_fin | non_stationary_regime | 1min | https://api.binance.com/api/v3/klines | partial
- frankfurter_fx | econ_fin | non_stationary_regime | daily | https://api.frankfurter.dev/v1/latest | contaminated
- treasury_yield_curve | econ_fin | non_stationary_regime | daily | https://home.treasury.gov/.../daily_treasury_yield_curve_rates | contaminated
- ecb_fx_daily | econ_fin | non_stationary_regime | daily | https://data-api.ecb.europa.eu/service/data/EXR | contaminated
- bitfinex_funding | econ_fin | non_stationary_regime | minute | https://api-pub.bitfinex.com/v2/tickers | clean
- defillama_tvl | econ_fin | non_stationary_regime | hourly | https://api.llama.fi/v2/historicalChainTvl | clean
- coinbase_spot | econ_fin | non_stationary_regime | minute | https://api.exchange.coinbase.com/products | partial
- alphavantage_intraday | econ_fin | non_stationary_regime | 1min | https://www.alphavantage.co/query | partial

## energy
- eia_grid_demand | energy | hierarchical, smooth_periodic | hourly | https://api.eia.gov/v2/electricity/rto/region-data/data/ | partial
- entsoe_load | energy | hierarchical | hourly | https://web-api.tp.entsoe.eu/api | partial
- caiso_oasis_5min | energy | hierarchical | 5min | http://oasis.caiso.com/oasisapi/SingleZip | partial
- uk_carbon_intensity | energy | smooth_periodic, bounded | 30min | https://api.carbonintensity.org.uk/intensity | clean
- octopus_agile | energy | bounded | 30min | https://api.octopus.energy/v1/products/AGILE-FLEX-22-11-25/.../standard-unit-rates | clean
- aemo_dispatch | energy | hierarchical | 5min | https://aemo.com.au/aemo/data/nem/priceanddemand | clean
- ercot_real_time | energy | hierarchical | 5min | https://www.ercot.com/api/1/services/read | partial
- nordpool_day_ahead | energy | hierarchical | hourly | https://www.nordpoolgroup.com/api/marketdata | clean
- elexon_imbalance | energy | non_stationary_regime | 30min | https://data.elexon.co.uk/bmrs/api/v1/ | clean
- eia_petroleum_daily | energy | non_stationary_regime | daily | https://api.eia.gov/v2/petroleum/.../data/ | partial

## healthcare
- wiki_pageviews_health | healthcare | count_discrete, hierarchical | daily | https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/... | partial
- google_health_trends | healthcare | hierarchical | daily | (pytrends) | partial
- delphi_covid_signals | healthcare | hierarchical | daily | https://api.delphi.cmu.edu/epidata/covidcast/ | clean
- nyc_311_health | healthcare | count_discrete | daily | https://data.cityofnewyork.us/resource/erm2-nwe9.json | clean
- rki_inzidenz | healthcare | hierarchical | daily | https://api.corona-zahlen.org/germany/history/cases | clean
- cdc_wastewater_nwss | healthcare | hierarchical | daily | https://data.cdc.gov/resource/2ew6-ywp6.json | clean
- openfda_drug_events | healthcare | count_discrete | daily | https://api.fda.gov/drug/event.json | clean
- usgs_emergency_alerts | healthcare | binary_state | daily | (FEMA OpenFEMA disasters) | clean
- nyc_health_data_emergency | healthcare | count_discrete | daily | https://a816-health.nyc.gov/... | clean
- cdc_respiratory_dashboard | healthcare | hierarchical | daily | https://data.cdc.gov/resource/.../ | clean

## nature (weather/climate/environment/geophysical)
- usgs_earthquakes_realtime | nature | count_discrete, chaotic_nonlinear | minute | https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson | partial
- usgs_streamflow_iv | nature | hierarchical, chaotic_nonlinear | 15min | https://waterservices.usgs.gov/nwis/iv/ | partial
- noaa_tides_coops | nature | smooth_periodic, hierarchical | 6min | https://api.tidesandcurrents.noaa.gov/api/prod/datagetter | clean
- ndbc_buoys | nature | hierarchical, chaotic_nonlinear | hourly | https://www.ndbc.noaa.gov/data/realtime2/ | partial
- openaq_v3 | nature | hierarchical | hourly | https://api.openaq.org/v3/ | partial
- purpleair_realtime | nature | hierarchical, noisy_clean_pair | 10min | https://api.purpleair.com/v1/sensors | clean
- noaa_swpc_planetary_k | nature | bounded, chaotic_nonlinear | hourly | https://services.swpc.noaa.gov/json/planetary_k_index_1m.json | clean
- noaa_swpc_solar_wind | nature | chaotic_nonlinear | minute | https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json | clean
- open_meteo_forecast | nature | hierarchical | hourly | https://api.open-meteo.com/v1/forecast | partial
- nasa_power_daily | nature | smooth_periodic | daily | https://power.larc.nasa.gov/api/temporal/daily/point | partial
- mauna_loa_co2_daily | nature | smooth_periodic | daily | https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_trend_gl.txt | partial (contaminated trend)
- usda_snotel_swe | nature | smooth_periodic, hierarchical | daily | https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data | clean
- inmet_weather_br | nature | hierarchical | hourly | https://apitempo.inmet.gov.br/ | clean
- ingv_seismicity | nature | count_discrete | minute | https://terremoti.ingv.it/api | clean
- nws_active_alerts | nature | binary_state, categorical | minute | https://api.weather.gov/alerts/active | clean
- airnow_aqi | nature | hierarchical, noisy_clean_pair | hourly | https://airnowapi.org/aq/ | partial
- gdacs_disasters | nature | count_discrete, zero_inflated_sparse | hourly | https://www.gdacs.org/xml/rss.xml | clean

## sales (consumer attention/retail proxy)
- hn_top_stories | sales | hierarchical, non_stationary_regime | minute | https://hacker-news.firebaseio.com/v0/topstories.json | partial
- wikimedia_pageviews | sales | hierarchical | daily | https://wikimedia.org/api/rest_v1/metrics/pageviews/top/... | partial
- npm_downloads_per_pkg | sales | hierarchical, count_discrete | daily | https://api.npmjs.org/downloads/range/... | partial
- pypi_downloads_pep | sales | hierarchical | daily | https://pypistats.org/api/packages/.../recent | partial
- spotify_charts | sales | hierarchical, categorical | daily | https://charts.spotify.com/charts/.../daily/global | partial
- box_office_mojo | sales | non_stationary_regime | daily | (no public API) | -
- crates_io_downloads | sales | count_discrete, hierarchical | daily | https://crates.io/api/v1/crates/... | clean
- itunes_rss | sales | hierarchical | daily | https://rss.applemarketingtools.com/api/v2/us/podcasts/top/100/podcasts.json | clean
- steam_charts_topgames | sales | hierarchical | minute | https://api.steampowered.com/ISteamChartsService | partial
- imdb_top_chart | sales | hierarchical, categorical | daily | https://www.imdb.com/chart/... | partial

## transport
- opensky_states | transport | count_discrete, hierarchical | minute | https://opensky-network.org/api/states/all | partial
- nyc_mta_subway | transport | hierarchical, smooth_periodic | minute | https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/... | clean
- citibike_station_status | transport | hierarchical, count_discrete | minute | https://gbfs.citibikenyc.com/gbfs/en/station_status.json | clean
- tfl_bus_arrivals | transport | hierarchical | minute | https://api.tfl.gov.uk/Line/.../Arrivals | clean
- bart_realtime | transport | count_discrete, hierarchical | minute | https://api.bart.gov/api/etd.aspx | partial
- nyc_311_complaints | transport | count_discrete | daily | https://data.cityofnewyork.us/resource/erm2-nwe9.json | partial
- vesselfinder_ais | transport | hierarchical | minute | (gated) | -
- caltrans_pems | transport | hierarchical | 5min | (gated) | -
- nrel_alt_fuel | transport | hierarchical | daily | https://developer.nrel.gov/api/alt-fuel-stations/v1.json | clean
- portland_streetcar | transport | hierarchical | minute | https://developer.trimet.org | clean
- gtfs_rt_norway_entur | transport | hierarchical | minute | https://api.entur.io/realtime/v1/services/vehicles | clean

## web_cloudops
- hn_realtime_max_item | web_cloudops | count_discrete | minute | https://hacker-news.firebaseio.com/v0/maxitem.json | partial
- github_events_firehose | web_cloudops | count_discrete | minute | https://api.github.com/events | clean
- cloudflare_radar_attacks | web_cloudops | bounded | hourly | https://api.cloudflare.com/client/v4/radar/attacks | clean
- aws_health_status | web_cloudops | binary_state, categorical | hourly | https://status.aws.amazon.com/data.json (RSS) | clean
- npm_downloads_total | web_cloudops | non_stationary_regime | daily | https://api.npmjs.org/downloads/range/... | partial
- pypi_total_downloads | web_cloudops | non_stationary_regime | daily | https://pypistats.org/api/... | partial
- so_questions_per_tag | web_cloudops | count_discrete, hierarchical | daily | https://api.stackexchange.com/2.3/questions | partial
- bgp_route_count_ripe | web_cloudops | count_discrete | daily | https://stat.ripe.net/data/ris-dumps/ | clean
- public_dns_responses | web_cloudops | count_discrete | hourly | https://stats.dns.icann.org/ | clean
- archive_org_wayback_count | web_cloudops | count_discrete | daily | https://archive.org/wayback/available | partial

## archetype-driven additions (to fill gaps)
- noaa_swpc_kp_index | nature | bounded (0-9), chaotic_nonlinear | hourly | https://services.swpc.noaa.gov/json/planetary_k_index_1m.json | clean
- nws_alerts_active | nature | binary_state | minute | https://api.weather.gov/alerts/active/count | clean
- noisy_clean_pair: PurpleAir + EPA AirNow at same lat/lon | nature | noisy_clean_pair | 10min/hourly | combined | clean
