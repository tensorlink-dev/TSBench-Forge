# DGP-class taxonomy

**Why this exists.** TSBench-Forge is a *generalization-across-domains* benchmark
for TSFMs. Two sources in the same `domain` and same `archetype` can still measure
completely different dynamical systems — e.g. `usgs_earthquakes_realtime` and
`swpc_solar_xray_flux` both live in `nature/count_discrete`, but earthquake counts
follow a self-organized-criticality process while solar X-ray flux follows a
different underlying stochastic driver. Scoring by `domain` count over-weights
whichever DGP class has more sources; scoring by DGP class weights the
generalization signal directly.

`dgp_class` is a categorical tag on every source, orthogonal to `domain` and
`archetype`. The eval-pool builder uses it to sample equal-weight per DGP class
per epoch, so a domain with 6 sources but 6 duplicate DGP classes contributes as
much as a domain with 2 sources of 2 distinct DGP classes.

## Field syntax

Add to any `sources.yaml` / `sources-slow.yaml` entry:

```yaml
- id: my_new_source
  domain: nature
  dgp_class: weather_field    # <-- one of the classes below
  archetypes: [...]
  ...
```

Multi-DGP sources (rare — usually indicates the source should be split): use a
list, `dgp_class: [weather_field, alert_stream]`.

## The classes

Grouped by domain for readability. Add new classes with a PR — the taxonomy
should stay small (<40) so the eval-pool sampler doesn't slice into cells with
one series each.

### `nature` (8 classes)

| class | example sources | dynamics |
|---|---|---|
| `weather_field` | METAR, Open-Meteo forecast, JMA | grid + station atmospheric state, seasonal + weather-system driven |
| `climate_field` | NOAA NCEI monthly, GISS regional | slow trends + interannual variability |
| `climate_index` | NOAA Drought Monitor, Mauna Loa CO₂ | scalar reductions of climate state |
| `hydrology` | USGS streamflow, USDA SNOTEL | terrestrial water cycle, seasonal + event-driven |
| `tidal_periodic` | NOAA CO-OPS tides | astronomically forced quasi-periodic |
| `seismic_event` | USGS earthquakes, INGV | self-organized-criticality, Gutenberg-Richter |
| `space_weather` | SWPC K-index / X-ray flux / solar wind | solar-driven, event burstiness |
| `air_quality` | Open-Meteo AQ, OpenAQ | meteorology + emissions coupling |

### `econ_fin` (10 classes)

| class | example sources | dynamics |
|---|---|---|
| `market_price` | Binance, Coinbase, CoinGecko | continuous-price trading dynamics, near-random-walk |
| `market_orderbook` | (none yet — could add L2 snapshots) | order-flow microstructure |
| `prediction_market` | Kalshi, Manifold, Polymarket | probability-bounded, event-driven convergence |
| `fx_rate` | Frankfurter FX, Banxico | central-bank + carry-trade driven |
| `monetary_policy_rate` | Treasury yield curve, ECB liquidity | policy-driven discrete regime shifts |
| `monetary_policy_flow` | ECB weekly bank liquidity | central-bank operations volume |
| `macro_economic_indicator` | (FRED mid-tail once added) | slow-cadence policy outputs |
| `macro_inflation` | Eurostat HICP | monthly rate-of-change, drift + stickiness |
| `macro_leading_indicator` | OECD CLI | forward-looking composite index |
| `macro_historical` | Maddison GDP | multi-century very-long-context |
| `development_indicator` | World Bank WDI | annual cross-country panel |

### `energy` (4 classes)

| class | example sources | dynamics |
|---|---|---|
| `grid_demand` | EIA hourly demand, AEMO | strong daily + weekly seasonality |
| `generation_mix` | UK carbon intensity, BPA wind | intermittent renewables + dispatch |
| `energy_price` | CAISO/NYISO LMP, EIA natgas spot | congestion + fuel + weather driven |
| `energy_supply_flow` | EIA weekly petroleum stocks | weekly balance flows |

### `transport` (5 classes)

| class | example sources | dynamics |
|---|---|---|
| `urban_transit_ridership` | Citibike, BART, TfL arrivals, MTA hourly | daily + weekly + calendar-shock |
| `rail_operations` | Amtrak realtime | schedule adherence, event-cascade delays |
| `aviation_traffic` | EUROCONTROL daily | seasonal + weather + ATC events |
| `aviation_congestion` | FAA NASSTATUS | zero-inflated, weather-triggered spikes |
| `shipping` | (MarineCadastre AIS once added) | port-cycle + trade flows |
| `air_traffic_flow` | OpenSky states | slow-moving spatial density |
| `winter_road_ops` | Iowa DOT plow AVL | weather-triggered fleet deployment; zero-inflated, seasonal on/off |

### `healthcare` (5 classes)

| class | example sources | dynamics |
|---|---|---|
| `syndromic_surveillance` | CDC NSSP ED visits, CDC FluView | early-warning, seasonal + outbreak |
| `disease_case_count` | RKI cases, CDC NNDSS | count process + reporting delay |
| `environmental_biosurveillance` | CDC NWSS wastewater | environmental proxy for community infection |
| `vital_statistics` | WHO GHO life expectancy | slow-moving demographic |
| `mortality` | (CDC WONDER once added) | population aggregate |

### `sales` (5 classes)

| class | example sources | dynamics |
|---|---|---|
| `attention_pageviews` | Wikipedia pageviews | Zipfian popularity + viral spikes |
| `download_counts` | npm, PyPI, crates | ecosystem adoption + release cycles |
| `content_ranking` | HN top stories, iTunes podcasts | rank-based, non-stationary at short timescales |
| `agricultural_progress` | USDA NASS crop progress | strongly seasonal |
| `agricultural_forecast` | USDA WASDE | month-over-month revision series |
| `agricultural_productivity` | FAO crop yields | yearly, weather + policy + tech |

### `web_cloudops` (5 classes)

| class | example sources | dynamics |
|---|---|---|
| `code_activity` | GitHub events firehose, gharchive | ecosystem-driven, weekly + release burst |
| `edit_stream` | Wikipedia recent changes | user-driven, near-Poisson within-domain |
| `qa_stream` | Stack Overflow questions | user-driven, weekly + event-driven |
| `network_flow` | RIPE announced prefixes | infra-driven, slow-changing |
| `alert_stream` | NWS active alerts, GDACS disasters | zero-inflated + burst |
| `service_health` | AWS Health RSS | zero-inflated, incident-driven |
| `observability_metric` | (BOOM-adjacent — not yet in catalog) | infra-vendor telemetry (would be a new class if added) |

## Guidance for adding a new source

Before adding a new source, ask:

1. **What DGP does it measure?** If the answer is "same DGP class as an existing
   source with different geography or ID," you're adding volume, not
   generalization signal. Consider whether the addition earns its place.
2. **Does it fill a gap?** Look at the coverage matrix — a class with 1 source
   is fragile; adding a second source of the same class hardens it, adding a
   different class expands coverage.
3. **Is it distinct from `archetype`?** The `archetype` tag captures the
   statistical shape (count_discrete, smooth_periodic, chaotic_nonlinear).
   `dgp_class` captures the underlying process. Two sources can share an
   archetype but differ in DGP.

## What the eval-pool builder does with this tag

`scraped_source.ScrapedLiveSource` uses `dgp_class` to sample equal-weight per
class per epoch. So a domain with 6 sources of 3 DGP classes contributes at the
class level, not source level. This prevents the "20 weather stations = 20 nature
votes" trap identified in the coverage skew review.

## Backfilling existing sources

The Priority 1-3 additions in this repo (2026-07-01) all carry `dgp_class`. The
77 pre-existing sources in `sources.yaml` do **not** — backfilling them is a
separate pass. When you touch an existing entry, add its `dgp_class` per this
taxonomy.
