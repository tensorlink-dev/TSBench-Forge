# Sweep-2 vetting — 2026-07-06

Rounds 114–155 (GLM 114–153, Sonnet 154–155): 162 unique accepted candidates,
every endpoint live-probed (scheme-less URLs re-probed with https).

## Probe-verified keyless (53) — the wiring queue
| candidate | domain/cadence | probe |
|---|---|---|
| Open Library recent changes stream (book edit/public | sales/irregular | 200 |
| Chicago food safety inspections (Socrata open data p | healthcare/irregular | 200 |
| iNaturalist observations API — species observation a | nature/irregular | 200 |
| OFAC sanctions list updates — RSS feed | econ_fin/irregular | 200 |
| URLScan.io — public scan submission stream | web_cloudops/irregular | 200 |
| ransomware_live_attacks | web_cloudops/irregular | 200 |
| seeclickfix_citizen_reports | sales/irregular | 200 |
| smhi_swedish_weather_observations | nature/half-hour | 200 |
| ocean_networks_canada_sensors | nature/half-hour | 200 |
| discogs_marketplace_sales | sales/irregular | 200 |
| vaers_vaccine_adverse_events | healthcare/irregular | 200 |
| stripe_status_incidents | web_cloudops/half-hour | 200 |
| tvmaze_show_updates | sales/irregular | 200 |
| glerl_great_lakes_levels | nature/half-hour | 200 |
| amprion_grid_events | energy/irregular | 200 |
| uk_ons_time_series | econ_fin/irregular | 200 |
| ca_chhs_facility_outbreaks | healthcare/irregular | 200 |
| nl_luchtmeetnet_air_quality | nature/half-hour | 200 |
| watttime_marginal_emissions | energy/irregular | 200 |
| celestrak_satellite_tle | transport/irregular | 200 |
| nasdaq_trader_halts | econ_fin/irregular | 200 |
| inciweb_wildfire_incidents | nature/irregular | 200 |
| haveibeenpwned_breaches | web_cloudops/irregular | 200 |
| wastewaterscan_pathogen_data | healthcare/weekly | 200 |
| speedrun_com_run_submissions | sales/irregular | 200 |
| pubmed_ncbi_publication_counts | healthcare/irregular | 200 |
| bitfinex_btcusd_30m_candles | econ_fin/half-hour | 200 |
| uptimerobot_monitors | web_cloudops/half-hour | 200 |
| poweroutage_us_outage_counts | energy/irregular | 200 |
| apnic_labs_ipv6_adoption | web_cloudops/half-hour | 200 |
| netblocks_internet_disruptions | web_cloudops/irregular | 200 |
| ams_fireball_reports | nature/irregular | 200 |
| geoscience_aus_earthquakes | nature/irregular | 200 |
| DC 311 service requests (Socrata) | sales/few-min | 200 |
| digitalocean_status_incidents | web_cloudops/irregular | 200 |
| atlassian_status_incidents | web_cloudops/irregular | 200 |
| eskom_loadshedding_status | energy/irregular | 200 |
| health_canada_recalls | healthcare/irregular | 200 |
| dartmouth_flood_observatory | nature/irregular | 200 |
| semantic_scholar_publications | sales/irregular | 200 |
| nerc_reliability_alerts | energy/irregular | 200 |
| peeringdb_network_ixp_changes | web_cloudops/irregular | 200 |
| musicbrainz_edit_activity | sales/irregular | 200 |
| bgs_uk_earthquakes | nature/irregular | 200 |
| esa_space_weather_events | nature/irregular | 200 |
| iaea_pris_reactor_status | energy/irregular | 200 |
| lichess_realtime_api | sales/few-min | 200 |
| caiso_flex_alert_feed | energy/irregular | 200 |
| govuk_mhra_drug_safety_update | healthcare/irregular | 200 |
| avo_alaska_volcano_alerts | nature/irregular | 200 |
| aemo_nemweb_market_notices | energy/irregular | 200 |
| stocktwits_trending_symbols | econ_fin/half-hour | 200 |
| deezer_chart_snapshots | sales/half-hour | 200 |

## Key-gated (env placeholders added): USDA NASS, ODPT Japan, Schiphol,
FAA NOTAM, Sonatype OSSIndex (+ commented eBay/Metaculus).

## Not viable from this host: ~45 404s (model-invented paths — Sonnet/GLM both
hallucinate exact endpoints at ~25-30% rate even when the provider is real),
~20 connection failures (host-blocked class), bot-walls (PHMSA, FMCSA, OPEC,
F-Droid, Fastly status), 13 need panel/parameter discovery first.
