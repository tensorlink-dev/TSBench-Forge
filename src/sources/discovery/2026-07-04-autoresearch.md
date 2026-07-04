# Autoresearch run — 2026-07-04

13 discovery rounds (`python -m source_discovery --out ...`, model z-ai/GLM-5.2 via
OpenRouter, `OPENROUTER_MAX_TOKENS=16000`); 12 produced output, 1 aborted with the
model spending the full 32k completion budget on reasoning (now a clean error).
**45 unique candidates** after dedupe by (domain, host) — full metadata in
`2026-07-04-candidates.json`. Every shortlisted endpoint below was live-probed
from the scrape host. Human review required before anything enters sources.yaml.

## Tier 1 — verified live, keyless, ready to wire in
| candidate | domain / cadence | probe |
|---|---|---|
| NIST NVD CVE 2.0 | web_cloudops / irregular | 200 |
| SEC EDGAR full-text search + current-filings Atom | econ_fin / irregular | 200 |
| SEC EDGAR per-filer submissions (data.sec.gov) | econ_fin / irregular | 200 |
| openFDA food recall enforcement | healthcare / irregular | 200 |
| NASA EONET natural-event tracker | nature / irregular | 200 |
| CISA KEV catalog | web_cloudops / irregular | 200 |
| NOAA SPC storm reports (today.csv) | nature / irregular | 200 |
| tsunami.gov event Atom feeds | nature / irregular | 200 |
| EA UK rainfall gauges (flood-monitoring) | nature / PT15M | 200 |
| WSDOT ferry vessel locations | transport / snapshot | 200 |
| npm registry _changes feed | sales / irregular | 200 |
| TfL road disruptions | transport / irregular | 200 |
| USGS elevated-volcanoes JSON (better than the HTML page proposed) | nature / irregular | 200 |
| EMSC seismic fdsnws | nature / irregular | 200 (vet-rejected only for empty `license` field — human override recommended, data is freely licensed) |

## Tier 2 — real endpoints, need a free key (placeholders in .env.example)
- **PurpleAir PM2.5, PT2M** — fills the healthcare/sub-hourly gap, the emptiest
  high-value cell in the coverage matrix. 403 without key.
- **ENTSO-E outage/unavailability events** (needs securityToken)
- **Global Fishing Watch vessel encounters** (401 without token)
- **WAQI real-time air quality** (token), **Twitch Helix viewer counts** (client id),
  **NASA FIRMS active fires** (MAP_KEY), **UK BMRS system frequency** (APIKey)

## Tier 3 — blocked or weak from this host
- PJM, CAIDA IODA, DOE OE-417, NG ESO data portal: connection-level unreachable (HTTP 000).
- ERCOT / Kickstarter / OpenSky: 403 bot/auth walls.
- Fingrid: 404 (proposed path wrong; portal exists).
- WHO DON, Alberta ER waits: HTML pages, need scraping adapters.
- RIPE RIS Live: websocket — scraper has no ws support.

## Bugs the run surfaced (fixed in this branch)
- `llm.propose` crashed with a bare TypeError when reasoning models returned
  `content=None` (`finish_reason="length"`); now retries once with doubled budget,
  then raises an informative error.
