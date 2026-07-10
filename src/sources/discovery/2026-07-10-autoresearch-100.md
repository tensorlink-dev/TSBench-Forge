# Autoresearch sweep — 100 rounds, 2026-07-10

103 attempts (`python -m source_discovery`, GLM-5.2 via OpenRouter), **100
succeeded, 3 failed** — all three failures were rounds 1/4/5 before the
reasoning fix below; zero failures in the 100 rounds after it. Full per-round
log in the sweep dir (`src/sources/discovered/sweep-2026-07-10/`, gitignored);
merged candidates in `2026-07-10-candidates-100rounds.json` (this dir).

## Totals

- 1295 proposals across 100 productive rounds; vet verdicts 562 accept /
  55 flag / 678 reject (rejects are mostly ledger re-proposals — the dedup
  working as designed).
- **663 unique candidates** after (domain, host) dedupe: 454 accept / 45 flag /
  164 reject.
- Proposal ledger grew 409 → **1004** entries.
- Accepts by domain: healthcare 96, nature 92, transport 87, web_cloudops 62,
  sales 47, econ_fin 38, energy 32. 370/454 accepts are sub-hourly or
  irregular — the high-value bands.

## Two client bugs found and fixed this sweep (`source_discovery/llm.py`)

1. **z-ai ignores `reasoning.max_tokens`.** GLM-5.2 was dying with
   `finish_reason=length` after burning the entire completion budget (even the
   2x retry, 64k tokens) on hidden reasoning — 3 of the first 5 rounds. The
   OpenRouter reasoning-cap parameter is silently ignored by the z-ai provider
   (verified with a live probe). Fix: `OPENROUTER_REASONING_ENABLED=false`
   sends `reasoning: {"enabled": false}`, which provably zeroes reasoning
   tokens. Side effect: rounds went from minutes to ~35 s and never died.
2. **Empty Block-2 responses.** Non-thinking GLM sometimes ends cleanly right
   after the "Block 2 — Candidate sources" header without emitting the JSON
   (rounds 28/36/37/39 produced 0 proposals). `llm.propose` now retries once
   when a response parses to zero candidates; no zero-proposal rounds after
   the fix except genuine all-reject rounds.

## Probe pass — ALL 273 unique accepted hosts (both passes)

Pass 1 (119 high-confidence sub-hourly/irregular hosts): 57× 200, 25× 404,
14× unreachable, 12× auth-gated, 4× websocket, 1× each 429/413/502.
Pass 2 (the remaining 154 accepted hosts): 61× 200, 38× 404, 28× unreachable,
21× auth-gated, 2× websocket. Every accept with a usable URL is probed
(3 templated URLs like `{NIGHTSCOUT_URL}` skipped). **Total: 118× HTTP 200.**
Raw results: `probe_results*.tsv` in the sweep dir.

Vetting coverage, to be precise about what each stage means: all 1295
proposals went through the automatic metadata vet (schema/denylist/dedup);
all 273 unique accepted hosts are live-probed; the DATA admission gate
(`--assess`) runs only after a source is wired and scraped — none have been.

Key-gated candidates with free key programs now have placeholders in
`.env(.example)`: WMATA, TransLink Vancouver, Cloudflare Radar, BART, Lichess,
NASA FIRMS, OpenChargeMap, AbuseIPDB, NVE Norway (EIA storage reuses
EIA_API_KEY). Not-free / no-key-program endpoints are documented as comments
there so nobody re-treads them (PulsePoint, Flightradar24, Stocktwits,
TreasuryDirect/AEMO/GBFS bot-walls, …).

Pass-2 200s skew toward bare-host probes from medium/low-confidence proposals
(a 200 on `sipc.org` verifies the host, not a feed); the concrete-endpoint
standouts there: Celestrak TLE updates, Deezer charts, TVMaze show updates,
Discogs marketplace, Semantic Scholar, PubMed eutils, MusicBrainz edits,
api.corona-zahlen.org (RKI), Keepa (key-gated in practice), Jamendo.

Caveats before admission: a 200 on a bare-host URL (cftc.gov, data.cdc.gov,
api.openaq.org, api.metro.net…) verifies the host, not the endpoint; some hits
are already wired or known-blocked (WSDOT ferries, Cloudflare Status,
data.ny.gov MTA; Polymarket is geo-blocked from the AU scrape host).

### Shortlist — probe-200, keyless, genuinely new, concrete endpoint

| candidate | domain / cadence | endpoint |
|---|---|---|
| USGS all-hour earthquake feed | nature / irregular | earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson |
| Geonet NZ quakes | nature / irregular | api.geonet.org.nz |
| Geoscience Australia earthquakes | nature / few-min | earthquakes.ga.gov.au/api/v1/events |
| PegelOnline Rhine river gauges | nature / few-min | pegelonline.wsv.de rest-api v2 |
| Ocean Networks Canada sensors | nature / sub-min | data.oceannetworks.ca/api |
| NOAA SWPC solar-flare event reports | nature / irregular | swpc.noaa.gov (json feeds under services.swpc.noaa.gov) |
| Energinet DK PowerSystemRightNow | energy / sub-min | api.energidataservice.dk (already probed 200 in 07-05 sweep) |
| Binance klines + funding rate | econ_fin / few-min–hourly | api.binance.com, fapi.binance.com |
| Bitfinex BTCUSD 30m candles | econ_fin / half-hour | api-pub.bitfinex.com |
| Gemini BTC trades | econ_fin / irregular | api.gemini.com/v1/trades/btcusd |
| Blockstream block arrivals | econ_fin / few-min | blockstream.info/api/blocks |
| Mempool.space unconfirmed count | econ_fin / sub-min | mempool.space/api/mempool |
| Nasdaq trade halts RSS | econ_fin / few-min | nasdaqtrader.com/rss.aspx?feed=tradehalts |
| openFDA device adverse events | healthcare / irregular | api.fda.gov/device/event.json |
| Citibike station status (GBFS) | transport / sub-min | gbfs.citibikenyc.com |
| MBTA service alerts | transport / few-min | api-v3.mbta.com/alerts |
| SEPTA alerts | transport / few-min | www3.septa.org/api/Alerts |
| LTA Singapore taxi availability | transport / half-hour | api.data.gov.sg |
| HN Algolia front page + Firebase topstories | sales / sub-min | hn.algolia.com, hacker-news.firebaseio.com |
| Lobste.rs hottest | sales / irregular | lobste.rs/hottest.json |
| OpenLibrary recent changes | sales / half-hour | openlibrary.org/recentchanges.json |
| Speedrun.com run submissions | sales / hourly | speedrun.com/api/v1/runs |
| GH Archive release events | sales / irregular | data.gharchive.org |
| GitHub public events | web_cloudops / irregular | api.github.com/events |
| NWS active alerts | web_cloudops / irregular | api.weather.gov |
| Statuspage incident feeds (Atlassian / DigitalOcean; Cloudflare already known) | web_cloudops / hourly | status.\*.com/api/v2/incidents.json |
| Ransomware.live victim feed | web_cloudops / irregular | api.ransomware.live/v1/victims |
| Wikimedia recent-changes stream | web_cloudops / irregular | stream.wikimedia.org (SSE — check scraper support) |
| URLScan public search | web_cloudops / few-min | urlscan.io/api/v1/search/ |

Next: standard admission pipeline per candidate (sources.yaml entry → dry-run →
scrape → `--assess` → cron → PR).
