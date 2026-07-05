# Autoresearch sweep — 100 rounds, 2026-07-04/05

Rounds 14–113 (GLM-5.2 via OpenRouter, 16k completion budget): **91 succeeded**,
9 failed cleanly on reasoning-budget exhaustion. Combined with rounds 1–13:
**142 unique novel candidates** (deduped by domain × host and against the
103-source catalog); full metadata in `2026-07-05-candidates-100rounds.json`.
Vet verdicts: 112 accept / 4 flag / 26 reject.

## Conviction signal (times independently proposed across rounds)
Steam players 55×, PurpleAir 38×, openFDA 30×, NASA FIRMS 24×, Twitch 23×,
ENTSO-E 40× (two hosts), SEC EDGAR 16×, Cloudflare 15×, WHO DON 15× — the model
converges hard on the same top sources, which independently validates the
2026-07-04 shortlist (all already wired in or key-gated).

## New probe-verified keyless candidates (this sweep)
| candidate | domain / cadence | probe |
|---|---|---|
| Eurostat monthly indicators API | econ_fin / P1M | 200 |
| NOAA NCEI storm-events CSV archive | nature / irregular | 200 |
| Cloudflare Status incidents JSON | web_cloudops / irregular | 200 |
| RIPE Atlas measurements API | web_cloudops / PT30M | 200 (needs measurement-id panel) |
| FDIC bank failures API | econ_fin / irregular | 200 |
| FEMA DisasterDeclarationsSummaries | nature / irregular | 200 (proposed path needed “Summaries” suffix) |
| Energinet DK datasets (Elspotprices, PowerSystemRightNow) | energy / PT1M-PT1H | 200 (dataset name lowercase) |

Blocked/wrong from probes: OpenChargeMap (403, needs key), NTSB AviationData
(404, path stale), ProMED (308 → HTML), AISStream (websocket), Product Hunt
(GraphQL + OAuth), PJM/ERCOT (host-blocked/bot-walled, known).

Next batch when desired: the seven 200s above can go through the standard
admission pipeline (entry → dry-run → scrape → --assess → cron → PR).

---

## 2026-07-05 (later) — ledger-aware rounds: 32 fresh candidates, all live-probed

10 rounds with the new proposal ledger active: 39 proposals, **zero repeats**
(vs Steam 55x blind), 32 accepts — every one a host never proposed in 113 prior
rounds. All probed from the scrape host:

### Verified live, keyless — ready for the admission pipeline (12)
| candidate | domain / cadence | note |
|---|---|---|
| Launch Library 2 (thespacedevs) | transport / irregular | 200 |
| bioRxiv/medRxiv details API | healthcare / irregular | 200, date-ranged |
| GeoNet NZ quakes | nature / irregular | 200; NZ counterpart to USGS/EMSC |
| **PEGELONLINE water gauges** | nature / PT15M | 200 via `www.pegelonline.wsv.de` (bare host connection-fails) |
| Jikan (MyAnimeList) episodes | sales / irregular | 200 |
| CFTC press-release RSS | econ_fin / irregular | 200 |
| CPSC recalls RSS | healthcare / irregular | 200 via `/Newsroom/CPSC-RSS-Feed` (proposed path 404) |
| MBTA v3 alerts | transport / irregular | 200 keyless (rate-limited) |
| ISC/SANS DShield portdata | web_cloudops / PT5M | 200 |
| SEPTA alerts | transport / irregular | 200 via `/api/Alerts/index.php` |
| World Bank WDI | econ_fin / P1Y | 200 |
| OECD MEI SDMX-JSON | econ_fin / P1M | 200 (scraper already parses SDMX-JSON) |

### Key-gated, free registration (ledger statuses updated)
abuse.ch trio (URLhaus/MalwareBazaar/ThreatFox — Auth-Key since 2025), Trakt
(client id), Finnhub, BEA, TriMet, OpenEnergyMonitor, BoardGameGeek (401s now),
AEMET (api_key param).

### Blocked / weak from this host
UN Comtrade, OpenHumans, IMO fireballs, SPP, INTERMAGNET, IMAGE magnetometers:
connection-level 000 (same class as Polymarket). crt.sh: flaky 502/404. FSIS
recalls: 403 bot-wall. Coral Reef Watch: product path needs research. DOL
claims: HTML page, needs a scraping adapter.
