## Block 1 — Gap Analysis

The current pool of 238 sources is strong on daily and hourly cadences across most domains, and contamination risk is well-managed (174 low, only 6 high). However, several **high-value gaps** remain, concentrated in the scarcest cadence bands:

**Healthcare is the biggest hole.** It has zero sources at sub-minute, few-minute, half-hour, or irregular cadences — all high-value cells with a target of 3. The existing healthcare sources are almost all daily or weekly (CDC, FDA, RKI, VAERS, etc.), which are useful but don't test models on rapid health-related dynamics. Air quality data is the most viable bridge: several citizen-science and environmental networks publish health-relevant sensor data at high frequency, and these are genuinely unlikely to appear in TSFM pretraining corpora.

**Irregular/event-driven cadence is under-filled everywhere.** Nature, sales, transport, and web_cloudops all have zero or near-zero irregular sources despite targets of 3. Irregular event streams are the gold standard for contamination resistance (future events can't be memorized) and for discriminating models (they're inherently non-periodic and bursty). The few existing irregular sources (elexon frequency, ripe atlas, abs cpi) are isolated.

**Web_cloudops × half-hour is empty.** The pool has many status-page sources at hourly or few-minute cadence, but none at half-hour. Statuspage.io-powered services expose a consistent JSON API (`/api/v2/incidents.json`) that can be polled at any cadence, making this an easy, high-confidence fill.

**Transport × irregular and × half-hour are thin.** Most transport sources are sub-minute realtime feeds (TfL, BART, CitiBike). GTFS-RT Service Alert feeds from transit agencies not yet in rotation provide genuinely irregular event streams — service disruptions, elevator outages, weather-related suspensions — that are impossible to memorize and hard to forecast.

**Energy × sub-min and × irregular remain thin** despite many proposed sources. Most TSOs publish at 5-minute or coarser; true sub-minute public energy data is rare (mostly grid frequency). I'm not confident enough in specific new endpoints to propose them without verification.

**Sales × irregular and × half-hour are thin.** The pool has many daily sales/download sources but few irregular event streams. New decentralized social platforms (Bluesky, Farcaster) are ideal: they postdate all model cutoffs, so their data is contamination-free by construction.

**Ranked gaps targeted:**
1. Healthcare × sub-min/few-min/half-hour/irregular (deficit 3 each)
2. Web_cloudops × half-hour (deficit 3)
3. Sales × irregular (deficit 3) and × half-hour (deficit 2)
4. Nature × irregular (deficit 3)
5. Transport × irregular (deficit 3)

---

## Block 2 — Candidate Sources
