## Block 1 — Gap Analysis

The current pool of 176 sources is strong on daily cadence (56 sources) and well-stocked in nature (45) and econ_fin (31), but it has a critical structural weakness: **irregular/event-driven cadences are almost absent** (only 1 source across all 7 domains, vs. a target of 2–3 per domain = 14–21 cells). This is the single largest gap by cell-count and also the highest-value gap because event-driven data is both contamination-resistant (future events can't have been pretrained on) and discriminative (models can't lean on simple seasonality).

**Over-represented:** Daily cadence (56 sources, many domains already at target), US-centric public-sector feeds, and air-quality/seismic/space-weather nature sources. The nature domain alone has 45 sources — more than energy (17), healthcare (19), and transport (19) combined with sales (21).

**Under-represented and high-priority:**
1. **Irregular/event-driven across all 7 domains** — deficit 3 in econ_fin, energy, healthcare, nature, sales, transport; deficit 2 in web_cloudops. This is the top priority.
2. **Healthcare sub-hourly (sub-min, few-min, half-hour)** — all at 0/3. Genuinely very hard to fill: most real-time health data is private (CGM, patient monitors), already rejected (PulsePoint, NHS, NightScout, OpenBCI), or already proposed (OpenAQ, ProMED, CDC HAN). I note this explicitly and focus my proposals where I can find genuinely new sources.
3. **web_cloudops/half-hour** (0/3) and **transport/half-hour** (1/3) — high-value sub-hourly cells with no or thin coverage.
4. **energy/sub-min** (1/3) and **energy/irregular** (0/3) — energy has strong few-min/hourly coverage from ISO/RTO feeds but lacks event-driven and sub-minute sources.

**Contamination posture is healthy:** 116/176 sources are low-risk, and most are live/streaming APIs whose future values don't yet exist. The 6 high-risk sources are a small fraction. The main contamination concern is that the pool leans heavily on well-known US/EU government APIs (USGS, NOAA, EIA, CDC, SEC) that could plausibly appear in future pretraining corpora — reinforcing the need for obscure and regional sources.

I was unable to find genuinely new public sources for healthcare/sub-min, healthcare/few-min, healthcare/half-hour, transport/irregular, and web_cloudops/half-hour that aren't already in CURRENT_SOURCES or ALREADY_PROPOSED. These gaps require either licensed data, private APIs, or creative repackaging of existing feeds — all flagged for human follow-up.

---

## Block 2 — Candidate Sources
