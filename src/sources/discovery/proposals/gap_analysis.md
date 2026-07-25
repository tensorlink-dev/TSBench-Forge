## Block 1 — Gap Analysis

The current pool of 176 sources is strong on daily and sub-hourly cadences but has critical holes in **irregular/event-driven** cells across nearly every domain, and in **healthcare sub-hourly** cells specifically. The pool is also heavily concentrated in nature (45) and econ_fin (31), while energy (17) and healthcare (19) are thinner.

**Over-represented:** Daily cadence (56 sources) and nature domain (45 sources). Many nature sources are seismic/meteorological feeds that, while live, may overlap in signal structure (event-driven geophysical processes). The econ_fin bucket is crypto-heavy (Binance, Coinbase, Kraken, Gemini, Bitfinex, Polymarket) — a monoculture within the domain.

**Under-represented (ranked by impact):**

1. **Irregular/event-driven cells (deficit 3 each in 7 domains):** This is the single largest gap. Only 1 irregular source exists (ripe_atlas). Irregular data is the gold standard for contamination resistance (future events can't be pretrained on) and for discriminating models (bursty, non-periodic, regime-switching patterns defeat seasonal-naive baselines).

2. **Healthcare sub-hourly (sub-min, few-min, half-hour — deficit 3 each):** Healthcare is the hardest domain to find real-time public data for. Most health surveillance is daily or weekly. The few sub-hourly sources (NYC 311, OpenAQ, NWS heat advisories) are already wired or rejected. This gap hurts because health signals have complex non-stationary dynamics (outbreak waves, policy interventions, seasonal surges).

3. **web_cloudops/half-hour (deficit 3):** Only status-page incident feeds exist at hourly; no half-hourly cybersecurity or infrastructure telemetry source is in rotation.

4. **energy/irregular (deficit 3):** Grid disturbance events, outage events, and emergency actions are absent. Most TSOs are either key-gated (ENTSO-E, PJM) or already proposed/rejected.

5. **Lower-priority gaps:** Monthly/quarterly/yearly cells across all domains have deficits, but these are less valuable because they're easier to find (national statistics offices) and less contamination-resistant (historical data is more likely in pretraining corpora).

**Contamination posture:** Good overall — 116 of 176 sources are low-risk. The main risk is that many "live" feeds (USGS, NWS, Open-Meteo) are well-known public APIs that could plausibly be in future TSFM training corpora. The irregular/event-driven gap is the most contamination-resistant cell to fill.

---

## Block 2 — Candidate Sources
