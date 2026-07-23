## Block 1 — Gap Analysis

**What the current pool over-covers:** The benchmark is heavily weighted toward daily cadence (56 sources) and sub-hourly real-time feeds (67 sources across sub-min, few-min, half-hour). Nature (45) and econ_fin (31) dominate the domain mix. Within nature, earthquake/volcano/seismic monitoring is saturated across six independent feeds (USGS, INGV, EMSC, GeoNet, BGS, Geoscience Australia), and weather is well-covered by Open-Meteo's multiple endpoints plus SMHI, JMA, and NDBC. Within econ_fin, crypto market data alone accounts for ~15 sources. The pool is strong on live/streaming feeds (22) and open APIs (147), which is excellent for contamination resistance.

**What is thin or absent:** The most critical gaps are the **irregular/event-driven** cells across nearly every domain — 9 of 10 domain×irregular cells have zero sources. Healthcare is the most severely under-served domain for high-frequency data: sub-min, few-min, half-hour, and irregular are all empty (deficit 3 each). This matters because healthcare event streams (outbreaks, adverse events, emergency dispatch) are among the hardest forecasting problems and are essentially absent from standard TSFM pretraining corpora. The web_cloudops/half-hour cell is also empty (deficit 3), and energy/irregular is missing entirely (deficit 3). Transport/irregular and sales/irregular are similarly empty.

**Ranked gaps targeted:**

1. **Irregular/event-driven cells across all domains** (deficit 3 each) — highest priority because they are both scarce and contamination-resistant by construction (future events cannot have been pretrained on).
2. **Healthcare high-frequency cells** (sub-min, few-min, half-hour — all deficit 3) — hardest to fill with public data, but highest discriminative value.
3. **web_cloudops/half-hour** (deficit 3) — no sources at all; internet traffic exchange data could fill this.
4. **transport/half-hour** (deficit 2) and **energy/irregular** (deficit 3) — moderate priority, fillable with government/infrastructure feeds.

Contamination concentration is well-managed (116 low, 54 medium, 6 high), so the primary concern is diversity and gap-filling rather than decontamination.

---

## Block 2 — Candidate Sources
