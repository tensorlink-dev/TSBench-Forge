# Block 1 — Gap Analysis

The current pool is enormous (~6560 source-series across 7 domains) and contamination is already well-managed (4245 low-risk, 216 high). The pool is **not** dominated by a single host — every domain's top host accounts for <31% of that domain's series. So the primary concern is **frequency holes**, not concentration or monoculture.

The only unfilled gap cells (by the precomputed coverage) are all in the **irregular** band (high-value) plus three low-value slow cells:

- **econ_fin/irregular** — deficit 2, high value. The existing pool covers crypto exchanges, treasury auctions, and SEC filings, but has very few truly irregular event-stream hosts. Corporate funding announcements and business filing events are an entirely different class of signal — boom/bust cycles, macro regime shifts — and no denylist or pretraining corpus maps to these hosts.
- **nature/irregular** — deficit 2, high value. The pool is rich in regular weather/hydrology station feeds but the irregular-event sources already proposed are nearly all USGS/NWS/NOAA/EMSC/Blitzortung variants — all rejected or wired. New, independent event-stream hosts (commercial alert APIs, UN disaster trackers) that are post-cutoff and key-gated are the most viable candidates.
- **energy/irregular** — deficit 1, high value. The existing pool and ALREADY_PROPOSED list already saturate this cell: every major ISO/RTO disturbance feed (EIA OE-417, NERC, PHMSA, ENTSO-E) and every national grid operator has been proposed, wired, or rejected. Finding an additional, truly novel, verifiable energy-irregular host that is not in either list proved very difficult — I am leaving this gap open rather than padding it with a guess.
- **web_cloudops/irregular** — deficit 1, high value. The pool and ALREADY_PROPOSED cover CISA KEV, Shodan, AbuseIPDB, Ransomware.live, BGPStream, and dozens of status-page feeds. An independent, key-gated malware-analysis submission stream from a different vendor is the best remaining candidate.

The three low-value gaps (nature/quarterly, sales/yearly, web_cloudops/quarterly, each deficit 1) I could not fill with a verifiable, novel host that is not in either list. Quarterly/yearly cadences are the hardest to source as live time-series APIs — most are published as static reports or wrapped in bulk downloads, and the few APIs that do exist (OECD, World Bank, UN Comtrade) are all in ALREADY_PROPOSED. I am not going to invent a quarterly web-ops feed.

---

# Block 2 — Candidate Sources
