### Block 1 — Gap analysis

The current pool of 176 sources is heavily concentrated in high-frequency (sub-minute, few-minute, hourly) and daily cadences, with excellent coverage of nature, finance, and web infrastructure domains. However, the pool suffers from severe structural gaps in **irregular/event-driven** and **low-frequency (weekly, monthly, quarterly, yearly)** bands across almost all domains. 

The most critical high-value gaps are the complete absence of irregular event streams in econ_fin, energy, nature, sales, and transport. Irregular series are highly prized because they lack the trivial periodicity that lets naive baselines cheat, and their event-driven nature makes them highly resistant to pretraining contamination. Furthermore, the healthcare domain is completely missing sub-minute, few-minute, and half-hourly cadences, indicating an over-reliance on daily/weekly public health reporting rather than operational or clinical feeds.

To address these gaps without re-proposing previously rejected sources, we must pivot to regional, industrial, or newly available open-data portals. For irregular econ_fin data, we target corporate action and auction event streams. For irregular nature data, we target specialized environmental event feeds (glacial lake outbursts, meteor detections). For healthcare high-frequency data, we target regional clinical and emergency resource feeds. For the low-frequency gaps, we target niche economic and environmental annual/quarterly releases that are too obscure or newly structured to be in standard pretraining corpora.

---

### Block 2 — Candidate sources
