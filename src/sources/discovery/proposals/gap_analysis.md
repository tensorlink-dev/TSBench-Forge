### Block 1 — Gap analysis

The current pool is heavily skewed toward high-frequency structured APIs (147 open APIs) and daily cadences (56 sources). While sub-minute and few-minute bands are well-populated, the **irregular/event-driven** and **half-hourly** bands are dangerously thin or entirely absent across nearly every domain. Specifically, the benchmark has **zero sources** for `econ_fin`, `energy`, `nature`, `sales`, and `transport` in the irregular cadence band, and a massive deficit (0–1 sources vs. a target of 3) in `healthcare` across all high-frequency and irregular cells. 

Because irregular event streams and post-cutoff data are the most contamination-resistant sources available, filling these gaps is the absolute top priority. Models cannot memorize the exact arrival times of future weather warnings, grid outages, or transit disruptions. The current over-representation of scheduled, predictable daily data (like treasury yields or daily weather observations) risks making the benchmark an easy target for seasonal-naive baselines. I am prioritizing high-value, irregular, and half-hourly cells that introduce non-deterministic event timing and regime shifts.

### Block 2 — Candidate sources
