---
id: DEC-TB-0006
type: decision
title: "Reference models (TimesFM-3, Toto-2 2.5B) join each round from a rented GPU that never sees the truth"
status: active
date: 2026-10-06
tags: [leaderboard, reference-models, lium, benchmark-integrity]
revisit_when: "a reference model's licence forbids publishing its scores; per-round pod cost or failure rate makes the rows unreliable; or Ephemeris serves one of these models (it then moves to the paracast rows)"
relations: {builds-on: "DEC-TB-0001 (paracast-scored rounds)"}
---

Rounds now also score two strong models Ephemeris does not serve: TimesFM-3
(google/timesfm-3.0-pytorch, non-commercial licence, benchmarked not served)
and Toto-2 2.5B (Datadog/Toto-2.0-2.5B). They need a GPU the round workflow
lacks, so:

1. The workflow builds the round's challenges and exports only what a
   forecaster may see: each context and horizon, never the truth
   (`run_paracast_round.py --export-contexts`).
2. `scripts/run_reference_pod.sh` rents a Lium GPU, runs
   `scripts/reference_forecasts.py` (each model zero-shot with its published
   defaults), fetches `<model>.npz`, and always deletes the pod. The pod gets
   no credentials.
3. The round scores the returned quantiles with every other row on one paired
   sample (`--reference-forecasts`). A fingerprint of the contexts travels
   both ways, so forecasts for any other challenge set are refused.

Optional by construction: without the `LIUM_API_KEY` / `LIUM_SSH_KEY` secrets,
or on any pod failure, the round publishes without these rows. They are
class `reference` in `models.json`.
