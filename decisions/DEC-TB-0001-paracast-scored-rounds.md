---
id: DEC-TB-0001
type: decision
title: "Leaderboard rounds are scored through the paracast endpoint, client-side"
status: active
date: 2026-08-14
tags: [leaderboard, paracast, publishing, automation, ensembles, routing]
revisit_when: "a model worth benchmarking is not servable from paracast (then the Lium pod path is the fallback for that model's rows and its results merge on the shared seed); or paracast's /feedback semantics change (per-model vs combined scoring, MASE floor, weight mutation) in a way that alters what the return leg teaches the ensemble; or round cadence needs to exceed daily identity (round_id would need a sub-daily scheme, which changes the published feed contract)"
relations: {}
---
Leaderboard rounds are produced by `scripts/run_paracast_round.py`: forecasts
come from the **paracast** serving endpoint (one `POST /forecast` panel over
the open-weights TSFMs), while **all scoring stays client-side** in
`evaluate`/`model_comparison` on the same seeded challenges and the same
per-source seasonal-naive-relative aggregation as the GPU-pod path. The
`.github/workflows/paracast-round.yml` cron publishes `docs/data/` twice a day,
which is what makes the cascade-frontend leaderboard live — the pod path
(`run_tsfm_comparison_lium.py`) remains for models paracast does not serve, and
merges on the shared challenge seed.

**Why the endpoint, not the pod.** The pod path needs a human, a rented GPU and
four incompatible torch stacks per run; rounds happened when someone had an
afternoon (two rounds ever published). paracast already keeps the panel warm
behind HTTP, so a round becomes a CPU-only loop a scheduled runner can drive.

**Why scoring is NOT delegated to paracast's `/feedback` metrics.** Its MASE
floor is naive-1 (the benchmark's is seasonal), it scores per-model EWMA state
rather than a paired per-challenge sample, and its pending table is a lossy
in-RAM LRU. Client-side scoring keeps paracast-scored rounds arithmetic-
identical to pod-scored ones, so the histories concatenate honestly.

**The return leg is deliberate.** After each round the realized truths are
POSTed to `/feedback`, so the benchmark's evaluation feeds paracast's
inverse-nCRPS EWMA ensemble weights and its router's training log
(`state/requests.jsonl`). `paracast-ensemble` and `paracast-router` compete as
first-class leaderboard rows — the round both *measures* whether combining and
routing beat the best single model, and *trains* the very weights doing the
combining. Neither side trusts the other's arithmetic: inference flows one way,
truths flow back.
