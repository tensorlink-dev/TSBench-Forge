# Conventions for `TSBench-Forge`

## TensorLink graph

This repo is a spoke of the company strategy graph (`TensorLink-AI/strategy`).
Node ID prefix for this repo: **TB**. Decisions live in `decisions/` as
`DEC-TB-####` nodes (frontmatter per `strategy/knowledge/schema.md`); outputs
(leaderboard movements, benchmark results) are distilled into hub evidence
nodes (`EV-CO-####`) by the weekly strategy sync. Cross-repo edges use
namespaced targets, e.g. `ME:EV-0021`.

## Design decisions

- **DEC-TB-0001** — Leaderboard rounds are scored through the paracast
  endpoint (inference only; scoring stays client-side on the shared seed) and
  published on a cron, making the public leaderboard live. Truths are POSTed
  back to `/feedback` so paracast's ensemble weights and router learn from
  every round. (`decisions/DEC-TB-0001-paracast-scored-rounds.md`)

- **DEC-TB-0002** — Eval windows are real or refused, never padded. The sampler
  serves the longest contiguous window up to the requested length, refuses
  below `MIN_MOTIF_LENGTH` (128) and redraws, so motif length is now
  **variable**. Replaces tile-padding, which was fabricating 15.8% of drawn
  windows. Self-contained: cascade reads the published parquet through its own
  pool builder and needs no change.
  (`decisions/DEC-TB-0002-honest-eval-windows.md`)
