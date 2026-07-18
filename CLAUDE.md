# Conventions for `TSBench-Forge`

## TensorLink graph

This repo is a spoke of the company strategy graph (`TensorLink-AI/strategy`).
Node ID prefix for this repo: **TB**. This repo currently carries no graph
nodes of its own — its outputs (leaderboard movements, benchmark results) are
distilled into hub evidence nodes (`EV-CO-####`) by the weekly strategy sync.
If real decisions accumulate here, add a `decisions/` dir with `DEC-TB-####`
nodes (frontmatter per `strategy/knowledge/schema.md`). Cross-repo edges use
namespaced targets, e.g. `ME:EV-0021`.
