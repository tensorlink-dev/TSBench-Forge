---
id: DEC-TB-0005
type: decision
title: "The paracast-ensemble row requests combine=mixture, what paracast serves by default"
status: active
date: 2026-10-06
tags: [leaderboard, paracast, ensemble, benchmark-integrity]
revisit_when: "paracast changes its default combine method, or a round shows mixture losing to vincentize on the same challenges"
relations: {builds-on: "DEC-TB-0001 (paracast-scored rounds)", evidence: "paracast experiments/router/RESULTS.md iteration 25 (full-panel verification)"}
---

The `paracast-ensemble` leaderboard row now asks paracast for
`combine="mixture"`. Until this change the client hard-coded
`combine="vincentize"` (equal-weight quantile averaging), overriding paracast's
own default. Since paracast 0.7.x that default is the accuracy-weighted
mixture, so the published row measured a combination no customer receives.

Measured on the same challenges, vincentize is the worst ensemble on every
corpus paracast was verified on: on a Weir round (1,280 challenges) it trails
the shipped weighted mixture by 1.5% CRPS / 2.0% MASE, and by up to 16% MASE on
fev-bench (paracast `experiments/router/results_blend/weir_verify.md`,
`ens_*.md`). The row therefore understated the product.

**Discontinuity.** Rounds from 2026-10-06 onward score the mixture; earlier
rounds scored vincentize. The same date is also paracast's switch to image
0.7.5 (`patchtst-fm-r2` replaces `patchtst-fm-r1`, accuracy weighting), so the
ensemble row changes method and membership at once. Readers of the history,
including the Weir dashboard, should mark the break rather than read across it.

Only the client default changes; `ModelSpec.ensemble(combine=...)` still
accepts `"vincentize"` for comparison runs.
