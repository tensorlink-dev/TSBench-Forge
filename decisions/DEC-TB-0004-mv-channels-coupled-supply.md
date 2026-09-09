---
id: DEC-TB-0004
type: decision
title: "mv_channels: forge supplies cross-predictiveness-verified coupled channels; multivariate scoring stays consumer-side"
status: active
date: 2026-09-08
tags: [multivariate, eval-pool, cascade-contract, sources, benchmark-integrity]
revisit_when: "cascade's Phase-2 builder ships and reads mv_channels (retest the arming numbers on their real packing); or the daily-and-faster MV pool share approaches the ~38% margin threshold and the A3 harvest strategy needs re-costing; or the audit's admission bar (min-gain 2%, dual worst-case placebo) produces verdicts the cascade-side reward measurably disagrees with; or GIFT-Eval-parity (~35% multivariate) stops being the reference point for forge's own Track-B benchmark"
relations: {consumed-by: "CA:DEC-CA-0041 (cascade per-variate multivariate scoring; owns the authoritative contract text in cascade docs/EVAL_POOL.md)", builds-on: "DEC-TB-0002 (honest windows), DEC-TB-0003 (jittered mix)"}
---

TSBench-Forge's multivariate role is **supply and labeling only**: tag which
of a source's parquet value columns form a genuinely coupled system, keep
them row-aligned on the shared `timestamp` index, and leave scoring to the
consumer. The interface is one catalog field:

```yaml
mv_channels: [TOTALDEMAND, RRP]   # ordered, <=8, columns of ONE (source, panel_row)
```

The authoritative contract text lives on the consumer side (cascade
`docs/EVAL_POOL.md`, paired with DEC-CA-0041); this node records the forge
half. The scoping clause is load-bearing: channels are columns **within a
single `(source, panel_row)`**, so a coupled group packs to one aligned
(C, L) window and counts as ONE series against every cascade cap (200/source,
call budget, point-estimate unit) and one bootstrap cluster. Cross-source
siblings (the four NSSP pathogen feeds sharing a state panel) can only become
multivariate by re-wiring into columns of one source (SoQL CASE-pivot) — four
separate sources can never be joined into a window by the consumer.

**Admission is cross-predictiveness, not correlation** (`source_discovery/
mv_audit.py`). Channel count is supply, not coupling; |corr| only dedups.
The test: per (source, panel_row), ridge autoregressions per channel —
own-lags vs own+sibling-lags — scored by held-out MAE at h=1 and a direct
h=16 (cascade's reward is a 64-step window; slow couplings don't show at one
step). Two measured failure modes shape the bar:

1. *Shared seasonality masquerades as coupling*: raw h=16 gains admitted
   everything; a bikes/docks feed gained MORE from time-shifted siblings than
   real ones. Every gain is therefore placebo-adjusted against rolled sibling
   copies (autocorrelation kept, alignment destroyed).
2. *No single surrogate offset is fair*: persistent levels (a yield curve)
   leak through a small shift; trends invert across a large shift's wrap
   seam. The subtraction uses the WORST of two offsets, floored at zero —
   deliberately conservative, because a false tag poisons the consumer's
   multivariate reward while a false negative only leaves supply untagged.

Outcome on the full catalog (2026-09-08): 260/3,096 admitted — correlated-
but-not-coupled rejected (treasury's 14 tenors), physical/economic coupling
admitted (demand→price 3/3 AEMO regions, load+wind, buoy meteorology, OHLCV).

**Curation above the statistics** (`mv_tag.py`): a held-out gain can be real
information and still not a coupled signal — IDs that dodge junk regexes (a
parking feed's `sourceelementkey` was the run's top "gain"), static registry
attributes, calendar columns, siblings *derived* from a channel (last_*,
cumulative aggregates, CI bounds, self-forecasts). Deterministic block rules
cut 260 → **202 tagged sources** (nature 62, energy 50, transport 30, web 24,
econ 19, healthcare 14, sales 3). Kept on purpose: hierarchical sets (total +
components) and published forecast-vs-actual pairs — genuine MV structures.
Evidence snapshot: `src/sources/mv_audit_report.json`.

**Evidence accounting** (why counts are in sources, not series): the KOTH
bootstrap clusters by source, so a 200-station coupled panel is one cluster —
channels and panel width buy zero evidence. Arming numbers as of tagging:
202 distinct coupled sources (vs the ~12 N_eff floor and min_clusters≈30
target — cleared everywhere, in every domain); daily-and-faster MV pool share
**11.8%** (1,420/12,060 series) vs the ~38% margin threshold. Closing that
share gap is the A3 harvest: many small coupled *feeds* (TSO demand+price,
multi-pollutant stations, OHLCV venues, sibling-metric feeds), never more
channels or stations on feeds already tagged.

**Everything here is inert until the consumer opts in** — cascade's builder
drops multichannel today, so tags change nothing in the live pool, and this
work shipped on `feat/mv-audit` without touching scraping, cron, or sync.

**Two consumers, deliberately different accounting** (do not "fix" one to
match the other): cascade packs a tagged group into one (C, L) window — one
series slot, one forecast call, source-level clusters. Forge's own benchmark
(Track B, future decision node) will instead expand a group into C linked
per-variate challenges with group-level bootstrap and an MV_FRAC round-mix
knob targeting GIFT-Eval's ~35% multivariate share. Same tag, two readers.

---

## Addendum (2026-09-09, `feat/mv-arming`): aligned with cascade PR #250

Cascade's activation is real and open as their PR #250 (DEC-CA-0041 +
`mv_score_from_block=9064800`, builder `MAX_MV_CHANNELS=8`). Facts that
re-shaped the forge plan:

- **Channel order is load-bearing pre-arming.** Their unarmed builder projects
  a tagged source to `mv_channels[0]` (not the densest column). `mv_order.py`
  reorders every tag so channel 0 IS the densest column — the projection
  becomes a no-op at merge. 189/202 already matched; 11 reordered; 2 flips
  unavoidable (densest column curated out for cause).
- **The scored pool is ~2,800 series, not our 10–22k eligible estimates** —
  the publisher samples with an even domain mix (~400/domain, DEC-CA-0032).
  Consequences: the 10k catalog-order harvest cutoff is not the binding
  constraint; adding series to a domain beyond its allocation buys nothing;
  the arming metric is **within-domain MV proportion**, not raw counts.
- **Their bar:** a group must show ≥3% measured joint-vs-marginal lift, and
  `gain × MV-share ≳ 1%` — hence the ~30% share target.
- **NSSP pivot direction confirmed** by DEC-CA-0041 (by state, across
  conditions; C=4). Wired here as `cdc_nssp_ed_pathogen_mix_daily_by_state`
  (SoQL `sum(case())`, 50 geographies × ~480 d, verified live); the forge
  audit admits it at **+0.342 adjusted gain** — strongest healthcare coupling
  in the catalog. Pathogen siblings stay wired until mv_score arms.

Cascade-ruler share (320-pt, fresh ≤4 d, this box's approximation,
2026-09-09): overall 16.9%; per-domain MV share — nature 33.6%, transport
18.6%, econ_fin 5.2%, energy 4.9%, web_cloudops 3.4%, healthcare 0.4% (→ ~7%
once the pivot has mainnet data), sales 0.2%. The remaining harvest is
**within-domain**: energy (tagged sources are young singles — they ripen on
their own), sales (USDA MPR price+volume columns), healthcare (the pivot +
respiratory panels), web (a few more IXPs). Transport/nature need nothing.
