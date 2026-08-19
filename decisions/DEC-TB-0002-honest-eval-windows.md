---
id: DEC-TB-0002
type: decision
title: "Eval windows are real or refused, never padded — length becomes variable"
status: active
date: 2026-08-18
tags: [eval-pool, data-quality, sampling, cascade-contract, benchmark-integrity]
revisit_when: "config.PROFILES grows a horizon longer than 48 (MIN_MOTIF_LENGTH is sized so context cannot go negative against the largest horizon, and would need re-deriving); or the share of series served short falls near zero as the cron accumulates contiguous history, at which point a fixed-length contract could be restored; or the near-constant band (>=95% of a window on one value, measured 2.1%) turns out to matter, which needs the fuller battery in source_discovery/quality.py rather than the cheap check in the draw path"
relations: {}
---
`ScrapedLiveSource` no longer guarantees that a motif is exactly the length the
caller asked for. It serves the longest **real** contiguous window up to that
length, refuses the series outright below `MIN_MOTIF_LENGTH` (128), and draws a
replacement so the pool size is unchanged.

**What it replaces.** `_extract_motif` had two paths that *tile-padded*: they
repeated the longest contiguous segment until it filled the requested length.
Measured 2026-08-18 over 1,500 motifs drawn through the real sampler at the
real 320-point geometry:

```
constant (<=1 distinct value)      2.5%
tile-padded (exact repeating period)  15.8%   <- one case: 16 real points x 20
------------------------------------------
degenerate                        18.9%
```

Nearly one eval window in five was trivially forecastable, and the larger part
of it was fabricated. A tiled window is exactly periodic, so any model that
finds the period scores perfectly on data that never happened. The code knew:
both paths returned `None` for the timestamps, with the comment "tiled
timestamps would be fictitious".

**Root cause, and why it is not "too few rows".** Eligibility and extraction
measured different quantities. `_index_available_series` counts *total*
observations across all daily files, deduped by timestamp, against
`min_series_length`. `_extract_motif` builds the window from the *longest
contiguous segment*, after `GAP_FACTOR=8` splits the series at every hole. A
series with 300 rows in 60-point chunks passes the first and cannot satisfy the
second. Padding was how that gap got papered over.

**Why not simply require 320 contiguous.** Measured cost, by longest
contiguous segment across 3,518 eligible series: 85.0% reach 320, 8.5% reach
192-319, 3.3% reach 128-191, 3.2% fall under 128. Requiring the full window
drops 15% of the pool — but it lands almost entirely on two domains, because
sales is 90% monthly and monthly series cluster at a **median segment of 306**,
about a dozen observations short of 320 after roughly 25 years of history.
Sales would go from 236 series to 9 while the equal-weight-per-domain sampler
kept handing it a seventh of every draw: each survivor drawn ~24 times per
1,500-motif eval. Repetition traded for fabrication is not an improvement. A
128 floor keeps 96.8% of series overall and 94% of sales+econ_fin.

**Consequences.**

- Motifs are variable length. They were already held as a list, never stacked,
  so the pool structure is unchanged — but any consumer that assumed uniform
  length must handle 128..N.
- A pool that genuinely runs dry now returns **short** and emits a
  `RuntimeWarning` naming the rejection reasons, instead of silently filling
  with padded windows. Bounded by `MAX_RESAMPLE_ROUNDS`.
- An unreadable frame used to return `zeros(length)` — a constant series
  indistinguishable downstream from a real flat feed. It now refuses.
- Zero-variance windows are rejected in the draw path. The cheap check lives in
  `ScrapedLiveSource._degenerate` rather than importing
  `source_discovery.quality`, because cascade consumes this module without that
  package. Windows sitting on one value for >=95% of their length (a further
  2.1%) are deliberately **kept**: a mostly-flat step function is real data, and
  separating it from a stuck sensor needs the fuller battery.

**No cascade change is needed, and the check that this is so matters.** An
earlier draft of this node asserted the opposite. Cascade does not import
`scraped_source`; it reads the published parquet through its own
`cascade/pool/sources/tsbench_forge.py` and windows it with its own
`PoolBuildConfig(context_length, horizon, min_context)`, whose `prepare_series`
already *refuses* a series shorter than `horizon + min_context` rather than
padding it. The tile-padding defect was only ever in this repo's reader, and so
is the fix.

The consumer that does change is this repo's own leaderboard path, and it was
already safe: `challenges.build_live_challenges` cuts a per-cadence
`config.PROFILES` shape from the motif and shrinks context to fit —
`ctx_len = min(ctx_len, len(motif) - horizon)` — holding the horizon fixed
because the horizon is the task definition. The largest horizon in `PROFILES`
is 48, so a 128-point floor leaves at least 80 points of context in the worst
case and `ctx_len` can never go negative. That is what makes 128 safe as a
floor, and it is the number to re-derive if `PROFILES` ever grows a longer
horizon.

**Two downstream layers had to move with it**, both found by running the real
leaderboard path rather than by reading:

- `ingest.FreshBuffer.sample_meta` slices a sub-window out of a pooled motif
  and assumed every one was at least `motif_len`; a ragged pool made
  `rng.integers(0, len(series) - length + 1)` raise `high <= 0`. It now takes
  `min(length, len(series))`.
- `challenges.build_live_challenges` cuts only the fresh `ctx_len + horizon`
  tail of the motif, and that tail can be flat where the whole motif was not —
  4 of 64 challenges, three of them with a constant truth as well. It now
  redraws past a flat context, bounded by `_FLAT_REDRAW_LIMIT` and driven by
  the challenge's own child rng so the set stays byte-reproducible. Verified:
  two builds of 64 challenges are byte-identical, with 0 flat and 0 periodic
  contexts, against 4 and 4 before.

**Measured end state**, 1,500 motifs through the real sampler at 320:

```
                       before   after
constant                 2.5%    0.0%
tile-padded             15.8%    0.0%
full-length window         --   73.6%
served short but real      --   26.4%   (min 134, median 196)
```

**Not addressed here.** 8.1% of eligible series are fast-cadence feeds
fragmented by gaps — those have the sampling rate to reach 320 and simply have
holes. That is a collection problem, and they return to full-length windows on
their own as the gaps stop.
