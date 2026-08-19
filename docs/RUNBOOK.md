# Runbook — the scrape host

Everything the catalog needs in order to keep collecting. Written after
noticing that if this box died, the data would survive but the *operation*
would not: the cron scripts existed in one place with no copy, and the crontab
had no backup that was not on the same disk.

## What runs

Eight scheduled jobs, all defined in [`ops/cron/crontab`](../ops/cron/crontab).
`flock -n` means a tick is skipped if the previous run is still going, so a
slow run never stacks — but see **Tick dropping** below, because that same
behaviour is how the schedule silently broke once already.

| when | job | what it does | writes |
|---|---|---|---|
| every minute | `scrape_band.sh '* * * * *'` | the `* * * * *` group — feeds returning a single "now" instant, where a skipped minute is unrecoverable. 78 sources, 46–52s | parquet |
| every 5 min | `scrape_band.sh '*/5 * * * *'` | the `*/5 * * * *` group — sources whose observed cadence equals our poll rate, so we are the bottleneck. 48 sources, ~11s | parquet |
| every 15 min | `scrape_all.sh` | sweeps every due source (8 workers, 12-minute start deadline). ~709s, so it occupies `:00`–`:12` of each quarter. **Scrape only** | parquet |
| `:13,:28,:43,:58` | `sync_hippius.sh` | mirrors `data/` + `sources.yaml` to the Hippius bucket, ~62s | S3 |
| 06:20 daily | `audit_daily.sh` | freshness audit: newest observation vs declared cadence | `logs/audit-<date>.json` |
| 06:43 daily | `pool_report.sh` | eval-pool composition: eligible sources/series per domain, domain × cadence grid, depth backlog | `logs/pool-<date>.json` |
| 07:13 daily | `grind_daily.sh` | finds and wires new sources, commits (never pushes). 40-min budget, now actually enforced | `sources.yaml`, `cron.yaml` |
| hourly :07 | `rotate_logs.sh` | the scrape logs are append-only and unbounded; they once reached 315MB, and a full disk looks exactly like a total upstream outage | — |

`grind_daily.sh` **refuses to run when `sources.yaml` or `cron.yaml` have
uncommitted changes**, so leaving the catalog dirty overnight silently costs a
day of grind. It also pins `GRIND_BRANCH=main` and will refuse off that branch —
worth knowing before leaving a feature branch checked out on this host.

### The grind's budget, and the veins that were eating it

Two things about the daily grind were true until 2026-08-18 and are worth
knowing, because both looked healthy from the outside — it committed every day
and cleared its floor every day.

**Its budgets were advisory.** `GRIND_MINUTES=40` and the 12-minute per-vein cap
were enforced from a callback the *sweep* had to choose to call, so a vein that
went quiet inside one slow request ran straight through both. Measured on the
2026-08-17 run:

| vein | budget | actual | |
|---|---|---|---|
| ckan | 720s | **2047s** | 2.8× over |
| pxweb | 720s | 910s | 1.3× over |
| whole run | 2400s | **3233s** | 54 min against a 40-minute budget |

That is why a job placed at `:13` to start in the gap between sweeps was running
until 08:08, across four sweep windows, on a box with an OOM history. The cap is
now also armed as a `SIGALRM`, which interrupts the blocking syscall itself.

**Ordering demoted bad veins but never removed them.** The back of the queue is
still a turn once the productive vein is mined out for the day, so the leftover
budget drained into veins measured not to pay:

```
ods_enum   24 wired / 1767s  =    74 s per wired source
arcgis      1 wired / 1259s  =  1259 s
ckan        1 wired / 4518s  =  4518 s   <- 75 minutes per source
```

A vein now gets benched once it has had `VEIN_TRIAL_SECONDS` (30 min cumulative)
and still cannot wire a source per `VEIN_BENCH_SECONDS_PER_WIRED` (30 min). The
bench is printed at the top of every run and reported in the JSON. It is
**reversible by deleting that vein's row from
`src/sources/discovered/grind/vein_stats.json`**, which puts it back on trial
from scratch — there is no automatic probation, by design. The board is never
emptied: if every vein would bench, the bench is dropped and logged.

Expect `pxweb` and `sdmx` to bench themselves within a few days on current form.
That is the mechanism working, not a fault — but when it happens, **the grind is
out of productive veins** and the fix is new ones, not a wider budget.

The honest read of the current state: `ods_enum` is the only vein paying, and
its own yield is falling (found 9 → 9 → 7 → 5; wired 7 → 5 → 4 → 2, with 3 of
yesterday's 5 finds duplicates). Benching buys back wasted minutes; it does not
add sources.

### Poll bands, and why `frequency` alone does nothing

A band is the **only** way to poll faster than the 15-minute sweep. `--all`
gates on `is_due()`, which returns True for anything hourly-or-faster, so a
`PT5M` source and a `PT1M` source both get fetched every 15 minutes when the
sweep is all that touches them. Declaring a fast cadence in `sources.yaml` buys
nothing on its own.

`cron.yaml` has described several bands for a long time, but until 2026-08-16
only `* * * * *` was ever executed — every other group fell through to the
sweep. That was invisible because it is not an error: the sources are polled,
just not at the rate they claim. Measured before the fix: of 93 GBFS feeds
declaring `PT5M`, exactly **one** was observing `PT5M`.

Adding a band is two steps, and doing one without the other is the trap:

1. put the ids in a `cron.yaml` group with that `cron:` expression, **and**
2. add a crontab line running `scrape_band.sh '<expr>' <log-tag> <deadline>`.

Groups slower than 15 minutes deliberately have no band — the sweep already
polls them more often than they need.

**Membership must be measured, not declared.** A source belongs in a band only
if *we* are the bottleneck: its observed cadence equals our poll rate. A feed
whose API returns timestamped history is publisher-limited — the sweep already
backfills every point via dedup, and polling it faster buys literally nothing.
When the `*/5` group was first examined it held 159 sources, of which **111 were
publisher-limited**. It now holds the 48 that measure as poll-bound.

### The minute tick, and the YAML loader that nearly ate it

The every-minute band does 46–52s of fetching against a 60s tick, so its setup
cost decides whether it fits. For a long time it did not: `scraper.py` parsed
the 3.7MB catalog with `yaml.safe_load`, the pure-Python loader, on every
invocation. Measured 2026-08-16 — **24.5s with `safe_load`, 4.3s with
`CSafeLoader`, identical parse**. The band was spending 40% of its tick on YAML
before issuing a request, and dropped ~20% of its ticks as a result.

Two false trails are worth recording, because both looked convincing:

- *"The sweep fix caused it."* No — the drop rate was **40%** before the
  sweep/upload split and **20%** after. That change halved it while also
  doubling sweep coverage.
- *"It's the interpreter import; we need a resident worker."* No — imports are
  **1.1s**. That figure came from an old comment rather than a measurement, and
  it pointed at an architectural rewrite when the fix was one line.

With libyaml the band runs ~53s of a 60s tick and drops **0%**. That headroom is
what makes a second band affordable at all — the `*/5` band had to be enabled,
measured, and reverted once before the loader was found, because on the old
loader it pushed the minute band from 20% to 82% dropped.

The tick has perhaps 7s of slack now. Measure before adding anything to it, and
measure the *gap between completions*, not the reported duration.

### Tick dropping

`flock -n` skipping a tick is the intended behaviour for one slow run. It is a
disaster when a job *routinely* overruns its interval, because nothing reports
it: the job keeps succeeding, its own duration looks healthy, and only the gap
between consecutive runs reveals that half of them never happened.

This has now happened twice, both times halving the sampling rate of feeds whose
missed observations cannot be backfilled:

- **2026-08-14, the fast band.** `scrape_fast.sh` shelled out once per id at 19s
  of import cost each, so a pass took ~13.5 min and twelve ticks in thirteen were
  dropped. Fixed by passing the whole id list to one process.
- **2026-08-16, the full sweep.** The Hippius upload ran inside `scrape_all.sh`
  under the sweep's own lock: 709s sweep + 337s serial upload = 17.4 min against
  a 15-minute cron. Every sweep completion for fifteen hours sat at `:27` or
  `:57`. Fixed by parallelizing the upload (337s → 62s) *and* moving it to its
  own job and its own lock, so a slow mirror can never delay a scrape again.

**Diagnose this by diffing consecutive completion times, not durations.** Both
times the durations looked perfectly healthy throughout. Keep `scrape_all.sh`
scrape-only; anything added to it comes out of the same 15-minute budget.

### Lock discipline

Every job has its **own** lock. An earlier version of this runbook claimed
`pool_report.sh` took `scrape_all`'s lock and waited up to 15 minutes for it;
neither was true — it has always had `/tmp/tsforge_pool.lock`, and `flock -n`
does not wait. What actually keeps the two heavy parquet readers apart is the
**clock**, which makes their scheduled minutes load-bearing:

Since the sweep now runs every 15 minutes rather than every 30, it occupies 48
minutes of every hour instead of 24. The only gaps are `:12`–`:15`, `:27`–`:30`,
`:42`–`:45` and `:57`–`:00`, and `pool_report.sh` (`:43`) and `grind_daily.sh`
(`:13`) are placed to start inside them. **Moving either without checking the
sweep windows will overlap them with eight scrape workers**, and this box has an
OOM history — 8 kills, most recently 2026-08-09 at 2.85GB, against 3.9GB total.
Both are `nice -n 10`, which helps with CPU and does nothing for memory.

## Restoring on a fresh box

```bash
git clone https://github.com/tensorlink-dev/TSBench-Forge.git
cd TSBench-Forge
python3 -m venv .venv && .venv/bin/pip install -e '.[all]'   # needs python3-venv
cp .env.example .env      # then fill it in — see "Secrets" below
ops/cron/install.sh       # scripts + crontab; --dry-run to preview
```

Then pull the data back. The bucket is a drop-in copy of the forge dir, so:

```bash
set -a; . .env; set +a
.venv/bin/python src/sources/sync_storage.py --pull
```

Nothing else is needed — the venv, the logs and the parquet tree all rebuild.

## What is and is not backed up

| | where | survives disk loss |
|---|---|---|
| scraped parquet, `sources.yaml` | Hippius bucket, mirrored 4×/hour | **yes** |
| code, cron scripts, crontab | this repo | **yes, once pushed** |
| `.env` (~98 API keys) | the scrape host only | **no** |
| `logs/`, `src/sources/discovered/` | the scrape host only | no — audit trail, not critical |

The bucket is append-only; objects are never deleted, so a bad local state
cannot propagate into a destroyed backup.

## Secrets

`.env` is gitignored and must stay that way. `.env.example` lists every key
**name** the catalog reads, with no values, so a rebuild is a known work-list
rather than an archaeology exercise.

Two things to know when restoring:

- A missing key **disables** the sources that need it rather than failing the
  run, so a partial `.env` still brings most of the catalog up. Fill the
  storage block first: without `HIPPIUS_S3_*` nothing is mirrored off the box.
- `.env` is sourced by cron **last-def-wins**, so append to it; never rewrite
  it in place, or you will silently drop keys that are still in use.

There is currently **no off-box copy of `.env`**. That is the single largest
gap in this runbook. Fixing it needs a decision on where an encrypted copy
lives (bucket + `age`, or a password manager); until then, treat the key list
in `.env.example` as the recovery inventory and expect to re-issue.

## Health checks

```bash
crontab -l                                    # the seven jobs
diff <(crontab -l) ops/cron/crontab           # installed == checked in?
tail -3 src/sources/data/_cron_all.log        # "sweep finished: N ok, 0 failed"
tail -2 src/sources/data/_cron_sync.log       # timestamped, with upload duration
tail -2 /root/cron/logs/audit.log             # ok / stale / unparsed counts
tail -1 /root/cron/logs/pool.log              # eligible sources and series
find src/sources/data -name "$(date -u +%F).parquet" | wc -l   # files written today

# Is the schedule actually being honoured? Gaps must be 15 min, not 30.
grep -a "sweep finished" src/sources/data/_cron_all.log | tail -5
```

A healthy sweep line reads `N ok, 0 failed, 0 skipped` and finishes inside its
15-minute window. "0 failed" counts sources that wrote a file, and retries mask
transient upstream errors — the audit is the honest measure of whether a source
is actually alive.

The last check is the one that catches tick dropping, and it is worth running
after any change to a scheduled job: a sweep can report a perfect `709s` on
every line while half its ticks are being silently discarded.

Drift between `/root/cron` and `ops/cron` is its own failure mode — twice a fix
has been made to the deployed copy and not the checked-in one, leaving
`install.sh` able to revert it on the next restore. The `diff` above is the
cheap guard; run it before trusting the repo as the source of truth.

## Polling politeness: `poll:` and the backfill-window gate

Two rules added 2026-08-19, after the audit showed three publishers pushing
back on our request rate — the day the catalog crossed ~3,000 sources:

- **`poll: <ISO duration>` on a source** overrides every cadence rule and sets
  an explicit contact rate. It gates on *attempts* (a `.last_attempt` marker in
  the source's data dir), not on successful writes — quotas count requests, and
  a source failing under an IP ban would otherwise come due on every sweep and
  keep renewing its own ban. Used for StackExchange (anonymous quota ~300
  req/day per IP; 10 sources at hourly = 240) and as a cheap "resume probe" on
  sources whose publisher currently 403s this host.
- **Backfill-window gate in `is_due()`**: a fast-declared feed whose URL
  re-serves a multi-day rolling window (`{YYYY-MM-DD-14d}`) backfills every
  point via dedup, so sub-hourly fetches buy nothing but load. Such feeds are
  refreshed roughly hourly on a stable per-id stagger (4 phases, 15 min apart)
  so one publisher's whole catalog does not come due on the same sweep.
  Measured before the rule: api.energy-charts.info took ~580 req/hour from 145
  such sources and answered ~200/hour with HTTP 429, each retry stalling a
  sweep worker 12s — which is where the sweep's 12-min deadline was going.

When a publisher blocks us outright (NOAA's www.ndbc.noaa.gov and
services.swpc.noaa.gov 403 this host's IP as of ~2026-07-30; browser headers
change nothing): rewire to a mirror rather than dropping the source.
`schema.rename: {api_name: stored_name}` exists for exactly this — it maps the
mirror's column names (and `_panel_*` keys) back to the original ones so the
series keep their accumulated history. See ndbc_buoy_realtime (CoastWatch
ERDDAP `cwwcNDBCMet`) for the pattern; gfz_hp30_nowcast / silso_sunspot_monthly
/ nasa_donki_solar_flares carry the coverage of the still-blocked SWPC feeds,
which sit on slow `poll:` resume-probes and come back by themselves if the
block lifts.
