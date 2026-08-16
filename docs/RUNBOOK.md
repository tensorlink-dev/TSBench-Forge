# Runbook — the scrape host

Everything the catalog needs in order to keep collecting. Written after
noticing that if this box died, the data would survive but the *operation*
would not: the cron scripts existed in one place with no copy, and the crontab
had no backup that was not on the same disk.

## What runs

Seven scheduled jobs, all defined in [`ops/cron/crontab`](../ops/cron/crontab).
`flock -n` means a tick is skipped if the previous run is still going, so a
slow run never stacks — but see **Tick dropping** below, because that same
behaviour is how the schedule silently broke once already.

| when | job | what it does | writes |
|---|---|---|---|
| every minute | `scrape_fast.sh` | polls only the `* * * * *` sources from `cron.yaml` — feeds that return a single "now" instant, where a skipped minute is unrecoverable. One scraper process for the whole band, not one per id | parquet |
| every 15 min | `scrape_all.sh` | sweeps every due source (8 workers, 12-minute start deadline). ~709s, so it occupies `:00`–`:12` of each quarter. **Scrape only** | parquet |
| `:13,:28,:43,:58` | `sync_hippius.sh` | mirrors `data/` + `sources.yaml` to the Hippius bucket, ~62s | S3 |
| 06:20 daily | `audit_daily.sh` | freshness audit: newest observation vs declared cadence | `logs/audit-<date>.json` |
| 06:43 daily | `pool_report.sh` | eval-pool composition: eligible sources/series per domain, domain × cadence grid, depth backlog | `logs/pool-<date>.json` |
| 07:13 daily | `grind_daily.sh` | finds and wires new sources, commits (never pushes) | `sources.yaml`, `cron.yaml` |
| hourly :07 | `rotate_logs.sh` | the scrape logs are append-only and unbounded; they once reached 315MB, and a full disk looks exactly like a total upstream outage | — |

`grind_daily.sh` **refuses to run when `sources.yaml` or `cron.yaml` have
uncommitted changes**, so leaving the catalog dirty overnight silently costs a
day of grind. It also pins `GRIND_BRANCH=main` and will refuse off that branch —
worth knowing before leaving a feature branch checked out on this host.

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
