# Runbook — the scrape host

Everything the catalog needs in order to keep collecting. Written after
noticing that if this box died, the data would survive but the *operation*
would not: the cron scripts existed in one place with no copy, and the crontab
had no backup that was not on the same disk.

## What runs

Five scheduled jobs, all defined in [`ops/cron/crontab`](../ops/cron/crontab).
`flock -n` means a tick is skipped if the previous run is still going, so a
slow run never stacks.

| when | job | what it does | writes |
|---|---|---|---|
| every minute | `scrape_fast.sh` | polls only the `* * * * *` sources from `cron.yaml` — feeds that return a single "now" instant, where a skipped minute is unrecoverable | parquet |
| every 15 min | `scrape_all.sh` | sweeps every due source (8 workers, 12-minute start deadline), then mirrors `data/` + `sources.yaml` to the Hippius bucket | parquet, S3 |
| 06:20 daily | `audit_daily.sh` | freshness audit: newest observation vs declared cadence | `logs/audit-<date>.json` |
| 06:40 daily | `pool_report.sh` | eval-pool composition: eligible sources/series per domain, domain × cadence grid, depth backlog | `logs/pool-<date>.json` |
| hourly :07 | `rotate_logs.sh` | the scrape logs are append-only and unbounded; they once reached 315MB, and a full disk looks exactly like a total upstream outage | — |

`grind_daily.sh` is checked in but **not scheduled** — it writes to the catalog
and commits, so enabling it is a deliberate act. Add its line to
`ops/cron/crontab` when you want it.

### Lock discipline

`pool_report.sh` takes **`scrape_all`'s** lock rather than its own, because
indexing every parquet costs ~200s and ~1.1GB and must not run alongside eight
scrape workers on a 4GB box. It waits up to 15 minutes and then skips the day:
a missed report beats an OOM that also stops the scraper.

`grind_daily.sh` deliberately does *not* take that lock — its sweeps are
network-bound and hold only a candidate list, so blocking would waste the
budget for no memory benefit.

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
| scraped parquet, `sources.yaml` | Hippius bucket, mirrored every 15 min | **yes** |
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
crontab -l                                    # the five jobs
tail -3 src/sources/data/_cron_all.log        # "sweep finished: N ok, 0 failed"
tail -2 /root/cron/logs/audit.log             # ok / stale / unparsed counts
tail -1 /root/cron/logs/pool.log              # eligible sources and series
find src/sources/data -name "$(date -u +%F).parquet" | wc -l   # files written today
```

A healthy sweep line reads `N ok, 0 failed, 0 skipped` and finishes inside its
15-minute window. "0 failed" counts sources that wrote a file, and retries mask
transient upstream errors — the audit is the honest measure of whether a source
is actually alive.
