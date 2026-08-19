#!/usr/bin/env bash
# Cron A (full sweep): scrape every source once. Rolling-window feeds (binance
# klines, event feeds, hourly/daily files) are fully backfilled by dedup on each
# poll, so a 15-min cadence loses nothing for them; the live-snapshot feeds are
# also swept here, on top of their per-minute appends from scrape_fast.sh.
#
# The Hippius mirror used to run here, inline. It is now Cron C
# (sync_hippius.sh) on its own lock: the 337s serial upload plus this 709s sweep
# came to 17.4 min against a 15-minute cron, so `flock -n` dropped every other
# tick and the entire catalog was sampled at half its configured cadence. Keep
# this script scrape-only -- anything added below eats the same headroom.
set -uo pipefail

REPO=/root/TSBench-Forge
cd "$REPO" || exit 1
set -a; . "$REPO/.env" 2>/dev/null; set +a
source "$REPO/.venv/bin/activate"

# Scrape everything. Transient upstream failures are logged per-source to
# data/<id>/_errors.log; don't abort the run — still upload what succeeded.
#
# --workers: the sweep is I/O-bound (waiting on ~600 due sources per tick), and
# scraper.py takes a lock per HOSTNAME, so concurrency never means two requests
# at one host. Single-threaded this took ~28 min against a 15-min cron, so flock
# was silently dropping every other tick and the sweep effectively ran half as
# often as configured. Workers are set against I/O, not the 2 cores.
#
# Tried 12 on 2026-08-07 and REVERTED: it is not faster on this box. Measured
# throughput, sources/sec: 8 workers 1.37 and 1.65, 12 workers 1.24 and 1.44,
# and peak RSS went 695MB -> 923MB. The sweep is not purely I/O-bound -- JSON
# and parquet parsing contend for the two cores -- so past ~8 concurrent
# fetches more workers buy nothing. Neither setting truncated (0 skipped).
# If sweeps approach the deadline again, raise --deadline-minutes or split the
# cadence tiers rather than adding workers.
# --deadline-minutes: stop STARTING sources at 12 min so the run always exits
# before the next tick; cadence tracking means anything skipped is simply due
# again next sweep. 2026-08-19: the catalog's +625 sources (08-13/14) had the
# sweep hitting this deadline on every pass with ~34 sources unstarted.
# Considered raising to 14 and DIDN'T: the daily jobs at :43/:13 sit in gaps
# that assume sweeps end at :12, and this box OOMs when heavy readers overlap.
# Instead the backfill-window gate in is_due() (same date) took ~160 fetches
# and the energy-charts 429 retry stalls out of each pass. If the monitor
# still reports deadline-hit after that, raise this to 14 AND move
# pool_report/grind_daily out of the :30-:44 / :00-:14 windows together.
python src/sources/scraper.py --all --workers 8 --deadline-minutes 12 \
    >> "$REPO/src/sources/data/_cron_all.log" 2>&1
