#!/usr/bin/env bash
# Cron A (full sweep): scrape every source once, then mirror to Hippius S3.
# Rolling-window feeds (binance klines, event feeds, hourly/daily files) are
# fully backfilled by dedup on each poll, so a 15-min cadence loses nothing for
# them; the live-snapshot feeds are also swept here (their per-minute appends
# from scrape_fast.sh get uploaded). Upload is size-diff based and append-only
# (see sync_storage.py), so re-running is idempotent.
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
# before the next tick with time left for the upload; cadence tracking means
# anything skipped is simply due again next sweep.
python src/sources/scraper.py --all --workers 8 --deadline-minutes 12 \
    >> "$REPO/src/sources/data/_cron_all.log" 2>&1

# Mirror data/ + sources.yaml to the Hippius bucket (tsbench-forge-sources).
python src/sources/sync_storage.py >> "$REPO/src/sources/data/_cron_sync.log" 2>&1
