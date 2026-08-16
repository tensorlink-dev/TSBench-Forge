#!/usr/bin/env bash
# Cron C (mirror): push data/ + sources.yaml to the Hippius bucket.
#
# Split out of scrape_all.sh on 2026-08-16. It used to run inside that script,
# under the same flock, and that was silently halving the whole catalog's
# sampling rate: the sweep took 709s and the serial upload another 337s, so the
# pair ran 17.4 min against a 15-minute cron and `flock -n` dropped every
# intervening tick. Every sweep completion for 15 hours straight landed at :27
# or :57 -- a stable 30-minute cycle where 15 was configured.
#
# On its own lock the two can overlap, which is safe in both directions:
# parquet writes are atomic (temp file + os.replace, see scraper.py) so an
# upload never reads a torn file, and the sync is size-diff idempotent and
# append-only, so a file mirrored mid-sweep is simply re-sent next pass when it
# has grown. If a sync overruns it now drops only its own tick, not a scrape.
#
# Offset to :13/:28/:43/:58 -- roughly a minute after the sweep that started at
# :00/:15/:30/:45 finishes -- so the freshest rows go up promptly.
set -uo pipefail

REPO=/root/TSBench-Forge
cd "$REPO" || exit 1
set -a; . "$REPO/.env" 2>/dev/null; set +a
source "$REPO/.venv/bin/activate"

python src/sources/sync_storage.py >> "$REPO/src/sources/data/_cron_sync.log" 2>&1
