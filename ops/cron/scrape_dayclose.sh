#!/usr/bin/env bash
# Close yesterday for every sub-daily feed whose URL asks for today only
# (scraper.py --day-close). The day's last regular fetch lands minutes before
# midnight and the next one asks for the new day, so without this every such
# series ends each UTC day in a 5-25 min hole, and cascade's reader splits the
# series there (2026-11-23: promlabs 10s feeds held 454k points but served the
# pool 4.6k). Runs at 00:20 UTC, before the 00:28 mirror sync.
set -uo pipefail
REPO=/root/TSBench-Forge
cd "$REPO" || exit 1
set -a; . "$REPO/.env" 2>/dev/null; set +a
source "$REPO/.venv/bin/activate"
nice -n 5 python src/sources/scraper.py --day-close 1 --workers 4 --deadline-minutes 30 \
    >> "$REPO/src/sources/data/_cron_dayclose.log" 2>&1
