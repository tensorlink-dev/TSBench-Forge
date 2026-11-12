#!/usr/bin/env bash
# Mirror the MarketCast bar store (/root/mkt/bars, written by mkt_record.sh)
# to the forge Hippius bucket under marketcast/bars/, then delete local days
# older than 30 that are confirmed in the bucket. See scripts/mkt_sync.py.
# Waits on the recorder's lock so it never uploads a pass half-written.
set -uo pipefail
REPO=/root/TSBench-Forge
cd "$REPO" || exit 1
set -a; . "$REPO/.env" 2>/dev/null; set +a
flock -w 600 /tmp/tsforge_mkt_record.lock \
    nice -n 10 "$REPO/.venv/bin/python" scripts/mkt_sync.py --store /root/mkt/bars --keep-days 30 \
    >> /root/cron/logs/mkt_sync.log 2>&1
