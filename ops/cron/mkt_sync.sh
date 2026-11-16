#!/usr/bin/env bash
# Mirror the MarketCast bar store (/root/mkt/bars, written by mkt_record.sh)
# to the forge Hippius bucket under marketcast/bars/, then delete local days
# older than 30 that are confirmed in the bucket. See scripts/mkt_sync.py.
# Does NOT take the recorder's lock: the first upload of the 5m/15m/1h banks
# is ~80k files and would block recording for its whole run. It need not:
# every day file is written atomically (temp + replace), and the days the
# recorder still rewrites are re-sent every pass (--hot-days).
set -uo pipefail
REPO=/root/TSBench-Forge
cd "$REPO" || exit 1
set -a; . "$REPO/.env" 2>/dev/null; set +a
nice -n 10 "$REPO/.venv/bin/python" scripts/mkt_sync.py --store /root/mkt/bars --keep-days 30 \
    >> /root/cron/logs/mkt_sync.log 2>&1
