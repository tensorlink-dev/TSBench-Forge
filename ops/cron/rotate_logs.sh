#!/usr/bin/env bash
# Cron D (log rotation): keep the scraper's append-only logs bounded.
#
# The scrape crons append to data/_cron_*.log every minute and never truncate.
# Left alone these grow without limit — _cron_all.log had reached 202MB and
# _cron_fast.log 113MB, together more than the entire scraped dataset — and the
# eventual failure mode is a full disk taking down every source at once, which
# looks like a catastrophic upstream outage rather than a housekeeping bug.
#
# One generation is kept (.1) so a crash's context survives the next rotation.
set -uo pipefail

MAX_BYTES=${MAX_BYTES:-52428800}      # 50MB
DATA=/root/TSBench-Forge/src/sources/data

for log in "$DATA"/_cron_*.log /root/cron/logs/audit.log; do
  [ -f "$log" ] || continue
  size=$(stat -c %s "$log" 2>/dev/null || echo 0)
  if [ "$size" -gt "$MAX_BYTES" ]; then
    mv -f "$log" "$log.1"
    : > "$log"
    echo "$(date -u +%FT%TZ) rotated $log at ${size} bytes" \
      >> /root/cron/logs/rotate.log
  fi
done
