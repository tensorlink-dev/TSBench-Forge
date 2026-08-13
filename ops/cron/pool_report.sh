#!/usr/bin/env bash
# Cron D (daily eval-pool composition): what the eval would actually draw from
# today -- eligible sources/series per domain, the domain x cadence grid, the
# backlog still short of min_series_length, and a fixed-seed sample draw.
#
# Distinct from the audit: the audit asks "is each source still fresh", this
# asks "what is the SHAPE of the pool". Both are report-only.
#
# Two scheduling constraints, both learned the hard way:
#   - it indexes every parquet in the tree (~200s, ~1.1GB peak on this 4GB box),
#     so it must not run alongside scrape_all's 8 workers. It takes the SAME
#     lock as scrape_all and waits for it, rather than racing it.
#   - it runs after the 06:20 audit so the two heavy readers never overlap.
set -uo pipefail

REPO=/root/TSBench-Forge
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DAY=$(date -u +%F)
OUT=/root/cron/logs/pool-$DAY.json

# -w 900: wait up to 15min for the in-flight scrape to finish, then give up
# rather than pile on. Skipping a day is strictly better than an OOM that also
# takes out the scraper.
{
  echo "== pool $DAY =="
  flock -w 900 /tmp/tsforge_scrape_all.lock \
    nice -n 10 python -m source_discovery --pool-report --pool-report-json "$OUT" \
    || echo "pool report skipped or failed (exit $?)"
} >> /root/cron/logs/pool.log 2>&1

# Keep the last 30 daily reports, matching the audit's retention.
ls -1t /root/cron/logs/pool-*.json 2>/dev/null | tail -n +31 | xargs -r rm -f
