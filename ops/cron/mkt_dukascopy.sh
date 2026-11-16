#!/usr/bin/env bash
# MarketCast FX beyond Hyperliquid: Dukascopy 1m bars for all 28 majors/crosses
# (scripts/mkt_dukascopy.py), then mirror to Hippius marketcast/dukascopy/.
#
# Each hourly run fetches yesterday for any pair still missing it, then walks
# the backfill back (2 years) at most 300 day-files per run, ~2 s apart. The
# feed is flaky (it answers 503 / times out in bursts), so a run that hits 5
# consecutive failures just stops; the next hour resumes from what is on disk.
# Not pruned locally (~1.2 GB for 2 years): local files are what tell the
# recorder a day is done.
set -uo pipefail
REPO=/root/TSBench-Forge
cd "$REPO" || exit 1
set -a; . "$REPO/.env" 2>/dev/null; set +a
STORE=/root/mkt/dukascopy
{
  nice -n 10 "$REPO/.venv/bin/python" -I scripts/mkt_dukascopy.py --store "$STORE" \
      --days-back 730 --max-requests 300 --pause 2
  nice -n 10 "$REPO/.venv/bin/python" scripts/mkt_sync.py --store "$STORE" \
      --prefix marketcast/dukascopy/ --hot-days 0 --keep-days 100000
} >> /root/cron/logs/mkt_dukascopy.log 2>&1
