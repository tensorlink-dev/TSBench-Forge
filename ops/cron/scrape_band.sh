#!/usr/bin/env bash
# Cron B (poll bands): poll ONE cron.yaml band and append locally. No S3 upload
# here -- sync_hippius.sh mirrors every 15 min and sweeps up these appends.
#
#   scrape_band.sh '* * * * *'    fast   4
#   scrape_band.sh '*/5 * * * *'  5min   3
#
# Replaces scrape_fast.sh (2026-08-16), which was this same logic with the band
# hardcoded. A second copy for the 5-minute band would have been the third
# near-identical script on this box, and a drifting copy of a cron script is
# precisely the failure that let a live fix sit unshipped for two days.
#
# WHY BANDS EXIST AT ALL. scraper.py --all gates on is_due(), which returns True
# for anything hourly-or-faster, so the 15-minute sweep polls a PT5M source and
# a PT1M source at exactly the same rate: every 15 minutes. Cadence finer than
# the sweep interval can only come from a band that runs more often. cron.yaml
# has described these bands for a long time, but only `* * * * *` was ever
# executed -- the `*/5` group's 146 sources were declared at five minutes and
# delivered at fifteen. Measured 2026-08-16: of 93 GBFS feeds declaring PT5M,
# exactly one was observing PT5M.
#
# ONE scraper process for the whole band, not one per id. Measured 2026-08-14:
# a single-id run costs 19s, of which 18s is importing pyarrow and httpx and 1s
# is the actual fetch. The old per-id loop took ~13.5 min for 69 sources, so
# `flock -n` dropped twelve ticks in thirteen and the band delivered ~13-minute
# resolution -- worse than the sweep it exists to improve on. Paying the import
# once and fetching with --workers brings a pass to well under a minute.
# --workers never issues two concurrent requests to one host (scraper._host_lock).
set -uo pipefail

BAND=${1:?usage: scrape_band.sh '<cron expr>' <log-suffix> <deadline-minutes>}
TAG=${2:?missing log suffix}
DEADLINE=${3:-4}

REPO=/root/TSBench-Forge
cd "$REPO" || exit 1
set -a; . "$REPO/.env" 2>/dev/null; set +a
source "$REPO/.venv/bin/activate"

LOG="$REPO/src/sources/data/_cron_${TAG}.log"

# Band member ids, straight from cron.yaml (single source of truth).
IDS=$(BAND="$BAND" python - <<'PY'
import os, yaml
band = os.environ["BAND"]
cfg = yaml.safe_load(open("src/sources/cron.yaml"))
ids = [str(s).split()[0]                      # tolerate stray "[...]" tokens
       for grp in cfg.get("schedules", []) if grp.get("cron") == band
       for s in grp.get("sources", [])]
print(",".join(dict.fromkeys(ids)))           # de-dup, keep order
PY
)

[ -z "$IDS" ] && { echo "$(date -u +%FT%TZ) band '$BAND': no ids in cron.yaml" \
    >> "$LOG"; exit 1; }

# --deadline-minutes: if a host hangs, stop starting new sources rather than let
# the run straddle several ticks. Anything not started is picked up next tick.
python src/sources/scraper.py --id "$IDS" --workers 8 \
    --deadline-minutes "$DEADLINE" >> "$LOG" 2>&1
