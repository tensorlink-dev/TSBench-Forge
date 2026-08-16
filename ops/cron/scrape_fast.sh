#!/usr/bin/env bash
# Cron A (fast band): poll ONLY the every-minute live-snapshot feeds and append
# locally. No S3 upload here — scrape_all.sh does the upload every 15 min, which
# sweeps up these accumulated appends too. Kept separate because these sources
# return a single "now" instant per call (see sources.yaml schema.timestamp_field
# == now()/current-state): a skipped minute is unrecoverable, so they alone need
# per-minute polling. The id list is derived from cron.yaml's `* * * * *` group
# so it stays in sync as the catalog changes.
#
# ONE scraper process for the whole band, not one per id. Measured 2026-08-14:
# a single-id run costs 19s, of which 18s is importing pyarrow and httpx and 1s
# is the actual fetch. The old per-id loop therefore took ~13.5 min to get
# through 69 sources (measured from the log: median gap between consecutive
# polls of the same id was 813s), so `flock -n` dropped twelve ticks in every
# thirteen and this band delivered ~13-minute resolution — worse than the
# 15-minute full sweep it exists to improve on, on exactly the sources whose
# missed minutes cannot be backfilled. Paying the import once and fetching with
# --workers brings a pass to well under a minute. --workers never issues two
# concurrent requests to the same host (scraper._host_lock).
set -uo pipefail

REPO=/root/TSBench-Forge
cd "$REPO" || exit 1
set -a; . "$REPO/.env" 2>/dev/null; set +a
source "$REPO/.venv/bin/activate"

# Every-minute source ids, straight from cron.yaml (single source of truth).
IDS=$(python - <<'PY'
import yaml
cfg = yaml.safe_load(open("src/sources/cron.yaml"))
ids = [str(s).split()[0]                      # tolerate stray "[...]" tokens
       for grp in cfg.get("schedules", []) if grp.get("cron") == "* * * * *"
       for s in grp.get("sources", [])]
print(",".join(dict.fromkeys(ids)))           # de-dup, keep order
PY
)

[ -z "$IDS" ] && { echo "$(date -u +%FT%TZ) fast band: no ids in cron.yaml" \
    >> "$REPO/src/sources/data/_cron_fast.log"; exit 1; }

# --deadline-minutes 4: a pass should take ~40s; if a host hangs, stop starting
# new sources rather than letting the run straddle several ticks. Anything not
# started is picked up by the next minute's run.
python src/sources/scraper.py --id "$IDS" --workers 8 --deadline-minutes 4 \
    >> "$REPO/src/sources/data/_cron_fast.log" 2>&1
