#!/usr/bin/env bash
# Cron A (fast band): poll ONLY the every-minute live-snapshot feeds and append
# locally. No S3 upload here — scrape_all.sh does the upload every 15 min, which
# sweeps up these accumulated appends too. Kept separate because these sources
# return a single "now" instant per call (see sources.yaml schema.timestamp_field
# == now()/current-state): a skipped minute is unrecoverable, so they alone need
# per-minute polling. The id list is derived from cron.yaml's `* * * * *` group
# so it stays in sync as the catalog changes.
set -uo pipefail

REPO=/root/TSBench-Forge
cd "$REPO" || exit 1
set -a; . "$REPO/.env" 2>/dev/null; set +a
source "$REPO/.venv/bin/activate"

# Every-minute source ids, straight from cron.yaml (single source of truth).
mapfile -t IDS < <(python - <<'PY'
import yaml
cfg = yaml.safe_load(open("src/sources/cron.yaml"))
for grp in cfg.get("schedules", []):
    if grp.get("cron") == "* * * * *":
        for s in grp.get("sources", []):
            print(str(s).split()[0])  # tolerate stray "[...]" tokens
PY
)

for id in "${IDS[@]}"; do
    python src/sources/scraper.py --id "$id" >> "$REPO/src/sources/data/_cron_fast.log" 2>&1
done
