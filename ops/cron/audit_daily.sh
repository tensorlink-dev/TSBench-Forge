#!/usr/bin/env bash
# Cron C (daily freshness audit): compare each source's newest OBSERVATION on
# disk against its declared cadence and log the stale/nodata/unparsed lists.
# Report-only — disabling a source is a deliberate act (--apply-disables),
# never done from cron. Findings land in /root/cron/logs/audit-<date>.json;
# the summary is appended to audit.log for quick grepping.
set -uo pipefail

REPO=/root/TSBench-Forge
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DAY=$(date -u +%F)
OUT=/root/cron/logs/audit-$DAY.json
{
  echo "== audit $DAY =="
  python -m source_discovery --audit --audit-json "$OUT"
} >> /root/cron/logs/audit.log 2>&1

# Keep the last 30 daily reports.
ls -1t /root/cron/logs/audit-*.json 2>/dev/null | tail -n +31 | xargs -r rm -f
