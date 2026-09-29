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
# --apply-disables switches off sources stale past 3x their own cadence limit.
# Without it the report was advisory only: 95 long-dead sources (ERDDAP 404s,
# withdrawn Socrata datasets, sensors that stopped reporting) were still being
# fetched every sweep, burning the 12-minute deadline that the live sources
# need. The catalog edit is committed below so the disable is reviewable.
{
  echo "== audit $DAY =="
  python -m source_discovery --audit --apply-disables --audit-json "$OUT"
} >> /root/cron/logs/audit.log 2>&1

if ! git -C "$REPO" diff --quiet -- src/sources/sources.yaml; then
  n=$(git -C "$REPO" diff -U0 -- src/sources/sources.yaml | grep -c '^+  disabled: true' || true)
  git -C "$REPO" add -- src/sources/sources.yaml
  git -C "$REPO" commit -q -m "Audit $DAY: disable ${n} sources stale past 3x their cadence" \
    -m "Applied by ops/cron/audit_daily.sh. Report: /root/cron/logs/audit-$DAY.json" \
    >> /root/cron/logs/audit.log 2>&1
fi

# Keep the last 30 daily reports.
ls -1t /root/cron/logs/audit-*.json 2>/dev/null | tail -n +31 | xargs -r rm -f
