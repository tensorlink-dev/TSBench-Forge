#!/usr/bin/env bash
# Cron E (daily source grind): find a few new sources, steered at whichever
# domains and cadence cells are thinnest, wire the survivors, commit.
#
# Unlike the other cron jobs this one WRITES to the repo: sources.yaml,
# cron.yaml, and a commit. Three things make that safe enough to leave alone:
#
#   - `wire --apply` re-verifies every candidate against the real scraper
#     before it reaches sources.yaml. Nothing is wired on the strength of a
#     sweep's own say-so, which is the same gate a hand-run wave goes through.
#   - the run is bounded: a wall-clock budget, a per-vein budget, and a cap of
#     TARGET sources per day, so a productive vein cannot land 200 overnight.
#   - it commits but never pushes. A bad day is a local `git revert`, and the
#     repo already carries unpushed history that is the operator's to send.
#
# Deliberately does NOT take the scrape lock. The sweeps are network-bound and
# hold only a candidate list, so unlike the pool report they can share the box
# with scrape_all; blocking on that lock would waste most of the budget.
set -uo pipefail

REPO=/root/TSBench-Forge
cd "$REPO" || exit 1
set -a; . "$REPO/.env" 2>/dev/null; set +a
source "$REPO/.venv/bin/activate"

DAY=$(date -u +%F)
TARGET=${GRIND_TARGET:-4}
MINUTES=${GRIND_MINUTES:-40}
BATCHES="$REPO/src/sources/discovered/grind"
LOG=/root/cron/logs/grind.log

{
  echo "== grind $DAY =="
  nice -n 10 python -m source_discovery --grind --apply \
    --grind-target "$TARGET" --grind-minutes "$MINUTES" --out "$BATCHES"
} >> "$LOG" 2>&1
rc=$?

# Commit only if the grind actually changed the catalog. `git diff --quiet`
# returns 1 when there ARE changes, which is the branch we want.
cd "$REPO" || exit 1
if ! git diff --quiet -- src/sources/sources.yaml src/sources/cron.yaml; then
    n=$(git diff --unified=0 -- src/sources/sources.yaml \
        | grep -c '^+- id:' || true)
    # Only the catalog is committed; src/sources/discovered/ is gitignored, so
    # the batch files stay on disk as a local audit trail of what was
    # considered rather than only what was kept.
    git add src/sources/sources.yaml src/sources/cron.yaml
    git commit -q -m "Daily grind $DAY: +${n} sources" \
        -m "Wired by /root/cron/grind_daily.sh, steered at the thinnest
domains and cadence cells. Every entry re-verified against the real
scraper by wire --apply before landing.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" \
        && echo "committed +${n}" >> "$LOG"
else
    echo "no catalog change (grind exit $rc)" >> "$LOG"
fi

# Keep the last 30 candidate batches; they are the audit trail for what was
# considered, not just what was wired.
ls -1t "$BATCHES"/grind-*.json 2>/dev/null | tail -n +31 | xargs -r rm -f
