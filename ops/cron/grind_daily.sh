#!/usr/bin/env bash
# Cron E (daily source grind): find new sources, steered at whichever domains
# and cadence cells are thinnest, wire the survivors, commit.
#
# Unlike the other cron jobs this one WRITES to the repo: sources.yaml,
# cron.yaml, and a commit. What makes that safe to leave alone:
#
#   - `wire --apply` re-verifies every candidate against the real scraper, and
#     the sweeps now pre-flight through the same gate, so nothing is wired on a
#     sweep's own say-so. Same gate a hand-run wave goes through.
#   - bounded three ways: a wall-clock budget, a per-vein budget, and TARGET
#     sources per day, so a suddenly-productive vein cannot land 200 overnight.
#   - it refuses to run on a dirty catalog, and commits ONLY the two files it
#     owns. An earlier version would have swept an operator's in-progress edits
#     into its own commit.
#   - it pins the branch. Whatever happens to be checked out is not a decision
#     this job should be making at 07:10.
#   - it commits but never pushes. A bad day is a local `git revert`.
#
# Exit codes from --grind mirror the three states worth distinguishing:
#   0  hit the target
#   1  short of target but at or above the floor  -> logged, not alarming
#   2  below the floor                            -> the veins have stopped
#                                                    producing, or something
#                                                    upstream changed shape
#
# Deliberately does NOT take the scrape lock: the sweeps are network-bound and
# hold only a candidate list, so blocking would waste most of the budget.
set -uo pipefail

REPO=${GRIND_REPO:-/root/TSBench-Forge}
BRANCH=${GRIND_BRANCH:-main}
TARGET=${GRIND_TARGET:-5}
MINIMUM=${GRIND_MINIMUM:-2}
MINUTES=${GRIND_MINUTES:-40}
LOG=/root/cron/logs/grind.log
BATCHES="$REPO/src/sources/discovered/grind"
DAY=$(date -u +%F)

cd "$REPO" || exit 1
mkdir -p "$(dirname "$LOG")" "$BATCHES"
say() { printf '%s\n' "$*" >> "$LOG"; }

say "== grind $DAY =="

# --- refuse to run on state we would be committing on someone's behalf -------
on=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
if [ "$on" != "$BRANCH" ]; then
    say "refusing: on branch '$on', expected '$BRANCH' (set GRIND_BRANCH)"
    exit 3
fi
if ! git diff --quiet -- src/sources/sources.yaml src/sources/cron.yaml; then
    say "refusing: catalog already has uncommitted changes; a run now would"
    say "          fold them into its own commit. Resolve, then re-run."
    exit 3
fi

set -a; . "$REPO/.env" 2>/dev/null; set +a
# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"

nice -n 10 python -m source_discovery --grind --apply \
    --grind-target "$TARGET" --grind-minimum "$MINIMUM" \
    --grind-minutes "$MINUTES" --out "$BATCHES" >> "$LOG" 2>&1
rc=$?

# --- commit only what this job owns ------------------------------------------
if git diff --quiet -- src/sources/sources.yaml src/sources/cron.yaml; then
    say "no catalog change (grind exit $rc)"
else
    n=$(git diff --unified=0 -- src/sources/sources.yaml \
        | grep -c '^+- id:' || true)
    git add -- src/sources/sources.yaml src/sources/cron.yaml
    git commit -q -m "Daily grind $DAY: +${n} sources" \
        -m "Wired by ops/cron/grind_daily.sh, steered at the thinnest domains
and cadence cells. Every entry pre-flighted by the sweep and re-verified
against the real scraper by wire --apply before landing.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" \
        && say "committed +${n} (exit $rc)"
fi

case "$rc" in
  0) say "target met" ;;
  1) say "SHORT of target $TARGET but at or above floor $MINIMUM" ;;
  2) say "BELOW floor $MINIMUM — veins may be exhausted or upstream changed" ;;
  *) say "grind failed with exit $rc" ;;
esac

# Keep the last 30 candidate batches; they are the audit trail for what was
# considered, not only what was kept.
ls -1t "$BATCHES"/grind-*.json 2>/dev/null | tail -n +31 | xargs -r rm -f
exit "$rc"
