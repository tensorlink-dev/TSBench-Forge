#!/usr/bin/env bash
# Reinstall the scrape host's scheduled jobs on a fresh box.
#
# The jobs used to live only in /root/cron with no copy anywhere, so losing the
# disk meant rebuilding the schedule from memory. They are checked in here
# instead; this script is the whole restore step.
#
#   ops/cron/install.sh              # install to the default /root/cron
#   CRON_DIR=/opt/cron ops/cron/install.sh
#   ops/cron/install.sh --dry-run    # show what would change
#
# Does NOT touch .env — the API keys are not in this repo and never will be.
# See docs/RUNBOOK.md for that part of the restore.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CRON_DIR=${CRON_DIR:-/root/cron}
DRY=${1:-}

say() { printf '%s\n' "$*"; }

if [[ ! -f "$REPO/.env" ]]; then
    say "WARNING: $REPO/.env is missing. Sources needing a key will be skipped"
    say "         and the Hippius mirror will not run. See docs/RUNBOOK.md."
fi
if [[ ! -x "$REPO/.venv/bin/python" ]]; then
    say "WARNING: $REPO/.venv is missing. Create it before the first tick:"
    say "         python3 -m venv .venv && .venv/bin/pip install -e '.[all]'"
fi

# The checked-in crontab names /root/cron because that is where it runs today.
# Retarget it when installing elsewhere, or the scripts land in one place and
# the schedule points at another.
render_crontab() {
    sed "s#/root/cron/#${CRON_DIR%/}/#g" "$REPO/ops/cron/crontab"
}

# Everything except this installer, which has no business being scheduled.
scripts() {
    find "$REPO/ops/cron" -maxdepth 1 -name '*.sh' ! -name 'install.sh' | sort
}

if [[ "$DRY" == "--dry-run" ]]; then
    say "would install scripts to $CRON_DIR:"
    scripts | sed 's#.*/#  #'
    say "would install this crontab:"
    render_crontab | sed 's/^/  /'
    exit 0
fi

mkdir -p "$CRON_DIR/logs"
# shellcheck disable=SC2046  # paths are repo-controlled, no spaces
install -m 0755 $(scripts) "$CRON_DIR/"
say "installed scripts to $CRON_DIR"

# Never clobber an existing crontab silently: keep a timestamped copy first.
if crontab -l >/dev/null 2>&1; then
    backup="$CRON_DIR/crontab.backup.$(date -u +%F-%H%M%S)"
    crontab -l > "$backup"
    say "backed up existing crontab to $backup"
fi
render_crontab | crontab -
say "installed crontab:"
crontab -l | grep -vE '^\s*(#|$)' | sed 's/^/  /'

say ""
say "Restore is not complete until .env is in place — see docs/RUNBOOK.md."
