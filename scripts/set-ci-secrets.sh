#!/usr/bin/env bash
# Populate the repo secrets the scrape workflow needs.
#
# Without these the workflow is a no-op that reports success: the "Pull today's
# archive" and "Mirror to Hippius" steps are gated on HIPPIUS_S3_ACCESS_KEY
# being non-empty, so with no secrets it scrapes ~1000 sources into an ephemeral
# runner and discards every row when the runner is destroyed.
#
# Values are piped from .env on stdin — never passed as arguments, so they do
# not land in the process list or your shell history.
#
# Prerequisite: gh auth login   (interactive; run it yourself first)

set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { echo "no .env in $(pwd)" >&2; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "run 'gh auth login' first" >&2; exit 1; }

set -a; . ./.env; set +a

printf '%s' "${HIPPIUS_S3_ACCESS_KEY:?missing in .env}" | gh secret set HIPPIUS_S3_ACCESS_KEY
printf '%s' "${HIPPIUS_S3_SECRET_KEY:?missing in .env}" | gh secret set HIPPIUS_S3_SECRET_KEY

# The whole .env as one secret: the workflow writes it back out and sources it,
# so every `auth: env:VAR` header and {ENVVAR} URL placeholder resolves. Adding a
# keyed source later means re-running this, not editing the workflow.
gh secret set SCRAPER_ENV < .env

echo
echo "secrets now set on the repo:"
gh secret list
echo
echo "verify with a real run:  gh workflow run 'Scrape live sources'"
echo "then confirm the 'Mirror to Hippius' step RUNS instead of being skipped."
