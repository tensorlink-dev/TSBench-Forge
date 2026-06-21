#!/bin/bash
# SessionStart hook: provision dependencies so tests, linters, and the demo run
# in Claude Code on the web sessions. Idempotent and non-interactive.
set -euo pipefail

# Only run in the remote (web) environment; local dev manages its own env.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# tsbench-forge's core path is numpy-only; pytest + ruff drive tests and lint.
python -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
python -m pip install --quiet numpy pytest ruff

echo "session-start: numpy + pytest + ruff ready"
