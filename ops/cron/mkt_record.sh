#!/usr/bin/env bash
# MarketCast forward bar recorder (cascade DEC-CA-0064 review M15).
#
# Hyperliquid serves only ~5000 candles per interval (1m = ~3.6 days), so the
# minute history MarketCast needs can only exist if someone records it as it
# closes. This runs cascade's own recorder (record_once, the code behind
# `cascade-pool mkt-record`) through scripts/mkt_record.py, which adds a
# rate-limit throttle and an adaptive lookback. The store layout is cascade's,
# so `cascade-pool publish --lane mkt --bar-store /root/mkt/bars` reads it.
#
# Code: an exported copy of cascade origin/main in /root/cascade-mkt (the
# /root/cascade checkout is read-only for us and far behind). Refresh it with
#   rm -rf /root/cascade-mkt && mkdir /root/cascade-mkt && \
#   git -C /root/cascade fetch -q origin && \
#   git -C /root/cascade archive origin/main | tar -x -C /root/cascade-mkt
# Interpreter: /root/cascade/.venv (numpy + stdlib are all this path needs).
#
# Cost: ~45MB RSS, ~3 min per normal 2h pass; ~0.2 MB/day/asset on disk.
set -uo pipefail
STORE=${MKT_STORE:-/root/mkt/bars}
mkdir -p "$STORE"
nice -n 5 /root/cascade/.venv/bin/python -I /root/TSBench-Forge/scripts/mkt_record.py \
    --cascade /root/cascade-mkt --store "$STORE" >> /root/cron/logs/mkt_record.log 2>&1
