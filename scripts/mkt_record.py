"""Record Hyperliquid 1m bars + funding for cascade's MarketCast (DEC-CA-0064 M15).

This is cascade's own recorder (``cascade.pool.marketcast_store.record_once``,
the code behind ``cascade-pool mkt-record``), run from an exported copy of
cascade origin/main. The store layout is cascade's, byte for byte, so
``cascade-pool publish --lane mkt --bar-store DIR`` can read it directly.

Two things are added around it, and neither changes what is written:

* **Throttle.** Hyperliquid allows 1200 request-weight per minute per IP. A
  candleSnapshot costs 20 + 1 per 60 candles, a fundingHistory 20 + 1 per 20
  rows. Cascade's recorder fires all 58 assets x 2 calls in ~30 s, so ~20 of
  them got HTTP 429 on this box. Here every call waits for its share of a
  1000/min budget (limit 1200), and a 429 backs off and retries.
* **Adaptive lookback.** A normal pass re-fetches 2 h. After a failed or
  missed pass, the lookback grows to cover everything since the last fully
  clean pass (+1 h), capped at 48 h (the API holds ~83 h of 1m bars).

usage: mkt_record.py --cascade DIR --store DIR [--state FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
from pathlib import Path

BUDGET_PER_MIN = 1000.0
MIN_LOOKBACK_H, MAX_LOOKBACK_H = 2.0, 48.0


def weight(body: dict) -> float:
    req = body.get("req", body)
    span_ms = max(0, int(req.get("endTime", 0)) - int(req.get("startTime", 0)))
    if body.get("type") == "candleSnapshot":
        return 20 + min(span_ms / 60_000, 5000) / 60
    if body.get("type") == "fundingHistory":
        return 20 + span_ms / 3_600_000 / 20
    return 20


def throttled(post):
    nxt = [0.0]

    def call(url, body, **kw):
        for attempt in range(5):
            wait = nxt[0] - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            nxt[0] = max(nxt[0], time.monotonic()) + weight(body) * 60 / BUDGET_PER_MIN
            try:
                return post(url, body, **kw)
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 4:
                    raise
                time.sleep(15 * (attempt + 1))
                nxt[0] = time.monotonic()
        raise RuntimeError("unreachable")
    return call


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cascade", type=Path, required=True, help="exported cascade tree")
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--state", type=Path, default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(a.cascade))
    from cascade.pool.marketcast_feeds import http_post_json
    from cascade.pool.marketcast_store import MktBarStore, record_once
    from cascade.shared.config import load_chain_config

    state_f = a.state or a.store / ".recorder_state.json"
    try:
        last_ok = float(json.loads(state_f.read_text())["last_clean_pass"])
    except (OSError, ValueError, KeyError):
        last_ok = 0.0
    now = time.time()
    gap_h = (now - last_ok) / 3600 + 1 if last_ok else MAX_LOOKBACK_H
    lookback_h = min(MAX_LOOKBACK_H, max(MIN_LOOKBACK_H, gap_h))

    cfg = load_chain_config(a.cascade / "chain.toml")
    t0 = time.monotonic()
    got = record_once(MktBarStore(a.store), cfg.marketcast, post=throttled(http_post_json),
                      now=now, lookback_minutes=int(lookback_h * 60))
    bad = {k: v for k, v in got.items() if isinstance(v, str)}
    mins = sum(v for v in got.values() if isinstance(v, int))
    print(f"{time.strftime('%FT%TZ', time.gmtime())} lookback {lookback_h:.1f}h: "
          f"{mins} minute(s) over {len(got) - len(bad)} asset(s), {len(bad)} failed, "
          f"{time.monotonic() - t0:.0f}s", flush=True)
    for k, v in sorted(bad.items()):
        print(f"  {k}: {v}")
    if got and not bad:
        a.store.mkdir(parents=True, exist_ok=True)
        tmp = state_f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"last_clean_pass": now}))
        tmp.replace(state_f)
    return 0 if got and len(bad) < len(got) else 1


if __name__ == "__main__":
    sys.exit(main())
