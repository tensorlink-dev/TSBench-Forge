"""Record Dukascopy 1m FX bars into a MarketCast-layout bar store.

Hyperliquid lists only EUR, JPY and GBP (vs USD); cascade's MarketCast
(DEC-CA-0064, open item 2) has no source for the other 25 of the 28
majors/crosses, and Hyperliquid's 1m depth (~3.6 days) leaves no minute
history for training (open item 10). Dukascopy serves free per-day 1m candle
files for all 28 pairs going back years:

    https://datafeed.dukascopy.com/datafeed/<PAIR>/<YYYY>/<MM-1>/<DD>/BID_candles_min_1.bi5

(month is 0-based; LZMA; 24-byte big-endian records: uint32 seconds into the
day, int32 open, close, low, high in points, float32 volume). A day's file
exists only once the day is over, so this is a T+1 recorder plus a backfill.

Written in cascade's MktBarStore layout so the same reader works:
    <store>/fx/<PAIR>/<YYYY-MM-DD>/{open,high,low,close,volume}.npy  float64[1440]
BID side only. Dukascopy fills minutes with no ticks with a flat zero-volume
candle; those are written as NaN (cascade's rule: a minute nobody traded is
missing, never interpolated). A day whose file is empty (weekend) is written
all-NaN, so it is not re-requested. Cascade does not read this source yet
(no adapter); this store is history waiting for one. Redistribution terms of
Dukascopy data are unverified (owner item).

Polite by construction: one request at a time, a pause between them, and the
run stops at the first run of consecutive 503s (the feed throttles that way).

usage: mkt_dukascopy.py --store DIR [--days-back 730] [--max-requests 600]
"""
from __future__ import annotations

import argparse
import datetime as dt
import lzma
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

CCY = ("EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY")   # Dukascopy base-quote order
PAIRS = tuple(a + b for i, a in enumerate(CCY) for b in CCY[i + 1:])  # 28
FIELDS = ("open", "high", "low", "close", "volume")
URL = "https://datafeed.dukascopy.com/datafeed/{p}/{y}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"


def scale(pair: str) -> float:
    return 1e3 if pair.endswith("JPY") else 1e5


def decode(raw: bytes, pair: str) -> dict[str, np.ndarray]:
    out = {f: np.full(1440, np.nan) for f in FIELDS}
    if not raw:
        return out
    b = lzma.decompress(raw)
    s = scale(pair)
    for k in range(len(b) // 24):
        t, o, c, lo, h, v = struct.unpack(">Iiiiif", b[k * 24:(k + 1) * 24])
        i = t // 60
        if not 0 <= i < 1440 or v <= 0:
            continue
        out["open"][i], out["high"][i], out["low"][i], out["close"][i] = o / s, h / s, lo / s, c / s
        out["volume"][i] = float(v)
    return out


def write_day(d: Path, arrs: dict[str, np.ndarray]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for f, a in arrs.items():
        tmp = d / f".{f}.npy.tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, a)
        tmp.replace(d / f"{f}.npy")


def fetch(url: str) -> bytes | int:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:   # noqa: S310 — fixed https host
            return r.read()
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return 599                            # network / timeout: transient, backs off like a 503


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--days-back", type=int, default=730)
    ap.add_argument("--max-requests", type=int, default=600)
    ap.add_argument("--pause", type=float, default=1.5)
    ap.add_argument("--pairs", default=",".join(PAIRS))
    a = ap.parse_args()
    today = dt.datetime.now(dt.UTC).date()
    # newest first: yesterday for every pair, then step back a day at a time
    todo = [(today - dt.timedelta(days=k), p) for k in range(1, a.days_back + 1)
            for p in a.pairs.split(",")]
    todo = [(day, p) for day, p in todo
            if not (a.store / "fx" / p / str(day) / "close.npy").exists()]
    got = empty = miss = req = streak = 0
    for day, p in todo:
        if req >= a.max_requests:
            break
        if req:
            time.sleep(a.pause)
        req += 1
        r = fetch(URL.format(p=p, y=day.year, m=day.month - 1, d=day.day))
        if isinstance(r, int):
            if r == 404:                      # before the pair's history starts
                write_day(a.store / "fx" / p / str(day), decode(b"", p))
                miss += 1
                continue
            streak += 1
            if streak >= 5:
                print(f"  stopping: {streak} consecutive HTTP {r} (throttled or down)", file=sys.stderr)
                break
            time.sleep(30 * streak)
            continue
        streak = 0
        arrs = decode(r, p)
        write_day(a.store / "fx" / p / str(day), arrs)
        if np.isfinite(arrs["close"]).any():
            got += 1
        else:
            empty += 1
    left = len(todo) - got - empty - miss
    print(f"{dt.datetime.now(dt.UTC):%FT%TZ} {req} request(s): {got} day(s) with bars, {empty} empty "
          f"(weekend/holiday), {miss} before history; {left} pair-day(s) still to fetch", flush=True)
    return 0 if streak < 5 else 1


if __name__ == "__main__":
    sys.exit(main())
