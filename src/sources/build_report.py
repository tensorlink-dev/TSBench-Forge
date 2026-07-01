#!/usr/bin/env python3
"""
Build a markdown report with one plot per data source.

Reads:  samples/<name>.{json,csv,xml,zip,txt}   (richer, format-clean payloads)
Reads:  data/<source_id>/*.parquet              (live scraper output)
Reads:  sources.yaml                            (metadata)
Writes: reports/plots/<source_id>.png
Writes: reports/REPORT.md

For each source we register a small loader that extracts (timestamp, value[s])
from the source's native payload format, then a generic plotter handles the
rest. Per-source loaders keep the report robust against scraper schema gaps.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from textwrap import shorten
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
REPORTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


def parse_ts(series, **kw):
    return pd.to_datetime(series, errors="coerce", utc=True, **kw)


# ──────────────────────────────────────────────────────────────────────────
# Per-source loaders. Each returns a long-form DataFrame with at least a
# 'ts' (datetime) column and one or more numeric/category columns.
# Some loaders return DataFrames with an extra 'panel' column for hierarchical
# sources.
# ──────────────────────────────────────────────────────────────────────────

def L(name: str) -> Path:
    return SAMPLES / name


def _epoch_to_dt(v, unit="auto"):
    if v is None:
        return None
    n = float(v)
    if unit == "auto":
        unit = "ms" if n > 1e12 else "s"
    return dt.datetime.fromtimestamp(n / (1000.0 if unit == "ms" else 1.0), dt.timezone.utc)


def load_polymarket_btc():
    d = json.load(open(L("polymarket_btc_history.json")))
    rows = [{"ts": _epoch_to_dt(x["t"], "s"), "price": x["p"]} for x in d["history"]]
    return pd.DataFrame(rows)


def load_kalshi_markets():
    """Use the per-market candlestick endpoint for one Kalshi BTC market."""
    d = json.load(open(L("kalshi_candles.json")))
    rows = []
    for c in d.get("candlesticks", []):
        ya = c.get("yes_ask", {}) or {}
        rows.append({
            "ts": _epoch_to_dt(c["end_period_ts"], "s"),
            "yes_close": float(ya.get("close_dollars") or 0),
            "yes_high": float(ya.get("high_dollars") or 0),
            "yes_low":  float(ya.get("low_dollars") or 0),
        })
    df = pd.DataFrame(rows).sort_values("ts")
    # filter null-only rows
    df = df[(df["yes_close"] > 0) | (df["yes_high"] > 0)]
    return df


def load_manifold_markets():
    """Cross-market recent-bets feed → bets-per-minute count series over the
    most recent ~75 min."""
    d = json.load(open(L("manifold_bets.json")))
    rows = [{"ts": _epoch_to_dt(b["createdTime"], "ms")} for b in d
            if b.get("createdTime")]
    df = pd.DataFrame(rows).dropna(subset=["ts"]).sort_values("ts")
    if df.empty:
        return df
    df = df.set_index("ts").assign(bets=1).resample("min")["bets"].sum().reset_index()
    return df


def load_coingecko():
    d = json.load(open(L("coingecko_btc_hourly.json")))
    rows = [{"ts": _epoch_to_dt(t, "ms"), "price_usd": p} for (t, p) in d["prices"]]
    return pd.DataFrame(rows)


def load_binance():
    d = json.load(open(L("binance_btcusdt_1m.json")))
    rows = [{"ts": _epoch_to_dt(r[0], "ms"),
             "open": float(r[1]), "high": float(r[2]),
             "low": float(r[3]), "close": float(r[4]),
             "volume": float(r[5])} for r in d]
    return pd.DataFrame(rows)


def load_coinbase():
    d = json.load(open(L("coinbase_btc_5min.json")))
    rows = [{"ts": _epoch_to_dt(r[0], "s"), "low": r[1], "high": r[2],
             "open": r[3], "close": r[4], "volume": r[5]} for r in d]
    return pd.DataFrame(rows)


def load_defillama():
    d = json.load(open(L("defillama_eth_tvl.json")))
    rows = [{"ts": _epoch_to_dt(x["date"], "s"), "tvl_usd": x["tvl"]} for x in d]
    return pd.DataFrame(rows)


def load_frankfurter():
    """Range query — daily ECB FX rates over ~16 months."""
    d = json.load(open(L("frankfurter_fx_range.json")))
    rows = []
    for date, rates in d["rates"].items():
        ts = pd.to_datetime(date, utc=True)
        for cur, rate in rates.items():
            rows.append({"ts": ts, "rate": rate, "panel": cur})
    return pd.DataFrame(rows).sort_values("ts")


def load_treasury():
    df = pd.read_csv(L("treasury_yield_curve.csv"))
    df["ts"] = pd.to_datetime(df["Date"], utc=True)
    df = df.drop(columns=["Date"]).set_index("ts").stack().reset_index()
    df.columns = ["ts", "panel", "yield_pct"]
    df["yield_pct"] = pd.to_numeric(df["yield_pct"], errors="coerce")
    return df


def load_uk_carbon():
    """30-day range — half-hourly actual + forecast."""
    d = json.load(open(L("uk_carbon_intensity_range.json")))["data"]
    rows = [{"ts": pd.to_datetime(r["from"], utc=True),
             "actual": r["intensity"].get("actual"),
             "forecast": r["intensity"].get("forecast")} for r in d]
    return pd.DataFrame(rows).sort_values("ts")


def load_uk_genmix():
    """7-day range — half-hourly fuel mix percentages."""
    d = json.load(open(L("uk_carbon_intensity_genmix_range.json")))["data"]
    rows = []
    for r in d:
        ts = pd.to_datetime(r["from"], utc=True)
        for x in r["generationmix"]:
            rows.append({"ts": ts, "perc": x["perc"], "panel": x["fuel"]})
    return pd.DataFrame(rows).sort_values("ts")


def load_octopus():
    d = json.load(open(L("octopus_agile_rates.json")))
    rows = [{"ts": pd.to_datetime(r["valid_from"], utc=True),
             "p_per_kwh": r["value_inc_vat"]} for r in d["results"]]
    return pd.DataFrame(rows).sort_values("ts")


def load_aemo():
    df = pd.read_csv(L("aemo_nem_5min.csv"))
    df["ts"] = pd.to_datetime(df["SETTLEMENTDATE"], utc=True)
    return df[["ts", "TOTALDEMAND", "RRP"]].rename(columns={"TOTALDEMAND": "demand_mw", "RRP": "price_aud"})


def load_caiso():
    """24-hour CAISO LMP at 5-min cadence (288 points)."""
    rows = []
    with zipfile.ZipFile(L("caiso_oasis_lmp_24h.zip")) as z:
        for name in z.namelist():
            if not name.endswith(".xml"):
                continue
            tree = ET.parse(z.open(name))
            for d in tree.iter():
                tag = d.tag.split("}")[-1]
                if tag != "REPORT_DATA":
                    continue
                ts = None; val = None; data_item = None
                for c in d:
                    t = c.tag.split("}")[-1]
                    if t == "INTERVAL_START_GMT": ts = c.text
                    elif t == "VALUE": val = c.text
                    elif t == "DATA_ITEM": data_item = c.text
                if ts and val and data_item == "LMP_PRC":
                    rows.append({"ts": pd.to_datetime(ts, utc=True), "lmp_usd": float(val)})
    return pd.DataFrame(rows).sort_values("ts")


def load_nyiso():
    df = pd.read_csv(L("nyiso_realtime_lmp.csv"))
    # Treat the "Time Stamp" as a wall-clock NY time and re-interpret as UTC.
    # Avoids tz database dependency on hosts that may not have IANA names.
    df["ts"] = pd.to_datetime(df["Time Stamp"], utc=True, errors="coerce")
    return df[["ts", "Name", "LBMP ($/MWHr)"]].rename(
        columns={"Name": "panel", "LBMP ($/MWHr)": "lmp_usd"}
    )


def load_elia():
    d = json.load(open(L("elia_solar_15min.json")))["results"]
    rows = [{"ts": pd.to_datetime(r["datetime"], utc=True),
             "measured_mw": r["measured"],
             "panel": r["region"]} for r in d]
    return pd.DataFrame(rows).sort_values("ts")


def load_uk_neso():
    raw = json.load(open(L("uk_neso_carbon_historic.json")))
    # CKAN structure: {"result": {"records": [...]}} or older saved {"records": [...]}
    if "records" in raw:
        d = raw["records"]
    elif "result" in raw and "records" in raw["result"]:
        d = raw["result"]["records"]
    else:
        d = []
    rows = []
    for r in d:
        rows.append({
            "ts": pd.to_datetime(r["DATETIME"], utc=True),
            "carbon_intensity": pd.to_numeric(r.get("CARBON_INTENSITY"), errors="coerce"),
            "gas_perc": pd.to_numeric(r.get("GAS_perc"), errors="coerce"),
            "wind_perc": pd.to_numeric(r.get("WIND_perc"), errors="coerce"),
        })
    return pd.DataFrame(rows).sort_values("ts")


def load_usgs_quakes():
    d = json.load(open(L("usgs_earthquakes.json")))
    rows = [{"ts": _epoch_to_dt(f["properties"]["time"], "ms"),
             "magnitude": f["properties"]["mag"]} for f in d["features"]]
    return pd.DataFrame(rows).sort_values("ts")


def load_usgs_streamflow():
    d = json.load(open(L("usgs_streamflow.json")))
    vals = d["value"]["timeSeries"][0]["values"][0]["value"]
    rows = [{"ts": pd.to_datetime(v["dateTime"], utc=True),
             "discharge_cfs": float(v["value"])} for v in vals]
    return pd.DataFrame(rows)


def load_noaa_tides():
    d = json.load(open(L("noaa_tides_coops.json")))
    rows = []
    for r in d["data"]:
        v = r.get("v") or ""
        try:
            rows.append({"ts": pd.to_datetime(r["t"], utc=True),
                         "water_level_m": float(v)})
        except (ValueError, TypeError):
            continue
    return pd.DataFrame(rows).sort_values("ts")


def load_ndbc_buoy():
    """NDBC realtime2 .txt — space-delimited with two header lines."""
    rows = []
    with open(L("ndbc_buoy_41008.txt")) as f:
        f.readline(); f.readline()                  # skip 2 header lines
        for line in f:
            parts = line.split()
            if len(parts) < 12:
                continue
            try:
                ts = dt.datetime(int(parts[0]), int(parts[1]), int(parts[2]),
                                 int(parts[3]), int(parts[4]), tzinfo=dt.timezone.utc)
            except ValueError:
                continue
            def f6(x): return None if x == "MM" else float(x)
            rows.append({
                "ts": ts,
                "wind_speed_ms": f6(parts[6]),
                "pressure_hpa": f6(parts[12]),
                "air_temp_c": f6(parts[13]),
                "water_temp_c": f6(parts[14]) if len(parts) > 14 else None,
            })
    return pd.DataFrame(rows).sort_values("ts")


def load_swpc_kp():
    d = json.load(open(L("swpc_planetary_k.json")))
    rows = [{"ts": pd.to_datetime(r["time_tag"], utc=True),
             "kp": r["estimated_kp"]} for r in d]
    return pd.DataFrame(rows).sort_values("ts")


def load_swpc_solar_wind():
    d = json.load(open(L("swpc_solar_wind_plasma.json")))
    rows = []
    for r in d[1:]:                                 # row 0 is header
        try:
            rows.append({
                "ts": pd.to_datetime(r[0], utc=True),
                "density": float(r[1]) if r[1] not in ("", None) else None,
                "speed": float(r[2]) if r[2] not in ("", None) else None,
                "temperature": float(r[3]) if r[3] not in ("", None) else None,
            })
        except Exception:                          # noqa: BLE001
            continue
    return pd.DataFrame(rows).sort_values("ts")


def load_open_meteo_forecast():
    d = json.load(open(L("open_meteo_forecast.json")))["hourly"]
    df = pd.DataFrame({
        "ts": pd.to_datetime(d["time"], utc=True),
        "temperature_2m": d["temperature_2m"],
        "precipitation": d["precipitation"],
        "wind_speed_10m": d["wind_speed_10m"],
    })
    return df


def load_open_meteo_aq():
    d = json.load(open(L("open_meteo_air_quality.json")))["hourly"]
    return pd.DataFrame({
        "ts": pd.to_datetime(d["time"], utc=True),
        "pm2_5": d["pm2_5"],
        "pm10": d["pm10"],
        "carbon_monoxide": d["carbon_monoxide"],
    })


def load_open_meteo_marine():
    d = json.load(open(L("open_meteo_marine.json")))["hourly"]
    return pd.DataFrame({
        "ts": pd.to_datetime(d["time"], utc=True),
        "wave_height": d["wave_height"],
        "wave_period": d.get("wave_period"),
    })


def load_sensor_community():
    """Use the daily archive CSV — 5-min cadence over a full day."""
    df = pd.read_csv(L("sensor_community_archive.csv"), sep=";")
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    rows = []
    for col, name in [("P1", "PM10"), ("P2", "PM2.5")]:
        sub = df[["ts", col]].copy()
        sub["value"] = pd.to_numeric(sub[col], errors="coerce")
        sub["panel"] = name
        rows.append(sub[["ts", "value", "panel"]])
    return pd.concat(rows, ignore_index=True).dropna(subset=["value"]).sort_values("ts")


def load_mauna_loa_co2():
    rows = []
    with open(L("mauna_loa_co2_daily.txt")) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                y, mo, dy = int(parts[0]), int(parts[1]), int(parts[2])
                trend = float(parts[4])
                rows.append({"ts": dt.datetime(y, mo, dy, tzinfo=dt.timezone.utc),
                             "co2_ppm_trend": trend})
            except ValueError:
                continue
    return pd.DataFrame(rows).sort_values("ts")


def load_nws_active_alerts():
    """Bin per-hour to convert events to a count-per-hour time series."""
    d = json.load(open(L("nws_active_alerts.json")))["features"]
    rows = []
    for f in d:
        p = f["properties"]
        rows.append({"ts": pd.to_datetime(p.get("sent"), utc=True)})
    df = pd.DataFrame(rows).dropna(subset=["ts"]).sort_values("ts")
    if df.empty:
        return df
    df = df.set_index("ts").assign(count=1).resample("h")["count"].sum().reset_index()
    return df


def load_ingv():
    """7-day INGV catalog → magnitude scatter."""
    rows = []
    with open(L("ingv_seismicity_week.txt")) as f:
        f.readline()                                # header
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 11:
                continue
            try:
                rows.append({"ts": pd.to_datetime(parts[1], utc=True),
                             "magnitude": float(parts[10])})
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows).sort_values("ts")


def load_usda_snotel():
    d = json.load(open(L("usda_snotel_swe.json")))
    rows = []
    for station in d:
        for el in station["data"]:
            for v in el["values"]:
                try:
                    rows.append({"ts": pd.to_datetime(v["date"], utc=True),
                                 "swe_in": v["value"]})
                except Exception:                  # noqa: BLE001
                    continue
    return pd.DataFrame(rows).sort_values("ts")


def load_gdacs():
    """Daily disaster-event counts."""
    from email.utils import parsedate_to_datetime
    txt = open(L("gdacs_disasters.xml"), encoding="utf-8-sig").read()
    rows = []
    item_re = re.compile(r"<item>(.*?)</item>", re.DOTALL)
    pub_re = re.compile(r"<pubDate>([^<]+)</pubDate>")
    type_re = re.compile(r"<gdacs:eventtype>([^<]+)</gdacs:eventtype>")
    def _rfc822(s):
        try: return parsedate_to_datetime(s).astimezone(dt.timezone.utc)
        except Exception: return None
    for body in item_re.findall(txt):
        ts_m = pub_re.search(body)
        et_m = type_re.search(body)
        if not ts_m: continue
        ts = _rfc822(ts_m.group(1).strip()) or pd.to_datetime(ts_m.group(1), utc=True, errors="coerce")
        if ts is None or pd.isna(ts):
            continue
        rows.append({"ts": pd.Timestamp(ts), "panel": (et_m.group(1) if et_m else "?")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.set_index("ts").groupby("panel").resample("D").size().reset_index(name="event_count")
    return df


def load_wikimedia_top():
    """The 'top articles' API is one-day snapshot. Convert to a real time
    series by tracking Main_Page (perennial #1) over the past 365 days."""
    d = json.load(open(L("wikimedia_main_page.json")))["items"]
    rows = [{"ts": pd.to_datetime(x["timestamp"][:8], utc=True),
             "views": x["views"]} for x in d]
    return pd.DataFrame(rows).sort_values("ts")


def load_wikimedia_per_article():
    d = json.load(open(L("wikimedia_per_article_pageviews.json")))["items"]
    rows = [{"ts": pd.to_datetime(x["timestamp"][:8], utc=True),
             "views": x["views"], "panel": x["article"]} for x in d]
    return pd.DataFrame(rows)


def load_hn_top():
    """Use HN /v0/item/{id}.json to fetch creation_time for top stories,
    then bin by hour to get an arrival-over-time series. Cached if already
    fetched."""
    cache = L("hn_top_arrivals.json")
    if not cache.exists():
        import urllib.request
        ids = json.load(open(L("hn_top_stories.json")))[:200]
        recs = []
        for sid in ids:
            try:
                with urllib.request.urlopen(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json", timeout=5) as r:
                    item = json.loads(r.read())
                recs.append({"id": sid, "time": item.get("time"), "score": item.get("score")})
            except Exception:                       # noqa: BLE001
                continue
        json.dump(recs, open(cache, "w"))
    recs = json.load(open(cache))
    rows = [{"ts": _epoch_to_dt(r["time"], "s"), "score": r.get("score") or 0}
            for r in recs if r.get("time")]
    df = pd.DataFrame(rows).dropna(subset=["ts"]).sort_values("ts")
    return df


def load_npm_lodash():
    d = json.load(open(L("npm_downloads_per_pkg.json")))
    rows = [{"ts": pd.to_datetime(x["day"], utc=True),
             "downloads": x["downloads"], "panel": d["package"]} for x in d["downloads"]]
    return pd.DataFrame(rows)


def load_pypi_numpy():
    d = json.load(open(L("pypi_downloads_per_pkg.json")))
    rows = [{"ts": pd.to_datetime(x["date"], utc=True),
             "downloads": x["downloads"], "panel": f"{d['package']}/{x.get('category','')}"}
            for x in d["data"]]
    return pd.DataFrame(rows)


def load_crates_serde():
    d = json.load(open(L("crates_io_downloads.json")))
    rows = [{"ts": pd.to_datetime(x["date"], utc=True),
             "downloads": x["downloads"]} for x in d["extra_downloads"]]
    return pd.DataFrame(rows)


def load_itunes():
    """The iTunes RSS feed only exposes a daily snapshot — no public history.
    Show the rank ↔ podcast scatter as a single-day series."""
    d = json.load(open(L("itunes_top_podcasts.json")))["feed"]
    ts = pd.to_datetime(d["updated"], utc=True, errors="coerce")
    if pd.isna(ts):
        ts = dt.datetime.now(dt.timezone.utc)
    rows = [{"ts": ts, "rank": i + 1, "panel": shorten(r["name"], 24)}
            for i, r in enumerate(d["results"][:25])]
    return pd.DataFrame(rows)


def load_reddit():
    d = json.load(open(L("reddit_top_daily.json")))["data"]["children"]
    rows = [{"ts": _epoch_to_dt(x["data"]["created_utc"], "s"),
             "score": x["data"]["score"]} for x in d]
    return pd.DataFrame(rows).sort_values("ts")


def load_opensky():
    """8 polls × 30 s = 4 min — aircraft count over NYC bounding box."""
    d = json.load(open(L("opensky_polled_series.json")))
    rows = [{"ts": _epoch_to_dt(x["time"] or x["ts_polled"], "s"),
             "aircraft_count": x["aircraft_count"]} for x in d]
    return pd.DataFrame(rows).sort_values("ts")


def load_citibike():
    """8 polls — total bikes available across all NYC Citi Bike stations."""
    d = json.load(open(L("citibike_polled_series.json")))
    rows = [{"ts": _epoch_to_dt(x.get("last_updated") or x["ts_polled"], "s"),
             "total_bikes_available": x["total_bikes_available"]} for x in d]
    return pd.DataFrame(rows).sort_values("ts")


def load_bart():
    """8 polls — number of train estimates pending at Powell St station."""
    d = json.load(open(L("bart_polled_series.json")))
    rows = [{"ts": _epoch_to_dt(x["ts_polled"], "s"),
             "estimate_count": x["estimate_count"]} for x in d]
    return pd.DataFrame(rows).sort_values("ts")


def load_tfl_arrivals():
    """8 polls — count of inbound trains predicted on Victoria line."""
    d = json.load(open(L("tfl_arrivals_polled_series.json")))
    rows = [{"ts": _epoch_to_dt(x["ts_polled"], "s"),
             "arrival_count": x["arrival_count"]} for x in d]
    return pd.DataFrame(rows).sort_values("ts")


def load_tfl_bikepoints():
    """8 polls — bikes available at single TfL bikepoint over 4 minutes."""
    d = json.load(open(L("tfl_bikepoints_polled_series.json")))
    rows = [{"ts": _epoch_to_dt(x["ts_polled"], "s"),
             "bikes_available": x["bikes_available"],
             "empty_docks": x["empty_docks"]} for x in d]
    return pd.DataFrame(rows).sort_values("ts")


def load_hn_max():
    """8 polls — diff = HN item arrival rate (~per 30 s)."""
    d = json.load(open(L("hn_max_item_polled_series.json")))
    rows = [{"ts": _epoch_to_dt(x["ts_polled"], "s"),
             "max_item": x["max_item"],
             "arrivals_since_prev": x["arrivals_since_prev"]} for x in d]
    return pd.DataFrame(rows).sort_values("ts").iloc[1:]    # drop the first poll's bogus diff


def load_github_events():
    """1.1k events sampled across 12 polls. Per-minute counts over the
    polling window plus the natural arrival distribution beforehand."""
    d = json.load(open(L("github_events.json")))
    rows = [{"ts": pd.to_datetime(e["created_at"], utc=True)} for e in d]
    df = pd.DataFrame(rows).dropna(subset=["ts"]).sort_values("ts")
    if df.empty:
        return df
    df = df.set_index("ts").assign(events=1).resample("min")["events"].sum().reset_index()
    # Drop trailing zeros where polling stopped
    df = df[df["events"] > 0]
    return df


def load_so_questions():
    d = json.load(open(L("stackoverflow_questions.json")))["items"]
    rows = [{"ts": _epoch_to_dt(x["creation_date"], "s"),
             "score": x.get("score", 0)} for x in d]
    return pd.DataFrame(rows).sort_values("ts")


def load_aws_health():
    """Daily incident counts. Extreme zero-inflation expected."""
    from email.utils import parsedate_to_datetime
    txt = open(L("aws_health_rss.xml")).read()
    item_re = re.compile(r"<item>(.*?)</item>", re.DOTALL)
    pub_re = re.compile(r"<pubDate>([^<]+)</pubDate>")

    def _rfc822(s: str):
        try:
            return parsedate_to_datetime(s).astimezone(dt.timezone.utc)
        except Exception:                                    # noqa: BLE001
            return None

    rows = []
    for body in item_re.findall(txt):
        ts_m = pub_re.search(body)
        if not ts_m:
            continue
        ts = _rfc822(ts_m.group(1).strip())
        if ts is None:
            continue
        rows.append({"ts": pd.Timestamp(ts)})
    df = pd.DataFrame(rows).dropna(subset=["ts"]).sort_values("ts")
    if df.empty:
        return df
    df = df.set_index("ts").assign(incidents=1).resample("D")["incidents"].sum().reset_index()
    return df


def load_nws_alerts_count():
    """8 polls — total / land / marine alert counts over 4 minutes."""
    d = json.load(open(L("nws_alerts_count_polled_series.json")))
    rows = [{"ts": _epoch_to_dt(x["ts_polled"], "s"),
             "total": x["total"],
             "land": x["land"],
             "marine": x["marine"]} for x in d]
    return pd.DataFrame(rows).sort_values("ts")


def load_ripestat():
    """6-month range query — prefix count series for AS3333 (RIPE NCC)."""
    d = json.load(open(L("ripestat_announced_prefixes_range.json")))["data"]
    rows = [{"ts": pd.to_datetime(x["timeline"][0]["starttime"], utc=True, errors="coerce") if x.get("timeline") else None,
             "prefix_count": len([p for p in [x.get("prefix")]] if x.get("prefix") else [])}
            for x in d.get("prefixes", [])]
    # The above is awkward — better just count visible-time per prefix
    rows = []
    for p in d.get("prefixes", []):
        for tl in p.get("timelines", []) or []:
            for ts_ in (tl.get("starttime"), tl.get("endtime")):
                if not ts_:
                    continue
                rows.append({"ts": pd.to_datetime(ts_, utc=True, errors="coerce"),
                             "prefix": p.get("prefix")})
    if not rows:
        # Fallback: just snapshot count
        return pd.DataFrame([{"ts": pd.to_datetime(d.get("query_endtime"), utc=True),
                              "prefix_count": len(d.get("prefixes", []))}])
    df = pd.DataFrame(rows).dropna(subset=["ts"])
    # Bin by week, count active prefixes
    df = df.set_index("ts").resample("W").nunique().reset_index().rename(columns={"prefix": "active_prefixes"})
    return df


def load_rki_cases():
    d = json.load(open(L("rki_germany_cases.json")))["data"]
    return pd.DataFrame([{"ts": pd.to_datetime(x["date"], utc=True),
                          "new_cases": x["cases"]} for x in d])


def load_rki_hosp():
    d = json.load(open(L("rki_germany_hospitalizations.json")))["data"]
    return pd.DataFrame([{"ts": pd.to_datetime(x["date"], utc=True),
                          "incidence_7d": x["incidence7Days"]} for x in d])


def load_wiki_health():
    d = json.load(open(L("wikimedia_health_influenza.json")))["items"]
    rows = [{"ts": pd.to_datetime(x["timestamp"][:8], utc=True),
             "views": x["views"], "panel": x["article"]} for x in d]
    return pd.DataFrame(rows)


def load_nyc311():
    """Hourly complaint counts (200 samples → ~last day)."""
    d = json.load(open(L("nyc311_complaints.json")))
    rows = [{"ts": pd.to_datetime(x["created_date"], utc=True)} for x in d]
    df = pd.DataFrame(rows).dropna(subset=["ts"]).sort_values("ts")
    if df.empty:
        return df
    df = df.set_index("ts").assign(complaints=1).resample("h")["complaints"].sum().reset_index()
    return df


def load_mempool():
    """8 polls of mempool.space — unconfirmed-tx count over 4 minutes."""
    d = json.load(open(L("mempool_unconfirmed.json")))
    rows = [{"ts": _epoch_to_dt(x["ts_polled"], "s"),
             "unconfirmed_count": x["count"],
             "vsize_bytes": x["vsize"]} for x in d]
    return pd.DataFrame(rows).sort_values("ts")


def load_binance_funding():
    d = json.load(open(L("binance_funding_rate.json")))
    rows = [{"ts": _epoch_to_dt(r["fundingTime"], "ms"),
             "funding_rate": float(r["fundingRate"]),
             "mark_price": float(r["markPrice"])} for r in d]
    return pd.DataFrame(rows).sort_values("ts")


def load_btc_blocks():
    d = json.load(open(L("bitcoin_block_arrivals.json")))
    rows = [{"ts": _epoch_to_dt(b["time"], "s"), "height": b["height"]} for b in d]
    df = pd.DataFrame(rows).dropna(subset=["ts"]).sort_values("ts")
    if df.empty:
        return df
    df = df.set_index("ts").assign(blocks=1).resample("h")["blocks"].sum().reset_index()
    return df


def load_polymarket_new():
    d = json.load(open(L("polymarket_new_markets.json")))
    rows = [{"ts": pd.to_datetime(x["createdAt"], utc=True, errors="coerce")} for x in d]
    df = pd.DataFrame(rows).dropna(subset=["ts"]).sort_values("ts")
    if df.empty:
        return df
    df = df.set_index("ts").assign(new_markets=1).resample("min")["new_markets"].sum().reset_index()
    df = df[df["new_markets"] > 0]
    return df


def load_swpc_xray():
    d = json.load(open(L("swpc_solar_xray_flux.json")))
    rows = [{"ts": pd.to_datetime(x["time_tag"], utc=True),
             "flux_wm2": x["flux"]} for x in d]
    return pd.DataFrame(rows).sort_values("ts")


def load_om_uv():
    d = json.load(open(L("open_meteo_uv_index.json")))["hourly"]
    return pd.DataFrame({
        "ts": pd.to_datetime(d["time"], utc=True),
        "uv_index": d["uv_index"],
    })


def load_hn_ask():
    d = json.load(open(L("hn_ask_stories.json")))
    rows = [{"ts": _epoch_to_dt(x["time"], "s"),
             "score": x.get("score") or 0,
             "descendants": x.get("descendants") or 0} for x in d if x.get("time")]
    return pd.DataFrame(rows).dropna(subset=["ts"]).sort_values("ts")



def load_tfl_status():
    """6 polls of TfL line status — count of disrupted lines + per-line severity."""
    d = json.load(open(L("tfl_line_status.json")))
    rows = []
    for r in d:
        rows.append({"ts": _epoch_to_dt(r["ts_polled"], "s"),
                     "is_disrupted": r["is_disrupted"],
                     "severity": r.get("severity"),
                     "panel": r["line"]})
    df = pd.DataFrame(rows).sort_values("ts")
    return df


def load_bpa():
    """BPA balancing 5-min: skip header lines, parse Date/Time + cols."""
    rows = []
    header = None
    with open(L("bpa_balancing.txt")) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("Date/Time"):
                header = line.split("\t")
                continue
            if header is None:
                continue
            parts = line.split("\t")
            if len(parts) < len(header):
                continue
            try:
                ts = pd.to_datetime(parts[0], utc=True, errors="coerce")
            except Exception:
                continue
            if pd.isna(ts):
                continue
            row = {"ts": ts}
            for i, name in enumerate(header[1:], start=1):
                try:
                    row[name.strip()] = float(parts[i])
                except (ValueError, IndexError):
                    row[name.strip()] = None
            rows.append(row)
    return pd.DataFrame(rows).sort_values("ts")


def load_neso_stor():
    """STOR notification log — bin per day, count events."""
    df = pd.read_csv(L("neso_stor_notifications.csv"))
    df["ts"] = pd.to_datetime(df["Notification Issued Date Time"], format="%d/%m/%Y %H:%M",
                              utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    if df.empty:
        return df
    # Bin per week given the very sparse cadence
    df = df.set_index("ts").assign(events=1).resample("W")["events"].sum().reset_index()
    df = df[df["events"] > 0]
    return df


def load_gharchive_per_minute():
    d = json.load(open(L("gharchive_per_minute.json")))
    rows = [{"ts": pd.to_datetime(x["minute"], utc=True),
             "event_count": x["event_count"]} for x in d]
    return pd.DataFrame(rows).sort_values("ts")


def load_binance_15m():
    d = json.load(open(L("binance_btcusdt_15m.json")))
    rows = [{"ts": _epoch_to_dt(r[0], "ms"),
             "close": float(r[4]), "volume": float(r[5])} for r in d]
    return pd.DataFrame(rows).sort_values("ts")


def load_binance_30m():
    d = json.load(open(L("binance_btcusdt_30m.json")))
    rows = [{"ts": _epoch_to_dt(r[0], "ms"),
             "close": float(r[4]), "volume": float(r[5])} for r in d]
    return pd.DataFrame(rows).sort_values("ts")


def load_eia_nyis():
    d = json.load(open(L("eia_nyis_hourly_demand.json")))["response"]["data"]
    rows = []
    for r in d:
        ts = pd.to_datetime(r["period"], utc=True, errors="coerce")
        v = r.get("value")
        try:
            v = float(v)
        except (ValueError, TypeError):
            v = None
        if pd.notna(ts) and v is not None:
            rows.append({"ts": ts, "demand_mwh": v})
    return pd.DataFrame(rows).sort_values("ts")


def load_usgs_dv():
    d = json.load(open(L("usgs_streamflow_daily.json")))
    vals = d["value"]["timeSeries"][0]["values"][0]["value"]
    rows = [{"ts": pd.to_datetime(v["dateTime"], utc=True),
             "discharge_cfs_daily_mean": float(v["value"])} for v in vals]
    return pd.DataFrame(rows).sort_values("ts")


def load_om_climate():
    d = json.load(open(L("open_meteo_climate_daily.json")))["daily"]
    return pd.DataFrame({"ts": pd.to_datetime(d["time"], utc=True),
                         "temperature_2m_mean": d["temperature_2m_mean"]}).sort_values("ts")


def load_npm_total():
    d = json.load(open(L("npm_total_daily.json")))
    rows = [{"ts": pd.to_datetime(x["day"], utc=True),
             "downloads": x["downloads"]} for x in d["downloads"]]
    return pd.DataFrame(rows).sort_values("ts")


def load_nws_heat():
    """Sparse heat-advisory subset — bin per hour."""
    d = json.load(open(L("nws_heat_advisories.json")))["features"]
    rows = [{"ts": pd.to_datetime(f["properties"]["sent"], utc=True, errors="coerce")} for f in d]
    df = pd.DataFrame(rows).dropna(subset=["ts"]).sort_values("ts")
    if df.empty:
        return df
    df = df.set_index("ts").assign(advisories=1).resample("h")["advisories"].sum().reset_index()
    df = df[df["advisories"] > 0]
    return df


def load_hn_algolia():
    """Bin HN stories by minute for arrival-rate series."""
    d = json.load(open(L("hn_algolia_stories.json")))["hits"]
    rows = [{"ts": pd.to_datetime(h["created_at"], utc=True)} for h in d]
    df = pd.DataFrame(rows).dropna(subset=["ts"]).sort_values("ts")
    if df.empty:
        return df
    df = df.set_index("ts").assign(stories=1).resample("min")["stories"].sum().reset_index()
    df = df[df["stories"] > 0]
    return df


def load_wikimedia_rc():
    """Bin Wikipedia recent-changes by minute."""
    d = json.load(open(L("wikimedia_recent_changes.json")))["query"]["recentchanges"]
    rows = [{"ts": pd.to_datetime(r["timestamp"], utc=True)} for r in d]
    df = pd.DataFrame(rows).dropna(subset=["ts"]).sort_values("ts")
    if df.empty:
        return df
    df = df.set_index("ts").assign(edits=1).resample("min")["edits"].sum().reset_index()
    df = df[df["edits"] > 0]
    return df


def load_metar():
    """METAR for KJFK over 6 hours."""
    d = json.load(open(L("metar_kjfk.json")))
    rows = [{"ts": _epoch_to_dt(r["obsTime"], "s"),
             "temp_c": r.get("temp"),
             "wspd_kt": r.get("wspd"),
             "altimeter": r.get("altim")} for r in d]
    return pd.DataFrame(rows).sort_values("ts")

# Map source-id -> loader
LOADERS: dict[str, Callable[[], pd.DataFrame]] = {
    # econ_fin
    "polymarket_btc_history": load_polymarket_btc,
    "kalshi_markets_snapshot": load_kalshi_markets,
    "manifold_markets_snapshot": load_manifold_markets,
    "coingecko_btc_hourly": load_coingecko,
    "binance_btcusdt_1m": load_binance,
    "coinbase_btc_5min": load_coinbase,
    "defillama_chain_tvl_daily": load_defillama,
    "frankfurter_fx_daily": load_frankfurter,
    "treasury_yield_curve_daily": load_treasury,
    # energy
    "uk_carbon_intensity_actual": load_uk_carbon,
    "uk_carbon_intensity_genmix": load_uk_genmix,
    "octopus_agile_tariff": load_octopus,
    "aemo_nem_5min": load_aemo,
    "caiso_oasis_lmp_5min": load_caiso,
    "nyiso_realtime_lmp_zonal": load_nyiso,
    "elia_solar_15min": load_elia,
    "uk_neso_carbon_historic_30min": load_uk_neso,
    # nature
    "usgs_earthquakes_realtime": load_usgs_quakes,
    "usgs_streamflow_iv": load_usgs_streamflow,
    "noaa_tides_coops": load_noaa_tides,
    "ndbc_buoy_realtime": load_ndbc_buoy,
    "swpc_planetary_k": load_swpc_kp,
    "swpc_solar_wind_plasma": load_swpc_solar_wind,
    "open_meteo_forecast": load_open_meteo_forecast,
    "open_meteo_air_quality": load_open_meteo_aq,
    "open_meteo_marine": load_open_meteo_marine,
    "sensor_community_pm25": load_sensor_community,
    "mauna_loa_co2_daily": load_mauna_loa_co2,
    "nws_active_alerts": load_nws_active_alerts,
    "ingv_seismicity": load_ingv,
    "usda_snotel_swe": load_usda_snotel,
    "gdacs_natural_disasters": load_gdacs,
    # sales
    "wikimedia_top_articles_daily": load_wikimedia_top,
    "wikimedia_per_article_pageviews": load_wikimedia_per_article,
    "hn_top_stories": load_hn_top,
    "npm_downloads_per_pkg": load_npm_lodash,
    "pypi_downloads_per_pkg": load_pypi_numpy,
    "crates_io_downloads": load_crates_serde,
    "itunes_top_podcasts": load_itunes,
    "reddit_top_daily": load_reddit,
    # transport
    "opensky_states_local": load_opensky,
    "citibike_station_status": load_citibike,
    "bart_realtime_etd": load_bart,
    "tfl_arrivals": load_tfl_arrivals,
    "tfl_bikepoints_status": load_tfl_bikepoints,
    # web_cloudops
    "hn_max_item": load_hn_max,
    "github_events_firehose": load_github_events,
    "stackoverflow_questions": load_so_questions,
    "aws_health_rss": load_aws_health,
    "nws_alerts_count": load_nws_alerts_count,
    "ripestat_announced_prefixes": load_ripestat,
    # healthcare
    "rki_germany_cases": load_rki_cases,
    "rki_germany_hospitalizations": load_rki_hosp,
    "wikimedia_health_pageviews": load_wiki_health,
    "nyc_311_complaints": load_nyc311,
    # ---- gap-filler additions ----
    "mempool_unconfirmed_count": load_mempool,
    "binance_btcusdt_funding_rate": load_binance_funding,
    "bitcoin_block_arrivals": load_btc_blocks,
    "polymarket_new_market_rate": load_polymarket_new,
    "swpc_solar_xray_flux": load_swpc_xray,
    "open_meteo_uv_index": load_om_uv,
    "hn_ask_stories": load_hn_ask,
    # ---- final-7 closing ----
    "tfl_line_status": load_tfl_status,
    "bpa_balancing_load_wind": load_bpa,
    "neso_stor_notifications": load_neso_stor,
    "gharchive_hourly_events": load_gharchive_per_minute,
    # ---- cross-frequency fillers ----
    "binance_btcusdt_15m": load_binance_15m,
    "binance_btcusdt_30m": load_binance_30m,
    "eia_nyis_hourly_demand": load_eia_nyis,
    "usgs_streamflow_daily": load_usgs_dv,
    "open_meteo_climate_daily": load_om_climate,
    "npm_total_daily_downloads": load_npm_total,
    "nws_heat_advisories": load_nws_heat,
    "hn_algolia_stories": load_hn_algolia,
    "wikimedia_recent_changes": load_wikimedia_rc,
    "metar_us_airports": load_metar,
}


# ──────────────────────────────────────────────────────────────────────────
# Plotter
# ──────────────────────────────────────────────────────────────────────────


def numeric_cols(df: pd.DataFrame) -> list[str]:
    """Columns whose non-null entries parse as numbers — treats sparse
    numeric columns (e.g. half the markets have no probability set) as
    numeric so we get a line plot rather than a value-counts bar."""
    out = []
    for c in df.columns:
        if c in ("ts", "panel"):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        non_null = s.notna()
        if non_null.sum() == 0:
            continue
        # Either ≥30% of all rows parse, or ≥80% of the originally-non-null
        # entries parse (catches int columns with many None values).
        orig_non_null = df[c].notna()
        if non_null.sum() / max(len(df), 1) >= 0.3 or (
            orig_non_null.sum() and non_null.sum() / orig_non_null.sum() >= 0.8
        ):
            out.append(c)
    return out


def plot_one(sid: str, src: dict, df: pd.DataFrame, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=110)
    ax.set_title(f"{sid}  ({src['domain']}, {src['frequency']}, {src['pretraining_novelty']})",
                 fontsize=10, loc="left")
    ax.grid(alpha=0.25, linewidth=0.4)

    if df is None or len(df) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout(); fig.savefig(out); plt.close(fig)
        return "no rows"

    df = df.dropna(subset=["ts"]).sort_values("ts")
    nums = numeric_cols(df)
    has_panel = "panel" in df.columns
    cat_cols = [c for c in df.columns if c not in ("ts", "panel") and c not in nums]

    cap = f"{len(df)} rows"

    if len(df) == 1 and not has_panel and not nums:
        ax.text(0.5, 0.5, shorten(str(df.iloc[0].to_dict()), 200),
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
    elif df["ts"].nunique() == 1 and has_panel and nums:
        # Genuine one-time snapshot of a panel (e.g. iTunes daily ranks).
        ycol = nums[0]
        sub = df.dropna(subset=[ycol]).copy()
        sub[ycol] = pd.to_numeric(sub[ycol], errors="coerce")
        sub = sub.dropna(subset=[ycol]).sort_values(ycol).head(25)
        ax.barh(range(len(sub))[::-1], sub[ycol].values)
        ax.set_yticks(range(len(sub))[::-1])
        ax.set_yticklabels([shorten(str(p), 32) for p in sub["panel"]], fontsize=7)
        ax.set_xlabel(f"{ycol} (snapshot at {df['ts'].iloc[0].strftime('%Y-%m-%d %H:%M UTC')})")
        cap = f"snapshot: {ycol} for top {len(sub)} of {df['panel'].nunique()} panels"
    elif has_panel and nums:
        ycol = nums[0]
        for k, sub in df.groupby("panel"):
            s = sub.copy()
            s[ycol] = pd.to_numeric(s[ycol], errors="coerce")
            s = s.dropna(subset=[ycol]).sort_values("ts")
            if len(s) >= 1:
                ax.plot(s["ts"], s[ycol], label=shorten(str(k), 22), linewidth=0.9)
        ax.set_ylabel(ycol)
        n_panels = df["panel"].nunique()
        if n_panels <= 12:
            ax.legend(loc="best", fontsize=7, ncols=2)
        cap = f"{len(df)} rows over {n_panels} panels"
    elif nums:
        for c in nums[:4]:
            y = pd.to_numeric(df[c], errors="coerce")
            ax.plot(df["ts"], y, label=c, linewidth=0.9)
        if len(nums) > 1:
            ax.legend(loc="best", fontsize=8)
        ax.set_ylabel(nums[0] if len(nums) == 1 else "value")
        cap = f"{len(df)} rows, {len(nums)} numeric series"
    elif cat_cols:
        col = cat_cols[0]
        vc = df[col].astype(str).value_counts().head(20)
        ax.barh(range(len(vc))[::-1], vc.values)
        ax.set_yticks(range(len(vc))[::-1])
        ax.set_yticklabels([shorten(str(x), 30) for x in vc.index], fontsize=7)
        ax.set_xlabel(f"count of {col}")
        cap = f"{len(df)} rows; top-20 of {col}"
    else:
        ax.text(0.5, 0.5, "no plottable column", ha="center", va="center", transform=ax.transAxes)

    if df["ts"].nunique() > 1:
        loc = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return cap


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────


def main() -> int:
    sources = yaml.safe_load(open(ROOT / "sources.yaml"))
    rows = []
    for src in sources:
        sid = src["id"]
        loader = LOADERS.get(sid)
        if loader is None:
            print(f"{sid:40s}  no loader")
            continue
        try:
            df = loader()
        except Exception as e:                      # noqa: BLE001
            print(f"{sid:40s}  load failed: {e}")
            df = pd.DataFrame()
        out = PLOTS_DIR / f"{sid}.png"
        try:
            cap = plot_one(sid, src, df, out)
        except Exception as e:                      # noqa: BLE001
            cap = f"plot failed: {e}"
            fig, ax = plt.subplots(figsize=(8, 2), dpi=100)
            ax.text(0.5, 0.5, cap, ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="red")
            ax.axis("off"); fig.savefig(out); plt.close(fig)
        rows.append((sid, src, cap))
        print(f"{sid:40s}  {cap}")

    # Markdown
    md = ["# Timeframe benchmark — source plots", ""]
    md.append(f"Generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} from `samples/*` payloads (verified {len(rows)} of {len(sources)} sources).")
    md.append("")
    md.append("Each plot title shows: `source_id (domain, ISO 8601 cadence, novelty)`.")
    md.append("Panels are stacked on shared axes with one line per panel value.")
    md.append("")
    counts: dict[str, list] = {}
    for sid, src, cap in rows:
        counts.setdefault(src["domain"], []).append((sid, src, cap))

    domain_order = ["econ_fin", "energy", "nature", "healthcare", "sales", "transport", "web_cloudops"]
    for d in domain_order:
        if d not in counts:
            continue
        md.append(f"## {d}  ({len(counts[d])} sources)")
        md.append("")
        for sid, src, cap in counts[d]:
            md.append(f"### `{sid}`")
            md.append(f"**{src['name']}**  ")
            md.append(f"archetypes: `{', '.join(src['archetypes'])}` · novelty: `{src['pretraining_novelty']}` · cadence: `{src['frequency']}`  ")
            if src.get("notes"):
                md.append(f"_{shorten(src['notes'], 220)}_  ")
            md.append(f"_{cap}_")
            md.append("")
            md.append(f"![{sid}](plots/{sid}.png)")
            md.append("")
    (REPORTS_DIR / "REPORT.md").write_text("\n".join(md))
    print(f"\nWrote {REPORTS_DIR/'REPORT.md'} ({len(rows)} sources)")

    # HTML mirror — quicker to view in a browser
    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Timeframe benchmark — source plots</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;line-height:1.45;color:#222;}",
        "h1{margin-bottom:.2em} h2{margin-top:2.4em;border-bottom:1px solid #ccc;padding-bottom:.2em;} h3{margin-top:1.6em;font-family:monospace;}",
        "img{max-width:100%;display:block;margin:.6em 0 1.6em 0;border:1px solid #eee;}",
        ".meta{color:#555;font-size:.92em} .cap{color:#999;font-size:.85em;font-style:italic}",
        ".tagrow code{background:#f3f3f3;padding:0 .3em;border-radius:.2em;font-size:.85em}",
        "</style></head><body>",
        f"<h1>Timeframe benchmark — source plots</h1>",
        f"<p>Generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}. {len(rows)} verified sources.</p>",
        "<p>Each plot title shows: <code>source_id (domain, ISO 8601 cadence, novelty)</code>.</p>",
    ]
    for d in domain_order:
        if d not in counts:
            continue
        html_parts.append(f"<h2>{d} — {len(counts[d])} sources</h2>")
        for sid, src, cap in counts[d]:
            html_parts.append(f"<h3>{sid}</h3>")
            html_parts.append(f"<p class='meta'><strong>{src['name']}</strong></p>")
            html_parts.append(
                f"<p class='tagrow'>archetypes <code>{', '.join(src['archetypes'])}</code> · "
                f"novelty <code>{src['pretraining_novelty']}</code> · "
                f"cadence <code>{src['frequency']}</code></p>"
            )
            if src.get("notes"):
                html_parts.append(f"<p class='meta'>{shorten(src['notes'], 220)}</p>")
            html_parts.append(f"<p class='cap'>{cap}</p>")
            html_parts.append(f"<img src='plots/{sid}.png' alt='{sid}'>")
    html_parts.append("</body></html>")
    (REPORTS_DIR / "REPORT.html").write_text("\n".join(html_parts))
    print(f"Wrote {REPORTS_DIR/'REPORT.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
