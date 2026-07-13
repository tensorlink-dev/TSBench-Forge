#!/usr/bin/env python3
"""Render a small-multiples gallery of the scraped series (most recent points).

    PYTHONPATH=src python3 scripts/plot_series_gallery.py [--n 20] [--points 1000]

Picks one series per source — every 2026-07 batch source first, then a
deterministic domain-diverse sample of the rest — and plots each one's most
recent <=points observations. Output: notebooks/figures/series_gallery.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

PRIORITY = [
    "gemini_btcusd_trades", "crw_virtual_station_sst", "ripe_atlas_rootserver_ping",
    "usace_cwms_timeseries", "datalakes_eawag_lake_obs", "sg_taxi_availability",
    "coinbase_spot_1m", "kraken_futures_btc_funding",
]

LINE = "#31688e"
INK, MUTED, GRID = "#333333", "#777777", "#e3e3e3"


def tail_series(src, spec, n_points):
    """Most recent <=n_points of one panel-expanded series (values, timestamps)."""
    import pandas as pd

    df = src._read_series_frame(spec["paths"])
    if df is None or df.empty:
        return None
    panel_row = spec.get("panel_row")
    if panel_row:
        for k, v in panel_row.items():
            col = f"_panel_{k}"
            if col in df.columns:
                df = df[df[col] == v]
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, format="mixed")
    if ts.notna().mean() <= 0.9:
        for fmt in ("%Y%m%d%H", "%Y %m %d %H %M", "%Y %m %d"):
            alt = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, format=fmt)
            if alt.notna().mean() > 0.9:
                ts = alt
                break
    # Year-less feed stamps (e.g. GLERL "MM-DD") can parse into nonsense dates;
    # keep only plausibly-modern timestamps.
    import pandas as pd
    lo, hi = pd.Timestamp("2000-01-01", tz="UTC"), pd.Timestamp("2100-01-01", tz="UTC")
    keep = ts.notna() & (ts >= lo) & (ts < hi)
    if keep.mean() < 0.5:
        return None
    df, ts = df[keep], ts[keep]
    order = np.argsort(ts.to_numpy(), kind="stable")
    df, ts = df.iloc[order], ts.iloc[order]
    # densest numeric column (mirrors ScrapedLiveSource._extract_motif)
    excluded = {"timestamp"} | {f"_panel_{k}" for k in (panel_row or {}).keys()}
    best_finite, best = -1, None
    for c in [c for c in df.columns if c not in excluded]:
        try:
            v = df[c].astype(float).to_numpy()
        except (TypeError, ValueError):
            continue
        n_fin = int(np.isfinite(v).sum())
        if n_fin > best_finite:
            best_finite, best = n_fin, v
    if best is None or best_finite < 8:
        return None
    vals = best[-n_points:]
    stamps = ts.dt.tz_localize(None).to_numpy()[-n_points:]
    return vals, stamps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--points", type=int, default=1000)
    ap.add_argument("--out", default=str(REPO / "notebooks/figures/series_gallery.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from scraped_source import ScrapedLiveSource

    src = ScrapedLiveSource(REPO / "src/sources/sources.yaml", REPO / "src/sources/data",
                            min_series_length=100)
    specs = src._catalog()
    by_source: dict[str, list[dict]] = {}
    for s in specs:
        by_source.setdefault(s["source_id"], []).append(s)

    candidates: list[dict] = []
    for sid in PRIORITY:
        if sid in by_source:
            candidates.append(by_source[sid][0])
    rng = np.random.default_rng(20260713)
    rest = [sid for sid in sorted(by_source) if sid not in PRIORITY]
    # domain-diverse fill: round-robin over domains
    by_domain: dict[str, list[str]] = {}
    for sid in rest:
        by_domain.setdefault(by_source[sid][0]["domain"], []).append(sid)
    for sids in by_domain.values():
        rng.shuffle(sids)
    while any(by_domain.values()):
        for d in sorted(by_domain):
            if by_domain[d]:
                candidates.append(by_source[by_domain[d].pop()][0])

    # Extract first; only successfully-extracted series get a panel.
    picked: list[tuple[dict, np.ndarray, np.ndarray]] = []
    for spec in candidates:
        if len(picked) >= args.n:
            break
        got = tail_series(src, spec, args.points)
        if got is not None:
            picked.append((spec, *got))

    ncols, nrows = 4, int(np.ceil(len(picked) / 4))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.0 * nrows), dpi=150)
    fig.patch.set_facecolor("white")

    plotted = 0
    for ax, (spec, vals, stamps) in zip(axes.flat, picked):
        ax.plot(stamps, vals, color=LINE, lw=1.0)
        pk = spec.get("panel_row") or {}
        sub = next(iter(pk.values()), "")
        title = spec["source_id"] + (f" · {str(sub)[:22]}" if sub else "")
        ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=3)
        span = (stamps[-1] - stamps[0]).astype("timedelta64[h]").astype(int)
        span_txt = f"{span/24:.1f}d" if span >= 48 else f"{span}h"
        ax.text(0.995, 0.98, f"{len(vals)} pts · {span_txt} · {spec['domain']}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7, color=MUTED)
        ax.grid(True, color=GRID, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(labelsize=7, colors=MUTED, length=2)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=4))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator(maxticks=4)))
        plotted += 1
    for ax in axes.flat[len(picked):]:
        ax.axis("off")

    fig.suptitle(f"TSBench-Forge series gallery — most recent ≤{args.points} points per series",
                 fontsize=12, color=INK, x=0.01, ha="left", y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(args.out, bbox_inches="tight")
    print(f"plotted {plotted}/{len(picked)} series -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
