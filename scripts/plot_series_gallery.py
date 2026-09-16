#!/usr/bin/env python3
"""Render the scraped series as paginated small-multiples galleries.

    PYTHONPATH=src python3 scripts/plot_series_gallery.py                # ALL series, 16/page
    PYTHONPATH=src python3 scripts/plot_series_gallery.py --sample 20   # one-page sample

Every panel-expanded series in the catalog gets a facet showing its most
recent <=points observations, extracted through the benchmark's own loader so
panels show exactly what the eval serves. Pages are written to
notebooks/figures/series_gallery/page_NN.png (sample mode: series_gallery.png).
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


def draw_panel(ax, mdates, spec, vals, stamps):
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
    loc = mdates.AutoDateLocator(maxticks=4)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def render_page(plt, mdates, panels, title, out_path, per_row=4):
    nrows = max(1, int(np.ceil(len(panels) / per_row)))
    fig, axes = plt.subplots(nrows, per_row, figsize=(16, 3.0 * nrows), dpi=150,
                             squeeze=False)
    fig.patch.set_facecolor("white")
    for ax, (spec, vals, stamps) in zip(axes.flat, panels):
        draw_panel(ax, mdates, spec, vals, stamps)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left", y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="plot only N series on one page (default: ALL series, paginated)")
    ap.add_argument("--per-page", type=int, default=16)
    ap.add_argument("--points", type=int, default=1000)
    ap.add_argument("--out-dir", default=str(REPO / "notebooks/figures/series_gallery"))
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

    # Order: priority sources first, then the rest alphabetically; panel rows
    # stay grouped under their source so facets read source-by-source.
    ordered: list[dict] = []
    for sid in PRIORITY:
        ordered.extend(by_source.get(sid, []))
    for sid in sorted(by_source):
        if sid not in PRIORITY:
            ordered.extend(by_source[sid])

    if args.sample:
        # one series per source until --sample panels, single page
        seen: set[str] = set()
        ordered = [s for s in ordered if not (s["source_id"] in seen or seen.add(s["source_id"]))]
        ordered = ordered[: args.sample]

    panels: list[tuple[dict, np.ndarray, np.ndarray]] = []
    skipped = 0
    for spec in ordered:
        got = tail_series(src, spec, args.points)
        if got is None:
            skipped += 1
            continue
        panels.append((spec, *got))

    if args.sample:
        out = REPO / "notebooks/figures/series_gallery.png"
        render_page(plt, mdates, panels,
                    f"TSBench-Forge series gallery — most recent ≤{args.points} points per series",
                    out)
        print(f"plotted {len(panels)} series ({skipped} skipped) -> {out}")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("page_*.png"):
        old.unlink()
    pages = [panels[i:i + args.per_page] for i in range(0, len(panels), args.per_page)]
    for p, chunk in enumerate(pages, 1):
        out = out_dir / f"page_{p:02d}.png"
        render_page(plt, mdates, chunk,
                    f"TSBench-Forge series gallery — page {p}/{len(pages)} "
                    f"(most recent ≤{args.points} points per series)", out)
        print(f"page {p:02d}: {len(chunk)} series -> {out}")
    print(f"total {len(panels)} series on {len(pages)} pages ({skipped} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
