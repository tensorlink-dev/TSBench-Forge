"""Multivariate (coupled-channel) audit: which sources earn an ``mv_channels`` tag.

The cascade eval pool is growing a multivariate mode (DEC-TB-0004): a source
whose parquet rows carry several value columns on one shared timestamp index
can be packed into a single aligned (C, L) window — *if* the channels are
genuinely coupled. The contract field is ``mv_channels: [col, ...]`` on a
catalog entry, scoped to columns within a single ``(source, panel_row)``.

This module is the admission gate for that tag. Column count is not the test,
and neither is correlation — a |corr| filter only removes duplicates. Under
per-variate scoring the only thing that earns a multivariate reward is
**cross-predictiveness**: sibling channels' *lags* must improve a held-out
one-step forecast of a channel beyond that channel's own lags. So per source
we fit small ridge autoregressions per channel — own-lags-only vs
own+sibling-lags — on the front of the aligned history and compare mean
absolute error on the held-out tail, at h=1 and a direct h=16.

Raw gain is not the admission bar, because it is inflated by shared
seasonality: a sibling with the same diurnal cycle "helps" even when its
actual values are scrambled in time (measured: a bikes/docks feed showed a
larger gain from time-shifted siblings than from real ones). So every gain is
placebo-adjusted — sibling lag blocks are re-scored with each sibling
circularly time-shifted (autocorrelation preserved, cross-information
destroyed), and the admitted quantity is ``real_gain - placebo_gain``. A
source is admitted when that adjusted gain clears ``--min-gain`` (default 2%)
on some channel at some horizon.

Evidence accounting (why the summary counts SOURCES, not series): cascade's
KOTH bootstrap clusters by source, so a 200-station panel with coupled
channels is one cluster — one unit of evidence. The numbers that decide
whether the multivariate mode can arm are printed at the end: distinct
admitted sources vs the ~12 N_eff floor (and the min_clusters≈30 target), and
the admitted share of the daily-and-faster pool vs the ~38% margin threshold.

Read-only: never writes the catalog. The tagging pass (A2) consumes the JSON
report this emits.

    # Full audit (streams one source at a time; safe on the 4GB box):
    .venv/bin/python -m source_discovery.mv_audit --out mv_audit_report.json

    # Spot-check a few sources:
    .venv/bin/python -m source_discovery.mv_audit --ids ndbc_buoy_realtime aemo_nem_5min
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Columns that are numeric on disk but are identifiers/geometry, not signals.
_JUNK_COL = re.compile(
    r"(^|_)(id|ids|uid|type|status|order|code|codes|category|catid|domainid"
    r"|lat|lon|latitude|longitude|year|month|day|hour|rank)$|shape__|_id$",
    re.IGNORECASE,
)

MAX_CHANNELS = 8          # contract cap on len(mv_channels)
COLLINEAR_ABS_CORR = 0.999  # duplicate-channel drop (NOT the admission test)
GAP_FACTOR = 8.0          # mirror the samplers: lag windows never span a gap
                          # > 8x the median spacing (weekends survive, cron
                          # outages don't fabricate one-step transitions)
HOLDOUT_FRAC = 0.25
HORIZONS = (1, 16)        # one-step + a mid-horizon direct forecast: cascade's
                          # reward is a 64-step window, and slow couplings
                          # (weather leading, demand leading price) only show
                          # up beyond one step
RIDGE_LAMBDA = 1.0        # on standardized channels
TARGET_POINTS = 4000      # aligned complete-case rows to collect per panel row
MIN_POINTS = 128          # below this the test is 'insufficient', not a verdict

# N_eff / margin thresholds from the cascade-side analysis (DEC-CA-0041):
# the summary prints progress against these, it does not enforce them.
NEFF_FLOOR_SOURCES = 12
MIN_CLUSTERS_TARGET = 30
POOL_SHARE_TARGET = 0.38


# --------------------------------------------------------------------- catalog


def _load_catalog(path: Path) -> list[dict]:
    import yaml

    cat = yaml.safe_load(path.read_text())
    return cat["sources"] if isinstance(cat, dict) else cat


def _is_daily_or_faster(freq: str | None) -> bool:
    f = str(freq or "")
    return f.startswith("PT") or f == "P1D"


# ----------------------------------------------------------------- disk access


def _newest_files(data_dir: Path, sid: str, max_files: int) -> list[Path]:
    files = sorted((data_dir / sid).glob("*.parquet"))
    return files[-max_files:] if files else []


def _scan_columns(files: list[Path]) -> tuple[list[str], list[str]]:
    """(panel_cols, candidate value cols) from the newest file's schema."""
    import pyarrow.parquet as pq

    try:
        names = pq.ParquetFile(files[-1]).schema_arrow.names
    except Exception:  # noqa: BLE001 — corrupt newest file: nothing to audit
        return [], []
    pcols = [c for c in names if c.startswith("_panel_")]
    vcols = [
        c for c in names
        if c != "timestamp" and not c.startswith("_panel_") and not _JUNK_COL.search(c)
    ]
    return pcols, vcols


def _collect(
    files: list[Path],
    pcols: list[str],
    vcols: list[str],
    n_rows: int,
    target_points: int,
):
    """Stream day-files newest-first and collect per-panel-row aligned frames.

    Returns ``{panel_key_tuple: DataFrame(timestamp + vcols)}`` for up to
    ``n_rows`` panel rows (the most populated ones in the newest file). Bounded
    memory: batches are filtered as they stream and collection stops per row at
    ``target_points`` — a 12M-row transit panel never sits in RAM whole.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    want = ["timestamp"] + pcols + vcols
    chosen: list[tuple] | None = None
    got: dict[tuple, list] = {}
    counts: dict[tuple, int] = {}

    for f in reversed(files):
        try:
            pf = pq.ParquetFile(f)
        except Exception:  # noqa: BLE001 — one corrupt day must not sink the audit
            continue
        cols = [c for c in pf.schema_arrow.names if c in want]
        if "timestamp" not in cols or not any(c in cols for c in vcols):
            continue
        for batch in pf.iter_batches(batch_size=65536, columns=cols):
            df = batch.to_pandas()
            if pcols:
                present = [c for c in pcols if c in df.columns]
                if len(present) < len(pcols):
                    continue
                if chosen is None:
                    # Most-populated rows of the newest file define the sample.
                    top = df.groupby(pcols).size().sort_values(ascending=False)
                    chosen = [k if isinstance(k, tuple) else (k,) for k in top.index[:n_rows]]
                    for k in chosen:
                        got[k], counts[k] = [], 0
                key = df[pcols].apply(tuple, axis=1)
                for k in chosen:
                    if counts[k] >= target_points:
                        continue
                    sub = df[key == k]
                    if len(sub):
                        got[k].append(sub.drop(columns=pcols))
                        counts[k] += len(sub)
            else:
                if chosen is None:
                    chosen = [()]
                    got[()], counts[()] = [], 0
                if counts[()] < target_points:
                    got[()].append(df)
                    counts[()] += len(df)
        if chosen is not None and all(counts[k] >= target_points for k in got):
            break

    out = {}
    for k, frames in got.items():
        if frames:
            out[k] = pd.concat(frames, ignore_index=True)
    return out


def _parse_ts(ts):
    """UTC-naive datetime64 array, or None when the feed's stamps don't parse.

    Mirrors the eval sampler's parsing (mixed first, then the feed-specific
    formats "mixed" can't infer: wikimedia's YYYYMMDDHH, NDBC's spaced stamps).
    """
    import pandas as pd

    parsed = pd.to_datetime(ts, errors="coerce", utc=True, format="mixed")
    if parsed.notna().mean() <= 0.9:
        for fmt in ("%Y%m%d%H", "%Y %m %d %H %M", "%Y %m %d"):
            alt = pd.to_datetime(ts, errors="coerce", utc=True, format=fmt)
            if alt.notna().mean() > 0.9:
                parsed = alt
                break
    if parsed.notna().mean() <= 0.9:
        return None
    return parsed.dt.tz_localize(None)


# ----------------------------------------------------- the cross-predictive test


def _lag_design(X: np.ndarray, seg_ids: np.ndarray, p: int):
    """One-step design matrices from segmented, standardized channels.

    Rows are targets at time t whose ``p`` lags all lie in the same contiguous
    segment (never spanning a gap). Returns ``(lags, y_index)`` where ``lags``
    is (n, p, C) and ``y_index`` the row positions of the targets in ``X``.
    """
    n = len(X)
    idx = np.arange(p, n)
    ok = np.ones(len(idx), dtype=bool)
    for k in range(p + 1):
        ok &= seg_ids[idx - k] == seg_ids[idx]
    idx = idx[ok]
    lags = np.stack([X[idx - k] for k in range(1, p + 1)], axis=1)  # (n, p, C)
    return lags, idx


def _ridge_mae(A_tr, y_tr, A_te, y_te) -> float:
    """Held-out MAE of a ridge fit (closed form, intercept via augmentation)."""
    A_tr = np.hstack([A_tr, np.ones((len(A_tr), 1))])
    A_te = np.hstack([A_te, np.ones((len(A_te), 1))])
    k = A_tr.shape[1]
    beta = np.linalg.solve(A_tr.T @ A_tr + RIDGE_LAMBDA * np.eye(k), A_tr.T @ y_tr)
    return float(np.mean(np.abs(A_te @ beta - y_te)))


def _cross_pred_gains(df, vcols: list[str]) -> dict:
    """Per-channel held-out gain of own+sibling lags over own lags.

    Returns ``{"n": int, "channels": [...], "gains": {col: adjusted_gain}}``
    or ``{"status": reason}`` when no verdict is possible. ``gain`` is the
    placebo-adjusted fractional MAE reduction — 0.03 means real sibling lags
    beat time-shifted (seasonality-matched) sibling lags by 3 points of the
    channel's own-lags error.
    """
    import pandas as pd

    ts = _parse_ts(df["timestamp"])
    if ts is None:
        return {"status": "no_clock"}
    df = df.assign(_ts=ts.to_numpy()).dropna(subset=["_ts"])
    df = df.sort_values("_ts").drop_duplicates(subset=["_ts"], keep="last")

    # Numeric, non-constant channels with decent coverage.
    cols, mat = [], []
    for c in vcols:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(v).mean() < 0.5:
            continue
        finite = v[np.isfinite(v)]
        if len(np.unique(finite)) < 3:
            continue
        cols.append(c)
        mat.append(v)
    if len(cols) < 2:
        return {"status": "lt2_channels"}

    X = np.column_stack(mat)
    keep_rows = np.isfinite(X).all(axis=1)
    X, t = X[keep_rows], df["_ts"].to_numpy()[keep_rows]
    if len(X) < MIN_POINTS:
        return {"status": "insufficient", "n": int(len(X))}

    # Duplicate-channel drop (this is dedup, not the admission test).
    keep: list[int] = []
    corr = np.corrcoef(X, rowvar=False)
    for j in range(len(cols)):
        if all(abs(corr[j, k]) < COLLINEAR_ABS_CORR or np.isnan(corr[j, k]) for k in keep):
            keep.append(j)
        if len(keep) == MAX_CHANNELS:
            break
    if len(keep) < 2:
        return {"status": "collinear"}
    cols = [cols[j] for j in keep]
    X = X[:, keep]

    # Segment at gaps so lag rows never span a cron outage.
    d = np.diff(t).astype("timedelta64[s]").astype(float)
    pos = d[d > 0]
    med = float(np.median(pos)) if pos.size else 0.0
    seg_ids = np.zeros(len(X), dtype=int)
    if med > 0:
        seg_ids[1:] = np.cumsum(d > GAP_FACTOR * med)

    n = len(X)
    p = 8 if n >= 600 else (4 if n >= 240 else 2)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    lags, idx = _lag_design(Z, seg_ids, p)
    if len(idx) < MIN_POINTS:
        return {"status": "insufficient", "n": int(len(idx))}

    # Placebo channels: rolled copies. Marginal dynamics (autocorrelation,
    # seasonality, persistence) survive; alignment with the target does not.
    # No single offset is a fair null for every dynamics type — persistent
    # levels leak through a small shift, trends invert across a large one's
    # wrap seam — so TWO surrogates are built and the admission subtracts the
    # stronger (worst-case) of them.
    lags_p_list = []
    for base, step in ((max(2 * p + 1, len(Z) // 20), 13), (len(Z) // 3, 37)):
        Zp = np.column_stack(
            [np.roll(Z[:, c], base + step * c) for c in range(Z.shape[1])]
        )
        lags_p_list.append(_lag_design(Zp, seg_ids, p)[0])

    by_horizon: dict[int, dict[str, float]] = {}
    for h in HORIZONS:
        # Direct h-step: the same lag block predicts X[i + h - 1]; targets must
        # stay inside the lag row's segment so a gap never fakes a transition.
        tgt = idx + (h - 1)
        ok = (tgt < len(Z)) & (seg_ids[np.minimum(tgt, len(Z) - 1)] == seg_ids[idx])
        idx_h, lags_h, tgt = idx[ok], lags[ok], tgt[ok]
        if len(idx_h) < MIN_POINTS:
            continue
        lagsp_h_list = [lp[ok] for lp in lags_p_list]
        cut = int(len(idx_h) * (1 - HOLDOUT_FRAC))
        g: dict[str, float] = {}
        for j, cname in enumerate(cols):
            y = Z[tgt, j]
            own = lags_h[:, :, j]                    # (n, p)
            full = lags_h.reshape(len(idx_h), -1)    # (n, p*C)
            mae_own = _ridge_mae(own[:cut], y[:cut], own[cut:], y[cut:])
            if mae_own < 1e-9:
                continue  # channel is trivially predictable; a gain here is noise
            mae_full = _ridge_mae(full[:cut], y[:cut], full[cut:], y[cut:])
            # Placebo designs: same shape as `full`, but siblings come from a
            # rolled copy (the target's own lags stay real in both).
            placebo = 0.0
            for lagsp_h in lagsp_h_list:
                plac = lags_h.copy()
                for c in range(len(cols)):
                    if c != j:
                        plac[:, :, c] = lagsp_h[:, :, c]
                plac = plac.reshape(len(idx_h), -1)
                mae_plac = _ridge_mae(plac[:cut], y[:cut], plac[cut:], y[cut:])
                placebo = max(placebo, 1.0 - mae_plac / mae_own)
            # Adjusted gain: real-sibling gain minus the WORST-CASE placebo,
            # floored at zero — a placebo that HURTS (ridge noise, wrap-seam
            # artifacts) must not donate credit to the real gain.
            real = 1.0 - mae_full / mae_own
            g[cname] = round(real - placebo, 4)
        if g:
            by_horizon[h] = g
    if not by_horizon:
        return {"status": "degenerate"}
    gains = {}
    for g in by_horizon.values():
        for c, v in g.items():
            gains[c] = max(gains.get(c, v), v)
    return {"n": int(len(idx)), "p": p, "channels": cols, "gains": gains,
            "by_horizon": {str(h): g for h, g in by_horizon.items()}}


# ------------------------------------------------------------------ per source


def audit_source(
    src: dict,
    data_dir: Path,
    *,
    min_gain: float,
    max_files: int,
    panel_rows: int,
) -> dict:
    sid = src["id"]
    out = {"id": sid, "domain": src.get("domain"), "frequency": src.get("frequency")}
    files = _newest_files(data_dir, sid, max_files)
    if not files:
        return {**out, "status": "no_data"}
    pcols, vcols = _scan_columns(files)
    if len(vcols) < 2:
        return {**out, "status": "univariate"}

    frames = _collect(files, pcols, vcols, panel_rows, TARGET_POINTS)
    if not frames:
        return {**out, "status": "no_rows"}

    rows = []
    for key, df in frames.items():
        res = _cross_pred_gains(df, vcols)
        res["panel_row"] = list(map(str, key)) if key else None
        rows.append(res)
    tested = [r for r in rows if "gains" in r]
    if not tested:
        # Every sampled row failed the same way; surface the dominant reason.
        reasons = [r.get("status", "?") for r in rows]
        return {**out, "status": max(set(reasons), key=reasons.count), "rows": rows}

    # Admit when at least half the tested panel rows show a channel whose
    # held-out error drops by min_gain — one lucky station must not tag a
    # 50-station panel, and one dud must not veto it.
    passing = [r for r in tested if max(r["gains"].values()) >= min_gain]
    admitted = len(passing) * 2 >= len(tested)
    # The tag proposal is the channel set of the best-evidenced row: channel
    # membership is a per-source contract, so rows must agree in practice —
    # disagreement shows up in `rows` for the A2 spot-check to catch.
    best = max(tested, key=lambda r: max(r["gains"].values()))
    return {
        **out,
        "status": "admitted" if admitted else "rejected",
        "mv_channels": best["channels"] if admitted else None,
        "best_gain": max(best["gains"].values()),
        "rows_tested": len(tested),
        "rows_passing": len(passing),
        "rows": rows,
    }


# --------------------------------------------------------------------- summary


def _series_count(data_dir: Path, sid: str, cap: int = 200) -> int:
    """Panel-row count of the newest day-file, capped like cascade's 200/source."""
    import pyarrow.parquet as pq

    files = sorted((data_dir / sid).glob("*.parquet"))
    if not files:
        return 0
    try:
        pf = pq.ParquetFile(files[-1])
    except Exception:  # noqa: BLE001
        return 0
    pcols = [c for c in pf.schema_arrow.names if c.startswith("_panel_")]
    if not pcols:
        return 1
    try:
        t = pf.read(columns=pcols).to_pandas()
    except Exception:  # noqa: BLE001
        return 1
    return min(cap, len(t.drop_duplicates())) or 1


def summarize(results: list[dict], catalog: list[dict], data_dir: Path) -> dict:
    from collections import Counter

    by_status = Counter(r["status"] for r in results)
    admitted = [r for r in results if r["status"] == "admitted"]
    adm_by_dom = Counter(r["domain"] for r in admitted)

    # Pool share on cascade's ruler: a coupled group is ONE series, so the MV
    # share is admitted sources' series over all daily-and-faster series.
    freq = {s["id"]: s.get("frequency") for s in catalog}
    fast_total = fast_mv = 0
    for s in catalog:
        if s.get("disabled") or not _is_daily_or_faster(s.get("frequency")):
            continue
        fast_total += _series_count(data_dir, s["id"])
    for r in admitted:
        if _is_daily_or_faster(freq.get(r["id"])):
            fast_mv += _series_count(data_dir, r["id"])
    share = fast_mv / fast_total if fast_total else 0.0

    return {
        "statuses": dict(by_status),
        "admitted_sources": len(admitted),
        "admitted_by_domain": dict(adm_by_dom),
        "neff_floor": NEFF_FLOOR_SOURCES,
        "min_clusters_target": MIN_CLUSTERS_TARGET,
        "fast_pool_series": fast_total,
        "fast_mv_series": fast_mv,
        "fast_mv_share": round(share, 4),
        "pool_share_target": POOL_SHARE_TARGET,
    }


# ------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--catalog", default="src/sources/sources.yaml", type=Path)
    ap.add_argument("--data-dir", default="src/sources/data", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="write full JSON report here")
    ap.add_argument("--ids", nargs="*", default=None, help="audit only these source ids")
    ap.add_argument("--min-gain", type=float, default=0.02,
                    help="held-out MAE reduction that admits a channel (default 2%%)")
    ap.add_argument("--max-files", type=int, default=90,
                    help="newest day-files to stream per source")
    ap.add_argument("--panel-rows", type=int, default=3,
                    help="panel rows sampled per source")
    args = ap.parse_args(argv)

    catalog = _load_catalog(args.catalog)
    todo = [s for s in catalog if not s.get("disabled")]
    if args.ids:
        todo = [s for s in todo if s["id"] in set(args.ids)]

    results = []
    for i, src in enumerate(todo, 1):
        r = audit_source(
            src, args.data_dir,
            min_gain=args.min_gain, max_files=args.max_files, panel_rows=args.panel_rows,
        )
        results.append(r)
        if r["status"] in ("admitted", "rejected"):
            print(f"[{i}/{len(todo)}] {r['id']}: {r['status']} "
                  f"(best_gain={r.get('best_gain')}, rows {r.get('rows_passing')}/{r.get('rows_tested')})",
                  file=sys.stderr)
        elif i % 200 == 0:
            print(f"[{i}/{len(todo)}] ...", file=sys.stderr)

    summary = summarize(results, catalog, args.data_dir)
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "results": results}, indent=1))
        print(f"report -> {args.out}", file=sys.stderr)

    print(json.dumps(summary, indent=2))
    adm = [r for r in results if r["status"] == "admitted"]
    for r in sorted(adm, key=lambda r: -r["best_gain"]):
        print(f"  {r['id']:50s} {r['domain']:12s} gain={r['best_gain']:+.3f} "
              f"C={len(r['mv_channels'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
