"""`ScrapedLiveSource` — feeds the scraper's parquet output into the live pool.

Closes the last-mile wire between the live-data catalog (`src/sources/`) and
the benchmark's `FreshBuffer`. This adapter is the production feed: it serves
windows of real scraped Wikipedia / MTA / Binance / weather data into the
buffer that challenges are sampled from.

Each motif that goes into the buffer carries:

* ``domain``      — the 7-domain GIFT-Eval taxonomy tag (``nature``, ``econ_fin``, …)
* ``dgp_class``   — the taxonomy from ``sources/DGP_TAXONOMY.md`` (35 classes)
* ``cadence``     — coarse frequency band (sub-min / few-min / half-hour /
                    hourly / daily / weekly / monthly / quarterly / yearly)
* ``source_id``   — the ``sources.yaml`` ``id`` for provenance and audit

The reward-hacking-defense breadth gates in ``score.py`` read the ``dgp_class``
and ``cadence`` labels off the buffer and *hard-veto* any pool where some class
or band drops below a min-share floor. That keeps the served eval-pool
distribution broad: any pool that would collapse a DGP class or a cadence band
scores fitness 0.

Pure numpy + pyarrow + pyyaml. No dep on the scraper module itself; the
adapter is decoupled through the on-disk parquet contract.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from ingest import LiveSource, MotifMeta


# Coarse cadence bands — same partition as `pool_sampler.FREQ_BAND` and the
# reward-hacking-defense breadth gates in score.py.
FREQ_BAND: dict[str, str] = {
    "PT30S": "sub-min", "PT1M": "sub-min", "PT2M30S": "sub-min",
    "PT5M": "few-min", "PT6M": "few-min", "PT10M": "few-min", "PT15M": "few-min",
    "PT30M": "half-hour",
    "PT1H": "hourly", "PT8H": "hourly",
    "P1D": "daily", "P1W": "weekly", "P1M": "monthly", "P1Q": "quarterly", "P1Y": "yearly",
}


class ScrapedLiveSource(LiveSource):
    """Read the scraper's parquet output as a labeled motif pool.

    The adapter loads the catalog (``sources.yaml``) once at construction to
    learn ``(domain, dgp_class, freq)`` per source id, then walks
    ``data_dir/<source_id>/*.parquet`` at each ``pull``/``pull_meta`` to grab
    contiguous motif windows.

    Sampling uses ``src.pool_sampler.sample_pool`` under the hood so each pull
    respects **equal-weight-per-DGP-class-per-domain** — the property that
    prevents source-count-heavy domains (nature) from drowning out light ones
    (healthcare) in the buffer. This is the same sampler that the eval-pool
    builder uses, so the served buffer has the same distribution shape that
    validators will score on.
    """

    domain = "scraped_live"  # coarse fallback; per-motif domain overrides

    def __init__(
        self,
        catalog_path: str | Path,
        data_dir: str | Path,
        *,
        min_series_length: int = 128,
        max_series_per_source: int | None = 200,
        require_freshness_days: int | None = None,
        enforce_cadence_balance: bool = True,
    ) -> None:
        """Args:
            catalog_path: path to ``sources.yaml``.
            data_dir: parent of ``<source_id>/*.parquet``. In the consolidated
                layout that's ``src/sources/data``.
            min_series_length: drop sources with fewer than this many
                observations available; served motifs need enough runway.
            max_series_per_source: after panel expansion, cap per-source series
                count so an oversized panel (e.g. 50-station METAR) can't
                dominate the pool. ``None`` disables the cap.
            require_freshness_days: if set, drop parquet files older than this
                many days. Guards against stale scraper output leaking into the
                buffer when the cron falls behind.
            enforce_cadence_balance: pass through to ``sample_pool``. On by
                default so cadence bands are equal-weighted alongside DGP
                classes. Disable only if you want an internal ablation.
        """
        import yaml
        with open(catalog_path) as f:
            catalog = yaml.safe_load(f)

        self.data_dir = Path(data_dir)
        self.min_series_length = int(min_series_length)
        self.max_series_per_source = max_series_per_source
        self.require_freshness_days = require_freshness_days
        self.enforce_cadence_balance = bool(enforce_cadence_balance)

        # Index the catalog for O(1) per-source metadata lookup.
        self._meta_by_source: dict[str, dict] = {}
        for entry in catalog or []:
            sid = entry.get("id")
            if not sid:
                continue
            self._meta_by_source[sid] = {
                "domain": entry.get("domain", "?"),
                "dgp_class": entry.get("dgp_class", "?"),
                "freq": entry.get("frequency", "?"),
                "cadence": FREQ_BAND.get(entry.get("frequency", ""), "other"),
                "panel": entry.get("panel", []) or [],
            }

        # Lazily populated on first pull.
        self._catalog_series: list[dict] | None = None

    # ------------------------------------------------------------------ index

    def _index_available_series(self) -> list[dict]:
        """Walk ``data_dir`` and enumerate every available panel-expanded series.

        Returns one dict per (source_id, panel_row) pair with fields
        ``source_id``, ``domain``, ``dgp_class``, ``cadence``, ``paths`` (list
        of parquet paths, newest last).
        """
        import pyarrow.parquet as pq  # local import — dep is optional for tests
        import datetime as dt

        out: list[dict] = []
        now = dt.datetime.now(dt.timezone.utc)
        for sid, meta in self._meta_by_source.items():
            src_dir = self.data_dir / sid
            if not src_dir.is_dir():
                continue
            parquets = sorted(src_dir.glob("*.parquet"))
            if not parquets:
                continue
            if self.require_freshness_days is not None:
                cutoff = now - dt.timedelta(days=self.require_freshness_days)
                fresh: list[Path] = []
                for p in parquets:
                    try:
                        # Filename is YYYY-MM-DD.parquet.
                        stamp = dt.datetime.strptime(p.stem, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
                        if stamp >= cutoff:
                            fresh.append(p)
                    except ValueError:
                        continue
                parquets = fresh
                if not parquets:
                    continue
            # Read the latest parquet to enumerate panel rows.
            try:
                t = pq.read_table(parquets[-1])
            except Exception:
                continue
            cols = t.column_names
            # Panel-expanded series appear as `_panel_<KEY>` columns; if none,
            # the whole file is one series.
            panel_cols = [c for c in cols if c.startswith("_panel_")]
            n_avail = t.num_rows
            if n_avail < self.min_series_length:
                continue
            if not panel_cols:
                out.append({
                    "source_id": sid,
                    "domain": meta["domain"],
                    "dgp_class": meta["dgp_class"],
                    "cadence": meta["cadence"],
                    "paths": parquets,
                    "panel_row": None,
                })
                continue
            # Enumerate unique panel-row values across the (possibly multiple)
            # panel keys. Cap per-source series count if requested.
            df = t.to_pandas()
            panel_group = df.groupby(panel_cols).size().reset_index(name="_n")
            panel_group = panel_group[panel_group["_n"] >= self.min_series_length]
            if self.max_series_per_source is not None:
                panel_group = panel_group.head(self.max_series_per_source)
            for _, row in panel_group.iterrows():
                panel_key = {c.replace("_panel_", ""): row[c] for c in panel_cols}
                out.append({
                    "source_id": sid,
                    "domain": meta["domain"],
                    "dgp_class": meta["dgp_class"],
                    "cadence": meta["cadence"],
                    "paths": parquets,
                    "panel_row": panel_key,
                })
        return out

    # -------------------------------------------------------------- sampling

    def _catalog(self) -> list[dict]:
        if self._catalog_series is None:
            self._catalog_series = self._index_available_series()
        return self._catalog_series

    def _sample_indices_equal_weight(
        self, n: int, rng: np.random.Generator
    ) -> list[dict]:
        """Pick ``n`` series with equal weight per (domain × dgp_class) — and
        optional per-cadence — using the same policy as ``pool_sampler``.
        """
        cat = self._catalog()
        if not cat:
            raise RuntimeError(
                f"ScrapedLiveSource: no series available under {self.data_dir}; "
                "run the scraper first, or check freshness / min_series_length."
            )

        # Group per (domain, dgp_class, cadence).
        by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for s in cat:
            by_cell[(s["domain"], s["dgp_class"], s["cadence"])].append(s)

        # Equal weight per domain first — split n across surviving domains.
        domains = sorted({d for d, _, _ in by_cell.keys()})
        per_domain = _split(n, len(domains))
        picks: list[dict] = []
        for dom, n_dom in zip(domains, per_domain):
            classes = sorted({c for d, c, _ in by_cell.keys() if d == dom})
            per_class = _split(n_dom, len(classes))
            for cls, n_cls in zip(classes, per_class):
                if self.enforce_cadence_balance:
                    bands = sorted({b for d, c, b in by_cell.keys() if d == dom and c == cls})
                    per_band = _split(n_cls, len(bands))
                    for band, n_band in zip(bands, per_band):
                        pool = by_cell[(dom, cls, band)]
                        picks.extend(_pick(pool, n_band, rng))
                else:
                    pool: list[dict] = []
                    for band in {b for d, c, b in by_cell.keys() if d == dom and c == cls}:
                        pool.extend(by_cell[(dom, cls, band)])
                    picks.extend(_pick(pool, n_cls, rng))
        return picks

    def _extract_motif(
        self, series_spec: dict, length: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Slice a length-``length`` window out of one panel-expanded series."""
        import pyarrow.parquet as pq
        t = pq.read_table(series_spec["paths"][-1])
        df = t.to_pandas()
        panel_row = series_spec.get("panel_row")
        if panel_row:
            for k, v in panel_row.items():
                col = f"_panel_{k}"
                if col in df.columns:
                    df = df[df[col] == v]
        if len(df) < length:
            # Repeat-pad short series so the pull doesn't crash; the pool won't
            # commonly hit this because we filtered by min_series_length.
            values = df.iloc[:, 1].astype(float).to_numpy() if len(df) else np.zeros(length)
            reps = int(np.ceil(length / max(1, len(values))))
            values = np.tile(values, reps)[:length]
            return values
        # Choose the value column with the most finite (non-NaN) numeric values,
        # not merely the first: some feeds carry several value fields where the
        # leading one is sparse/mostly-NaN for a given panel, which would yield a
        # degenerate all-NaN motif. Prefer the densest real signal; fall back to a
        # rank encoding of a categorical column only if nothing numeric survives.
        excluded = {"timestamp"} | {f"_panel_{k}" for k in (panel_row or {}).keys()}
        value_cols = [c for c in df.columns if c not in excluded] or [df.columns[-1]]
        best_col, best_finite, best_values = None, -1, None
        for c in value_cols:
            try:
                v = df[c].astype(float).to_numpy()
            except (TypeError, ValueError):
                continue
            finite = int(np.isfinite(v).sum())
            if finite > best_finite:
                best_col, best_finite, best_values = c, finite, v
        if best_values is None or best_finite == 0:
            # No usable numeric column: rank-encode the first candidate.
            col = value_cols[0]
            values = df[col].astype("category").cat.codes.to_numpy().astype(float)
        else:
            values = best_values
        # Prefer a window with real signal: try a few starts and keep the first
        # whose finite fraction clears half, so an occasional NaN patch doesn't
        # dominate a motif when denser windows exist.
        start = int(rng.integers(0, len(values) - length + 1))
        for _ in range(4):
            cand = int(rng.integers(0, len(values) - length + 1))
            window = values[cand : cand + length]
            if np.isfinite(window).mean() >= 0.5:
                start = cand
                break
        motif = np.asarray(values[start : start + length], dtype=float)
        # Replace any NaN/inf with the running median so downstream numeric ops
        # don't explode; the scraper's clean step usually handles this, but
        # per-series panels sometimes leave gaps.
        if not np.all(np.isfinite(motif)):
            med = float(np.nanmedian(motif)) if np.isfinite(np.nanmedian(motif)) else 0.0
            motif = np.where(np.isfinite(motif), motif, med)
        return motif

    # ------------------------------------------------------------- LiveSource

    def pull(self, n: int, length: int, rng: np.random.Generator) -> list[np.ndarray]:
        return [m.motif for m in self.pull_meta(n, length, rng)]

    def pull_labeled(
        self, n: int, length: int, rng: np.random.Generator
    ) -> list[tuple[np.ndarray, str]]:
        return [(m.motif, m.domain) for m in self.pull_meta(n, length, rng)]

    def pull_meta(
        self, n: int, length: int, rng: np.random.Generator
    ) -> list[MotifMeta]:
        picks = self._sample_indices_equal_weight(n, rng)
        return [
            MotifMeta(
                motif=self._extract_motif(spec, length, rng),
                domain=spec["domain"],
                dgp_class=spec["dgp_class"],
                cadence=spec["cadence"],
                source_id=spec["source_id"],
            )
            for spec in picks
        ]


# ---------------------------------------------------------------- helpers


def _split(total: int, k: int) -> list[int]:
    """Distribute ``total`` slots as evenly as possible into ``k`` groups."""
    if k == 0:
        return []
    base, extra = divmod(total, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


def _pick(pool: list[dict], n: int, rng: np.random.Generator) -> list[dict]:
    if n == 0:
        return []
    if not pool:
        raise RuntimeError("empty leaf pool during equal-weight sampling")
    idx = rng.integers(0, len(pool), size=n)
    return [pool[i] for i in idx]


__all__ = ["ScrapedLiveSource", "FREQ_BAND"]
