"""Daily-updated public feeds — the live layer fed by real, refreshing sources.

``live_feeds.py`` wires real series from a handful of static CSV datasets; this
module covers the **daily-updated public sources** catalogued in the sibling
``horizon-forge`` project, re-expressed against this repo's own
:class:`~feeds.DatedLiveSource` contract. **There is no dependency on
horizon-forge**: only the stdlib + numpy, with the network injected via the
``live_feeds`` ``fetch`` callable exactly like the rest of the live layer.

What was ported
---------------
* horizon-forge's declarative **JSON path engine** (:func:`_walk` /
  :func:`_tokenize` / :func:`_apply`), extended here with nested-list
  *flattening* (for doubly-iterated paths like ``[].data[].values[].value``) and
  optional per-record *filtering* (to drop, e.g., PyPI's duplicate mirror rows).
* Adapters for the four payload shapes the daily catalogue actually emits —
  list-of-records JSON (:class:`DatedJsonFeed`), date-keyed-object JSON
  (``dict_keyed`` :class:`DatedJsonFeed`), CSV (reusing
  :class:`live_feeds.DatedCsvFeed`), and whitespace tables
  (:class:`DatedTextFeed`) — plus **panel expansion**, so one templated endpoint
  (``{ARTICLE}`` / ``{PACKAGE}`` / ``{CRATE}``) fans out into many per-instance
  series.
* The curated catalogue itself (:data:`DAILY_REGISTRY`).

What was left behind
--------------------
httpx, pandas, pyarrow, parquet storage, and the cron layer. horizon-forge
*accumulates* history into per-day parquet via an external scheduler; here we
lean on the fact that these endpoints already return a trailing window of
history, fetch it on demand, and let :func:`live_feeds.cached_fetch` snapshot it.

How it plugs in
---------------
Every adapter is a drop-in :class:`feeds.DatedLiveSource` that stamps each window
with the availability time of its last point, so it composes directly with
:class:`feeds.AsOfLiveSource` (vintage gating) and :class:`feeds.DedupFreshBuffer`
(cross-epoch dedup). :func:`build_daily_live_source` mirrors
:func:`live_feeds.build_real_live_source`, fanning the whole catalogue (panels
fully expanded) into a multi-domain ``MixtureLiveSource``.

Catalogue scope
---------------
15 of the 18 daily (``P1D``) sources in horizon-forge's catalogue are ported —
every one that decodes to a univariate numeric daily series. Three are
intentionally excluded because they are *not* time series and forcing them into a
forecasting benchmark would be wrong (see :data:`EXCLUDED`): ``itunes_top_podcasts``
(a daily list of podcast *ids*), ``wikimedia_top_articles_daily`` (a one-day
cross-section of the top articles, not a series — the per-article panel feed
covers Wikipedia pageviews as a proper series instead), and
``neso_stor_notifications`` (irregular event rows with a non-numeric value).

Production note
---------------
Like ``live_feeds``, the bundled URLs are pinned snapshots with fixed date ranges
— fine for proving the wiring and for research. Some endpoints (PyPI, crates,
Treasury, Mauna Loa global trend) return only a few months/hundreds of points, so
they suit shorter ``motif_len`` than the long climate/finance feeds; see each
spec's ``min_points``. A real deployment points these at as-of vendor endpoints
whose new rows are timestamped after the commit beacon.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from domains import _finalize
from feeds import DatedLiveSource, DatedMotif
from ingest import LiveSource
from live_feeds import DatedCsvFeed, Fetcher, cached_fetch, urllib_fetch

# --------------------------------------------------------------------------- #
# JSON path engine (ported from horizon-forge's sources/scraper.py)
# --------------------------------------------------------------------------- #
#
# A path is a dotted/bracketed spec naming where a timestamp or value sequence
# lives inside a JSON payload. ``[]`` lifts the rest of the path over every
# element of a list (returning an aligned list); ``[N]`` indexes a single one.
#
#     'a.b'              -> obj['a']['b']
#     'a[].b'            -> [item['b'] for item in obj['a']]
#     'a[0].b'           -> obj['a'][0]['b']
#     '[].data[].v'      -> nested list, flattened by parse_json_series


def _tokenize(path: str) -> list[tuple[str, str]]:
    """Tokenize a path into ``('key', name)`` / ``('idx', '' | 'N')`` pairs."""
    tokens: list[tuple[str, str]] = []
    i, n = 0, len(path)
    while i < n:
        c = path[i]
        if c == ".":
            i += 1
        elif c == "[":
            j = path.index("]", i)
            tokens.append(("idx", path[i + 1 : j]))
            i = j + 1
        else:
            j = i
            while j < n and path[j] not in ".[":
                j += 1
            tokens.append(("key", path[i:j]))
            i = j
    return tokens


def _apply(obj: Any, tokens: list[tuple[str, str]]) -> Any:
    """Apply tokens to ``obj``. An iteration token (``[]``) lifts the remaining
    path over every element, returning a list aligned with the parent list."""
    if not tokens:
        return obj
    kind, val = tokens[0]
    rest = tokens[1:]

    if kind == "key":
        if isinstance(obj, dict):
            return _apply(obj.get(val), rest)
        if isinstance(obj, list):
            return [_apply(item, [(kind, val)] + rest) for item in obj]
        return None

    # idx
    if val == "":  # iterate every element
        if not isinstance(obj, list):
            return None
        return [_apply(item, rest) for item in obj]
    try:
        idx = int(val)
    except ValueError:
        return None
    if isinstance(obj, list):
        if -len(obj) <= idx < len(obj):
            return _apply(obj[idx], rest)
        return None
    if isinstance(obj, dict):  # numeric-key fallback
        return _apply(obj.get(val), rest)
    return None


def _walk(obj: Any, path: str) -> Any:
    """Walk a JSON-like object using a dotted/bracketed path (``""`` -> ``obj``)."""
    if path == "":
        return obj
    return _apply(obj, _tokenize(path))


def _flatten(x: Any) -> list[Any]:
    """Flatten arbitrarily-nested lists into a flat list of scalars.

    Doubly-iterated paths (``[].data[].values[].value``) resolve to a list of
    lists; since the timestamp and value paths share a prefix they nest
    identically, so flattening both yields aligned scalar sequences.
    """
    if isinstance(x, list):
        out: list[Any] = []
        for e in x:
            out.extend(_flatten(e))
        return out
    return [x]


# --------------------------------------------------------------------------- #
# Timestamp coercion
# --------------------------------------------------------------------------- #

_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y-%m",
    "%Y%m%d",
    "%m/%d/%Y",
    "%Y-%m-%dT%H:%M:%S",
)


def _to_posix(value: Any) -> float | None:
    """Coerce an epoch number or a date/ISO string into POSIX seconds (UTC).

    Handles the shapes the daily catalogue emits: epoch seconds or milliseconds
    (DefiLlama), plain ``YYYY-MM-DD`` (RKI, Open-Meteo, Frankfurter), Wikipedia's
    ``YYYYMMDD00`` stamps, and full ISO-8601 with an offset (USGS). Returns
    ``None`` for anything unparseable so the caller can drop the row.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        n = float(value)
        return n / 1000.0 if n > 1e12 else n  # > 1e12 -> epoch milliseconds
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.isdigit():
            # An all-digit string is a packed date stamp, not an epoch (epoch
            # timestamps arrive as JSON numbers in this catalogue). Wikipedia
            # uses YYYYMMDD00 (a trailing hour field); others YYYYMMDD.
            if len(s) == 10 and s.endswith("00"):
                s = s[:8]
            if len(s) == 8:
                try:
                    return datetime.strptime(s, "%Y%m%d").replace(tzinfo=UTC).timestamp()
                except ValueError:
                    pass
            n = float(s)  # large numeric string -> epoch fallback
            return n / 1000.0 if n > 1e12 else n
        try:  # ISO-8601 (accept a trailing 'Z')
            parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.timestamp()
        except ValueError:
            pass
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=UTC).timestamp()
            except ValueError:
                continue
    return None


def _sorted_arrays(pairs: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Sort ``(time, value)`` pairs ascending by time into ``(values, times)``."""
    pairs.sort(key=lambda p: p[0])
    times = np.asarray([p[0] for p in pairs], dtype=float)
    values = np.asarray([p[1] for p in pairs], dtype=float)
    return values, times


def parse_json_series(
    payload: str | bytes | Any,
    timestamp_field: str,
    value_field: str,
    *,
    filter_field: str = "",
    filter_value: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract aligned, time-sorted ``(values, times)`` from a list-shaped payload.

    ``timestamp_field`` / ``value_field`` are path specs (see :func:`_walk`); each
    must resolve to a (possibly nested) list. The two are flattened and zipped,
    rows with a non-numeric value or unparseable timestamp are dropped, and — when
    ``filter_field`` is given — only rows whose filter cell equals ``filter_value``
    are kept (e.g. PyPI returns a ``with_mirrors`` and a ``without_mirrors`` row
    per date). ``times`` is POSIX seconds.
    """
    data = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
    ts_raw = _walk(data, timestamp_field)
    val_raw = _walk(data, value_field)
    if not isinstance(ts_raw, list) or not isinstance(val_raw, list):
        raise ValueError(
            f"paths did not resolve to lists: "
            f"timestamp_field={timestamp_field!r} value_field={value_field!r}"
        )
    ts_seq = _flatten(ts_raw)
    val_seq = _flatten(val_raw)
    filt_seq = _flatten(_walk(data, filter_field)) if filter_field else None

    pairs: list[tuple[float, float]] = []
    for i in range(min(len(ts_seq), len(val_seq))):
        if filt_seq is not None and (i >= len(filt_seq) or filt_seq[i] != filter_value):
            continue
        ts = _to_posix(ts_seq[i])
        if ts is None:
            continue
        try:
            v = float(val_seq[i])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        pairs.append((ts, v))
    return _sorted_arrays(pairs)


def parse_json_dict_series(
    payload: str | bytes | Any,
    dict_field: str,
    value_field: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(values, times)`` from a *date-keyed object* payload.

    For shapes like Frankfurter's ``{"rates": {"2020-01-02": {"EUR": 0.89}, ...}}``:
    ``dict_field`` names the date-keyed dict (the keys are timestamps) and
    ``value_field`` is the sub-path read from each value (``""`` if the value is
    already a scalar).
    """
    data = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
    d = _walk(data, dict_field)
    if not isinstance(d, dict):
        raise ValueError(f"dict_field {dict_field!r} did not resolve to an object")

    pairs: list[tuple[float, float]] = []
    for key, raw in d.items():
        ts = _to_posix(key)
        if ts is None:
            continue
        cell = _walk(raw, value_field) if value_field else raw
        try:
            v = float(cell)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        pairs.append((ts, v))
    return _sorted_arrays(pairs)


def parse_whitespace_series(
    text: str,
    *,
    date_cols: tuple[int, ...] = (0, 1, 2),
    value_col: int = -1,
    comment: str = "#",
    missing_below: float = -900.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse a whitespace-delimited table (e.g. NOAA's Mauna Loa CO2 trend file).

    ``date_cols`` are the year/month/day column indices, ``value_col`` the value
    column. Comment lines and rows with a sentinel missing value (``<=
    missing_below``, NOAA uses ``-999.99``) are skipped.
    """
    pairs: list[tuple[float, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(comment):
            continue
        parts = line.split()
        try:
            y, m, d = (int(parts[date_cols[0]]), int(parts[date_cols[1]]), int(parts[date_cols[2]]))
            v = float(parts[value_col])
        except (IndexError, ValueError):
            continue
        if not np.isfinite(v) or v <= missing_below:
            continue
        try:
            ts = datetime(y, m, d, tzinfo=UTC).timestamp()
        except ValueError:
            continue
        pairs.append((ts, v))
    return _sorted_arrays(pairs)


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #


def _window_dated(
    series: np.ndarray,
    times: np.ndarray,
    domain: str,
    url: str,
    n: int,
    length: int,
    rng: np.random.Generator,
) -> list[DatedMotif]:
    """Serve ``n`` random ``length``-windows, z-scored and stamped with end time."""
    if series.size < length:
        raise ValueError(f"feed {url!r} has {series.size} points, need >= {length}")
    out: list[DatedMotif] = []
    for _ in range(n):
        start = int(rng.integers(0, series.size - length + 1))
        end = start + length - 1
        out.append((_finalize(series[start : start + length]), domain, float(times[end])))
    return out


@dataclass
class DatedJsonFeed(DatedLiveSource):
    """A real :class:`feeds.DatedLiveSource` over a daily JSON endpoint.

    Fetches once (via the injected ``fetch``), parses one timestamp path and one
    value path into a time-sorted series, caches it in memory, and serves random
    ``length``-windows — z-scored with :func:`domains._finalize` and stamped with
    each window's last-point availability time. Set ``dict_keyed=True`` for
    date-keyed-object payloads (then ``timestamp_field`` names the dict and
    ``value_field`` is the sub-path per value). ``filter_field`` / ``filter_value``
    keep only matching records.
    """

    url: str
    domain: str = "json"  # type: ignore[assignment]
    timestamp_field: str = ""
    value_field: str = ""
    dict_keyed: bool = False
    filter_field: str = ""
    filter_value: Any = None
    fetch: Fetcher = field(default=urllib_fetch)
    _series: np.ndarray | None = field(default=None, init=False, repr=False)
    _times: np.ndarray | None = field(default=None, init=False, repr=False)

    def _load(self) -> tuple[np.ndarray, np.ndarray]:
        if self._series is None or self._times is None:
            text = self.fetch(self.url)
            if self.dict_keyed:
                vals, times = parse_json_dict_series(text, self.timestamp_field, self.value_field)
            else:
                vals, times = parse_json_series(
                    text,
                    self.timestamp_field,
                    self.value_field,
                    filter_field=self.filter_field,
                    filter_value=self.filter_value,
                )
            if vals.size == 0:
                raise ValueError(f"feed {self.url!r} parsed to an empty series")
            self._series, self._times = vals, times
        return self._series, self._times

    def pull_dated(self, n: int, length: int, rng: np.random.Generator) -> list[DatedMotif]:
        series, times = self._load()
        return _window_dated(series, times, self.domain, self.url, n, length, rng)


@dataclass
class DatedTextFeed(DatedLiveSource):
    """A :class:`feeds.DatedLiveSource` over a whitespace-delimited daily table."""

    url: str
    domain: str = "text"  # type: ignore[assignment]
    date_cols: tuple[int, ...] = (0, 1, 2)
    value_col: int = -1
    comment: str = "#"
    fetch: Fetcher = field(default=urllib_fetch)
    _series: np.ndarray | None = field(default=None, init=False, repr=False)
    _times: np.ndarray | None = field(default=None, init=False, repr=False)

    def _load(self) -> tuple[np.ndarray, np.ndarray]:
        if self._series is None or self._times is None:
            vals, times = parse_whitespace_series(
                self.fetch(self.url),
                date_cols=self.date_cols,
                value_col=self.value_col,
                comment=self.comment,
            )
            if vals.size == 0:
                raise ValueError(f"feed {self.url!r} parsed to an empty series")
            self._series, self._times = vals, times
        return self._series, self._times

    def pull_dated(self, n: int, length: int, rng: np.random.Generator) -> list[DatedMotif]:
        series, times = self._load()
        return _window_dated(series, times, self.domain, self.url, n, length, rng)


# --------------------------------------------------------------------------- #
# Curated catalogue of daily-updated public sources
# --------------------------------------------------------------------------- #

# Panel instance lists (carried over from horizon-forge's sources.yaml panels).
_WIKI_ARTICLES = (
    "Cleveland_Clinic", "Bitcoin", "Quantum_computing", "Tariff", "Eurovision_Song_Contest",
    "GPT-4", "Climate_change", "Artificial_intelligence", "Electric_vehicle", "NATO",
    "Federal_Reserve", "SpaceX", "Apple_Inc.", "Tesla", "NVIDIA", "Recession", "Inflation",
    "World_Cup", "Super_Bowl", "Olympics", "Earthquake", "Hurricane", "Wildfire", "Volcano",
    "Stock_market", "Renewable_energy", "Nuclear_power", "Cybersecurity", "Cryptocurrency",
    "Blockchain",
)
_WIKI_HEALTH = (
    "Influenza", "Cancer", "Cleveland_Clinic", "Tylenol", "Ozempic",
    "Respiratory_syncytial_virus", "Measles", "Long_COVID",
)
_NPM_PACKAGES = (
    "react", "lodash", "axios", "vue", "tslib", "yargs", "@aws-sdk/client-s3", "rxjs", "chalk",
    "dotenv", "typescript", "webpack", "eslint", "prettier", "jest", "react-dom", "next",
    "tailwindcss", "zod", "@types/node", "vite", "express", "@babel/core", "esbuild", "rollup",
    "postcss", "sass", "socket.io", "graphql", "bun",
)
_PYPI_PACKAGES = (
    "numpy", "pandas", "requests", "torch", "pyarrow", "httpx", "ruff", "pydantic",
    "huggingface-hub", "matplotlib", "scikit-learn", "tensorflow", "transformers", "scipy",
    "fastapi", "flask", "django", "sqlalchemy", "jupyter", "notebook", "polars", "duckdb",
    "lxml", "beautifulsoup4", "aiohttp", "rich", "click", "typer", "pytest", "black",
)
_CRATES = (
    "serde", "tokio", "clap", "anyhow", "regex", "rand", "reqwest", "itertools", "rayon", "log",
    "serde_json", "thiserror", "futures", "hyper", "axum", "sqlx", "chrono", "uuid", "tonic",
    "bytes", "tracing", "lazy_static", "dotenv", "env_logger", "parking_lot",
)


@dataclass(frozen=True)
class DailyFeedSpec:
    """One curated daily source: enough to construct a dated adapter.

    ``kind`` selects the parser/adapter: ``"json"`` (list-of-records),
    ``"json_dict"`` (date-keyed object), ``"csv"``, or ``"text_ws"`` (whitespace
    table). ``panel`` (with a ``{KEY}`` placeholder in ``url``) fans the spec out
    into one feed per instance, all sharing ``domain``.
    """

    domain: str
    url: str
    kind: str = "json"
    # json / json_dict:
    timestamp_field: str = ""
    value_field: str = ""
    filter_field: str = ""
    filter_value: Any = None
    # csv:
    value_column: int | str = -1
    date_column: int | str = 0
    has_header: bool = True
    # text_ws:
    date_cols: tuple[int, ...] = (0, 1, 2)
    value_col: int = -1
    comment: str = "#"
    # panel fan-out:
    panel_key: str = ""
    panel: tuple[str, ...] = ()
    min_points: int = 0
    note: str = ""


DAILY_REGISTRY: dict[str, DailyFeedSpec] = {
    # --- list-of-records JSON, single series ------------------------------- #
    "defi_tvl": DailyFeedSpec(
        domain="defi_tvl",
        url="https://api.llama.fi/v2/historicalChainTvl/Ethereum",
        timestamp_field="[].date",
        value_field="[].tvl",
        min_points=1500,
        note="Daily Ethereum total value locked (USD), full history. DefiLlama.",
    ),
    "climate_open_meteo": DailyFeedSpec(
        domain="climate_open_meteo",
        url=(
            "https://climate-api.open-meteo.com/v1/climate?latitude=40.71&longitude=-74.0"
            "&start_date=2000-01-01&end_date=2025-12-31&models=EC_Earth3P_HR"
            "&daily=temperature_2m_mean"
        ),
        timestamp_field="daily.time",
        value_field="daily.temperature_2m_mean",
        min_points=2000,
        note="Daily mean 2m temperature (CMIP6 EC-Earth3P-HR), NYC. Open-Meteo.",
    ),
    "river_streamflow": DailyFeedSpec(
        domain="river_streamflow",
        url=(
            "https://waterservices.usgs.gov/nwis/dv/?format=json&sites=01646500"
            "&parameterCd=00060&statCd=00003&startDT=2010-01-01&endDT=2025-12-31"
        ),
        timestamp_field="value.timeSeries[0].values[0].value[].dateTime",
        value_field="value.timeSeries[0].values[0].value[].value",
        min_points=3000,
        note="Daily mean streamflow (cfs), Potomac River. USGS NWIS.",
    ),
    "snow_water_equiv": DailyFeedSpec(
        domain="snow_water_equiv",
        url=(
            "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data?stationTriplets=908:WA:SNTL"
            "&elements=WTEQ&duration=DAILY&beginDate=2010-10-01&endDate=2025-06-01"
        ),
        timestamp_field="[].data[].values[].date",
        value_field="[].data[].values[].value",
        min_points=1500,
        note="Daily snow water equivalent (in), WA SNOTEL station. USDA NRCS.",
    ),
    "epidemic_cases": DailyFeedSpec(
        domain="epidemic_cases",
        url="https://api.corona-zahlen.org/germany/history/cases/1200",
        timestamp_field="data[].date",
        value_field="data[].cases",
        min_points=600,
        note="Daily reported COVID-19 cases, Germany. RKI / corona-zahlen.org.",
    ),
    "hospitalization_rate": DailyFeedSpec(
        domain="hospitalization_rate",
        url="https://api.corona-zahlen.org/germany/history/hospitalization/1200",
        timestamp_field="data[].date",
        value_field="data[].incidence7Days",
        min_points=600,
        note="Daily 7-day hospitalization incidence, Germany. RKI / corona-zahlen.org.",
    ),
    "npm_total_downloads": DailyFeedSpec(
        domain="npm_total_downloads",
        url="https://api.npmjs.org/downloads/range/2024-06-01:2025-12-01/npm",
        timestamp_field="downloads[].day",
        value_field="downloads[].downloads",
        min_points=400,
        note="Daily total npm package downloads (registry-wide). npm registry.",
    ),
    # --- date-keyed object JSON -------------------------------------------- #
    "fx_usd_eur": DailyFeedSpec(
        domain="fx_usd_eur",
        url="https://api.frankfurter.dev/v1/2015-01-01..2025-12-31?from=USD&to=EUR",
        kind="json_dict",
        timestamp_field="rates",
        value_field="EUR",
        min_points=2000,
        note="Daily USD->EUR reference rate (ECB, weekdays). Frankfurter.",
    ),
    # --- CSV --------------------------------------------------------------- #
    "treasury_10y_yield": DailyFeedSpec(
        domain="treasury_10y_yield",
        url=(
            "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
            "daily-treasury-rates.csv/2024/all?type=daily_treasury_yield_curve"
            "&field_tdr_date_value=2024"
        ),
        kind="csv",
        value_column="10 Yr",
        date_column="Date",
        min_points=200,
        note="Daily 10-year Treasury par yield (weekdays, 2024). US Treasury.",
    ),
    # --- whitespace text --------------------------------------------------- #
    "co2_mauna_loa": DailyFeedSpec(
        domain="co2_mauna_loa",
        url="https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_trend_gl.txt",
        kind="text_ws",
        date_cols=(0, 1, 2),
        value_col=4,
        min_points=300,
        note="Daily global atmospheric CO2 trend (ppm). NOAA GML.",
    ),
    # --- panel JSON (one series per instance) ------------------------------ #
    "wiki_pageviews": DailyFeedSpec(
        domain="wiki_pageviews",
        url=(
            "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/"
            "all-access/all-agents/{ARTICLE}/daily/2020010100/2025123100"
        ),
        timestamp_field="items[].timestamp",
        value_field="items[].views",
        panel_key="ARTICLE",
        panel=_WIKI_ARTICLES,
        min_points=1500,
        note="Daily English Wikipedia pageviews per article (panel of 30). Wikimedia.",
    ),
    "wiki_health_pageviews": DailyFeedSpec(
        domain="wiki_health_pageviews",
        url=(
            "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/"
            "all-access/all-agents/{ARTICLE}/daily/2020010100/2025123100"
        ),
        timestamp_field="items[].timestamp",
        value_field="items[].views",
        panel_key="ARTICLE",
        panel=_WIKI_HEALTH,
        min_points=1500,
        note="Daily Wikipedia pageviews for medical articles (panel of 8). Wikimedia.",
    ),
    "npm_downloads": DailyFeedSpec(
        domain="npm_downloads",
        url="https://api.npmjs.org/downloads/range/2024-06-01:2025-12-01/{PACKAGE}",
        timestamp_field="downloads[].day",
        value_field="downloads[].downloads",
        panel_key="PACKAGE",
        panel=_NPM_PACKAGES,
        min_points=400,
        note="Daily npm downloads per package (panel of 30). npm registry.",
    ),
    "pypi_downloads": DailyFeedSpec(
        domain="pypi_downloads",
        url="https://pypistats.org/api/packages/{PACKAGE}/overall?mirrors=true",
        timestamp_field="data[].date",
        value_field="data[].downloads",
        filter_field="data[].category",
        filter_value="with_mirrors",
        panel_key="PACKAGE",
        panel=_PYPI_PACKAGES,
        min_points=120,
        note="Daily PyPI downloads per package (panel of 30, ~180d). pypistats.org.",
    ),
    "crates_downloads": DailyFeedSpec(
        domain="crates_downloads",
        url="https://crates.io/api/v1/crates/{CRATE}/downloads",
        timestamp_field="meta.extra_downloads[].date",
        value_field="meta.extra_downloads[].downloads",
        panel_key="CRATE",
        panel=_CRATES,
        min_points=80,
        note="Daily crates.io downloads per crate (panel of 25, ~90d). crates.io.",
    ),
}


# Daily P1D sources from horizon-forge that are deliberately NOT ported, because
# they are not univariate numeric time series (forcing them into a forecasting
# benchmark would be meaningless).
EXCLUDED: dict[str, str] = {
    "itunes_top_podcasts": "a daily list of podcast ids (a ranking snapshot), not a numeric series",
    "wikimedia_top_articles_daily": (
        "a one-day cross-section of the top articles, not a series; wiki_pageviews covers "
        "Wikipedia pageviews as a proper per-article series instead"
    ),
    "neso_stor_notifications": "irregular STOR event rows with a non-numeric value column",
}


def _expand_url(url: str, key: str, value: str) -> str:
    return url.replace("{" + key + "}", value) if key else url


def _build_one(spec: DailyFeedSpec, url: str, fetch: Fetcher) -> DatedLiveSource:
    """Construct the dated adapter for ``spec`` at a (possibly templated) ``url``."""
    if spec.kind == "json":
        return DatedJsonFeed(
            url=url,
            domain=spec.domain,
            timestamp_field=spec.timestamp_field,
            value_field=spec.value_field,
            filter_field=spec.filter_field,
            filter_value=spec.filter_value,
            fetch=fetch,
        )
    if spec.kind == "json_dict":
        return DatedJsonFeed(
            url=url,
            domain=spec.domain,
            timestamp_field=spec.timestamp_field,
            value_field=spec.value_field,
            dict_keyed=True,
            fetch=fetch,
        )
    if spec.kind == "csv":
        return DatedCsvFeed(
            url=url,
            domain=spec.domain,
            value_column=spec.value_column,
            date_column=spec.date_column,
            has_header=spec.has_header,
            fetch=fetch,
        )
    if spec.kind == "text_ws":
        return DatedTextFeed(
            url=url,
            domain=spec.domain,
            date_cols=spec.date_cols,
            value_col=spec.value_col,
            comment=spec.comment,
            fetch=fetch,
        )
    raise ValueError(f"unknown feed kind {spec.kind!r} for {spec.domain!r}")


def expand_daily_feeds(
    name: str, *, fetch: Fetcher | None = None
) -> list[tuple[DatedLiveSource, float]]:
    """All ``(feed, weight)`` components for ``name`` — one per panel instance.

    Non-panel specs yield a single weight-1 feed; a panel spec yields one feed per
    instance, each weighted ``1/len(panel)`` so the *source* (not each instance)
    carries unit weight in a mixture.
    """
    if name not in DAILY_REGISTRY:
        raise KeyError(f"unknown daily feed {name!r}; known: {sorted(DAILY_REGISTRY)}")
    spec = DAILY_REGISTRY[name]
    f = fetch or cached_fetch()
    if not spec.panel:
        return [(_build_one(spec, spec.url, f), 1.0)]
    w = 1.0 / len(spec.panel)
    return [
        (_build_one(spec, _expand_url(spec.url, spec.panel_key, inst), f), w)
        for inst in spec.panel
    ]


def make_daily_feed(
    name: str, *, instance: int = 0, fetch: Fetcher | None = None
) -> DatedLiveSource:
    """Construct a single dated feed from the catalogue.

    For a panel spec, ``instance`` selects which panel member (default the first).
    Use :func:`expand_daily_feeds` to get every instance, or
    :func:`build_daily_live_source` for the whole multi-domain mixture.
    """
    return expand_daily_feeds(name, fetch=fetch)[instance][0]


def build_daily_live_source(
    names: list[str] | None = None,
    *,
    fetch: Fetcher | None = None,
    weights: dict[str, float] | None = None,
):
    """A multi-domain feed over real daily data — the analogue of
    :func:`live_feeds.build_real_live_source`.

    Every selected spec is fully expanded (panels included) into a
    ``MixtureLiveSource``, each spec contributing equal total weight (overridable
    via ``weights``) regardless of how many panel instances it has. The shared
    ``fetch`` (a cache by default) means each underlying URL is pulled at most
    once. For vintage gating, prefer a single :func:`make_daily_feed` wrapped in
    :class:`feeds.AsOfLiveSource` — the mixture composes feeds as plain (undated)
    sources, like the CSV analogue.

    Note: the short-history feeds (``pypi_downloads``, ``crates_downloads``,
    ``treasury_10y_yield``) cannot serve windows longer than their few-hundred
    points; subset ``names`` (or raise the pool/motif length accordingly) when
    mixing them with the long climate/finance feeds.
    """
    from domains import MixtureLiveSource

    chosen = names or list(DAILY_REGISTRY)
    f = fetch or cached_fetch()
    components: list[tuple[LiveSource, float]] = []
    for name in chosen:
        scale = (weights or {}).get(name, 1.0)
        for feed, w in expand_daily_feeds(name, fetch=f):
            components.append((feed, w * scale))
    return MixtureLiveSource(components)
