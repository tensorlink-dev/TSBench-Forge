"""Daily-updated public JSON feeds — the live layer fed by real, refreshing APIs.

``live_feeds.py`` wires real series from *CSV* endpoints; this module covers the
other half of the public-data world: **JSON APIs that publish a fresh daily
point**. It is a self-contained port of the data-ingestion ideas in the sibling
``horizon-forge`` project — specifically its declarative JSON *path engine* and a
curated catalogue of daily endpoints — re-expressed against this repo's own
:class:`~feeds.DatedLiveSource` contract. **There is no dependency on
horizon-forge**: only the stdlib + numpy, with the network injected via a
``fetch`` callable exactly like the rest of the live layer.

What was ported, and what was deliberately left behind
------------------------------------------------------
* **Ported:** the generic JSON path engine (:func:`_walk` / :func:`_tokenize` /
  :func:`_apply`) that pulls a timestamp array and a value array out of an
  arbitrarily-nested payload using a dotted/bracketed path spec (e.g.
  ``data[].date`` or ``value.timeSeries[0].values[0].value[].value``), plus the
  curated daily endpoints worth keeping.
* **Left behind:** httpx, pandas, pyarrow, parquet storage, and the cron layer.
  horizon-forge *accumulates* history into per-day parquet via an external
  scheduler; here we lean on the fact that these endpoints already return a
  trailing window of history, fetch it on demand, and let
  :func:`live_feeds.cached_fetch` snapshot it. That keeps the module within this
  repo's "stdlib + numpy, injected fetch" discipline and needs no standing
  infrastructure.

How it plugs in
---------------
:class:`DatedJsonFeed` is a drop-in :class:`feeds.DatedLiveSource`: it stamps each
window with the availability time of its last point, so it composes directly with
:class:`feeds.AsOfLiveSource` (vintage gating) and :class:`feeds.DedupFreshBuffer`
(cross-epoch dedup). :func:`build_daily_live_source` mirrors
:func:`live_feeds.build_real_live_source`, returning a multi-domain
``MixtureLiveSource`` over the catalogue.

Production note
---------------
Like ``live_feeds``, the bundled URLs are pinned snapshots (some carry fixed date
ranges) — fine for proving the wiring and for research. A real deployment points
:class:`DatedJsonFeed` at an as-of endpoint whose new rows are timestamped after
the commit beacon and wraps it in ``feeds.AsOfLiveSource`` + ``DedupFreshBuffer``.
Several catalogue endpoints need an identifying ``User-Agent`` or token (noted per
entry); supply that through the injected ``fetch`` rather than here.
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
from live_feeds import Fetcher, cached_fetch, urllib_fetch

# --------------------------------------------------------------------------- #
# JSON path engine (ported from horizon-forge's sources/scraper.py)
# --------------------------------------------------------------------------- #
#
# A path is a dotted/bracketed spec naming where a timestamp or value sequence
# lives inside a JSON payload. ``[]`` lifts the rest of the path over every
# element of a list (returning an aligned list); ``[N]`` indexes a single one.
#
#     'a.b'        -> obj['a']['b']
#     'a[].b'      -> [item['b'] for item in obj['a']]
#     'a[0].b'     -> obj['a'][0]['b']


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

    Handles the shapes the daily catalogue actually emits: epoch seconds or
    milliseconds (DefiLlama), plain ``YYYY-MM-DD`` (RKI, Open-Meteo), and full
    ISO-8601 timestamps with an offset (USGS). Returns ``None`` for anything
    unparseable so the caller can drop the row and keep arrays aligned.
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
        try:  # a bare numeric string is an epoch
            n = float(s)
            return n / 1000.0 if n > 1e12 else n
        except ValueError:
            pass
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


def parse_json_series(
    payload: str | bytes | Any,
    timestamp_field: str,
    value_field: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract aligned ``(values, times)`` arrays from a JSON payload.

    ``timestamp_field``/``value_field`` are path specs (see :func:`_walk`) that
    must each resolve to a list; the two lists are zipped, rows with a
    non-numeric value or unparseable timestamp are dropped (keeping the arrays
    aligned), and the result is sorted ascending by time so the *last* point of
    any window is its availability time. ``times`` is POSIX seconds.
    """
    data = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
    ts_seq = _walk(data, timestamp_field)
    val_seq = _walk(data, value_field)
    if not isinstance(ts_seq, list) or not isinstance(val_seq, list):
        raise ValueError(
            f"paths did not resolve to aligned lists: "
            f"timestamp_field={timestamp_field!r} value_field={value_field!r}"
        )

    pairs: list[tuple[float, float]] = []
    for raw_ts, raw_val in zip(ts_seq, val_seq, strict=False):  # truncate to the shorter
        ts = _to_posix(raw_ts)
        if ts is None:
            continue
        try:
            v = float(raw_val)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        pairs.append((ts, v))

    pairs.sort(key=lambda p: p[0])
    times = np.asarray([p[0] for p in pairs], dtype=float)
    values = np.asarray([p[1] for p in pairs], dtype=float)
    return values, times


# --------------------------------------------------------------------------- #
# The adapter
# --------------------------------------------------------------------------- #


@dataclass
class DatedJsonFeed(DatedLiveSource):
    """A real :class:`feeds.DatedLiveSource` over a daily JSON endpoint.

    Fetches once (via the injected ``fetch``), parses one timestamp path and one
    value path into a time-sorted series, caches it in memory, and serves random
    ``length``-windows — each z-scored with :func:`domains._finalize` so it lands
    on the synthetic zoo's scale, and stamped with the availability time of its
    last point so it drops straight into :class:`feeds.AsOfLiveSource`. Raises if
    the series is shorter than a requested window.
    """

    url: str
    domain: str = "json"  # type: ignore[assignment]
    timestamp_field: str = ""
    value_field: str = ""
    fetch: Fetcher = field(default=urllib_fetch)
    _series: np.ndarray | None = field(default=None, init=False, repr=False)
    _times: np.ndarray | None = field(default=None, init=False, repr=False)

    def _load(self) -> tuple[np.ndarray, np.ndarray]:
        if self._series is None or self._times is None:
            text = self.fetch(self.url)
            vals, times = parse_json_series(text, self.timestamp_field, self.value_field)
            if vals.size == 0:
                raise ValueError(f"feed {self.url!r} parsed to an empty series")
            self._series, self._times = vals, times
        return self._series, self._times

    def pull_dated(self, n: int, length: int, rng: np.random.Generator) -> list[DatedMotif]:
        series, times = self._load()
        if series.size < length:
            raise ValueError(f"feed {self.url!r} has {series.size} points, need >= {length}")
        out: list[DatedMotif] = []
        for _ in range(n):
            start = int(rng.integers(0, series.size - length + 1))
            end = start + length - 1
            out.append((_finalize(series[start : start + length]), self.domain, float(times[end])))
        return out


# --------------------------------------------------------------------------- #
# Curated catalogue of daily-updated public JSON feeds
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DailyFeedSpec:
    """One curated daily JSON feed: enough to construct a :class:`DatedJsonFeed`."""

    domain: str
    url: str
    timestamp_field: str
    value_field: str
    min_points: int = 0
    note: str = ""


# Daily-refreshing public endpoints whose response already carries a trailing
# window of history (so an on-demand fetch yields a usable series with no
# scraper), are auth-free, and decode to a clean single numeric series. Panel /
# date-templated / snapshot-only / auth-gated daily sources from the catalogue
# (Wikimedia per-article, npm/pypi/crates panels, Frankfurter `latest`, iTunes
# top-100) are intentionally omitted from this first pass — they need URL
# templating or per-instance iteration the on-demand path doesn't yet do.
DAILY_REGISTRY: dict[str, DailyFeedSpec] = {
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
}


def make_daily_feed(name: str, *, fetch: Fetcher | None = None) -> DatedJsonFeed:
    """Construct a :class:`DatedJsonFeed` from the daily catalogue by name."""
    if name not in DAILY_REGISTRY:
        raise KeyError(f"unknown daily feed {name!r}; known: {sorted(DAILY_REGISTRY)}")
    spec = DAILY_REGISTRY[name]
    return DatedJsonFeed(
        url=spec.url,
        domain=spec.domain,
        timestamp_field=spec.timestamp_field,
        value_field=spec.value_field,
        fetch=fetch or cached_fetch(),
    )


def build_daily_live_source(
    names: list[str] | None = None,
    *,
    fetch: Fetcher | None = None,
    weights: dict[str, float] | None = None,
):
    """A multi-domain feed over real daily JSON data — the JSON analogue of
    :func:`live_feeds.build_real_live_source`.

    Pass ``names`` to subset the catalogue and ``weights`` to skew the mix
    (defaults to an equal blend). The shared ``fetch`` (a cache by default) means
    each underlying URL is pulled at most once. For vintage gating, prefer a
    single :func:`make_daily_feed` wrapped in :class:`feeds.AsOfLiveSource` — the
    mixture composes the feeds as plain (undated) sources, like the CSV analogue.
    """
    from domains import MixtureLiveSource

    chosen = names or list(DAILY_REGISTRY)
    f = fetch or cached_fetch()
    components: list[tuple[LiveSource, float]] = []
    for name in chosen:
        w = (weights or {}).get(name, 1.0)
        components.append((make_daily_feed(name, fetch=f), w))
    return MixtureLiveSource(components)
