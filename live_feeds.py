"""Real public time-series feeds — the live layer wired to actual data.

``ingest.py`` defines the ``LiveSource`` contract with a synthetic stand-in, and
``feeds.py`` adds the *discipline* a real deployment needs (as-of gating, cross-
epoch dedup, an HTTP/CSV skeleton). This module supplies the missing piece: a set
of **concrete adapters pointed at real, publicly hosted series**, plus a curated
registry that composes them into a genuinely multi-domain *real* feed — the
real-data analogue of ``domains.default_live_source``.

Why this matters
----------------
The benchmark's anti-memorisation guarantee only bites if the "live" motifs are
real data a miner could not reproduce by fitting the synthetic generator. Until
real series actually flow through the pipeline, that guarantee is scaffolding.
This module makes it concrete: ``build_real_live_source()`` returns a
``MixtureLiveSource`` spanning climate, solar activity, atmospheric chemistry,
finance, and weather — five distinct real-world data-generating processes.

Design choices that keep it consensus-safe and testable
-------------------------------------------------------
* **Injected fetcher.** Every adapter takes a ``fetch`` callable (``url -> text``)
  so the network policy lives with the caller and tests run fully offline. The
  default is a small on-disk cache over ``urllib`` (:func:`cached_fetch`), so a
  notebook re-runs without re-hitting the network and validators that share a
  cache snapshot stay byte-identical.
* **Per-motif z-scoring.** Each served window is normalised with the same
  :func:`domains._finalize` the synthetic zoo uses, so real and synthetic motifs
  land on a common scale and the per-challenge error normalisation in ``score``
  stays well-conditioned regardless of a feed's native units.
* **As-of ready.** :class:`DatedCsvFeed` parses the date column and stamps every
  window with the availability time of its *last* point, so it drops straight into
  :class:`feeds.AsOfLiveSource` for vintage gating against the commit beacon.

Production note
---------------
These public endpoints are *static history* — fine for proving the wiring and for
research, but a real deployment must point :class:`DatedCsvFeed` at an as-of /
vintage vendor endpoint whose new rows are timestamped after the commit beacon,
and wrap it in ``feeds.AsOfLiveSource`` + ``feeds.DedupFreshBuffer``. The registry
URLs here are pinned to specific hosts; treat them as examples, not guarantees.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from domains import _finalize
from feeds import DatedLiveSource, DatedMotif
from ingest import LiveSource

# A fetcher maps a URL to its raw text body. Injected everywhere so the network
# policy and offline testing stay external to the adapters.
Fetcher = Callable[[str], str]


# --------------------------------------------------------------------------- #
# Fetching: a plain urllib fetcher and a cached wrapper around it
# --------------------------------------------------------------------------- #


def urllib_fetch(url: str, timeout: float = 30.0) -> str:  # pragma: no cover - network
    """Fetch ``url`` as UTF-8 text via the standard library (no extra deps)."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "tsbench-forge/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def default_cache_dir() -> str:
    """Where cached feed bodies live; override with ``TSBENCH_FEED_CACHE``."""
    return os.environ.get(
        "TSBENCH_FEED_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "tsbench-forge", "feeds"),
    )


def cached_fetch(
    cache_dir: str | None = None,
    inner: Fetcher = urllib_fetch,
) -> Fetcher:
    """Return a :data:`Fetcher` that memoises bodies on disk under ``cache_dir``.

    The first pull of a URL hits ``inner`` (the network); subsequent pulls — in
    the same process, a later notebook run, or another validator sharing the
    cache directory — read the file. Keying is ``sha256(url)`` so the cache is
    flat and filesystem-safe. This is what lets a feed be a *fixed snapshot*: all
    validators that share the snapshot reconstruct identical pools.
    """
    root = cache_dir or default_cache_dir()
    os.makedirs(root, exist_ok=True)

    def fetch(url: str) -> str:
        path = os.path.join(root, hashlib.sha256(url.encode("utf-8")).hexdigest() + ".txt")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return fh.read()
        body = inner(url)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.replace(tmp, path)  # atomic: a crashed write never leaves a half cache
        return body

    return fetch


# --------------------------------------------------------------------------- #
# CSV parsing helpers
# --------------------------------------------------------------------------- #


def _split_csv_line(line: str) -> list[str]:
    """Split one CSV line, tolerating simple double-quoted fields (no embedded commas)."""
    return [cell.strip().strip('"').strip("'") for cell in line.split(",")]


def _resolve_column(header: list[str] | None, column: int | str) -> int:
    """Resolve a column given by name or index to an integer index."""
    if isinstance(column, int):
        return column
    if header is None:
        raise ValueError(f"column {column!r} given by name but the CSV has no header")
    try:
        return header.index(column)
    except ValueError as exc:
        raise ValueError(f"column {column!r} not in header {header}") from exc


def parse_csv_column(
    text: str,
    value_column: int | str,
    *,
    has_header: bool = True,
    date_column: int | str | None = None,
    date_formats: tuple[str, ...] = ("%Y-%m-%d", "%Y-%m", "%Y/%m/%d", "%m/%d/%Y"),
) -> tuple[np.ndarray, np.ndarray | None]:
    """Parse one numeric column (and optionally a date column) out of CSV text.

    Returns ``(values, times)`` where ``times`` is ``None`` when no ``date_column``
    is given, else an array of POSIX seconds aligned with ``values``. Rows whose
    value cell is non-numeric are skipped (so sparse / annotated feeds still
    parse); a row is dropped entirely if its date cannot be parsed when dates are
    requested, keeping ``values`` and ``times`` aligned.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return np.empty(0, dtype=float), (None if date_column is None else np.empty(0))
    header = _split_csv_line(lines[0]) if has_header else None
    rows = lines[1:] if has_header else lines
    vcol = _resolve_column(header, value_column)
    dcol = None if date_column is None else _resolve_column(header, date_column)

    values: list[float] = []
    times: list[float] = []
    for raw in rows:
        cells = _split_csv_line(raw)
        if vcol >= len(cells):
            continue
        try:
            v = float(cells[vcol])
        except ValueError:
            continue
        if not np.isfinite(v):
            continue
        if dcol is not None:
            if dcol >= len(cells):
                continue
            ts = _parse_date(cells[dcol], date_formats)
            if ts is None:
                continue
            times.append(ts)
        values.append(v)

    vals = np.asarray(values, dtype=float)
    tarr = np.asarray(times, dtype=float) if dcol is not None else None
    return vals, tarr


def _parse_date(cell: str, formats: tuple[str, ...]) -> float | None:
    for fmt in formats:
        try:
            dt = datetime.strptime(cell, fmt).replace(tzinfo=UTC)
            return dt.timestamp()
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------- #
# The adapters
# --------------------------------------------------------------------------- #


@dataclass
class CsvFeed(LiveSource):
    """A real :class:`LiveSource` over one numeric column of a CSV endpoint.

    Fetches once (via the injected ``fetch``), caches the parsed series in memory,
    and serves random ``length``-windows, each z-scored with :func:`_finalize` so
    it matches the synthetic zoo's conditioning. Raises if the series is shorter
    than a requested window — curate long-enough feeds, or shrink ``motif_len``.
    """

    url: str
    domain: str = "csv"  # type: ignore[assignment]
    value_column: int | str = -1
    has_header: bool = True
    fetch: Fetcher = field(default=urllib_fetch)
    _series: np.ndarray | None = field(default=None, init=False, repr=False)

    def series(self) -> np.ndarray:
        """The full parsed series (fetched and cached on first access)."""
        if self._series is None:
            text = self.fetch(self.url)
            vals, _ = parse_csv_column(
                text, self.value_column, has_header=self.has_header
            )
            if vals.size == 0:
                raise ValueError(f"feed {self.url!r} parsed to an empty series")
            self._series = vals
        return self._series

    def _windows(
        self, series: np.ndarray, n: int, length: int, rng: np.random.Generator
    ) -> list[tuple[int, np.ndarray]]:
        if series.size < length:
            raise ValueError(
                f"feed {self.url!r} has {series.size} points, need >= {length}"
            )
        out: list[tuple[int, np.ndarray]] = []
        for _ in range(n):
            start = int(rng.integers(0, series.size - length + 1))
            window = series[start : start + length]
            out.append((start + length - 1, _finalize(window)))
        return out

    def pull(self, n: int, length: int, rng: np.random.Generator) -> list[np.ndarray]:
        return [w for _, w in self._windows(self.series(), n, length, rng)]


@dataclass
class DatedCsvFeed(DatedLiveSource):
    """A :class:`CsvFeed` that also reads a date column, so it can be as-of gated.

    Each window is stamped with the availability time of its *last* point (POSIX
    seconds) — a realistic "this window became knowable at its end" clock. Wrap in
    :class:`feeds.AsOfLiveSource` to admit only windows ending after the commit
    beacon, the same vintage discipline a real vendor feed needs.
    """

    url: str
    domain: str = "csv"  # type: ignore[assignment]
    value_column: int | str = -1
    date_column: int | str = 0
    has_header: bool = True
    fetch: Fetcher = field(default=urllib_fetch)
    _series: np.ndarray | None = field(default=None, init=False, repr=False)
    _times: np.ndarray | None = field(default=None, init=False, repr=False)

    def _load(self) -> tuple[np.ndarray, np.ndarray]:
        if self._series is None or self._times is None:
            text = self.fetch(self.url)
            vals, times = parse_csv_column(
                text,
                self.value_column,
                has_header=self.has_header,
                date_column=self.date_column,
            )
            if vals.size == 0:
                raise ValueError(f"feed {self.url!r} parsed to an empty series")
            assert times is not None
            self._series, self._times = vals, times
        return self._series, self._times

    def pull_dated(self, n: int, length: int, rng: np.random.Generator) -> list[DatedMotif]:
        series, times = self._load()
        if series.size < length:
            raise ValueError(
                f"feed {self.url!r} has {series.size} points, need >= {length}"
            )
        out: list[DatedMotif] = []
        for _ in range(n):
            start = int(rng.integers(0, series.size - length + 1))
            end = start + length - 1
            out.append((_finalize(series[start : start + length]), self.domain, float(times[end])))
        return out


# --------------------------------------------------------------------------- #
# Curated registry of real, publicly hosted series
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeedSpec:
    """One curated public feed: enough to construct a :class:`CsvFeed`/`DatedCsvFeed`."""

    domain: str
    url: str
    value_column: int | str
    date_column: int | str
    has_header: bool = True
    min_points: int = 0
    note: str = ""


# Five genuinely distinct real-world data-generating processes. Each is long
# enough for the default working window and spans a different domain, so a high
# score means "good across real worlds," not "good at one series." URLs are
# pinned to widely-mirrored hosts; swap in your own vendor feeds for production.
REGISTRY: dict[str, FeedSpec] = {
    "climate_temp": FeedSpec(
        domain="climate_temp",
        url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        value_column="Temp",
        date_column="Date",
        min_points=3650,
        note="Daily minimum temperatures, Melbourne 1981-1990.",
    ),
    "solar_sunspots": FeedSpec(
        domain="solar_sunspots",
        url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv",
        value_column="Sunspots",
        date_column="Month",
        min_points=2800,
        note="Monthly mean sunspot counts, 1749-1983.",
    ),
    "atmospheric_co2": FeedSpec(
        domain="atmospheric_co2",
        url="https://raw.githubusercontent.com/datasets/co2-ppm-daily/main/data/co2-ppm-daily.csv",
        value_column="value",
        date_column="date",
        min_points=10000,
        note="Daily atmospheric CO2 concentration (ppm), Mauna Loa.",
    ),
    "finance_equity": FeedSpec(
        domain="finance_equity",
        url="https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv",
        value_column="AAPL.Close",
        date_column="Date",
        min_points=500,
        note="Daily AAPL closing price, 2015-2017.",
    ),
    "weather_seattle": FeedSpec(
        domain="weather_seattle",
        url="https://raw.githubusercontent.com/vega/vega-datasets/main/data/seattle-weather.csv",
        value_column="temp_max",
        date_column="date",
        min_points=1400,
        note="Daily max temperature, Seattle 2012-2015.",
    ),
}


def make_feed(name: str, *, fetch: Fetcher | None = None, dated: bool = False):
    """Construct a :class:`CsvFeed` (or :class:`DatedCsvFeed`) from the registry."""
    if name not in REGISTRY:
        raise KeyError(f"unknown feed {name!r}; known: {sorted(REGISTRY)}")
    spec = REGISTRY[name]
    f = fetch or cached_fetch()
    if dated:
        return DatedCsvFeed(
            url=spec.url,
            domain=spec.domain,
            value_column=spec.value_column,
            date_column=spec.date_column,
            has_header=spec.has_header,
            fetch=f,
        )
    return CsvFeed(
        url=spec.url,
        domain=spec.domain,
        value_column=spec.value_column,
        has_header=spec.has_header,
        fetch=f,
    )


def build_real_live_source(
    names: list[str] | None = None,
    *,
    fetch: Fetcher | None = None,
    weights: dict[str, float] | None = None,
):
    """A multi-domain feed over *real* data — the live analogue of the zoo.

    Mirrors :func:`domains.default_live_source` but every component is a real
    public series. Pass ``names`` to subset the registry and ``weights`` to skew
    the mix (defaults to an equal blend). The shared ``fetch`` (a cache by
    default) means each underlying URL is pulled at most once.
    """
    from domains import MixtureLiveSource

    chosen = names or list(REGISTRY)
    f = fetch or cached_fetch()
    components: list[tuple[LiveSource, float]] = []
    for name in chosen:
        w = (weights or {}).get(name, 1.0)
        components.append((make_feed(name, fetch=f), w))
    return MixtureLiveSource(components)
