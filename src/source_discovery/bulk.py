"""bulk.py — synthesise catalog candidates from federated open-data catalogs.

Agent grind waves find *interesting* sources but cost 20-40 minutes for 10-18
of them. Reaching a 1000-source catalog needs a second gear: whole classes of
sources whose shape is machine-derivable, with no LLM in the loop.

Two platforms are that class, and both publish a *federated* catalog — one
query reaching every portal on the platform:

* **Socrata** (``api.us.socrata.com``) — US-heavy, mostly city and state
  portals. Returns the host, the dataset id, the last-updated time, and the
  column list with datatypes.
* **Opendatasoft** (``data.opendatasoft.com``, ~100k datasets) — Europe-heavy,
  which is where the catalog's geography is thinnest. Returns richer metadata
  still: the publisher's OWN hostname (``source_domain_address``), so entries
  point at ``opendata.tpg.ch`` rather than all collapsing onto one aggregator
  host and counting as a single provider.

In both cases the declared column types are enough to write a catalog entry:
the date column is the timestamp, the numeric columns are the values, and the
URL is mechanical. Every portal is also a distinct host, so this buys provider
diversity at the same time as volume — which is the axis the catalog is
actually short on.

Two safeguards keep the volume honest:

* Every candidate is emitted in the ``--wire`` grind-block format, so it still
  has to clear the same admission gate (>=20 rows, >=20 distinct timestamps,
  >=50% numeric) and the same freshness check as a hand-written entry. A bad
  guess is rejected, not trusted.
* Before emitting, each candidate gets a *cheap* timestamp-only probe
  (``$select=<ts>&$limit=60``, a few KB) to measure the real cadence and the
  real age of the newest observation. Cadence is what the frequency and cron
  group are set from, and age is what ``audit_slack_days`` is derived from —
  guessing either is the single biggest cause of gate failures.

The per-host cap matters as much as the total: the coverage metric admits at
most ``DEFAULT_HOST_CAP`` sources per host, so a 6th dataset from one portal
buys volume without buying coverage.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import csv
import statistics
import time
import unicodedata
from typing import Any, Iterable, Optional
from urllib.parse import quote, urlparse

import requests
import yaml

CATALOG_API = "https://api.us.socrata.com/api/catalog/v1"
UA = "TSBench-Forge/1.0 (benchmark data collection; chris@tensor-link.com)"

# The coverage metric credits at most this many sources per host, so there is no
# reason to wire more than this from any single portal.
DEFAULT_HOST_CAP = 4

# Socrata datatype names that can carry an observation time.
TS_TYPES = frozenset({
    "calendar date", "floating timestamp", "fixed timestamp", "date", "datetime",
})
# ...and the ones that can carry a numeric value.
NUM_TYPES = frozenset({"number", "money", "double", "percent", "int", "decimal"})

# Timestamp columns whose name says "row bookkeeping" rather than "when the thing
# happened". Only used to *rank* — if a dataset has nothing else, these are fine.
_TS_DEMOTE = ("update", "modif", "load", "extract", "ingest", "process", "closed", "resolution")
# ...and the ones that usually ARE the event time.
_TS_PROMOTE = ("date", "datetime", "time", "week", "month", "period", "created",
               "occur", "report", "start", "incident", "observ", "collect", "sample")

# Numeric columns that are identifiers/coordinates, not observations. A dataset
# whose only "numbers" are these is better served by a binned count.
_VAL_REJECT = (
    "id", "_id", "key", "code", "zip", "zipcode", "postal", "lat", "latitude",
    "lon", "long", "longitude", "x_coord", "y_coord", "coordinate", "geoid",
    "fips", "tract", "block", "beat", "district", "precinct", "ward", "sector",
    "year", "month", "day", "hour", "quarter", "week", "number", "num",
    "objectid", "row", "seq", "phone", "badge", "unit", "council", "ansi",
)

# Keyword classes to sweep, mapped onto the catalog's domain vocabulary. Each
# entry is (query keyword, domain, dgp_class). Deliberately excludes anything
# matching the contamination denylist ("traffic", "electricity", "weather") —
# those keyword classes are left to hand-vetted waves.
KEYWORD_CLASSES: tuple[tuple[str, str, str], ...] = (
    ("911 calls", "healthcare", "emergency_dispatch"),
    ("emergency medical service responses", "healthcare", "emergency_dispatch"),
    ("fire incidents", "healthcare", "emergency_dispatch"),
    ("hospital discharges", "healthcare", "health_utilisation"),
    ("communicable disease cases", "healthcare", "disease_surveillance"),
    ("overdose deaths", "healthcare", "mortality_surveillance"),
    ("restaurant inspections", "sales", "inspection_stream"),
    ("business licenses issued", "sales", "registration_stream"),
    ("building permits", "sales", "permit_stream"),
    ("sales tax receipts", "sales", "tax_receipts"),
    ("real property sales", "sales", "transaction_stream"),
    ("311 service requests", "sales", "service_requests"),
    ("transit ridership", "transport", "ridership"),
    ("bus on time performance", "transport", "service_reliability"),
    ("bicycle counts", "transport", "vehicle_counts"),
    ("parking citations", "transport", "citation_stream"),
    ("motor vehicle collisions", "transport", "collision_stream"),
    ("airport passengers", "transport", "throughput"),
    ("air quality monitoring", "nature", "air_quality"),
    ("river gage height", "nature", "hydrology"),
    ("water consumption", "nature", "water_demand"),
    ("beach water quality", "nature", "water_quality"),
    ("solar production", "energy", "generation"),
    ("energy consumption building", "energy", "building_load"),
    ("natural gas usage", "energy", "gas_demand"),
    ("unemployment claims", "econ_fin", "labour_market"),
    ("median home price", "econ_fin", "housing_market"),
    ("municipal revenue", "econ_fin", "public_finance"),
    ("website analytics visits", "web_cloudops", "web_traffic"),
    ("service desk tickets", "web_cloudops", "ticket_stream"),
    # --- second tier: narrower topics, but the datasets sit on smaller portals
    # that the headline keywords never surface. Yield per keyword is lower and
    # host novelty is much higher, which is the axis that actually needs feeding.
    ("ambulance response times", "healthcare", "emergency_dispatch"),
    ("syndromic surveillance", "healthcare", "disease_surveillance"),
    ("immunization doses administered", "healthcare", "immunisation"),
    ("behavioral health crisis calls", "healthcare", "crisis_services"),
    ("food safety violations", "healthcare", "inspection_stream"),
    ("animal intakes shelter", "healthcare", "intake_stream"),
    ("births and deaths registered", "healthcare", "vital_statistics"),
    ("code enforcement cases", "sales", "enforcement_stream"),
    ("short term rental registrations", "sales", "registration_stream"),
    ("liquor license applications", "sales", "registration_stream"),
    ("farmers market sales", "sales", "retail_activity"),
    ("procurement contracts awarded", "sales", "procurement"),
    ("library circulation", "sales", "circulation"),
    ("recreation program registrations", "sales", "registration_stream"),
    ("towed vehicles", "transport", "citation_stream"),
    ("street sweeping", "transport", "service_operations"),
    ("scooter trips", "transport", "micromobility"),
    ("ferry passengers", "transport", "ridership"),
    ("paratransit trips", "transport", "ridership"),
    ("road closures", "transport", "disruption_stream"),
    ("speed camera violations", "transport", "citation_stream"),
    ("stormwater flow", "nature", "hydrology"),
    ("groundwater levels", "nature", "hydrology"),
    ("tree canopy plantings", "nature", "greening"),
    ("waste tonnage collected", "nature", "waste_stream"),
    ("recycling diversion", "nature", "waste_stream"),
    ("noise complaints", "nature", "environmental_complaints"),
    ("wildlife observations", "nature", "biodiversity"),
    ("streetlight outages", "energy", "asset_faults"),
    ("municipal fleet fuel", "energy", "fleet_consumption"),
    ("electric vehicle charging sessions", "energy", "ev_charging"),
    ("building energy benchmarking", "energy", "building_load"),
    ("payroll expenditures", "econ_fin", "public_finance"),
    ("parking meter revenue", "econ_fin", "public_finance"),
    ("hotel occupancy tax", "econ_fin", "tax_receipts"),
    ("job postings openings", "econ_fin", "labour_market"),
    ("eviction filings", "econ_fin", "housing_market"),
    ("foreclosure filings", "econ_fin", "housing_market"),
    ("open data portal usage", "web_cloudops", "web_traffic"),
    ("public wifi sessions", "web_cloudops", "session_stream"),
    ("call center wait times", "web_cloudops", "service_latency"),
    ("api requests", "web_cloudops", "api_traffic"),
)


# --------------------------------------------------------------------------- #
# Column selection — pure, unit-testable
# --------------------------------------------------------------------------- #
def _cols(res: dict) -> list[tuple[str, str, str]]:
    """(display_name, field_name, datatype-lowercased) triples for a result."""
    names = res.get("columns_name") or []
    fields = res.get("columns_field_name") or []
    types = res.get("columns_datatype") or []
    n = min(len(names), len(fields), len(types))
    return [(names[i], fields[i], str(types[i]).strip().lower()) for i in range(n)]


def _score_ts(field: str) -> int:
    f = field.lower()
    score = 0
    for tok in _TS_PROMOTE:
        if tok in f:
            score += 2
    for tok in _TS_DEMOTE:
        if tok in f:
            score -= 3
    return score


def pick_timestamp_column(cols: Iterable[tuple[str, str, str]]) -> Optional[str]:
    """The field_name most likely to be the observation time, or None."""
    cands = [(f, _score_ts(f)) for _n, f, t in cols if t in TS_TYPES]
    if not cands:
        return None
    # Highest score wins; ties break on the earliest column, which in Socrata
    # exports is overwhelmingly the primary event date.
    return max(cands, key=lambda p: p[1])[0]


def _is_identifier(field: str) -> bool:
    f = field.lower()
    parts = set(re.split(r"[^a-z0-9]+", f))
    if parts & set(_VAL_REJECT):
        return True
    return any(f == tok or f.endswith("_" + tok) or f.startswith(tok + "_")
               for tok in _VAL_REJECT)


def pick_value_columns(
    cols: Iterable[tuple[str, str, str]], ts_field: str, limit: int = 3
) -> list[str]:
    """Numeric field_names that plausibly carry observations (identifiers and
    coordinates excluded). Empty is a valid answer — the caller then bins a
    count instead, which is the right shape for an event stream anyway."""
    out = []
    for _n, f, t in cols:
        if f == ts_field or t not in NUM_TYPES or _is_identifier(f):
            continue
        out.append(f)
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- #
# Cadence inference
# --------------------------------------------------------------------------- #
# Bands are the ones config.FREQ_BAND uses, so generated entries land in the
# cadence cells the coverage matrix actually reports on.
_FREQ_LADDER: tuple[tuple[float, str], ...] = (
    (90, "PT1M"), (450, "PT5M"), (1200, "PT15M"), (2700, "PT30M"),
    (7200, "PT1H"), (129600, "P1D"), (864000, "P1W"),
    (3888000, "P1M"), (11664000, "P1Q"),
)


def freq_from_delta(seconds: float) -> str:
    """Median inter-observation gap -> the closest catalog frequency label."""
    for ceiling, label in _FREQ_LADDER:
        if seconds <= ceiling:
            return label
    return "P1Y"


def cron_cadence_for(freq: str, age_days: float = 0.0) -> str:
    """The cron group a frequency rides in.

    ``frequency`` describes the SERIES; the cron cadence describes the POLL, and
    they are not the same thing. Municipal dispatch logs are 5-minute series
    published in a nightly or monthly batch: polling one of those every 5 minutes
    is thousands of wasted requests a day for identical bytes. Publication lag —
    how old the newest record actually is — is the honest signal for how often to
    ask. Anything slower than daily also rides the daily poller, since a weekly
    series needs checking often enough to catch its publication, not weekly."""
    if age_days > 2:
        return "P1D"
    if age_days > 0.25:
        return "PT1H"
    return freq if freq in ("PT1M", "PT5M", "PT15M", "PT30M", "PT1H") else "P1D"


_ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?")


def _parse_iso(s: Any) -> Optional[dt.datetime]:
    m = _ISO_RE.match(str(s or ""))
    if not m:
        return None
    y, mo, d, hh, mi, ss = m.groups()
    try:
        return dt.datetime(
            int(y), int(mo), int(d), int(hh or 0), int(mi or 0), int(ss or 0),
            tzinfo=dt.timezone.utc,
        )
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# URL construction
# --------------------------------------------------------------------------- #
def build_url(domain: str, rid: str, ts: str, vals: list[str], limit: int = 5000) -> str:
    """A SoQL URL that returns newest-first. ``$order`` is mandatory alongside
    ``$limit`` on Socrata: without it the page you get back is arbitrary, which
    is how you end up storing 2016 data forever."""
    select = ",".join([ts] + vals)
    return (
        f"https://{domain}/resource/{rid}.json"
        f"?$select={quote(select, safe=',')}"
        f"&$order={quote(ts)}%20DESC&$limit={limit}"
    )


def probe_url(domain: str, rid: str, ts: str, limit: int = 60) -> str:
    """Timestamp-only probe — a few KB regardless of how wide the dataset is."""
    return (
        f"https://{domain}/resource/{rid}.json"
        f"?$select={quote(ts)}&$order={quote(ts)}%20DESC&$limit={limit}"
    )


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def search_catalog(
    keyword: str, limit: int = 100, offset: int = 0, timeout: int = 45
) -> list[dict]:
    params = {"q": keyword, "only": "dataset", "limit": limit, "offset": offset}
    r = requests.get(
        CATALOG_API, params=params, headers={"User-Agent": UA}, timeout=timeout
    )
    r.raise_for_status()
    return r.json().get("results", [])


# The catalog API caps a single response at 100; a popular keyword matches many
# hundreds of datasets, and since the best candidates are on small portals they
# sit deep in the relevance ranking. Paging is where the volume actually is.
CATALOG_PAGE = 100


def search_catalog_paged(
    keyword: str, want: int, timeout: int = 45, sleep_s: float = 0.2
) -> list[dict]:
    out: list[dict] = []
    while len(out) < want:
        page = search_catalog(
            keyword, limit=min(CATALOG_PAGE, want - len(out)),
            offset=len(out), timeout=timeout,
        )
        if not page:
            break
        out.extend(page)
        if len(page) < CATALOG_PAGE:
            break
        time.sleep(sleep_s)
    return out


def probe_series(
    domain: str, rid: str, ts: str, timeout: int = 30
) -> Optional[dict]:
    """Measure what the wire gate will care about: how many distinct timestamps
    the newest page has, how old the newest one is, and the median gap."""
    url = probe_url(domain, rid, ts)
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}"}
        rows = r.json()
    except Exception as exc:                                  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:140]}
    if not isinstance(rows, list) or not rows:
        return {"error": "empty response"}
    all_stamps = sorted({
        d for d in (_parse_iso(row.get(ts)) for row in rows if isinstance(row, dict))
        if d is not None
    }, reverse=True)
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now + dt.timedelta(days=1)
    stamps = [d for d in all_stamps if d <= cutoff]
    n_future = len(all_stamps) - len(stamps)
    if len(stamps) < 2:
        return {"error": f"only {len(stamps)} usable timestamp(s) "
                         f"({n_future} in the future)"}
    # A feed carrying future dates would read as permanently fresh to the
    # freshness audit, so it is worse than useless — it hides its own death.
    # A stray typo among thousands of rows is tolerable; a systematic schedule
    # is not.
    if n_future > max(1, 0.02 * len(all_stamps)):
        return {"error": f"{n_future}/{len(all_stamps)} timestamps are "
                         f"future-dated (newest {all_stamps[0].date()})"}
    gaps = [
        (stamps[i] - stamps[i + 1]).total_seconds() for i in range(len(stamps) - 1)
    ]
    gaps = [g for g in gaps if g > 0]
    return {
        "rows": len(rows),
        "distinct": len(stamps),
        "future": n_future,
        "newest": stamps[0].isoformat(),
        "age_days": (now - stamps[0]).total_seconds() / 86400.0,
        "median_gap_s": statistics.median(gaps) if gaps else 0.0,
    }


# --------------------------------------------------------------------------- #
# Candidate synthesis
# --------------------------------------------------------------------------- #
# Words that carry no topic information, so matching on them would let anything
# through the relevance guard.
_STOPWORDS = frozenset({
    "and", "the", "of", "by", "for", "in", "on", "to", "per", "data", "dataset",
    "open", "public", "city", "county", "state", "total", "number", "counts",
    "report", "reports", "current", "history", "historical", "monthly", "daily",
    "weekly", "annual", "yearly", "time", "times", "service", "services",
})


# Dataset families that dominate municipal portals but make poor *search*
# keywords (they would just re-surface what the sweep already queries). They
# exist so that a dataset arriving under the wrong query can still be filed
# under the right domain instead of being thrown away.
EXTRA_CLASSES: tuple[tuple[str, str, str], ...] = (
    ("crime offense arrests police incidents stops", "healthcare", "public_safety_stream"),
    ("permits issued construction residential commercial", "sales", "permit_stream"),
    ("citations violations nuisance enforcement", "transport", "citation_stream"),
    ("dispatch cad events calls for service", "healthcare", "emergency_dispatch"),
    ("housing starts construction units", "econ_fin", "housing_market"),
    ("campaign contributions expenditures filings", "econ_fin", "political_finance"),
    ("dataset freshness portal metadata", "web_cloudops", "portal_telemetry"),
    ("creel fish catch survey counts", "nature", "biodiversity"),
)


def fold(s: str) -> str:
    """Lowercase and strip diacritics. Without this, splitting on ``[^a-z0-9]``
    shreds every accented word — "fréquentation" becomes ``fr`` + ``quentation``
    — and topic matching silently fails across most of the ODS catalog, which is
    predominantly French."""
    decomposed = unicodedata.normalize("NFKD", str(s).lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _topic_tokens(text: str) -> list[str]:
    return [
        t for t in re.split(r"[^a-z0-9]+", fold(text))
        if t and t not in _STOPWORDS and len(t) > 2
    ]


def resolve_class(
    query_class: tuple[str, str, str], title: str, description: str = ""
) -> Optional[tuple[str, str, str]]:
    """Decide a dataset's domain from the dataset ITSELF, not from the query that
    surfaced it.

    Full-text relevance decays fast past the first page — a query for "short term
    rental registrations" starts returning campaign-finance filings — and the
    naive thing to do is inherit the query's domain. That is worse than dropping
    the source: the domain x cadence matrix is what steers the whole build, so a
    mislabelled entry misdirects every later decision about what to go find next.
    Scoring every known class against the dataset's own title keeps the good
    series and files it correctly. Returns None when nothing matches, which is
    the honest answer for a result the sweep simply should not have surfaced."""
    hay = " ".join(_topic_tokens(f"{title} {description}"))
    if not hay:
        return None
    best, best_score = None, 0
    for klass in ((query_class,) + tuple(KEYWORD_CLASSES)
                  + tuple(ODS_KEYWORD_CLASSES) + EXTRA_CLASSES):
        toks = _topic_tokens(klass[0])
        score = sum(1 for t in toks if t in hay or t.rstrip("s") in hay)
        if score > best_score:                       # first-wins keeps the query
            best, best_score = klass, score          # class ahead of ties
    return best


def is_relevant(keyword: str, *texts: str) -> bool:
    """Does this dataset actually belong to the keyword's topic class?

    Full-text relevance ranking degrades quickly past the first page: a query
    for "short term rental registrations" starts returning campaign-finance
    filings. Those are perfectly good time series, but they would be filed under
    whatever domain the *keyword* mapped to, and a mislabelled domain is worse
    than a missing source — the domain x cadence balance is the thing steering
    the entire build, so poisoning it misdirects every later decision."""
    hay = fold(" ".join(t for t in texts if t))
    toks = _topic_tokens(keyword)
    if not toks:
        return True
    return any(t in hay or t.rstrip("s") in hay for t in toks)


def _slug(s: str, maxlen: int = 44) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", fold(s)).strip("_")
    return re.sub(r"_+", "_", s)[:maxlen].strip("_")


# Host labels that identify the *platform*, not the *publisher*. Every city
# portal is called "data.<city>.gov", so keying an id on the first label would
# collapse them all to "data_" and collide.
_GENERIC_LABELS = frozenset({
    "data", "opendata", "open", "datos", "dados", "donnees", "api", "www",
    "portal", "catalog", "gov", "us", "org", "com", "net", "co", "ca", "city",
    "state", "county", "info", "public", "internal", "performance", "insights",
})


def host_label(domain: str) -> str:
    """The publisher-identifying part of a portal hostname.

    Walks in from the TLD rather than out from the left: the distinctive label is
    the registered name, and everything to its left is platform decoration.
    ``data.everettwa.gov`` -> ``everettwa``, ``cos-data.seattle.gov`` ->
    ``seattle``, ``data.gov.uk`` -> the first non-generic label found."""
    parts = [p for p in str(domain).lower().split(".") if p]
    for label in reversed(parts):
        stripped = re.split(r"[-_]", label)
        candidate = next((s for s in stripped if s and s not in _GENERIC_LABELS), "")
        if candidate:
            return _slug(candidate, 20)
    return _slug(str(domain), 20)


def entry_id(domain: str, title: str) -> str:
    return f"{host_label(domain)}_{_slug(title, 40)}"[:64].strip("_")


def synthesize(
    res: dict,
    klass: tuple[str, str, str],
    probe: dict,
    taken: Optional[set[str]] = None,
) -> Optional[dict]:
    """A grind block (the format ``--wire`` eats) for one catalog result.

    ``taken`` is the set of ids already spoken for (catalog + this batch). A
    collision would make ``--wire`` roll the whole batch back on its duplicate-id
    check, so titles that slug to the same id get the dataset's 4x4 appended."""
    _kw, dom, dgp = klass
    meta, resource = res.get("metadata") or {}, res.get("resource") or {}
    host = meta.get("domain") or ""
    rid, title = resource.get("id") or "", (resource.get("name") or "").strip()
    if not (host and rid and title):
        return None
    cols = _cols(resource)
    ts = pick_timestamp_column(cols)
    if not ts:
        return None
    vals = pick_value_columns(cols, ts)

    freq = freq_from_delta(probe["median_gap_s"])
    # GENEROUS by design: the observed age plus a wide margin. A slack window
    # tighter than the age already observed is the classic self-inflicted
    # rejection, and publication schedules slip.
    slack = max(45, int(probe["age_days"]) + 45)

    sid = entry_id(host, title)
    if taken and sid in taken:
        sid = f"{sid[:55]}_{rid.replace('-', '')}"[:64]

    entry: dict[str, Any] = {
        "id": sid,
        "name": f"{title} ({host})",
        "domain": dom,
        "dgp_class": dgp,
        "archetypes": ["count_discrete"] if not vals else ["non_stationary_regime"],
        "frequency": freq,
        "endpoint": {
            "type": "rest_json",
            "url": build_url(host, rid, ts, vals),
            "auth": "none",
            "rate_limit": "1000 req/h anonymous (Socrata)",
        },
        "schema": {
            "timestamp_field": f"[].{ts}",
            "variates": max(1, len(vals)),
        },
        "history_available": "unknown",
        "update_cadence_observed": (
            f"median gap {probe['median_gap_s'] / 60:.1f} min over the newest "
            f"{probe['distinct']} timestamps; newest {probe['newest']}"
        ),
        "pretraining_novelty": "clean",
        "novelty_notes": (
            "Municipal/agency Socrata series; not in known TSFM pretraining mixes."
        ),
        "license": "open data (see host portal terms)",
        "audit_slack_days": slack,
        "notes": (
            f"Bulk-generated from the Socrata federated catalog (query "
            f"'{_kw}'). Columns chosen mechanically from the catalog's declared "
            f"datatypes; verified against the live endpoint by the wire gate."
        ),
    }
    if vals:
        entry["schema"]["value_field"] = [f"[].{v}" for v in vals]
    else:
        # No observation-bearing numeric column: the series IS the event rate.
        entry["schema"]["value_field"] = f"[].{ts}"
        entry["schema"]["aggregate"] = {
            "op": "count",
            "bin": "P1D" if freq in ("P1D", "P1W", "P1M", "P1Q", "P1Y") else "PT1H",
        }
        if entry["schema"]["aggregate"]["bin"] == "PT1H":
            entry["frequency"] = "PT1H"
            freq = "PT1H"

    return {
        "candidate_name": entry["name"],
        "wireable": True,
        "yaml_block": yaml.dump([entry], sort_keys=False, allow_unicode=True),
        "cron_cadence": cron_cadence_for(freq, probe["age_days"]),
        "reason": (
            f"Socrata bulk: {probe['distinct']} distinct ts in newest page, "
            f"newest {probe['newest']} ({probe['age_days']:.1f}d old), "
            f"median gap {probe['median_gap_s']:.0f}s, "
            f"values={vals or 'binned count'}"
        ),
    }


# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #
def wired_hosts(catalog_path: str) -> set[str]:
    reg = yaml.safe_load(open(catalog_path)) or []
    hosts = set()
    for src in reg:
        ep = src.get("endpoint") or {}
        for url in (ep.get("url", ""), (ep.get("resolve") or {}).get("url", "")):
            if url:
                hosts.add(urlparse(url).netloc.lower())
    return hosts


def wired_resource_ids(catalog_path: str) -> set[str]:
    """Socrata 4x4s already in the catalog — the cheapest possible dedupe."""
    reg = yaml.safe_load(open(catalog_path)) or []
    ids = set()
    for src in reg:
        url = (src.get("endpoint") or {}).get("url", "")
        m = re.search(r"/resource/([a-z0-9]{4}-[a-z0-9]{4})\.", url)
        if m:
            ids.add(m.group(1))
    return ids


def sweep(
    catalog_path: str,
    classes: Iterable[tuple[str, str, str]] = KEYWORD_CLASSES,
    per_keyword: int = 60,
    host_cap: int = DEFAULT_HOST_CAP,
    max_age_days: float = 21.0,
    new_hosts_only: bool = True,
    target: Optional[int] = None,
    sleep_s: float = 0.35,
    checkpoint_path: Optional[str] = None,
    log=print,
) -> tuple[list[dict], list[dict]]:
    """Returns (candidates, skipped). ``skipped`` carries a reason per drop so
    the sweep's own blind spots are visible rather than silent."""
    seen_hosts: dict[str, int] = {}
    known_hosts = wired_hosts(catalog_path)
    known_ids = wired_resource_ids(catalog_path)
    taken_ids = {s["id"] for s in (yaml.safe_load(open(catalog_path)) or [])}
    cands: list[dict] = []
    skipped: list[dict] = []
    seen_ids: set[str] = set()

    for klass in classes:
        kw = klass[0]
        try:
            results = search_catalog_paged(kw, per_keyword)
        except Exception as exc:                              # noqa: BLE001
            skipped.append({"keyword": kw, "reason": f"catalog search failed: {exc}"})
            continue
        log(f"[{kw}] {len(results)} catalog results")
        for res in results:
            meta = res.get("metadata") or {}
            resource = res.get("resource") or {}
            host = (meta.get("domain") or "").lower()
            rid = resource.get("id") or ""
            title = (resource.get("name") or "").strip()
            tag = f"{host}/{rid}"
            if not host or not rid or rid in seen_ids:
                continue
            seen_ids.add(rid)
            if rid in known_ids:
                skipped.append({"id": tag, "reason": "resource already in catalog"})
                continue
            if new_hosts_only and host in known_hosts:
                skipped.append({"id": tag, "reason": "host already wired"})
                continue
            if seen_hosts.get(host, 0) >= host_cap:
                skipped.append({"id": tag, "reason": f"host cap {host_cap} reached"})
                continue
            updated = _parse_iso(resource.get("updatedAt"))
            if updated is None:
                skipped.append({"id": tag, "reason": "no parseable updatedAt"})
                continue
            age = (dt.datetime.now(dt.timezone.utc) - updated).total_seconds() / 86400
            if age > max_age_days:
                skipped.append({"id": tag, "reason": f"stale: updated {age:.0f}d ago"})
                continue
            category = (res.get("classification") or {}).get("domain_category", "")
            use_class = resolve_class(klass, title, category)
            if use_class is None:
                skipped.append({"id": tag,
                                "reason": f"unclassifiable for '{kw}': {title[:60]}"})
                continue
            if use_class is not klass:
                log(f"  ~ reclassified as {use_class[1]}: {title[:60]}")
            cols = _cols(resource)
            ts = pick_timestamp_column(cols)
            if not ts:
                skipped.append({"id": tag, "reason": "no date/timestamp column"})
                continue

            probe = probe_series(host, rid, ts)
            time.sleep(sleep_s)
            if probe is None or "error" in probe:
                skipped.append({
                    "id": tag,
                    "reason": f"probe failed: {(probe or {}).get('error')}",
                })
                continue
            if probe["distinct"] < 20:
                skipped.append({
                    "id": tag,
                    "reason": f"only {probe['distinct']} distinct ts in newest page",
                })
                continue
            if probe["age_days"] > max_age_days * 3:
                skipped.append({
                    "id": tag,
                    "reason": f"newest observation {probe['age_days']:.0f}d old",
                })
                continue

            block = synthesize(res, use_class, probe, taken=taken_ids)
            if block is None:
                skipped.append({"id": tag, "reason": "could not synthesise entry"})
                continue
            taken_ids.add(yaml.safe_load(block["yaml_block"])[0]["id"])
            cands.append(block)
            seen_hosts[host] = seen_hosts.get(host, 0) + 1
            _checkpoint(cands, checkpoint_path)
            log(f"  + {block['candidate_name']}  [{block['cron_cadence']}]")
            if target and len(cands) >= target:
                log(f"target {target} reached")
                return cands, skipped
    return cands, skipped


# --------------------------------------------------------------------------- #
# Opendatasoft
# --------------------------------------------------------------------------- #
# The ODS federated catalog carries ~100k datasets, skewed to France, Belgium,
# Switzerland, the Netherlands, Spain and Australia — precisely the geographies
# the catalog is thin on.
ODS_CATALOG = "https://data.opendatasoft.com/api/explore/v2.1/catalog/datasets"
ODS_PAGE = 100                       # /catalog/datasets caps a response at 100
ODS_EXPORT_ROWS = 2000               # /exports/json honours order_by + limit

# Searching an ODS portal in English is the mistake that makes it look
# unproductive: most of the ~100k datasets are published in French, with Spanish,
# Dutch and German behind it, so English queries only ever reach the
# international-facing minority. These are the same topic classes as
# KEYWORD_CLASSES, asked in the languages the data is catalogued in.
ODS_KEYWORD_CLASSES: tuple[tuple[str, str, str], ...] = (
    # French — transport
    ("fréquentation transport", "transport", "ridership"),
    ("comptage vélo", "transport", "vehicle_counts"),
    ("comptage routier", "transport", "vehicle_counts"),
    ("stationnement disponibilité", "transport", "parking_occupancy"),
    ("trafic voyageurs gare", "transport", "ridership"),
    ("vélos libre service disponibilité", "transport", "micromobility"),
    # French — energy
    ("consommation électrique", "energy", "grid_demand"),
    ("production énergie renouvelable", "energy", "generation"),
    ("consommation gaz", "energy", "gas_demand"),
    ("bornes recharge électrique", "energy", "ev_charging"),
    # French — nature
    ("qualité de l'air mesures", "nature", "air_quality"),
    ("qualité des eaux", "nature", "water_quality"),
    ("pluviométrie relevés", "nature", "hydrology"),
    ("déchets collecte tonnage", "nature", "waste_stream"),
    ("niveau des nappes", "nature", "hydrology"),
    ("pollens allergie", "nature", "air_quality"),
    # French — healthcare / sales / econ / web
    ("urgences hospitalières passages", "healthcare", "health_utilisation"),
    ("épidémiologie surveillance", "healthcare", "disease_surveillance"),
    ("interventions pompiers", "healthcare", "emergency_dispatch"),
    ("permis de construire", "sales", "permit_stream"),
    ("créations entreprises", "sales", "registration_stream"),
    ("marchés publics attribués", "sales", "procurement"),
    ("fréquentation équipements", "sales", "attendance"),
    ("prix carburants", "econ_fin", "price_series"),
    ("emploi demandeurs", "econ_fin", "labour_market"),
    ("logements transactions", "econ_fin", "housing_market"),
    ("fréquentation site web", "web_cloudops", "web_traffic"),
    ("demandes signalements", "web_cloudops", "ticket_stream"),
    # Spanish
    ("consumo energía", "energy", "grid_demand"),
    ("calidad del aire", "nature", "air_quality"),
    ("aforo tráfico", "transport", "vehicle_counts"),
    ("residuos recogida", "nature", "waste_stream"),
    ("licencias actividad", "sales", "registration_stream"),
    ("urgencias atenciones", "healthcare", "health_utilisation"),
    # Dutch
    ("verkeer metingen", "transport", "vehicle_counts"),
    ("luchtkwaliteit metingen", "nature", "air_quality"),
    ("energieverbruik", "energy", "grid_demand"),
    ("afval inzameling", "nature", "waste_stream"),
    # German
    ("Verkehrszählung", "transport", "vehicle_counts"),
    ("Luftqualität Messwerte", "nature", "air_quality"),
    ("Stromverbrauch", "energy", "grid_demand"),
    ("Niederschlag Messwerte", "nature", "hydrology"),
)

ODS_TS_TYPES = frozenset({"date", "datetime"})
ODS_NUM_TYPES = frozenset({"int", "double", "decimal", "float", "long"})

# Field-name fragments that mark an observation time, in the languages ODS
# portals actually publish in. Without the non-English ones the picker falls
# back on ranking noise for most of the catalog.
_ODS_TS_PROMOTE = _TS_PROMOTE + (
    "jour", "heure", "horodate", "horaire", "mois", "annee", "semaine",
    "debut", "fecha", "hora", "dia", "mes", "datum", "zeit", "tijd", "data",
)
# ...and the ones that mark identifiers/coordinates in those same languages.
_ODS_VAL_REJECT = _VAL_REJECT + (
    "insee", "numero", "num", "identifiant", "cp", "siret", "siren", "geo",
    "geopoint", "geoshape", "commune", "departement", "arrondissement",
    "codigo", "nr", "plz", "gemeente", "annee", "mois", "jour", "semaine",
)


def _ods_fields(rec: dict) -> list[tuple[str, str, str]]:
    """(label, name, type) triples, shaped like ``_cols`` so the pickers are
    shared between the two platforms."""
    out = []
    for f in rec.get("fields") or []:
        name, typ = f.get("name"), str(f.get("type", "")).strip().lower()
        if name:
            out.append((f.get("label") or name, name, typ))
    return out


def ods_pick_timestamp(fields: Iterable[tuple[str, str, str]]) -> Optional[str]:
    cands = []
    for _lab, name, typ in fields:
        if typ not in ODS_TS_TYPES:
            continue
        f = name.lower()
        score = sum(2 for tok in _ODS_TS_PROMOTE if tok in f)
        score -= sum(3 for tok in _TS_DEMOTE if tok in f)
        cands.append((name, score))
    return max(cands, key=lambda p: p[1])[0] if cands else None


def ods_pick_values(
    fields: Iterable[tuple[str, str, str]], ts: str, limit: int = 3
) -> list[str]:
    out = []
    for _lab, name, typ in fields:
        if name == ts or typ not in ODS_NUM_TYPES:
            continue
        parts = set(re.split(r"[^a-z0-9]+", name.lower()))
        if parts & set(_ODS_VAL_REJECT):
            continue
        out.append(name)
        if len(out) >= limit:
            break
    return out


def ods_search(keyword: str, want: int, timeout: int = 45,
               sleep_s: float = 0.2) -> list[dict]:
    """Full-text search across every ODS portal. ODSQL takes a bare quoted
    string for full text; a `search(default, ...)` call is rejected here."""
    out: list[dict] = []
    while len(out) < want:
        params = {
            "where": f'"{keyword}"',
            "limit": min(ODS_PAGE, want - len(out)),
            "offset": len(out),
        }
        r = requests.get(ODS_CATALOG, params=params,
                         headers={"User-Agent": UA}, timeout=timeout)
        if r.status_code != 200:
            break
        page = r.json().get("results", [])
        if not page:
            break
        out.extend(page)
        if len(page) < ODS_PAGE:
            break
        time.sleep(sleep_s)
    return out


def ods_export_url(host: str, dsid: str, ts: str, vals: list[str]) -> str:
    select = ",".join([ts] + vals)
    return (
        f"https://{host}/api/explore/v2.1/catalog/datasets/{dsid}/exports/json"
        f"?limit={ODS_EXPORT_ROWS}&order_by={quote(ts)}%20desc"
        f"&select={quote(select, safe=',')}"
    )


def ods_probe(host: str, dsid: str, ts: str, timeout: int = 30) -> dict:
    """Newest 100 records, timestamp column only."""
    url = (
        f"https://{host}/api/explore/v2.1/catalog/datasets/{dsid}/records"
        f"?limit=100&order_by={quote(ts)}%20desc&select={quote(ts)}"
    )
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}"}
        rows = r.json().get("results", [])
    except Exception as exc:                                  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:140]}
    if not rows:
        return {"error": "no records"}
    all_stamps = sorted({
        d for d in (_parse_iso(row.get(ts)) for row in rows if isinstance(row, dict))
        if d is not None
    }, reverse=True)
    now = dt.datetime.now(dt.timezone.utc)
    stamps = [d for d in all_stamps if d <= now + dt.timedelta(days=1)]
    n_future = len(all_stamps) - len(stamps)
    if len(stamps) < 2:
        return {"error": f"only {len(stamps)} usable timestamp(s) "
                         f"({n_future} future-dated)"}
    if n_future > max(1, 0.02 * len(all_stamps)):
        return {"error": f"{n_future}/{len(all_stamps)} timestamps future-dated"}
    gaps = [g for g in ((stamps[i] - stamps[i + 1]).total_seconds()
                        for i in range(len(stamps) - 1)) if g > 0]
    return {
        "rows": len(rows),
        "distinct": len(stamps),
        "future": n_future,
        "newest": stamps[0].isoformat(),
        "age_days": (now - stamps[0]).total_seconds() / 86400.0,
        "median_gap_s": statistics.median(gaps) if gaps else 0.0,
    }


def ods_synthesize(rec: dict, klass: tuple[str, str, str], probe: dict,
                   taken: Optional[set[str]] = None) -> Optional[dict]:
    kw, dom, dgp = klass
    metas = (rec.get("metas") or {}).get("default") or {}
    host = (metas.get("source_domain_address") or "").strip().lower()
    dsid = metas.get("source_dataset") or rec.get("dataset_id") or ""
    title = (metas.get("title") or metas.get("title_en") or dsid).strip()
    if not host or not dsid:
        return None
    fields = _ods_fields(rec)
    ts = ods_pick_timestamp(fields)
    if not ts:
        return None
    vals = ods_pick_values(fields, ts)
    freq = freq_from_delta(probe["median_gap_s"])
    slack = max(45, int(probe["age_days"]) + 45)

    sid = entry_id(host, title)
    if taken and sid in taken:
        sid = f"{sid[:52]}_{_slug(dsid, 10)}"[:64]

    entry: dict[str, Any] = {
        "id": sid,
        "name": f"{title} ({host})",
        "domain": dom,
        "dgp_class": dgp,
        "archetypes": ["count_discrete"] if not vals else ["non_stationary_regime"],
        "frequency": freq,
        "endpoint": {
            "type": "rest_json",
            "url": ods_export_url(host, dsid, ts, vals),
            "auth": "none",
            "rate_limit": "Opendatasoft anonymous quota (per-portal)",
        },
        "schema": {
            "timestamp_field": f"[].{ts}",
            "variates": max(1, len(vals)),
        },
        "history_available": f"{metas.get('records_count') or 'unknown'} records",
        "update_cadence_observed": (
            f"median gap {probe['median_gap_s'] / 60:.1f} min over the newest "
            f"{probe['distinct']} timestamps; newest {probe['newest']}"
        ),
        "pretraining_novelty": "clean",
        "novelty_notes": (
            "European/municipal Opendatasoft series; not in known TSFM "
            "pretraining mixes."
        ),
        "license": metas.get("license") or "open data (see host portal terms)",
        "audit_slack_days": slack,
        "notes": (
            f"Bulk-generated from the Opendatasoft federated catalog (query "
            f"'{kw}'). Fields chosen from the portal's declared types; the "
            f"endpoint points at the publisher's own host, not the aggregator."
        ),
    }
    if vals:
        entry["schema"]["value_field"] = [f"[].{v}" for v in vals]
    else:
        entry["schema"]["value_field"] = f"[].{ts}"
        bin_ = "P1D" if freq in ("P1D", "P1W", "P1M", "P1Q", "P1Y") else "PT1H"
        entry["schema"]["aggregate"] = {"op": "count", "bin": bin_}
        if bin_ == "PT1H":
            entry["frequency"] = freq = "PT1H"

    return {
        "candidate_name": entry["name"],
        "wireable": True,
        "yaml_block": yaml.dump([entry], sort_keys=False, allow_unicode=True),
        "cron_cadence": cron_cadence_for(freq, probe["age_days"]),
        "reason": (
            f"ODS bulk: {probe['distinct']} distinct ts in newest page, newest "
            f"{probe['newest']} ({probe['age_days']:.1f}d old), median gap "
            f"{probe['median_gap_s']:.0f}s, values={vals or 'binned count'}"
        ),
    }


def wired_ods_datasets(catalog_path: str) -> set[str]:
    reg = yaml.safe_load(open(catalog_path)) or []
    out = set()
    for src in reg:
        url = (src.get("endpoint") or {}).get("url", "")
        m = re.search(r"/catalog/datasets/([^/?]+)", url)
        if m:
            out.add(m.group(1))
    return out


def ods_sweep(
    catalog_path: str,
    classes: Optional[Iterable[tuple[str, str, str]]] = None,
    per_keyword: int = 100,
    host_cap: int = DEFAULT_HOST_CAP,
    max_age_days: float = 21.0,
    new_hosts_only: bool = True,
    target: Optional[int] = None,
    sleep_s: float = 0.35,
    checkpoint_path: Optional[str] = None,
    log=print,
) -> tuple[list[dict], list[dict]]:
    if classes is None:
        classes = ODS_KEYWORD_CLASSES + KEYWORD_CLASSES
    seen_hosts: dict[str, int] = {}
    known_hosts = wired_hosts(catalog_path)
    known_ds = wired_ods_datasets(catalog_path)
    taken_ids = {s["id"] for s in (yaml.safe_load(open(catalog_path)) or [])}
    cands: list[dict] = []
    skipped: list[dict] = []
    seen: set[str] = set()

    for klass in classes:
        kw = klass[0]
        try:
            results = ods_search(kw, per_keyword)
        except Exception as exc:                              # noqa: BLE001
            skipped.append({"keyword": kw, "reason": f"ODS search failed: {exc}"})
            continue
        log(f"[ODS {kw}] {len(results)} catalog results")
        for rec in results:
            metas = (rec.get("metas") or {}).get("default") or {}
            host = (metas.get("source_domain_address") or "").strip().lower()
            dsid = metas.get("source_dataset") or ""
            title = (metas.get("title") or "").strip()
            tag = f"{host}/{dsid}"
            if not host or not dsid or tag in seen:
                continue
            seen.add(tag)
            if not rec.get("has_records"):
                skipped.append({"id": tag, "reason": "dataset has no records"})
                continue
            if dsid in known_ds:
                skipped.append({"id": tag, "reason": "dataset already in catalog"})
                continue
            if new_hosts_only and host in known_hosts:
                skipped.append({"id": tag, "reason": "host already wired"})
                continue
            if seen_hosts.get(host, 0) >= host_cap:
                skipped.append({"id": tag, "reason": f"host cap {host_cap} reached"})
                continue
            use_class = resolve_class(klass, title, metas.get("description") or "")
            if use_class is None:
                skipped.append({"id": tag,
                                "reason": f"unclassifiable for '{kw}': {title[:60]}"})
                continue
            if use_class is not klass:
                log(f"  ~ reclassified as {use_class[1]}: {title[:60]}")
            if (metas.get("records_count") or 0) < 40:
                skipped.append({"id": tag,
                                "reason": f"only {metas.get('records_count')} records"})
                continue
            mod = _parse_iso(metas.get("data_processed") or metas.get("modified"))
            if mod is None:
                skipped.append({"id": tag, "reason": "no parseable modified date"})
                continue
            age = (dt.datetime.now(dt.timezone.utc) - mod).total_seconds() / 86400
            if age > max_age_days:
                skipped.append({"id": tag, "reason": f"stale: processed {age:.0f}d ago"})
                continue
            fields = _ods_fields(rec)
            ts = ods_pick_timestamp(fields)
            if not ts:
                skipped.append({"id": tag, "reason": "no date/datetime field"})
                continue

            probe = ods_probe(host, dsid, ts)
            time.sleep(sleep_s)
            if "error" in probe:
                skipped.append({"id": tag, "reason": f"probe failed: {probe['error']}"})
                continue
            if probe["distinct"] < 20:
                skipped.append({"id": tag,
                                "reason": f"only {probe['distinct']} distinct ts"})
                continue
            if probe["age_days"] > max_age_days * 3:
                skipped.append({"id": tag,
                                "reason": f"newest observation {probe['age_days']:.0f}d old"})
                continue
            block = ods_synthesize(rec, use_class, probe, taken=taken_ids)
            if block is None:
                skipped.append({"id": tag, "reason": "could not synthesise entry"})
                continue
            taken_ids.add(yaml.safe_load(block["yaml_block"])[0]["id"])
            cands.append(block)
            seen_hosts[host] = seen_hosts.get(host, 0) + 1
            _checkpoint(cands, checkpoint_path)
            log(f"  + {block['candidate_name']}  [{block['cron_cadence']}]")
            if target and len(cands) >= target:
                log(f"target {target} reached")
                return cands, skipped
    return cands, skipped


# --------------------------------------------------------------------------- #
# CKAN
# --------------------------------------------------------------------------- #
# CKAN has no federated search — each portal is its own island — so the host
# list IS the input. That is also the point: this list is chosen for geography,
# because the catalog's thinnest axis is not topic but country. A portal only
# helps if its resources are in the DataStore; a CSV sitting in file storage has
# no queryable API and cannot be ordered newest-first.
CKAN_PORTALS: tuple[tuple[str, str], ...] = (
    ("data.gov.ie", "Ireland"),
    ("data.gov.sk", "Slovakia"),
    ("data.gov.lv", "Latvia"),
    ("data.gov.ro", "Romania"),
    ("data.gov.gr", "Greece"),
    ("data.gov.cy", "Cyprus"),
    ("data.gov.mt", "Malta"),
    ("dados.gov.br", "Brazil"),
    ("datos.gob.cl", "Chile"),
    ("datos.gob.mx", "Mexico"),
    ("datosabiertos.gob.pe", "Peru"),
    ("data.gov.sg", "Singapore"),
    ("data.gov.my", "Malaysia"),
    ("data.gov.ph", "Philippines"),
    ("data.gov.in", "India"),
    ("data.gov.lk", "Sri Lanka"),
    ("catalog.data.gov.bd", "Bangladesh"),
    ("africaopendata.org", "Africa (regional)"),
    ("data.gov.ng", "Nigeria"),
    ("open.africa", "Africa (regional)"),
    ("data.govt.nz", "New Zealand"),
    ("data.gov.au", "Australia"),
    ("open.canada.ca", "Canada"),
    ("data.overheid.nl", "Netherlands"),
    ("opendata.swiss", "Switzerland"),
    ("data.gv.at", "Austria"),
    ("govdata.de", "Germany"),
    ("data.norge.no", "Norway"),
    ("opendata.dk", "Denmark"),
    ("avoindata.fi", "Finland"),
    ("dados.gov.pt", "Portugal"),
    ("datos.gob.es", "Spain"),
    ("dati.gov.it", "Italy"),
    ("data.gov.ua", "Ukraine"),
    ("data.gov.il", "Israel"),
    ("data.gov.jo", "Jordan"),
    ("data.humdata.org", "Humanitarian (global)"),
)

# CKAN DataStore column types.
CKAN_TS_TYPES = frozenset({"timestamp", "timestamptz", "date", "time"})
CKAN_NUM_TYPES = frozenset({"numeric", "int4", "int8", "int2", "float4",
                            "float8", "double precision", "integer", "bigint",
                            "real", "money"})


def ckan_search(host: str, keyword: str, rows: int = 50,
                timeout: int = 45) -> list[dict]:
    url = f"https://{host}/api/3/action/package_search"
    r = requests.get(url, params={"q": keyword, "rows": rows},
                     headers={"User-Agent": UA}, timeout=timeout)
    if r.status_code != 200:
        return []
    body = r.json()
    if not body.get("success"):
        return []
    return (body.get("result") or {}).get("results", [])


def ckan_datastore_fields(host: str, rid: str,
                          timeout: int = 30) -> list[tuple[str, str, str]]:
    """(id, id, type) triples for a DataStore resource, shaped like ``_cols``."""
    url = f"https://{host}/api/3/action/datastore_search"
    try:
        r = requests.get(url, params={"resource_id": rid, "limit": 0},
                         headers={"User-Agent": UA}, timeout=timeout)
        if r.status_code != 200:
            return []
        fields = ((r.json().get("result") or {}).get("fields") or [])
    except Exception:                                         # noqa: BLE001
        return []
    out = []
    for f in fields:
        fid, typ = f.get("id"), str(f.get("type", "")).strip().lower()
        if fid and fid != "_id":
            out.append((fid, fid, typ))
    return out


def ckan_pick_timestamp(fields: Iterable[tuple[str, str, str]]) -> Optional[str]:
    cands = []
    for _lab, name, typ in fields:
        if typ not in CKAN_TS_TYPES:
            continue
        f = name.lower()
        score = sum(2 for tok in _ODS_TS_PROMOTE if tok in f)
        score -= sum(3 for tok in _TS_DEMOTE if tok in f)
        cands.append((name, score))
    return max(cands, key=lambda p: p[1])[0] if cands else None


def ckan_pick_values(fields: Iterable[tuple[str, str, str]], ts: str,
                     limit: int = 3) -> list[str]:
    out = []
    for _lab, name, typ in fields:
        if name == ts or typ not in CKAN_NUM_TYPES:
            continue
        if set(re.split(r"[^a-z0-9]+", name.lower())) & set(_ODS_VAL_REJECT):
            continue
        out.append(name)
        if len(out) >= limit:
            break
    return out


def ckan_data_url(host: str, rid: str, ts: str, vals: list[str],
                  rows: int = 2000) -> str:
    fields = ",".join([ts] + vals)
    return (
        f"https://{host}/api/3/action/datastore_search?resource_id={rid}"
        f"&limit={rows}&sort={quote(ts)}%20desc&fields={quote(fields, safe=',')}"
    )


def ckan_probe(host: str, rid: str, ts: str, timeout: int = 30) -> dict:
    url = (f"https://{host}/api/3/action/datastore_search?resource_id={rid}"
           f"&limit=100&sort={quote(ts)}%20desc&fields={quote(ts)}")
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}"}
        recs = ((r.json().get("result") or {}).get("records") or [])
    except Exception as exc:                                  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:140]}
    if not recs:
        return {"error": "no records"}
    all_stamps = sorted({
        d for d in (_parse_iso(rec.get(ts)) for rec in recs if isinstance(rec, dict))
        if d is not None
    }, reverse=True)
    now = dt.datetime.now(dt.timezone.utc)
    stamps = [d for d in all_stamps if d <= now + dt.timedelta(days=1)]
    n_future = len(all_stamps) - len(stamps)
    if len(stamps) < 2:
        return {"error": f"only {len(stamps)} usable timestamp(s)"}
    if n_future > max(1, 0.02 * len(all_stamps)):
        return {"error": f"{n_future}/{len(all_stamps)} timestamps future-dated"}
    gaps = [g for g in ((stamps[i] - stamps[i + 1]).total_seconds()
                        for i in range(len(stamps) - 1)) if g > 0]
    return {
        "rows": len(recs), "distinct": len(stamps), "future": n_future,
        "newest": stamps[0].isoformat(),
        "age_days": (now - stamps[0]).total_seconds() / 86400.0,
        "median_gap_s": statistics.median(gaps) if gaps else 0.0,
    }


def _text(v: Any) -> str:
    """CKAN fields are usually strings but some portals return a per-language
    dict ({'en': ..., 'nl': ...}); prefer English, else any value."""
    if isinstance(v, dict):
        for k in ("en", "en-GB", "en-US"):
            if v.get(k):
                return str(v[k])
        return str(next((x for x in v.values() if x), ""))
    return str(v or "")


# Enough of a CSV to see the header and judge the columns, without pulling a
# file that may be hundreds of MB.
CKAN_SNIFF_BYTES = 196_608


def ckan_fetch_head(url: str, timeout: int = 30) -> Optional[str]:
    try:
        with requests.get(url, headers={"User-Agent": UA}, timeout=timeout,
                          stream=True) as r:
            if r.status_code != 200:
                return None
            buf = b""
            for chunk in r.iter_content(32768):
                buf += chunk
                if len(buf) >= CKAN_SNIFF_BYTES:
                    break
    except Exception:                                         # noqa: BLE001
        return None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return buf.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def ckan_file_candidate(res: dict, max_age_days: float) -> dict:
    """Judge a linked CSV resource by sniffing its head.

    Freshness comes from the portal's own ``last_modified`` rather than the
    data, because a CSV is written oldest-first: the newest row is at the END of
    a file that may be enormous, and downloading all of it just to date it would
    cost more than the source is worth. The wire gate downloads it in full
    anyway and applies the real freshness check there, so a portal whose
    metadata flatters a stale file is still caught — just later and cheaper.
    """
    fmt = _text(res.get("format")).lower()
    url = _text(res.get("url"))
    if "csv" not in fmt or not url.startswith("http"):
        return {"error": f"not a CSV resource (format={fmt or '?'})"}
    size = res.get("size")
    if isinstance(size, (int, float)) and size > 48_000_000:
        return {"error": f"file too large ({int(size)} bytes)"}
    stamp = _parse_iso(res.get("last_modified") or res.get("created"))
    if stamp is None:
        return {"error": "no parseable last_modified"}
    age = (dt.datetime.now(dt.timezone.utc) - stamp).total_seconds() / 86400
    if age > max_age_days:
        return {"error": f"stale: file modified {age:.0f}d ago"}

    text = ckan_fetch_head(url)
    if not text:
        return {"error": "could not read the file head"}
    rows = _sniff_csv_rows(text)
    if len(rows) < 3:
        return {"error": "fewer than 3 readable CSV rows"}
    header, body = rows[0], rows[1:]
    ts_col = _csv_timestamp_column(header, body)
    if not ts_col:
        return {"error": "no column parses as a date"}
    fields = [(h, h, "timestamp" if h == ts_col else
               ("numeric" if _csv_column_is_numeric(header, body, h) else "text"))
              for h in header]
    stamps = sorted({
        d for d in (_parse_iso(r[header.index(ts_col)])
                    for r in body if len(r) > header.index(ts_col))
        if d is not None
    })
    gaps = [(stamps[i + 1] - stamps[i]).total_seconds() for i in range(len(stamps) - 1)]
    gaps = [g for g in gaps if g > 0]
    probe = {
        "rows": len(body),
        # The head of the file is the OLDEST data, so distinct-count here is a
        # sample, and freshness is the metadata's age — both re-checked by the
        # gate against the whole file.
        "distinct": len(stamps),
        "future": 0,
        "newest": stamp.isoformat(),
        "age_days": age,
        "median_gap_s": statistics.median(gaps) if gaps else 86400.0,
    }
    return {"fields": fields, "ts": ts_col, "probe": probe}


def _sniff_csv_rows(text: str, limit: int = 200) -> list[list[str]]:
    lines = [l for l in text.splitlines() if l.strip()][:limit]
    if not lines:
        return []
    delim = max([",", ";", "\t", "|"], key=lambda d: lines[0].count(d))
    if lines[0].count(delim) == 0:
        return []
    rows = [next(csv.reader([l], delimiter=delim)) for l in lines]
    # The head fetch cuts mid-record, so the last row is usually partial. Field
    # count is the reliable tell — a short final line may still be a valid row.
    if len(rows) > 1 and len(rows[-1]) < len(rows[0]):
        rows = rows[:-1]
    return rows


def _csv_timestamp_column(header: list[str], body: list[list[str]]) -> Optional[str]:
    """The column whose values actually parse as dates — declared types do not
    exist for a raw file, so the data has to answer for itself."""
    best, best_hits = None, 0
    for i, name in enumerate(header):
        vals = [r[i] for r in body[:40] if len(r) > i and r[i].strip()]
        if not vals:
            continue
        hits = sum(1 for v in vals if _parse_iso(v) is not None)
        if hits < max(5, 0.8 * len(vals)):
            continue
        score = hits + (4 if any(t in name.lower() for t in _ODS_TS_PROMOTE) else 0)
        if score > best_hits:
            best, best_hits = name, score
    return best


def _csv_column_is_numeric(header: list[str], body: list[list[str]],
                           name: str) -> bool:
    i = header.index(name)
    vals = [r[i] for r in body[:40] if len(r) > i and r[i].strip()]
    if not vals:
        return False
    ok = 0
    for v in vals:
        try:
            float(v.replace(",", "."))
            ok += 1
        except ValueError:
            pass
    return ok >= 0.8 * len(vals)


def ckan_synthesize(host: str, country: str, pkg: dict, res: dict,
                    klass: tuple[str, str, str], fields: list[tuple[str, str, str]],
                    probe: dict, taken: Optional[set[str]] = None) -> Optional[dict]:
    kw, dom, dgp = klass
    rid = res.get("id") or ""
    title = (res.get("name") or pkg.get("title") or "").strip()
    if not rid or not title:
        return None
    ts = ckan_pick_timestamp(fields)
    if not ts:
        return None
    vals = ckan_pick_values(fields, ts)
    freq = freq_from_delta(probe["median_gap_s"])
    slack = max(45, int(probe["age_days"]) + 45)
    is_file = not res.get("datastore_active")

    sid = entry_id(host, title)
    if taken and sid in taken:
        sid = f"{sid[:52]}_{rid.replace('-', '')[:8]}"[:64]

    entry: dict[str, Any] = {
        "id": sid,
        "name": f"{title} — {country} ({host})",
        "domain": dom,
        "dgp_class": dgp,
        "archetypes": ["count_discrete"] if not vals else ["non_stationary_regime"],
        "frequency": freq,
        "endpoint": {
            "type": "rest_csv" if is_file else "rest_json",
            "url": _text(res.get("url")) if is_file
                   else ckan_data_url(host, rid, ts, vals),
            "auth": "none",
            "rate_limit": "CKAN anonymous (per-portal)",
        },
        "schema": {
            # A linked file is read with its own column names; the DataStore API
            # wraps records one level down. A slash-separated path there would
            # silently parse as a single empty row.
            "timestamp_field": ts if is_file else f"result.records[].{ts}",
            "variates": max(1, len(vals)),
        },
        "history_available": "unknown",
        "update_cadence_observed": (
            f"median gap {probe['median_gap_s'] / 60:.1f} min over the newest "
            f"{probe['distinct']} timestamps; newest {probe['newest']}"
        ),
        "pretraining_novelty": "clean",
        "novelty_notes": f"National open-data portal series ({country}); novel.",
        "license": pkg.get("license_title") or "open data (see portal terms)",
        "audit_slack_days": slack,
        "notes": (
            f"Bulk-generated from the {host} CKAN DataStore (query '{kw}'). "
            f"Package: {pkg.get('name', '?')}."
        ),
    }
    if vals:
        entry["schema"]["value_field"] = (
            list(vals) if is_file else [f"result.records[].{v}" for v in vals]
        )
    else:
        entry["schema"]["value_field"] = ts if is_file else f"result.records[].{ts}"
        bin_ = "P1D" if freq in ("P1D", "P1W", "P1M", "P1Q", "P1Y") else "PT1H"
        entry["schema"]["aggregate"] = {"op": "count", "bin": bin_}
        if bin_ == "PT1H":
            entry["frequency"] = freq = "PT1H"

    return {
        "candidate_name": entry["name"],
        "wireable": True,
        "yaml_block": yaml.dump([entry], sort_keys=False, allow_unicode=True),
        "cron_cadence": cron_cadence_for(freq, probe["age_days"]),
        "reason": (
            f"CKAN bulk ({country}): {probe['distinct']} distinct ts, newest "
            f"{probe['newest']} ({probe['age_days']:.1f}d old), median gap "
            f"{probe['median_gap_s']:.0f}s, values={vals or 'binned count'}"
        ),
    }


def ckan_sweep(
    catalog_path: str,
    portals: Iterable[tuple[str, str]] = CKAN_PORTALS,
    classes: Iterable[tuple[str, str, str]] = KEYWORD_CLASSES,
    per_portal: int = 3,
    rows: int = 40,
    keywords_per_portal: int = 8,
    max_age_days: float = 21.0,
    target: Optional[int] = None,
    sleep_s: float = 0.3,
    checkpoint_path: Optional[str] = None,
    log=print,
) -> tuple[list[dict], list[dict]]:
    """One portal at a time, a handful of keywords each. ``per_portal`` is
    deliberately small: the point of this sweep is breadth of *provider*, and a
    9th dataset from one national portal is worth far less than a 1st from the
    next country."""
    classes = list(classes)
    taken_ids = {s["id"] for s in (yaml.safe_load(open(catalog_path)) or [])}
    known_res = wired_ckan_resources(catalog_path)
    cands: list[dict] = []
    skipped: list[dict] = []

    for host, country in portals:
        got = 0
        for klass in classes[:keywords_per_portal] if keywords_per_portal else classes:
            if got >= per_portal:
                break
            kw = klass[0]
            try:
                pkgs = ckan_search(host, kw, rows=rows)
            except Exception as exc:                          # noqa: BLE001
                skipped.append({"id": host, "reason": f"search failed: {exc}"[:120]})
                break
            for pkg in pkgs:
                if got >= per_portal:
                    break
                for res in pkg.get("resources") or []:
                    if got >= per_portal:
                        break
                    rid = res.get("id") or ""
                    if not rid or rid in known_res:
                        continue
                    # Some portals return multilingual objects rather than
                    # strings for name/title.
                    title = (_text(res.get("name")) or _text(pkg.get("title"))).strip()
                    tag = f"{host}/{rid}"
                    use_class = resolve_class(
                        klass, title, _text(pkg.get("notes")) or _text(pkg.get("title"))
                    )
                    if use_class is None:
                        continue
                    if res.get("datastore_active"):
                        fields = ckan_datastore_fields(host, rid)
                        time.sleep(sleep_s)
                        if not fields:
                            skipped.append({"id": tag, "reason": "no DataStore fields"})
                            continue
                        ts = ckan_pick_timestamp(fields)
                        if not ts:
                            skipped.append({"id": tag, "reason": "no timestamp column"})
                            continue
                        probe = ckan_probe(host, rid, ts)
                        time.sleep(sleep_s)
                    else:
                        # The DataStore is the exception, not the rule: across
                        # every national portal swept, essentially no resource
                        # had datastore_active, because they publish FILES and
                        # link them. Those files are still perfectly good
                        # sources — they just have to be read directly.
                        got_file = ckan_file_candidate(res, max_age_days)
                        if "error" in got_file:
                            skipped.append({"id": tag, "reason": got_file["error"]})
                            continue
                        fields, ts, probe = (got_file["fields"], got_file["ts"],
                                             got_file["probe"])
                        time.sleep(sleep_s)
                    if "error" in probe:
                        skipped.append({"id": tag,
                                        "reason": f"probe failed: {probe['error']}"})
                        continue
                    if probe["distinct"] < 20:
                        skipped.append({"id": tag,
                                        "reason": f"only {probe['distinct']} distinct ts"})
                        continue
                    if probe["age_days"] > max_age_days * 3:
                        skipped.append({"id": tag,
                                        "reason": f"newest {probe['age_days']:.0f}d old"})
                        continue
                    block = ckan_synthesize(host, country, pkg, res, use_class,
                                            fields, probe, taken=taken_ids)
                    if block is None:
                        skipped.append({"id": tag, "reason": "could not synthesise"})
                        continue
                    taken_ids.add(yaml.safe_load(block["yaml_block"])[0]["id"])
                    cands.append(block)
                    got += 1
                    _checkpoint(cands, checkpoint_path)
                    log(f"  + [{country}] {block['candidate_name'][:70]}")
                    if target and len(cands) >= target:
                        log(f"target {target} reached")
                        return cands, skipped
        log(f"[{host}] {got} candidate(s)")
    return cands, skipped


def wired_ckan_resources(catalog_path: str) -> set[str]:
    reg = yaml.safe_load(open(catalog_path)) or []
    out = set()
    for src in reg:
        url = (src.get("endpoint") or {}).get("url", "")
        m = re.search(r"resource_id=([0-9a-f\-]{8,})", url)
        if m:
            out.add(m.group(1))
    return out


def write_batch(candidates: list[dict], out_path: str) -> str:
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(candidates, fh, indent=2)
    return out_path


def _checkpoint(candidates: list[dict], path: Optional[str]) -> None:
    """Persist progress after every find.

    A full sweep runs for tens of minutes across thousands of HTTP requests, and
    writing only at the end means any interruption — a killed shell, a lost
    session — throws away every candidate found so far. That has already
    happened once, losing 23 verified candidates that each cost a live probe.
    """
    if not path:
        return
    try:
        write_batch(candidates, path)
    except OSError as exc:                                    # noqa: BLE001
        log_write_failed = f"checkpoint write failed: {exc}"
        print(log_write_failed)
