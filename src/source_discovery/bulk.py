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
    # --- third tier: narrower still. Yield per keyword keeps falling, but the
    # datasets sit on portals the broader terms never reach.
    ("water main breaks", "nature", "asset_faults"),
    ("wastewater treatment flow", "nature", "water_quality"),
    ("snow removal operations", "transport", "service_operations"),
    ("pothole repairs", "transport", "service_operations"),
    ("signal light outages", "transport", "asset_faults"),
    ("red light camera violations", "transport", "citation_stream"),
    ("airport operations delays", "transport", "service_reliability"),
    ("rail crossing incidents", "transport", "collision_stream"),
    ("emergency shelter occupancy", "healthcare", "capacity_utilisation"),
    ("homeless services encounters", "healthcare", "service_encounters"),
    ("lead testing results", "healthcare", "environmental_health"),
    ("mosquito surveillance", "healthcare", "vector_surveillance"),
    ("nursing home staffing", "healthcare", "staffing_levels"),
    ("prescription monitoring", "healthcare", "pharmacy_dispensing"),
    ("child care enrollment", "sales", "enrollment"),
    ("school attendance daily", "sales", "attendance"),
    ("parks facility reservations", "sales", "registration_stream"),
    ("marriage licenses issued", "sales", "registration_stream"),
    ("vehicle registrations", "sales", "registration_stream"),
    ("court case filings", "econ_fin", "case_filings"),
    ("property tax collections", "econ_fin", "tax_receipts"),
    ("payroll headcount", "econ_fin", "labour_market"),
    ("utility assistance applications", "econ_fin", "public_assistance"),
    ("electric meter readings", "energy", "building_load"),
    ("streetlight energy usage", "energy", "asset_consumption"),
    ("municipal solar generation", "energy", "generation"),
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
    query_class: Optional[tuple[str, str, str]], title: str, description: str = "",
    min_score: int = 1,
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
    the honest answer for a result the sweep simply should not have surfaced.

    `min_score` is how many of a class's tokens the title has to hit. One is
    right for a keyword sweep, where the query already established the topic and
    this call is only checking the result did not drift. It is far too loose
    without that anchor: a publisher-first walk hands this function raw titles,
    many of them not in English, so token overlap with an English keyword list is
    near zero and a single incidental hit wins by default. Measured on the ODS
    federation at min_score=1: French headcount statistics filed as transport,
    lottery draws as healthcare. Raise it and those become None, which is
    correct — they are datasets this build has no class for."""
    hay = " ".join(_topic_tokens(f"{title} {description}"))
    if not hay:
        return None
    best, best_score = None, min_score - 1
    # A publisher-first sweep has no query to inherit from, so query_class is
    # None and the dataset's own title has to carry the classification alone.
    head = (query_class,) if query_class is not None else ()
    for klass in (head + tuple(KEYWORD_CLASSES)
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
    # Country-code TLDs. Without these a European host walks in from the right
    # and stops on the TLD itself: erddap.emodnet-physics.eu -> "eu", which
    # names a continent rather than a publisher and collides with every other
    # .eu source.
    "eu", "uk", "ie", "fr", "de", "es", "it", "nl", "be", "pt", "at", "ch",
    "no", "se", "fi", "dk", "is", "pl", "cz", "gr", "ro", "hu", "si", "sk",
    "au", "nz", "jp", "kr", "in", "sg", "za", "br", "mx", "cl", "ar", "il",
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
def host_counts(catalog_path: str) -> dict[str, int]:
    """How many LIVE sources each host already carries.

    The per-host cap has to count what is already in the catalog, not just what
    this sweep found. Counting only within a run lets a host that already holds
    five sources collect several more, every one of them past the point where the
    coverage metric stops crediting them — effort spent on volume that reads as
    breadth and is not.
    """
    reg = yaml.safe_load(open(catalog_path)) or []
    counts: dict[str, int] = {}
    for src in reg:
        if src.get("disabled"):
            continue
        url = (src.get("endpoint") or {}).get("url", "")
        if url:
            h = urlparse(url).netloc.lower()
            counts[h] = counts.get(h, 0) + 1
    return counts


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
    seen_hosts: dict[str, int] = host_counts(catalog_path)
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
    # --- second pass: aimed at the cadence cells that stay thin (half-hour,
    # weekly, irregular) and at energy provider diversity, which is the
    # catalogue's worst-concentrated domain.
    ("places disponibles parking", "transport", "parking_occupancy"),
    ("temps d'attente", "web_cloudops", "service_latency"),
    ("perturbations trafic", "transport", "disruption_stream"),
    ("travaux voirie chantiers", "transport", "disruption_stream"),
    ("incidents réseau électrique", "energy", "asset_faults"),
    ("coupures électricité", "energy", "outage_stream"),
    ("production photovoltaïque", "energy", "generation"),
    ("production éolienne", "energy", "generation"),
    ("épisodes de pollution", "nature", "air_quality"),
    ("débit rivière hydrométrie", "nature", "hydrology"),
    ("température station météo", "nature", "meteorology"),
    ("consommation eau potable", "nature", "water_demand"),
    ("affluence piscine équipement", "sales", "attendance"),
    ("alertes sanitaires", "healthcare", "health_alerts"),
    ("passages urgences hebdomadaire", "healthcare", "health_utilisation"),
    ("ocupación aparcamiento", "transport", "parking_occupancy"),
    ("caudal río", "nature", "hydrology"),
    ("producción eólica", "energy", "generation"),
    ("incidencias tráfico", "transport", "disruption_stream"),
    ("Pegelstand Gewässer", "nature", "hydrology"),
    ("Parkhaus Belegung", "transport", "parking_occupancy"),
    ("Störungen Netz", "energy", "asset_faults"),
    ("Photovoltaik Erzeugung", "energy", "generation"),
    ("parkeergarage bezetting", "transport", "parking_occupancy"),
    ("waterstand meting", "nature", "hydrology"),
    ("qualidade do ar", "nature", "air_quality"),
    ("consumo de energia", "energy", "grid_demand"),
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
                   taken: Optional[set[str]] = None,
                   origin: Optional[str] = None) -> Optional[dict]:
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
    provenance = origin or f"query '{kw}'"

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
            f"Bulk-generated from the Opendatasoft federated catalog "
            f"({provenance}). Fields chosen from the portal's declared types; "
            f"the endpoint points at the publisher's own host, not the "
            f"aggregator."
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
    seen_hosts: dict[str, int] = host_counts(catalog_path)
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
# ODS, publisher-first
# --------------------------------------------------------------------------- #
# `ods_sweep` above picks a keyword and sees which publishers turn up. That is
# the right shape for finding DATASETS and the wrong shape for finding
# PUBLISHERS: popular topics are dominated by the same large portals, so you
# re-meet them keyword after keyword. Measured, on this federation: a breadth
# run examined 1,680 datasets and produced ONE candidate, with 372 skips reading
# "host already wired".
#
# Inverting it -- enumerate publishers, then ask each one for its best series --
# turns host count into the thing being maximised rather than a by-product.
#
# The enumeration is the awkward part. /catalog/facets caps a facet at 100
# values and this endpoint rejects the `facet=name(limit:N)` syntax, so the list
# has to be partitioned. Excluding everything already seen works but grows the
# `where` clause without bound; partitioning on a name prefix keeps every URL
# short, and recursing only into prefixes that come back full means the walk
# costs about one request per 100 publishers found.
ODS_FACETS = "https://data.opendatasoft.com/api/explore/v2.1/catalog/facets"
ODS_FACET_CAP = 100                  # /catalog/facets caps a facet at 100 values
ODS_PREFIX_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-"
# 4 truncates on 'data', which is the most common prefix in the federation
# by a wide margin -- the depth limit is a backstop, the request budget in
# ods_publishers is what actually bounds the walk.
ODS_MAX_PREFIX_DEPTH = 8

# A keyword sweep never surfaces these because no keyword class describes them.
# Walking a portal end-to-end does. Lottery draws are the sharpest case: they
# have a clean timestamp, a dense numeric column and daily cadence, so every
# structural check passes -- and they are i.i.d. uniform by construction. A
# forecasting benchmark that contains them is measuring nothing and paying for
# the privilege. The rest are portals' own test fixtures.
_ODS_REJECT_TITLE = re.compile(
    # Games of chance. "lottery" alone would also take out National Lottery
    # FUND grant payments, which are a real spending series, so the draw words
    # need draw context; loto/keno/euromillions carry it on their own.
    # (euromillion\b would never match "EuroMillions" -- same trap as
    # climatolog\b not matching "climatology".)
    r"\b(loto|lotto|keno|euromillions?|powerball|jackpots?|raffles?|sorteos?)\b"
    r"|\b(lottery|tirage|draw)\b[^,;]{0,24}"
    r"\b(result|resultat|résultat|number|numero|winning|draw|gagnant)"
    r"|\b(winning|gagnant)\b[^,;]{0,16}\b(number|numero|combinaison)"
    # Portals' own test fixtures.
    r"|\b(dummy|sample[_ ]data|test[_ ]dataset|random[_ ]walk|"
    r"dedicated[_ -]integers|all[_ ]geometries)\b",
    re.I,
)

# Without a query class to anchor on, one token of overlap is noise.
ODS_PUBLISHER_MIN_CLASS_SCORE = 2



# Publisher-first has no query to anchor a domain to, and the title cannot carry
# it alone: `resolve_class` scores an English/French municipal keyword list
# against raw titles and lets one token win, which on this federation filed
# French headcount statistics as transport, lottery draws as healthcare, and
# "Hourly electricity consumption by feeder" as nature (it matched the class
# "water consumption" on the word consumption). Raising the threshold fixes the
# wrong answers and loses the right ones.
#
# So classification here is CLOSED, the same shape that fixed ERDDAP, keyed on
# a field the publisher filled in rather than on prose: ODS `theme`. The whole
# federation uses 116 distinct values, enumerated by prefix walk, and this map
# covers every one that plausibly carries a time series. Themes absent from the
# map -- Orthoimagery, Elevation, Land cover, Archaeology, Politics, Census --
# are skipped rather than guessed. Skipping is the cheap error here: an
# unclassified dataset costs one candidate, a misclassified one distorts the
# domain x cadence matrix that steers what the build looks for next.
ODS_THEME_DOMAIN: dict[str, str] = {
    # nature
    "environment": "nature",
    "environnement": "nature",
    "environmental monitoring facilities": "nature",
    "meteorological geographical features": "nature",
    "hydrography": "nature",
    "hydrographie": "nature",
    "climate & sustainability": "nature",
    "environment, climate": "nature",
    "territory and environment": "nature",
    "farming, environment": "nature",
    "water & agriculture": "nature",
    "sea regions": "nature",
    "natural risk zones": "nature",
    # energy
    "energy": "energy",
    "electricity": "energy",
    "conventional fuels": "energy",
    "environment & energy": "energy",
    # transport
    "transport": "transport",
    "transportation": "transport",
    "trasporti": "transport",
    "transports, déplacements": "transport",
    "transport, movements": "transport",
    "transport & mobility": "transport",
    "mobility": "transport",
    "mobility and transport": "transport",
    "mobilité": "transport",
    "mobiliteit": "transport",
    "traffic": "transport",
    "routes / transports": "transport",
    "07-territories, transport & services": "transport",
    # healthcare
    "health": "healthcare",
    "santé": "healthcare",
    "santé / social": "healthcare",
    "chronic deases": "healthcare",          # the federation's own typo
    "services, social": "healthcare",
    # econ_fin
    "economy": "econ_fin",
    "economie": "econ_fin",
    "economy and finance": "econ_fin",
    "finance": "econ_fin",
    "employment": "econ_fin",
    "industry": "econ_fin",
    "production": "econ_fin",
    "03-economy, employment": "econ_fin",
    # sales
    "housing & properties": "sales",
    "housing": "sales",
    "property and planning": "sales",
    "tourism": "sales",
    "tourism, culture, sport and leisure": "sales",
    # web_cloudops
    "digital and telecommunications": "web_cloudops",
}


# Exact values will not do the job: portals define their own themes, so the
# federation has far more than the ~116 the global facet shows, in at least six
# languages ("Solidarité et soutien à l'activité", "Itinerarios y horarios",
# "Mobiliteit"). Worse, /catalog/facets cannot enumerate them cleanly --
# startswith is case-sensitive AND returns the CO-OCCURRING themes of matching
# datasets, so a prefix walk returns neither a filtered nor a complete list.
#
# So the closed vocabulary is a pattern list rather than a value list. It stays
# closed -- a theme matching nothing here is skipped, never guessed -- but it
# generalises across a portal's own naming. Substring matching is safe on this
# field in a way it was not on titles: "transport" appearing in a curator's
# theme label means the theme is about transport, whereas in prose it can be
# one incidental word in a sentence about something else.
ODS_THEME_PATTERNS: tuple[tuple[str, str], ...] = (
    # nature
    (r"environn?ement|environmental|ambiente|umwelt|milieu", "nature"),
    (r"hydro|water|\beau\b|\beaux\b|acqua|agua|rivi[eè]re|nappe", "nature"),
    (r"m[ée]t[ée]o|weather|climat|clima\b|atmosph", "nature"),
    (r"qualit[ée] de l.air|air quality|luchtkwaliteit|pollution", "nature"),
    (r"biodiv|habitat|esp[eè]ce|species|faune|flore|for[eê]t|forest", "nature"),
    (r"agricultur|farming|agricol|agraria|p[eê]che|fisher", "nature"),
    (r"risque naturel|natural risk|s[ée]isme|seismic|inondation|flood", "nature"),
    (r"d[ée]chet|waste|recycl", "nature"),
    (r"\bmer\b|marine|maritim|oc[ée]an|ocean|coastal|littoral", "nature"),
    # energy
    (r"[ée]nerg|energy|energia|energie", "energy"),
    (r"[ée]lectric|electric|elettric|el[ée]ctric|strom", "energy"),
    (r"chauffage|heating|fuel|carburant|petrol|p[ée]trole", "energy"),
    # transport
    (r"transport|trasport|verkehr|vervoer", "transport"),
    (r"mobilit|d[ée]placement|movement", "transport"),
    (r"traffic|trafic|traffico|circulation routi", "transport"),
    (r"stationnement|parking|aparcamiento", "transport"),
    (r"v[ée]lo|bicycl|\bbike\b|cycl(ing|isme)|fiets", "transport"),
    (r"\bbus\b|tramway|\btram\b|m[ée]tro\b|rail|train|horaires?\b", "transport"),
    (r"a[ée]roport|airport|\bport\b|navigation", "transport"),
    (r"itinerario|horario|fahrplan|timetable|dessertes?\b", "transport"),
    # healthcare
    (r"sant[ée]|health|salute|salud|gesundheit|sanitar", "healthcare"),
    (r"solidarit|social|handicap|autonomie|m[ée]dico|medical", "healthcare"),
    (r"h[oô]pital|hospital|urgence|emergency care", "healthcare"),
    (r"petite enfance|childcare|cr[eè]che|enfance et jeunesse", "healthcare"),
    # econ_fin
    (r"[ée]conom|economy|wirtschaft", "econ_fin"),
    (r"financ|budget|fiscal|imp[oô]t|\btax\b|comptes? public", "econ_fin"),
    (r"emploi|employment|travail|labour|labor market|arbeit|ch[oô]mage", "econ_fin"),
    (r"industr|manufactur|production", "econ_fin"),
    (r"entreprise|business|commerce|trade|march[ée] public", "econ_fin"),
    # sales
    (r"logement|housing|immobil|real estate|propert|wohnen|vivienda", "sales"),
    (r"tourism|tourisme|turismo|tourismus|h[oô]tell", "sales"),
    (r"consommation des m[ée]nages|retail|vente", "sales"),
    # web_cloudops
    (r"num[ée]rique|digital|t[ée]l[ée]com|telecom|internet|broadband|"
     r"informatique", "web_cloudops"),
)

_ODS_THEME_RE: tuple[tuple[Any, str], ...] = tuple(
    (re.compile(pat, re.I), dom) for pat, dom in ODS_THEME_PATTERNS)

# A keyword sweep only ever sees datasets a topic query matched. A publisher
# walk sees the WHOLE portal, and an open-data portal is mostly registries:
# address bases, defibrillator locations, charge points, cemetery records. Those
# pass every structural check, because a registry built incrementally has a
# per-row "created"/"last updated" column that looks exactly like an observation
# clock and numeric columns that are coordinates and identifiers. Half of the
# first publisher-first batch was registries; all of them carried one of these.
_ODS_RECORD_TS = re.compile(
    # "maj" also arrives glued: datemajobs, majdate. Anchored to date/obs words
    # so it cannot fire on major/majoration.
    r"(^|_)maj(_|$)|date_?maj|maj_?(date|obs)|mise[_ ]?a[_ ]?jour|"
    r"mise[_ ]en[_ ]service|"
    r"creat|instal|publi|demande|saisie|enregistr|validation|"
    r"import|integration|depot|inscription|derniere",
    re.I,
)
# Numeric columns that identify or locate a row rather than measure anything.
# Checked as a substring, unlike _ODS_VAL_REJECT's token match, because these
# arrive glued together (uiccountrycode, organisationnumber, c_xy_precis).
_ODS_ID_VAL = re.compile(
    r"^[xy]$|(^|_)xy(_|$)|^g?id$|coord|geo|uic|insee|siret|siren|"
    # Dublin Core metadata fields describe the RECORD, never a measurement.
    r"^dc_|commune|departement|arrondissement|^wmo$|"
    r"(id|code|numero|number|num)$|^(id|code|no)_|"
    # "numbershort" is an identifier, "number_of_passengers" is a measurement.
    r"^(number|numero|nombre)(?!s?[_ ]?(of|de|d))",
    re.I,
)

# The theme is the publisher's own filing, and publishers file by DEPARTMENT:
# a health agency tags its air-quality monitors "Santé". These few phrases name
# the measured process unambiguously enough to outrank that. Kept deliberately
# short -- every entry here is a hole in the closed vocabulary.
ODS_TITLE_OVERRIDE: tuple[tuple[str, str, str], ...] = (
    (r"qualit[ée] de l.air|air quality|luchtkwaliteit|"
     r"(^|\W)(pm10|pm2[.,]?5|no2|ozone)(\W|$)", "nature", "air_quality"),
    (r"prix|tarif|price|precio|preis", "econ_fin", "price_level"),
    # Same process, two languages, and it was landing in two domains: the
    # English EPC in nature, the French DPE in sales.
    (r"energy performance certificate|\bepc\b|"
     r"diagnostics? de performances? [ée]nerg[ée]tiques?|\bdpe\b",
     "energy", "building_energy_rating"),
)
_ODS_TITLE_OVERRIDE_RE = tuple(
    (re.compile(p, re.I), d, c) for p, d, c in ODS_TITLE_OVERRIDE)


# Used when the theme fixes the domain but no keyword class in that domain
# matches the title well enough to name the process more precisely.
ODS_DOMAIN_DEFAULT_CLASS: dict[str, str] = {
    "nature": "environmental_measurement",
    "energy": "energy_consumption",
    "transport": "transport_activity",
    "healthcare": "public_health_indicator",
    "econ_fin": "economic_indicator",
    "sales": "property_market_activity",
    "web_cloudops": "digital_service_usage",
}


def _best_class(pool: Iterable[tuple[str, str, str]], hay: str,
                min_score: int) -> Optional[tuple[str, str, str]]:
    best, best_score = None, min_score - 1
    for klass in pool:
        toks = _topic_tokens(klass[0])
        score = sum(1 for t in toks if t in hay or t.rstrip("s") in hay)
        if score > best_score:
            best, best_score = klass, score
    return best


def ods_publisher_class(themes: Any, title: str,
                        description: str = "") -> Optional[tuple[str, str, str]]:
    """Domain from the publisher's own `theme`, dgp_class from the title.

    The title is still allowed to name the process, but only from classes that
    already live in the theme's domain — so a bad title match costs precision
    inside a domain instead of putting the source in the wrong one.
    """
    for rx, dom, dgp in _ODS_TITLE_OVERRIDE_RE:
        if rx.search(title):
            return (dgp, dom, dgp)
    if isinstance(themes, str):
        themes = [themes]
    doms = set()
    for t in (themes or []):
        if not isinstance(t, str):
            continue
        exact = ODS_THEME_DOMAIN.get(t.strip().lower())
        if exact:
            doms.add(exact)
            continue
        doms.update(dom for rx, dom in _ODS_THEME_RE if rx.search(t))
    if len(doms) != 1:
        return None            # unmapped, or two themes that disagree
    dom = doms.pop()
    pool = [k for k in (tuple(KEYWORD_CLASSES) + tuple(ODS_KEYWORD_CLASSES)
                        + EXTRA_CLASSES) if k[1] == dom]
    hay = " ".join(_topic_tokens(f"{title} {description}"))
    hit = _best_class(pool, hay, ODS_PUBLISHER_MIN_CLASS_SCORE) if hay else None
    return hit or (dom, dom, ODS_DOMAIN_DEFAULT_CLASS[dom])


def _ods_facet_page(prefix: str, timeout: int) -> Optional[list[str]]:
    """Publisher names under `prefix`, or None if the call failed."""
    params = {"facet": "source_domain_address"}
    if prefix:
        params["where"] = f'startswith(source_domain_address, "{prefix}")'
    try:
        r = requests.get(ODS_FACETS, params=params,
                         headers={"User-Agent": UA}, timeout=timeout)
        if r.status_code != 200:
            return None
        facets = (r.json().get("facets") or [{}])[0].get("facets") or []
    except Exception:                                         # noqa: BLE001
        return None
    return [f["name"] for f in facets if f.get("name")]


def ods_publishers(timeout: int = 60, sleep_s: float = 0.25,
                   max_requests: int = 1500, log=print) -> list[str]:
    """Every publisher hostname in the ODS federation, by bounded prefix walk.

    Breadth-first with a hard request budget rather than plain recursion. The
    fan-out is 37 per level, so a run where most prefixes come back full would
    cost 37^depth requests — a budget makes the worst case a truncated answer
    the caller is told about, instead of a sweep that never returns.
    """
    found: set[str] = set()
    queue: list[str] = [""]
    spent, truncated = 0, []
    while queue and spent < max_requests:
        prefix = queue.pop(0)
        names = _ods_facet_page(prefix, timeout)
        spent += 1
        time.sleep(sleep_s)
        if names is None:
            continue
        found.update(names)
        # A full page means the facet was truncated and there is more behind it.
        if len(names) < ODS_FACET_CAP:
            continue
        if len(prefix) >= ODS_MAX_PREFIX_DEPTH:
            truncated.append(prefix or "(root)")
            continue
        queue.extend(prefix + ch for ch in ODS_PREFIX_ALPHABET)
    if queue:
        log(f"  [ods-pub] request budget {max_requests} spent with "
            f"{len(queue)} prefixes unexplored")
    if truncated:
        log(f"  [ods-pub] still full at max depth, publishers below these are "
            f"unreachable: {', '.join(truncated[:8])}")
    return sorted(found)


def ods_publisher_datasets(host: str, want: int = 25, timeout: int = 45) -> list[dict]:
    """That publisher's most recently processed datasets, newest first."""
    r = requests.get(ODS_CATALOG, params={
        "where": f'source_domain_address="{host}"',
        "order_by": "data_processed desc",
        "limit": min(ODS_PAGE, want),
    }, headers={"User-Agent": UA}, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return r.json().get("results", [])


def ods_publisher_sweep(
    catalog_path: str,
    host_cap: int = 2,
    scan_per_host: int = 25,
    max_age_days: float = 21.0,
    publishers: Optional[Iterable[str]] = None,
    target: Optional[int] = None,
    sleep_s: float = 0.3,
    checkpoint_path: Optional[str] = None,
    log=print,
) -> tuple[list[dict], list[dict]]:
    known_hosts = wired_hosts(catalog_path)
    seen_hosts = host_counts(catalog_path)
    known_ds = wired_ods_datasets(catalog_path)
    taken_ids = {s["id"] for s in (yaml.safe_load(open(catalog_path)) or [])}
    cands: list[dict] = []
    skipped: list[dict] = []

    if publishers is None:
        log("[ods-pub] enumerating the federation...")
        publishers = ods_publishers(log=log)
    publishers = [h for h in publishers if h]
    log(f"[ods-pub] {len(publishers)} publishers; "
        f"{sum(1 for h in publishers if h in known_hosts)} already wired")

    for host in publishers:
        if host in known_hosts or seen_hosts.get(host, 0) >= host_cap:
            skipped.append({"id": host, "reason": "host already wired"})
            continue
        try:
            records = ods_publisher_datasets(host, scan_per_host)
        except Exception as exc:                              # noqa: BLE001
            skipped.append({"id": host, "reason": f"listing failed: {exc}"})
            continue
        time.sleep(sleep_s)
        got = 0
        for rec in records:
            if got >= host_cap:
                break
            metas = (rec.get("metas") or {}).get("default") or {}
            dsid = metas.get("source_dataset") or rec.get("dataset_id") or ""
            title = (metas.get("title") or "").strip()
            tag = f"{host}/{dsid}"
            if not dsid or dsid in known_ds or not rec.get("has_records"):
                continue
            if (metas.get("records_count") or 0) < 40:
                continue
            mod = _parse_iso(metas.get("data_processed") or metas.get("modified"))
            if mod is None:
                continue
            age = (dt.datetime.now(dt.timezone.utc) - mod).total_seconds() / 86400
            if age > max_age_days:
                # Sorted newest-first, so the rest of this publisher is older.
                skipped.append({"id": host,
                                "reason": f"nothing processed inside {max_age_days:.0f}d "
                                          f"(newest {age:.0f}d)"})
                break
            fields = _ods_fields(rec)
            ts = ods_pick_timestamp(fields)
            if not ts:
                continue
            if _ODS_RECORD_TS.search(ts):
                skipped.append({"id": tag,
                                "reason": f"registry: '{ts}' is a record-keeping "
                                          f"date, not an observation clock"})
                continue
            vals = ods_pick_values(fields, ts)
            vals = [v for v in vals if not _ODS_ID_VAL.search(v)]
            if not vals:
                skipped.append({"id": tag,
                                "reason": "no numeric field that measures anything"})
                continue
            # Publisher-first means no keyword to inherit a class from, so the
            # title has to carry it alone; an unclassifiable dataset is skipped
            # rather than guessed into a domain it would then distort.
            if _ODS_REJECT_TITLE.search(f"{title} {dsid}"):
                skipped.append({"id": tag,
                                "reason": f"not a forecastable process: {title[:50]}"})
                continue
            use_class = ods_publisher_class(
                metas.get("theme"), title, metas.get("description") or "")
            if use_class is None:
                skipped.append({
                    "id": tag,
                    "reason": f"theme not in the map: {metas.get('theme')}"})
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
            block = ods_synthesize(rec, use_class, probe, taken=taken_ids,
                                   origin="publisher walk, theme-classified")
            if block is None:
                skipped.append({"id": tag, "reason": "could not synthesise entry"})
                continue
            taken_ids.add(yaml.safe_load(block["yaml_block"])[0]["id"])
            known_ds.add(dsid)
            cands.append(block)
            got += 1
            seen_hosts[host] = seen_hosts.get(host, 0) + 1
            _checkpoint(cands, checkpoint_path)
            log(f"  + [{use_class[1]}] {block['candidate_name'][:66]}")
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


# Resource "names" that describe the FILE rather than the data. Naming an entry
# after one of these produces ids like `ie_csv`, which say nothing about the
# series and collide across every package on the portal.
_GENERIC_RESOURCE_NAMES = frozenset({
    "csv", "data", "dataset", "download", "file", "link", "resource", "export",
    "csv file", "data file", "download csv", "csv download", "open data",
    "données", "fichier", "datos", "archivo", "bestand", "daten",
})


def _resource_title(res: dict, pkg: dict) -> str:
    name = _text(res.get("name")).strip()
    pkg_title = _text(pkg.get("title")).strip()
    if not name or name.lower() in _GENERIC_RESOURCE_NAMES:
        return pkg_title or name
    if pkg_title and name.lower() not in pkg_title.lower():
        return f"{pkg_title} — {name}"
    return pkg_title or name


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


def _content_length(url: str, timeout: int = 20) -> Optional[int]:
    try:
        r = requests.head(url, headers={"User-Agent": UA}, timeout=timeout,
                          allow_redirects=True)
        return int(r.headers.get("Content-Length") or 0) or None
    except Exception:                                         # noqa: BLE001
        return None


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
    if not size:
        # `size` is frequently absent, and the scraper caps a response at 48MB —
        # so without this the file is downloaded twice before anyone learns it
        # is too big: once here and once at wire time.
        declared = _content_length(url)
        if declared and declared > 48_000_000:
            return {"error": f"Content-Length {declared} exceeds the 48MB cap"}
    stamp = _parse_iso(res.get("last_modified") or res.get("created")
                       or res.get("metadata_modified"))
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
        d for d in (_parse_loose_date(r[header.index(ts_col)])
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


_LOOSE_DATE_RES = (
    # DD/MM/YYYY and DD-MM-YYYY dominate European portals; a sniffer that only
    # accepts ISO finds no timestamp column on most of them and the whole
    # portal reads as unusable.
    (re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})"), ("d", "m", "y")),
    (re.compile(r"^(\d{4})(\d{2})(\d{2})$"), ("y", "m", "d")),
)


def _parse_loose_date(v: Any) -> Optional[dt.datetime]:
    """ISO first, then the common national orderings. Day-first is assumed for
    ambiguous DD/MM vs MM/DD — these are European portals, and guessing wrong
    only shifts a date within a month, never the cadence the column is chosen
    for."""
    got = _parse_iso(v)
    if got is not None:
        return got
    s = str(v or "").strip()
    for rx, order in _LOOSE_DATE_RES:
        m = rx.match(s)
        if not m:
            continue
        parts = dict(zip(order, m.groups()))
        try:
            return dt.datetime(int(parts["y"]), int(parts["m"]), int(parts["d"]),
                               tzinfo=dt.timezone.utc)
        except ValueError:
            return None
    return None


def _csv_timestamp_column(header: list[str], body: list[list[str]]) -> Optional[str]:
    """The column whose values actually parse as dates — declared types do not
    exist for a raw file, so the data has to answer for itself."""
    best, best_hits = None, 0
    for i, name in enumerate(header):
        vals = [r[i] for r in body[:40] if len(r) > i and r[i].strip()]
        if not vals:
            continue
        hits = sum(1 for v in vals if _parse_loose_date(v) is not None)
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
    title = _resource_title(res, pkg)
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
                    # One resource per package. A package's other resources are
                    # nearly always the SAME series in another language or
                    # format — opendata.swiss offered Swiss weekly deaths three
                    # times over — and wiring each would triple-count one series
                    # as three sources while adding no information.
                    break
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



# --------------------------------------------------------------------------- #
# ERDDAP
# --------------------------------------------------------------------------- #
# ERDDAP is the third federated shape, and structurally the best one yet: ~60
# INDEPENDENTLY OPERATED servers (each a distinct host, which is the axis the
# catalog is short on) that all speak an identical REST dialect. Every server
# exposes its own catalog as a queryable table -- `tabledap/allDatasets` --
# carrying datasetID, title, institution and, decisively, `maxTime`: the newest
# observation the dataset actually holds. No other platform hands over the
# freshness of the DATA rather than of the METADATA, which was the single
# biggest source of wasted probes on Socrata/ODS/CKAN.
#
# Two ERDDAP-specific traps, both of which would poison the benchmark rather
# than merely waste a request:
#
# * Many servers publish MODEL OUTPUT next to observations -- forecasts,
#   hindcasts, climatologies, reanalyses. A forecast grid is not an observed
#   series; predicting a prediction measures nothing. They also carry maxTime in
#   the FUTURE, so they read as permanently fresh (`_ERDDAP_REJECT_TITLE`, plus
#   a hard future-maxTime cut).
# * Half the variables on a typical mooring are QC flags, coordinates and
#   instrument ids, all numeric. Wiring those yields a "series" that is a
#   constant 1 (`_ERDDAP_DROP_VAR`).
#
# The URL needs no date token: ERDDAP understands server-relative constraints
# (`time>=now-3days`), so a generated entry stays fresh forever with no
# templating.
ERDDAP_SERVERS: tuple[tuple[str, str], ...] = (
    ("https://erddap.observations.voiceoftheocean.org/erddap", "VOTO"),
    ("https://erddap.ogsl.ca/erddap", "SLGO-OGSL"),
    ("https://coastwatch.pfeg.noaa.gov/erddap", "CSWC"),
    ("https://apdrc.soest.hawaii.edu/erddap", "APDRC"),
    ("https://www.ncei.noaa.gov/erddap", "NCEI"),
    ("https://erddap.bco-dmo.org/erddap", "BCODMO"),
    ("https://erddap.emodnet.eu/erddap", "EMODnet"),
    ("https://erddap.emodnet-physics.eu/erddap", "EMODnet Physics"),
    ("https://erddap.marine.ie/erddap", "MII"),
    ("https://cwcgom.aoml.noaa.gov/erddap", "CSCGOM"),
    ("https://erddap.sensors.ioos.us/erddap", "IOOS-Sensors"),
    ("http://erddap.cencoos.org/erddap", "CeNCOOS ERDDAP"),
    ("https://data.neracoos.org/erddap", "NERACOOS"),
    ("https://gliders.ioos.us/erddap", "NGDAC"),
    ("https://pae-paha.pacioos.hawaii.edu/erddap", "PacIOOS"),
    ("https://sccoos.org/erddap", "SCCOOS"),
    ("http://erddap.secoora.org/erddap", "SECOORA"),
    ("http://osmc.noaa.gov/erddap", "OSMC"),
    ("http://dap.onc.uvic.ca/erddap", "ONC"),
    ("http://erddap.oceantrack.org/erddap", "OTN"),
    ("https://oceanwatch.pifsc.noaa.gov/erddap", "PIFSC"),
    ("https://erddap.dataexplorer.oceanobservatories.org/erddap", "OOI"),
    ("https://erddap-goldcopy.dataexplorer.oceanobservatories.org/erddap", "OOI Goldcopy"),
    ("http://www.myroms.org:8080/erddap", "MYROMS"),
    ("http://tds.marine.rutgers.edu/erddap", "RUTGERS"),
    ("https://comet.nefsc.noaa.gov/erddap", "NEFSC"),
    ("https://opendap.co-ops.nos.noaa.gov/erddap", "COOPS-NOS"),
    ("https://gcoos5.geos.tamu.edu/erddap", "GCOO5-TAMU"),
    ("https://gcoos4.tamu.edu/erddap", "GCOO4-TAMU"),
    ("https://apps.glerl.noaa.gov/erddap", "GLERL"),
    ("https://spraydata.ucsd.edu/erddap", "UCSD"),
    ("https://salishsea.eos.ubc.ca/erddap", "UBC"),
    ("http://bmlsc.ucdavis.edu:8080/erddap", "BMLSC"),
    ("https://upwell.pfeg.noaa.gov/erddap", "UAF"),
    ("https://www.ifremer.fr/erddap", "IFREMER"),
    ("https://data.pmel.noaa.gov/pmel/erddap", "PMEL"),
    ("https://ferret.pmel.noaa.gov/alamo/erddap", "ALAMO"),
    ("https://ferret.pmel.noaa.gov/socat/erddap", "SOCAT"),
    ("https://catalogue.hakai.org/erddap", "Hakai"),
    ("https://polarwatch.noaa.gov/erddap", "POLARWATCH"),
    ("https://geoport.usgs.esipfed.org/erddap", "USGS"),
    ("https://erddap.incois.gov.in/erddap", "INCOIS"),
    ("https://www.smartatlantic.ca/erddap", "SmartAtlantic"),
    ("https://erddap.griidc.org/erddap", "GRIIDC"),
    ("https://atn.ioos.us/erddap", "ATN-IOOS"),
    ("https://pub-data.diver.orr.noaa.gov/erddap", "DIVER"),
    ("https://erddap.gcoos.org/erddap", "GCOOS"),
    ("https://basin.ceoe.udel.edu/erddap", "CEOE"),
    ("https://cioosatlantic.ca/erddap", "CIOOS Atlantic"),
    ("https://data.cioospacific.ca/erddap", "CIOOS Pacific"),
    ("http://erddap.emso.eu/erddap", "EMSO ERIC"),
    ("https://coastwatch.noaa.gov/erddap", "NCCO"),
    ("https://canwinerddap.ad.umanitoba.ca/erddap", "CanWIN"),
    ("https://oceanview.pfeg.noaa.gov/erddap", "NOAA Oceanview"),
    ("https://erddap.oa.iode.org/erddap", "IOC-IODE-OA"),
    ("https://linkedsystems.uk/erddap", "NOC-BODC Linked Systems"),
    ("https://erddap.bio-oracle.org/erddap", "Bio-Oracle"),
    ("https://erddap.ondeckdata.com/erddap", "CFRF"),
)

# Variables that are numeric but are not observations: QC flags, coordinates,
# instrument/deployment bookkeeping. On a typical mooring these OUTNUMBER the
# real measurements, and a QC flag column is a constant, which is worse than no
# series at all.
_ERDDAP_DROP_VAR = re.compile(
    r"(?:^|_)(?:qc|qa|qartod|flag|flags|test|tests|agg|mask|status|"
    r"instrument\d*|deployment|profile|trajectory|rowsize|row_size)(?:$|_)"
    r"|^(?:latitude|longitude|lat|lon|depth|altitude|z|station|station_id|"
    r"platform|crs|id|time_modified|actual_time|time_created|precise_time|"
    r"sample_time|obs_time|feature_type_instance)$",
    re.I,
)
_ERDDAP_NUM_TYPES = frozenset({"double", "float", "int", "long", "short"})

# Titles that mean "this is a model, not a measurement". A forecast dataset is
# both scientifically wrong for the benchmark (forecasting a forecast measures
# nothing) and operationally poisonous: its newest timestamp is in the FUTURE,
# so the freshness audit can never see it die.
_ERDDAP_REJECT_TITLE = re.compile(
    r"\b(forecast|hindcast|nowcast|prediction|predicted|model|modell?ed|"
    r"simulation|reanalys[ie]s|climatolog[a-z]*|projection|scenario|"
    r"particle track|trajectory analysis)\b"
    # ERDDAP's own placeholder title, left in place by many installs. It names
    # nothing, so the entry it would generate is unidentifiable in the catalog.
    r"|^data from a (local|remote) source",
    re.I,
)

# A glider/mooring DEPLOYMENT rather than a station: the id carries the launch
# timestamp ("AsterSEA068-20260804T0908"). These are fresh today and dead in
# three weeks when the instrument is recovered -- a source that is guaranteed to
# go stale is worse than no source, because it consumes an audit slot forever.
_ERDDAP_DEPLOYMENT = re.compile(r"\d{8}T\d{3,6}|[-_]\d{8}$")

# Probe windows, widest-first fallback. A 10-minute mooring fills 20 distinct
# timestamps in hours; a monthly cruise series needs years. Starting narrow and
# widening only on failure keeps the common case to a few KB.
_ERDDAP_WINDOWS: tuple[int, ...] = (2, 14, 120, 1200)

ERDDAP_DEFAULT_CLASS = ("ocean marine buoy mooring water observations",
                        "nature", "environmental_sensor")

# ERDDAP classification is deliberately CLOSED, unlike the catalog sweeps'.
# `resolve_class` scores a title against every municipal keyword class there is,
# which is right for a general-purpose portal and actively harmful here: an
# ocean federation has exactly one domain, so every cross-domain match it finds
# is a false one. Left open, it filed "NOAA Ship ... Near Real Time" under sales
# (the token "real" hitting a real-estate class) and "RADS Altimeter" under
# econ_fin -- 12 of 50 candidates mislabelled, in the very matrix that steers
# what the build goes looking for next. So the domain is fixed and only the
# dgp_class is inferred.
ERDDAP_CLASSES: tuple[tuple[str, str, str], ...] = (
    ("wave height swell period directional waves", "nature", "ocean_waves"),
    ("water temperature salinity conductivity density currents", "nature",
     "ocean_physics"),
    ("tide sea level water level surge datum", "nature", "sea_level"),
    ("meteorological air temperature wind barometric pressure humidity",
     "nature", "met_station"),
    ("river discharge streamflow stage hydrology runoff", "nature", "hydrology"),
    ("chlorophyll oxygen nutrient nitrate ph carbon dioxide pco2", "nature",
     "biogeochemistry"),
    ("radiation irradiance solar par light", "nature", "radiation"),
    ("acoustic detection fish tag telemetry biodiversity", "nature",
     "biodiversity"),
    ("glider profiler mooring underway ship transect", "nature",
     "ocean_physics"),
)


def erddap_class(title: str, institution: str = "") -> tuple[str, str, str]:
    """Pick a dgp_class from the ocean/met vocabulary; domain is always nature."""
    hay = " ".join(_topic_tokens(f"{title} {institution}"))
    best, best_score = ERDDAP_DEFAULT_CLASS, 0
    for klass in ERDDAP_CLASSES:
        score = sum(1 for t in _topic_tokens(klass[0])
                    if t in hay or t.rstrip("s") in hay)
        if score > best_score:
            best, best_score = klass, score
    return best


def erddap_platform_key(title: str, dsid: str) -> str:
    """What PHYSICAL platform a dataset comes from, for within-server spread.

    A server's fresh list is sorted by platform, so taking the first N datasets
    takes N instruments on ONE buoy -- `A01_met`, `A01_waves`, `A01_ocean_001m`
    are three depths of the same mooring in the same water, which is volume
    without information. Station codes lead the title and are the only reliably
    machine-findable part of them, so a leading token carrying a digit ("A01",
    "(41024", "W60-G500") keys the platform; titles with no such token are left
    alone, because there the distinguishing part is prose and over-eager
    deduping would throw away genuinely separate sites (each EMODnet HF radar
    station is titled "Near Real Time Surface Ocean ... by HFR-<site>").
    """
    first = (title or "").strip().split(" ")[0] if title else ""
    if first and any(ch.isdigit() for ch in first):
        return re.sub(r"[^a-z0-9]", "", first.lower())
    return _slug(title or dsid, 60)


# A tabledap query has no LIMIT clause: the window IS the limit. When a
# dataset's declared maxTime lies, or the probe widens to its 1200-day fallback
# against a high-rate instrument, the honest answer to the query is hundreds of
# megabytes -- and parsing that into Python dicts costs an order of magnitude
# more again. One such response took this 4GB box to 2.6GB RSS and into swap,
# which slows the production scrape cron sharing the machine. A probe that needs
# more than a few MB is a probe whose answer is "too big to wire" anyway.
ERDDAP_PROBE_BYTES = 8_000_000
ERDDAP_CATALOG_BYTES = 48_000_000        # allDatasets on a 10k-dataset server


def erddap_get(url: str, timeout: int = 60,
               max_bytes: int = ERDDAP_PROBE_BYTES) -> dict:
    with requests.get(url, headers={"User-Agent": UA}, timeout=timeout,
                      stream=True) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        buf = bytearray()
        for chunk in r.iter_content(65536):
            buf += chunk
            if len(buf) > max_bytes:
                raise RuntimeError(f"response exceeds {max_bytes // 1_000_000}MB")
        return json.loads(bytes(buf).decode("utf-8", "replace"))


def _erddap_rows(payload: dict) -> list[dict]:
    """ERDDAP's uniform response shape: {"table": {columnNames, rows}}."""
    table = (payload or {}).get("table") or {}
    names = table.get("columnNames") or []
    return [dict(zip(names, row)) for row in table.get("rows") or []]


def erddap_datasets(
    base: str, max_age_days: float = 7.0, horizon_days: float = 0.5,
    timeout: int = 90,
) -> list[dict]:
    """Every tabledap dataset on one server whose newest observation is recent.

    The `maxTime<=` half of the constraint is not symmetry for its own sake: it
    is what keeps forecast and model-output datasets -- which legitimately carry
    timestamps months ahead -- out of the candidate pool.
    """
    now = dt.datetime.now(dt.timezone.utc)
    lo = (now - dt.timedelta(days=max_age_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    hi = (now + dt.timedelta(days=horizon_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    cols = "datasetID,title,institution,minTime,maxTime,dataStructure"
    url = (f"{base}/tabledap/allDatasets.json?{quote(cols, safe=',')}"
           f"&maxTime%3E={lo}&maxTime%3C={hi}")
    try:
        rows = _erddap_rows(erddap_get(url, timeout=timeout,
                                       max_bytes=ERDDAP_CATALOG_BYTES))
    except RuntimeError as exc:
        # ERDDAP answers an empty result set with HTTP 404, not an empty table.
        # Reported as an error it reads as "server down" and hides the real
        # finding, which is that the freshness window was simply too tight.
        if "404" in str(exc):
            return []
        raise
    out = []
    for r in rows:
        if r.get("datasetID") == "allDatasets":
            continue
        if (r.get("dataStructure") or "table") != "table":
            continue
        out.append(r)
    return out


def erddap_variables(base: str, dsid: str, timeout: int = 45) -> list[tuple[str, str]]:
    rows = _erddap_rows(erddap_get(f"{base}/info/{quote(dsid)}/index.json",
                                   timeout=timeout))
    return [
        (r.get("Variable Name") or "", (r.get("Data Type") or "").lower())
        for r in rows if r.get("Row Type") == "variable"
    ]


def erddap_pick_values(
    variables: Iterable[tuple[str, str]], limit: int = 4
) -> list[str]:
    """The observation-bearing columns, QC/coordinate/bookkeeping ones removed."""
    out = []
    for name, typ in variables:
        if not name or name == "time":
            continue
        if typ not in _ERDDAP_NUM_TYPES:
            continue
        if _ERDDAP_DROP_VAR.search(name):
            continue
        out.append(name)
        if len(out) >= limit:
            break
    return out


def erddap_data_url(base: str, dsid: str, vals: list[str], days: int) -> str:
    """`time>=now-Nd` is evaluated by the SERVER, so the entry never goes stale
    and needs no date templating in the scraper."""
    select = ",".join(["time"] + vals)
    return (f"{base}/tabledap/{quote(dsid)}.json"
            f"?{quote(select, safe=',')}&time%3E=now-{days}days")


def erddap_probe(
    base: str, dsid: str, vals: list[str], timeout: int = 60,
    windows: Iterable[int] = _ERDDAP_WINDOWS,
) -> dict:
    """Fetch real rows and measure what the wire gate will measure.

    Returns the window that worked, so the emitted URL asks for the same span --
    a window tuned to the dataset's own cadence rather than a global guess.
    """
    now = dt.datetime.now(dt.timezone.utc)
    last_err = "no window tried"
    for days in windows:
        url = erddap_data_url(base, dsid, vals, days)
        try:
            rows = _erddap_rows(erddap_get(url, timeout=timeout))
        except Exception as exc:                              # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"[:120]
            # Widening a window that already blew the size cap only asks for a
            # bigger version of the same answer.
            if "exceeds" in str(exc):
                return {"error": f"{days}d window {exc}"}
            continue
        if not rows:
            last_err = f"no rows in {days}d window"
            continue
        all_stamps = [_parse_iso(r.get("time")) for r in rows]
        all_stamps = [s for s in all_stamps if s is not None]
        cutoff = now + dt.timedelta(days=1)
        stamps = sorted({s for s in all_stamps if s <= cutoff}, reverse=True)
        n_future = len([s for s in all_stamps if s > cutoff])
        if n_future > max(1, 0.02 * len(all_stamps)):
            return {"error": f"{n_future}/{len(all_stamps)} timestamps are "
                             f"future-dated -- model output, not observations"}
        if len(stamps) < 20:
            last_err = f"only {len(stamps)} distinct ts in {days}d window"
            continue
        # Rows per distinct timestamp: >1 means several stations/depths share a
        # timestamp, i.e. a PANEL. The gate reads one series per entry, so a
        # panel silently collapses into whichever row lands last.
        ratio = len(rows) / len(stamps)
        if ratio > 1.5:
            return {"error": f"panel: {len(rows)} rows over {len(stamps)} "
                             f"distinct timestamps ({ratio:.1f}x)"}
        kept = []
        for v in vals:
            good = sum(1 for r in rows if isinstance(r.get(v), (int, float)))
            if good >= 0.5 * len(rows):
                kept.append(v)
        if not kept:
            return {"error": f"no value column is >=50% numeric over {len(rows)} rows"}
        gaps = [(stamps[i] - stamps[i + 1]).total_seconds()
                for i in range(len(stamps) - 1)]
        gaps = [g for g in gaps if g > 0]
        return {
            "rows": len(rows),
            "distinct": len(stamps),
            "window_days": days,
            "values": kept,
            "newest": stamps[0].isoformat(),
            "age_days": (now - stamps[0]).total_seconds() / 86400.0,
            "median_gap_s": statistics.median(gaps) if gaps else 0.0,
        }
    return {"error": last_err}


def _erddap_slack(median_gap_s: float, age_days: float) -> int:
    """Slack proportional to the CADENCE, not one global number.

    The Socrata generator's flat 45-day floor is right for a monthly municipal
    extract and useless for a 10-minute buoy: a mooring that goes dark stays
    "fresh" for six weeks. Slack has to be generous enough to absorb ordinary
    publication slip and tight enough that death is visible.
    """
    if median_gap_s <= 7200:                     # sub-daily: hours matter
        return max(3, int(age_days) + 3)
    if median_gap_s <= 129600:                   # daily-ish
        return max(14, int(age_days) + 10)
    return max(45, int(age_days) + 45)


def erddap_synthesize(
    base: str, label: str, ds: dict, klass: tuple[str, str, str],
    probe: dict, taken: Optional[set[str]] = None,
) -> Optional[dict]:
    host = urlparse(base).netloc.lower()
    dsid = ds.get("datasetID") or ""
    title = (ds.get("title") or dsid).strip()
    inst = (ds.get("institution") or label or "").strip()
    vals = probe.get("values") or []
    if not (host and dsid and vals):
        return None
    _kw, dom, dgp = klass
    freq = freq_from_delta(probe["median_gap_s"])
    sid = entry_id(host, title)
    if taken and sid in taken:
        sid = f"{sid[:52]}_{_slug(dsid, 10)}"[:64]

    entry: dict[str, Any] = {
        "id": sid,
        "name": f"{title} ({inst or host})",
        "domain": dom,
        "dgp_class": dgp,
        "archetypes": ["non_stationary_regime", "seasonal_multi"],
        "frequency": freq,
        "endpoint": {
            "type": "rest_json",
            "url": erddap_data_url(base, dsid, vals, probe["window_days"]),
            "auth": "none",
            "rate_limit": None,
        },
        "schema": {
            # ERDDAP answers with parallel arrays under table.rows, so the
            # fields are POSITIONAL: column 0 is always the requested `time`.
            "timestamp_field": "table.rows[][0]",
            "value_field": [f"table.rows[][{i}]" for i in range(1, len(vals) + 1)],
            "variates": len(vals),
        },
        "history_available": (
            f"from {(ds.get('minTime') or '?')[:10]} (server retains full series; "
            f"the entry polls a rolling {probe['window_days']}d window)"
        ),
        "update_cadence_observed": (
            f"median gap {probe['median_gap_s'] / 60:.1f} min over "
            f"{probe['distinct']} distinct timestamps; newest {probe['newest']}"
        ),
        "pretraining_novelty": "clean",
        "novelty_notes": (
            "Operational ocean/met observing platform served over ERDDAP; the "
            "live rolling window post-dates every known TSFM pretraining cutoff."
        ),
        "license": f"open data -- {inst or label} via ERDDAP",
        "audit_slack_days": _erddap_slack(probe["median_gap_s"], probe["age_days"]),
        "notes": (
            f"Bulk-generated from the {label} ERDDAP server ({host}). Variables "
            f"chosen mechanically from the dataset's declared types with QC "
            f"flags, coordinates and instrument ids excluded; verified against "
            f"the live endpoint by the wire gate. Columns: {', '.join(vals)}."
        ),
    }
    return {
        "candidate_name": entry["name"],
        "wireable": True,
        "yaml_block": yaml.dump([entry], sort_keys=False, allow_unicode=True),
        "cron_cadence": cron_cadence_for(freq, probe["age_days"]),
        "reason": (
            f"ERDDAP {label}: {probe['distinct']} distinct ts in a "
            f"{probe['window_days']}d window, newest {probe['newest']} "
            f"({probe['age_days']:.2f}d old), median gap "
            f"{probe['median_gap_s']:.0f}s, values={vals}"
        ),
    }


def erddap_sweep(
    catalog_path: str,
    servers: Iterable[tuple[str, str]] = ERDDAP_SERVERS,
    host_cap: int = DEFAULT_HOST_CAP,
    max_age_days: float = 14.0,
    per_server: Optional[int] = None,
    new_hosts_only: bool = False,
    target: Optional[int] = None,
    sleep_s: float = 0.3,
    checkpoint_path: Optional[str] = None,
    log=print,
) -> tuple[list[dict], list[dict]]:
    """Sweep the ERDDAP federation, at most `host_cap` datasets per server.

    `new_hosts_only` defaults to FALSE here, unlike the catalog sweeps: on
    Socrata a wired host means the portal is already represented, but an ERDDAP
    server is an entire regional observing system, and the three already in the
    catalog carry one dataset each out of hundreds.
    """
    seen_hosts = host_counts(catalog_path)
    known_hosts = wired_hosts(catalog_path)
    known_ds = wired_erddap_datasets(catalog_path)
    taken_ids = {s["id"] for s in (yaml.safe_load(open(catalog_path)) or [])}
    cands: list[dict] = []
    skipped: list[dict] = []
    cap = per_server if per_server is not None else host_cap

    for base, label in servers:
        base = base.rstrip("/")
        host = urlparse(base).netloc.lower()
        if new_hosts_only and host in known_hosts:
            skipped.append({"id": host, "reason": "host already wired"})
            continue
        room = cap - seen_hosts.get(host, 0)
        if room <= 0:
            skipped.append({"id": host, "reason": f"host cap {cap} reached"})
            continue
        try:
            datasets = erddap_datasets(base, max_age_days=max_age_days)
        except Exception as exc:                              # noqa: BLE001
            skipped.append({"id": host,
                            "reason": f"catalog unreachable: "
                                      f"{type(exc).__name__}: {exc}"[:110]})
            continue
        log(f"[{label} {host}] {len(datasets)} fresh tabledap datasets, room {room}")
        got = 0
        platforms: set[str] = set()
        for ds in datasets:
            if got >= room:
                break
            dsid = ds.get("datasetID") or ""
            title = (ds.get("title") or "").strip()
            tag = f"{host}/{dsid}"
            pkey = erddap_platform_key(title, dsid)
            if pkey in platforms:
                skipped.append({"id": tag,
                                "reason": f"same platform '{pkey}' as an "
                                          f"earlier pick on this server"})
                continue
            if f"{host}|{dsid}" in known_ds:
                skipped.append({"id": tag, "reason": "dataset already in catalog"})
                continue
            if _ERDDAP_REJECT_TITLE.search(f"{title} {dsid}"):
                skipped.append({"id": tag, "reason": f"model output: {title[:50]}"})
                continue
            if _ERDDAP_DEPLOYMENT.search(dsid) or _ERDDAP_DEPLOYMENT.search(title):
                skipped.append({"id": tag,
                                "reason": f"one-off deployment: {dsid[:40]}"})
                continue
            try:
                variables = erddap_variables(base, dsid)
            except Exception as exc:                          # noqa: BLE001
                skipped.append({"id": tag,
                                "reason": f"info failed: {type(exc).__name__}"})
                continue
            time.sleep(sleep_s)
            if not any(n == "time" for n, _ in variables):
                skipped.append({"id": tag, "reason": "no time variable"})
                continue
            vals = erddap_pick_values(variables)
            if not vals:
                skipped.append({"id": tag,
                                "reason": "no non-QC numeric variable"})
                continue
            probe = erddap_probe(base, dsid, vals)
            time.sleep(sleep_s)
            if "error" in probe:
                skipped.append({"id": tag, "reason": probe["error"]})
                continue
            use_class = erddap_class(title, ds.get("institution") or "")
            block = erddap_synthesize(base, label, ds, use_class, probe,
                                      taken=taken_ids)
            if block is None:
                skipped.append({"id": tag, "reason": "could not synthesise"})
                continue
            taken_ids.add(yaml.safe_load(block["yaml_block"])[0]["id"])
            cands.append(block)
            got += 1
            platforms.add(pkey)
            seen_hosts[host] = seen_hosts.get(host, 0) + 1
            _checkpoint(cands, checkpoint_path)
            log(f"  + {block['candidate_name'][:72]}  [{block['cron_cadence']}]")
            if target and len(cands) >= target:
                log(f"target {target} reached")
                return cands, skipped
    return cands, skipped


def wired_erddap_datasets(catalog_path: str) -> set[str]:
    """`host|datasetID` for every ERDDAP entry already wired."""
    reg = yaml.safe_load(open(catalog_path)) or []
    out = set()
    for src in reg:
        url = (src.get("endpoint") or {}).get("url", "")
        m = re.search(r"^(https?://[^/]+).*/(?:tabledap|griddap)/([^/?.]+)", url)
        if m:
            out.add(f"{urlparse(m.group(1)).netloc.lower()}|{m.group(2)}")
    return out



# --------------------------------------------------------------------------- #
# GBFS
# --------------------------------------------------------------------------- #
# The bike/scooter-share standard, with an official registry of every system
# that speaks it. It is the largest single list of live feeds available -- and
# the clearest illustration of why SOURCE count and HOST count are different
# numbers: 1,536 registered systems resolve to only ~134 distinct hostnames,
# because operators like nextbike, urbansharing and Lyft serve dozens of cities
# each from one endpoint. Wiring all 1,536 would multiply the source count by a
# factor the coverage metric would refuse to credit, and would hand one operator
# a hundred entries. So the cap here is per HOST, not per system, and it is
# tight.
#
# Station status is a genuine panel -- one series per dock -- which the gate
# admits (>=20 entities x >=20 numeric rows) and which the eval pool caps at 25
# series per feed downstream. Free-floating vehicle feeds are deliberately NOT
# used: their panel ids are individual bikes that appear and vanish, so the
# "series" churns its own identity between polls.
GBFS_SYSTEMS_CSV = ("https://raw.githubusercontent.com/MobilityData/gbfs/"
                    "master/systems.csv")

# GBFS 3.0 renamed the availability field; both spellings are live in the wild.
_GBFS_COUNT_FIELDS = ("num_bikes_available", "num_vehicles_available")
_GBFS_DOCK_FIELDS = ("num_docks_available", "num_docks_disabled")


def gbfs_systems(timeout: int = 60) -> list[dict]:
    r = requests.get(GBFS_SYSTEMS_CSV, headers={"User-Agent": UA}, timeout=timeout)
    r.raise_for_status()
    return list(csv.DictReader(r.text.splitlines()))


def gbfs_feeds(discovery_url: str, timeout: int = 30) -> dict[str, str]:
    """feed name -> url, across both GBFS layouts.

    Pre-3.0 keys the feed list by language (`data.en.feeds`); 3.0 drops the
    language level (`data.feeds`). Assuming either one silently loses half the
    registry.
    """
    r = requests.get(discovery_url, headers={"User-Agent": UA}, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    data = (r.json() or {}).get("data") or {}
    feeds: list[dict] = []
    if isinstance(data.get("feeds"), list):
        feeds = data["feeds"]
    else:
        for lang in ("en", "fr", "de", "es", "nl", "no", "fi", "pl", "it"):
            if isinstance(data.get(lang), dict) and data[lang].get("feeds"):
                feeds = data[lang]["feeds"]
                break
        else:
            for val in data.values():
                if isinstance(val, dict) and isinstance(val.get("feeds"), list):
                    feeds = val["feeds"]
                    break
    return {f.get("name", ""): f.get("url", "") for f in feeds if f.get("url")}


def gbfs_probe(url: str, timeout: int = 30) -> dict:
    """Measure the panel: how many stations, and do they carry a usable clock?"""
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}"}
        payload = r.json()
    except Exception as exc:                                  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:120]}
    stations = ((payload or {}).get("data") or {}).get("stations")
    if not isinstance(stations, list) or not stations:
        return {"error": "no data.stations array"}
    if len(stations) < 20:
        return {"error": f"only {len(stations)} stations (gate needs 20 entities)"}
    field = next((f for f in _GBFS_COUNT_FIELDS
                  if sum(1 for s in stations if isinstance(s.get(f), (int, float)))
                  >= 0.5 * len(stations)), None)
    if not field:
        return {"error": "no availability count on >=50% of stations"}
    if not any(isinstance(s.get("station_id"), (str, int)) for s in stations):
        return {"error": "stations carry no station_id to key the panel on"}
    reported = [s.get("last_reported") for s in stations]
    has_clock = sum(1 for v in reported if isinstance(v, (int, float, str)) and v)
    if has_clock < 0.5 * len(stations):
        return {"error": "no last_reported on >=50% of stations"}
    docks = [f for f in _GBFS_DOCK_FIELDS
             if sum(1 for s in stations if isinstance(s.get(f), (int, float)))
             >= 0.5 * len(stations)]
    ttl = payload.get("ttl")
    return {
        "stations": len(stations),
        "value_fields": [field] + docks[:1],
        "ttl": ttl if isinstance(ttl, (int, float)) and ttl > 0 else None,
    }


def gbfs_synthesize(system: dict, url: str, probe: dict,
                    taken: Optional[set[str]] = None) -> Optional[dict]:
    host = urlparse(url).netloc.lower()
    name = (system.get("Name") or "").strip()
    location = (system.get("Location") or "").strip()
    country = (system.get("Country Code") or "").strip()
    if not (host and name):
        return None
    label = f"{name} ({location})" if location else name
    # Dedupe on the id that actually ships. This generator is the only one that
    # prefixes, so comparing the bare sid against a set of wired ids would never
    # match and a collision would overwrite a source instead of adding one.
    sid = f"gbfs_{entry_id(host, f'{name} {location}')}"[:64]
    if taken and sid in taken:
        sid = f"{sid[:52]}_{_slug(system.get('System ID') or '', 10)}"[:64]
    vals = probe["value_fields"]

    entry: dict[str, Any] = {
        "id": sid,
        "name": f"{label} — per-station availability (GBFS panel)",
        "domain": "transport",
        "dgp_class": "micromobility_availability",
        "archetypes": ["count_discrete", "bounded", "smooth_periodic",
                       "hierarchical"],
        # The SERIES moves per minute, but polling ~100 systems that often costs
        # more sweep budget than dock occupancy at 1-minute resolution is worth.
        "frequency": "PT5M",
        "endpoint": {
            "type": "rest_json",
            "url": url,
            "auth": "none",
            "rate_limit": None,
        },
        "schema": {
            "timestamp_field": "data.stations[].last_reported",
            "value_field": [f"data.stations[].{v}" for v in vals],
            "panel_field": "data.stations[].station_id",
            "variates": len(vals),
        },
        "history_available": "P0D — live snapshot, history accrues from wiring",
        "update_cadence_observed": (
            f"{probe['stations']} stations; ttl "
            f"{probe['ttl'] if probe['ttl'] is not None else 'unset'}s"
        ),
        "pretraining_novelty": "clean",
        "novelty_notes": (
            "Live dock occupancy; the snapshot series begins at wiring time and "
            "so post-dates every known TSFM pretraining cutoff."
        ),
        "license": "GBFS open data (see operator terms)",
        "audit_slack_days": 3,
        "notes": (
            f"Bulk-generated from the MobilityData GBFS registry "
            f"({country or '??'}, system id {system.get('System ID') or '?'}). "
            f"Station panel keyed on station_id; free-floating vehicle feeds are "
            f"excluded because their panel ids churn between polls."
        ),
    }
    return {
        "candidate_name": entry["name"],
        "wireable": True,
        "yaml_block": yaml.dump([entry], sort_keys=False, allow_unicode=True),
        "cron_cadence": "PT5M",
        "reason": (f"GBFS registry: {probe['stations']} stations, fields "
                   f"{vals}, host {host}"),
    }


def gbfs_sweep(
    catalog_path: str,
    host_cap: int = 2,
    target: Optional[int] = None,
    sleep_s: float = 0.3,
    checkpoint_path: Optional[str] = None,
    log=print,
) -> tuple[list[dict], list[dict]]:
    seen_hosts = host_counts(catalog_path)
    known_urls = {(s.get("endpoint") or {}).get("url", "")
                  for s in (yaml.safe_load(open(catalog_path)) or [])}
    taken_ids = {s["id"] for s in (yaml.safe_load(open(catalog_path)) or [])}
    cands: list[dict] = []
    skipped: list[dict] = []
    try:
        systems = gbfs_systems()
    except Exception as exc:                                  # noqa: BLE001
        return [], [{"id": "registry", "reason": f"registry unreachable: {exc}"}]
    log(f"[GBFS] {len(systems)} registered systems")

    for system in systems:
        disc = (system.get("Auto-Discovery URL") or "").strip()
        name = (system.get("Name") or "").strip()
        if not disc.startswith("http"):
            continue
        if (system.get("Authentication Type") or "").strip() not in ("", "None"):
            skipped.append({"id": name, "reason": "feed requires authentication"})
            continue
        dhost = urlparse(disc).netloc.lower()
        if seen_hosts.get(dhost, 0) >= host_cap:
            skipped.append({"id": name, "reason": f"host cap {host_cap} reached"})
            continue
        try:
            feeds = gbfs_feeds(disc)
        except Exception as exc:                              # noqa: BLE001
            skipped.append({"id": name,
                            "reason": f"discovery failed: {type(exc).__name__}"})
            continue
        time.sleep(sleep_s)
        url = feeds.get("station_status") or ""
        if not url:
            skipped.append({"id": name, "reason": "no station_status feed "
                                                  "(free-floating system)"})
            continue
        if url in known_urls:
            skipped.append({"id": name, "reason": "feed already in catalog"})
            continue
        host = urlparse(url).netloc.lower()
        if seen_hosts.get(host, 0) >= host_cap:
            skipped.append({"id": name, "reason": f"host cap {host_cap} reached"})
            continue
        probe = gbfs_probe(url)
        time.sleep(sleep_s)
        if "error" in probe:
            skipped.append({"id": name, "reason": probe["error"]})
            continue
        block = gbfs_synthesize(system, url, probe, taken=taken_ids)
        if block is None:
            skipped.append({"id": name, "reason": "could not synthesise"})
            continue
        taken_ids.add(yaml.safe_load(block["yaml_block"])[0]["id"])
        cands.append(block)
        known_urls.add(url)
        seen_hosts[host] = seen_hosts.get(host, 0) + 1
        seen_hosts[dhost] = seen_hosts.get(dhost, 0) + 1
        _checkpoint(cands, checkpoint_path)
        log(f"  + {block['candidate_name'][:70]}  [{probe['stations']} stations]")
        if target and len(cands) >= target:
            log(f"target {target} reached")
            return cands, skipped
    return cands, skipped



# --------------------------------------------------------------------------- #
# Socrata, publisher-first
# --------------------------------------------------------------------------- #
# Same inversion as ods_publisher_sweep, and the same two hazards: no query to
# anchor a class to, and a portal walk that returns the whole portal rather than
# datasets about a topic. Both are handled the same way -- classify off the
# publisher's own filing (`classification.domain_category`, Socrata's analogue
# of ODS `theme`) and reject registry timestamps and identifier columns.
#
# Enumerating the domains is the part that differs. Socrata has no domains
# endpoint (/api/catalog/v1/domains is a 404 on both the US and EU catalogs), so
# the list comes from scrolling the federated catalog and collecting
# metadata.domain. That surfaces a lot of abandoned Socrata-hosted portals:
# of 249 domains found, 164 were unwired and production, and only 58 had
# anything updated in the last three weeks. Measure liveness before sweeping --
# it is one request per domain and skips two thirds of the work.
SOCRATA_NONPROD = ("demo", "qa", "beta", "test", "staging", "sandbox",
                   "training", "internal")


def socrata_is_production(domain: str) -> bool:
    """Token-wise, not substring: these markers turn up in any position and with
    either separator (budget-QA-reporting.data.socrata.com), while a substring
    test would also reject legitimate names containing them."""
    parts = set(re.split(r"[.\-_]+", domain.lower()))
    return not (parts & set(SOCRATA_NONPROD))


def socrata_domains(scroll_pages: int = 400, timeout: int = 60,
                    sleep_s: float = 0.15, log=print) -> list[str]:
    """Every domain the federated catalog will admit to, by scroll paging."""
    doms: set[str] = set()
    scroll, pages = None, 0
    while pages < scroll_pages:
        params: dict[str, Any] = {"only": "dataset", "limit": 100}
        if scroll:
            params["scroll_id"] = scroll
        try:
            r = requests.get(CATALOG_API, params=params,
                             headers={"User-Agent": UA}, timeout=timeout)
            res = r.json().get("results", []) if r.status_code == 200 else []
        except Exception:                                     # noqa: BLE001
            break
        if not res:
            break
        for x in res:
            d = (x.get("metadata") or {}).get("domain")
            if d:
                doms.add(d.lower())
        scroll = res[-1]["resource"]["id"]
        pages += 1
        time.sleep(sleep_s)
    log(f"  [socrata-pub] {pages * 100} datasets scanned, {len(doms)} domains")
    return sorted(doms)


def socrata_domain_datasets(domain: str, want: int = 25,
                            timeout: int = 45) -> list[dict]:
    """That domain's most recently updated datasets, newest first.

    Both `domains` and `search_context` are required: with `domains` alone the
    big portals answer 0 (data.austintexas.gov reports 0 vs 705 with context).
    """
    r = requests.get(CATALOG_API, params={
        "domains": domain, "search_context": domain,
        "only": "dataset", "order": "updatedAt DESC",
        "limit": min(CATALOG_PAGE, want),
    }, headers={"User-Agent": UA}, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return r.json().get("results", [])


def socrata_publisher_sweep(
    catalog_path: str,
    host_cap: int = 2,
    scan_per_host: int = 25,
    max_age_days: float = 21.0,
    domains: Optional[Iterable[str]] = None,
    target: Optional[int] = None,
    sleep_s: float = 0.3,
    checkpoint_path: Optional[str] = None,
    log=print,
) -> tuple[list[dict], list[dict]]:
    known_hosts = wired_hosts(catalog_path)
    seen_hosts = host_counts(catalog_path)
    known_ids = wired_resource_ids(catalog_path)
    taken_ids = {s["id"] for s in (yaml.safe_load(open(catalog_path)) or [])}
    cands: list[dict] = []
    skipped: list[dict] = []

    if domains is None:
        log("[socrata-pub] enumerating domains...")
        domains = socrata_domains(log=log)
    domains = [d for d in domains if d and socrata_is_production(d)]
    log(f"[socrata-pub] {len(domains)} production domains; "
        f"{sum(1 for d in domains if d in known_hosts)} already wired")

    for domain in domains:
        if domain in known_hosts or seen_hosts.get(domain, 0) >= host_cap:
            skipped.append({"id": domain, "reason": "host already wired"})
            continue
        try:
            records = socrata_domain_datasets(domain, scan_per_host)
        except Exception as exc:                              # noqa: BLE001
            skipped.append({"id": domain, "reason": f"listing failed: {exc}"})
            continue
        time.sleep(sleep_s)
        got = 0
        for res in records:
            if got >= host_cap:
                break
            resource = res.get("resource") or {}
            rid = resource.get("id") or ""
            title = (resource.get("name") or "").strip()
            tag = f"{domain}/{rid}"
            if not rid or rid in known_ids:
                continue
            updated = _parse_iso(resource.get("updatedAt"))
            if updated is None:
                continue
            age = (dt.datetime.now(dt.timezone.utc) - updated).total_seconds() / 86400
            if age > max_age_days:
                # Sorted newest-first, so the rest of this portal is older.
                skipped.append({"id": domain,
                                "reason": f"nothing updated inside "
                                          f"{max_age_days:.0f}d (newest {age:.0f}d)"})
                break
            if _ODS_REJECT_TITLE.search(title):
                skipped.append({"id": tag,
                                "reason": f"not a forecastable process: {title[:50]}"})
                continue
            category = (res.get("classification") or {}).get("domain_category", "")
            use_class = ods_publisher_class([category] if category else [], title,
                                            resource.get("description") or "")
            if use_class is None:
                skipped.append({"id": tag,
                                "reason": f"category not in the map: {category!r}"})
                continue
            cols = _cols(resource)
            ts = pick_timestamp_column(cols)
            if not ts:
                skipped.append({"id": tag, "reason": "no date/timestamp column"})
                continue
            if _ODS_RECORD_TS.search(ts):
                skipped.append({"id": tag,
                                "reason": f"registry: '{ts}' is a record-keeping "
                                          f"date, not an observation clock"})
                continue
            vals = [v for v in pick_value_columns(cols, ts)
                    if not _ODS_ID_VAL.search(v)]
            if not vals:
                skipped.append({"id": tag,
                                "reason": "no numeric field that measures anything"})
                continue

            probe = probe_series(domain, rid, ts)
            time.sleep(sleep_s)
            if probe is None or "error" in probe:
                skipped.append({"id": tag,
                                "reason": f"probe failed: {(probe or {}).get('error')}"})
                continue
            if probe["distinct"] < 20:
                skipped.append({"id": tag,
                                "reason": f"only {probe['distinct']} distinct ts"})
                continue
            if probe["age_days"] > max_age_days * 3:
                skipped.append({"id": tag,
                                "reason": f"newest observation {probe['age_days']:.0f}d old"})
                continue
            block = synthesize(res, use_class, probe, taken=taken_ids)
            if block is None:
                skipped.append({"id": tag, "reason": "could not synthesise entry"})
                continue
            taken_ids.add(yaml.safe_load(block["yaml_block"])[0]["id"])
            known_ids.add(rid)
            cands.append(block)
            got += 1
            seen_hosts[domain] = seen_hosts.get(domain, 0) + 1
            _checkpoint(cands, checkpoint_path)
            log(f"  + [{use_class[1]}] {block['candidate_name'][:66]}")
            if target and len(cands) >= target:
                log(f"target {target} reached")
                return cands, skipped
    return cands, skipped



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
