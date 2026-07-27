"""bulk.py — synthesise catalog candidates from federated open-data catalogs.

Agent grind waves find *interesting* sources but cost 20-40 minutes for 10-18
of them. Reaching a 1000-source catalog needs a second gear: whole classes of
sources whose shape is machine-derivable, with no LLM in the loop.

Socrata is that class. Its federated catalog API searches every Socrata domain
at once and returns, per dataset, the host, the dataset id, the last-updated
time, and the full column list *with datatypes*. That is everything needed to
write a catalog entry: the ``Calendar date`` column is the timestamp, the
``Number`` columns are the values, and the URL is mechanical. Every city portal
is also a distinct host, so this buys provider diversity at the same time as
volume — which is the axis the catalog is actually short on.

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
import statistics
import time
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
def _slug(s: str, maxlen: int = 44) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")
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
            results = search_catalog(kw, limit=per_keyword)
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

            block = synthesize(res, klass, probe, taken=taken_ids)
            if block is None:
                skipped.append({"id": tag, "reason": "could not synthesise entry"})
                continue
            taken_ids.add(yaml.safe_load(block["yaml_block"])[0]["id"])
            cands.append(block)
            seen_hosts[host] = seen_hosts.get(host, 0) + 1
            log(f"  + {block['candidate_name']}  [{block['cron_cadence']}]")
            if target and len(cands) >= target:
                log(f"target {target} reached")
                return cands, skipped
    return cands, skipped


def write_batch(candidates: list[dict], out_path: str) -> str:
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(candidates, fh, indent=2)
    return out_path
