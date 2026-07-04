#!/usr/bin/env python3
"""
Timeframe benchmark — unified scraper.

Drives data ingestion entirely from sources.yaml. No per-source bespoke files.

Per-source behaviour:

    fetch -> parse -> dedupe -> append to data/<id>/<YYYY-MM-DD>.parquet

The parser is generic: each entry's `endpoint.type` selects a fetcher (rest_json,
rest_csv, rss). The `schema` block names where the timestamp + value(s) live
inside the payload. For payloads with shapes that don't decompose neatly, the
scraper falls back to writing the raw payload as a single timestamped row
(timestamp = now, value = json string) — this is the fallback for
"snapshot" sources whose meaning is "what the API said at poll time".

Idempotency: deduplication is on the timestamp column, scoped per UTC date
file. Re-running the same minute does not duplicate.

Failures are logged to data/<id>/_errors.log with HTTP code + message.

Usage:
    python scraper.py --id <source_id>            # single source
    python scraper.py --all                       # iterate every source
    python scraper.py --domain energy             # subset
    python scraper.py --id <id> --dry-run         # fetch + parse but do not write
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import logging
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

try:
    import httpx
except ImportError:                                # noqa: BLE001
    sys.stderr.write("scraper.py requires httpx. Install: pip install httpx pyarrow pyyaml\n")
    raise

try:
    import pyarrow as pa                           # noqa: F401
    import pyarrow.parquet as pq                   # noqa: F401
except ImportError:
    sys.stderr.write("scraper.py requires pyarrow. Install: pip install pyarrow\n")
    raise


ROOT = Path(__file__).resolve().parent
SOURCES_YAML = ROOT / "sources.yaml"
DATA_DIR = ROOT / "data"

logging.basicConfig(
    level=os.environ.get("SCRAPER_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("timeframe-scraper")

UTC = dt.timezone.utc


# ──────────────────────────────────────────────────────────────────────────
# Source loading
# ──────────────────────────────────────────────────────────────────────────


def load_sources() -> list[dict]:
    with open(SOURCES_YAML) as f:
        sources = yaml.safe_load(f)
    if not isinstance(sources, list):
        raise ValueError("sources.yaml must be a YAML list")
    return sources


def get_source(sid: str) -> dict:
    for s in load_sources():
        if s["id"] == sid:
            return s
    raise KeyError(f"source id not found: {sid}")


# ──────────────────────────────────────────────────────────────────────────
# URL templating
# ──────────────────────────────────────────────────────────────────────────

_TODAY_PATTERNS = [
    ("{YYYYMMDD}", "%Y%m%d"),
    ("{YYYY-MM-DD}", "%Y-%m-%d"),
    ("{YYYY/MM/DD}", "%Y/%m/%d"),
    ("{YYYY}", "%Y"),
    ("{MM}", "%m"),
    ("{DD}", "%d"),
    ("{YYYYMM}", "%Y%m"),
]

_OFFSET_DATE_RE = re.compile(r"\{YYYY-MM-DD-(\d+)d\}")
_OFFSET_HOUR_RE = re.compile(r"\{H-(\d+)\}")


def expand_url(url: str, now: Optional[dt.datetime] = None) -> str:
    """Substitute date tokens in URL templates. Tokens left untouched if value
    not derivable from current UTC date.

    Static tokens: {YYYY}, {MM}, {DD}, {YYYYMMDD}, {YYYY-MM-DD},
    {YYYY/MM/DD}, {YYYYMM}, {H} (current UTC hour, 0-23, unpadded),
    {HH} (zero-padded).

    Relative-offset tokens (regex):
      {YYYY-MM-DD-Nd}   today minus N days, formatted YYYY-MM-DD
                        (e.g. {YYYY-MM-DD-30d} = 30 days ago).
      {H-N}             current UTC hour minus N (wraps at midnight UTC)
                        (e.g. {H-1} = previous hour).

    Tokens like {ARTICLE} or {PACKAGE} or {CRATE} are *not* expanded — these
    are panel-iteration placeholders; the caller should pre-expand them.
    """
    now = now or dt.datetime.now(UTC)
    # Relative offsets first so they don't get partially consumed by the literal
    # {H} / {YYYY-MM-DD} substitutions below.
    url = _OFFSET_DATE_RE.sub(
        lambda m: (now - dt.timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d"),
        url,
    )
    url = _OFFSET_HOUR_RE.sub(
        lambda m: str((now - dt.timedelta(hours=int(m.group(1)))).hour),
        url,
    )
    if "{HH}" in url:
        url = url.replace("{HH}", f"{now.hour:02d}")
    if "{H}" in url:
        url = url.replace("{H}", str(now.hour))
    if "{ISO_DATE}" in url:
        url = url.replace("{ISO_DATE}", now.strftime("%Y-%m-%d"))
    if "{ISO_DATETIME}" in url:
        url = url.replace("{ISO_DATETIME}", now.strftime("%Y-%m-%dT%H:%M:%S"))
    for token, fmt in _TODAY_PATTERNS:
        if token in url:
            url = url.replace(token, now.strftime(fmt))
    return url


# ──────────────────────────────────────────────────────────────────────────
# Fetchers
# ──────────────────────────────────────────────────────────────────────────


def _resolve_auth(auth_field: Optional[str]) -> dict[str, str]:
    """Return http headers derived from an `auth:` field.

    Supported forms:
      auth: none                           → {}
      auth: env:VAR                        → header inferred from VAR suffix
                                             (*_USER_AGENT → User-Agent,
                                              *_API_KEY    → X-API-Key,
                                              *_TOKEN      → Authorization: Bearer,
                                              default      → Authorization: Bearer)
      auth: env:VAR:HeaderName             → explicit header name
                                             (e.g. env:LTA_ACCOUNT_KEY:AccountKey
                                              → AccountKey: <value>)
    """
    if not auth_field or auth_field == "none":
        return {}
    if auth_field.startswith("env:"):
        rest = auth_field[4:]
        # env:VAR:HeaderName — explicit header name
        if ":" in rest:
            var, header_name = rest.split(":", 1)
            val = os.environ.get(var)
            if not val:
                log.warning("env var %s referenced but unset; sending no auth", var)
                return {}
            return {header_name: val}
        var = rest
        val = os.environ.get(var)
        if not val:
            log.warning("env var %s referenced but unset; sending no auth", var)
            return {}
        # Inferred header from VAR suffix
        if var.endswith("USER_AGENT"):
            return {"User-Agent": val}
        if var.endswith("API_KEY"):
            return {"X-API-Key": val}
        if var.endswith("TOKEN"):
            return {"Authorization": f"Bearer {val}"}
        return {"Authorization": f"Bearer {val}"}
    return {}


def _expand_env(url: str) -> str:
    """Substitute {ENVVAR} placeholders in URL from os.environ.

    Used for APIs that require the key in the URL query string rather than a
    header (EIA `?api_key=...`, Banxico `?token=...`, USDA NASS `?key=...`).
    Unset variables are left in place so the leftover-placeholder check downstream
    catches the misconfiguration.
    """
    return _ENV_PLACEHOLDER_RE.sub(
        lambda m: os.environ.get(m.group(1), m.group(0)),
        url,
    )


_ENV_PLACEHOLDER_RE = re.compile(r"\{([A-Z][A-Z0-9_]+)\}")


def _http_get(url: str, headers: dict[str, str]) -> httpx.Response:
    # Default UA acceptable for most APIs; crates.io/Reddit need a contact.
    headers = {"User-Agent": "timeframe-scraper/0.1 (+https://github.com/timeframe-bench)", **headers}
    timeout = httpx.Timeout(30.0, connect=10.0)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        return client.get(url, headers=headers)


def _expand_panel(url: str, panel_row: dict[str, str]) -> str:
    """Substitute panel placeholders ({ARTICLE}, {PACKAGE}, {CRATE}, ...)."""
    for key, val in panel_row.items():
        url = url.replace("{" + key + "}", val)
    return url


_PLACEHOLDER_RE = re.compile(r"\{([A-Z_]+)\}")


def fetch_payload(src: dict, panel_row: Optional[dict[str, str]] = None) -> tuple[bytes, str]:
    ep = src["endpoint"]
    url = expand_url(ep["url"])
    if panel_row:
        url = _expand_panel(url, panel_row)
    url = _expand_env(url)                                    # {ENVVAR} → os.environ[VAR]
    leftover = _PLACEHOLDER_RE.findall(url)
    if leftover:
        raise RuntimeError(
            f"unresolved placeholders {leftover} in URL — define `panel` in sources.yaml or pass --panel"
        )
    headers = _resolve_auth(ep.get("auth"))
    # Up to 3 attempts. Two failure modes are retried: transient transport
    # faults (DNS blips, SSL handshake stalls — observed on WSL2 / long-haul
    # routes), and 429/503 throttling with Retry-After honoured (capped) —
    # pypistats in particular 429s bursts of panel rows.
    for attempt in range(3):
        try:
            resp = _http_get(url, headers)
        except httpx.TransportError as e:
            if attempt == 2:
                raise
            wait = 5 * (attempt + 1)
            log.info("transport error for %s (%s) — retrying in %ds", url[:100], e, wait)
            time.sleep(wait)
            continue
        if resp.status_code in (429, 503) and attempt < 2:
            try:
                wait = min(float(resp.headers.get("Retry-After") or 15), 60.0)
            except ValueError:  # HTTP-date form; just use the default
                wait = 15.0
            log.info("HTTP %d for %s — retrying in %.0fs", resp.status_code, url[:100], wait)
            time.sleep(wait)
            continue
        break
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.content, resp.headers.get("content-type", "")


# ──────────────────────────────────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────────────────────────────────


def _tokenize(path: str) -> list[tuple[str, str]]:
    """Tokenize path syntax into [('key',name) | ('idx', '' or 'N')] pairs.

    Examples:
        'data.foo'      -> [('key','data'), ('key','foo')]
        'a[].b'         -> [('key','a'), ('idx',''), ('key','b')]
        'prices[][0]'   -> [('key','prices'), ('idx',''), ('idx','0')]
        'a[3].b'        -> [('key','a'), ('idx','3'), ('key','b')]
    """
    tokens: list[tuple[str, str]] = []
    i, n = 0, len(path)
    while i < n:
        c = path[i]
        if c == ".":
            i += 1
        elif c == "[":
            j = path.index("]", i)
            tokens.append(("idx", path[i + 1:j]))
            i = j + 1
        else:
            j = i
            while j < n and path[j] not in ".[":
                j += 1
            tokens.append(("key", path[i:j]))
            i = j
    return tokens


def _apply(obj: Any, tokens: list[tuple[str, str]]) -> Any:
    """Apply tokens to obj. Iteration tokens (`[]`) lift the rest of the path
    over every element, returning a list aligned with the parent list."""
    if not tokens:
        return obj
    kind, val = tokens[0]
    rest = tokens[1:]

    if kind == "key":
        if obj is None:
            return None
        if isinstance(obj, dict):
            return _apply(obj.get(val), rest)
        if isinstance(obj, list):
            return [_apply(item, [(kind, val)] + rest) for item in obj]
        return None

    # idx
    if val == "":                                        # iterate
        if obj is None:
            return None
        if not isinstance(obj, list):
            return None
        out = [_apply(item, rest) for item in obj]
        # A second `[]` deeper in the path yields a list per element; flatten
        # one level so `a[].b[].c` gives one flat sequence instead of nesting
        # (which collapsed multi-level feeds like SNOTEL to a single record).
        if any(isinstance(x, list) for x in out):
            flat: list = []
            for x in out:
                flat.extend(x) if isinstance(x, list) else flat.append(x)
            return flat
        return out
    # explicit index
    try:
        idx = int(val)
    except ValueError:
        return None
    if isinstance(obj, list):
        if -len(obj) <= idx < len(obj):
            return _apply(obj[idx], rest)
        return None
    if isinstance(obj, dict):                            # numeric key fallback
        return _apply(obj.get(val), rest)
    return None


def _walk(obj: Any, path: str) -> Any:
    """Walk a JSON-like object using a dotted/bracketed path."""
    if path == "":
        return obj
    return _apply(obj, _tokenize(path))


def _epoch_to_iso(v: Any) -> Optional[str]:
    """Heuristically convert a number-or-string ts into an ISO UTC string."""
    if v is None:
        return None
    if isinstance(v, str):
        # Already a parseable string — best-effort, return as-is
        return v
    if isinstance(v, (int, float)):
        n = float(v)
        # epoch ms vs s: anything > 1e12 is ms
        if n > 1e12:
            n /= 1000.0
        try:
            return dt.datetime.fromtimestamp(n, UTC).isoformat()
        except (ValueError, OverflowError, OSError):
            return None
    return str(v)


def _records_from_json(data: Any, schema: dict) -> list[dict]:
    """Build a list of {timestamp, value} records from a JSON payload.

    If `value_field` is a list, each row carries a dict of named values.

    Falls back to a single row (timestamp=now, value=raw_json) if the
    declared paths don't yield aligned arrays — useful for snapshot endpoints.
    """
    ts_path = schema.get("timestamp_field", "")
    val_path = schema.get("value_field", "")
    now_iso = dt.datetime.now(UTC).isoformat()

    # Special: timestamp_field == 'now()' -> snapshot semantics
    if ts_path == "now()":
        return [{"timestamp": now_iso, "value": json.dumps(data)[:200000]}]

    # Try to walk
    try:
        ts_seq = _walk(data, ts_path) if ts_path else None
    except Exception as e:                                  # noqa: BLE001
        log.debug("ts walk failed (%s); fallback snapshot row", e)
        return [{"timestamp": now_iso, "value": json.dumps(data)[:200000]}]

    if isinstance(val_path, list):
        try:
            val_seqs = {p.split(".")[-1].strip("[]"): _walk(data, p) for p in val_path}
        except Exception:
            return [{"timestamp": now_iso, "value": json.dumps(data)[:200000]}]
    else:
        try:
            val_seqs = {"value": _walk(data, val_path)} if val_path else {}
        except Exception:
            return [{"timestamp": now_iso, "value": json.dumps(data)[:200000]}]

    if isinstance(ts_seq, list):
        rows: list[dict] = []
        for i, ts in enumerate(ts_seq):
            row = {"timestamp": _epoch_to_iso(ts) or now_iso}
            for name, seq in val_seqs.items():
                if isinstance(seq, list) and i < len(seq):
                    row[name] = seq[i]
                else:
                    row[name] = seq
            rows.append(row)
        return rows

    if ts_seq is None and val_seqs:
        # snapshot with one current value
        row = {"timestamp": now_iso}
        for name, seq in val_seqs.items():
            row[name] = seq
        return [row]

    # last resort
    return [{"timestamp": _epoch_to_iso(ts_seq) or now_iso,
             "value": json.dumps(data)[:200000]}]


def _records_from_csv(text: str, schema: dict) -> list[dict]:
    f = io.StringIO(text)
    reader = csv.DictReader(f)
    ts_field = schema.get("timestamp_field", "").split("/")[-1]
    val_fields = schema.get("value_field")
    if isinstance(val_fields, str):
        val_fields = [val_fields]

    rows = []
    for r in reader:
        ts_raw = r.get(ts_field) or next(iter(r.values()), None)
        row = {"timestamp": _epoch_to_iso(ts_raw) or ts_raw}
        for vf in (val_fields or []):
            short = vf.split("/")[-1]
            row[short] = r.get(short)
        if not val_fields:
            row.update({k: v for k, v in r.items() if k != ts_field})
        rows.append(row)
    return rows


def _records_from_zip_xml(blob: bytes, schema: dict) -> list[dict]:
    """Minimal CAISO-style ZIP-XML extractor: walk every leaf XML element and
    emit a row whenever we find both an INTERVAL_START_GMT and a VALUE sibling
    inside the same parent.  Sufficient for verification and downstream parquet
    appends; replace with a proper xmlschema parser if production-grade
    fidelity is needed."""
    import xml.etree.ElementTree as ET

    rows = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in z.namelist():
            if not name.lower().endswith(".xml"):
                continue
            tree = ET.parse(z.open(name))
            for parent in tree.iter():
                ts = None
                val = None
                for child in parent:
                    tag = child.tag.split("}")[-1]
                    if tag in ("INTERVAL_START_GMT", "DATA_ITEM_START_TIME"):
                        ts = child.text
                    elif tag in ("VALUE",):
                        val = child.text
                if ts and val:
                    rows.append({"timestamp": ts, "value": val})
    return rows


def _records_from_rss(text: str, schema: dict) -> list[dict]:
    """RSS items -> rows of {timestamp=pubDate, value=title|category}."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise RuntimeError(f"RSS parse error: {e}")

    rows = []
    for item in root.iter("item"):
        ts = item.findtext("pubDate")
        val = item.findtext("category") or item.findtext("title") or ""
        rows.append({"timestamp": _epoch_to_iso(ts) or ts, "value": val})
    if not rows:
        # Atom feeds (tsunami.gov, many gov alert streams): namespaced
        # <entry> elements with <updated>/<title>/<category term=...>.
        for entry in root.iter():
            if entry.tag.rsplit("}", 1)[-1] != "entry":
                continue
            ts = val = None
            for child in entry:
                tag = child.tag.rsplit("}", 1)[-1]
                if tag == "updated":
                    ts = child.text
                elif tag == "category" and child.get("term"):
                    val = child.get("term")
                elif tag == "title" and val is None:
                    val = child.text
            rows.append({"timestamp": _epoch_to_iso(ts) or ts, "value": val or ""})
    return rows


# ──────────────────────────────────────────────────────────────────────────
# Storage (parquet append + dedupe by timestamp)
# ──────────────────────────────────────────────────────────────────────────


def write_records(sid: str, records: list[dict], dry_run: bool = False) -> int:
    if not records:
        return 0
    today = dt.datetime.now(UTC).strftime("%Y-%m-%d")
    out_dir = DATA_DIR / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{today}.parquet"

    # Coerce all values to strings so concat across heterogeneous payloads is
    # never blocked by Arrow type unification. Downstream consumers cast as
    # needed; raw fidelity is preserved.
    str_records = [{k: (None if v is None else str(v)) for k, v in r.items()} for r in records]
    new_table = pa.Table.from_pylist(str_records)

    def _all_string(t: pa.Table) -> pa.Table:
        return t.cast(pa.schema([(f.name, pa.string()) for f in t.schema]))

    new_table = _all_string(new_table)
    if out_path.exists():
        existing = _all_string(pq.read_table(out_path))
        try:
            combined = pa.concat_tables(
                [existing, new_table], promote_options="default"
            )
        except (TypeError, pa.lib.ArrowInvalid):
            combined = pa.concat_tables([existing, new_table])
    else:
        combined = new_table

    # Dedupe on the full row so genuinely distinct measurements at the same
    # timestamp (multiple stations/zones/events) are preserved. Re-runs of the
    # same poll fold identical rows.
    df = combined.to_pandas()
    df = df.drop_duplicates(keep="first")
    n_written = len(df)
    if dry_run:
        log.info("[dry-run] %s: would write %d unique rows to %s", sid, n_written, out_path)
        return n_written
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_path)
    return n_written


def log_error(sid: str, msg: str) -> None:
    out_dir = DATA_DIR / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    err = out_dir / "_errors.log"
    with err.open("a") as f:
        f.write(f"{dt.datetime.now(UTC).isoformat()} {msg}\n")


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────


def parse_payload(src: dict, blob: bytes, content_type: str) -> list[dict]:
    ep_type = src["endpoint"]["type"]
    schema = src.get("schema", {})

    # gzip magic 0x1f 0x8b — used by e.g. gharchive .json.gz files. Decompress
    # transparently before dispatching.
    if blob[:2] == b"\x1f\x8b":
        import gzip
        blob = gzip.decompress(blob)

    if ep_type == "ndjson_minute_counts":
        return _records_from_ndjson_minute_counts(blob)
    if ep_type == "rest_json":
        data = json.loads(blob)
        # SDMX-JSON detection: response with both `dataSets` and `structure` needs
        # a per-dimension-index decoder (OECD, Eurostat, ECB, IMF).
        if isinstance(data, dict) and "dataSets" in data and "structure" in data:
            return _records_from_sdmx_json(data, schema)
        return _records_from_json(data, schema)
    if ep_type == "rest_csv":
        text = blob.decode("utf-8", errors="replace")
        # Some "csv" sources actually return zipped XML (CAISO).
        if blob[:2] == b"PK":
            return _records_from_zip_xml(blob, schema)
        return _records_from_csv(text, schema)
    if ep_type in ("rss",):
        return _records_from_rss(blob.decode("utf-8", errors="replace"), schema)
    if ep_type == "rest_xml":
        return _records_from_xml(blob, schema)
    if ep_type == "html_table":
        return _records_from_html_table(blob, schema, src["endpoint"])
    if ep_type == "rest_xlsx":
        return _records_from_xlsx(blob, schema)
    if ep_type == "s3":
        # Treated as a CSV/JSON over HTTPS
        if "json" in content_type:
            return _records_from_json(json.loads(blob), schema)
        return _records_from_csv(blob.decode("utf-8", errors="replace"), schema)
    raise NotImplementedError(f"endpoint.type {ep_type!r} not handled")


# ──────────────────────────────────────────────────────────────────────────
# New parsers: XML, HTML table, XLSX, SDMX-JSON
# ──────────────────────────────────────────────────────────────────────────


def _records_from_ndjson_minute_counts(blob: bytes) -> list[dict]:
    """Aggregate an NDJSON event stream (GH Archive hourly files) into
    per-minute event counts. Events are one JSON object per line; we extract
    `created_at` with a regex instead of json-decoding every event because the
    decompressed hourly files run to hundreds of MB."""
    from collections import Counter
    counts = Counter(
        m.group(1)
        for m in re.finditer(rb'"created_at":"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})', blob)
    )
    return [
        {"timestamp": minute.decode() + ":00Z", "value": n}
        for minute, n in sorted(counts.items())
    ]


def _records_from_xml(blob: bytes, schema: dict) -> list[dict]:
    """Parse XML → list of dicts. Schema keys give slash-separated element paths.

    Example schema for FAA NASSTATUS (returns <AIRPORT_STATUS_INFORMATION>...):
        timestamp_field: AIRPORT_STATUS_INFORMATION/Update_Time
        value_field: AIRPORT_STATUS_INFORMATION/Delay_type/ARPT/Delay/Min
    """
    import xml.etree.ElementTree as ET
    root = ET.fromstring(blob)
    ts_path = schema.get("timestamp_field", "")
    val_paths = schema.get("value_field", [])
    if isinstance(val_paths, str):
        val_paths = [val_paths]

    def _find_all(node, path: str) -> list:
        # xpath-lite: consume the root name if it matches, then use the tail as a
        # relative path.
        parts = [p for p in path.split("/") if p]
        if parts and node.tag == parts[0]:
            parts = parts[1:]
        return list(node.iter(parts[-1])) if parts else []

    def _text(node, path: str) -> Optional[str]:
        elts = _find_all(node, path)
        return elts[0].text if elts else None

    ts_val = _text(root, ts_path) if ts_path else None
    records: list[dict] = []
    if val_paths:
        # Iterate value elements as siblings; parallel-align by index.
        value_elts = _find_all(root, val_paths[0])
        for elt in value_elts:
            rec = {"timestamp": ts_val}
            try:
                rec["value"] = float(elt.text) if elt.text else None
            except (ValueError, TypeError):
                rec["value"] = elt.text
            records.append(rec)
    if not records:
        # No value elts found → single-row snapshot at the timestamp.
        records = [{"timestamp": ts_val, "value": None}]
    return records


def _records_from_html_table(blob: bytes, schema: dict, endpoint: dict) -> list[dict]:
    """Parse an HTML page → rows from a table matching endpoint.table_selector.

    Schema:
        timestamp_field: <column name in the header row>
        value_field: <column name in the header row>
    Endpoint (optional):
        table_selector: CSS selector for the target table (default: first table)
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(blob, "lxml")
    selector = endpoint.get("table_selector", "table")
    table = soup.select_one(selector)
    if table is None:
        return []
    rows = table.find_all("tr")
    if not rows:
        return []
    headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
    ts_col = schema.get("timestamp_field", headers[0] if headers else "")
    val_col = schema.get("value_field", headers[1] if len(headers) > 1 else "")
    if isinstance(val_col, list):
        val_col = val_col[0]
    records: list[dict] = []
    for tr in rows[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if len(cells) < len(headers):
            continue
        row = dict(zip(headers, cells))
        try:
            val = float(row.get(val_col, "").replace(",", ""))
        except (ValueError, AttributeError):
            val = row.get(val_col)
        records.append({"timestamp": row.get(ts_col), "value": val})
    return records


def _records_from_xlsx(blob: bytes, schema: dict) -> list[dict]:
    """Parse .xlsx → rows using openpyxl. Reads the first sheet's first two
    columns as (timestamp, value) unless schema overrides column names.
    """
    import io
    from openpyxl import load_workbook
    wb = load_workbook(io.BytesIO(blob), read_only=True, data_only=True)
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)
    header = next(rows_iter, None)
    if not header:
        return []
    ts_col = schema.get("timestamp_field", header[0])
    val_col = schema.get("value_field", header[1] if len(header) > 1 else None)
    if isinstance(val_col, list):
        val_col = val_col[0]
    ts_i = header.index(ts_col) if ts_col in header else 0
    val_i = header.index(val_col) if val_col in header else 1
    records: list[dict] = []
    for row in rows_iter:
        if row is None or all(c is None for c in row):
            continue
        ts = row[ts_i] if ts_i < len(row) else None
        val = row[val_i] if val_i < len(row) else None
        try:
            val = float(val) if val is not None else None
        except (ValueError, TypeError):
            pass
        records.append({"timestamp": str(ts) if ts is not None else None, "value": val})
    return records


def _records_from_sdmx_json(data: dict, schema: dict) -> list[dict]:
    """Decode SDMX-JSON → flat rows. Used by OECD, Eurostat, ECB, IMF.

    SDMX-JSON stores observations under `dataSets[0].series` keyed by
    dimension-index strings (e.g. "0:2:1"), with each observation as
    `{"time_idx": [value, ...]}`. The time-period lookup lives in
    `structure.dimensions.observation`.
    """
    if "dataSets" not in data or "structure" not in data:
        return []
    ds = data["dataSets"][0] if data["dataSets"] else {}
    struct = data["structure"]

    # Time labels: structure.dimensions.observation is a list; the entry with
    # id == "TIME_PERIOD" gives the labels.
    obs_dims = struct.get("dimensions", {}).get("observation", [])
    time_labels: list[str] = []
    for d in obs_dims:
        if d.get("id") == "TIME_PERIOD":
            time_labels = [v.get("id", "") for v in d.get("values", [])]
            break

    records: list[dict] = []
    series = ds.get("series", {})
    if series:
        for _key, s in series.items():
            for obs_key, val in (s.get("observations") or {}).items():
                try:
                    idx = int(obs_key.split(":")[0])
                except (ValueError, IndexError):
                    continue
                t = time_labels[idx] if 0 <= idx < len(time_labels) else str(idx)
                v = val[0] if isinstance(val, list) and val else val
                records.append({"timestamp": t, "value": v})
    else:
        # Some SDMX-JSON responses put observations flat on dataSets[0].observations.
        for obs_key, val in (ds.get("observations") or {}).items():
            try:
                idx = int(obs_key.split(":")[-1])
            except (ValueError, IndexError):
                continue
            t = time_labels[idx] if 0 <= idx < len(time_labels) else str(idx)
            v = val[0] if isinstance(val, list) and val else val
            records.append({"timestamp": t, "value": v})
    return records


def run_one(sid: str, dry_run: bool = False) -> int:
    src = get_source(sid)
    if src.get("disabled"):
        log.info("skip %s (disabled: %s)", sid, src.get("disabled_reason", "no reason given"))
        return 0
    log.info("fetch %s -> %s", sid, src["endpoint"]["url"][:120])

    panel = src.get("panel") or [None]   # list of dicts or [None] for single-poll
    total_written = 0
    for panel_row in panel:
        try:
            blob, ct = fetch_payload(src, panel_row=panel_row)
        except Exception as e:                              # noqa: BLE001
            log.error("fetch failed for %s%s: %s", sid, f" [{panel_row}]" if panel_row else "", e)
            log_error(sid, f"FETCH {e} {panel_row or ''}")
            continue
        try:
            rows = parse_payload(src, blob, ct)
            # `panel_concat: true` marks a panel that paginates ONE series over
            # time (e.g. one CSV per calendar year) rather than enumerating
            # distinct series — skip the `_panel_` tagging so the chunks merge.
            if panel_row and not src.get("panel_concat"):
                for r in rows:
                    r.update({"_panel_" + k: v for k, v in panel_row.items()})
        except Exception as e:                              # noqa: BLE001
            log.error("parse failed for %s%s: %s", sid, f" [{panel_row}]" if panel_row else "", e)
            log_error(sid, f"PARSE {e} {panel_row or ''}")
            continue
        n = write_records(sid, rows, dry_run=dry_run)
        log.info("%s%s: wrote %d rows (raw %d)", sid, f" [{panel_row}]" if panel_row else "", n, len(rows))
        total_written += n
    return total_written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", help="single source id from sources.yaml")
    ap.add_argument("--all", action="store_true", help="run every source")
    ap.add_argument("--domain", help="only run sources in this domain")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch + parse but do not write parquet")
    ap.add_argument("--list", action="store_true", help="print all source ids and exit")
    args = ap.parse_args()

    sources = load_sources()
    if args.list:
        for s in sources:
            flag = "  [DISABLED]" if s.get("disabled") else ""
            print(f"{s['id']:35s}  {s['domain']:13s}  {s['frequency']}{flag}")
        return 0

    targets: Iterable[str]
    if args.id:
        targets = [args.id]
    elif args.all:
        targets = [s["id"] for s in sources]
    elif args.domain:
        targets = [s["id"] for s in sources if s["domain"] == args.domain]
    else:
        ap.print_help()
        return 1

    failed = 0
    for sid in targets:
        try:
            run_one(sid, dry_run=args.dry_run)
        except Exception as e:                              # noqa: BLE001
            log.exception("%s: unrecoverable error", sid)
            log_error(sid, f"UNCAUGHT {e}")
            failed += 1
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
