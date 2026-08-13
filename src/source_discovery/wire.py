"""wire.py — the probe-verify-and-wire step between discovery and the catalog.

This closes the pipeline's missing middle. Discovery (llm.py) proposes, vet.py
pre-filters metadata, but until now a human/agent had to hand-convert accepted
candidates into ``sources.yaml`` entries and verify them ad hoc. This module
takes candidate blocks with *fully specified* endpoints, replays each one
through the REAL scraper (``fetch_payload`` + ``parse_payload``), gates on the
same admission bar used throughout the catalog build, and — only with
``apply=True`` — appends the survivors to ``sources.yaml``, registers them in
``cron.yaml``, and settles their ledger statuses.

Input format (one JSON array, two shapes accepted per item):

1. grind block (what endpoint-discovery agents emit)::

     {"candidate_name": "...", "wireable": true, "yaml_block": "- id: ...",
      "cron_cadence": "PT5M", "reason": "..."}

   ``wireable: false`` items are settled straight to the ledger — ``key-gated``
   if the reason mentions keys/tokens/registration, else ``rejected``.

2. a plain catalog entry dict (``{"id": ..., "endpoint": {...}, ...}``).

Verification gate (same as the 500-source build): >=20 rows AND >=20 distinct
timestamps AND >=50% rows numeric, OR >=20 panel entities with >=20 numeric
rows. Panel sources are fetched per panel row, exactly as the scraper does.

Safety rails learned the hard way (see repo memory):
* yaml_block is HTML-unescaped SAFELY (``&amp;`` family only) — never
  ``html.unescape``, which mangles ``&timespan`` via the legacy ``&times``
  entity.
* all writes are atomic (temp + ``os.replace``);
* after applying, the catalog is re-parsed and the entry count must equal
  before + wired, with zero duplicate ids — else the write is rolled back.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import yaml

from . import config
from .ledger import ledger_path as _ledger_path

# Cadences that have their own cron group; anything slower rides the daily one.
CRON_CADENCES = ("PT1M", "PT2M30S", "PT5M", "PT15M", "PT30M", "PT1H", "P1D")

# A non-wireable reason mentioning any of these ⇒ key-gated, not rejected.
KEY_MARKERS = (
    "needs key", "api key", "api_key", "apikey", "token", "oauth", "credential",
    "x-api-key", "secret", "contact-email", "contact email", "user-agent",
    "user agent", "register", "account", "sign up", "signup", "subscription",
)

_SAFE_ENTITIES = (
    ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
    ("&quot;", '"'), ("&#39;", "'"), ("&#x27;", "'"),
)


def safe_unescape(s: str) -> str:
    """Undo the HTML escaping agents apply to yaml_block. ONLY the six safe
    entities — html.unescape would corrupt URLs like ``...&timespan=`` through
    the semicolon-less legacy ``&times`` entity."""
    for ent, ch in _SAFE_ENTITIES:
        s = s.replace(ent, ch)
    return s


def _atomic_write(path: str | Path, text: str) -> None:
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _scraper():
    """Import the real scraper (src/sources is not a package)."""
    sources_dir = str(Path(__file__).resolve().parent.parent / "sources")
    if sources_dir not in sys.path:
        sys.path.insert(0, sources_dir)
    import scraper  # noqa: PLC0415
    return scraper


def _url_key(entry: dict) -> str:
    """Duplicate-detection key for an endpoint. POST APIs (GraphQL, StatCan
    WDS, PxWeb) multiplex many series over ONE url — the body is part of the
    identity, else the second source on a host's POST endpoint is falsely
    deduped.

    GET payloads multiplex too. energy-charts answers one `public_power`
    URL with twenty parallel production-type arrays; a national grid's solar
    output and its nuclear output are different processes, not one source
    entered twice. So the selected series — value_field, plus the timestamp
    path and any panel pin that picks a lane out of the payload — belongs in
    the identity for the same reason the POST body does. Two entries that
    agree on url AND selection are still duplicates and still collapse."""
    ep = entry.get("endpoint") or {}
    url = ep.get("url", "")
    body = ep.get("body") or ep.get("body_form") or ep.get("query")
    if body is not None:
        url += "#" + json.dumps(body, sort_keys=True, default=str)
    sch = entry.get("schema") or {}
    sel = {k: sch.get(k) for k in ("value_field", "timestamp_field", "panel_field",
                                   "where", "sheet", "record_path")
           if sch.get(k) is not None}
    if sel:
        url += "@" + json.dumps(sel, sort_keys=True, default=str)
    return url


def _is_num(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


@dataclasses.dataclass
class VerifyResult:
    ok: bool
    rows: int = 0
    distinct_ts: int = 0
    numeric_rows: int = 0
    panel_entities: int = 0
    newest: Optional[str] = None
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def verify_entry(entry: dict, scraper_mod=None) -> VerifyResult:
    """Replay one catalog entry through the real scraper and apply the gate.

    The gate checks volume (rows / distinct timestamps / numeric) AND
    freshness: the newest parsed observation must be inside the audit's
    cadence-scaled staleness limit. Without the freshness leg, sources that
    page oldest-first pass verification while fetching a frozen historical
    window — six such landed in the catalog before this check existed."""
    sc = scraper_mod or _scraper()
    try:
        panel = entry.get("panel")
        if isinstance(panel, list) and panel:
            recs: list[dict] = []
            for prow in panel:
                blob, ctype = sc.fetch_payload(entry, prow)
                recs += sc.parse_payload(entry, blob, ctype)
        else:
            blob, ctype = sc.fetch_payload(entry)
            recs = sc.parse_payload(entry, blob, ctype)
    except Exception as exc:  # noqa: BLE001 — any fetch/parse fault is a verdict
        return VerifyResult(ok=False, error=f"{type(exc).__name__}: {str(exc)[:160]}")

    ts = {r.get("timestamp") for r in recs}
    ents = {r[k] for r in recs for k in r if k.startswith("_panel")}

    def _row_numeric(r: dict) -> bool:
        return any(_is_num(v) for k, v in r.items()
                   if k != "timestamp" and not k.startswith("_"))

    nn = sum(1 for r in recs if _row_numeric(r))
    ok = ((len(recs) >= 20 and len(ts) >= 20 and nn >= 0.5 * len(recs))
          or (len(ents) >= 20 and nn >= 20))

    from . import audit  # noqa: PLC0415 — avoid import cycle at module load
    newest = None
    for t in ts:
        got = audit.parse_ts(t)
        if got and (newest is None or got > newest):
            newest = got
    if ok:
        if newest is None:
            ok = False
            err = "no parsable timestamps — cannot confirm freshness"
        else:
            import datetime as dt  # noqa: PLC0415
            age = dt.datetime.now(dt.timezone.utc) - newest
            slack = entry.get("audit_slack_days")  # declared publication lag
            limit = (dt.timedelta(days=float(slack)) if slack
                     else audit.staleness_threshold(entry.get("frequency", "")))
            if age > limit:
                ok = False
                err = (f"stale at wire time: newest observation {newest.isoformat()} "
                       f"is {age.days}d old vs {limit.days}d limit — likely an "
                       f"oldest-first page or a fixed historical window")
            else:
                err = None
    else:
        err = None
    return VerifyResult(ok=ok, rows=len(recs), distinct_ts=len(ts),
                        numeric_rows=nn, panel_entities=len(ents),
                        newest=newest.isoformat() if newest else None, error=err)


def _parse_candidate(item: dict) -> tuple[str, Optional[dict], str, Optional[str]]:
    """Normalise one input item → (name, entry-or-None, reason, cadence)."""
    if "yaml_block" in item or "candidate_name" in item:
        name = str(item.get("candidate_name") or "?")
        reason = str(item.get("reason") or "")
        cadence = item.get("cron_cadence")
        if not item.get("wireable") or not item.get("yaml_block"):
            return name, None, reason, cadence
        loaded = yaml.safe_load(safe_unescape(item["yaml_block"]))
        entry = loaded[0] if isinstance(loaded, list) else loaded
        if not isinstance(entry, dict):
            raise ValueError("yaml_block did not parse to a mapping")
        return name, entry, reason, cadence
    if "id" in item and "endpoint" in item:
        return str(item["id"]), item, "", None
    raise ValueError(f"unrecognised candidate shape: {sorted(item)[:6]}")


def _entry_yaml(entry: dict) -> str:
    body = yaml.safe_dump(entry, sort_keys=False, default_flow_style=False,
                          width=100, allow_unicode=True)
    return "- " + body.replace("\n", "\n  ").rstrip() + "\n"


def _cron_group(cron: dict, every: str) -> dict:
    for g in cron.get("schedules", []):
        if g.get("every") == every:
            return g
    g = {"every": every, "sources": []}
    cron.setdefault("schedules", []).append(g)
    return g


def _settle_ledger(ledger_file: Path, wired: set[str], key_gated: set[str],
                   rejected: set[str]) -> dict[str, int]:
    """Move matching *proposed* ledger entries to their terminal status.

    Matches by entry name. Sticky statuses (wired/key-gated/retired) are never
    downgraded. Uses plain json.dumps with default ensure_ascii=True — em-dash
    names must stay escaped or every ledger line rewrites."""
    if not ledger_file.exists():
        return {}
    entries = json.loads(ledger_file.read_text())
    changed: dict[str, int] = {}
    for e in entries:
        if e.get("status") != "proposed":
            continue
        name = e.get("name")
        new = ("wired" if name in wired else
               "key-gated" if name in key_gated else
               "rejected" if name in rejected else None)
        if new:
            e["status"] = new
            changed[new] = changed.get(new, 0) + 1
    _atomic_write(ledger_file, json.dumps(entries, indent=2) + "\n")
    return changed


def wire_batch(candidates_file: str | Path, catalog_path: str | Path,
               label: str, apply: bool = False,
               cron_path: str | Path | None = None) -> dict:
    """Verify a candidate batch against the live scraper; optionally wire it.

    Returns a JSON-able report. With ``apply=False`` (default) nothing is
    written — the report shows exactly what *would* be wired.
    """
    sc = _scraper()
    catalog_path = Path(catalog_path)
    cron_path = Path(cron_path) if cron_path else catalog_path.parent / "cron.yaml"

    sources = yaml.safe_load(catalog_path.read_text())
    existing_ids = {s["id"] for s in sources}
    existing_urls = {_url_key(s) for s in sources}

    items = json.loads(Path(candidates_file).read_text())
    to_wire: list[tuple[str, dict, Optional[str]]] = []
    wired_names: set[str] = set()
    key_names: set[str] = set()
    rej_names: set[str] = set()
    detail: list[dict] = []

    for item in items:
        try:
            name, entry, reason, cadence = _parse_candidate(item)
        except Exception as exc:  # noqa: BLE001
            detail.append({"name": item.get("candidate_name") or item.get("id"),
                           "verdict": "malformed", "error": str(exc)[:160]})
            rej_names.add(str(item.get("candidate_name") or ""))
            continue
        if entry is None:
            low = reason.lower()
            gated = any(m in low for m in KEY_MARKERS)
            (key_names if gated else rej_names).add(name)
            detail.append({"name": name,
                           "verdict": "key-gated" if gated else "rejected",
                           "reason": reason[:160]})
            continue
        sid = entry.get("id")
        url = _url_key(entry)
        if sid in existing_ids or (url and url in existing_urls):
            wired_names.add(name)
            detail.append({"name": name, "id": sid, "verdict": "duplicate"})
            continue
        if entry.get("domain") not in config.DOMAINS:
            rej_names.add(name)
            detail.append({"name": name, "id": sid, "verdict": "bad-domain",
                           "domain": entry.get("domain")})
            continue
        res = verify_entry(entry, sc)
        detail.append({"name": name, "id": sid,
                       "verdict": "wire" if res.ok else "verify-fail",
                       **res.as_dict()})
        if res.ok:
            to_wire.append((name, entry, cadence))
            existing_ids.add(sid)
            existing_urls.add(url)
        else:
            rej_names.add(name)

    report: dict = {
        "label": label,
        "applied": False,
        "wired_ids": [e["id"] for _, e, _ in to_wire],
        "counts": {"wire": len(to_wire), "duplicate": len(wired_names),
                   "key_gated": len(key_names), "rejected": len(rej_names)},
        "detail": detail,
    }
    if not apply or not to_wire:
        if apply:  # nothing passed, but ledger settlement still applies
            report["ledger_changes"] = _settle_ledger(
                _ledger_path(catalog_path), wired_names, key_names, rej_names)
            report["applied"] = True
        return report

    # -- apply: sources.yaml ------------------------------------------------
    before = len(sources)
    text = catalog_path.read_text()
    if not text.endswith("\n"):
        text += "\n"
    blocks = "".join(_entry_yaml(e) for _, e, _ in to_wire)
    _atomic_write(catalog_path, text + f"\n# -- wired batch: {label} --\n" + blocks)

    # Post-write validation: reparse; count must be before+wired, ids unique.
    reparsed = yaml.safe_load(catalog_path.read_text())
    ids = [s["id"] for s in reparsed]
    if len(reparsed) != before + len(to_wire) or len(ids) != len(set(ids)):
        _atomic_write(catalog_path, text)  # roll back
        raise RuntimeError(
            f"catalog validation failed after append ({before}+{len(to_wire)} "
            f"expected, got {len(reparsed)}, dup_ids={len(ids) - len(set(ids))}) "
            "— rolled back")

    # -- cron.yaml ------------------------------------------------------------
    cron = yaml.safe_load(cron_path.read_text())
    for _, e, cadence in to_wire:
        freq = cadence if cadence in CRON_CADENCES else e.get("frequency", "P1D")
        every = freq if freq in CRON_CADENCES else "P1D"
        grp = _cron_group(cron, every)
        if e["id"] not in grp["sources"]:
            grp["sources"].append(e["id"])
    _atomic_write(cron_path, yaml.safe_dump(cron, sort_keys=False,
                                            default_flow_style=False, width=100))

    # -- ledger ---------------------------------------------------------------
    wired_names |= {n for n, _, _ in to_wire}
    report["ledger_changes"] = _settle_ledger(
        _ledger_path(catalog_path), wired_names, key_names, rej_names)
    report["applied"] = True
    report["catalog_entries"] = len(reparsed)
    return report
