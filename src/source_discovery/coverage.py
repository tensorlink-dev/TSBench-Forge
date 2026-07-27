"""Deterministic coverage analysis over the existing catalog (no LLM, no network).

Loads ``sources.yaml``, normalises each entry into the agent's ``CURRENT_SOURCES``
schema, and builds the domain × cadence coverage matrix the agent maps in Phase 1
and the gap ranking it uses in Phase 2. All pure functions of the on-disk catalog.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from . import config, duration


# Map the catalog's `pretraining_novelty` field onto the agent's risk scale.
_NOVELTY_TO_RISK = {
    "clean": "low",
    "partial": "medium",
    "contaminated": "high",
    "unknown": "medium",
}


def _access_method(entry: dict) -> str:
    """Infer the agent's access_method from the catalog endpoint + cadence."""
    ep = entry.get("endpoint", {}) or {}
    auth = str(ep.get("auth") or "none").lower()
    cadence = str(entry.get("update_cadence_observed") or "").lower()
    if any(w in cadence for w in ("realtime", "real-time", "stream", "every ~", "every 1", "second")):
        return "live_stream"
    if auth.startswith("env:") or auth in ("key", "token", "oauth"):
        return "licensed"
    etype = str(ep.get("type") or "").lower()
    if "bulk" in etype or "zip" in etype or "xlsx" in etype:
        return "bulk_download"
    return "open_api"


def load_registry(catalog_path: str | Path,
                  include_disabled: bool = False) -> list[dict]:
    """Load ``sources.yaml`` and normalise entries to the CURRENT_SOURCES schema.

    Each returned dict carries the fields the agent's prompt documents plus the
    raw ``id`` / ``dgp_class`` / ``cadence`` for the coverage matrix.
    """
    import yaml

    with open(catalog_path) as f:
        raw = yaml.safe_load(f) or []

    out: list[dict] = []
    for e in raw:
        if not e.get("id"):
            continue
        # A disabled source contributes no data; counting it as coverage hides
        # the hole it left behind.
        if e.get("disabled") and not include_disabled:
            continue
        freq = e.get("frequency", "")
        novelty = str(e.get("pretraining_novelty") or "unknown").lower()
        ep = e.get("endpoint", {}) or {}
        out.append(
            {
                "id": e["id"],
                "name": e.get("name", e["id"]),
                "domain": e.get("domain", "?"),
                "dgp_class": e.get("dgp_class", "?"),
                "frequency": freq,
                "cadence": band_for(freq),
                "access_method": _access_method(e),
                "url_or_endpoint": ep.get("url", ""),
                "license": e.get("license", "?"),
                "series_count": len(e.get("panel", []) or []) or 1,
                "typical_length": e.get("history_available", "?"),
                "first_available_date": _first_available(e),
                "contamination_risk": _NOVELTY_TO_RISK.get(novelty, "medium"),
                "archetypes": e.get("archetypes", []) or [],
            }
        )
    return out


def _first_available(entry: dict) -> str:
    """Best-effort earliest-data date; falls back to history horizon or 'unknown'."""
    ver = entry.get("verification", {}) or {}
    lv = ver.get("last_verified")
    if isinstance(lv, str) and lv[:4].isdigit():
        return lv
    return str(entry.get("history_available") or "unknown")


def band_for(freq: str) -> str:
    """Cadence band for any ISO-8601 frequency, not just the pre-listed ones.

    ``config.FREQ_BAND`` only names the frequencies the catalog happened to use
    when it was written, so anything newer (PT10S, PT4M, P3D, P30D, P2W, P3M)
    silently fell into "irregular" and distorted the matrix. Parse the duration
    and bucket it; the literal string "irregular" stays irregular.
    """
    if freq in config.FREQ_BAND:
        return config.FREQ_BAND[freq]
    seconds = _period_seconds(freq)
    if seconds is None:
        return "irregular"
    if seconds < 60:
        return "sub-min"
    # Boundaries follow config.FREQ_BAND's existing convention, where the
    # 15-minute grid MTU counts as few-min rather than half-hour; the fallback
    # must not invent a second, conflicting one.
    if seconds <= 900:
        return "few-min"
    if seconds <= 1800:
        return "half-hour"
    if seconds <= 10800:
        return "hourly"
    if seconds <= 3 * 86400:
        return "daily"
    if seconds <= 14 * 86400:
        return "weekly"
    if seconds <= 45 * 86400:
        return "monthly"
    if seconds <= 200 * 86400:
        return "quarterly"
    return "yearly"


def _period_seconds(freq: str) -> int | None:
    return duration.period_seconds(freq)


def host_of(entry: dict) -> str:
    from urllib.parse import urlparse
    return urlparse(entry.get("url_or_endpoint") or "").netloc


def diversity(registry: list[dict], cap: int = 5) -> dict:
    """Per-domain provider diversity, and a host-capped "effective" count.

    A domain whose sources nearly all come from one API is not as covered as its
    headline number suggests — 53 of energy's sources are per-country variants
    of one host. ``effective`` counts at most ``cap`` sources per host, which is
    the number to steer by when deciding where the next wave should look.
    """
    out: dict[str, dict] = {}
    for domain in sorted({s["domain"] for s in registry}):
        rows = [s for s in registry if s["domain"] == domain]
        hosts = Counter(host_of(s) for s in rows)
        top_host, top_n = hosts.most_common(1)[0] if hosts else ("", 0)
        out[domain] = {
            "sources": len(rows),
            "hosts": len(hosts),
            "top_host": top_host,
            "top_host_share": round(top_n / len(rows), 3) if rows else 0.0,
            "effective": sum(min(n, cap) for n in hosts.values()),
        }
    return out


def coverage_matrix(registry: list[dict]) -> dict[tuple[str, str], int]:
    """Count sources per (domain, cadence-band) cell."""
    m: dict[tuple[str, str], int] = defaultdict(int)
    for s in registry:
        m[(s["domain"], s["cadence"])] += 1
    return dict(m)


def _target_for(cell: tuple[str, str]) -> int:
    _, band = cell
    if band in config.HIGH_VALUE_BANDS:
        return config.TARGET_PER_HIGH_VALUE_CELL
    return config.TARGET_PER_CELL


def gap_cells(registry: list[dict]) -> list[dict]:
    """Cells below their target, ranked by deficit then high-value-band priority.

    Returns dicts ``{domain, cadence, have, target, deficit, high_value}`` sorted
    worst-first, so Phase-3 proposals can target the biggest holes.
    """
    m = coverage_matrix(registry)
    gaps: list[dict] = []
    for domain in config.DOMAINS:
        for band in config.CADENCE_BANDS:
            cell = (domain, band)
            have = m.get(cell, 0)
            target = _target_for(cell)
            if have < target:
                gaps.append(
                    {
                        "domain": domain,
                        "cadence": band,
                        "have": have,
                        "target": target,
                        "deficit": target - have,
                        "high_value": band in config.HIGH_VALUE_BANDS,
                    }
                )
    gaps.sort(key=lambda g: (-g["deficit"], not g["high_value"], g["domain"]))
    return gaps


def summarize(registry: list[dict]) -> dict:
    """A compact, human- and prompt-readable coverage summary."""
    by_domain = Counter(s["domain"] for s in registry)
    by_cadence = Counter(s["cadence"] for s in registry)
    by_risk = Counter(s["contamination_risk"] for s in registry)
    by_access = Counter(s["access_method"] for s in registry)
    return {
        "n_sources": len(registry),
        "by_domain": dict(by_domain.most_common()),
        "by_cadence": dict(by_cadence.most_common()),
        "by_contamination_risk": dict(by_risk.most_common()),
        "by_access_method": dict(by_access.most_common()),
        "by_provider_diversity": diversity(registry),
        "gap_cells": gap_cells(registry),
    }


def render_matrix(registry: list[dict], cap: int = 5) -> str:
    """The domain × cadence grid as text, with provider concentration alongside.

    Printed by ``--coverage`` (on stderr, so the summary JSON keeps stdout to
    itself) because the balance question is asked far more often than it is
    scripted, and a JSON blob is the wrong shape for reading a matrix.
    """
    m = coverage_matrix(registry)
    bands = list(config.CADENCE_BANDS)
    domains = sorted({s["domain"] for s in registry},
                     key=lambda d: -sum(m.get((d, b), 0) for b in bands))
    div = diversity(registry, cap=cap)
    w = 10
    head = f"{'domain':14s}" + "".join(f"{b:>{w}s}" for b in bands)
    head += f"{'TOTAL':>8s}{'EFFCTV':>8s}{'HOSTS':>7s}{'TOP HOST':>9s}"
    lines = [head, "-" * len(head)]
    for d in domains:
        row = f"{d:14s}" + "".join(f"{m.get((d, b), 0):>{w}d}" for b in bands)
        info = div[d]
        lines.append(row + f"{info['sources']:>8d}{info['effective']:>8d}"
                           f"{info['hosts']:>7d}{info['top_host_share']:>8.0%} ")
    lines.append("-" * len(head))
    totals = f"{'TOTAL':14s}" + "".join(
        f"{sum(m.get((d, b), 0) for d in domains):>{w}d}" for b in bands)
    lines.append(totals + f"{len(registry):>8d}"
                          f"{sum(v['effective'] for v in div.values()):>8d}")
    empty = [(d, b) for d in domains for b in bands if not m.get((d, b))]
    if empty:
        lines.append("empty cells: " + ", ".join(f"{d}/{b}" for d, b in empty))
    lines.append(f"(EFFCTV counts at most {cap} sources per host — a domain built "
                 f"on one API is less covered than its total suggests)")
    return "\n".join(lines)
