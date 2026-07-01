"""Deterministic coverage analysis over the existing catalog (no LLM, no network).

Loads ``sources.yaml``, normalises each entry into the agent's ``CURRENT_SOURCES``
schema, and builds the domain × cadence coverage matrix the agent maps in Phase 1
and the gap ranking it uses in Phase 2. All pure functions of the on-disk catalog.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from . import config


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


def load_registry(catalog_path: str | Path) -> list[dict]:
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
                "cadence": config.FREQ_BAND.get(freq, "irregular"),
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
        "gap_cells": gap_cells(registry),
    }
