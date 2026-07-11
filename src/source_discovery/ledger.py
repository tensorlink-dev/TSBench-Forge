"""Persistent proposal ledger — the agent's memory across discovery runs.

Without it every run starts blind and the model re-proposes its favourite
sources indefinitely (observed: Steam 55x, PurpleAir 38x across 100 rounds).
The ledger closes the loop three ways:

1. **Prompt**: ``build_inputs`` injects a compact ALREADY_PROPOSED list, and the
   system prompt instructs the model that re-proposals are wasted output.
2. **Vet**: anything whose (domain, host) is already on the ledger is
   hard-rejected, so repeats never reach a human reviewer twice.
3. **Persistence**: every full run upserts its vetted proposals back into the
   ledger (times_proposed increments; a human-set status like ``wired`` or
   ``key-gated`` is never downgraded by an automated run).

The ledger lives next to the run artifacts (``sources/discovery/``) and is
committed, so agent memory survives machines and branches.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

LEDGER_NAME = "proposal_ledger.json"

# Statuses a human (or the wiring pipeline) sets; automated runs never overwrite.
_STICKY = {"wired", "key-gated", "retired"}


def ledger_path(catalog_path: str | Path) -> Path:
    return Path(catalog_path).parent / "discovery" / LEDGER_NAME


def candidate_key(candidate: dict) -> str:
    url = str(candidate.get("url_or_endpoint") or "")
    if url and "://" not in url:
        # Scheme-less URLs parse to an empty netloc, collapsing the key to the
        # (unstable) name and letting re-proposals through. Same fix as vet._host.
        url = "//" + url
    host = urlparse(url).netloc
    return f"{candidate.get('domain', '?')}|{host or candidate.get('name', '?')}"


def load(path: str | Path) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    entries = json.loads(p.read_text())
    return {e["key"]: e for e in entries}


def save(path: str | Path, ledger: dict[str, dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(ledger.values(), key=lambda e: (-e.get("times_proposed", 1), e["key"]))
    p.write_text(json.dumps(ordered, indent=2) + "\n")


def update(path: str | Path, results, run_date: str) -> int:
    """Upsert vetted proposals; returns the number of NEW ledger entries."""
    ledger = load(path)
    new = 0
    for r in results:
        c = r.candidate
        key = candidate_key(c)
        e = ledger.get(key)
        if e is None:
            new += 1
            ledger[key] = {
                "key": key,
                "name": str(c.get("name", ""))[:80],
                "domain": c.get("domain"),
                "frequency": c.get("frequency"),
                "status": "rejected" if r.verdict == "reject" else "proposed",
                "first_proposed": run_date,
                "times_proposed": 1,
            }
        else:
            e["times_proposed"] = int(e.get("times_proposed", 1)) + 1
            if e.get("status") not in _STICKY and r.verdict == "reject":
                e["status"] = "rejected"
    save(path, ledger)
    return new


def prompt_block(ledger: dict[str, dict], limit: int = 500) -> list[str]:
    """Compact ALREADY_PROPOSED list for the agent prompt.

    One terse string per entry — the block exists to be *checked against*, not
    reasoned about, and a large structured block measurably drowns reasoning
    models in their own deliberation budget.
    """
    entries = sorted(ledger.values(), key=lambda e: -e.get("times_proposed", 1))[:limit]
    return [f"{e['key'].split('|', 1)[1]} [{e['status']}]" for e in entries]
