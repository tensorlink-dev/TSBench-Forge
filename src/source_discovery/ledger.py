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
            # Keep a few of the OTHER names seen for this host. The key is
            # domain|host, so every dataset on a host collapses into one entry
            # and only the first name was ever kept — which is why the prompt
            # could say "api.fda.gov — FDA MAUDE adverse event reports" while a
            # model happily proposed "FDA Drug Shortage Reports" 57 times. The
            # aliases give it the vocabulary to recognise its own idea.
            name = str(c.get("name", ""))[:80]
            if name and name != e.get("name"):
                aka = e.setdefault("also_proposed", [])
                if name not in aka and len(aka) < 4:
                    aka.append(name)
            if e.get("status") not in _STICKY and r.verdict == "reject":
                e["status"] = "rejected"
    save(path, ledger)
    return new


def prompt_block(ledger: dict[str, dict], limit: int = 2000) -> list[str]:
    """Compact ALREADY_PROPOSED list for the agent prompt.

    One terse string per entry — the block exists to be *checked against*, not
    reasoned about, and a large structured block measurably drowns reasoning
    models in their own deliberation budget.

    Each line carries the host **and** the name, because the key is
    ``domain|netloc`` and a host alone cannot be matched against what a model
    actually writes. Measured 2026-08-13: the block said ``api.fda.gov
    [wired]`` and the model proposed "FDA Drug Shortage Reports" — a different
    dataset on a listed host — which the vet then rejected as seen 57x. Both
    models under test spent their entire budget re-proposing such entries.

    The limit is generous for the same reason: the block is ~20kB against a
    million-token window, so hiding half the ledger to save tokens bought
    nothing and let the long tail be re-proposed indefinitely.
    """
    entries = sorted(ledger.values(), key=lambda e: -e.get("times_proposed", 1))[:limit]
    out = []
    for e in entries:
        host = e["key"].split("|", 1)[1]
        names = [n for n in [str(e.get("name") or "").strip(),
                             *(e.get("also_proposed") or [])] if n and n != host]
        # "e.g." is load-bearing: the key is domain|host, so the WHOLE HOST is
        # excluded, not just the dataset named. Rendering a bare name invited
        # exactly the mistake this block exists to prevent — a second dataset
        # on a host already listed.
        label = host if not names else f"{host} — e.g. {'; '.join(names[:3])}"
        out.append(f"{label} [{e['status']}]")
    return out
