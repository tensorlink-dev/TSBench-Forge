"""Unattended daily source-finding, steered by where the catalog is thinnest.

The bulk sweeps in ``bulk.py`` each know how to mine one vein. This decides
*which* vein to run today and *which* of its finds to keep, so the catalog
grows toward balance instead of toward whatever happens to be easiest to
harvest — which, left alone, is web_cloudops: two unsteered waves took it from
14% to 40% of the catalog in two days.

Three steering decisions, in order:

1. **Domain** — rank by the host-capped ``effective`` count from
   ``coverage.diversity``, not by raw source count. A domain whose sources all
   come from one API is not as covered as its headline says; energy's 209
   sources are 145 effective. Weakest domains go first.

2. **Vein** — only run veins that can actually serve today's target domains.
   The keyword-driven sweeps (arcgis, ods, ckan) get their keyword list
   filtered to those domains; the single-remit veins (sdmx→econ_fin,
   gbfs→transport, misskey/ixp→web_cloudops) are only queued when their domain
   is targeted.

3. **Cadence** — sweeps cannot be asked for a granularity; the cadence is a
   property of the data, discovered late. So this over-collects and then keeps
   the candidates landing in the thinnest (domain × cadence) cells. That is the
   only point where granularity balance can be steered at all.

``config.gap_cells`` is deliberately not used: its floors (2 per cell, 3 for
high-value bands) were written for a catalog of tens of sources and are met
almost everywhere at 2.3k, so it reports nothing useful. Balance here is
relative — how thin a cell is against its domain's own spread — not a fixed
floor.

Everything is bounded: a wall-clock deadline, a per-vein deadline, and a cap on
how many finds may come from one host, because five sources from one API is the
opposite of what this is for.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Optional, Sequence

import yaml

from . import bulk, config, coverage

# Over-collect this many times the wire target, so cadence selection has
# something to choose between. Higher is better balance but slower.
OVERCOLLECT = 3
# No more than this many wired sources from one host in a single run.
PER_HOST_CAP = 2
DEFAULT_TARGET = 4
DEFAULT_MINUTES = 45
# A vein that has found nothing in this long is not going to; move on.
VEIN_MINUTES = 12

ALL_DOMAINS = tuple(config.DOMAINS)


class _DeadlineReached(Exception):
    """Raised out of a sweep's log callback to stop it at its budget."""


def _deadline_log(deadline: float):
    """A ``log`` callback that aborts its sweep once the budget is spent.

    The sweeps take no deadline argument, so without this the first slow vein
    eats the whole run: a measured ArcGIS pass ran 1060s against a 12-minute
    budget and no other vein got a turn. They do call ``log`` once per keyword
    and checkpoint after every find, so raising from here stops them promptly
    and the finds so far survive in the checkpoint file.
    """

    def _log(_message: str) -> None:
        if time.monotonic() >= deadline:
            raise _DeadlineReached()

    return _log


def _resume_checkpoint(path: Path) -> list[dict]:
    """Finds a sweep had banked before it was stopped."""
    try:
        blocks = json.loads(path.read_text())
    except Exception:  # noqa: BLE001 — no checkpoint just means nothing banked
        return []
    return blocks if isinstance(blocks, list) else []


class Vein:
    """One bulk sweep plus what it can be steered toward.

    ``keyworded`` veins accept a ``(keyword, domain, dgp_class)`` list and can
    be pointed at any domain. The rest serve a fixed remit and are only worth
    running when that remit is what is thin.
    """

    def __init__(self, name: str, fn: Callable, domains: Sequence[str],
                 keyworded: bool = False, kwargs: Optional[dict] = None):
        self.name = name
        self.fn = fn
        self.domains = tuple(domains)
        self.keyworded = keyworded
        self.kwargs = kwargs or {}

    def serves(self, domain: str) -> bool:
        return domain in self.domains


def _veins() -> tuple[Vein, ...]:
    """Vein registry, best-yielding first within each remit.

    Order matters: the first vein that can serve a target domain runs first, so
    put the ones measured to still be productive ahead of the thin ones.
    ERDDAP, PeerTube and the publisher sweeps are omitted -- each was measured
    empty (see the exhausted-veins notes), and re-running them daily would burn
    the budget before a productive vein got a turn.
    """
    return (
        Vein("arcgis", bulk.arcgis_sweep, ALL_DOMAINS, keyworded=True,
             kwargs={"host_cap": 1, "per_keyword": 400, "sleep_s": 0.2}),
        Vein("sdmx", bulk.sdmx_sweep, ("econ_fin",),
             kwargs={"host_cap": 2, "max_flows": 300}),
        Vein("gbfs", bulk.gbfs_sweep, ("transport",), kwargs={"host_cap": 1}),
        Vein("ods", bulk.ods_sweep, ALL_DOMAINS, keyworded=True,
             kwargs={"host_cap": 1, "per_keyword": 100}),
        Vein("ckan", bulk.ckan_sweep, ALL_DOMAINS, keyworded=True,
             kwargs={"per_portal": 2, "keywords_per_portal": 6}),
        Vein("pxweb", bulk.pxweb_sweep, ("econ_fin", "healthcare", "nature",
                                         "sales", "energy"),
             kwargs={"host_cap": 1}),
        Vein("misskey", bulk.misskey_sweep, ("web_cloudops",),
             kwargs={"host_cap": 1}),
        Vein("ixp", bulk.ixp_sweep, ("web_cloudops",), kwargs={"host_cap": 1}),
    )


def weakest_domains(catalog_path: str | Path, n: int = 3) -> list[dict]:
    """Domains ranked thinnest-first by host-capped effective count."""
    registry = coverage.load_registry(catalog_path)
    div = coverage.diversity(registry)
    rows = [{"domain": d, **v} for d, v in div.items()]
    for d in ALL_DOMAINS:
        if d not in div:
            # A domain with nothing at all is maximally thin, and would
            # otherwise be invisible to a ranking built from what exists.
            rows.append({"domain": d, "sources": 0, "hosts": 0,
                         "top_host": "", "top_host_share": 0.0, "effective": 0})
    rows.sort(key=lambda r: r["effective"])
    return rows[:n]


def cell_counts(catalog_path: str | Path) -> Counter:
    """Live source count per (domain, cadence-band)."""
    registry = coverage.load_registry(catalog_path)
    return Counter((s["domain"], s["cadence"]) for s in registry)


def _block_domain_band(block: dict) -> tuple[str, str]:
    """Read (domain, cadence-band) off a candidate's yaml_block."""
    try:
        entry = yaml.safe_load(block.get("yaml_block") or "")[0]
    except Exception:  # noqa: BLE001 — a malformed block is wire's problem, not ours
        return "?", "?"
    return (str(entry.get("domain") or "?"),
            coverage.band_for(str(entry.get("frequency") or "")))


def _block_host(block: dict) -> str:
    try:
        entry = yaml.safe_load(block.get("yaml_block") or "")[0]
    except Exception:  # noqa: BLE001
        return "?"
    from urllib.parse import urlparse
    return urlparse((entry.get("endpoint") or {}).get("url", "")).netloc.lower()


def rank_candidates(blocks: list[dict], counts: Counter,
                    target_domains: Sequence[str],
                    per_host_cap: int = PER_HOST_CAP) -> list[dict]:
    """Pick the finds that most improve balance.

    Sorted by: is it a target domain, then how thin its (domain, cadence) cell
    is, then whether that band is high-value. The per-host cap is applied while
    selecting rather than after, so a productive host cannot crowd out the
    thinner cells it happens to outrank.
    """
    scored = []
    for i, b in enumerate(blocks):
        domain, band = _block_domain_band(b)
        scored.append((
            0 if domain in target_domains else 1,
            counts.get((domain, band), 0),
            0 if band in config.HIGH_VALUE_BANDS else 1,
            i,                      # stable: preserve discovery order in ties
            b,
        ))
    scored.sort(key=lambda t: t[:4])

    picked, host_used = [], Counter()
    for _, _, _, _, b in scored:
        host = _block_host(b)
        if host and host_used[host] >= per_host_cap:
            continue
        host_used[host] += 1
        picked.append(b)
    return picked


def run(catalog_path: str | Path,
        target: int = DEFAULT_TARGET,
        minutes: float = DEFAULT_MINUTES,
        n_domains: int = 3,
        checkpoint_dir: Optional[str | Path] = None,
        log=print) -> dict:
    """Sweep for candidates until the target or the deadline, then rank them.

    Does NOT wire anything — returns the ranked batch so the caller decides.
    """
    deadline = time.monotonic() + minutes * 60.0
    checkpoint_dir = Path(checkpoint_dir or tempfile.mkdtemp(prefix="grind-"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weak = weakest_domains(catalog_path, n_domains)
    target_domains = [w["domain"] for w in weak]
    log(f"[grind] thinnest domains: " +
        ", ".join(f"{w['domain']}({w['effective']})" for w in weak))

    want = target * OVERCOLLECT
    found: list[dict] = []
    ran: list[dict] = []

    for vein in _veins():
        if len(found) >= want or time.monotonic() >= deadline:
            break
        serving = [d for d in target_domains if vein.serves(d)]
        if not serving:
            continue
        kwargs = dict(vein.kwargs)
        if vein.keyworded:
            kws = tuple(k for k in bulk.KEYWORD_CLASSES if k[1] in serving)
            if vein.name == "arcgis":
                kws = tuple(k for k in bulk.ARCGIS_EXTRA_KEYWORDS
                            if k[1] in serving) + kws
            if not kws:
                continue
            kwargs["keywords" if vein.name == "arcgis" else "classes"] = kws
        # Stop on the overall deadline, not on a small per-vein budget: a short
        # VEIN_MINUTES is legitimate config and must not end the whole run.
        remaining = deadline - time.monotonic()
        if remaining <= 30:
            break
        budget = min(VEIN_MINUTES * 60.0, remaining)
        log(f"[grind] vein {vein.name} for {serving} "
            f"({budget / 60:.0f} min budget)")
        started = time.monotonic()
        vein_deadline = started + budget
        # Checkpoint so a vein stopped at its budget still hands over what it
        # banked, rather than losing a slow vein's whole contribution.
        ckpt = Path(checkpoint_dir) / f"{vein.name}.json"
        kwargs["checkpoint_path"] = str(ckpt)
        stopped = False
        try:
            cands, _skipped = vein.fn(
                str(catalog_path), target=want - len(found),
                log=_deadline_log(vein_deadline), **kwargs)
        except _DeadlineReached:
            stopped = True
            cands = _resume_checkpoint(ckpt)
        except Exception as exc:  # noqa: BLE001 — one dead vein must not end the run
            log(f"[grind] vein {vein.name} failed: {type(exc).__name__}: {exc}")
            ran.append({"vein": vein.name, "domains": serving, "found": 0,
                        "error": f"{type(exc).__name__}: {exc}"})
            continue
        elapsed = time.monotonic() - started
        found.extend(cands)
        ran.append({"vein": vein.name, "domains": serving,
                    "found": len(cands), "seconds": round(elapsed, 1),
                    "stopped_at_budget": stopped})
        log(f"[grind] vein {vein.name}: {len(cands)} in {elapsed:.0f}s"
            f"{' (hit budget)' if stopped else ''} ({len(found)}/{want})")

    counts = cell_counts(catalog_path)
    ranked = rank_candidates(found, counts, target_domains)
    return {
        "target_domains": target_domains,
        "weakest": weak,
        "veins_run": ran,
        "found": len(found),
        "ranked": ranked[:target],
        "surplus": len(ranked) - min(len(ranked), target),
    }
