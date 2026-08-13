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
from typing import Callable, Optional, Sequence  # noqa: F401  (Callable in sigs)

import yaml

from . import bulk, config, coverage

# Over-collect this many times the wire target, so cadence selection has
# something to choose between. Higher is better balance but slower.
OVERCOLLECT = 3
# No more than this many wired sources from one host in a single run.
PER_HOST_CAP = 2
DEFAULT_TARGET = 5
# Below this a run is a problem to look at, not a quiet day: the veins have
# stopped producing, or something upstream changed shape. Kept separate from
# the target so a 3-of-5 day reads as "short" and a 1-of-5 day reads as "broken".
DEFAULT_MINIMUM = 2
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


# Per-vein history, so the order can follow what actually works rather than
# what worked when this was written.
VEIN_STATS_NAME = "vein_stats.json"


def load_stats(stats_path: str | Path | None) -> dict[str, dict]:
    if not stats_path or not Path(stats_path).is_file():
        return {}
    try:
        data = json.loads(Path(stats_path).read_text())
    except Exception:  # noqa: BLE001 — a corrupt stats file must not stop a run
        return {}
    return data if isinstance(data, dict) else {}


def record_stats(stats_path: str | Path | None, rows: list[dict]) -> None:
    """Accumulate seconds/found/wired per vein. Best-effort; never raises."""
    if not stats_path:
        return
    stats = load_stats(stats_path)
    for r in rows:
        s = stats.setdefault(r["vein"], {"seconds": 0.0, "found": 0,
                                         "wired": 0, "runs": 0})
        s["seconds"] += float(r.get("seconds") or 0.0)
        s["found"] += int(r.get("found") or 0)
        s["wired"] += int(r.get("wired") or 0)
        s["runs"] += 1
    try:
        Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
        Path(stats_path).write_text(json.dumps(stats, indent=2, sort_keys=True))
    except Exception:  # noqa: BLE001
        pass


def _yield_rate(stats: dict, name: str) -> float | None:
    """Wired per minute for a vein, or None when it has never been tried.

    Rated on WIRED, not found: a vein that produces candidates the gate throws
    out is worse than useless, because it also spends the budget. Measured
    2026-08-13 — arcgis 2 finds in 1801s and none survived verification, while
    ckan managed 4 in 684s. Ordering on found alone would still have put the
    slow one first.
    """
    s = stats.get(name)
    if not s or not s.get("seconds"):
        return None
    return 60.0 * float(s.get("wired", 0)) / float(s["seconds"])


def _veins() -> tuple[Vein, ...]:
    """Vein registry, best-yielding first within each remit.

    This is the STATIC order, used only until vein_stats.json has history; see
    _plan, which re-sorts by measured wired-per-minute. It is listed
    cheapest-first rather than best-first, because a vein that gives up in a
    minute costs almost nothing to try and a slow one can eat the whole budget:
    measured 2026-08-13, arcgis spent 969s to produce one candidate that failed
    verification, while ods was exhausted in 61s.

    ERDDAP, PeerTube and the publisher sweeps are omitted -- each was measured
    empty (see the exhausted-veins notes), and re-running them daily would burn
    the budget before a productive vein got a turn.
    """
    return (
        # First because it is the only vein measured to actually deliver:
        # five wired in ~4 minutes where a keyword grind managed zero in 35.
        # It walks portals already known to work rather than hunting new ones.
        Vein("ods_enum", bulk.ods_enum_sweep,
             ("energy", "transport", "nature"), kwargs={"host_cap": 2}),
        Vein("ods", bulk.ods_sweep, ALL_DOMAINS, keyworded=True,
             kwargs={"host_cap": 1, "per_keyword": 100}),
        Vein("ckan", bulk.ckan_sweep, ALL_DOMAINS, keyworded=True,
             kwargs={"per_portal": 2, "keywords_per_portal": 6}),
        Vein("gbfs", bulk.gbfs_sweep, ("transport",), kwargs={"host_cap": 1}),
        Vein("misskey", bulk.misskey_sweep, ("web_cloudops",),
             kwargs={"host_cap": 1}),
        Vein("ixp", bulk.ixp_sweep, ("web_cloudops",), kwargs={"host_cap": 1}),
        Vein("pxweb", bulk.pxweb_sweep, ("econ_fin", "healthcare", "nature",
                                         "sales", "energy"),
             kwargs={"host_cap": 1}),
        Vein("sdmx", bulk.sdmx_sweep, ("econ_fin",),
             kwargs={"host_cap": 2, "max_flows": 300}),
        Vein("arcgis", bulk.arcgis_sweep, ALL_DOMAINS, keyworded=True,
             kwargs={"host_cap": 1, "per_keyword": 400, "sleep_s": 0.2}),
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


def _plan(target_domains: Sequence[str],
          stats: Optional[dict] = None) -> list[tuple["Vein", list[str]]]:
    """Veins to try: thin domains first, best-performing first within that.

    Two orderings compose here.

    The outer one is by domain — veins serving a thin domain go before the
    rest. The fallback pass matters for the "N wired per run" promise:
    preferring the three thinnest domains is right, but when their veins are
    mined out it is the difference between four sources and zero.

    The inner one is by measured wired-per-minute, so the budget goes where it
    has been paying. Veins never tried sort FIRST rather than last — an
    unmeasured vein is the cheapest information available, and sorting it last
    would mean a vein that never got a turn can never earn one.
    """
    stats = stats or {}

    def rank(v: "Vein") -> tuple[int, float]:
        rate = _yield_rate(stats, v.name)
        # (0, ...) explores the untried; (1, -rate) exploits the best known.
        return (1, -rate) if rate is not None else (0, 0.0)

    ordered = sorted(_veins(), key=rank)
    plan = [(v, [d for d in target_domains if v.serves(d)]) for v in ordered]
    first = [(v, ds) for v, ds in plan if ds]
    rest = [(v, [d for d in ALL_DOMAINS if v.serves(d)])
            for v, ds in plan if not ds]
    return first + rest


def run(catalog_path: str | Path,
        target: int = DEFAULT_TARGET,
        minimum: int = DEFAULT_MINIMUM,
        minutes: float = DEFAULT_MINUTES,
        n_domains: int = 3,
        checkpoint_dir: Optional[str | Path] = None,
        wire_fn: Optional[Callable[[list[dict]], int]] = None,
        stats_path: Optional[str | Path] = None,
        log=print) -> dict:
    """Sweep, rank and (with ``wire_fn``) keep going until ``target`` are WIRED.

    Without ``wire_fn`` this only collects and ranks — the old behaviour, used
    by dry runs.

    With one, the loop closes: wire re-verifies every candidate against the
    real scraper, so *found* and *wired* are different numbers, and a run that
    wired the ranked batch once could deliver far fewer than asked. Each vein's
    finds are wired as they arrive and the shortfall drives the next vein,
    until the target is met, the veins run out, or the clock does.
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
    pool: list[dict] = []          # collected, not yet submitted to wire
    ran: list[dict] = []
    wired = 0
    wire_attempts: list[dict] = []

    def _flush(final: bool = False) -> None:
        """Rank what is pooled and wire enough to cover the shortfall."""
        nonlocal wired, pool
        if not wire_fn or not pool or wired >= target:
            return
        short = target - wired
        ranked = rank_candidates(pool, cell_counts(catalog_path), target_domains)
        origin = {id(b): b.get("_grind_vein") for b in pool}
        # Submit more than the shortfall: wire re-verifies against the live
        # scraper and rejects some, so a 1:1 submission systematically
        # under-delivers.
        batch = ranked[:max(short * 2, short + 2)]
        if not batch:
            return
        got = wire_fn(batch)
        wired += got
        wire_attempts.append({"submitted": len(batch), "wired": got})
        log(f"[grind] wired {got}/{len(batch)} submitted "
            f"({wired}/{target} for this run)")
        # Credit the wins to the veins that supplied them, so the ordering
        # learns from wired rather than merely found.
        if got:
            share = {}
            for b in batch[:got]:
                v = origin.get(id(b))
                if v:
                    share[v] = share.get(v, 0) + 1
            for row in ran:
                row["wired"] = row.get("wired", 0) + share.get(row["vein"], 0)
        sent = {id(b) for b in batch}
        pool = [b for b in pool if id(b) not in sent]

    stats = load_stats(stats_path)
    for vein, serving in _plan(target_domains, stats):
        if wired >= target:
            break
        if not wire_fn and len(found) >= want:
            break
        if time.monotonic() >= deadline:
            break
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
        for c in cands:
            c["_grind_vein"] = vein.name
        found.extend(cands)
        pool.extend(cands)
        ran.append({"vein": vein.name, "domains": serving,
                    "found": len(cands), "seconds": round(elapsed, 1),
                    "stopped_at_budget": stopped})
        log(f"[grind] vein {vein.name}: {len(cands)} in {elapsed:.0f}s"
            f"{' (hit budget)' if stopped else ''} ({len(found)} found)")
        _flush()

    _flush(final=True)
    if wire_fn:
        # Only runs that actually wired teach anything about wired-rate.
        record_stats(stats_path, ran)

    counts = cell_counts(catalog_path)
    ranked = rank_candidates(pool, counts, target_domains)
    return {
        "target_domains": target_domains,
        "weakest": weak,
        "veins_run": ran,
        "found": len(found),
        "target": target,
        "minimum": minimum,
        "wired": wired if wire_fn else None,
        "met_target": (wired >= target) if wire_fn else None,
        "met_minimum": (wired >= minimum) if wire_fn else None,
        "wire_attempts": wire_attempts,
        "ranked": ranked[:target],
        "surplus": max(0, len(ranked) - target),
    }
