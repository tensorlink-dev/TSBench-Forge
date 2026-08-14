"""CLI for the source-discovery agent.

    # Show current coverage + the biggest gaps (deterministic, no model):
    python -m source_discovery --coverage

    # Emit the exact prompt the agent would receive, without calling a model:
    python -m source_discovery --dry-run

    # Full run (needs OPENROUTER_API_KEY): propose -> vet -> write outputs:
    python -m source_discovery --out src/sources/discovered

    # Vet a candidate list produced elsewhere (e.g. by an interactive agent):
    python -m source_discovery --vet candidates.json --out src/sources/discovered

    # Auto-assess the DATA of an already-scraped source (admission gate):
    python -m source_discovery --assess aemo_nem_5min --data-dir src/sources/data

    # Verify endpoint-complete candidates against the REAL scraper; --apply wires
    # survivors into sources.yaml + cron.yaml and settles the proposal ledger:
    python -m source_discovery --wire batch.json --label grind-w1 [--apply]

    # Freshness audit: newest observation on disk vs declared cadence:
    python -m source_discovery --audit [--audit-json out.json] [--apply-disables]

    # Composition of the EVAL POOL (what clears min_series_length on disk),
    # which is not the same as the catalog's composition:
    python -m source_discovery --pool-report [--pool-report-json out.json]

    # Is the pipeline RUNNING (cron heartbeats, sweep outcomes, per-host
    # failures, sources writing nothing)? Seconds, not minutes — unlike --audit
    # it never opens a parquet:
    python -m source_discovery --monitor [--monitor-json out.json] [--quiet]

    # Bulk-generate candidates from the Socrata federated catalog (no model);
    # writes a batch in --wire's own format, so pipe it straight into --wire:
    python -m source_discovery --bulk-socrata batch.json --bulk-target 60
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import sys

from . import (audit, bulk, coverage, grind, llm, monitor, pool_report, quality,
               runner, wire)

_DEFAULT_CATALOG = os.path.join(os.path.dirname(__file__), os.pardir, "sources", "sources.yaml")
_DEFAULT_DATA = os.path.join(os.path.dirname(__file__), os.pardir, "sources", "data")
_DEFAULT_CRON = os.path.join(os.path.dirname(__file__), os.pardir, "sources", "cron.yaml")
# The audit/pool/grind logs live outside the repo, on the scrape host only.
# --monitor skips those heartbeats where the directory is absent rather than
# reporting three missing jobs on every developer machine.
_DEFAULT_OPS_LOGS = os.environ.get("TSFORGE_OPS_LOGS", "/root/cron/logs")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="source_discovery")
    ap.add_argument("--catalog", default=_DEFAULT_CATALOG, help="path to sources.yaml")
    ap.add_argument("--out", default="src/sources/discovered", help="output directory")
    ap.add_argument("--coverage", action="store_true",
                    help="print the coverage summary + ranked gaps and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the assembled agent prompt and exit (no model call)")
    ap.add_argument("--vet", metavar="FILE",
                    help="vet a candidate JSON array from FILE instead of calling the model")
    ap.add_argument("--assess", metavar="SOURCE_ID",
                    help="auto-assess the DATA of an already-scraped source (admission gate)")
    ap.add_argument("--data-dir", default=_DEFAULT_DATA,
                    help="scraped parquet dir for --assess (default src/sources/data)")
    ap.add_argument("--wire", metavar="FILE",
                    help="verify a candidate batch against the real scraper "
                         "(grind blocks or plain entries); wires with --apply")
    ap.add_argument("--apply", action="store_true",
                    help="with --wire: actually append survivors to sources.yaml, "
                         "register in cron.yaml, and settle the ledger")
    ap.add_argument("--label", default="unlabeled",
                    help="with --wire: batch label written above the appended block")
    ap.add_argument("--grind", action="store_true",
                    help="find N new sources steered at the thinnest domains "
                         "and cadence cells; --apply wires the survivors")
    ap.add_argument("--grind-target", type=int, default=grind.DEFAULT_TARGET,
                    help="how many sources to wire this run (default 5)")
    ap.add_argument("--grind-minimum", type=int, default=grind.DEFAULT_MINIMUM,
                    help="below this the run exits 2 (broken, not just short)")
    ap.add_argument("--grind-minutes", type=float, default=grind.DEFAULT_MINUTES,
                    help="wall-clock budget for the sweep phase (default 45)")
    ap.add_argument("--grind-domains", type=int, default=3,
                    help="how many of the thinnest domains to target")
    ap.add_argument("--pool-report", action="store_true",
                    help="composition of the EVAL POOL (what clears "
                         "min_series_length on disk), not of the catalog")
    ap.add_argument("--pool-report-json", metavar="FILE",
                    help="with --pool-report: also write the full report JSON")
    ap.add_argument("--monitor", action="store_true",
                    help="pipeline health: cron heartbeats, sweeps, failures, silence")
    ap.add_argument("--monitor-json", metavar="FILE",
                    help="with --monitor: also write the full report JSON")
    ap.add_argument("--monitor-window-hours", type=float,
                    default=monitor.DEFAULT_WINDOW_HOURS,
                    help="with --monitor: failure-rollup window (default 24)")
    ap.add_argument("--ops-log-dir", default=_DEFAULT_OPS_LOGS,
                    help="with --monitor: dir holding audit/pool/grind logs")
    ap.add_argument("--quiet", action="store_true",
                    help="with --monitor: print only when there are findings")
    ap.add_argument("--audit", action="store_true",
                    help="freshness audit: newest observation on disk vs cadence")
    ap.add_argument("--audit-json", metavar="FILE",
                    help="with --audit: also write full findings JSON to FILE")
    ap.add_argument("--apply-disables", action="store_true",
                    help="with --audit: disable sources stale past 3x their limit")
    ap.add_argument("--bulk-socrata", metavar="OUT_FILE",
                    help="sweep the Socrata federated catalog and write a "
                         "--wire-format candidate batch to OUT_FILE (no model)")
    ap.add_argument("--bulk-ods", metavar="OUT_FILE",
                    help="same, over the Opendatasoft federated catalog "
                         "(~100k datasets, Europe-heavy)")
    ap.add_argument("--bulk-ckan", metavar="OUT_FILE",
                    help="same, over a curated list of national CKAN portals "
                         "(chosen for geographic spread, not volume)")
    ap.add_argument("--bulk-erddap", metavar="OUT_FILE",
                    help="same, over the ERDDAP federation (~60 independently "
                         "operated ocean/met observing servers)")
    ap.add_argument("--bulk-ods-publishers", metavar="OUT_FILE",
                    help="walk the Opendatasoft federation PUBLISHER-first: "
                         "enumerate every portal, then take the best series "
                         "each one has. Finds hosts where --bulk-ods, which is "
                         "keyword-first, keeps re-meeting the same big portals")
    ap.add_argument("--bulk-socrata-publishers", metavar="OUT_FILE",
                    help="same inversion over Socrata: enumerate domains from "
                         "the federated catalog, then take the best series each "
                         "one has (Socrata has no domains endpoint)")
    ap.add_argument("--bulk-ixp", metavar="OUT_FILE",
                    help="sweep IXP Manager traffic graphers listed in the "
                         "public IXP database (web_cloudops)")
    ap.add_argument("--bulk-peertube", metavar="OUT_FILE",
                    help="sweep the PeerTube instance index for per-instance "
                         "video publication rates (web_cloudops)")
    ap.add_argument("--bulk-gbfs", metavar="OUT_FILE",
                    help="sweep the GBFS bikeshare system index for station "
                         "occupancy series (transport)")
    ap.add_argument("--bulk-misskey", metavar="OUT_FILE",
                    help="sweep Misskey/Sharkey instances for hourly note "
                         "publication rates via /api/charts (web_cloudops)")
    ap.add_argument("--bulk-misskey-charts", default="notes",
                    help="comma-separated Misskey charts to take per host "
                         f"({','.join(bulk.MISSKEY_CHARTS)}); a second chart "
                         "needs --bulk-host-cap 2 to clear the first pass")
    ap.add_argument("--bulk-arcgis", metavar="OUT_FILE",
                    help="sweep ArcGIS Hub for self-hosted local-government "
                         "feature services (skips the vendor's shared "
                         "services*.arcgis.com tenancy)")
    ap.add_argument("--bulk-sdmx", metavar="OUT_FILE",
                    help="sweep SDMX 2.1 REST endpoints (central banks and "
                         "international agencies; econ_fin-heavy, SDMX-CSV "
                         "over the existing rest_csv reader)")
    ap.add_argument("--bulk-max-flows", type=int, default=400,
                    help="with --bulk-sdmx: how deep to read each agency's "
                         "dataflow list (default 400)")
    ap.add_argument("--bulk-pxweb", metavar="OUT_FILE",
                    help="walk PX-Web statistics-office subject trees "
                         "(econ_fin-heavy; json-stat2 over POST)")
    ap.add_argument("--bulk-per-host", type=int, default=None,
                    help="with --bulk-erddap: datasets to take per server "
                         "(default: --bulk-host-cap); with "
                         "--bulk-ods-publishers: datasets to SCAN per portal "
                         "(default 25)")
    ap.add_argument("--bulk-per-keyword", type=int, default=None,
                    help="catalog results to page through per keyword "
                         "(default 200 Socrata / 100 ODS)")
    ap.add_argument("--bulk-target", type=int, default=None,
                    help="with --bulk-socrata: stop once N candidates are found")
    ap.add_argument("--bulk-host-cap", type=int, default=bulk.DEFAULT_HOST_CAP,
                    help="with --bulk-socrata: max candidates per host "
                         f"(default {bulk.DEFAULT_HOST_CAP}, matching the "
                         "coverage metric's per-host credit cap)")
    ap.add_argument("--bulk-any-host", action="store_true",
                    help="with --bulk-socrata: allow already-wired hosts "
                         "(default is new providers only)")
    ap.add_argument("--bulk-scan-instances", type=int, default=600,
                    help="with --bulk-peertube: how deep to read the instance "
                         "index, which is sorted biggest-first (default 600 "
                         "of ~1900 listed)")
    ap.add_argument("--bulk-max-age-days", type=float, default=21.0,
                    help="with --bulk-socrata: reject datasets whose newest "
                         "observation is older than this (default 21)")
    ap.add_argument("--bulk-keywords", metavar="KW", nargs="+",
                    help="with --bulk-socrata: sweep only these keywords "
                         "(default: the full built-in keyword-class list)")
    args = ap.parse_args(argv)

    if args.bulk_arcgis:
        cands, skipped = bulk.arcgis_sweep(
            args.catalog, host_cap=args.bulk_host_cap,
            per_keyword=args.bulk_per_keyword or 300,
            max_age_days=args.bulk_max_age_days,
            target=args.bulk_target, checkpoint_path=args.bulk_arcgis,
            log=lambda m: print(m, file=sys.stderr),
        )
        bulk.write_batch(cands, args.bulk_arcgis)
        reasons = {}
        for s in skipped:
            key = re.sub(r"\d+", "N", s.get("reason", "?"))[:80]
            reasons[key] = reasons.get(key, 0) + 1
        print(json.dumps({
            "candidates": len(cands), "out": args.bulk_arcgis,
            "skipped": len(skipped),
            "skip_reasons": dict(sorted(reasons.items(), key=lambda p: -p[1])[:14]),
        }, indent=2))
        return 0 if cands else 1

    if args.bulk_sdmx:
        cands, skipped = bulk.sdmx_sweep(
            args.catalog, host_cap=args.bulk_host_cap,
            max_flows=args.bulk_max_flows,
            target=args.bulk_target, checkpoint_path=args.bulk_sdmx,
            log=lambda m: print(m, file=sys.stderr),
        )
        bulk.write_batch(cands, args.bulk_sdmx)
        reasons = {}
        for s in skipped:
            key = re.sub(r"\d+", "N", s.get("reason", "?"))[:80]
            reasons[key] = reasons.get(key, 0) + 1
        print(json.dumps({
            "candidates": len(cands), "out": args.bulk_sdmx,
            "skipped": len(skipped),
            "skip_reasons": dict(sorted(reasons.items(), key=lambda p: -p[1])[:14]),
        }, indent=2))
        return 0 if cands else 1

    if args.bulk_pxweb:
        cands, skipped = bulk.pxweb_sweep(
            args.catalog, host_cap=args.bulk_host_cap,
            max_age_days=max(args.bulk_max_age_days, 120.0),
            target=args.bulk_target, checkpoint_path=args.bulk_pxweb,
            log=lambda m: print(m, file=sys.stderr),
        )
        bulk.write_batch(cands, args.bulk_pxweb)
        reasons = {}
        for s in skipped:
            key = re.sub(r"\d+", "N", s.get("reason", "?"))[:80]
            reasons[key] = reasons.get(key, 0) + 1
        print(json.dumps({
            "candidates": len(cands), "out": args.bulk_pxweb,
            "skipped": len(skipped),
            "skip_reasons": dict(sorted(reasons.items(), key=lambda p: -p[1])[:14]),
        }, indent=2))
        return 0 if cands else 1

    if args.bulk_peertube:
        cands, skipped = bulk.peertube_sweep(
            args.catalog, host_cap=1,
            scan_instances=args.bulk_scan_instances,
            max_age_days=args.bulk_max_age_days,
            target=args.bulk_target, checkpoint_path=args.bulk_peertube,
            log=lambda m: print(m, file=sys.stderr),
        )
        bulk.write_batch(cands, args.bulk_peertube)
        reasons = {}
        for s in skipped:
            key = re.sub(r"\d+", "N", s.get("reason", "?"))[:80]
            reasons[key] = reasons.get(key, 0) + 1
        print(json.dumps({
            "candidates": len(cands), "out": args.bulk_peertube,
            "skipped": len(skipped),
            "skip_reasons": dict(sorted(reasons.items(), key=lambda p: -p[1])[:14]),
        }, indent=2))
        return 0 if cands else 1

    if args.bulk_gbfs:
        cands, skipped = bulk.gbfs_sweep(
            args.catalog, host_cap=args.bulk_host_cap,
            target=args.bulk_target, checkpoint_path=args.bulk_gbfs,
            log=lambda m: print(m, file=sys.stderr),
        )
        bulk.write_batch(cands, args.bulk_gbfs)
        reasons = {}
        for sk in skipped:
            key = re.sub(r"\d+", "N", sk.get("reason", "?"))[:80]
            reasons[key] = reasons.get(key, 0) + 1
        print(json.dumps({
            "candidates": len(cands), "out": args.bulk_gbfs,
            "skipped": len(skipped),
            "skip_reasons": dict(sorted(reasons.items(), key=lambda p: -p[1])[:14]),
        }, indent=2))
        return 0 if cands else 1

    if args.bulk_misskey:
        charts = [c.strip() for c in args.bulk_misskey_charts.split(",")
                  if c.strip()]
        unknown = [c for c in charts if c not in bulk.MISSKEY_CHARTS]
        if unknown:
            print(f"unknown misskey chart(s): {', '.join(unknown)}; "
                  f"known: {', '.join(bulk.MISSKEY_CHARTS)}", file=sys.stderr)
            return 2
        cands, skipped = bulk.misskey_sweep(
            args.catalog, host_cap=args.bulk_host_cap or 1, charts=charts,
            target=args.bulk_target, checkpoint_path=args.bulk_misskey,
            log=lambda m: print(m, file=sys.stderr),
        )
        bulk.write_batch(cands, args.bulk_misskey)
        reasons = {}
        for s in skipped:
            key = re.sub(r"\d+", "N", s.get("reason", "?"))[:80]
            reasons[key] = reasons.get(key, 0) + 1
        print(json.dumps({
            "candidates": len(cands), "out": args.bulk_misskey,
            "skipped": len(skipped),
            "unexamined": sum(1 for s in skipped
                              if s.get("reason", "").startswith("unexamined")),
            "skip_reasons": dict(sorted(reasons.items(), key=lambda p: -p[1])[:14]),
        }, indent=2))
        return 0 if cands else 1

    if args.bulk_ixp:
        cands, skipped = bulk.ixp_sweep(
            args.catalog, host_cap=args.bulk_host_cap,
            target=args.bulk_target, checkpoint_path=args.bulk_ixp,
            log=lambda m: print(m, file=sys.stderr),
        )
        bulk.write_batch(cands, args.bulk_ixp)
        reasons = {}
        for s in skipped:
            key = re.sub(r"\d+", "N", s.get("reason", "?"))[:80]
            reasons[key] = reasons.get(key, 0) + 1
        print(json.dumps({
            "candidates": len(cands), "out": args.bulk_ixp,
            "skipped": len(skipped),
            "skip_reasons": dict(sorted(reasons.items(), key=lambda p: -p[1])[:14]),
        }, indent=2))
        return 0 if cands else 1

    if args.bulk_socrata_publishers:
        cands, skipped = bulk.socrata_publisher_sweep(
            args.catalog, host_cap=args.bulk_host_cap,
            scan_per_host=args.bulk_per_host or 25,
            max_age_days=args.bulk_max_age_days,
            target=args.bulk_target,
            checkpoint_path=args.bulk_socrata_publishers,
            log=lambda m: print(m, file=sys.stderr),
        )
        bulk.write_batch(cands, args.bulk_socrata_publishers)
        reasons = {}
        for s in skipped:
            key = re.sub(r"\d+", "N", s.get("reason", "?"))[:80]
            reasons[key] = reasons.get(key, 0) + 1
        print(json.dumps({
            "candidates": len(cands), "out": args.bulk_socrata_publishers,
            "skipped": len(skipped),
            "skip_reasons": dict(sorted(reasons.items(), key=lambda p: -p[1])[:14]),
        }, indent=2))
        return 0 if cands else 1

    if args.bulk_ods_publishers:
        cands, skipped = bulk.ods_publisher_sweep(
            args.catalog, host_cap=args.bulk_host_cap,
            scan_per_host=args.bulk_per_host or 25,
            max_age_days=args.bulk_max_age_days,
            target=args.bulk_target,
            checkpoint_path=args.bulk_ods_publishers,
            log=lambda m: print(m, file=sys.stderr),
        )
        bulk.write_batch(cands, args.bulk_ods_publishers)
        reasons = {}
        for s in skipped:
            key = re.sub(r"\d+", "N", s.get("reason", "?"))[:80]
            reasons[key] = reasons.get(key, 0) + 1
        print(json.dumps({
            "candidates": len(cands), "out": args.bulk_ods_publishers,
            "skipped": len(skipped),
            "skip_reasons": dict(sorted(reasons.items(), key=lambda p: -p[1])[:14]),
        }, indent=2))
        return 0 if cands else 1

    if args.bulk_erddap:
        cands, skipped = bulk.erddap_sweep(
            args.catalog, host_cap=args.bulk_host_cap,
            per_server=args.bulk_per_host,
            max_age_days=args.bulk_max_age_days,
            # Always allow wired hosts: an ERDDAP server is a whole regional
            # observing system, and the three already in the catalog carry one
            # dataset each. The per-host cap is what bounds concentration.
            new_hosts_only=False,
            target=args.bulk_target, checkpoint_path=args.bulk_erddap,
            log=lambda m: print(m, file=sys.stderr),
        )
        bulk.write_batch(cands, args.bulk_erddap)
        reasons: dict[str, int] = {}
        for s in skipped:
            key = re.sub(r"\d+", "N", s.get("reason", "?"))[:80]
            reasons[key] = reasons.get(key, 0) + 1
        print(json.dumps({
            "candidates": len(cands), "out": args.bulk_erddap,
            "skipped": len(skipped),
            "skip_reasons": dict(sorted(reasons.items(), key=lambda p: -p[1])[:14]),
        }, indent=2))
        return 0 if cands else 1

    if args.bulk_ckan:
        cands, skipped = bulk.ckan_sweep(
            args.catalog, max_age_days=args.bulk_max_age_days,
            target=args.bulk_target, checkpoint_path=args.bulk_ckan,
            log=lambda m: print(m, file=sys.stderr),
        )
        bulk.write_batch(cands, args.bulk_ckan)
        reasons: dict[str, int] = {}
        for s in skipped:
            key = re.sub(r"\d+", "N", s.get("reason", "?"))[:80]
            reasons[key] = reasons.get(key, 0) + 1
        print(json.dumps({
            "candidates": len(cands), "out": args.bulk_ckan,
            "skipped": len(skipped),
            "skip_reasons": dict(sorted(reasons.items(), key=lambda p: -p[1])[:12]),
        }, indent=2))
        return 0 if cands else 1

    if args.bulk_socrata or args.bulk_ods:
        classes = bulk.KEYWORD_CLASSES
        if args.bulk_keywords:
            want = {k.lower() for k in args.bulk_keywords}
            classes = tuple(c for c in classes if c[0].lower() in want)
            if not classes:
                print(f"no built-in keyword class matches {sorted(want)}; known: "
                      + ", ".join(sorted(c[0] for c in bulk.KEYWORD_CLASSES)),
                      file=sys.stderr)
                return 2
        sweep_fn = bulk.ods_sweep if args.bulk_ods else bulk.sweep
        out_path = args.bulk_ods or args.bulk_socrata
        # ODS defaults to its own multilingual class list; passing the English
        # one explicitly would silently override that and re-run an
        # English-only sweep over a mostly-French catalog.
        if args.bulk_ods and not args.bulk_keywords:
            classes = None
        kwargs = dict(
            classes=classes, host_cap=args.bulk_host_cap,
            max_age_days=args.bulk_max_age_days,
            new_hosts_only=not args.bulk_any_host, target=args.bulk_target,
            checkpoint_path=out_path,
            log=lambda m: print(m, file=sys.stderr),
        )
        if args.bulk_per_keyword:
            kwargs["per_keyword"] = args.bulk_per_keyword
        elif not args.bulk_ods:
            kwargs["per_keyword"] = 200
        cands, skipped = sweep_fn(args.catalog, **kwargs)
        bulk.write_batch(cands, out_path)
        reasons: dict[str, int] = {}
        for s in skipped:
            key = re.sub(r"\d+", "N", s.get("reason", "?"))
            reasons[key] = reasons.get(key, 0) + 1
        print(json.dumps({
            "candidates": len(cands),
            "out": out_path,
            "hosts": len({c["candidate_name"].rsplit("(", 1)[-1] for c in cands}),
            "skipped": len(skipped),
            "skip_reasons": dict(sorted(reasons.items(), key=lambda p: -p[1])),
        }, indent=2))
        return 0 if cands else 1

    if args.wire:
        rep = wire.wire_batch(args.wire, args.catalog, label=args.label,
                              apply=args.apply)
        print(json.dumps(rep, indent=2))
        return 0 if rep["counts"]["wire"] or not rep["detail"] else 1

    if args.grind:
        import datetime as _dt
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M")
        out_dir = Path(args.out or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        wired_names: list[str] = []
        wire_reports: list[dict] = []

        def _wire(blocks: list[dict]) -> int:
            """Wire one batch; returns how many actually landed.

            Feeding the count back is what lets the sweep keep going until the
            run's target is genuinely met — wire re-verifies against the live
            scraper, so submitting N never means N wired.
            """
            part = out_dir / f"grind-{stamp}-{len(wire_reports) + 1}.json"
            bulk.write_batch(blocks, str(part))
            rep = wire.wire_batch(str(part), args.catalog,
                                  label=f"grind-{stamp}", apply=True)
            wire_reports.append({"batch": str(part), **rep["counts"]})
            wired_names.extend(d.get("name") for d in rep.get("detail", [])
                               if d.get("verdict") == "wire")
            return int(rep["counts"].get("wire", 0))

        plan = grind.run(args.catalog, target=args.grind_target,
                         minimum=args.grind_minimum,
                         minutes=args.grind_minutes,
                         n_domains=args.grind_domains,
                         wire_fn=_wire if args.apply else None,
                         stats_path=out_dir / grind.VEIN_STATS_NAME,
                         log=lambda m: print(m, file=sys.stderr))
        report = {k: plan[k] for k in
                  ("target_domains", "veins_run", "found", "target", "minimum",
                   "wired", "met_target", "met_minimum", "surplus")}
        if args.apply:
            report["wire_batches"] = wire_reports
            report["wired_names"] = wired_names
        else:
            batch = out_dir / f"grind-{stamp}.json"
            bulk.write_batch(plan["ranked"], str(batch))
            report["batch"] = str(batch)
            report["ranked"] = [b.get("candidate_name") for b in plan["ranked"]]
        print(json.dumps(report, indent=2))
        if args.apply:
            # Three states, because they need different responses: hit the
            # target, fell short of it, or dropped under the floor -- which
            # means the veins stopped producing or something changed shape.
            if plan["met_target"]:
                return 0
            return 1 if plan["met_minimum"] else 2
        return 0 if plan["ranked"] else 1

    if args.pool_report:
        report = pool_report.build(args.catalog, args.data_dir)
        if args.pool_report_json:
            Path(args.pool_report_json).write_text(
                json.dumps(report, indent=2) + "\n")
        print(pool_report.summarize(report))
        print()
        print(pool_report.render_table(report))
        print()
        print(json.dumps({k: report[k] for k in (
            "totals", "eligible_sources_by_domain", "eligible_series_by_domain",
            "ineligible_on_depth_by_domain", "sampled_by_domain",
            "sampled_by_cadence")}, indent=2))
        return 0 if not report["sample_error"] else 1

    if args.monitor:
        ops = args.ops_log_dir if os.path.isdir(args.ops_log_dir or "") else None
        cron = _DEFAULT_CRON if os.path.exists(_DEFAULT_CRON) else None
        report = monitor.build(args.catalog, args.data_dir, cron_path=cron,
                               ops_log_dir=ops,
                               window_hours=args.monitor_window_hours)
        if args.monitor_json:
            Path(args.monitor_json).write_text(json.dumps(report, indent=2) + "\n")
        if args.quiet:
            # Cron-friendly: silence on a healthy pipeline, one line otherwise,
            # so a mailed run means something went wrong.
            if report["findings"]:
                print(monitor.summarize(report))
                for f in report["findings"]:
                    print(f"  [{f['severity']}] {f['check']}: {f['detail']}")
        else:
            print(monitor.render(report))
        return {"ok": 0, "warn": 1, "critical": 2}[report["status"]]

    if args.audit:
        findings = audit.audit_catalog(args.catalog, args.data_dir)
        summary = audit.summarize(findings)
        if args.audit_json:
            Path(args.audit_json).write_text(json.dumps(findings, indent=2) + "\n")
        if args.apply_disables:
            summary["disabled_now"] = audit.apply_disables(args.catalog, findings)
        print(json.dumps(summary, indent=2))
        return 0 if not summary["stale"] else 1

    if args.assess:
        q = quality.assess_scraped_source(args.catalog, args.data_dir, args.assess)
        disc = None
        if q.discrimination is not None:
            d = q.discrimination
            disc = {"ok": d.ok, "predictability": d.predictability,
                    "naive_error": d.naive_error, "spread": d.spread,
                    "n_windows": d.n_windows, "reasons": d.reasons}
        report = {
            "source": args.assess,
            "admitted": q.ok,
            "series_ok": f"{q.n_series_ok}/{q.n_series}",
            "reasons": q.reasons,
            "discrimination": disc,
            "per_series_metrics": [p.metrics for p in q.per_series],
        }
        print(json.dumps(report, indent=2, default=str))
        return 0 if q.ok else 1

    if args.coverage:
        reg = coverage.load_registry(args.catalog)
        summary = coverage.summarize(reg)
        # stdout is the machine contract: the summary JSON and nothing else, so
        # `--coverage | python -c 'json.load(sys.stdin)'` keeps working. The
        # human-facing matrix and gap list go to stderr, which a terminal shows
        # anyway and a pipeline can drop.
        print(coverage.render_matrix(reg), file=sys.stderr)
        print(json.dumps(summary, indent=2, default=str))
        gaps = summary["gap_cells"]
        print(f"\n{len(gaps)} under-target cells; top 10 gaps:", file=sys.stderr)
        for g in gaps[:10]:
            star = " *high-value*" if g["high_value"] else ""
            print(f"  {g['domain']:<12} {g['cadence']:<10} have={g['have']} "
                  f"target={g['target']}{star}", file=sys.stderr)
        return 0

    if args.dry_run:
        inputs = runner.build_inputs(args.catalog)
        print("===== SYSTEM =====\n" + llm.system_prompt())
        print("\n===== USER =====\n" + llm.build_user_message(inputs))
        return 0

    if args.vet:
        res = runner.run_vet(args.vet, args.catalog, args.out)
        print(json.dumps(res, indent=2))
        return 0

    # Full run.
    cfg = llm.OpenRouterConfig.from_env()
    if not cfg.enabled:
        print("OPENROUTER_API_KEY not set. Use --coverage, --dry-run, or --vet <file>.",
              file=sys.stderr)
        return 2
    res = runner.run_discovery(args.catalog, cfg, args.out)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
