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

from . import audit, bulk, coverage, llm, quality, runner, wire

_DEFAULT_CATALOG = os.path.join(os.path.dirname(__file__), os.pardir, "sources", "sources.yaml")
_DEFAULT_DATA = os.path.join(os.path.dirname(__file__), os.pardir, "sources", "data")


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
    ap.add_argument("--bulk-max-age-days", type=float, default=21.0,
                    help="with --bulk-socrata: reject datasets whose newest "
                         "observation is older than this (default 21)")
    ap.add_argument("--bulk-keywords", metavar="KW", nargs="+",
                    help="with --bulk-socrata: sweep only these keywords "
                         "(default: the full built-in keyword-class list)")
    args = ap.parse_args(argv)

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
        print(coverage.render_matrix(reg))
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
