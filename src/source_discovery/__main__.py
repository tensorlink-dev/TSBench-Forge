"""CLI for the source-discovery agent.

    # Show current coverage + the biggest gaps (deterministic, no model):
    python -m source_discovery --coverage

    # Emit the exact prompt the agent would receive, without calling a model:
    python -m source_discovery --dry-run

    # Full run (needs OPENROUTER_API_KEY): propose -> vet -> write outputs:
    python -m source_discovery --out src/sources/discovered

    # Vet a candidate list produced elsewhere (e.g. by an interactive agent):
    python -m source_discovery --vet candidates.json --out src/sources/discovered
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import coverage, llm, runner

_DEFAULT_CATALOG = os.path.join(os.path.dirname(__file__), os.pardir, "sources", "sources.yaml")


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
    args = ap.parse_args(argv)

    if args.coverage:
        reg = coverage.load_registry(args.catalog)
        summary = coverage.summarize(reg)
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
