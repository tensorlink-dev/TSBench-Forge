"""Orchestrate discovery: build inputs -> (propose) -> vet -> write outputs.

Pure-Python glue. The only non-deterministic step is :func:`llm.propose`; the
rest (loading the registry, building the coverage summary, vetting proposals,
writing files) is deterministic and unit-tested.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config, coverage, ledger, llm, vet


def target_coverage() -> dict:
    """The TARGET_COVERAGE input: per-cell source targets + the priority bands."""
    return {
        "domains": list(config.DOMAINS),
        "cadence_bands": list(config.CADENCE_BANDS),
        "target_per_cell": config.TARGET_PER_CELL,
        "high_value_bands": list(config.HIGH_VALUE_BANDS),
        "target_per_high_value_cell": config.TARGET_PER_HIGH_VALUE_CELL,
        "note": (
            "Fill each (domain x cadence) cell to its target. Live/sub-hourly and "
            "irregular/event-driven cells are the highest value (scarce + "
            "contamination-resistant)."
        ),
    }


def build_inputs(catalog_path: str | Path) -> dict:
    """Assemble the agent inputs (deterministic, no LLM).

    The prompt's CURRENT_SOURCES is one terse string per source, not the full
    registry: reasoning models burn their completion budget deliberating over
    large structured blocks (observed: ~40% of sweep rounds dying with
    finish_reason=length once the full 63kB registry + ledger were in the
    prompt). The full registry still backs the coverage matrix and vetting.
    """
    from urllib.parse import urlparse

    registry = coverage.load_registry(catalog_path)
    led = ledger.load(ledger.ledger_path(catalog_path))
    compact = [
        f"{e['id']} [{e['domain']}/{e['cadence']}] "
        f"{urlparse(str(e.get('url_or_endpoint', ''))).netloc}"
        for e in registry
    ]
    return {
        "registry": registry,  # kept for vetting; not sent verbatim
        "ledger": led,  # kept for vetting; compact form goes in the prompt
        "current_sources": compact,
        "coverage_summary": coverage.summarize(registry),
        "target_coverage": target_coverage(),
        "contamination_denylist": list(config.CONTAMINATION_DENYLIST),
        "model_cutoffs": dict(config.MODEL_CUTOFFS),
        "already_proposed": ledger.prompt_block(led),
    }


def _write_outputs(
    out_dir: Path, gap_analysis: str, results: list[vet.VetResult]
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"accept": 0, "flag": 0, "reject": 0}
    for r in results:
        counts[r.verdict] += 1

    if gap_analysis:
        (out_dir / "gap_analysis.md").write_text(gap_analysis + "\n")

    vetted = [
        {**r.candidate, "_verdict": r.verdict, "_vet_reasons": r.reasons}
        for r in results
    ]
    (out_dir / "candidates.json").write_text(json.dumps(vetted, indent=2, default=str) + "\n")
    return counts


def run_discovery(
    catalog_path: str | Path,
    cfg: llm.OpenRouterConfig,
    out_dir: str | Path,
) -> dict:
    """Full run: propose via the model, vet, and write outputs. Returns a summary.

    Vetted proposals (accepted AND rejected) are upserted into the proposal
    ledger so subsequent runs see them in ALREADY_PROPOSED and the vet
    hard-rejects re-proposals — the agent explores instead of re-treading.
    """
    import datetime as dt

    inputs = build_inputs(catalog_path)
    gap_analysis, candidates = llm.propose(inputs, cfg)
    results = vet.vet_all(candidates, inputs["registry"], ledger=inputs["ledger"])
    counts = _write_outputs(Path(out_dir), gap_analysis, results)
    new = ledger.update(
        ledger.ledger_path(catalog_path), results, dt.date.today().isoformat()
    )
    return {"proposed": len(candidates), **counts, "ledger_new": new, "out_dir": str(out_dir)}


def run_vet(
    candidates_file: str | Path,
    catalog_path: str | Path,
    out_dir: str | Path,
) -> dict:
    """Vet a candidate list produced elsewhere (no model call). Returns a summary."""
    candidates = json.loads(Path(candidates_file).read_text())
    if not isinstance(candidates, list):
        raise ValueError("candidates file must be a JSON array of proposal objects")
    registry = coverage.load_registry(catalog_path)
    results = vet.vet_all(candidates, registry)
    counts = _write_outputs(Path(out_dir), "", results)
    return {"proposed": len(candidates), **counts, "out_dir": str(out_dir)}
