"""Driver: run N foundation models against the live benchmark and test the gaps.

This is the reusable core behind ``notebooks/tsfm_significance.ipynb`` and
``scripts/run_tsfm_comparison_lium.py``. Keeping it as an importable module (not
notebook cells) means the same code path is smoke-tested locally on the classical
panel and then executed on the GPU — no copy-paste drift.

Flow
----
1. :func:`build_challenges` — deterministic challenge set from the scraped live
   catalog (``src/sources/data``), commit-reveal seeded.
2. :func:`load_models` — construct the TSFM roster via ``tsfm_adapters.load_tsfm``,
   skipping any that fail to import/load, always keeping the classical
   ``probabilistic_panel`` as reference rungs.
3. :func:`run_comparison` — one inference pass per model, then the leaderboard
   (seasonal-naive-relative shifted gmean, the primary ranking) plus the paired
   significance tests from :mod:`model_comparison`.

Every number is reproducible from ``(seed, motif_len, n_challenges, data_dir)``.
The honest breadth caveat (few source series → non-independent challenges) is
carried through in the returned ``note``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import model_comparison as mc
from challenges import build_live_challenges
from evaluate import leaderboard, probabilistic_panel
from ingest import FreshBuffer
from scraped_source import ScrapedLiveSource

# The 10-model roster (short names resolved by tsfm_adapters.load_tsfm). Order is
# display order; any that fail to load are skipped with a logged reason.
DEFAULT_ROSTER = (
    "chronos2",
    "chronos-bolt",
    "timesfm25",
    "toto2",
    "tirex",
    "moirai2",
    "flowstate",
    "tabpfn-ts",
    "sundial",
    "timesfm",
)


def _rng(*parts):
    return np.random.default_rng(abs(hash(parts)) % (2**32))


def build_challenges(
    data_dir: str,
    *,
    catalog: str = "src/sources/sources.yaml",
    motif_len: int = 304,
    n_challenges: int = 256,
    pool_size: int = 96,
    seed: str = "tsfm-significance-v1",
) -> list:
    """Deterministic challenge set from the scraped live catalog."""
    source = ScrapedLiveSource(catalog, data_dir, min_series_length=motif_len)
    buffer = FreshBuffer(source, pool_size=pool_size, motif_len=motif_len)
    return build_live_challenges(buffer, _rng(seed, "reveal"), n_challenges)


def load_models(
    roster=DEFAULT_ROSTER,
    *,
    device: str = "cuda",
    include_classical: bool = True,
    verbose: bool = True,
) -> tuple[dict, list[dict]]:
    """Load the TSFM roster, skipping failures. Returns ``(models, load_report)``.

    ``load_report`` records per-model ``{name, loaded, error}`` so the notebook can
    show exactly which models are in the run and why any are missing — a partial
    roster still yields a valid (smaller) comparison.
    """
    from tsfm_adapters import load_tsfm

    # A tiny probe forecast forces each adapter's lazy import + weight load now, so
    # "loaded" means "actually runnable" — a model whose extra is missing or whose
    # weights fail is excluded here rather than exploding mid-leaderboard.
    probe_ctx = np.sin(np.linspace(0, 12, 128)).astype(float)
    probe_meta = {"horizon": 8}

    models: dict = {}
    report: list[dict] = []
    for name in roster:
        try:
            adapter = load_tsfm(name, device=device) if _accepts_device(name) else load_tsfm(name)
            fc = adapter(probe_ctx, probe_meta)
            assert len(np.asarray(fc.mean)) >= 1 and fc.quantiles, "empty forecast"
            models[name] = adapter
            report.append({"name": name, "loaded": True, "error": None})
            if verbose:
                print(f"  ok   {name}")
        except Exception as e:  # noqa: BLE001 — a missing extra must not kill the run
            report.append({"name": name, "loaded": False, "error": f"{type(e).__name__}: {e}"})
            if verbose:
                print(f"  skip {name}: {type(e).__name__}: {e}")
    if include_classical:
        # Reference rungs (seasonal_naive, drift, ar1, ewma, strong, parrot).
        for name, fc in probabilistic_panel().items():
            models.setdefault(name, fc)
    return models, report


def _accepts_device(name: str) -> bool:
    """True for adapters whose __init__ takes a ``device`` kwarg."""
    return name.lower() not in {"timesfm", "timesfm25", "timesfm-2.5", "tirex"}


def run_comparison(
    data_dir: str,
    *,
    catalog: str = "src/sources/sources.yaml",
    roster=DEFAULT_ROSTER,
    device: str = "cuda",
    motif_len: int = 304,
    n_challenges: int = 256,
    seed: str = "tsfm-significance-v1",
    metrics=("crps", "mase"),
    out_dir: str | None = None,
) -> dict:
    """End-to-end: challenges → per-challenge scores → leaderboard + paired tests.

    Returns a JSON-serialisable dict with ``leaderboard``, ``load_report``,
    ``note``, and per-metric ``friedman`` / ``vs_baseline`` / ``pairwise`` blocks.
    Writes ``results.json`` under ``out_dir`` when given.
    """
    challenges = build_challenges(
        data_dir, catalog=catalog, motif_len=motif_len, n_challenges=n_challenges, seed=seed
    )
    models, load_report = load_models(roster, device=device)
    board = leaderboard(models, challenges)
    scores = mc.score_models(models, challenges)
    note = mc.source_clustered_note(scores)

    # Per-challenge scores are what let separate group runs merge into ONE
    # significance analysis: the challenge set is identical across groups (fixed
    # seed), so aligning by source_ids and unioning the model arrays reconstructs
    # the full paired matrix. See scripts/merge_tsfm_results.py.
    _first = next(iter(scores.values()))
    per_challenge = {
        "source_ids": _first.source_ids.tolist(),
        "models": {
            name: {"mase": s.mase.tolist(), "crps": s.crps.tolist(), "weight": s.weight.tolist()}
            for name, s in scores.items()
        },
    }

    out: dict = {
        "config": {
            "data_dir": str(data_dir),
            "motif_len": motif_len,
            "n_challenges": len(challenges),
            "seed": seed,
            "roster": list(roster),
        },
        "load_report": load_report,
        "note": note,
        "leaderboard": [_clean(row) for row in board],
        "per_challenge": per_challenge,
        "by_metric": {},
    }
    for metric in metrics:
        fr = mc.friedman_omnibus(scores, metric=metric)
        vb = mc.compare_to_baseline(scores, metric=metric)
        pw = mc.pairwise_significance(scores, metric=metric)
        out["by_metric"][metric] = {
            "friedman": {k: _clean(v) for k, v in fr.items()},
            "vs_baseline": [_clean(r) for r in vb],
            "pairwise": {
                "names": pw["names"],
                "p": pw["p"].tolist(),
                "win": pw["win"].tolist(),
                "better": pw["better"].tolist(),
            },
        }
    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "results.json").write_text(json.dumps(out, indent=2, default=_json_default))
    return out


def _clean(v):
    """Make numpy scalars/rows JSON-friendly."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
