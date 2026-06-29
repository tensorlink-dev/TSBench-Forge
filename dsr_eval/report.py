"""Reporting: per-seed CSV plus a Table-2-style markdown summary (mean +/- std).

The runner emits one row per ``(model, system, seed)``; this module aggregates over
seeds and renders the result two ways:

* a flat ``results.csv`` (every seed, every metric) for downstream analysis, and
* a ``summary.md`` mirroring Table 2 of Durstewitz et al. (arXiv:2602.16864): one
  section per system, models as rows, the dynamics metrics (and MASE, for contrast) as
  columns, each cell ``mean +/- std`` across seeds.

The metric set and arrow direction are fixed here so every report reads the same way.
"""

from __future__ import annotations

import csv
import math
import os
from collections.abc import Sequence

# (key, header, lower-is-better) -- the columns of the summary table, in order.
_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("d_stsp", "D_stsp", True),
    ("d_h", "D_H", True),
    ("vpt", "VPT (lambda-t)", False),
    ("lyap_gen", "lambda_max(gen)", False),
    ("lyap_gap", "|d lambda|", True),
    ("mase", "MASE", True),
)
_CSV_FIELDS: tuple[str, ...] = (
    "model", "system", "seed",
    "d_stsp", "d_h", "vpt", "lyap_gen", "lyap_true", "lyap_gap", "d_ky_ref", "mase", "n_gen",
)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def aggregate(rows: Sequence[dict]) -> dict[tuple[str, str], dict[str, tuple[float, float]]]:
    """Aggregate per-seed rows to ``{(system, model): {metric: (mean, std)}}``."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((str(r["system"]), str(r["model"])), []).append(r)
    out: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    metric_keys = [k for k, _, _ in _METRICS] + ["lyap_true", "d_ky_ref"]
    for key, group in groups.items():
        out[key] = {m: _mean_std([float(g[m]) for g in group]) for m in metric_keys}
    return out


def write_csv(rows: Sequence[dict], path: str) -> str:
    """Write the flat per-seed rows to ``path`` (one row per model/system/seed)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return path


def _fmt(mean: float, std: float) -> str:
    if math.isnan(mean):
        return "n/a"
    return f"{mean:.3f} +/- {std:.3f}"


def summary_markdown(rows: Sequence[dict], *, cfg=None, n_seeds: int | None = None) -> str:
    """Render the Table-2-style markdown summary string."""
    agg = aggregate(rows)
    systems = sorted({s for s, _ in agg})
    seeds = n_seeds if n_seeds is not None else len({r["seed"] for r in rows})

    lines: list[str] = ["# DSR evaluation summary", ""]
    settings = [f"seeds: {seeds}"]
    if cfg is not None:
        settings += [
            f"embed_dim: {cfg.embed_dim}", f"bins: {cfg.bins}", f"sigma: {cfg.sigma}",
            f"VPT epsilon: {cfg.vpt_threshold}", f"D_stsp estimator: {cfg.estimator}",
        ]
    lines += [
        "Fixed settings (must match across runs for cross-run comparison): "
        + ", ".join(settings),
        "",
        "Arrows: down = lower is better, up = higher is better. "
        "MASE saturates on chaos; the dynamics metrics do not.",
        "",
    ]

    header = "| model | " + " | ".join(
        f"{h} {'v' if lower else '^'}" for _, h, lower in _METRICS
    ) + " |"
    sep = "|" + "---|" * (len(_METRICS) + 1)

    for system in systems:
        models = sorted(m for s, m in agg if s == system)
        lam_true, _ = agg[(system, models[0])]["lyap_true"] if models else (float("nan"), 0.0)
        d_ky, _ = agg[(system, models[0])]["d_ky_ref"] if models else (float("nan"), 0.0)
        title = f"## {system}"
        if not math.isnan(lam_true):
            title += f"  (reference lambda_max = {lam_true:.3f}, D_KY = {d_ky:.3f})"
        lines += [title, "", header, sep]
        for model in models:
            cells = [_fmt(*agg[(system, model)][k]) for k, _, _ in _METRICS]
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def write_reports(rows: Sequence[dict], out_dir: str, *, cfg=None) -> tuple[str, str]:
    """Write ``results.csv`` and ``summary.md`` into ``out_dir``; return both paths."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")
    md_path = os.path.join(out_dir, "summary.md")
    write_csv(rows, csv_path)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(summary_markdown(rows, cfg=cfg))
    return csv_path, md_path


__all__ = [
    "aggregate",
    "summary_markdown",
    "write_csv",
    "write_reports",
]
