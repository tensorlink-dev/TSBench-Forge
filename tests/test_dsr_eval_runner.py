"""End-to-end DSR runner + reporting.

Exercises the full protocol on the classical panel baselines (no torch needed): model
resolution, the per-sample metric row, multi-seed aggregation, and the CSV / markdown
report. Also checks the load-bearing contrast -- a model that collapses chaos to a
periodic repeat is exposed by D_H even though it is invisible to the geometry metric.
"""

from __future__ import annotations

import numpy as np
import pytest

from dsr_eval.datasets import build_sample
from dsr_eval.report import aggregate, summary_markdown, write_reports
from dsr_eval.runner import (
    MetricConfig,
    evaluate_sample,
    resolve_forecaster,
    run_dsr_eval,
)
from dsr_eval.systems import get_system


def test_resolve_forecaster() -> None:
    assert callable(resolve_forecaster("seasonal_naive"))
    assert callable(resolve_forecaster("context_parrot"))
    sentinel = object()

    def fn(ctx, meta=None):
        return sentinel

    assert resolve_forecaster(fn) is fn  # callables pass through untouched
    with pytest.raises(KeyError):
        resolve_forecaster("no_such_model")


def test_evaluate_sample_keys_and_finiteness() -> None:
    sample = build_sample(get_system("lorenz"), 0, context_len=256, long_len=1500)
    row = evaluate_sample(resolve_forecaster("seasonal_naive"), sample)
    for key in ("d_stsp", "d_h", "vpt", "lyap_gen", "lyap_true", "lyap_gap", "mase", "n_gen"):
        assert key in row
    assert np.isfinite(row["d_stsp"]) and row["d_stsp"] >= 0.0
    assert 0.0 <= row["d_h"] <= 1.0
    assert row["mase"] > 0.0
    assert row["lyap_true"] == pytest.approx(0.9056, abs=1e-3)


def test_seasonal_naive_collapse_exposed_by_dh() -> None:
    # seasonal_naive repeats its last cycle -> a periodic rollout. Its power spectrum
    # is sharply peaked where true Lorenz is broadband, so D_H is large.
    sample = build_sample(get_system("lorenz"), 1, context_len=256, long_len=1500)
    row = evaluate_sample(resolve_forecaster("seasonal_naive"), sample)
    assert row["d_h"] > 0.5
    assert row["lyap_gen"] == pytest.approx(0.0, abs=0.2)  # no chaos in a clean repeat


def test_run_dsr_eval_row_count_and_columns() -> None:
    rows = run_dsr_eval(
        ["seasonal_naive", "context_parrot"],
        ["lorenz", "rossler"],
        seeds=(0, 1),
        context_len=256,
        long_len=1200,
    )
    assert len(rows) == 2 * 2 * 2  # models x systems x seeds
    models = {r["model"] for r in rows}
    systems = {r["system"] for r in rows}
    assert models == {"seasonal_naive", "context_parrot"}
    assert systems == {"lorenz", "rossler"}


def test_aggregate_mean_std() -> None:
    rows = run_dsr_eval(
        ["seasonal_naive"], ["lorenz"], seeds=(0, 1, 2), context_len=256, long_len=1200
    )
    agg = aggregate(rows)
    mean, std = agg[("lorenz", "seasonal_naive")]["d_h"]
    assert np.isfinite(mean) and std >= 0.0


def test_write_reports(tmp_path) -> None:
    rows = run_dsr_eval(
        ["seasonal_naive"], ["lorenz"], seeds=(0, 1), context_len=256, long_len=1200
    )
    cfg = MetricConfig()
    csv_path, md_path = write_reports(rows, str(tmp_path / "out"), cfg=cfg)
    assert csv_path.endswith("results.csv") and md_path.endswith("summary.md")
    csv_text = (tmp_path / "out" / "results.csv").read_text()
    assert "d_stsp" in csv_text and "seasonal_naive" in csv_text
    md = summary_markdown(rows, cfg=cfg, n_seeds=2)
    assert "## lorenz" in md and "D_stsp" in md and "reference lambda_max" in md
