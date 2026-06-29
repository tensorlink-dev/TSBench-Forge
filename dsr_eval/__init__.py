"""Dynamical-systems-reconstruction (DSR) evaluation for time-series foundation models.

A benchmark that scores a TSFM on whether it reconstructs the *dynamics* of a chaotic
system -- attractor geometry, long-term temporal structure, chaos signature -- not just
its short-horizon point error. See ``dsr_eval/README.md`` for the protocol, the metric
definitions, and the convergence/sensitivity caveats. Follows Durstewitz et al.,
"Why a Dynamical Systems Perspective is Needed to Advance Time Series Modeling"
(arXiv:2602.16864).

Entry points:

* :func:`dsr_eval.runner.run_dsr_eval` -- run the eval programmatically.
* ``python -m dsr_eval`` -- the CLI.
* :mod:`dsr_eval.metrics` -- the DSR metrics (``D_stsp``, ``D_H``, Lyapunov, ``D_KY``,
  ``VPT``), extending the repo's :mod:`dsr_metrics`.
* :mod:`dsr_eval.systems` / :mod:`dsr_eval.datasets` -- chaotic-system generators and
  the context / continuation / long-rollout split.
"""

from __future__ import annotations

from dsr_eval.datasets import DSRSample, build_dataset, build_sample
from dsr_eval.runner import MetricConfig, evaluate_sample, run_dsr_eval
from dsr_eval.systems import ChaoticSystem, available_systems, get_system, integrate

__all__ = [
    "ChaoticSystem",
    "DSRSample",
    "MetricConfig",
    "available_systems",
    "build_dataset",
    "build_sample",
    "evaluate_sample",
    "get_system",
    "integrate",
    "run_dsr_eval",
]
