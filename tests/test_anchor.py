"""Anchor validation: a hollow `strong` is detected, a good one passes."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import structured_challenges
from config import HORIZON
from score import (
    AnchorValidationError,
    default_panel,
    panel_from_env,
    validate_panel,
)


def _challenges(n: int = 96):
    # Structured trend+seasonal+AR series where the strong anchor genuinely earns
    # its lead — the right setting to test that validation passes a good anchor.
    return structured_challenges(n, seed=1)


def test_default_anchor_validates_on_structured_data() -> None:
    report = validate_panel(_challenges())
    assert report["valid"]
    assert report["runner_up"] is not None
    assert report["margin"] >= 0.02


def test_hollow_anchor_is_flagged() -> None:
    # Swap `strong` for a deliberately useless flat-zero forecaster: it cannot
    # lead the baselines, so validation must fail.
    def hollow(context, meta=None):
        return np.zeros(HORIZON)

    panel = default_panel(strong_model=hollow)
    report = validate_panel(_challenges(), panel)
    assert not report["valid"]


def test_require_raises_on_hollow_anchor() -> None:
    def hollow(context, meta=None):
        return np.zeros(HORIZON)

    panel = default_panel(strong_model=hollow)
    with pytest.raises(AnchorValidationError):
        validate_panel(_challenges(), panel, require=True)


def test_panel_from_env_defaults_to_numpy_anchor() -> None:
    panel = panel_from_env({})
    assert set(panel) == set(default_panel())
    # Unknown choice also falls back to the numpy default rather than erroring.
    panel2 = panel_from_env({"TSBENCH_STRONG": "nonexistent-model"})
    assert "strong" in panel2
