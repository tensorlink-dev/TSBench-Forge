"""Sandboxed submission execution: valid forecasts pass, abuse fails closed."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from config import HORIZON
from sandbox import SandboxLimits, run_submission

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX rlimits only")

_CLEAN = (
    "import numpy as np\n"
    "def forecast(context):\n"
    "    level = np.mean(context[-12:])\n"
    "    slope = (context[-1] - context[0]) / len(context)\n"
    f"    return level + slope * np.arange(1, {HORIZON + 1})\n"
)


def _ctx() -> np.ndarray:
    return np.linspace(0.0, 10.0, 256)


def test_clean_submission_runs_and_returns_valid_forecast() -> None:
    res = run_submission(_CLEAN, _ctx())
    assert res.ok and res.status == "ok"
    assert res.prediction is not None
    assert res.prediction.shape == (HORIZON,)
    assert np.all(np.isfinite(res.prediction))


def test_prefilter_rejects_hardcoded_submission_without_executing() -> None:
    table = ", ".join(str(0.1 * i) for i in range(40))
    cheat = f"import requests\nANSWERS=[{table}]\ndef forecast(c):\n    return ANSWERS\n"
    res = run_submission(cheat, _ctx())
    assert not res.ok and res.status == "rejected"
    assert res.findings  # static analysis caught it pre-execution


def test_wrong_length_forecast_is_an_error() -> None:
    bad = "import numpy as np\ndef forecast(c):\n    return np.zeros(3)\n"
    res = run_submission(bad, _ctx())
    assert not res.ok and res.status == "error"
    assert "length" in (res.error or "")


def test_nonfinite_forecast_is_an_error() -> None:
    bad = (
        "import numpy as np\n"
        "def forecast(c):\n"
        f"    return np.full({HORIZON}, np.inf)\n"
    )
    res = run_submission(bad, _ctx())
    assert not res.ok and res.status == "error"


def test_submission_exception_is_captured() -> None:
    bad = "def forecast(c):\n    raise RuntimeError('kaboom')\n"
    res = run_submission(bad, _ctx())
    assert not res.ok and res.status == "error"
    assert "kaboom" in (res.error or "")


def test_timeout_kills_runaway_submission() -> None:
    spin = "def forecast(c):\n    while True:\n        pass\n"
    res = run_submission(spin, _ctx(), limits=SandboxLimits(cpu_seconds=2, wall_seconds=2.0))
    assert not res.ok and res.status in {"timeout", "error"}


def test_network_is_blocked_in_sandbox() -> None:
    # A submission that survives the pre-filter (dynamic import) but tries to use
    # the network at runtime must fail closed.
    net = (
        "import numpy as np\n"
        "def forecast(c):\n"
        "    mod = __import__('socket')\n"
        "    mod.socket()\n"
        f"    return np.zeros({HORIZON})\n"
    )
    # __import__ trips the pre-filter; disable it to test the runtime guard.
    res = run_submission(net, _ctx(), prefilter=False)
    assert not res.ok and res.status == "error"
    assert "network" in (res.error or "").lower() or "socket" in (res.error or "").lower()
