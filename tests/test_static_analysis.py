"""Static-analysis gate: flag hardcoded answers, network, dynamic exec, writes."""

from __future__ import annotations

from static_analysis import ARRAY_FLAG_THRESHOLD, scan_submission

CLEAN = (
    "import numpy as np\n"
    "def forecast(context):\n"
    "    level = np.mean(context[-12:])\n"
    "    slope = (context[-1] - context[0]) / len(context)\n"
    "    return level + slope * np.arange(1, 49)\n"
)


def test_clean_numpy_submission_has_no_findings() -> None:
    assert scan_submission(CLEAN) == []


def test_cheating_submission_has_at_least_three_findings() -> None:
    nums = ", ".join(str(i) for i in range(ARRAY_FLAG_THRESHOLD + 2))
    code = (
        "import numpy as np\n"
        "import requests\n"
        f"ANSWERS = [{nums}]\n"
        "def forecast(context):\n"
        "    return np.array(eval('ANSWERS'))\n"
    )
    findings = scan_submission(code)
    assert len(findings) >= 3


def test_each_detector_fires() -> None:
    assert any("network" in f for f in scan_submission("import socket"))
    assert any("network" in f for f in scan_submission("from urllib import request"))
    assert any("dynamic" in f for f in scan_submission("exec('print(1)')"))
    assert any("dynamic" in f for f in scan_submission("__import__('os')"))
    assert any("file write" in f for f in scan_submission("open('cache','w')"))
    assert any("file write" in f for f in scan_submission("open('c', mode='a')"))

    big = "[" + ", ".join("1.0" for _ in range(ARRAY_FLAG_THRESHOLD)) + "]"
    assert any("array" in f for f in scan_submission(f"X = {big}"))


def test_threshold_boundary() -> None:
    just_under = "[" + ", ".join("1" for _ in range(ARRAY_FLAG_THRESHOLD - 1)) + "]"
    assert not any("array" in f for f in scan_submission(f"X = {just_under}"))
    nested = "[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]"
    assert any("array" in f for f in scan_submission(f"GRID = {nested}"))


def test_read_only_open_not_flagged() -> None:
    assert scan_submission("open('data.csv')") == []
    assert scan_submission("open('data.csv', 'r')") == []


def test_unparseable_submission_still_flagged() -> None:
    findings = scan_submission("import requests\ndef (:bad syntax")
    assert findings  # regex fallback catches the network import
