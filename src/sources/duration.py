"""ISO-8601 duration parsing — the one definition, with no heavy dependencies.

Split out of ``scraper.py`` so the deterministic, network-free consumers can
band a cadence without importing the scraper. ``scraper`` imports httpx and
pyarrow at module scope and *raises* when they are missing, so a five-line
duration parse used to drag the entire ingestion stack into
``source_discovery.coverage`` and ``source_discovery.audit`` — which is why the
free ``--coverage`` gate (whose CI job installs only pyyaml + numpy) died at
import time and emitted nothing on stdout.
"""

from __future__ import annotations

import re

_DUR_RE = re.compile(
    r"^P(?:(?P<y>\d+)Y)?(?:(?P<mo>\d+)M)?(?:(?P<w>\d+)W)?(?:(?P<d>\d+)D)?"
    r"(?:T(?:(?P<h>\d+)H)?(?:(?P<mi>\d+)M)?(?:(?P<s>\d+)S)?)?$"
)
_DUR_UNITS = {"y": 31_536_000, "mo": 2_592_000, "w": 604_800, "d": 86_400,
              "h": 3600, "mi": 60, "s": 1}


def period_seconds(freq: str) -> int | None:
    """ISO-8601 duration -> seconds. None if unparseable (e.g. 'irregular')."""
    m = _DUR_RE.match(str(freq or "").strip())
    if not m or not any(m.groupdict().values()):
        return None
    return sum(int(v) * _DUR_UNITS[k] for k, v in m.groupdict().items() if v)
