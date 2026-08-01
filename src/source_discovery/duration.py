"""ISO-8601 duration parsing — stdlib only, no network, no optional deps.

``scraper._period_seconds`` is the original home of this parser, but importing
it drags in the whole scraper module, which hard-fails at import time without
``httpx``/``pyarrow``. That is fine inside the scrape job and wrong everywhere
else: ``--coverage`` and ``--audit`` are deterministic, offline analyses of the
catalog and must run on a bare ``pyyaml`` install. Duplicating ~six lines of
regex is cheaper than making every caller install an HTTP client;
``test_period_seconds_matches_scraper`` pins the two implementations together.
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
