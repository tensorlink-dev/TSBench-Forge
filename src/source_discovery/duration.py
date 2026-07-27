"""ISO-8601 duration parsing, with no dependencies beyond the stdlib.

``coverage`` and ``audit`` both need to turn a catalog ``frequency`` string
into seconds, and both used to reach into ``src/sources/scraper.py`` for it.
That import is expensive in the wrong way: scraper.py hard-requires ``httpx``
and ``pyarrow`` at module scope, so a deterministic, network-free command like
``--coverage`` died with ``ModuleNotFoundError`` anywhere those two are not
installed — including the Auto-search workflow, which installs only pyyaml and
numpy by design (it never scrapes). The parse itself is a regex over a string;
it does not belong behind an HTTP client.

``scraper._period_seconds`` keeps its own copy so that ``python scraper.py``
stays runnable without ``src`` on the path. The two are pinned together by
``tests/test_source_discovery.py::test_period_seconds_matches_scraper``, which
runs whenever httpx/pyarrow are importable.
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
