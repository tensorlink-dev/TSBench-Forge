"""Discovery-agent configuration: the deterministic inputs and the vetting rules.

These are the knobs the agent reasons over and the deterministic code enforces.
The agent *proposes*; the values here are what its proposals are checked against.
Everything is data — no LLM, no network — so it is unit-testable in isolation.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Contamination denylist
# --------------------------------------------------------------------------- #
# Datasets known or strongly suspected to be in TSFM pretraining corpora. A
# proposal is rejected if its name/url matches any of these tokens (or an obvious
# repackaging). Lowercased substring match — deliberately broad, since the whole
# point is to catch "ETTh1 under a different name". Extend freely.
CONTAMINATION_DENYLIST: tuple[str, ...] = (
    "ett", "etth1", "etth2", "ettm1", "ettm2",
    "electricity", "ecl", "electricityloaddiagrams",
    "traffic", "pems",
    "weather",  # the ETT-companion "weather" bundle (not real met feeds — see note)
    "exchange", "exchange_rate",
    "ili", "illness",
    "m1", "m2", "m3", "m4", "m5",  # M-competitions
    "monash",
    "wikipedia web traffic", "web-traffic-time-series", "wikipedia pageviews dataset",
    "tourism",
    "solar-energy", "solar_al", "solar 10-minute",
    "london smart meters", "smart meters in london",
    "gluonts", "darts dataset", "sktime dataset",
    "lotsa",  # the aggregate pretraining corpus itself
)

# The denylist token "weather" is intentionally broad and *will* flag legitimate
# real weather feeds (METAR, open-meteo, …). Those are allowed via this allowlist
# of substrings — a proposal that matches "weather" but ALSO matches one of these
# is treated as a genuine live feed, not the ETT-companion bundle.
DENYLIST_ALLOW_OVERRIDES: tuple[str, ...] = (
    "metar", "open-meteo", "open_meteo", "noaa", "nws", "ndbc", "metoffice",
    "aemet", "dwd", "jma", "ecmwf", "meteostat", "synop",
)


# --------------------------------------------------------------------------- #
# Model training-data cutoffs (ESTIMATES — verify before trusting)
# --------------------------------------------------------------------------- #
# Data published strictly after max(these) is contamination-free by construction.
# These are best-effort public estimates of pretraining-corpus cutoffs for the
# TSFMs this benchmark evaluates; they are NOT authoritative. Treat as a floor and
# update from model cards. ISO date strings.
MODEL_CUTOFFS: dict[str, str] = {
    "chronos": "2024-01-01",     # Amazon Chronos (T5-based), 2024 release
    "timesfm": "2024-02-01",     # Google TimesFM
    "moirai": "2024-03-01",      # Salesforce Moirai / LOTSA corpus
    "timegpt": "2023-12-01",     # Nixtla TimeGPT
    "lag-llama": "2023-06-01",   # Lag-Llama
}


def contamination_free_after() -> str:
    """The date after which data is contamination-free for *all* evaluated models.

    ``max(MODEL_CUTOFFS)`` — a source whose data begins (or only has fresh rows)
    after this cannot be in any evaluated model's pretraining set.
    """
    return max(MODEL_CUTOFFS.values()) if MODEL_CUTOFFS else "1970-01-01"


# --------------------------------------------------------------------------- #
# Coverage taxonomy + targets
# --------------------------------------------------------------------------- #
# GIFT-Eval-style domains this benchmark stratifies over.
DOMAINS: tuple[str, ...] = (
    "nature", "econ_fin", "transport", "energy",
    "sales", "healthcare", "web_cloudops",
)

# Coarse cadence bands (mirrors ``scraped_source.FREQ_BAND`` so the coverage
# matrix here lines up with the sampler that actually serves the pool).
FREQ_BAND: dict[str, str] = {
    "PT30S": "sub-min", "PT1M": "sub-min", "PT2M30S": "sub-min",
    "PT5M": "few-min", "PT6M": "few-min", "PT10M": "few-min", "PT15M": "few-min",
    "PT30M": "half-hour",
    "PT1H": "hourly", "PT8H": "hourly",
    "P1D": "daily", "P1W": "weekly", "P1M": "monthly", "P1Q": "quarterly", "P1Y": "yearly",
}
CADENCE_BANDS: tuple[str, ...] = (
    "sub-min", "few-min", "half-hour", "hourly",
    "daily", "weekly", "monthly", "quarterly", "yearly", "irregular",
)

# Target number of sources per (domain × cadence-band) cell. A single global
# floor keeps the target legible; the runner reports which cells fall below it.
# Sub-daily live cells are weighted up because fresh, high-frequency feeds are the
# scarcest and most contamination-resistant.
TARGET_PER_CELL: int = 2
HIGH_VALUE_BANDS: tuple[str, ...] = ("sub-min", "few-min", "half-hour", "irregular")
TARGET_PER_HIGH_VALUE_CELL: int = 3


# --------------------------------------------------------------------------- #
# Candidate schema (what a well-formed proposal must contain)
# --------------------------------------------------------------------------- #
CANDIDATE_REQUIRED_FIELDS: tuple[str, ...] = (
    "name", "domain", "frequency", "access_method", "url_or_endpoint",
    "license", "estimated_series_count", "estimated_length",
    "first_available_date", "supports_live_future_tasks",
    "contamination_risk", "contamination_reasoning", "gap_filled",
    "difficulty_note", "adapter_notes", "confidence", "verify",
)
ACCESS_METHODS: tuple[str, ...] = (
    "open_api", "bulk_download", "live_stream", "licensed", "scrape",
)
RISK_LEVELS: tuple[str, ...] = ("low", "medium", "high")
CONFIDENCE_LEVELS: tuple[str, ...] = ("high", "medium", "low")
