"""Deterministic vetting of agent proposals — the code half of the loop.

The agent proposes; this rejects or flags each candidate with *machine-checkable*
rules before any human looks or any scraper runs:

* **schema**       — required fields present, enums valid;
* **denylist**     — name/url matches a known pretraining dataset (or a
                     repackaging of one) → hard reject;
* **duplicate**    — same domain and host/url as a source already in rotation;
* **contamination**— the agent's own risk claim is sanity-checked against its
                     ``first_available_date`` vs the model cutoffs and against
                     the "too easy to find" heuristic.

This is the "leakage check" the system prompt says runs downstream. The other
downstream gate — the *discrimination filter* — can only run after a source is
actually scraped (it needs real windows to score with ``score.panel_fitness``),
so it lives in the main benchmark, not here; ``verdict`` never claims a source
discriminates, only that it is safe and novel enough to be worth scraping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse

from . import config


@dataclass
class VetResult:
    """Outcome of vetting one candidate."""

    candidate: dict
    ok: bool
    verdict: str  # "accept" | "flag" | "reject"
    reasons: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return str(self.candidate.get("name", "<unnamed>"))


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""


def _denylisted(text: str) -> str | None:
    """Return the offending denylist token, or None. Honours the allow-overrides."""
    t = text.lower()
    if any(ok in t for ok in config.DENYLIST_ALLOW_OVERRIDES):
        # A legitimate live feed that merely contains a denylist word (e.g. a real
        # NOAA "weather" station) — not the ETT-companion bundle.
        deny_hits = [d for d in config.CONTAMINATION_DENYLIST if d in t]
        real_hits = [d for d in deny_hits if d not in ("weather", "solar-energy")]
        return real_hits[0] if real_hits else None
    for d in config.CONTAMINATION_DENYLIST:
        # Word-ish match: bare tokens like "m4" shouldn't fire inside "pm4x"; use
        # a boundary check for short alphanumeric tokens, substring for phrases.
        if len(d) <= 3 or d.isalnum() and len(d) <= 4:
            import re
            if re.search(rf"(?<![a-z0-9]){re.escape(d)}(?![a-z0-9])", t):
                return d
        elif d in t:
            return d
    return None


def _schema_errors(c: dict) -> list[str]:
    errs: list[str] = []
    for fld in config.CANDIDATE_REQUIRED_FIELDS:
        if fld not in c or c[fld] in (None, ""):
            errs.append(f"missing/empty field '{fld}'")
    if c.get("access_method") and c["access_method"] not in config.ACCESS_METHODS:
        errs.append(f"invalid access_method '{c['access_method']}'")
    if c.get("contamination_risk") and c["contamination_risk"] not in config.RISK_LEVELS:
        errs.append(f"invalid contamination_risk '{c['contamination_risk']}'")
    if c.get("confidence") and c["confidence"] not in config.CONFIDENCE_LEVELS:
        errs.append(f"invalid confidence '{c['confidence']}'")
    return errs


def vet_candidate(c: dict, registry: list[dict], ledger: dict | None = None) -> VetResult:
    """Vet one proposal against the schema, denylist, registry, ledger, and cutoffs."""
    reasons: list[str] = []

    # 0) Ledger — a (domain, host) that was already proposed in ANY prior run is
    # a wasted proposal: reject so repeats never reach a human reviewer twice.
    if ledger:
        from . import ledger as _ledger

        prior = ledger.get(_ledger.candidate_key(c))
        if prior is not None:
            return VetResult(c, ok=False, verdict="reject", reasons=[
                f"already proposed on {prior.get('first_proposed')} "
                f"(status={prior.get('status')}, seen {prior.get('times_proposed')}x) "
                "— propose a genuinely new source"
            ])

    # 1) Schema — a malformed proposal can't be trusted or acted on.
    schema_errs = _schema_errors(c)
    if schema_errs:
        return VetResult(c, ok=False, verdict="reject", reasons=schema_errs)

    # 2) Denylist — a known-contaminated dataset (or repackaging) is a hard no.
    hay = f"{c.get('name', '')} {c.get('url_or_endpoint', '')} {c.get('contamination_reasoning', '')}"
    hit = _denylisted(hay)
    if hit is not None:
        return VetResult(c, ok=False, verdict="reject",
                         reasons=[f"matches contamination denylist token '{hit}'"])

    # 3) Duplicate — same domain + same host as an existing source.
    host = _host(c.get("url_or_endpoint", ""))
    if host:
        for s in registry:
            if s["domain"] == c.get("domain") and _host(s.get("url_or_endpoint", "")) == host:
                reasons.append(f"same host+domain as existing source '{s['id']}' ({host})")
                break

    # 4) Contamination sanity — the agent's own claim vs the evidence.
    live = bool(c.get("supports_live_future_tasks"))
    fad = str(c.get("first_available_date", ""))
    cutoff = config.contamination_free_after()
    risk = c.get("contamination_risk")
    if risk == "low" and not live:
        # A low-risk claim on non-live data must be justified by a post-cutoff date.
        if not (fad[:10] > cutoff):
            reasons.append(
                f"claims low contamination but is not live and first_available_date "
                f"'{fad}' is not clearly after model cutoff {cutoff} — justify or downgrade"
            )
    if c.get("confidence") == "high" and "verify" in c and not str(c["verify"]).strip():
        reasons.append("confidence=high with an empty verify note — high-confidence "
                       "claims still need a check")

    verdict = "flag" if reasons else "accept"
    return VetResult(c, ok=True, verdict=verdict, reasons=reasons)


def vet_all(
    candidates: list[dict], registry: list[dict], ledger: dict | None = None
) -> list[VetResult]:
    """Vet a candidate list; accepted/flagged first, rejected last."""
    results = [vet_candidate(c, registry, ledger) for c in candidates]
    order = {"accept": 0, "flag": 1, "reject": 2}
    results.sort(key=lambda r: order[r.verdict])
    return results
