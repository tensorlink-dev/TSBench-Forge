"""Data-source discovery agent: propose new, uncontaminated catalog sources.

An LLM-driven curation tool that reviews the existing source catalog, finds
coverage gaps (domain / cadence / contamination), and proposes concrete new
sources to fill them. Its output is a candidate list, never a decision — and
almost all of the vetting is **automatic**:

* :mod:`vet` — deterministic *metadata* pre-filters on each proposal (schema,
  contamination denylist, duplicate of an existing source), before anything is
  fetched;
* :mod:`quality` — the deterministic *data* admission gate that runs on the
  actual fetched series (finite/variance/flatline/spike checks + a behavioural
  discrimination filter), with **no human and no LLM**.

A human is in the loop only where one is genuinely required — licensing / legal
sign-off for paywalled or contract sources. The agent never touches a forecast or
a score.

This is the *opposite* of the removed forge: not a bounded optimizer in the
consensus path, but an offline discovery step — vetted automatically by code —
where a non-deterministic model is exactly the right tool.
"""

from . import config, coverage, llm, quality, runner, vet

__all__ = ["config", "coverage", "llm", "quality", "runner", "vet"]
