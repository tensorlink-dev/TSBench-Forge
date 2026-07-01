"""Data-source discovery agent: propose new, uncontaminated catalog sources.

An LLM-driven curation tool that reviews the existing source catalog, finds
coverage gaps (domain / cadence / contamination), and proposes concrete new
sources to fill them. Its output is a **vetted candidate list**, never a decision:
proposals are checked by deterministic code (:mod:`vet`) and a human before any
source is added or scraped, and the agent never touches a forecast or a score.

This is the *opposite* of the removed forge: not a bounded optimizer in the
consensus path, but an offline, human-vetted discovery step where a
non-deterministic model is exactly the right tool.
"""

from . import config, coverage, llm, runner, vet

__all__ = ["config", "coverage", "llm", "runner", "vet"]
