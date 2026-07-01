# source_discovery — the catalog-curation agent

An LLM-driven **data-source discovery** tool. It keeps the benchmark's source
pool diverse and uncontaminated by reviewing what's in `sources.yaml`, ranking
the coverage gaps, and proposing concrete new sources to fill them.

Its output is a **vetted candidate list, never a decision.** Every proposal is
checked by deterministic code and — for anything licensed or paywalled — a human,
before a source is ever added or scraped. The agent never scores a model and never
touches a forecast.

> **How this differs from the removed forge.** The forge was an LLM wedged into
> the benchmark's *scoring/consensus path*, optimizing a handful of bounded
> scalars — a job a classical optimizer does exactly, so it was removed. This is
> the opposite: an *offline*, human-vetted *discovery* step, where a
> non-deterministic model is precisely the right tool and nothing it emits reaches
> a score or requires validator consensus.

## The loop

```
sources.yaml ──► coverage.py ──► (gaps) ──► llm.py (the agent) ──► vet.py ──► candidates.json
                deterministic                 proposals            deterministic   (human review
                                                                                     + scraper)
```

1. **`coverage.py`** loads the catalog, normalises each entry to the agent's
   `CURRENT_SOURCES` schema, and computes the domain × cadence coverage matrix and
   the ranked gap cells — all deterministic.
2. **`llm.py`** sends the system prompt (`system_prompt.md`) + the four inputs
   (`CURRENT_SOURCES`, `TARGET_COVERAGE`, `CONTAMINATION_DENYLIST`,
   `MODEL_CUTOFFS`) to a model and parses back a gap analysis + a JSON candidate
   array. This is the only non-deterministic step.
3. **`vet.py`** rejects/flags each candidate by machine-checkable rules: schema,
   contamination **denylist** (catches "ETTh1 under another name"), **duplicate**
   of an existing source (same host + domain), and a **contamination sanity**
   check of the agent's own risk claim vs. its `first_available_date` and the
   model cutoffs.

The second downstream gate the system prompt mentions — the **discrimination
filter** — can only run *after* a source is scraped (it needs real windows to
score with `score.panel_fitness`), so it lives in the main benchmark, not here.

## Usage

```bash
# Coverage + the biggest gaps (deterministic, no model, no key):
python -m source_discovery --coverage

# See the exact prompt the agent would receive (no model call):
python -m source_discovery --dry-run

# Full run — needs OPENROUTER_API_KEY. Writes gap_analysis.md + candidates.json:
export OPENROUTER_API_KEY=sk-or-...
export OPENROUTER_MODEL=anthropic/claude-opus-4.8   # default; any OpenRouter model
python -m source_discovery --out src/sources/discovered

# Vet a candidate list produced elsewhere (e.g. by an interactive Claude session):
python -m source_discovery --vet candidates.json --out src/sources/discovered
```

Because the deterministic half runs without a key, the tool is useful even with no
model wired: `--coverage` reports the gaps, and `--vet` runs the denylist / dedup /
schema checks on any candidate list — so an interactive agent can do the discovery
and this tool does the vetting.

## Configuration (`config.py`)

- `CONTAMINATION_DENYLIST` — known pretraining datasets (ETT, Electricity, Traffic,
  Monash, M1–M5, …) with an allow-override list so real weather/solar *feeds* are
  not caught by the ETT-companion bundle names.
- `MODEL_CUTOFFS` — **estimated** training cutoffs per evaluated TSFM; data after
  `max(...)` is contamination-free by construction. Update from model cards.
- `DOMAINS` / `CADENCE_BANDS` / `TARGET_PER_CELL` — the stratification and the
  per-cell source targets (sub-hourly-live and irregular cells weighted up).

## Output

Written under `--out` (default `src/sources/discovered/`, gitignored):

- `gap_analysis.md` — the agent's human-readable Phase-1/2 write-up.
- `candidates.json` — every proposal with `_verdict` (`accept`/`flag`/`reject`) and
  `_vet_reasons`. Accepted/flagged sort first. An operator turns accepted
  candidates into `sources.yaml` entries (then `scraper.py --id <new>` verifies).
