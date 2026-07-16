# source_discovery — the catalog-curation agent

An LLM-driven **data-source discovery** tool. It keeps the benchmark's source
pool diverse and uncontaminated by reviewing what's in `sources.yaml`, ranking
the coverage gaps, and proposing concrete new sources to fill them.

Its output is a candidate list, never a decision — and the vetting is
**automatic**. Every proposal is checked by deterministic code at two stages
(metadata before fetching, then the actual data after), with a human in the loop
only where one is genuinely required: licensing / legal sign-off for paywalled or
contract sources. The agent never scores a model and never touches a forecast.

> **Why an LLM here is safe.** This is an *offline*, automatically-vetted
> *discovery* step: nothing the model emits reaches a score or requires validator
> consensus, so a non-deterministic model is precisely the right tool for the
> open-ended judgement the task needs.

## The loop

```
sources.yaml ─► coverage.py ─► llm.py (agent) ─► vet.py ─────► scraper.py ─► quality.py ──► rotation
              deterministic     proposals       metadata vet   (fetch)      DATA auto-vet
              (gap matrix)                       (pre-fetch)                 (post-fetch)
```

1. **`coverage.py`** loads the catalog, normalises each entry to the agent's
   `CURRENT_SOURCES` schema, and computes the domain × cadence coverage matrix and
   the ranked gap cells — all deterministic.
2. **`llm.py`** sends the system prompt (`system_prompt.md`) + the four inputs
   (`CURRENT_SOURCES`, `TARGET_COVERAGE`, `CONTAMINATION_DENYLIST`,
   `MODEL_CUTOFFS`) to a model and parses back a gap analysis + a JSON candidate
   array. This is the only non-deterministic step.
3. **`vet.py`** — *metadata* pre-filters (before fetching): schema, contamination
   **denylist** (catches "ETTh1 under another name"), **duplicate** of an existing
   source (same host + domain), and a contamination-claim sanity check.
4. **`quality.py`** — the *data* admission gate (after the proposal is scraped),
   fully automatic:
   - **intrinsic per-series checks** — non-finite (NaN/inf) fraction, length,
     constant / near-constant **variance**, degenerate value-cardinality (stuck /
     quantized sensor), flatline runs, single-spike variance domination;
   - **a discrimination filter** — reject if the series are **unforecastable**
     (near-zero autocorrelation ⇒ pure noise) or **trivially solved** (a
     seasonal-naive copy already nails them). SNR and spectral flatness are
     reported as *diagnostics*, not hard gates, because real noisy feeds and pure
     noise overlap on those — the behavioural filter is the honest test.

Both vetting stages run with **no human and no LLM**. `quality.py` is the concrete
form of the system prompt's "discrimination filter + leakage check" for the data
itself; it reuses the benchmark's own panel, so "admittable" means the same thing
as "a useful challenge".

## Usage

```bash
# Coverage + the biggest gaps (deterministic, no model, no key):
python -m source_discovery --coverage

# See the exact prompt the agent would receive (no model call):
python -m source_discovery --dry-run

# Full run — needs OPENROUTER_API_KEY. Writes gap_analysis.md + candidates.json:
export OPENROUTER_API_KEY=sk-or-...
export OPENROUTER_MODEL=z-ai/glm-5.2   # default; any OpenRouter model
python -m source_discovery --out src/sources/discovered

# Vet a candidate list produced elsewhere (e.g. by an interactive Claude session):
python -m source_discovery --vet candidates.json --out src/sources/discovered

# Auto-assess the DATA of an already-scraped source (the admission gate):
python -m source_discovery --assess aemo_nem_5min --data-dir src/sources/data
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
