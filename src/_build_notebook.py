"""Build example.ipynb — a full walkthrough notebook for tsbench-forge."""

from __future__ import annotations

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells = []


def md(text: str) -> None:
    cells.append(new_markdown_cell(text.strip("\n")))


def code(src: str) -> None:
    cells.append(new_code_cell(src.strip("\n")))


md(r"""
# tsbench-forge — a guided walkthrough

`tsbench-forge` is a **hard-to-game benchmark for time-series foundation models
(TSFMs)** built on **real public data only**. The distribution the benchmark tests
is exactly the distribution of the ingested feeds (climate, energy, markets,
transport, …), drawn from the live catalog under `src/sources/` — there is no
synthetic generation.

Two halves work together:

1. **Assembling challenges (anti-gaming).** A fresh, breadth-balanced pool of
   *real* motifs is sampled and split into observed context / hidden truth, with
   challenges derived deterministically from a commit-reveal seed so every
   validator agrees. Freshness / as-of gating and light truth-preserving
   augmentation are the layers that keep it hard to memorise.
2. **The evaluation (the part that scores a model).** A probabilistic leaderboard
   (MASE / WQL / CRPS) that ranks a real TSFM against a reference panel, plus a
   *headroom* check that the benchmark can actually separate a better-than-classical
   model, and a *foundational breadth* read across real domains.

This notebook runs **numpy-only, fully offline** — no API key, no network, no
GPU. It reads locally-scraped parquet (`src/sources/data`) if present, else the
committed trimmed fixture (`tests/fixtures/sources_data`), exactly like
`src/demo.py`.
""")

code(r"""
import os, sys
# Make the tsbench-forge modules importable whether this runs from the repo
# root or from notebooks/ (the modules live in src/).
for _p in ("src", os.path.join("..", "src")):
    if os.path.isdir(_p):
        sys.path.insert(0, os.path.abspath(_p))
        break

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from config import CONTEXT_LEN, HORIZON
from scraped_source import ScrapedLiveSource
from ingest import FreshBuffer
from challenges import build_live_challenges
from seed import rng_for
from score import (
    default_panel, panel_fitness, validate_panel, domain_coverage,
    stratified_fitness, foundational_fitness,
)
from evaluate import leaderboard, probabilistic_panel, headroom, benchmark_has_headroom, ProbForecast
from sandbox import run_submission

BLOCK_HASH = "0xnotebook-demo"
MOTIF_LEN, POOL_SIZE, N_REVEAL = 384, 96, 128


def _data_dir():
    # Prefer freshly-scraped data; fall back to the committed trimmed fixture.
    here = os.path.abspath(sys.path[0])
    live = os.path.join(here, "sources", "data")
    fixture = os.path.join(here, os.pardir, "tests", "fixtures", "sources_data")
    if os.path.isdir(live) and any(
        f.endswith(".parquet") for _, _, fs in os.walk(live) for f in fs
    ):
        return live
    return os.path.abspath(fixture)


CATALOG = os.path.join(os.path.abspath(sys.path[0]), "sources", "sources.yaml")
print(f"context={CONTEXT_LEN}  horizon={HORIZON}  challenges={N_REVEAL}")
print(f"live catalog data: {_data_dir()}")
""")

md(r"""
## 1. A fresh, breadth-balanced pool of real motifs

The benchmark samples *real* windows from the live catalog. `ScrapedLiveSource`
reads the scraped parquet, and `FreshBuffer` builds a pool balanced across
domain × dgp_class × cadence — broad enough to certify a *foundation* model, not
just one process.
""")

code(r"""
source = ScrapedLiveSource(CATALOG, _data_dir(), min_series_length=MOTIF_LEN)
buffer = FreshBuffer(source, pool_size=POOL_SIZE, motif_len=MOTIF_LEN)
buffer.refresh(np.random.default_rng(0xC0FFEE))
print("pool domains    :", dict(Counter(buffer.pool_domains)))
print("pool dgp_classes:", len(set(buffer.pool_dgp_classes)), "classes")
print("pool cadences   :", dict(Counter(buffer.pool_cadences)))
""")

md(r"""
## 2. Commit-reveal: the challenges are a pure function of the seed

The concrete challenges are derived from `seed = H(block_hash ‖ epoch ‖ tag)`,
revealed only after miners commit. Every draw flows through the beacon-derived
`rng`, so a given `(pool, seed)` yields a byte-identical challenge set across all
validators — the cornerstone of commit-reveal consensus.
""")

code(r"""
reveal = build_live_challenges(buffer, rng_for(BLOCK_HASH, 0, "reveal"), N_REVEAL)
a = build_live_challenges(buffer, rng_for(BLOCK_HASH, 0, "chk"), 8)
b = build_live_challenges(buffer, rng_for(BLOCK_HASH, 0, "chk"), 8)
identical = all(np.array_equal(x.context, y.context) and np.array_equal(x.truth, y.truth)
                for x, y in zip(a, b))
print(f"revealed {len(reveal)} challenges")
print(f"two independent replays are byte-identical: {identical}")
""")

md(r"""
## 3. The reference panel & the validity gate

Validity comes from a *frozen reference panel*, not from trusting the data. A
challenge only counts if a panel of known-quality forecasters ranks in its known
order. The `strong` anchor must genuinely lead — `validate_panel` checks that
loudly, so a hollow anchor fails instead of silently making the gate meaningless.

> **Read this if `valid=False`.** On raw real data the *default numpy classical
> anchor* does not always lead the simple baselines. That is the validity gate
> doing its job: it is telling you the anchor is too weak to certify real-world
> skill. The fix is **not** to ignore it but to swap in an independently-validated
> zero-shot TSFM via `default_panel(strong_model=...)` — see `independent_eval.py`
> and `experiments/independent_validation.ipynb`.
""")

code(r"""
panel = default_panel()
res = panel_fitness(reveal, panel)
errs = res["errors"]
order = sorted(errs, key=errs.get)

plt.figure(figsize=(8, 4))
plt.bar(order, [errs[m] for m in order], color=["#2a9d8f" if m=="strong" else "#999" for m in order])
plt.title("Reference panel — mean normalised error (lower = better)")
plt.ylabel("normalised MAE"); plt.grid(alpha=0.3, axis="y"); plt.show()

vp = validate_panel(reveal, panel)
print(f"anchor validation: strong leads '{vp['runner_up']}' by {vp['margin']:+.3f}  -> valid={vp['valid']}")
if not vp["valid"]:
    print("NOTE: the numpy classical anchor does not lead on raw real data — expected;")
    print("      the fix is a real TSFM anchor via default_panel(strong_model=...).")
""")

md(r"""
## 4. Evaluating a model — the leaderboard (MASE / WQL / CRPS)

This is the half that **scores an actual TSFM**. A forecaster emits a
`ProbForecast` (mean + quantiles) and is judged on the metrics the TSFM
literature uses. Here we score the panel itself; to add a real model:

```python
from tsfm_adapters import load_tsfm      # pip install -e ".[chronos]"
models = {"chronos": load_tsfm("chronos"), **probabilistic_panel()}
leaderboard(models, reveal)
```
""")

code(r"""
board = leaderboard(probabilistic_panel(), reveal)
print(f"{'rank':>4}  {'model':<16}{'MASE':>8}{'WQL':>8}{'CRPS':>8}")
for r in board:
    print(f"{r['rank']:>4}  {r['model']:<16}{r['mase']:>8.3f}{r['wql']:>8.3f}{r['crps']:>8.3f}")

names = [r["model"] for r in board]
plt.figure(figsize=(8, 4))
plt.bar(names, [r["mase"] for r in board], color="#264653")
plt.title("Leaderboard — MASE (lower = better)"); plt.ylabel("MASE")
plt.grid(alpha=0.3, axis="y"); plt.show()
""")

md(r"""
## 5. Headroom — can the benchmark certify a *better* model?

A benchmark only certifies TSFMs if a genuinely better model scores measurably
better than the classical anchor. We inject a deliberately-superior probe (the
true future + small noise — better than any real model, but not perfect) and
confirm the benchmark rewards it. If this failed, no leaderboard here would be
trustworthy.
""")

code(r"""
probe_rng = np.random.default_rng(7)
truth_by_id = {id(ch.context): np.asarray(ch.truth, dtype=float) for ch in reveal}

def superior_probe(context, meta=None):
    tgt = truth_by_id[id(context)]
    noisy = tgt + probe_rng.normal(0.0, 0.05 * (np.std(tgt) + 1e-8), size=tgt.shape)
    deciles = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    return ProbForecast(mean=noisy, quantiles={q: noisy for q in deciles})

hr = headroom(superior_probe, reveal)
print(f"a superior model beats the anchor by {hr['mase_margin']:+.3f} MASE / "
      f"{hr['wql_margin']:+.3f} WQL")
print(f"benchmark_has_headroom = {benchmark_has_headroom(superior_probe, reveal)}")
""")

md(r"""
## 6. A sample challenge and the anchor's forecast

What a single challenge actually looks like: an observed context, the hidden
truth over the horizon, and the `strong` anchor's prediction.
""")

code(r"""
ch = reveal[0]
pred = panel["strong"](ch.context, getattr(ch, "meta", None))
n = len(ch.context)
plt.figure(figsize=(11, 4))
plt.plot(np.arange(n), ch.context, color="#264653", label="context (observed)")
plt.plot(np.arange(n, n+HORIZON), ch.truth, color="#2a9d8f", label="truth (hidden)")
plt.plot(np.arange(n, n+HORIZON), pred, "--", color="#e76f51", label="strong forecast")
plt.axvline(n, color="grey", ls=":", alpha=0.6)
plt.title(f"Sample challenge  (domain: {getattr(ch,'meta',{}).get('domain','?')})")
plt.legend(); plt.grid(alpha=0.3); plt.show()
""")

md(r"""
## 7. Submissions run in a sandbox, not just a linter

The static scan is a cheap pre-filter; the **real** boundary is isolated,
resource-limited execution. A clean forecaster runs; a hardcoded/cheating one is
rejected.
""")

code(r"""
clean = (
    "import numpy as np\n"
    "def forecast(context):\n"
    "    level = np.mean(context[-12:])\n"
    "    slope = (context[-1] - context[0]) / len(context)\n"
    f"    return level + slope * np.arange(1, {HORIZON + 1})\n"
)
table = ", ".join(str(round(0.1*i, 2)) for i in range(40))
cheat = ("import numpy as np\nimport requests\n"
         f"ANSWERS = [{table}]\n"
         "def forecast(context):\n    return np.array(eval('ANSWERS'))\n")

for name, code_str in (("clean numpy forecaster", clean), ("hardcoded/cheating", cheat)):
    r = run_submission(code_str, reveal[0].context)
    detail = f"shape={r.prediction.shape}" if r.ok else (r.error or r.findings)
    print(f"{name:<24} -> status={r.status}  ({detail})")
""")

md(r"""
## 8. Foundational breadth — good *across worlds*, not one process

A high score should mean "generalises across domains." Coverage is measured (the
effective number of data-generating processes), breadth gates check the pool
spans enough dgp_classes and cadences, and fitness is reported per domain so
narrowness can't hide in an average. Recall `fitness = spread · max(0, ordering)`.
""")

code(r"""
cov = domain_coverage(reveal)
fnd = foundational_fitness(
    reveal, panel,
    dgp_classes=buffer.pool_dgp_classes,
    cadences=buffer.pool_cadences,
)
print(f"spans {cov['n_domains']} real domains; effective (exp-entropy) = {cov['effective_domains']:.2f}")
print(f"coverage_gate={fnd['coverage_gate']:.2f}  "
      f"dgp_class_breadth_gate={fnd['dgp_class_breadth_gate']:.0f}  "
      f"cadence_breadth_gate={fnd['cadence_breadth_gate']:.0f}  "
      f"parrot_gate={fnd['parrot_gate']:.2f}")

print(f"\n{'domain':<16}{'n':>4}{'spread':>8}{'order':>8}{'strong':>8}")
for dom, m in sorted(stratified_fitness(reveal, panel).items(), key=lambda kv: -kv[1]['n']):
    print(f"{dom:<16}{m['n']:>4}{m['spread']:>8.2f}{m['ordering']:>+8.2f}{m['difficulty']:>8.2f}")
""")

md(r"""
## Recap

- **Live catalog** → challenges are real motifs, not synthetic; the tested
  distribution is exactly the ingested feeds.
- **Commit-reveal** → identical, reproducible challenges for every validator.
- **Panel + `validate_panel`** → challenges only count when the anchor genuinely
  leads (numpy anchor `valid=False` on raw real data is expected; fix is a real
  TSFM anchor).
- **Leaderboard (MASE/WQL/CRPS) + `load_tsfm`** → scores an actual TSFM.
- **Headroom** → confirms the benchmark can certify a better-than-classical model.
- **Sandbox** → submissions are executed in isolation, not just linted.
- **Foundational breadth** → valid and discriminating *across* real domains.

Swap in a real anchor (`default_panel(strong_model=...)`) and a real model
(`tsfm_adapters.load_tsfm`) for production. See `src/demo.py` for the same flow as
a script, and `experiments/` for the real-live-feed and independent-validation
walkthroughs.
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
import os

_out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks", "example.ipynb")
nbf.write(nb, _out)
print(f"wrote {_out} with {len(cells)} cells")
