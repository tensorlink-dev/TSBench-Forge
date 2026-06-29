"""Build experiments/independent_validation.ipynb — the anchor-validation walkthrough.

Dependency-free builder (writes nbformat-v4 JSON directly). Running the notebook
needs the ``notebook`` extra; to validate a *real TSFM* also install ``.[chronos]``
and stage weights, and ``.[strong]`` enables the statsforecast fallback used here.

    python experiments/build_independent_validation_notebook.py
"""

from __future__ import annotations

import json
import os

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(src: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": src.strip("\n"),
        }
    )


md(r"""
# Proving an **independently-validated** anchor (TSFM-ready)

The benchmark's validity rests on the `strong` anchor genuinely being good — but
checking that on the forge's *own* challenges is circular. This notebook does the
non-circular thing: it establishes the anchor's quality on a **held-out, real,
external benchmark the forge never touches** (commodity / transport / demography),
then promotes the validated anchor and confirms it also leads on the forge.

It resolves the **strongest anchor this environment can run**: a real TSFM
(Chronos/TimesFM) if `.[chronos]` + weights are present, else a literature-
validated `statsforecast` model (`.[strong]`), else the numpy placeholder — and
tells you which. So the *same* notebook validates a real neural TSFM where one is
installed, and a real classical model otherwise.

**Requirements:** `pip install -e ".[notebook,strong]"` (add `chronos` for a TSFM)
and network on first run (feeds are then cached).
""")

code(r"""
import sys, os
# tsbench-forge modules live in src/; make them importable from repo root or experiments/.
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "src")))

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from config import WEAK_STATE
from evaluate import leaderboard, probabilistic_panel
from forge_loop import run_forge, manifest_for
from generate import build_challenges
from seed import rng_for
from score import validate_panel
from independent_eval import (
    VALIDATION_REGISTRY, validation_benchmark, resolve_anchor,
    is_independently_validated, validated_panel, leakage_gap, forge_buffer,
)

anchor = resolve_anchor()           # TSFM -> statsforecast -> numpy
print(f"Resolved anchor: [{anchor.kind}]  {anchor.detail}")
for name, spec in VALIDATION_REGISTRY.items():
    print(f"  held-out: {name:<20} {spec['note']}")
""")

md(r"""
## 1. The held-out external benchmark

Real series in domains **disjoint from the forge feed** — so a good score here is
independent evidence of skill, not a property of the forge's distribution.
""")

code(r"""
bench = validation_benchmark(n_per_domain=24)
print("held-out tasks per domain:", dict(Counter(c.meta["domain"] for c in bench)))

doms = list(VALIDATION_REGISTRY)
fig, axes = plt.subplots(len(doms), 1, figsize=(11, 2.0*len(doms)), sharex=True)
for ax, dom in zip(axes, doms):
    ch = next(c for c in bench if c.meta["domain"] == dom)
    ax.plot(np.arange(len(ch.context)), ch.context, color="#264653")
    ax.plot(np.arange(len(ch.context), len(ch.context)+len(ch.truth)), ch.truth, color="#2a9d8f")
    ax.set_ylabel(dom, rotation=0, ha="right", va="center"); ax.grid(alpha=0.3)
axes[0].set_title("Held-out tasks: context (dark) + horizon truth (green)"); plt.tight_layout(); plt.show()
""")

md(r"""
## 2. Independent validation — the anchor must beat every classical baseline

Zero-shot on the held-out set (nothing is fit to the forge). If the anchor does
not lead, it is **not** fit to certify TSFM quality — the gate the README demands.
""")

code(r"""
res = is_independently_validated(anchor.forecaster, name=f"anchor[{anchor.kind}]", benchmark=bench)
print(f"{'rank':>4}  {'model':<22}{'MASE':>9}{'WQL':>8}{'CRPS':>8}")
for r in res.board:
    star = "  <- anchor" if r["model"].startswith("anchor[") else ""
    print(f"{r['rank']:>4}  {r['model']:<22}{r['mase']:>9.3f}{r['wql']:>8.3f}{r['crps']:>8.3f}{star}")
print(f"\n=> {'VALIDATED' if res.validated else 'NOT validated'}: "
      f"leads best baseline by {res.margin:+.3f} MASE"
      + (f"  (beaten by {res.beaten_by})" if res.beaten_by else ""))

names = [r["model"] for r in res.board]
colors = ["#2a9d8f" if n.startswith("anchor[") else "#999" for n in names]
plt.figure(figsize=(8,4)); plt.bar(names, [r["mase"] for r in res.board], color=colors)
plt.xticks(rotation=30, ha="right"); plt.ylabel("MASE (held-out, lower=better)")
plt.title("Independent validation on the external held-out set"); plt.grid(alpha=0.3, axis="y"); plt.show()
""")

md(r"""
## 3. Harden a forge benchmark, then promote the validated anchor

The forge runs with the cheap numpy panel (its job is *hardening*); we then promote
the **independently-validated** anchor to `strong` and confirm it still leads on the
forge's own reveal via `validate_panel`. Because the anchor's quality was
established externally in step 2, this is no longer circular.
""")

code(r"""
buffer = forge_buffer(pool_size=96, motif_len=384)
buffer.refresh(np.random.default_rng(0xC0FFEE))
final_state, log = run_forge(buffer, 12, "0xindep-nb", WEAK_STATE)
mhash = manifest_for(final_state)
reveal = build_challenges(final_state, buffer, rng_for("0xindep-nb", 12, mhash), 96)
print(f"forge fitness {log[0].fitness:.3f} -> {log[-1].fitness:.3f}; revealed {len(reveal)} challenges")

panel, _ = validated_panel(anchor)
vp = validate_panel(reveal, panel)
print(f"validate_panel on forge reveal: runner-up '{vp['runner_up']}', "
      f"anchor leads by {vp['margin']:+.3f} -> valid={vp['valid']}")
""")

md(r"""
## 4. Where the validated anchor lands on the leaderboard, and the leakage gap
""")

code(r"""
models = {f"anchor[{anchor.kind}]": anchor.forecaster, **probabilistic_panel()}
board = leaderboard(models, reveal)
print(f"{'rank':>4}  {'model':<22}{'MASE':>8}{'WQL':>8}")
for r in board:
    print(f"{r['rank']:>4}  {r['model']:<22}{r['mase']:>8.3f}{r['wql']:>8.3f}")

gap = leakage_gap(anchor.forecaster, reveal, benchmark=bench)
print(f"\nleakage detector  held-out MASE={gap['static_mase']:.3f}  "
      f"forge MASE={gap['forge_mase']:.3f}  gap={gap['gap']:+.3f}")
print("Track this gap across epochs: a forge score racing ahead of the held-out score == leakage.")
""")

md(r"""
## 5. Validating a real neural TSFM

Everything above is anchor-agnostic. To validate **Chronos** (or TimesFM):

```bash
pip install -e ".[chronos]"      # torch + chronos-forecasting
# stage weights once (validator-side), e.g. set HF_HOME to a pre-downloaded cache
```

```python
anchor = resolve_anchor(prefer_tsfm="chronos")   # now resolves to the real TSFM
```

and re-run the notebook. The held-out leaderboard in step 2 is then the
**independent validation of a real TSFM**; step 3 promotes it to the benchmark's
validity anchor. Nothing else changes.

## Recap

- **Independent** = quality established on a held-out, real, external set the forge
  never touches — breaking the circularity of self-validation.
- `resolve_anchor` always runs the strongest anchor available (TSFM → classical →
  numpy) and reports which, so a run is never silently on the placeholder.
- A validated anchor makes `validate_panel` meaningful; the leakage gap keeps it
  honest over time.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "independent_validation.ipynb")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)
print(f"wrote {out} with {len(cells)} cells")
