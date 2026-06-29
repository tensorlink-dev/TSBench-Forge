"""Build experiments/live_feeds.ipynb — the real-data walkthrough notebook.

Deliberately dependency-free: it writes nbformat-v4 JSON directly (no ``nbformat``
needed to *build* the notebook). Running the resulting notebook needs the
``notebook`` extra (``pip install -e ".[notebook]"``) for matplotlib/jupyter, and
network access to the public feeds on first run (results are then cached).

    python experiments/build_live_feeds_notebook.py
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
# tsbench-forge on **real data** — a live-feed experiment

The companion to `example.ipynb`, but every "live" motif here is a **real public
time series** pulled through `live_feeds.py`, not the synthetic zoo. We run the
whole pipeline on them: build a multi-domain real feed, run the forge,
commit-reveal, validate the panel, score a MASE/WQL/CRPS leaderboard, check
headroom, and read per-domain breadth.

**Requirements:** `pip install -e ".[notebook]"` and outbound access to the feed
hosts on first run (bodies are then cached under `~/.cache/tsbench-forge/feeds`,
so reruns are offline). If a host is blocked, the feed build raises — set
`TSBENCH_FEED_CACHE` to a pre-populated directory, or fall back to
`domains.default_live_source()`.
""")

code(r"""
import sys, os
# Make the tsbench-forge modules (in src/) importable whether this runs from
# the repo root or from experiments/.
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "src")))

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from config import WEAK_STATE, CONTEXT_LEN, HORIZON
from ingest import FreshBuffer
from forge_loop import run_forge, manifest_for
from generate import build_challenges
from seed import rng_for
from score import default_panel, panel_fitness, validate_panel, domain_coverage, stratified_fitness
from evaluate import leaderboard, probabilistic_panel, headroom, benchmark_has_headroom, ProbForecast
from live_feeds import REGISTRY, build_real_live_source, make_feed, cached_fetch

BLOCK_HASH = "0xlive-notebook"
EPOCHS, MOTIF_LEN, POOL = 18, 384, 96
print(f"context={CONTEXT_LEN} horizon={HORIZON} epochs={EPOCHS}")
for name, spec in REGISTRY.items():
    print(f"  {name:<18} {spec.domain:<18} {spec.note}")
""")

md(r"""
## 1. The real feeds

Five genuinely distinct real-world processes — climate, solar activity,
atmospheric CO₂, equity prices, weather. A shared cached fetcher means each URL is
pulled at most once. Below, one z-scored window from each.
""")

code(r"""
fetch = cached_fetch()          # on-disk cache; first run hits the network
fig, axes = plt.subplots(len(REGISTRY), 1, figsize=(11, 2.0*len(REGISTRY)), sharex=True)
for ax, name in zip(axes, REGISTRY):
    feed = make_feed(name, fetch=fetch)
    win = feed.pull(1, MOTIF_LEN, np.random.default_rng(0))[0]
    ax.plot(win, color="#264653"); ax.set_ylabel(name, rotation=0, ha="right", va="center")
    ax.grid(alpha=0.3)
axes[0].set_title("One z-scored window from each real feed"); plt.tight_layout(); plt.show()
""")

md(r"""
## 2. A real multi-domain pool

`build_real_live_source()` blends the registry into one `MixtureLiveSource` — the
real-data analogue of `domains.default_live_source()`.
""")

code(r"""
source = build_real_live_source(fetch=fetch)
buffer = FreshBuffer(source, pool_size=POOL, motif_len=MOTIF_LEN)
buffer.refresh(np.random.default_rng(0xC0FFEE))
print("pool domains:", dict(Counter(buffer.pool_domains)))
""")

md(r"""
## 3. Run the forge — on real data

Starting from a deliberately weak (synthetic-heavy, pre-fittable) state, the forge
keeps one-knob moves only when the panel becomes more discriminating-and-valid.
`fitness = spread · max(0, ordering) · gate`.
""")

code(r"""
final_state, log = run_forge(buffer, EPOCHS, BLOCK_HASH, WEAK_STATE)
epochs = [s.epoch for s in log]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(epochs, [s.fitness for s in log], marker="o"); ax[0].set(title="Forge fitness (real feed)", xlabel="epoch", ylabel="fitness"); ax[0].grid(alpha=0.3)
ax[1].plot(epochs, [s.gate for s in log], marker="o", label="gate")
ax[1].plot(epochs, [s.ordering for s in log], marker="s", label="ordering")
ax[1].plot(epochs, [s.spread for s in log], marker="^", label="spread")
ax[1].set(title="gate / ordering / spread", xlabel="epoch"); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
print(f"fitness {log[0].fitness:.3f} -> {log[-1].fitness:.3f}  ({sum(1 for s in log if s.decision=='keep')} KEEPs)")
""")

md(r"""
## 4. Commit-reveal — identical challenges for every validator
""")

code(r"""
mhash = manifest_for(final_state)
a = build_challenges(final_state, buffer, rng_for(BLOCK_HASH, EPOCHS, mhash), 8)
b = build_challenges(final_state, buffer, rng_for(BLOCK_HASH, EPOCHS, mhash), 8)
identical = all(np.array_equal(x.context, y.context) and np.array_equal(x.truth, y.truth) for x, y in zip(a, b))
reveal = build_challenges(final_state, buffer, rng_for(BLOCK_HASH, EPOCHS, mhash), 128)
print(f"manifest={mhash[:16]}...  two replays byte-identical: {identical}  revealed {len(reveal)} challenges")
""")

md(r"""
## 5. Panel validity + leaderboard (MASE / WQL / CRPS) on real data

> **Read this if `valid=False`.** On real data the *default numpy anchor* does not
> always lead — a simple `ewma`/`drift` can edge it out on noisy equities or
> sunspots. That is the validity gate doing its job: it is telling you the anchor
> is too weak to certify real-world skill, exactly the README's "panel quality is
> load-bearing" caveat made concrete. The fix is **not** to ignore it but to swap
> in an independently-validated zero-shot TSFM via
> `default_panel(strong_model=...)`; on the synthetic zoo (`example.ipynb`) the
> numpy anchor *does* lead, which is why the demo uses it.
""")

code(r"""
panel = default_panel()
vp = validate_panel(reveal, panel)
print(f"anchor validation: strong leads '{vp['runner_up']}' by {vp['margin']:.3f} -> valid={vp['valid']}")

board = leaderboard(probabilistic_panel(), reveal)
print(f"\n{'rank':>4}  {'model':<16}{'MASE':>8}{'WQL':>8}{'CRPS':>8}")
for r in board:
    print(f"{r['rank']:>4}  {r['model']:<16}{r['mase']:>8.3f}{r['wql']:>8.3f}{r['crps']:>8.3f}")

names = [r["model"] for r in board]
plt.figure(figsize=(8,4)); plt.bar(names, [r["mase"] for r in board], color="#264653")
plt.title("Leaderboard on real data — MASE (lower=better)"); plt.ylabel("MASE"); plt.grid(alpha=0.3, axis="y"); plt.show()
""")

md(r"""
To put a real TSFM on this board (needs the `chronos` extra + staged weights):

```python
from tsfm_adapters import load_tsfm
board = leaderboard({"chronos": load_tsfm("chronos"), **probabilistic_panel()}, reveal)
```
""")

md(r"""
## 6. Headroom — can the benchmark certify a *better* model?
""")

code(r"""
probe_rng = np.random.default_rng(7)
truth_by_id = {id(ch.context): np.asarray(ch.truth, dtype=float) for ch in reveal}
def superior_probe(context, meta=None):
    tgt = truth_by_id[id(context)]
    noisy = tgt + probe_rng.normal(0.0, 0.05*(np.std(tgt)+1e-8), size=tgt.shape)
    d = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
    return ProbForecast(mean=noisy, quantiles={q: noisy for q in d})
hr = headroom(superior_probe, reveal)
print(f"superior probe beats anchor by {hr['mase_margin']:+.3f} MASE / {hr['wql_margin']:+.3f} WQL")
print(f"benchmark_has_headroom = {benchmark_has_headroom(superior_probe, reveal)}")
""")

md(r"""
## 7. Foundational breadth across **real** domains

Coverage is measured (effective number of DGPs), and fitness is read per domain so
narrowness can't hide in an average. On real data this surfaces a genuine signal:
the synthetic share is pre-fittable (low `gate`) while the real domains are not,
and some real series (e.g. equities) are intrinsically hard to order — the
validity-gate ↔ hard-DGP tension, made visible on real data.
""")

code(r"""
cov = domain_coverage(reveal)
print(f"spans {cov['n_domains']} domains; effective (exp-entropy) = {cov['effective_domains']:.2f}\n")
print(f"{'domain':<18}{'n':>4}{'spread':>8}{'order':>8}{'gate':>7}")
for dom, m in sorted(stratified_fitness(reveal, panel).items(), key=lambda kv: -kv[1]['n']):
    print(f"{dom:<18}{m['n']:>4}{m['spread']:>8.2f}{m['ordering']:>+8.2f}{m['gate']:>7.2f}")
""")

md(r"""
## 8. As-of (vintage) gating on real timestamps

A real deployment must only serve data timestamped *after* the commit beacon.
`DatedCsvFeed` reads the date column and stamps each window with its end time;
`feeds.AsOfLiveSource` admits only post-commit windows.
""")

code(r"""
from datetime import datetime, UTC
from feeds import AsOfLiveSource
dated = make_feed("climate_temp", fetch=fetch, dated=True)
commit = datetime(1988, 1, 1, tzinfo=UTC).timestamp()
gated = AsOfLiveSource(inner=dated, commit_time=commit)
windows = gated.pull_dated(5, 304, np.random.default_rng(0))
for _, dom, ts in windows:
    print(f"  {dom}: window ends {datetime.fromtimestamp(ts, UTC).date()}  (>= 1988-01-01)")
print("all post-commit:", all(ts > commit for _, _, ts in windows))
""")

md(r"""
## Recap

- Real, multi-domain feeds flow through the **same** forge → reveal → panel →
  leaderboard → headroom pipeline as the synthetic demo.
- The forge hardens against the pre-fittable synthetic share (gate ↑); real
  motifs carry structure the generator-fitter cannot reconstruct.
- `DatedCsvFeed` + `AsOfLiveSource` give vintage discipline on real timestamps.

For production: point `DatedCsvFeed` at an as-of vendor endpoint, wrap it in
`feeds.AsOfLiveSource` + `feeds.DedupFreshBuffer`, swap a real zero-shot TSFM into
`default_panel(strong_model=...)`, and put `tsfm_adapters.load_tsfm(...)` on the
leaderboard.
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

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_feeds.ipynb")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)
print(f"wrote {out} with {len(cells)} cells")
