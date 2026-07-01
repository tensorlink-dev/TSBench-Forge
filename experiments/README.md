# experiments

Runnable experiment notebooks for tsbench-forge and the dependency-free scripts
that build them.

## live_feeds.ipynb

The real-data walkthrough: the full commit-reveal → panel → MASE/WQL/CRPS
leaderboard → headroom → breadth pipeline run on **real public series** (climate,
solar, atmospheric CO₂, equities, weather) via `live_feeds.py`. Challenges are
assembled with `build_live_challenges` over the live buffer — real data only, no
synthetic generation.

```bash
pip install -e ".[notebook]"                       # matplotlib + jupyter
python experiments/build_live_feeds_notebook.py    # (re)generate the .ipynb
jupyter notebook experiments/live_feeds.ipynb
```

Notes:

- **Network on first run.** The notebook pulls a handful of CSVs and caches them
  under `~/.cache/tsbench-forge/feeds` (override with `TSBENCH_FEED_CACHE`), so
  reruns are offline. If a feed host is blocked, point `TSBENCH_FEED_CACHE` at a
  pre-populated cache.
- **`valid=False` is expected on some seeds.** On real data the *default numpy
  anchor* does not always lead the simple baselines — the validity gate correctly
  flagging that the anchor is too weak to certify real-world skill. The fix is to
  swap in an independently-validated zero-shot TSFM via
  `default_panel(strong_model=...)`, not to ignore the gate. See the notebook's
  section 5.

The builder writes nbformat-v4 JSON directly, so **building** the notebook needs
no extra dependencies (only **running** it does).

## independent_validation.ipynb

Proves an **independently-validated anchor**: it establishes a model's quality on a
held-out, real, external benchmark (gold / NYC taxi / births — disjoint from the
live benchmark feed), then promotes the validated anchor to `strong` and re-checks
`validate_panel` on the live benchmark's own reveal.

```bash
pip install -e ".[notebook,strong]"                          # statsforecast fallback
python experiments/build_independent_validation_notebook.py  # (re)generate the .ipynb
jupyter notebook experiments/independent_validation.ipynb
```

- `resolve_anchor()` runs the strongest anchor available — a real **TSFM** with
  `pip install -e ".[chronos]"` + staged weights, else a literature-validated
  **statsforecast** model, else the numpy placeholder — and reports which. With
  `.[chronos]` this notebook is the independent validation of a *real neural TSFM*.
- Held-out feeds are pulled on first run and cached, same as `live_feeds.ipynb`.
