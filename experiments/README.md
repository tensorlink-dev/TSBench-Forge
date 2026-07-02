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
- **Reading the scores.** The panel of baselines reports `spread` (discrimination)
  and the parrot gate; `strong` is a strong classical baseline / leaderboard rung,
  not a validity gate — on real data a naive model legitimately wins on some
  series, so there is no "strong must lead" requirement. To put a real TSFM on the
  leaderboard, pass `default_panel(strong_model=...)` and `tsfm_adapters.load_tsfm`.

The builder writes nbformat-v4 JSON directly, so **building** the notebook needs
no extra dependencies (only **running** it does).
