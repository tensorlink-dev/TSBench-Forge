"""DEC-TB-0003: the per-round mix jitter that makes the benchmark composition
unpredictable.

The old sampler split every round identically — equal share per domain, then
per class — so a miner who inspected a few rounds (the eval pool is published
daily) knew the exact composition to tune against, including tiny cells whose
same handful of series were guaranteed slots every single day. These tests pin
the four replacements:

1. Dirichlet-jittered domain mix, uniform in expectation (``mix_jitter_alpha``);
2. without-replacement picks inside a cell (``_pick``);
3. per-round class rotation — tiny cells lose their guaranteed daily quota
   (``class_keep_frac``);
4. a per-round series bag — which series are eligible at all rotates
   (``series_bag_frac``);

plus the two invariants that make the jitter safe to ship: byte-determinism
per (catalog, seed) for validator consensus, and MIN_CLASS_SLOTS-block
allocation so the dgp-breadth hard-veto gate (share >= 2%) can never fire on a
class the round chose to include.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scraped_source import (MIN_CLASS_SLOTS, ScrapedLiveSource, _jittered_split,
                            _pick)

DOMAINS = ["econ_fin", "energy", "nature", "transport"]
CLASSES = ["alpha", "beta", "gamma"]
PER_CELL = 4  # sources per (domain, class) cell -> 48 series total


@pytest.fixture(scope="module")
def jitter_tree(tmp_path_factory) -> tuple[Path, Path]:
    """4 domains x 3 classes x 4 sources, one plain hourly series each."""
    tmp_path = tmp_path_factory.mktemp("jitter")
    catalog = []
    for dom in DOMAINS:
        for cls in CLASSES:
            for i in range(PER_CELL):
                catalog.append({
                    "id": f"{dom}_{cls}_{i}",
                    "domain": dom,
                    "dgp_class": f"{dom}_{cls}",
                    "frequency": "PT1H",
                })
    yaml_path = tmp_path / "sources.yaml"
    yaml_path.write_text(yaml.safe_dump(catalog))
    data_dir = tmp_path / "data"
    rng = np.random.default_rng(0)
    for entry in catalog:
        d = data_dir / entry["id"]
        d.mkdir(parents=True)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=300, freq="h"),
            "value": rng.normal(0, 1, size=300).cumsum(),
        })
        pq.write_table(pa.Table.from_pandas(df), d / "2026-07-01.parquet")
    return yaml_path, data_dir


def _source(tree, **kw) -> ScrapedLiveSource:
    return ScrapedLiveSource(tree[0], tree[1], min_series_length=128, **kw)


def _keys(picks) -> list[tuple]:
    return [ScrapedLiveSource._series_key(p) for p in picks]


# ------------------------------------------------------------- determinism


def test_same_seed_gives_byte_identical_picks(jitter_tree):
    """Consensus: every validator replaying the same beacon draws the same
    round, jitter and all."""
    a = _source(jitter_tree)._sample_indices_equal_weight(
        64, np.random.default_rng(7))
    b = _source(jitter_tree)._sample_indices_equal_weight(
        64, np.random.default_rng(7))
    assert _keys(a) == _keys(b)


def test_different_seeds_give_different_domain_mixes(jitter_tree):
    """The point of the change: the realised mix varies round to round."""
    src = _source(jitter_tree)
    mixes = set()
    for seed in range(10):
        picks = src._sample_indices_equal_weight(
            64, np.random.default_rng(seed))
        counts = Counter(p["domain"] for p in picks)
        mixes.add(tuple(counts[d] for d in DOMAINS))
    # The old sampler produced exactly one mix, every round, forever.
    assert len(mixes) >= 5


def test_expected_mix_is_still_uniform(jitter_tree):
    """The jitter must not change long-run breadth incentives: averaged over
    many rounds the mix converges back to equal-weight per domain."""
    src = _source(jitter_tree)
    totals: Counter[str] = Counter()
    rounds = 60
    for seed in range(rounds):
        picks = src._sample_indices_equal_weight(
            64, np.random.default_rng(seed))
        totals.update(p["domain"] for p in picks)
    expected = 64 * rounds / len(DOMAINS)
    for dom in DOMAINS:
        assert 0.8 * expected <= totals[dom] <= 1.2 * expected, (dom, totals)


# ------------------------------------------------------ gate-safety invariant


def test_every_class_present_clears_the_breadth_gate_floor(jitter_tree):
    """The dgp gate hard-vetoes any class below 2% share (3 of 128). A class
    is either absent from a round or present with >= MIN_CLASS_SLOTS picks —
    never token presence."""
    src = _source(jitter_tree)
    for seed in range(50):
        picks = src._sample_indices_equal_weight(
            128, np.random.default_rng(seed))
        counts = Counter(p["dgp_class"] for p in picks)
        assert min(counts.values()) >= MIN_CLASS_SLOTS, (seed, counts)


def test_every_domain_appears_in_every_round(jitter_tree):
    """The one-block domain floor: coverage never collapses to a subset."""
    src = _source(jitter_tree)
    for seed in range(30):
        picks = src._sample_indices_equal_weight(
            64, np.random.default_rng(seed))
        assert {p["domain"] for p in picks} == set(DOMAINS), seed


# --------------------------------------------------------------- rotation


def test_class_rotation_benches_classes_per_round_but_not_forever(jitter_tree):
    src = _source(jitter_tree, class_keep_frac=0.5, series_bag_frac=1.0)
    all_classes = {f"{d}_{c}" for d in DOMAINS for c in CLASSES}
    seen_overall: set[str] = set()
    some_round_missed_one = False
    for seed in range(20):
        picks = src._sample_indices_equal_weight(
            64, np.random.default_rng(seed))
        present = {p["dgp_class"] for p in picks}
        seen_overall |= present
        if present != all_classes:
            some_round_missed_one = True
    assert some_round_missed_one  # no guaranteed daily quota for any class
    assert seen_overall == all_classes  # but every class still gets rounds


def test_series_bag_rotates_which_series_can_appear(jitter_tree):
    """With a 0.5 bag and quotas large enough to exhaust every cell, a series
    missing from a round means the bag excluded it — and across rounds
    different series are excluded."""
    src = _source(jitter_tree, series_bag_frac=0.5, class_keep_frac=1.0,
                  mix_jitter_alpha=None)
    rounds = []
    for seed in range(10):
        picks = src._sample_indices_equal_weight(
            96, np.random.default_rng(seed))  # 2x the catalog's 48 series
        rounds.append(frozenset(_keys(picks)))
    assert len(set(rounds)) >= 5  # eligibility differs round to round
    union = set().union(*rounds)
    assert any(len(r) < len(union) for r in rounds)


# ------------------------------------------------------- without replacement


def test_picks_within_a_cell_are_distinct_until_the_cell_is_exhausted():
    pool = [{"source_id": f"s{i}", "panel_row": None} for i in range(8)]
    rng = np.random.default_rng(3)
    got = _pick(pool, 6, rng)
    keys = [p["source_id"] for p in got]
    assert len(set(keys)) == 6
    # Past exhaustion: every series once, then dupes.
    got = _pick(pool, 11, np.random.default_rng(3))
    keys = [p["source_id"] for p in got]
    assert set(keys) == {f"s{i}" for i in range(8)}
    assert len(keys) == 11


# ------------------------------------------------------------ _jittered_split


def test_jittered_split_conserves_total_and_floors_every_group():
    for seed in range(40):
        counts = _jittered_split(128, 7, 4.0, np.random.default_rng(seed))
        assert sum(counts) == 128
        assert len(counts) == 7
        assert min(counts) >= MIN_CLASS_SLOTS


def test_jittered_split_concentrates_tiny_topups():
    """Rejection top-ups (n < one block per group) go to few groups in usable
    chunks instead of spraying token 1-slot counts."""
    for seed in range(20):
        counts = _jittered_split(4, 7, 4.0, np.random.default_rng(seed))
        assert sum(counts) == 4
        assert sum(1 for c in counts if c > 0) == 1


def test_neutral_knobs_restore_the_exact_uniform_split(jitter_tree):
    src = _source(jitter_tree, mix_jitter_alpha=None, series_bag_frac=1.0,
                  class_keep_frac=1.0)
    for seed in (1, 2):
        picks = src._sample_indices_equal_weight(
            48, np.random.default_rng(seed))
        counts = Counter(p["domain"] for p in picks)
        assert all(counts[d] == 12 for d in DOMAINS)


# ---------------------------------------------------- K-draw pooled verdict


def test_round_draws_are_deterministic_and_differ_between_draws():
    """K jittered draws from one beacon seed: byte-identical across
    validators, genuinely different from each other."""
    from challenges import build_round_draws
    from conftest import live_buffer
    from seed import rng_for

    s1 = build_round_draws(live_buffer(pool_size=48), rng_for("kd", 1, "m"),
                           32, k_draws=3)
    s2 = build_round_draws(live_buffer(pool_size=48), rng_for("kd", 1, "m"),
                           32, k_draws=3)
    assert [len(s) for s in s1] == [32, 32, 32]
    for a, b in zip(s1, s2):
        for x, y in zip(a, b):
            assert np.array_equal(x.context, y.context)
            assert np.array_equal(x.truth, y.truth)
    draw_ids = [tuple(ch.meta["source_id"] for ch in s) for s in s1]
    assert len(set(draw_ids)) == 3  # each draw refreshed its own jittered pool


def test_pooled_verdict_bootstraps_once_over_all_draws():
    """evaluate_pooled: one forecast per challenge across all K draws, one
    pooled bootstrap CI around the point verdict."""
    from challenges import build_round_draws
    from conftest import live_buffer
    from evaluate import evaluate_pooled, probabilistic
    from score import seasonal_naive
    from seed import rng_for

    sets = build_round_draws(live_buffer(pool_size=48), rng_for("pv", 1, "m"),
                             16, k_draws=3)
    calls = 0
    base = probabilistic(seasonal_naive)

    def counting(ctx, meta=None):
        nonlocal calls
        calls += 1
        return base(ctx, meta)

    res = evaluate_pooled(counting, sets, n_boot=200, seed=11)
    assert calls == sum(len(s) for s in sets)  # K cheap evals, no re-forecasts
    assert res["n_draws"] == 3.0 and res["n"] == float(calls)
    for k in ("mase", "wql", "crps"):
        lo, hi = res[f"{k}_ci95"]
        assert lo <= res[k] <= hi
        assert res[f"{k}_boot_std"] > 0.0
    # Deterministic given the seed — the consensus requirement.
    again = evaluate_pooled(base, sets, n_boot=200, seed=11)
    assert again["mase_ci95"] == res["mase_ci95"]
