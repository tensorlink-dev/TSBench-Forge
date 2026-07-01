"""Foundational breadth: domain tagging, the coverage metric, gate, and strata.

The benchmark forecasts real data, so breadth comes from the live catalog itself.
These cover:

1. the coverage/diversity metric + gate + stratified reporting
   (``score.domain_coverage`` / ``coverage_gate`` / ``stratified_fitness`` /
   ``foundational_fitness``) — pure logic, independent of the data source;
2. that live challenges inherit and carry their source domain; and
3. that a broad live feed scores a full coverage gate while a single-domain feed
   is penalised.
"""

from __future__ import annotations

import numpy as np

from challenges import Challenge, build_live_challenges
from conftest import RandomWalkSource, live_buffer
from ingest import FreshBuffer, MixtureLiveSource
from score import (
    challenge_domain,
    coverage_gate,
    domain_coverage,
    foundational_fitness,
    panel_fitness,
    stratified_fitness,
)
from seed import rng_for


def _fake(domains: list[str]) -> list[Challenge]:
    """Challenges carrying only a domain tag (enough for coverage accounting)."""
    return [
        Challenge(context=np.zeros(4), truth=np.zeros(2), mode="x", meta={"domain": d})
        for d in domains
    ]


# --------------------------------------------------------------------------- #
# Coverage metric, gate, stratified reporting (pure logic)
# --------------------------------------------------------------------------- #


def test_effective_domains_matches_diversity() -> None:
    balanced = domain_coverage(_fake(["a", "b", "c", "d"] * 5))
    assert balanced["n_domains"] == 4
    assert abs(float(balanced["effective_domains"]) - 4.0) < 1e-6  # exp(entropy)=4

    single = domain_coverage(_fake(["a"] * 12))
    assert float(single["effective_domains"]) == 1.0  # exp(0)

    # Skew: one dominant domain + rare others => effective count well below raw.
    skewed = domain_coverage(_fake(["a"] * 17 + ["b", "c", "d"]))
    assert skewed["n_domains"] == 4
    assert float(skewed["effective_domains"]) < 2.0


def test_coverage_gate_is_monotone_and_targeted() -> None:
    narrow = _fake(["a"] * 20)
    broad = _fake(["a", "b", "c", "d", "e"] * 4)
    assert coverage_gate(broad) > coverage_gate(narrow)
    assert coverage_gate(broad) == 1.0  # 5 effective >= target 4
    assert coverage_gate(narrow, target_effective_domains=4.0) < 0.3  # 1/4


# --------------------------------------------------------------------------- #
# MixtureLiveSource construction guard (now lives in ingest)
# --------------------------------------------------------------------------- #


def test_mixture_rejects_bad_weights() -> None:
    for bad in ([], [(RandomWalkSource(), 0.0)], [(RandomWalkSource(), -1.0)]):
        try:
            MixtureLiveSource(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for components={bad!r}")


# --------------------------------------------------------------------------- #
# Live challenges carry their source domain
# --------------------------------------------------------------------------- #


def test_live_buffer_is_multi_domain() -> None:
    buf = live_buffer(pool_size=48, motif_len=384)
    assert len(set(buf.pool_domains)) >= 3  # the real catalog spans many domains
    # dgp_class / cadence labels are present for the breadth gates.
    assert all(c is not None for c in buf.pool_dgp_classes)
    assert all(c is not None for c in buf.pool_cadences)


def test_live_challenges_tagged_with_domain() -> None:
    chs = build_live_challenges(live_buffer(), rng_for("ch", 1, "m"), 48)
    for ch in chs:
        assert challenge_domain(ch) != "unknown"
    assert len({challenge_domain(c) for c in chs}) >= 3


def test_stratified_fitness_partitions_the_set() -> None:
    chs = build_live_challenges(live_buffer(), rng_for("strat", 1, "m"), 60)
    report = stratified_fitness(chs)
    assert set(report) == {challenge_domain(c) for c in chs}
    assert sum(int(v["n"]) for v in report.values()) == len(chs)


def test_panel_fitness_reports_coverage() -> None:
    res = panel_fitness(build_live_challenges(live_buffer(), rng_for("rep", 1, "m"), 48))
    assert float(res["coverage"]["effective_domains"]) > 1.0
    assert 0.0 <= float(res["coverage_gate"]) <= 1.0
    assert isinstance(res["fitness"], float)


# --------------------------------------------------------------------------- #
# Broad feed vs narrow feed: the coverage gate penalises narrowness
# --------------------------------------------------------------------------- #


def test_live_feed_is_broad() -> None:
    chs = build_live_challenges(live_buffer(pool_size=64), rng_for("brd", 1, "m"), 64)
    eff = float(domain_coverage(chs)["effective_domains"])
    assert eff >= 3.0  # many effective DGP domains per evaluation


def test_foundational_fitness_penalises_a_narrow_feed() -> None:
    broad_chs = build_live_challenges(live_buffer(pool_size=64), rng_for("pen", 1, "m"), 64)
    narrow_buf = FreshBuffer(RandomWalkSource(), pool_size=64, motif_len=384)  # single domain
    narrow_chs = build_live_challenges(narrow_buf, rng_for("pen", 1, "m"), 64)

    fb = foundational_fitness(broad_chs)
    fn = foundational_fitness(narrow_chs)
    assert float(fb["coverage_gate"]) > float(fn["coverage_gate"])
    assert float(fn["coverage_gate"]) < 1.0  # single domain -> gate below 1
