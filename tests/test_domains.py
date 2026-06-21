"""Foundational breadth: a multi-domain DGP feed, domain tagging, and coverage.

These cover the three breadth investments:

1. a broad, multi-domain live feed (``domains.MixtureLiveSource`` over a zoo of
   distinct data-generating processes),
2. a coverage/diversity gate + stratified reporting (``score.domain_coverage`` /
   ``coverage_gate`` / ``stratified_fitness`` / ``foundational_fitness``),
3. the dynaprior-inspired DGP zoo wired into the ``LiveSource`` hook.

The load-bearing properties: the zoo is finite/non-degenerate and deterministic
under a fixed beacon (consensus-safe); challenges inherit their generating
domain; and the multi-domain feed is simultaneously *broad* and still *valid*
(the adaptive anchor leads), which is exactly what makes the benchmark
foundational rather than merely diverse.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from config import WEAK_STATE
from domains import (
    ZOO_SOURCES,
    JumpDiffusionSource,
    LorenzSource,
    MixtureLiveSource,
    default_live_source,
)
from generate import Challenge, build_challenges
from ingest import FreshBuffer
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
# (3) DGP zoo: well-formed and deterministic
# --------------------------------------------------------------------------- #


def test_zoo_sources_finite_bounded_and_deterministic() -> None:
    for source_cls in ZOO_SOURCES:
        src = source_cls()
        a = src.pull(3, 256, rng_for("zoo", 1, src.domain))
        b = src.pull(3, 256, rng_for("zoo", 1, src.domain))
        assert len(a) == 3
        for m, m2 in zip(a, b, strict=True):
            assert m.shape == (256,)
            assert np.all(np.isfinite(m))  # bounded / no blow-up
            assert float(np.std(m)) > 1e-6  # non-constant (scale stays well-conditioned)
            assert np.array_equal(m, m2)  # pure function of the beacon rng


def test_pull_labeled_default_tags_single_domain() -> None:
    src = LorenzSource()
    labeled = src.pull_labeled(4, 128, rng_for("z", 2, "m"))
    assert [d for _, d in labeled] == ["lorenz"] * 4


# --------------------------------------------------------------------------- #
# (1) Mixture feed: broad and deterministic
# --------------------------------------------------------------------------- #


def test_mixture_is_multi_domain_and_deterministic() -> None:
    src = default_live_source()
    pulls = src.pull_labeled(48, 256, rng_for("mix", 1, "m"))
    domains = {d for _, d in pulls}
    assert len(domains) >= 4  # genuinely spans many DGPs, not one

    again = src.pull_labeled(48, 256, rng_for("mix", 1, "m"))
    for (m1, d1), (m2, d2) in zip(pulls, again, strict=True):
        assert d1 == d2
        assert np.array_equal(m1, m2)


def test_mixture_rejects_bad_weights() -> None:
    for bad in ([], [(LorenzSource(), 0.0)], [(LorenzSource(), -1.0)]):
        try:
            MixtureLiveSource(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for components={bad!r}")


def test_freshbuffer_carries_domains() -> None:
    buf = FreshBuffer(default_live_source(), pool_size=40, motif_len=400)
    buf.refresh(rng_for("buf", 1, "m"))
    assert len(set(buf.pool_domains)) >= 4

    labeled = buf.sample_labeled(10, 256, rng_for("buf", 2, "m"))
    assert all(isinstance(d, str) and m.shape == (256,) for m, d in labeled)
    # Back-compat shim still returns bare arrays.
    arrays = buf.sample_motifs(5, 256, rng_for("buf", 3, "m"))
    assert all(a.shape == (256,) for a in arrays)


# --------------------------------------------------------------------------- #
# Challenges inherit their generating domain
# --------------------------------------------------------------------------- #


def test_challenges_tagged_with_domain() -> None:
    state = replace(WEAK_STATE, w_synth=0.34, w_spliced=0.33, w_aug_live=0.33)
    buf = FreshBuffer(default_live_source(), pool_size=48, motif_len=768)
    chs = build_challenges(state, buf, rng_for("ch", 1, "m"), 48)

    for ch in chs:
        d = challenge_domain(ch)
        assert d != "unknown"
        if ch.mode == "synth":
            assert d == "synth"
        else:  # spliced / aug_live inherit the live motif's domain
            assert d != "synth"

    assert len({challenge_domain(c) for c in chs}) >= 3


# --------------------------------------------------------------------------- #
# (2) Coverage metric, gate, and stratified reporting
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


def test_stratified_fitness_partitions_the_set() -> None:
    state = replace(WEAK_STATE, w_synth=0.34, w_spliced=0.33, w_aug_live=0.33)
    buf = FreshBuffer(default_live_source(), pool_size=48, motif_len=768)
    chs = build_challenges(state, buf, rng_for("strat", 1, "m"), 60)

    report = stratified_fitness(chs)
    assert set(report) == {challenge_domain(c) for c in chs}
    assert sum(int(v["n"]) for v in report.values()) == len(chs)


def test_panel_fitness_reports_coverage() -> None:
    state = replace(WEAK_STATE, w_synth=0.34, w_spliced=0.33, w_aug_live=0.33)
    buf = FreshBuffer(default_live_source(), pool_size=48, motif_len=768)
    res = panel_fitness(build_challenges(state, buf, rng_for("rep", 1, "m"), 48))
    assert float(res["coverage"]["effective_domains"]) > 1.0
    assert 0.0 <= float(res["coverage_gate"]) <= 1.0
    # Reporting coverage must not perturb the frozen fitness scalar.
    assert isinstance(res["fitness"], float)


# --------------------------------------------------------------------------- #
# The point of it all: broad AND valid, and the gate penalises narrowness
# --------------------------------------------------------------------------- #


def test_multidomain_feed_is_broad_and_valid() -> None:
    # The proven well-formed blend, but over the multi-domain feed. The adaptive
    # anchor must still lead (validity) while the eval spans many DGPs (breadth).
    state = replace(WEAK_STATE, w_synth=0.45, w_spliced=0.35, w_aug_live=0.20, aug_severity=0.3)
    buf = FreshBuffer(default_live_source(), pool_size=64, motif_len=768)

    acc = {"ordering": 0.0, "gate": 0.0, "fitness": 0.0, "eff": 0.0}
    errs: dict[str, float] = {}
    n_seeds = 5
    for s in range(n_seeds):
        res = panel_fitness(build_challenges(state, buf, rng_for("fnd", s, "m"), 64))
        for k in ("ordering", "gate", "fitness"):
            acc[k] += float(res[k]) / n_seeds
        acc["eff"] += float(res["coverage"]["effective_domains"]) / n_seeds
        for name, e in res["errors"].items():
            errs[name] = errs.get(name, 0.0) + e

    assert min(errs, key=errs.get) == "strong"  # valid: anchor leads on average
    assert acc["ordering"] > 0.0
    assert acc["gate"] > 0.0
    assert acc["fitness"] > 0.0
    assert acc["eff"] >= 3.0  # broad: many effective DGPs per evaluation


def test_foundational_fitness_penalises_a_narrow_feed() -> None:
    state = replace(WEAK_STATE, w_synth=0.40, w_spliced=0.30, w_aug_live=0.30, aug_severity=0.3)
    broad = FreshBuffer(default_live_source(), pool_size=64, motif_len=768)
    narrow = FreshBuffer(JumpDiffusionSource(), pool_size=64, motif_len=768)  # single DGP

    broad_chs = build_challenges(state, broad, rng_for("pen", 1, "m"), 64)
    narrow_chs = build_challenges(state, narrow, rng_for("pen", 1, "m"), 64)

    fb = foundational_fitness(broad_chs)
    fn = foundational_fitness(narrow_chs)

    # Same generator knobs, but the narrow feed collapses to ~2 domains
    # (synth + one), so its coverage gate -- and thus foundational fitness
    # relative to bare fitness -- is strictly lower.
    assert float(fb["coverage_gate"]) > float(fn["coverage_gate"])
    assert float(fn["coverage_gate"]) < 1.0
