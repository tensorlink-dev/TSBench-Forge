"""The calibration replay scores policies honestly: no peeking, exact composition.

Rounds here are synthetic results.json payloads with planted structure: model
"a" wins the energy domain and "b" wins nature, so a per-domain policy should
find both leaders while any single-model policy has to give one domain away.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("replay_calibration", REPO / "scripts/replay_calibration.py")
rc = importlib.util.module_from_spec(_spec)
# Registered before executing: @dataclass looks its own module up in
# sys.modules to resolve the postponed annotations.
sys.modules["replay_calibration"] = rc
_spec.loader.exec_module(rc)

N_PER_DOMAIN = 30
DOMAIN_MAP = {f"e{i}": "energy" for i in range(3)} | {f"n{i}": "nature" for i in range(3)}


def payload(a_energy: float = 0.5, b_nature: float = 0.5, seed: int = 0) -> dict:
    """One round: 30 energy + 30 nature challenges over 6 sources."""
    rng = np.random.default_rng(seed)
    sources = [f"e{i % 3}" for i in range(N_PER_DOMAIN)] + [f"n{i % 3}" for i in range(N_PER_DOMAIN)]
    energy = np.arange(len(sources)) < N_PER_DOMAIN
    jitter = lambda: 1.0 + 0.01 * rng.standard_normal(len(sources))  # noqa: E731
    crps = {
        "seasonal_naive": np.ones(len(sources)),
        "a": np.where(energy, a_energy, 0.9) * jitter(),
        "b": np.where(energy, 0.9, b_nature) * jitter(),
        "paracast-ensemble": np.full(len(sources), 0.7) * jitter(),
    }
    models = {
        name: {"crps": v.tolist(), "mase": v.tolist(), "weight": [1.0] * len(sources)}
        for name, v in crps.items()
    }
    return {
        "config": {"roster": ["a", "b", "paracast-ensemble"]},
        "per_challenge": {"source_ids": sources, "models": models},
    }


def make_round(label: str, **kw):
    return rc.round_from_results(label, payload(**kw), DOMAIN_MAP)


def test_compose_takes_each_challenge_from_the_chosen_model():
    r = make_round("r0")
    choice = np.where(r.domains == "energy", "a", "b")
    s = rc.compose("mix", r.scores, choice)
    energy = r.domains == "energy"
    np.testing.assert_array_equal(s.crps[energy], r.scores["a"].crps[energy])
    np.testing.assert_array_equal(s.crps[~energy], r.scores["b"].crps[~energy])


def test_domain_leader_finds_each_domains_winner():
    history = [make_round(f"r{i}", seed=i) for i in range(3)]
    current = make_round("r3", seed=3)
    choice = rc.domain_leader(None)(history, current, ["a", "b"])
    assert set(choice[current.domains == "energy"]) == {"a"}
    assert set(choice[current.domains == "nature"]) == {"b"}


def test_policies_never_see_the_round_they_are_scored_on():
    # History says "a" wins nature too; the current round says "b" does. A
    # policy that peeked would pick "b" for nature.
    history = [make_round(f"r{i}", b_nature=0.95, seed=i) for i in range(3)]
    current = make_round("r3", b_nature=0.3, seed=3)
    choice = rc.domain_leader(None)(history, current, ["a", "b"])
    assert "b" not in set(choice[current.domains == "nature"])


def test_trailing_window_only_uses_recent_rounds():
    # Old rounds: "b" is dreadful in nature. Latest round: "b" wins nature.
    old = [make_round(f"r{i}", b_nature=2.0, seed=i) for i in range(4)]
    new = [make_round("r4", b_nature=0.2, seed=4)]
    current = make_round("r5", seed=5)
    assert set(rc.domain_leader(1)(old + new, current, ["a", "b"])[current.domains == "nature"]) == {"b"}
    assert set(rc.domain_leader(None)(old + new, current, ["a", "b"])[current.domains == "nature"]) == {"a"}


def test_small_domains_fall_back_to_the_global_leader():
    r = make_round("r0")
    assert r.domain_rel("energy") is not None
    # A domain with fewer challenges than the floor produces no evidence.
    idx = np.flatnonzero(r.domains == "energy")[: rc.MIN_DOMAIN_CHALLENGES - 1]
    small = rc.Round("s", rc.subset_scores(r.scores, idx), r.members, r.domains[idx])
    assert small.domain_rel("energy") is None


def test_replay_end_to_end_prefers_domain_selection_here(tmp_path):
    for i in range(5):
        d = tmp_path / f"2026-09-{10 + i:02d}"
        d.mkdir()
        (d / "results.json").write_text(json.dumps(payload(seed=i)))
    rounds, skipped = rc.load_rounds(tmp_path, DOMAIN_MAP)
    assert len(rounds) == 5 and not skipped
    result = rc.replay(rounds, [None])
    assert len(result["per_round"]) == 4
    summary = rc.summarize(result, rounds)
    by_row = {r["row"]: r for r in summary["table"]}
    domain = by_row["domain_leader[all]|members"]["mean_crps_rel"]
    single = by_row["trailing_leader[all]|members"]["mean_crps_rel"]
    assert domain < single
    assert by_row["domain_leader[all]|members"]["wins_vs_ensemble"] == 4
    assert summary["domain_leaders"] == {"energy": {"a": 5}, "nature": {"b": 5}}


def test_rounds_missing_the_ensemble_are_skipped(tmp_path):
    p = payload()
    del p["per_challenge"]["models"]["paracast-ensemble"]
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "results.json").write_text(json.dumps(p))
    rounds, skipped = rc.load_rounds(tmp_path, DOMAIN_MAP)
    assert rounds == [] and skipped == ["x"]


@pytest.mark.parametrize(
    "freq,bucket",
    [
        ("PT15M", "sub_daily"),
        ("PT1H", "sub_daily"),
        ("P1D", "daily"),
        ("P1W", "slower"),
        ("P1M", "slower"),  # a month, not a minute: "M" before the "T"
        ("P1Y", "slower"),
        ("weekly", "unknown"),
        (None, "unknown"),
    ],
)
def test_cadence_bucket_reads_iso_durations(freq, bucket):
    assert rc.cadence_bucket(freq) == bucket


def test_a_gap_withholds_the_most_recent_rounds():
    # Rounds 0-3: "b" is dreadful in nature. Round 4: "b" wins nature. With a
    # gap of 1, the policy scored on round 5 must not see round 4.
    rounds = [make_round(f"r{i}", b_nature=2.0, seed=i) for i in range(4)]
    rounds += [make_round("r4", b_nature=0.2, seed=4), make_round("r5", seed=5)]
    no_gap = rc.replay(rounds, [1], gap=0)["per_round"][-1]["choices"]
    gapped = rc.replay(rounds, [1], gap=1)["per_round"][-1]["choices"]
    key = "domain_leader[1]|members"
    assert no_gap[key].get("b") == N_PER_DOMAIN
    assert "b" not in gapped[key]


def test_cadence_breakdown_is_reported_per_bucket():
    cadence = {s: ("daily" if s.startswith("e") else "slower") for s in DOMAIN_MAP}
    rounds = [rc.round_from_results(f"r{i}", payload(seed=i), DOMAIN_MAP, cadence) for i in range(3)]
    summary = rc.summarize(rc.replay(rounds, [None]), rounds)
    assert set(summary["by_cadence"]) == {"daily", "slower"}


@pytest.mark.parametrize("text,expected", [("1,4,all", [1, 4, None]), (" 2 ", [2])])
def test_parse_windows(text, expected):
    assert rc.parse_windows(text) == expected
