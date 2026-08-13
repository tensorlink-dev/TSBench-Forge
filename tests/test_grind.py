"""grind: which vein runs today, and which of its finds are worth keeping.

The sweeps themselves are covered by test_bulk. What matters here is the
steering, because an unsteered grind drifts toward whatever is easiest to
harvest -- two unsteered waves took web_cloudops from 14% to 40% of the
catalog. So: thinnest domain first, veins matched to that domain, and finds
ranked by which (domain, cadence) cell they fill.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from source_discovery import grind


def _entry(sid: str, domain: str, freq: str, host: str) -> dict:
    return {
        "id": sid, "domain": domain, "dgp_class": "x", "frequency": freq,
        "endpoint": {"type": "rest_json", "url": f"https://{host}/api/{sid}"},
    }


def _catalog(tmp_path: Path, entries: list[dict]) -> Path:
    p = tmp_path / "sources.yaml"
    p.write_text(yaml.safe_dump(entries))
    return p


def _all_domains_catalog(tmp_path: Path, thinnest: str = "energy") -> Path:
    """Every domain populated, ``thinnest`` deliberately the weakest.

    Fixtures with one domain are a trap: the six absent ones score 0 effective
    and correctly rank ahead of it, so the vein under test never gets picked.
    """
    entries = []
    for d in grind.ALL_DOMAINS:
        n = 1 if d == thinnest else 6
        entries += [_entry(f"{d}_{i}", d, "PT1H", f"{d}{i}.example")
                    for i in range(n)]
    return _catalog(tmp_path, entries)


def _block(name: str, domain: str, freq: str, host: str) -> dict:
    entry = _entry(name, domain, freq, host)
    return {"candidate_name": name, "wireable": True, "cron_cadence": "P1D",
            "yaml_block": yaml.safe_dump([entry]), "reason": "test"}


def test_weakest_domain_uses_effective_not_raw_counts(tmp_path):
    """A domain whose sources all sit on one host is thinner than its headline.
    Ranking on raw counts would send the grind to the wrong place."""
    entries = []
    # 20 sources, all one host -> effective 5 (the cap).
    entries += [_entry(f"e{i}", "energy", "PT1H", "one.example")
                for i in range(20)]
    # 8 sources across 8 hosts -> effective 8, so healthier despite fewer.
    entries += [_entry(f"h{i}", "healthcare", "PT1H", f"h{i}.example")
                for i in range(8)]
    # Rank all of them: the five domains absent from this fixture are empty and
    # legitimately rank ahead of both (see the next test), so compare the two
    # this case is actually about.
    ranked = grind.weakest_domains(_catalog(tmp_path, entries), n=99)
    order = [r["domain"] for r in ranked]
    energy = next(r for r in ranked if r["domain"] == "energy")
    assert order.index("energy") < order.index("healthcare"), (
        "20 sources on one host is thinner than 8 across 8")
    assert energy["effective"] == 5 < energy["sources"] == 20


def test_domains_absent_from_the_catalog_rank_as_thinnest(tmp_path):
    """A domain with nothing at all is invisible to a ranking built only from
    what exists, and is exactly where the grind should go."""
    entries = [_entry(f"n{i}", "nature", "PT1H", f"n{i}.example")
               for i in range(5)]
    weak = grind.weakest_domains(_catalog(tmp_path, entries), n=3)
    assert all(w["effective"] == 0 for w in weak)
    assert "nature" not in {w["domain"] for w in weak}


def test_ranking_prefers_the_thinnest_cadence_cell(tmp_path):
    """Cadence cannot be requested from a sweep, so balance is steered here:
    among finds in a target domain, the thinnest band wins."""
    counts = Counter({("energy", "hourly"): 40, ("energy", "sub-min"): 1})
    blocks = [
        _block("fat", "energy", "PT1H", "a.example"),
        _block("thin", "energy", "PT1M", "b.example"),
    ]
    ranked = grind.rank_candidates(blocks, counts, ["energy"])
    assert [b["candidate_name"] for b in ranked] == ["thin", "fat"]


def test_ranking_puts_target_domains_first(tmp_path):
    counts = Counter({("energy", "hourly"): 40, ("web_cloudops", "hourly"): 0})
    blocks = [
        _block("offtarget", "web_cloudops", "PT1H", "a.example"),
        _block("ontarget", "energy", "PT1H", "b.example"),
    ]
    ranked = grind.rank_candidates(blocks, counts, ["energy"])
    assert ranked[0]["candidate_name"] == "ontarget", (
        "an empty cell in a healthy domain must not outrank the thin domain")


def test_one_host_cannot_take_the_whole_run(tmp_path):
    """Five sources from one API is the opposite of what this is for."""
    counts = Counter()
    blocks = [_block(f"s{i}", "energy", "PT1H", "same.example")
              for i in range(5)]
    blocks.append(_block("other", "energy", "PT1H", "other.example"))
    ranked = grind.rank_candidates(blocks, counts, ["energy"],
                                   per_host_cap=2)
    hosts = Counter(grind._block_host(b) for b in ranked)
    assert hosts["same.example"] == 2
    assert hosts["other.example"] == 1
    assert len(ranked) == 3


def test_malformed_block_does_not_sink_the_ranking(tmp_path):
    counts = Counter()
    blocks = [{"candidate_name": "bad", "yaml_block": "{{not yaml"},
              _block("good", "energy", "PT1H", "a.example")]
    ranked = grind.rank_candidates(blocks, counts, ["energy"])
    assert {b["candidate_name"] for b in ranked} == {"bad", "good"}
    assert ranked[0]["candidate_name"] == "good", "the parseable one ranks first"


def test_a_slow_vein_is_stopped_at_its_budget(tmp_path, monkeypatch):
    """Measured: ArcGIS ran 1060s against a 12-minute budget and no other vein
    got a turn. The sweeps take no deadline, so the budget is enforced from the
    log callback -- and the finds banked before the stop must survive."""
    import json as _json

    calls = {"slow": 0, "fast": 0}

    def slow_sweep(catalog, target=None, log=print, checkpoint_path=None,
                   **kw):
        calls["slow"] += 1
        Path(checkpoint_path).write_text(_json.dumps(
            [_block("banked", "energy", "PT1H", "slow.example")]))
        for _ in range(10_000):          # a sweep that would run forever
            log("still working")
        raise AssertionError("deadline never fired")   # pragma: no cover

    def fast_sweep(catalog, target=None, log=print, checkpoint_path=None,
                   **kw):
        calls["fast"] += 1
        return [_block("quick", "energy", "PT5M", "fast.example")], []

    monkeypatch.setattr(grind, "_veins", lambda: (
        grind.Vein("slow", slow_sweep, ("energy",)),
        grind.Vein("fast", fast_sweep, ("energy",)),
    ))
    # Zero budget rather than a small one: a timing race here would make the
    # test flaky in exactly the direction that hides the bug.
    monkeypatch.setattr(grind, "VEIN_MINUTES", 0)

    out = grind.run(_all_domains_catalog(tmp_path), target=2, minutes=5,
                    n_domains=1, checkpoint_dir=tmp_path / "ck",
                    log=lambda m: None)

    assert calls["slow"] == 1 and calls["fast"] == 1, (
        "the slow vein must not starve the ones behind it")
    slow_row = next(r for r in out["veins_run"] if r["vein"] == "slow")
    assert slow_row["stopped_at_budget"] is True
    assert slow_row["found"] == 1, "checkpointed finds survive the stop"
    assert {b["candidate_name"] for b in out["ranked"]} == {"banked", "quick"}


def test_a_failing_vein_does_not_end_the_run(tmp_path, monkeypatch):
    def broken(catalog, target=None, log=print, checkpoint_path=None, **kw):
        raise RuntimeError("upstream down")

    def works(catalog, target=None, log=print, checkpoint_path=None, **kw):
        return [_block("ok", "energy", "PT1H", "a.example")], []

    monkeypatch.setattr(grind, "_veins", lambda: (
        grind.Vein("broken", broken, ("energy",)),
        grind.Vein("works", works, ("energy",)),
    ))
    out = grind.run(_all_domains_catalog(tmp_path), target=2, minutes=5,
                    n_domains=1, checkpoint_dir=tmp_path / "ck",
                    log=lambda m: None)
    rows = {r["vein"]: r for r in out["veins_run"]}
    assert "upstream down" in rows["broken"]["error"]
    assert rows["works"]["found"] == 1
    assert [b["candidate_name"] for b in out["ranked"]] == ["ok"]


def test_every_domain_has_at_least_one_vein():
    """A domain no vein serves can never be filled, so the grind would report
    it as thinnest forever and find nothing."""
    veins = grind._veins()
    for domain in grind.ALL_DOMAINS:
        assert any(v.serves(domain) for v in veins), f"no vein serves {domain}"


def test_keyworded_veins_cover_every_domain():
    """Keyword filtering is how a general vein gets pointed at a thin domain;
    an empty filter would silently skip the vein."""
    for domain in grind.ALL_DOMAINS:
        from source_discovery import bulk
        pool = ([k for k in bulk.KEYWORD_CLASSES if k[1] == domain]
                + [k for k in bulk.ARCGIS_EXTRA_KEYWORDS if k[1] == domain])
        assert pool, f"no keyword classes for {domain}"


def test_exhausted_veins_are_not_scheduled():
    """ERDDAP, PeerTube and the publisher sweeps were each measured empty.
    Running them daily would burn the budget before a live vein got a turn."""
    names = {v.name for v in grind._veins()}
    assert not (names & {"erddap", "peertube", "ods_publishers",
                         "socrata_publishers", "socrata"})
