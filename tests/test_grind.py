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

import pytest
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


def _vein_yielding(name: str, blocks, calls=None):
    def sweep(catalog, target=None, log=print, checkpoint_path=None, **kw):
        if calls is not None:
            calls.append(name)
        return list(blocks), []
    return sweep


def test_keeps_sweeping_until_the_target_is_actually_WIRED(tmp_path, monkeypatch):
    """wire re-verifies against the live scraper, so submitting N never means N
    wired. A run that wires the ranked batch once and stops silently delivers
    fewer sources than asked."""
    calls = []
    monkeypatch.setattr(grind, "_veins", lambda: (
        grind.Vein("v1", _vein_yielding("v1", [
            _block("a", "energy", "PT1H", "a.example")], calls), ("energy",)),
        grind.Vein("v2", _vein_yielding("v2", [
            _block("b", "energy", "PT5M", "b.example")], calls), ("energy",)),
        grind.Vein("v3", _vein_yielding("v3", [
            _block("c", "energy", "PT1M", "c.example")], calls), ("energy",)),
    ))
    # Only every other submitted candidate survives verification.
    seen = {"n": 0}

    def wire_fn(batch):
        got = 0
        for _ in batch:
            seen["n"] += 1
            got += seen["n"] % 2
        return got

    out = grind.run(_all_domains_catalog(tmp_path), target=2, minutes=5,
                    n_domains=1, checkpoint_dir=tmp_path / "ck",
                    wire_fn=wire_fn, log=lambda m: None)
    assert out["wired"] >= 2, "must keep going until the target is met"
    assert out["met_target"] is True
    assert len(calls) >= 2, "one vein could not have supplied enough"


def test_stops_as_soon_as_the_target_is_met(tmp_path, monkeypatch):
    """The budget is not a quota to spend: hitting the number ends the run."""
    calls = []
    monkeypatch.setattr(grind, "_veins", lambda: tuple(
        grind.Vein(f"v{i}", _vein_yielding(f"v{i}", [
            _block(f"s{i}", "energy", "PT1H", f"h{i}.example")], calls),
            ("energy",)) for i in range(5)))
    out = grind.run(_all_domains_catalog(tmp_path), target=1, minutes=5,
                    n_domains=1, checkpoint_dir=tmp_path / "ck",
                    wire_fn=lambda batch: len(batch), log=lambda m: None)
    assert out["wired"] >= 1 and out["met_target"] is True
    assert len(calls) == 1, f"should have stopped after one vein, ran {calls}"


def test_falls_back_to_other_domains_rather_than_returning_short(tmp_path,
                                                                 monkeypatch):
    """Preferring the thin domains is right; returning zero because their veins
    are mined out is not. A source in a healthy domain beats nothing."""
    calls = []
    monkeypatch.setattr(grind, "_veins", lambda: (
        grind.Vein("thin_only", _vein_yielding("thin_only", [], calls),
                   ("energy",)),
        grind.Vein("elsewhere", _vein_yielding("elsewhere", [
            _block("w", "web_cloudops", "PT1H", "w.example")], calls),
            ("web_cloudops",)),
    ))
    out = grind.run(_all_domains_catalog(tmp_path, thinnest="energy"),
                    target=1, minutes=5, n_domains=1,
                    checkpoint_dir=tmp_path / "ck",
                    wire_fn=lambda batch: len(batch), log=lambda m: None)
    assert "elsewhere" in calls, "fallback pass must run when the thin vein is dry"
    assert out["wired"] == 1 and out["met_target"] is True


def test_shortfall_is_reported_honestly(tmp_path, monkeypatch):
    """A dry day must not look like a good one — cron reads met_target."""
    monkeypatch.setattr(grind, "_veins", lambda: (
        grind.Vein("dry", _vein_yielding("dry", []), ("energy",)),))
    out = grind.run(_all_domains_catalog(tmp_path), target=4, minutes=5,
                    n_domains=1, checkpoint_dir=tmp_path / "ck",
                    wire_fn=lambda batch: len(batch), log=lambda m: None)
    assert out["wired"] == 0 and out["met_target"] is False


def test_without_wire_fn_it_only_collects(tmp_path, monkeypatch):
    """Dry runs must not wire anything."""
    monkeypatch.setattr(grind, "_veins", lambda: (
        grind.Vein("v", _vein_yielding("v", [
            _block("a", "energy", "PT1H", "a.example")]), ("energy",)),))
    out = grind.run(_all_domains_catalog(tmp_path), target=2, minutes=5,
                    n_domains=1, checkpoint_dir=tmp_path / "ck",
                    log=lambda m: None)
    assert out["wired"] is None and out["met_target"] is None
    assert [b["candidate_name"] for b in out["ranked"]] == ["a"]


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


def test_untried_veins_are_explored_before_known_ones(tmp_path):
    """An unmeasured vein is the cheapest information available. Sorting it
    last would mean a vein that never got a turn can never earn one."""
    stats = {"ckan": {"seconds": 600, "found": 4, "wired": 4, "runs": 2}}
    order = [v.name for v, _ in grind._plan(list(grind.ALL_DOMAINS), stats)]
    assert order.index("arcgis") < order.index("ckan"), "untried explores first"


def test_ordering_follows_wired_not_found(tmp_path):
    """arcgis produced candidates that all failed verification while ckan's
    survived; ranking on `found` would still have put the slow one first."""
    stats = {
        "arcgis": {"seconds": 1801, "found": 2, "wired": 0, "runs": 2},
        "ckan":   {"seconds": 684,  "found": 4, "wired": 3, "runs": 2},
        "sdmx":   {"seconds": 683,  "found": 2, "wired": 1, "runs": 2},
    }
    order = [v.name for v, _ in grind._plan(list(grind.ALL_DOMAINS), stats)]
    measured = [n for n in order if n in stats]
    assert measured == ["ckan", "sdmx", "arcgis"]


def test_stats_accumulate_across_runs(tmp_path):
    p = tmp_path / "vein_stats.json"
    grind.record_stats(p, [{"vein": "ckan", "seconds": 300, "found": 2,
                            "wired": 1}])
    grind.record_stats(p, [{"vein": "ckan", "seconds": 200, "found": 1,
                            "wired": 2}])
    s = grind.load_stats(p)["ckan"]
    assert (s["seconds"], s["found"], s["wired"], s["runs"]) == (500, 3, 3, 2)
    assert grind._yield_rate(grind.load_stats(p), "ckan") == pytest.approx(
        60 * 3 / 500)
    assert grind._yield_rate(grind.load_stats(p), "never_run") is None


def test_corrupt_stats_file_does_not_stop_a_run(tmp_path):
    p = tmp_path / "vein_stats.json"
    p.write_text("{{ not json")
    assert grind.load_stats(p) == {}


def test_wins_are_credited_to_the_vein_that_supplied_them(tmp_path, monkeypatch):
    """Without attribution the ordering learns nothing: every vein would look
    equally good regardless of whose candidates survived the gate."""
    stats_path = tmp_path / "vein_stats.json"
    monkeypatch.setattr(grind, "_veins", lambda: (
        grind.Vein("dud", _vein_yielding("dud", [
            _block("bad", "energy", "PT1H", "bad.example")]), ("energy",)),
        grind.Vein("good", _vein_yielding("good", [
            _block("ok", "energy", "PT5M", "ok.example")]), ("energy",)),
    ))
    # Only the candidate named "ok" survives verification.
    def wire_fn(batch):
        return sum(1 for b in batch if b["candidate_name"] == "ok")

    grind.run(_all_domains_catalog(tmp_path), target=1, minutes=5, n_domains=1,
              checkpoint_dir=tmp_path / "ck", wire_fn=wire_fn,
              stats_path=stats_path, log=lambda m: None)
    s = grind.load_stats(stats_path)
    assert s["good"]["wired"] == 1
    assert s.get("dud", {}).get("wired", 0) == 0


def test_minimum_is_distinct_from_target(tmp_path, monkeypatch):
    """A 3-of-5 day is short; a 1-of-5 day means the veins stopped producing.
    Cron needs to tell those apart, so they are separate signals."""
    monkeypatch.setattr(grind, "_veins", lambda: (
        grind.Vein("v", _vein_yielding("v", [
            _block(f"s{i}", "energy", "PT1H", f"h{i}.example")
            for i in range(3)]), ("energy",)),))
    out = grind.run(_all_domains_catalog(tmp_path), target=5, minimum=2,
                    minutes=5, n_domains=1, checkpoint_dir=tmp_path / "ck",
                    wire_fn=lambda b: len(b), log=lambda m: None)
    assert out["wired"] == 3
    assert out["met_target"] is False and out["met_minimum"] is True


def test_below_the_floor_is_reported_separately(tmp_path, monkeypatch):
    monkeypatch.setattr(grind, "_veins", lambda: (
        grind.Vein("v", _vein_yielding("v", [
            _block("only", "energy", "PT1H", "h.example")]), ("energy",)),))
    out = grind.run(_all_domains_catalog(tmp_path), target=5, minimum=2,
                    minutes=5, n_domains=1, checkpoint_dir=tmp_path / "ck",
                    wire_fn=lambda b: len(b), log=lambda m: None)
    assert out["wired"] == 1
    assert out["met_target"] is False and out["met_minimum"] is False


def test_ods_enum_is_first_in_the_static_order():
    """It is the only vein measured to deliver: five wired in ~4 minutes where
    a keyword grind managed zero in 35."""
    assert grind._veins()[0].name == "ods_enum"


# --- benching: demotion is not enough, something has to leave ---------------
#
# _plan sorts a bad vein to the back, and the back is still a turn once
# everything ahead of it is mined out for the day. Measured 2026-08-17: ckan
# took 2047s of a 2400s run to wire nothing, three days running.

def test_a_vein_that_cannot_pay_for_its_budget_is_benched():
    """The real vein_stats.json numbers that motivated this."""
    benched = grind.benched_veins({
        "ods_enum": {"seconds": 1767, "found": 39, "wired": 24, "runs": 5},
        "arcgis":   {"seconds": 1259, "found": 1,  "wired": 1,  "runs": 2},
        "ckan":     {"seconds": 4518, "found": 1,  "wired": 1,  "runs": 3},
    })
    assert set(benched) == {"ckan"}, (
        "ckan cost 75 min per wired source; arcgis cost 21 and still pays")
    assert "min per wired source" in benched["ckan"]


def test_a_vein_still_under_trial_is_not_benched():
    """A slow start must not bench something that would have paid. pxweb had
    910s and nothing to show for it, which is not yet evidence."""
    assert grind.benched_veins(
        {"pxweb": {"seconds": 910, "found": 0, "wired": 0, "runs": 1}}) == {}


def test_a_vein_that_never_wires_is_benched_once_tried_enough():
    got = grind.benched_veins(
        {"pxweb": {"seconds": grind.VEIN_TRIAL_SECONDS + 1,
                   "found": 3, "wired": 0, "runs": 4}})
    assert "pxweb" in got and "0 wired" in got["pxweb"], (
        "found-but-never-wired is the worse case: it spends the budget twice")


def test_an_untried_vein_is_never_benched():
    """_plan explores the untried first; benching them would close that door
    before it opened."""
    assert grind.benched_veins({}) == {}
    assert grind.benched_veins({"gbfs": {"seconds": 0, "wired": 0}}) == {}


def test_a_benched_vein_does_not_get_a_turn(tmp_path, monkeypatch):
    calls = {"dead": 0, "live": 0}

    def dead(catalog, target=None, log=print, checkpoint_path=None, **kw):
        calls["dead"] += 1
        return [], []

    def live(catalog, target=None, log=print, checkpoint_path=None, **kw):
        calls["live"] += 1
        return [_block("kept", "energy", "PT1H", "live.example")], []

    monkeypatch.setattr(grind, "_veins", lambda: (
        grind.Vein("dead", dead, ("energy",)),
        grind.Vein("live", live, ("energy",)),
    ))
    stats = tmp_path / "vein_stats.json"
    stats.write_text(__import__("json").dumps({
        "dead": {"seconds": 4518, "found": 1, "wired": 0, "runs": 3},
        "live": {"seconds": 100, "found": 5, "wired": 5, "runs": 1},
    }))

    out = grind.run(_all_domains_catalog(tmp_path), target=2, minutes=5,
                    n_domains=1, checkpoint_dir=tmp_path / "ck",
                    stats_path=stats, log=lambda m: None)

    assert calls == {"dead": 0, "live": 1}
    assert "dead" in out["benched"], "and it is reported, not silently dropped"


def test_benching_never_empties_the_board(tmp_path, monkeypatch):
    """A run with no veins is worse than a run with a bad one — and it would
    freeze the stats that are the only way back off the bench."""
    calls = {"n": 0}

    def only(catalog, target=None, log=print, checkpoint_path=None, **kw):
        calls["n"] += 1
        return [], []

    monkeypatch.setattr(grind, "_veins",
                        lambda: (grind.Vein("only", only, ("energy",)),))
    stats = tmp_path / "vein_stats.json"
    stats.write_text(__import__("json").dumps(
        {"only": {"seconds": 9999, "found": 0, "wired": 0, "runs": 9}}))

    out = grind.run(_all_domains_catalog(tmp_path), target=2, minutes=5,
                    n_domains=1, checkpoint_dir=tmp_path / "ck",
                    stats_path=stats, log=lambda m: None)

    assert calls["n"] == 1
    assert out["benched"] == {}, "the bench is dropped, and says so"


# --- the budget was advisory, and read as a guarantee -----------------------

def test_a_vein_that_never_logs_is_still_stopped_at_its_budget(
        tmp_path, monkeypatch):
    """The cooperative check can only fire when the sweep chooses to log.
    Measured 2026-08-17: ckan ran 2047s against a 720s budget and pushed a
    40-minute run to 54, straight through four sweep windows."""
    import json as _json
    import time as _time

    if not grind._can_arm_alarm():                    # pragma: no cover
        pytest.skip("SIGALRM unavailable here")

    def silent(catalog, target=None, log=print, checkpoint_path=None, **kw):
        Path(checkpoint_path).write_text(_json.dumps(
            [_block("banked", "energy", "PT1H", "silent.example")]))
        _time.sleep(30)                # never logs, so nothing checks the clock
        raise AssertionError("the alarm never fired")   # pragma: no cover

    def after(catalog, target=None, log=print, checkpoint_path=None, **kw):
        return [_block("quick", "energy", "PT5M", "after.example")], []

    monkeypatch.setattr(grind, "_veins", lambda: (
        grind.Vein("silent", silent, ("energy",)),
        grind.Vein("after", after, ("energy",)),
    ))
    monkeypatch.setattr(grind, "VEIN_MINUTES", 1.0 / 60.0)      # 1 second

    began = _time.monotonic()
    out = grind.run(_all_domains_catalog(tmp_path), target=2, minutes=5,
                    n_domains=1, checkpoint_dir=tmp_path / "ck",
                    log=lambda m: None)
    elapsed = _time.monotonic() - began

    assert elapsed < 10, f"ran {elapsed:.0f}s against a 1s budget"
    row = next(r for r in out["veins_run"] if r["vein"] == "silent")
    assert row["stopped_at_budget"] is True
    assert row["found"] == 1, "checkpointed finds survive the hard stop too"
    assert {b["candidate_name"] for b in out["ranked"]} == {"banked", "quick"}


def test_the_alarm_is_disarmed_after_a_vein_finishes(tmp_path, monkeypatch):
    """A timer left armed would fire into whatever ran next — including the
    wire step, which writes sources.yaml."""
    import signal as _signal
    import time as _time

    if not grind._can_arm_alarm():                    # pragma: no cover
        pytest.skip("SIGALRM unavailable here")

    def quick(catalog, target=None, log=print, checkpoint_path=None, **kw):
        return [_block("q", "energy", "PT1H", "q.example")], []

    monkeypatch.setattr(grind, "_veins",
                        lambda: (grind.Vein("quick", quick, ("energy",)),))
    monkeypatch.setattr(grind, "VEIN_MINUTES", 1.0 / 60.0)

    before = _signal.getsignal(_signal.SIGALRM)
    grind.run(_all_domains_catalog(tmp_path), target=1, minutes=5,
              n_domains=1, checkpoint_dir=tmp_path / "ck", log=lambda m: None)

    assert _signal.getsignal(_signal.SIGALRM) is before, "handler restored"
    assert _signal.getitimer(_signal.ITIMER_REAL) == (0.0, 0.0), "disarmed"
    _time.sleep(1.5)          # would raise here if the timer were still live


def test_the_deadline_survives_a_sweep_that_swallows_exceptions(
        tmp_path, monkeypatch):
    """bulk.py has 47 blanket `except Exception` handlers so one dead host
    cannot end a pass. A deadline derived from Exception would be eaten by
    every one of them."""
    import time as _time

    if not grind._can_arm_alarm():                    # pragma: no cover
        pytest.skip("SIGALRM unavailable here")

    def swallower(catalog, target=None, log=print, checkpoint_path=None, **kw):
        for _ in range(60):
            try:
                _time.sleep(0.5)
            except Exception:            # noqa: BLE001 — the pattern under test
                pass
        raise AssertionError("the deadline was swallowed")   # pragma: no cover

    monkeypatch.setattr(grind, "_veins",
                        lambda: (grind.Vein("swallower", swallower,
                                            ("energy",)),))
    monkeypatch.setattr(grind, "VEIN_MINUTES", 1.0 / 60.0)

    began = _time.monotonic()
    out = grind.run(_all_domains_catalog(tmp_path), target=1, minutes=5,
                    n_domains=1, checkpoint_dir=tmp_path / "ck",
                    log=lambda m: None)
    assert _time.monotonic() - began < 10
    assert out["veins_run"][0]["stopped_at_budget"] is True
