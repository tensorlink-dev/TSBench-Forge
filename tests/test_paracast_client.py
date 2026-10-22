"""Tests for src/paracast_client.py — the paracast-scored round path.

The wire shapes faked here mirror paracast_common.schema (ForecastRequest /
ForecastResponse / FeedbackRequest): quantile keys are ``f"{level:.6g}"``
strings, forecasts align with the submitted series, `meta.request_id` is the
feedback handle. If paracast's contract changes, update BOTH the fake below
and the client.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import paracast_client as pc
from challenges import Challenge
from evaluate import DEFAULT_QUANTILES, evaluate_forecaster


class FakeEndpoint:
    """Transport-level fake: answers like the paracast router, records calls."""

    def __init__(self, fail_models: set[str] | None = None):
        self.calls: list[dict] = []
        self.feedbacks: list[dict] = []
        self.fail_models = fail_models or set()
        self._req_no = 0

    def __call__(self, method, url, body, headers):
        path = url.split("://", 1)[-1].split("/", 1)[1]
        payload = json.loads(body) if body else None
        self.calls.append({"method": method, "path": "/" + path,
                           "payload": payload, "headers": headers})
        if path == "health":
            return 200, json.dumps({"status": "ok", "healthy": ["m1"]}).encode()
        if path == "models":
            return 200, json.dumps({
                "models": [
                    {"name": "chronos2", "enabled": True, "healthy": True},
                    {"name": "tirex2", "enabled": True, "healthy": True},
                    {"name": "moirai2", "enabled": False, "healthy": False},
                    {"name": "yinglong-300m", "enabled": True, "healthy": False},
                ],
                "router_policy": "heuristic", "gate_threshold": 0.5,
            }).encode()
        if path == "feedback":
            self.feedbacks.append(payload)
            return 200, json.dumps({"request_id": payload["request_id"],
                                    "metrics": {}, "weights": {}}).encode()
        assert path == "forecast"
        if payload.get("model") in self.fail_models:
            return 503, json.dumps({"detail": "worker unavailable"}).encode()
        self._req_no += 1
        horizon = payload["horizon"]
        levels = payload["quantiles"]  # server sorts; ours arrive sorted
        forecasts = []
        for series in payload["series"]:
            base = float(np.mean(series))
            forecasts.append({
                "quantiles": {
                    f"{q:.6g}": [base + (q - 0.5) * 2.0] * horizon for q in levels
                }
            })
        return 200, json.dumps({
            "forecasts": forecasts,
            "meta": {"mode": payload["mode"], "models_used": [payload.get("model") or "pool"],
                     "request_id": f"req-{self._req_no}"},
        }).encode()


def _challenge(value: float, n_ctx: int, horizon: int, freq: str, source: str) -> Challenge:
    return Challenge(
        context=np.full(n_ctx, value, dtype=float),
        truth=np.full(horizon, value, dtype=float),
        mode="live",
        meta={"freq": freq, "source_id": source, "unseen_frac": 1.0},
    )


@pytest.fixture
def challenges():
    # Two (horizon, freq) groups → prefetch must issue two requests per spec.
    return [
        _challenge(10.0, 64, 24, "PT1H", "src-a"),
        _challenge(20.0, 64, 24, "PT1H", "src-a"),
        _challenge(30.0, 32, 14, "P1D", "src-b"),
    ]


def _client(fake):
    return pc.ParacastClient("http://x/", transport=fake, retries=0)


def test_panel_filters_enabled_and_healthy():
    fake = FakeEndpoint()
    assert _client(fake).panel() == ["chronos2", "tirex2"]


def test_prefetch_groups_by_horizon_and_freq(challenges):
    fake = FakeEndpoint()
    result = pc.prefetch_forecasts(_client(fake), challenges,
                                   [pc.ModelSpec.explicit("chronos2")], verbose=False)
    reqs = [c["payload"] for c in fake.calls if c["path"] == "/forecast"]
    assert sorted((r["horizon"], r["freq"]) for r in reqs) == [(14, "D"), (24, "h")]
    assert all(r["mode"] == "explicit" and r["model"] == "chronos2" for r in reqs)
    # every request asks for the dense CRPS grid the evaluator scores on
    assert all(r["quantiles"] == list(pc.REQUEST_QUANTILES) for r in reqs)
    assert set(result.forecasts["chronos2"]) == {0, 1, 2}


def test_prefetch_parses_probforecast(challenges):
    fake = FakeEndpoint()
    result = pc.prefetch_forecasts(_client(fake), challenges,
                                   [pc.ModelSpec.explicit("chronos2")], verbose=False)
    fc = result.forecasts["chronos2"][0]
    assert fc.mean.shape == (24,)
    # float keys, median as the point path, WQL deciles all present
    assert np.allclose(fc.mean, fc.quantiles[0.5])
    for q in DEFAULT_QUANTILES:
        assert q in fc.quantiles and fc.quantiles[q].shape == (24,)


def test_forecaster_scores_through_evaluate(challenges):
    fake = FakeEndpoint()
    result = pc.prefetch_forecasts(_client(fake), challenges,
                                   [pc.ModelSpec.explicit("chronos2")], verbose=False)
    metrics = evaluate_forecaster(result.forecaster("chronos2"), challenges)
    assert metrics["n"] == 3.0
    # fake forecasts the context mean == truth, so the point error is ~0
    assert metrics["mase"] < 1e-6
    assert np.isfinite(metrics["crps"])


def test_failed_model_is_skipped_not_fatal(challenges):
    fake = FakeEndpoint(fail_models={"tirex2"})
    result = pc.prefetch_forecasts(
        _client(fake), challenges,
        [pc.ModelSpec.explicit("chronos2"), pc.ModelSpec.explicit("tirex2")],
        verbose=False)
    assert "chronos2" in result.forecasts
    assert "tirex2" not in result.forecasts
    assert "tirex2" in result.errors


def test_pseudo_model_specs():
    ens, route = pc.ModelSpec.ensemble(), pc.ModelSpec.route()
    assert (ens.model_id, ens.mode, ens.model) == (pc.ENSEMBLE_ID, "ensemble", None)
    assert (route.model_id, route.mode, route.model) == (pc.ROUTER_ID, "route", None)


def test_ensemble_row_requests_the_served_mixture():
    # DEC-TB-0005: the published row must measure what paracast serves.
    assert pc.ModelSpec.ensemble().combine == "mixture"
    assert pc.ModelSpec.ensemble("vincentize").combine == "vincentize"


def test_feedback_mirrors_truths(challenges):
    fake = FakeEndpoint()
    client = _client(fake)
    result = pc.prefetch_forecasts(client, challenges,
                                   [pc.ModelSpec.explicit("chronos2")], verbose=False)
    out = pc.send_feedback(client, challenges, result.requests, verbose=False)
    assert out == {"sent": 2, "failed": 0, "skipped": 0}
    by_id = {rec.request_id: rec for rec in result.requests}
    for fb in fake.feedbacks:
        rec = by_id[fb["request_id"]]
        expect = [np.asarray(challenges[i].truth).tolist() for i in rec.indices]
        assert fb["actuals"] == expect


def test_feedback_mode_filter(challenges):
    fake = FakeEndpoint()
    client = _client(fake)
    result = pc.prefetch_forecasts(client, challenges,
                                   [pc.ModelSpec.explicit("chronos2")], verbose=False)
    out = pc.send_feedback(client, challenges, result.requests,
                           modes=("ensemble",), verbose=False)
    assert out["sent"] == 0 and out["skipped"] == 2


def test_clean_series_interpolates_nonfinite():
    x = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
    assert pc._clean_series(x) == [1.0, 2.0, 3.0, 4.0, 5.0]
    with pytest.raises(pc.ParacastError):
        pc._clean_series(np.array([np.nan, 1.0]))


def test_iso_freq_mapping():
    assert pc.iso_freq_to_pandas("PT1H") == "h"
    assert pc.iso_freq_to_pandas("P1D") == "D"
    assert pc.iso_freq_to_pandas("PT9H") is None  # unknown → no hint, never a guess
    assert pc.iso_freq_to_pandas(None) is None


def test_4xx_raises_without_retry():
    calls = []

    def transport(method, url, body, headers):
        calls.append(url)
        return 400, b'{"detail": "bad quantiles"}'

    client = pc.ParacastClient("http://x", transport=transport, retries=3)
    with pytest.raises(pc.ParacastError) as e:
        client.forecast([[1.0, 2.0]], horizon=2)
    assert e.value.status == 400
    assert len(calls) == 1  # contract errors must not burn retries


def test_5xx_retries():
    calls = []

    def transport(method, url, body, headers):
        calls.append(url)
        if len(calls) < 2:
            return 503, b"cold start"
        return 200, json.dumps({"status": "ok"}).encode()

    client = pc.ParacastClient("http://x", transport=transport, retries=2, retry_wait=0.0)
    assert client.health() == {"status": "ok"}
    assert len(calls) == 2


def test_models_json_covers_every_publishable_id():
    """Every id a paracast round can publish has display metadata.

    Guards the same seam as test_publish_round's committed-round check: the
    tracker renders whatever ids appear in a round, and models.json is its
    display-name/class registry.
    """
    from pathlib import Path

    doc = json.loads(
        (Path(__file__).resolve().parent.parent / "docs/data/models.json").read_text()
    )
    registered = doc["models"]
    for mid in (*pc.KNOWN_PANEL_IDS, pc.ENSEMBLE_ID, pc.ROUTER_ID):
        assert mid in registered, f"models.json lacks display metadata for {mid!r}"
    assert registered[pc.ENSEMBLE_ID]["class"] == "ensemble"
    assert registered[pc.ROUTER_ID]["class"] == "ensemble"


# --------------------------------------------------------------------------- #
# Public Ephemeris API front door (base_url ending in /api/v1)
# --------------------------------------------------------------------------- #


class PublicFakeEndpoint(FakeEndpoint):
    """Answers like the Ephemeris gateway: /api/v1 routes, series objects,
    an unwrapped /models list, health at /api/status. Forecasts are delegated
    to the raw fake after translating the request back."""

    def __call__(self, method, url, body, headers):
        path = url.split("://", 1)[-1].split("/", 1)[1]
        if path == "api/status":
            self.calls.append({"method": method, "path": "/" + path,
                               "payload": None, "headers": headers})
            return 200, json.dumps({"state": "operational", "models": [
                {"name": "chronos2", "state": "ready", "enabled": True},
                {"name": "tirex2", "state": "warming", "enabled": True},
                {"name": "moirai2", "state": "unavailable", "enabled": False},
            ]}).encode()
        assert path.startswith("api/v1/"), path
        rest = path[len("api/v1/"):]
        raw_url = "http://x/" + rest
        if rest == "models":
            status, raw = super().__call__(method, raw_url, body, headers)
            return status, json.dumps(json.loads(raw)["models"]).encode()
        if rest == "forecast":
            payload = json.loads(body)
            assert all(isinstance(s, dict) for s in payload["series"])
            raw_payload = {**payload, "series": [s["values"] for s in payload["series"]]}
            status, raw = super().__call__(method, raw_url, json.dumps(raw_payload).encode(),
                                           headers)
            self.calls[-1]["public_payload"] = payload
            return status, raw
        return super().__call__(method, raw_url, body, headers)


def _public(fake, **kw):
    return pc.ParacastClient("https://eph.example/api/v1/", transport=fake, retries=0, **kw)


def test_public_flavour_is_detected_from_the_path():
    assert _public(PublicFakeEndpoint()).public
    assert not _client(FakeEndpoint()).public
    assert not pc.ParacastClient("https://host:20004").public


def test_public_forecast_sends_series_objects_full_context_and_a_key(challenges):
    fake = PublicFakeEndpoint()
    result = pc.prefetch_forecasts(_public(fake), challenges,
                                   [pc.ModelSpec.explicit("chronos2")], verbose=False)
    assert len(result.forecasts["chronos2"]) == 3
    sent = [c for c in fake.calls if "public_payload" in c]
    assert len(sent) == 2
    for call in sent:
        p = call["public_payload"]
        assert "freq" not in p  # per series, not per request
        assert {s["freq"] for s in p["series"]} <= {"h", "D"}
        assert p["context_len"] == max(len(s["values"]) for s in p["series"])
        assert call["headers"]["Idempotency-Key"].startswith("weir-")
    assert len({c["headers"]["Idempotency-Key"] for c in sent}) == 2


def test_public_retry_reuses_the_idempotency_key():
    keys = []

    def transport(method, url, body, headers):
        keys.append(headers.get("Idempotency-Key"))
        if len(keys) < 2:
            return 503, b"busy"
        return 200, json.dumps({"forecasts": [{"quantiles": {"0.5": [1.0]}}],
                                "meta": {"request_id": "r"}}).encode()

    client = pc.ParacastClient("https://eph.example/api/v1", transport=transport,
                               retries=2, retry_wait=0.0)
    client.forecast([[1.0, 2.0]], horizon=1)
    assert len(keys) == 2 and keys[0] == keys[1] is not None


def test_public_models_list_and_status_map_to_the_raw_shapes():
    fake = PublicFakeEndpoint()
    client = _public(fake)
    assert client.panel() == ["chronos2", "tirex2"]
    health = client.health()
    assert health == {"status": "ok", "healthy": ["chronos2"], "unhealthy": ["tirex2"]}
    assert fake.calls[-1]["path"] == "/api/status"


def test_public_batches_stay_inside_the_output_budget():
    horizon = 400  # 64 x 400 x 19 quantiles would be 486,400 values
    chs = [_challenge(float(i), 32, horizon, "P1D", "src") for i in range(40)]
    fake = PublicFakeEndpoint()
    pc.prefetch_forecasts(_public(fake), chs, [pc.ModelSpec.explicit("chronos2")],
                          verbose=False)
    sizes = [len(c["public_payload"]["series"]) for c in fake.calls if "public_payload" in c]
    cap = pc.PUBLIC_MAX_OUTPUT_VALUES // (horizon * len(pc.REQUEST_QUANTILES))
    assert sum(sizes) == 40 and max(sizes) == cap


def test_public_feedback_goes_to_the_gateway_route(challenges):
    fake = PublicFakeEndpoint()
    client = _public(fake)
    result = pc.prefetch_forecasts(client, challenges,
                                   [pc.ModelSpec.explicit("chronos2")], verbose=False)
    out = pc.send_feedback(client, challenges, result.requests, verbose=False)
    assert out == {"sent": 2, "failed": 0, "skipped": 0}
    assert len(fake.feedbacks) == 2
