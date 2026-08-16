"""Score the benchmark through the paracast inference endpoint.

paracast (github.com/TensorLink-AI/paracast) serves the open-weights TSFM
panel behind ONE HTTP endpoint (`POST /forecast`) with three modes: `explicit`
(name a model), `route` (its heuristic policy picks one) and `ensemble` (the
whole pool, combined). That turns a leaderboard round from "rent a GPU pod and
build ten venvs" into a CPU-only HTTP loop — which is what lets rounds run on a
schedule instead of by hand (see scripts/run_paracast_round.py).

Division of labour, deliberately:

* **paracast is inference only.** All scoring stays client-side in
  :mod:`evaluate` / :mod:`model_comparison`, on the same seeded challenges and
  the same seasonal-naive-relative aggregation as the GPU path — so
  paracast-scored rounds are directly comparable to pod-scored ones. paracast's
  own `/feedback` metrics are NOT used for the leaderboard: its MASE floor is
  naive-1 (ours is seasonal), and its numbers are per-model EWMA state, not a
  per-challenge paired sample.
* **`/feedback` is the return path, not the scoreboard.** After a round we post
  the realized truths back so paracast's inverse-nCRPS EWMA weights (its
  ensembling) and `state/requests.jsonl` (its router's training set) learn from
  every benchmark round. The leaderboard and the serving weights improve from
  the same evaluation without either trusting the other's arithmetic.

The client is stdlib-only (urllib) on top of the repo's numpy core: no httpx,
no pydantic — the wire contract is mirrored here as plain dicts and pinned by
tests/test_paracast_client.py against the shapes in paracast_common.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import numpy as np

from evaluate import _CRPS_QUANTILES, ProbForecast

# Leaderboard ids for the two pseudo-models the endpoint itself contributes.
# They compete as first-class rows: the ensemble tests whether combining the
# panel beats its best member, the router whether paracast's policy picks well.
ENSEMBLE_ID = "paracast-ensemble"
ROUTER_ID = "paracast-router"

# Snapshot of the enabled serving panel (paracast models.yaml, 2026-08). The
# live roster is ALWAYS discovered from GET /models at run time — this tuple
# exists so tests can pin docs/data/models.json display entries for every id
# a paracast round can publish without needing a live endpoint.
KNOWN_PANEL_IDS: tuple[str, ...] = (
    "chronos2",
    "timesfm25",
    "toto2-313m",
    "toto2-2.5b",
    "tirex2",
    "patchtst-fm-r1",
    "flowstate-r1",
    "yinglong-300m",
)

# The dense grid evaluate.py scores CRPS on. Requesting it directly means the
# evaluator interpolates the model's OWN quantile function at every level it
# needs (the WQL deciles are the 0.1/0.2/…/0.9 subset), instead of a coarse
# decile band being re-interpolated.
REQUEST_QUANTILES: tuple[float, ...] = _CRPS_QUANTILES

# ISO-8601 sampling interval (challenge meta["freq"], the FREQ_SEASONALITY
# keys) → the pandas-style alias paracast's feature extractor understands.
# Unknown intervals send freq=None: paracast treats the hint as optional, and a
# wrong alias is worse than none (FlowState's scale-factor table keys on it).
ISO_FREQ_TO_PANDAS: dict[str, str] = {
    "PT30S": "30s",
    "PT1M": "min",
    "PT2M30S": "150s",
    "PT5M": "5min",
    "PT6M": "6min",
    "PT10M": "10min",
    "PT15M": "15min",
    "PT30M": "30min",
    "PT1H": "h",
    "PT8H": "8h",
    "P1D": "D",
    "P1W": "W",
    "P1M": "MS",
    "P1Q": "QS",
    "P1Y": "YS",
}


def iso_freq_to_pandas(freq: str | None) -> str | None:
    """Map a challenge's ISO-8601 interval to a pandas alias (None if unknown)."""
    return ISO_FREQ_TO_PANDAS.get(freq or "")


class ParacastError(RuntimeError):
    """An HTTP or contract error from the paracast endpoint."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class ParacastClient:
    """Minimal JSON client for the paracast router API.

    ``transport`` is injectable for tests: a callable
    ``(method, url, body_bytes|None, headers) -> (status, body_bytes)``. The
    default uses urllib. Retries cover transient 5xx / connection failures —
    the endpoint cold-starts workers (weight downloads), so first contact after
    a deploy can 503 briefly.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 300.0,
        retries: int = 2,
        retry_wait: float = 2.0,
        transport=None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.retry_wait = retry_wait
        self._transport = transport or self._urllib_transport

    # -- transport ----------------------------------------------------------

    def _urllib_transport(self, method, url, body, headers):
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:  # non-2xx still carries a JSON body
            return e.code, e.read()

    def _call(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = self.base_url + path
        body = None if payload is None else json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                status, raw = self._transport(method, url, body, headers)
            except OSError as e:  # DNS/conn/timeout — retryable
                last = ParacastError(f"{method} {path}: {e}")
                status, raw = None, b""
            else:
                if status is not None and status < 400:
                    try:
                        return json.loads(raw.decode() or "{}")
                    except json.JSONDecodeError as e:
                        raise ParacastError(f"{method} {path}: bad JSON: {e}", status) from e
                detail = raw.decode(errors="replace")[:500]
                last = ParacastError(f"{method} {path}: HTTP {status}: {detail}", status)
                if status is not None and status < 500:
                    raise last  # 4xx is a contract bug, not a blip — never retry
            if attempt < self.retries:
                time.sleep(self.retry_wait * (2**attempt))
        raise last  # type: ignore[misc]

    # -- endpoints ----------------------------------------------------------

    def forecast(
        self,
        series: list[list[float]],
        *,
        horizon: int,
        mode: str = "explicit",
        model: str | None = None,
        freq: str | None = None,
        quantiles: tuple[float, ...] = REQUEST_QUANTILES,
        combine: str = "vincentize",
    ) -> dict:
        payload: dict = {
            "mode": mode,
            "series": series,
            "horizon": horizon,
            "quantiles": list(quantiles),
            "combine": combine,
        }
        if model is not None:
            payload["model"] = model
        if freq is not None:
            payload["freq"] = freq
        return self._call("POST", "/forecast", payload)

    def feedback(self, request_id: str, actuals: list[list[float]]) -> dict:
        return self._call("POST", "/feedback", {"request_id": request_id, "actuals": actuals})

    def models(self) -> dict:
        return self._call("GET", "/models")

    def health(self) -> dict:
        return self._call("GET", "/health")

    def panel(self) -> list[str]:
        """Names of the enabled+healthy serving panel, in manifest order."""
        doc = self.models()
        return [m["name"] for m in doc.get("models", []) if m.get("enabled") and m.get("healthy")]


# --------------------------------------------------------------------------- #
# Prefetch: batched inference over a challenge set
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSpec:
    """One leaderboard row to fetch from the endpoint.

    ``model_id`` is the published leaderboard id; ``mode``/``model`` are the
    request parameters (explicit rows set ``model``, the pseudo-models set only
    ``mode``).
    """

    model_id: str
    mode: str = "explicit"
    model: str | None = None
    combine: str = "vincentize"

    @staticmethod
    def explicit(name: str) -> ModelSpec:
        return ModelSpec(model_id=name, mode="explicit", model=name)

    @staticmethod
    def ensemble(combine: str = "vincentize") -> ModelSpec:
        return ModelSpec(model_id=ENSEMBLE_ID, mode="ensemble", combine=combine)

    @staticmethod
    def route() -> ModelSpec:
        return ModelSpec(model_id=ROUTER_ID, mode="route")


@dataclass
class RequestRecord:
    """One served /forecast call, kept so its truths can be fed back."""

    model_id: str
    mode: str
    request_id: str | None
    indices: list[int]  # challenge indices, in the order their series were sent


@dataclass
class PrefetchResult:
    """Per-model ProbForecasts for a challenge list, plus the feedback ledger."""

    forecasts: dict[str, dict[int, ProbForecast]] = field(default_factory=dict)
    requests: list[RequestRecord] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)  # model_id -> reason

    def forecaster(self, model_id: str):
        """A :data:`evaluate.Forecaster` replaying the prefetched forecasts.

        Keyed by context array identity (the convention evaluate.leaderboard's
        replay cache already uses), so the standard scoring paths — which only
        see ``(context, meta)`` — hit the prefetched result for their challenge.
        """
        table = self._context_table.get(model_id)
        if table is None:
            raise KeyError(f"no prefetched forecasts for {model_id!r}")

        def replay(context, meta=None, _table=table):
            try:
                return _table[id(context)]
            except KeyError:
                raise KeyError(
                    "context not in prefetched set — score the same challenge "
                    "list that was prefetched"
                ) from None

        return replay

    # id(context) -> ProbForecast, built by prefetch_forecasts (kept separate
    # from `forecasts` so the index-keyed dict stays JSON-friendly/debuggable).
    _context_table: dict[str, dict[int, ProbForecast]] = field(default_factory=dict)


def _clean_series(context: np.ndarray) -> list[float]:
    """A JSON-safe, finite copy of a context (paracast rejects non-finite points).

    Non-finite values are linearly interpolated over the finite neighbours
    (edge values held), matching how the scraper treats gaps. A context with
    fewer than two finite points cannot be repaired and raises.
    """
    x = np.asarray(context, dtype=float).copy()
    finite = np.isfinite(x)
    if finite.sum() < 2:
        raise ParacastError("context has fewer than 2 finite observations")
    if not finite.all():
        idx = np.arange(x.size)
        x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    return [float(v) for v in x]


def _parse_forecast(entry: dict, horizon: int) -> ProbForecast:
    """One response ``forecasts[i]`` dict → :class:`evaluate.ProbForecast`.

    Quantile keys arrive as strings (``"0.5"``); the evaluator wants float keys
    and horizon-length arrays. The median is the point path — the panel is
    quantile-native, and the median is what paracast's own skill tracker scores
    MASE on, so point and probabilistic metrics describe the same forecast.
    """
    qs: dict[float, np.ndarray] = {}
    for key, vals in entry["quantiles"].items():
        arr = np.asarray(vals, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != horizon:
            raise ParacastError(
                f"quantile {key} has shape {arr.shape}, expected ({horizon},)"
            )
        qs[float(key)] = arr
    if 0.5 not in qs:
        raise ParacastError("response quantiles lack the median (0.5)")
    return ProbForecast(mean=qs[0.5], quantiles=qs)


def prefetch_forecasts(
    client: ParacastClient,
    challenges: list,
    specs: list[ModelSpec],
    *,
    chunk: int = 64,
    verbose: bool = True,
) -> PrefetchResult:
    """Fetch every spec's forecast for every challenge, batched per request.

    Challenges are grouped by ``(horizon, freq)`` — the two per-request scalars
    in paracast's contract — and each group is sent in ``chunk``-sized batches,
    so a 256-challenge round is ~a dozen requests per model rather than 256.

    A spec that fails (down worker, 400 on some group) is recorded in
    ``errors`` and simply absent from ``forecasts`` — one dead model must not
    kill the round, mirroring ``tsfm_comparison.load_models`` semantics.
    """
    groups: dict[tuple[int, str | None], list[int]] = {}
    for i, ch in enumerate(challenges):
        meta = getattr(ch, "meta", None) or {}
        horizon = int(len(np.asarray(ch.truth)))
        freq = iso_freq_to_pandas(meta.get("freq"))
        groups.setdefault((horizon, freq), []).append(i)

    result = PrefetchResult()
    for spec in specs:
        got: dict[int, ProbForecast] = {}
        table: dict[int, ProbForecast] = {}
        try:
            for (horizon, freq), indices in groups.items():
                for start in range(0, len(indices), chunk):
                    batch = indices[start : start + chunk]
                    series = [_clean_series(challenges[i].context) for i in batch]
                    resp = client.forecast(
                        series,
                        horizon=horizon,
                        mode=spec.mode,
                        model=spec.model,
                        freq=freq,
                        combine=spec.combine,
                    )
                    forecasts = resp.get("forecasts", [])
                    if len(forecasts) != len(batch):
                        raise ParacastError(
                            f"{spec.model_id}: {len(forecasts)} forecasts "
                            f"for {len(batch)} series"
                        )
                    for i, entry in zip(batch, forecasts, strict=True):
                        fc = _parse_forecast(entry, horizon)
                        got[i] = fc
                        table[id(challenges[i].context)] = fc
                    result.requests.append(
                        RequestRecord(
                            model_id=spec.model_id,
                            mode=spec.mode,
                            request_id=(resp.get("meta") or {}).get("request_id"),
                            indices=list(batch),
                        )
                    )
        except ParacastError as e:
            result.errors[spec.model_id] = str(e)
            if verbose:
                print(f"  skip {spec.model_id}: {e}")
            continue
        result.forecasts[spec.model_id] = got
        result._context_table[spec.model_id] = table
        if verbose:
            print(f"  ok   {spec.model_id} ({len(got)} forecasts)")
    return result


def send_feedback(
    client: ParacastClient,
    challenges: list,
    records: list[RequestRecord],
    *,
    modes: tuple[str, ...] = ("explicit", "route", "ensemble"),
    verbose: bool = True,
) -> dict[str, int]:
    """POST realized truths back for every recorded request.

    This is the "results feed ensembles and routing" leg: paracast scores each
    pending request's per-model forecasts against these actuals, updates its
    inverse-nCRPS EWMA ensemble weights on disk, and logs the outcome beside
    the request features for a future learned router. Failures are counted, not
    raised — feedback is a donation, never a round blocker. Truths are known
    the moment challenges are built (context/truth split of already-scraped
    data), so this can run immediately after prefetch; paracast's pending table
    is an in-RAM LRU (512), so it should not wait longer than it has to.
    """
    sent = failed = skipped = 0
    for rec in records:
        if rec.request_id is None or rec.mode not in modes:
            skipped += 1
            continue
        actuals = [
            [float(v) for v in np.asarray(challenges[i].truth, dtype=float)]
            for i in rec.indices
        ]
        try:
            client.feedback(rec.request_id, actuals)
            sent += 1
        except ParacastError as e:
            failed += 1
            if verbose:
                print(f"  feedback failed ({rec.model_id}): {e}")
    return {"sent": sent, "failed": failed, "skipped": skipped}
