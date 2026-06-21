"""Challenge generation: augmentation, synthetic composition, and the splice.

Anti-gaming role
----------------
Three independent layers live here, each closing a different vector:

* **Augmentation** (`jitter`, `magnitude_warp`, `history_cutout`, `time_warp`)
  perturbs real motifs so exact-match memorisation fails even when the
  underlying motif repeats.
* **Synthetic composition** (`_piecewise_trend`, `_seasonality`,
  `_regime_ar_noise`, `_spikes`) manufactures structure that live data happens
  not to cover this epoch, closing coverage gaps.
* **Real-motif splice** blends a synthetic scaffold with a live motif, so a
  model tuned to pure-synthetic fingerprints still meets genuine real-world
  texture it cannot have pre-fit.

The **blend controller** (`assemble_challenge`) picks the mode per challenge
from the (seeded) `GeneratorState` weights, so a miner cannot know in advance
whether a given challenge is `synth`, `spliced`, or `aug_live`.

Every random draw flows through the passed-in ``rng`` (beacon-derived), so the
entire challenge set is a deterministic function of the revealed seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import CONTEXT_LEN, HORIZON, SEASONAL_PERIODS, GeneratorState
from ingest import FreshBuffer

SERIES_LEN = CONTEXT_LEN + HORIZON
_MAX_CHANGEPOINTS = 4
_EPS = 1e-8

# Amplitude of the synthetic "fingerprint" -- a deterministic, noise-looking
# pattern (see ``_fingerprint``) baked into every synthetic scaffold. It is the
# concrete thing a generator-fitting miner exploits: reconstructable if you know
# the generator, but invisible to the legitimate panel. The ``overfit`` detector
# is handed it (via the oracle) and nothing else is, which is what lets the
# detector flag a benchmark that leans too hard on pre-fittable synthetic data.
_FINGERPRINT_AMP = 1.2


# --------------------------------------------------------------------------- #
# Challenge container
# --------------------------------------------------------------------------- #


@dataclass
class Challenge:
    """One forecasting task handed to the panel.

    ``meta['oracle']`` carries the synthetic process's noise-free continuation
    (the synthetic *contribution* to the truth). It is consumed only by the
    ``overfit`` detector to simulate a worst-case generator-fitting miner; real
    forecasters ignore ``meta`` entirely. It is ``None`` for ``aug_live``, where
    there is no synthetic structure to exploit.

    ``meta['domain']`` names the data-generating process behind the challenge:
    ``"synth"`` for pure-synthetic scaffolds, or the live motif's source domain
    (e.g. ``"lorenz"``, ``"random_walk"``) for ``spliced`` / ``aug_live``. The
    scorer aggregates by it to measure coverage and report skill per domain.
    """

    context: np.ndarray
    truth: np.ndarray
    mode: str
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Synthetic primitives
# --------------------------------------------------------------------------- #


def _piecewise_trend(
    length: int, base: float, slope0: float, n_changepoints: int, rng: np.random.Generator
) -> np.ndarray:
    """Piecewise-linear trend with ``n_changepoints`` slope breaks *inside the context*.

    The breaks are placed within the observed window (not the horizon), so the
    final segment establishes a *trackable* current regime: a locally-adaptive
    forecaster (``strong``) reads the latest slope and continues it, while a
    global-slope model (``drift``) is contaminated by the stale earlier regimes
    and flat models (``ewma``/``naive``) miss the trend entirely. Raising
    ``changepoint_prob`` therefore adds discriminating, in-order structure -- it
    widens ``strong``'s lead -- rather than just adding unpredictable noise. Late
    breaks (little context left to estimate the new slope) supply the
    "too hard" regime that eventually defeats even ``strong``.
    """
    slope_change = 0.25
    cps: set[int] = set()
    if n_changepoints > 0:
        # Keep changepoints within the context window so the resulting regime is
        # observable; allow them close to its end to create the hard cases.
        hi = max(2, CONTEXT_LEN - 4)
        n_eff = min(n_changepoints, hi - 1)
        cps = {int(c) for c in rng.choice(np.arange(1, hi), size=n_eff, replace=False)}
    out = np.empty(length)
    level, slope = base, slope0
    for t in range(length):
        if t in cps:
            slope += rng.normal(0.0, slope_change)
        level += slope
        out[t] = level
    return out


def _seasonality(
    length: int, periods: tuple[int, ...], amps: tuple[float, ...], phases: tuple[float, ...]
) -> np.ndarray:
    """Sum of sinusoids at the recipe's periods (multi-period seasonality)."""
    t = np.arange(length)
    s = np.zeros(length)
    for p, a, ph in zip(periods, amps, phases, strict=True):
        s += a * np.sin(2 * np.pi * t / p + ph)
    return s


def _regime_ar_noise(
    length: int, phi: float, regime_switch_prob: float, rng: np.random.Generator
) -> np.ndarray:
    """Regime-switched AR(1) noise.

    The variance flips between a calm and a volatile regime; ``phi`` sets
    persistence. The innovation scale is normalised by ``sqrt(1 - phi**2)`` so
    the *stationary variance is constant in* ``phi``: raising ``phi`` makes the
    noise more **predictable** (higher lag-1 autocorrelation) without simply
    making it louder. That is what turns ``noise_ar_phi`` into a clean
    discrimination knob -- AR-aware forecasters (``strong``, ``ar1``) capture the
    extra predictable swing while flat/trend models cannot. It is also the
    structure the ``overfit`` detector exploits: a miner who knows the exact
    ``phi`` and regime path predicts the noise's conditional mean better than
    ``strong`` can estimate it.
    """
    sigmas = (0.5, 1.6)
    innov_scale = float(np.sqrt(max(1.0 - phi * phi, 1e-6)))
    noise = np.empty(length)
    prev = 0.0
    regime = 0
    for t in range(length):
        if rng.random() < regime_switch_prob:
            regime = 1 - regime
        prev = phi * prev + rng.normal(0.0, sigmas[regime] * innov_scale)
        noise[t] = prev
    return noise


def _spikes(length: int, prob: float, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Sparse, unpredictable additive spikes (zero conditional mean)."""
    mask = rng.random(length) < prob
    return mask * rng.normal(0.0, scale, size=length)


def _fingerprint(length: int, rng: np.random.Generator, n_comp: int = 3) -> np.ndarray:
    """Deterministic, noise-looking pattern that only a generator-fitter can predict.

    A sum of short, *non-seasonal*, incommensurate sinusoids. To the legitimate
    panel (which only searches the declared ``SEASONAL_PERIODS``) it is
    indistinguishable from irreducible noise, so it inflates every honest model's
    error equally. But it is a fixed function of the recipe's RNG draws, so the
    ``overfit`` detector -- standing in for a miner who has reverse-engineered the
    generator -- reconstructs and cancels it. The gap this opens on synthetic
    challenges is exactly the gaming signal the detector exists to expose.
    """
    t = np.arange(length)
    out = np.zeros(length)
    for _ in range(n_comp):
        period = rng.uniform(4.0, 11.0)
        phase = rng.uniform(0.0, 2 * np.pi)
        out += np.sin(2 * np.pi * t / period + phase)
    return (_FINGERPRINT_AMP / np.sqrt(n_comp)) * out


# --------------------------------------------------------------------------- #
# Augmentations (applied to live motifs)
# --------------------------------------------------------------------------- #


def jitter(x: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise scaled to the series' own variability."""
    sd = severity * (float(np.std(x)) + _EPS)
    return x + rng.normal(0.0, sd, size=x.shape)


def magnitude_warp(
    x: np.ndarray, severity: float, rng: np.random.Generator, knots: int = 4
) -> np.ndarray:
    """Multiply by a smooth random curve (slowly-varying gain)."""
    n = len(x)
    kx = np.linspace(0, n - 1, knots)
    ky = 1.0 + rng.normal(0.0, severity, size=knots)
    curve = np.interp(np.arange(n), kx, ky)
    return x * curve


def time_warp(
    x: np.ndarray, severity: float, rng: np.random.Generator, knots: int = 4
) -> np.ndarray:
    """Smoothly speed up / slow down local time (phase distortion)."""
    n = len(x)
    kx = np.linspace(0, n - 1, knots)
    speed = np.clip(1.0 + rng.normal(0.0, severity, size=knots), 0.2, None)
    fine = np.interp(np.arange(n), kx, speed)
    cum = np.cumsum(fine)
    cum = cum / cum[-1] * (n - 1)
    return np.interp(np.arange(n), cum, x)


def history_cutout(x: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Blank a contiguous chunk of *history* (context only), holding last value.

    Forces robustness to missing context; never touches the horizon, so the
    truth stays a faithful continuation.
    """
    span = int(severity * CONTEXT_LEN * 0.5)
    if span < 1:
        return x
    start = int(rng.integers(0, max(1, CONTEXT_LEN - span)))
    out = x.copy()
    out[start : start + span] = out[start]
    return out


_AUGMENTATIONS = {
    "jitter": jitter,
    "magnitude_warp": magnitude_warp,
    "time_warp": time_warp,
    "history_cutout": history_cutout,
}


# --------------------------------------------------------------------------- #
# Recipe grammar
# --------------------------------------------------------------------------- #


@dataclass
class Recipe:
    """A concrete, sampled specification for one challenge.

    Sampling the recipe separately from assembling it keeps the grammar
    inspectable and the generation deterministic: a given ``(state, rng)`` yields
    a fixed recipe, and a fixed recipe yields a fixed series.
    """

    mode: str
    periods: tuple[int, ...]
    amps: tuple[float, ...]
    phases: tuple[float, ...]
    base: float
    slope0: float
    n_changepoints: int
    spike_prob: float
    splice_weight: float
    aug_ops: tuple[str, ...]


def sample_recipe(state: GeneratorState, rng: np.random.Generator) -> Recipe:
    """Draw a challenge recipe from the generator state.

    The mode is chosen here by the **blend controller** using the (normalised)
    state weights, so the composition of the test set is set by the forge, not
    knowable to miners ahead of the reveal.
    """
    w_synth, w_spliced, w_aug_live = state.blend_weights()
    mode = str(
        rng.choice(["synth", "spliced", "aug_live"], p=[w_synth, w_spliced, w_aug_live])
    )

    n_periods = int(rng.integers(1, 3))
    periods = tuple(int(p) for p in rng.choice(SEASONAL_PERIODS, size=n_periods, replace=False))
    amps = tuple(float(rng.uniform(0.4, 1.0)) for _ in periods)
    phases = tuple(float(rng.uniform(0.0, 2 * np.pi)) for _ in periods)

    base = float(rng.normal(0.0, 5.0))
    slope0 = float(rng.normal(0.0, 0.05))
    n_changepoints = int(rng.binomial(_MAX_CHANGEPOINTS, state.changepoint_prob))
    spike_prob = 0.005 + 0.05 * state.aug_severity
    splice_weight = float(rng.uniform(0.35, 0.6))

    # The number of augmentation ops scales with severity; which ops is random.
    n_aug = int(np.clip(round(state.aug_severity * len(_AUGMENTATIONS)), 0, len(_AUGMENTATIONS)))
    aug_ops = tuple(
        str(o) for o in rng.choice(list(_AUGMENTATIONS), size=n_aug, replace=False)
    ) if n_aug > 0 else ()

    return Recipe(
        mode=mode,
        periods=periods,
        amps=amps,
        phases=phases,
        base=base,
        slope0=slope0,
        n_changepoints=n_changepoints,
        spike_prob=spike_prob,
        splice_weight=splice_weight,
        aug_ops=aug_ops,
    )


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def _build_synth(
    state: GeneratorState, recipe: Recipe, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(series, oracle)`` for a synthetic scaffold.

    ``oracle`` is the noise-free conditional-mean continuation over the horizon
    (trend + seasonality + AR conditional mean) -- what a perfect
    generator-fitter would predict. Spikes are excluded because they have zero
    conditional mean.
    """
    trend = _piecewise_trend(SERIES_LEN, recipe.base, recipe.slope0, recipe.n_changepoints, rng)
    seas = _seasonality(SERIES_LEN, recipe.periods, recipe.amps, recipe.phases)
    noise = _regime_ar_noise(SERIES_LEN, state.noise_ar_phi, state.regime_switch_prob, rng)
    spikes = _spikes(SERIES_LEN, recipe.spike_prob, 4.0, rng)
    fingerprint = _fingerprint(SERIES_LEN, rng)
    series = trend + seas + noise + spikes + fingerprint

    phi = state.noise_ar_phi
    h = np.arange(1, HORIZON + 1)
    ar_cond_mean = noise[CONTEXT_LEN - 1] * (phi**h)
    # The generator-fitter predicts everything deterministic -- trend,
    # seasonality, the AR conditional mean, and (crucially) the fingerprint --
    # leaving only the unpredictable innovations and spikes as its error.
    oracle = trend[CONTEXT_LEN:] + seas[CONTEXT_LEN:] + ar_cond_mean + fingerprint[CONTEXT_LEN:]
    return series, oracle


def _scale_motif(motif: np.ndarray, target_std: float) -> np.ndarray:
    """Center a motif and rescale it to a target standard deviation."""
    m = motif - motif.mean()
    s = float(np.std(m))
    if s < _EPS:
        return m
    return m * (target_std / s)


def assemble_challenge(
    state: GeneratorState,
    recipe: Recipe,
    buffer: FreshBuffer,
    rng: np.random.Generator,
) -> Challenge:
    """Assemble one ``Challenge`` according to ``recipe.mode``.

    * ``synth``    -- pure synthetic scaffold.
    * ``spliced``  -- synthetic scaffold blended with a scaled live motif; the
      oracle is the synthetic *contribution* only, so the detector is blind to
      the spliced-in real texture.
    * ``aug_live`` -- a live motif put through the recipe's augmentations; no
      synthetic oracle exists.
    """
    if recipe.mode == "synth":
        series, oracle = _build_synth(state, recipe, rng)
        context, truth = series[:CONTEXT_LEN], series[CONTEXT_LEN:]
        # Pure-synthetic challenges have no live texture -- their own domain.
        meta = {"mode": "synth", "oracle": oracle, "domain": "synth"}

    elif recipe.mode == "spliced":
        series, oracle = _build_synth(state, recipe, rng)
        scaffold_std = float(np.std(series)) + _EPS
        motif, domain = buffer.sample_labeled(1, SERIES_LEN, rng)[0]
        motif = _scale_motif(motif, scaffold_std)
        w = recipe.splice_weight
        blended = (1.0 - w) * series + w * motif
        context, truth = blended[:CONTEXT_LEN], blended[CONTEXT_LEN:]
        # The detector knows the synthetic part exactly but not the live motif;
        # the challenge inherits the spliced-in motif's data-generating domain.
        meta = {"mode": "spliced", "oracle": (1.0 - w) * oracle, "domain": domain}

    elif recipe.mode == "aug_live":
        motif, domain = buffer.sample_labeled(1, SERIES_LEN, rng)[0]
        motif = motif.copy()
        for op in recipe.aug_ops:
            motif = _AUGMENTATIONS[op](motif, state.aug_severity, rng)
        context, truth = motif[:CONTEXT_LEN], motif[CONTEXT_LEN:]
        meta = {"mode": "aug_live", "oracle": None, "domain": domain}

    else:  # pragma: no cover - guarded by sample_recipe
        raise ValueError(f"unknown mode {recipe.mode!r}")

    return Challenge(
        context=np.asarray(context, dtype=float),
        truth=np.asarray(truth, dtype=float),
        mode=recipe.mode,
        meta=meta,
    )


def build_challenges(
    state: GeneratorState,
    buffer: FreshBuffer,
    rng: np.random.Generator,
    n: int,
) -> list[Challenge]:
    """Deterministically assemble ``n`` challenges from one beacon RNG.

    Each challenge gets an independent child stream via ``rng.spawn`` so the set
    is order-stable and byte-reproducible: the cornerstone of cross-validator
    consensus and the determinism test. ``children[0]`` is reserved for a
    one-time pool refresh so that the per-challenge streams (``children[1:]``)
    are identical whether or not the buffer was already populated.
    """
    children = rng.spawn(n + 1)
    buffer.ensure(children[0])
    challenges: list[Challenge] = []
    for child in children[1:]:
        recipe = sample_recipe(state, child)
        challenges.append(assemble_challenge(state, recipe, buffer, child))
    return challenges
