"""The forge: a keep/revert autoresearch loop over ``GeneratorState``.

Anti-gaming role
----------------
Every other layer closes a *static* gaming vector. This one closes the *dynamic*
vector -- **benchmark saturation**. Forecasters improve and memorise; a frozen
benchmark slowly leaks and stops discriminating. The forge keeps the benchmark a
moving target by treating its own design as an optimization problem: each epoch
it proposes a small change to the generator, keeps it only if the reference panel
becomes *more* discriminating-and-valid, and reverts otherwise.

Determinism / consensus
-----------------------
The proposer is the LLM boundary and is non-deterministic, so it must run *once*
per epoch (never per-validator). The forge commits a hashed manifest of the state
it settles on; the concrete challenges validators replay are then derived
deterministically from ``H(block_hash || epoch || manifest_hash)`` (see
``seed.py``). The keep/revert search below uses *common random numbers* (the same
beacon seed for the incumbent and the candidate) so the decision reflects the
state change, not sampling noise -- a standard variance-reduction trick, and the
reason the trajectory is legible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace

import numpy as np

from config import N_CHALLENGES, GeneratorState
from generate import build_challenges
from ingest import FreshBuffer
from score import PanelModel, default_panel, panel_fitness
from seed import beacon_seed, manifest_hash, rng_for

# Per-knob nudge sizes for the stub proposer. Blend weights take larger steps
# because they are renormalised (a small raw change barely moves the share).
_STEP = {
    "w_synth": 0.15,
    "w_spliced": 0.15,
    "w_aug_live": 0.15,
    "changepoint_prob": 0.05,
    "regime_switch_prob": 0.05,
    "aug_severity": 0.10,
    "noise_ar_phi": 0.10,
}
_KNOBS = tuple(_STEP)


@dataclass
class ForgeStep:
    """One row of the forge log: what was tried and whether it stuck."""

    epoch: int
    knob: str | None
    decision: str  # "init" | "keep" | "revert"
    fitness: float
    spread: float
    ordering: float
    difficulty: float
    gate: float
    state: GeneratorState


def manifest_for(state: GeneratorState) -> str:
    """Canonical commit hash of a state.

    Sorted-key JSON so the hash is stable across runs/machines -- the value the
    forge would publish on-chain before the seed (and thus the challenges) is
    revealed.
    """
    payload = json.dumps(asdict(state.normalized().clamped()), sort_keys=True)
    return manifest_hash(payload)


def propose_mutation(
    state: GeneratorState, log: list[ForgeStep], rng: np.random.Generator
) -> tuple[GeneratorState, str]:
    """Propose a one-knob mutation. **This is the LLM boundary (stubbed).**

    >>> THE LLM GOES HERE <<<  A real deployment replaces the body of this
    function with a single call to the forge LLM, handing it the log and
    ``program.md`` and asking for exactly one knob change. The contract the rest
    of the system relies on is intentionally tiny: *return a new state that
    differs from ``state`` in at most one knob.*

    The stub is a hand-coded approximation of the ``program.md`` policy so the
    demo tells the right story without an LLM in the loop:

    * **target the weakest validity signal first** -- if the generator-fitting
      ``gate`` is low (synthetic data is pre-fittable), shift mass from ``synth``
      to real/spliced modes; if the panel ``ordering`` is poor, ease the
      trend-break difficulty so the anchor leads again;
    * **then chase discrimination** (``spread``) by raising augmentation /
      changepoint richness;
    * **keep exploring** -- a fraction of proposals are random single-knob nudges,
      so the loop's KEEP/REVERT machinery (not the heuristic) is what actually
      decides, exactly as it would with a noisy LLM proposer.
    """
    last = log[-1]
    roll = rng.random()

    if roll < 0.30:  # exploration: one random knob, random direction
        knob = _KNOBS[int(rng.integers(len(_KNOBS)))]
        direction = 1.0 if rng.random() < 0.5 else -1.0
    elif last.gate < 0.65:  # generator-fitter winning -> de-emphasise synthetic
        knob = "w_aug_live" if rng.random() < 0.5 else "w_spliced"
        direction = 1.0
    elif last.ordering < 0.6:  # panel mis-ordered -> ease trend-break difficulty
        knob = "changepoint_prob"
        direction = -1.0
    else:  # validity holds -> chase spread
        knob = "aug_severity" if rng.random() < 0.5 else "changepoint_prob"
        direction = 1.0

    proposed = replace(state, **{knob: getattr(state, knob) + direction * _STEP[knob]})
    return proposed.normalized().clamped(), knob


def evaluate_state(
    state: GeneratorState,
    buffer: FreshBuffer,
    block_hash: str,
    n_challenges: int,
    panel: dict[str, PanelModel],
    n_seeds: int = 4,
) -> dict[str, float]:
    """Estimate a state's fitness as a mean over a fixed bank of challenge seeds.

    The forge is optimizing the *expected* discrimination of a state, so a single
    sampled challenge set (which carries real sampling noise) is too noisy to
    drive keep/revert. Averaging over a small, fixed bank of beacon seeds -- the
    same bank for every state, all run -- yields a stable objective: common random
    numbers turn the search into a clean hill-climb instead of a random walk.
    """
    keys = ("fitness", "spread", "ordering", "difficulty", "gate")
    acc = {k: 0.0 for k in keys}
    for s in range(n_seeds):
        rng = rng_for(block_hash, 10_000 + s, "forge-eval")
        res = panel_fitness(build_challenges(state, buffer, rng, n_challenges), panel)
        for k in keys:
            acc[k] += float(res[k])
    return {k: acc[k] / n_seeds for k in keys}


def run_forge(
    buffer: FreshBuffer,
    epochs: int,
    block_hash: str,
    init_state: GeneratorState,
    n_challenges: int = N_CHALLENGES,
    panel: dict[str, PanelModel] | None = None,
    n_seeds: int = 4,
) -> tuple[GeneratorState, list[ForgeStep]]:
    """Run the keep/revert forge loop and return ``(final_state, log)``.

    Each epoch: propose a one-knob mutation (the LLM boundary) -> assemble
    ``n_challenges`` and score with the reference panel over the fixed seed bank
    -> KEEP if the candidate's fitness beats the incumbent's, else REVERT. The
    panel is frozen for the whole run, so the logged fitness is a stable yardstick
    and the trajectory is monotone non-decreasing.
    """
    panel = panel or default_panel()
    state = init_state.normalized().clamped()

    def score(s: GeneratorState) -> dict[str, float]:
        return evaluate_state(s, buffer, block_hash, n_challenges, panel, n_seeds)

    incumbent = score(state)
    log: list[ForgeStep] = [_step(0, None, "init", incumbent, state)]

    for epoch in range(1, epochs + 1):
        candidate, knob = propose_mutation(state, log, rng_for(block_hash, epoch, "propose"))
        cand = score(candidate)

        if cand["fitness"] > incumbent["fitness"]:
            state, incumbent, decision = candidate, cand, "keep"
        else:
            decision = "revert"

        log.append(_step(epoch, knob, decision, incumbent, state))

    return state, log


def _step(
    epoch: int, knob: str | None, decision: str, m: dict[str, float], state: GeneratorState
) -> ForgeStep:
    return ForgeStep(
        epoch=epoch,
        knob=knob,
        decision=decision,
        fitness=m["fitness"],
        spread=m["spread"],
        ordering=m["ordering"],
        difficulty=m["difficulty"],
        gate=m.get("gate", float("nan")),
        state=state,
    )


def committed_seed(block_hash: str, epoch: int, state: GeneratorState) -> int:
    """The seed validators would replay for a committed state (commit-reveal).

    Provided for completeness/demos: ties the settled state to the public beacon
    so any validator can reconstruct the exact challenge set.
    """
    return beacon_seed(block_hash, epoch, manifest_for(state))
