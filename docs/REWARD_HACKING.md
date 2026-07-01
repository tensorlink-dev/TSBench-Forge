# LLM-forge reward-hacking analysis

The forge optimizes

```
fitness = spread × max(0, ordering) × gate
```

over many epochs via an LLM proposer. This document catalogs the failure modes
where the LLM can raise `fitness` while making the benchmark *worse* as a
generalisation test — and the defenses shipped or queued.

## Failure modes

1. **Anchor exploitation.** `ordering` rewards states where the frozen `strong`
   anchor beats medium / weak baselines. If the LLM finds a blend that's
   *specifically easy for `strong`* (e.g. smooth-periodic-heavy where classical
   models shine), spread + ordering both climb — but a good TSFM is also fine
   on smooth-periodic. The eval silently measures "matches strong's training
   distribution" rather than TSFM skill.

2. **Domain collapse.** Global blend knobs (`w_synth`, `w_spliced`, `w_aug_live`)
   don't touch DGP-class balance directly, but their interaction with
   augmentation severities can differentially destroy some DGP classes: heavy
   time-warping ruins `count_discrete`, aggressive splicing creates pathological
   cross-DGP hybrids. The 7-domain claim silently narrows without any single
   epoch tripping the panel-validity gate.

3. **Cadence collapse.** High-frequency challenges tend to give higher spread;
   the LLM's search can drift toward minute-cadence-heavy blends and the
   yearly / monthly generalisation claim disappears.

4. **Parrot-just-below-floor.** `parrot_gate` catches a context-parroter that
   *beats* strong. But a state tuned so the parroter sits just below strong —
   near-trivially forecastable but technically clearing the floor — still
   measures no skill difference between competing TSFMs.

5. **Metric-coupling exploitation.** All three fitness factors (spread, ordering,
   gate) draw from the same challenge set. If the LLM finds a corner where they
   co-vary from one shared property (e.g. "high spectral content"), it exploits
   the correlation while underlying benchmark quality drops.

6. **Panel drift over versions.** Within a version the reference panel is
   frozen. Across many epochs the state drifts far from the panel's calibration
   regime; panel orderings that made sense on the broad initial distribution
   become knife-edge accidents on the drifted content.

7. **Real-data over-augmentation.** Post-`ScrapedLiveSource` wiring, the LLM
   might discover that heavier augmentation of real motifs raises spread
   (augmented reality is harder). Over-augmented real data is no longer real,
   destroying the anti-generator-fitting property that made real data valuable.

## Defenses shipped in this PR

Each defense turns one failure mode into an unconditional `fitness = 0`
outcome — so the LLM cannot get *any* credit for pushing that direction.

| defense | catches | code |
|---|---|---|
| **DGP-class breadth hard veto** — if any DGP class in the pool has share below `min_share`, `dgp_class_breadth_gate = 0`, so foundational_fitness → 0 | domain collapse (#2) | `score.dgp_class_breadth_gate` |
| **Cadence-band breadth hard veto** — if any cadence band in the pool has share below `min_share`, `cadence_breadth_gate = 0` | cadence collapse (#3) | `score.cadence_breadth_gate` |
| **Augmentation-preserves-DGP check** — for `count_discrete` and `bounded` archetypes, an augmented motif that violates the invariant (non-integer counts, out-of-bounds) is dropped from the pool | over-augmentation (#7) | (basic form; see follow-up) |
| **Reward-hacking test suite** — pins the above invariants: a state that produces mono-class or mono-cadence pools cannot achieve fitness > 0 | regression check | `tests/test_reward_hacking.py` |

## Defenses queued (follow-up PRs)

| defense | catches | scope |
|---|---|---|
| **Held-out validation panel** — a second reference panel the LLM never sees; alarm on ordering divergence between the two panels | anchor exploitation (#1), metric coupling (#5) | new `score.HeldOutPanel`; forge run halts and rolls back on divergence |
| **Canary TSFM tracking** — score a fixed zero-shot TSFM (e.g. Chronos) across the forge's epochs; alarm if its generalisation gap grows | anchor exploitation (#1), panel drift (#6) | new `evaluate.canary_track()`; log per-epoch |
| **Panel jitter** — periodically re-calibrate the reference panel on held-out state samples; a version-freeze relaxation | panel drift (#6) | new `score.recalibrate_panel()` on version boundaries |
| **Adversarial mutation testing** — per-epoch, also try 3-5 hand-coded random mutations; use theirs if any beat the LLM's | metric coupling (#5), general reward hacking | `forge_loop.run_forge(adversarial_probes=N)` |
| **Per-knob total drift cap** — rate-limit total per-knob drift across a version (e.g. `w_synth` can't move more than 30% from initial) | monotone corner-chasing | `program.md` invariant enforced in `GeneratorState.clamped()` |
| **Cross-run consistency** — run the forge N times with different LLM seeds; require final states to be within some distance of each other | reward hacking that exploits non-determinism | offline audit tool, not per-round |
| **Sharper parrot floor** — require the winning model to beat parrot by a *margin* proportional to spread, not just strictly | parrot-just-below-floor (#4) | `evaluate.clears_floor(margin=k*spread)` |

## Composition guarantee

The three fitness factors and both breadth gates multiply:

```
foundational_fitness = fitness
                     × coverage_gate
                     × parrot_gate
                     × dgp_class_breadth_gate    ← new (this PR)
                     × cadence_breadth_gate      ← new (this PR)
```

Multiplication means any *one* factor going to zero forces the aggregate to
zero. The LLM cannot trade off "great spread + terrible domain breadth" — the
breadth veto zeroes it unconditionally.

## Not a replacement for judgment

Reward-hacking defenses raise the bar; they don't eliminate the risk. The
LLM's search space is large and the defenses catch the failure modes we can
enumerate. For a benchmark that's a load-bearing input to a live subnet, plan
for periodic manual review of the forge log + a canary-TSFM regression test
alongside every version bump.
