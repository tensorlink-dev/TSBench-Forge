#!/usr/bin/env python3
"""Replay ensemble-calibration policies over stored paracast rounds.

The question: would a policy calibrated on *past* rounds have beaten the
served `paracast-ensemble` on the *next* round? Every policy here sees only
rounds strictly before the one it is scored on, and every score is computed by
`model_comparison.leaderboard_from_scores` -- the leaderboard's own
per-source seasonal-naive-relative shifted gmean -- so a policy's number sits
on the same scale as the published ranks.

What stored rounds can and cannot answer
----------------------------------------
Each round artifact keeps per-challenge CRPS/MASE for every model, not the
forecasts. That is enough to replay any *selection* policy exactly (a
challenge takes the chosen model's own score), but not a re-*weighted*
ensemble: a mixture's CRPS is not a function of its members' CRPS. The
weighting replay needs the member forecasts that rounds do not yet persist.

Policies (each a per-challenge choice of which model's score to take):

* ``trailing_leader[W]``  -- the single best model over the last W rounds.
* ``domain_leader[W]``    -- per data domain (econ_fin, energy, ...), the best
  model on that domain over the last W rounds; the domain-profile question,
  and the closest stored-data analogue of a per-customer calibration.
* ``source_leader[W]``    -- the same per source series, falling back to the
  domain leader where a source has too little history.
* ``hindsight_*``         -- the same choices made with the round's own
  results: not a policy, a ceiling on what the selection could ever buy.

Each runs over two candidate sets: the panel members alone, and the members
plus the served ensemble (so "keep the ensemble here" is a choice it can make).

Caveats the report repeats: the twice-daily runs are not independent rounds,
and ~20 days of retained artifacts cannot test windows of 14-28 days.

    python scripts/replay_calibration.py --rounds-dir replay_rounds \
        --out replay_report.json --markdown replay_report.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from model_comparison import (  # noqa: E402
    ModelScores,
    leaderboard_from_scores,
    scores_from_serialized,
)

ENSEMBLE = "paracast-ensemble"
ROUTER = "paracast-router"
PSEUDO = {ENSEMBLE, ROUTER}
BASELINE = "seasonal_naive"
METRICS = ("crps", "mase")
UNKNOWN = "unknown"

# A domain or source needs this many challenges in a round before its score
# there counts as evidence; below it one wild series decides the "leader".
MIN_DOMAIN_CHALLENGES = 20
MIN_SOURCE_CHALLENGES = 3

CATALOGS = ("src/sources/sources.yaml", "src/sources/sources-slow.yaml")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


@dataclass
class Round:
    """One stored round: aligned per-challenge scores plus challenge tags."""

    label: str
    scores: dict[str, ModelScores]
    members: list[str]
    domains: np.ndarray
    cadences: np.ndarray | None = None
    # Filled lazily: {(scope, key): {model: crps_rel}}.
    _rel_cache: dict = field(default_factory=dict, repr=False)

    @property
    def sources(self) -> np.ndarray:
        return self.scores[BASELINE].source_ids

    def rel(self, idx: np.ndarray | None = None, key=None) -> dict[str, float]:
        """crps_rel per model, over the challenges in ``idx`` (all if None)."""
        cache_key = key if key is not None else "all"
        if cache_key not in self._rel_cache:
            subset = self.scores if idx is None else subset_scores(self.scores, idx)
            rows = leaderboard_from_scores(subset, metrics=("crps",))
            self._rel_cache[cache_key] = {r["model"]: r["crps_rel"] for r in rows}
        return self._rel_cache[cache_key]

    def domain_rel(self, domain: str) -> dict[str, float] | None:
        idx = np.flatnonzero(self.domains == domain)
        if idx.size < MIN_DOMAIN_CHALLENGES:
            return None
        return self.rel(idx, key=("domain", domain))

    def source_rel(self, source: str) -> dict[str, float] | None:
        idx = np.flatnonzero(self.sources == source)
        if idx.size < MIN_SOURCE_CHALLENGES:
            return None
        return self.rel(idx, key=("source", source))


def _catalog(repo: Path = REPO) -> list[dict]:
    entries: list[dict] = []
    for rel in CATALOGS:
        path = repo / rel
        if path.exists():
            loaded = yaml.load(path.read_text(), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
            entries += [e for e in loaded or [] if isinstance(e, dict) and "id" in e]
    return entries


def load_domain_map(repo: Path = REPO) -> dict[str, str]:
    return {e["id"]: e.get("domain") or UNKNOWN for e in _catalog(repo)}


# How fast a source's newest window moves. The split matters because a slow
# source can serve the *same* eval window on consecutive rounds, and then
# "last round's winner on this series" is scored on data it was chosen on.
CADENCE_BUCKETS = ("sub_daily", "daily", "slower", UNKNOWN)
# ISO 8601 reuses "M": months before the "T", minutes after it.
_DATE_UNITS = {"Y": 365 * 86400, "M": 30 * 86400, "W": 7 * 86400, "D": 86400}
_TIME_UNITS = {"H": 3600, "M": 60, "S": 1}
_ISO = re.compile(r"P(?:(\d+)([YMWD]))?(?:T(\d+)([HMS]))?")


def cadence_bucket(freq: str | None) -> str:
    m = _ISO.fullmatch(str(freq or "").strip())
    if not m or not any(m.groups()):
        return UNKNOWN
    seconds = 0
    if m.group(1):
        seconds += int(m.group(1)) * _DATE_UNITS[m.group(2)]
    if m.group(3):
        seconds += int(m.group(3)) * _TIME_UNITS[m.group(4)]
    if seconds < 86400:
        return "sub_daily"
    return "daily" if seconds == 86400 else "slower"


def load_cadence_map(repo: Path = REPO) -> dict[str, str]:
    return {e["id"]: cadence_bucket(e.get("frequency")) for e in _catalog(repo)}


def round_from_results(
    label: str, results: dict, domain_map: dict[str, str], cadence_map: dict[str, str] | None = None
) -> Round | None:
    """Build a Round from a results.json payload, or None if it cannot be replayed."""
    pc = results.get("per_challenge")
    if not pc:
        return None
    scores = scores_from_serialized(pc)
    if BASELINE not in scores or ENSEMBLE not in scores:
        return None
    roster = (results.get("config") or {}).get("roster") or []
    members = [m for m in roster if m not in PSEUDO and m in scores]
    if len(members) < 2:
        return None
    domains = np.array([domain_map.get(s, UNKNOWN) for s in pc["source_ids"]])
    cadences = np.array([(cadence_map or {}).get(s, UNKNOWN) for s in pc["source_ids"]])
    return Round(label=label, scores=scores, members=members, domains=domains, cadences=cadences)


def load_rounds(
    rounds_dir: Path, domain_map: dict[str, str], cadence_map: dict[str, str] | None = None
) -> tuple[list[Round], list[str]]:
    """Every ``<label>/results.json`` under ``rounds_dir``, oldest first.

    Labels are expected to sort chronologically (the workflow names each
    directory by the artifact's creation timestamp).
    """
    rounds, skipped = [], []
    for path in sorted(rounds_dir.glob("*/results.json")):
        label = path.parent.name
        rnd = round_from_results(label, json.loads(path.read_text()), domain_map, cadence_map)
        if rnd is None:
            skipped.append(label)
        else:
            rounds.append(rnd)
    return rounds, skipped


# --------------------------------------------------------------------------- #
# Composing a policy's scores
# --------------------------------------------------------------------------- #


def subset_scores(scores: dict[str, ModelScores], idx: np.ndarray) -> dict[str, ModelScores]:
    return {
        n: ModelScores(n, s.mase[idx], s.crps[idx], s.weight[idx], s.source_ids[idx])
        for n, s in scores.items()
    }


def compose(name: str, scores: dict[str, ModelScores], choice: Sequence[str]) -> ModelScores:
    """A synthetic model whose challenge i is scored as ``choice[i]`` was."""
    choice = np.asarray(choice)
    base = scores[BASELINE]
    mase = np.empty_like(base.mase, dtype=float)
    crps = np.empty_like(base.crps, dtype=float)
    weight = np.empty_like(base.weight, dtype=float)
    for model in np.unique(choice):
        idx = choice == model
        src = scores[model]
        mase[idx], crps[idx], weight[idx] = src.mase[idx], src.crps[idx], src.weight[idx]
    return ModelScores(name, mase, crps, weight, base.source_ids)


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #

# A policy maps (history, current round, candidates) to a per-challenge choice.
Policy = Callable[[list[Round], Round, list[str]], np.ndarray]


def _mean_skill(tables: list[dict[str, float] | None], candidates: list[str]) -> dict[str, float]:
    """Mean crps_rel per candidate over the tables it appears in."""
    acc: dict[str, list[float]] = {}
    for table in tables:
        if not table:
            continue
        for m in candidates:
            v = table.get(m)
            if v is not None and np.isfinite(v):
                acc.setdefault(m, []).append(v)
    return {m: float(np.mean(v)) for m, v in acc.items()}


def _best(skill: dict[str, float]) -> str | None:
    return min(skill, key=skill.get) if skill else None


def trailing_leader(window: int | None) -> Policy:
    def policy(history, current, candidates):
        recent = history if window is None else history[-window:]
        best = _best(_mean_skill([r.rel() for r in recent], candidates))
        return np.full(current.sources.shape, best or candidates[0], dtype=object)

    return policy


def domain_leader(window: int | None) -> Policy:
    def policy(history, current, candidates):
        recent = history if window is None else history[-window:]
        fallback = _best(_mean_skill([r.rel() for r in recent], candidates)) or candidates[0]
        choice = np.full(current.sources.shape, fallback, dtype=object)
        for d in np.unique(current.domains):
            best = _best(_mean_skill([r.domain_rel(d) for r in recent], candidates))
            if best is not None:
                choice[current.domains == d] = best
        return choice

    return policy


def source_leader(window: int | None) -> Policy:
    by_domain = domain_leader(window)

    def policy(history, current, candidates):
        recent = history if window is None else history[-window:]
        choice = by_domain(history, current, candidates)
        for s in np.unique(current.sources):
            best = _best(_mean_skill([r.source_rel(s) for r in recent], candidates))
            if best is not None:
                choice[current.sources == s] = best
        return choice

    return policy


def hindsight_leader(history, current, candidates):
    return trailing_leader(1)([current], current, candidates)


def hindsight_domain_leader(history, current, candidates):
    return domain_leader(1)([current], current, candidates)


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #


def window_label(w: int | None) -> str:
    return "all" if w is None else str(w)


def build_policies(windows: Sequence[int | None]) -> dict[str, Policy]:
    policies: dict[str, Policy] = {}
    for w in windows:
        policies[f"trailing_leader[{window_label(w)}]"] = trailing_leader(w)
        policies[f"domain_leader[{window_label(w)}]"] = domain_leader(w)
        policies[f"source_leader[{window_label(w)}]"] = source_leader(w)
    policies["hindsight_leader"] = hindsight_leader
    policies["hindsight_domain_leader"] = hindsight_domain_leader
    return policies


CANDIDATE_SETS: dict[str, Callable[[Round], list[str]]] = {
    "members": lambda r: list(r.members),
    "members+ensemble": lambda r: [*r.members, ENSEMBLE],
}


def replay(
    rounds: list[Round], windows: Sequence[int | None], min_history: int = 1, gap: int = 0
) -> dict:
    """Score every policy on every round with ``min_history`` usable rounds before it.

    ``gap`` withholds the most recent rounds from the policy: with ``gap=2`` a
    policy scored on round i sees only rounds before i-2 (a full day, at two
    rounds a day). If a policy's edge collapses as the gap opens, it was being
    scored on windows it had effectively already seen.
    """
    policies = build_policies(windows)
    per_round = []
    for i in range(min_history + gap, len(rounds)):
        history, current = rounds[: i - gap], rounds[i]
        synthetic: dict[str, ModelScores] = {}
        choices: dict[str, dict[str, int]] = {}
        for set_name, pick in CANDIDATE_SETS.items():
            candidates = pick(current)
            for pol_name, policy in policies.items():
                name = f"{pol_name}|{set_name}"
                choice = policy(history, current, candidates)
                synthetic[name] = compose(name, current.scores, choice)
                choices[name] = dict(Counter(choice.tolist()))
        everything = {**current.scores, **synthetic}
        rows = leaderboard_from_scores(everything, metrics=METRICS)
        by_cadence: dict[str, dict[str, dict[str, float]]] = {m: {} for m in METRICS}
        if current.cadences is not None:
            for bucket in CADENCE_BUCKETS:
                idx = np.flatnonzero(current.cadences == bucket)
                if idx.size >= MIN_DOMAIN_CHALLENGES:
                    sub = leaderboard_from_scores(subset_scores(everything, idx), metrics=METRICS)
                    for m in METRICS:
                        by_cadence[m][bucket] = {r["model"]: r[f"{m}_rel"] for r in sub}
        per_round.append(
            {
                "round": current.label,
                "history": i - gap,
                **{f"{m}_rel": {r["model"]: r[f"{m}_rel"] for r in rows} for m in METRICS},
                **{f"{m}_rel_by_cadence": by_cadence[m] for m in METRICS},
                "choices": choices,
            }
        )
    return {
        "gap": gap,
        "per_round": per_round,
        "policies": list(policies),
        "candidate_sets": list(CANDIDATE_SETS),
    }


def _diff_table(per_round: list[dict], key: Callable[[dict], dict | None]) -> list[dict]:
    names = sorted({n for r in per_round for n in (key(r) or {})})
    table = []
    for name in names:
        diffs, vals = [], []
        for r in per_round:
            t = key(r) or {}
            v, e = t.get(name), t.get(ENSEMBLE)
            if v is None or e is None:
                continue
            vals.append(v)
            diffs.append(v - e)
        if not vals:
            continue
        diffs_a = np.asarray(diffs)
        table.append(
            {
                "row": name,
                "n_rounds": len(vals),
                "mean_rel": float(np.mean(vals)),
                "mean_diff_vs_ensemble": float(diffs_a.mean()),
                "wins_vs_ensemble": int((diffs_a < 0).sum()),
                "losses_vs_ensemble": int((diffs_a > 0).sum()),
                "wilcoxon_p": _wilcoxon(diffs_a),
            }
        )
    table.sort(key=lambda r: r["mean_rel"])
    return table


def summarize(result: dict, rounds: list[Round]) -> dict:
    """Per metric: mean relative score per row and its record against the served ensemble.

    Policies always *choose* by CRPS, the metric the leaderboard ranks on; MASE
    is reported for the same choices, to show whether a CRPS win costs the
    point forecast (a mixture can widen the fan while dragging the median).
    """
    per_round = result["per_round"]
    out: dict = {"stability": rank_stability(rounds), "domain_leaders": domain_leader_counts(rounds)}
    for m in METRICS:
        cad_key = f"{m}_rel_by_cadence"
        buckets = sorted({b for r in per_round for b in r.get(cad_key, {})})
        out[m] = {
            "table": _diff_table(per_round, lambda r, m=m: r.get(f"{m}_rel")),
            "by_cadence": {
                b: _diff_table(per_round, lambda r, b=b, k=cad_key: r.get(k, {}).get(b))
                for b in buckets
            },
        }
    return out


def _wilcoxon(diffs: np.ndarray) -> float | None:
    nonzero = diffs[diffs != 0]
    if nonzero.size < 6:
        return None
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    return float(wilcoxon(nonzero).pvalue)


def rank_stability(rounds: list[Round]) -> dict:
    """How well one round's member ranking predicts the next one's."""
    taus, same_leader = [], 0
    for prev, cur in zip(rounds, rounds[1:]):
        common = [m for m in prev.members if m in cur.members]
        if len(common) < 3:
            continue
        a = [prev.rel()[m] for m in common]
        b = [cur.rel()[m] for m in common]
        taus.append(_kendall_tau(a, b))
        same_leader += int(common[int(np.argmin(a))] == common[int(np.argmin(b))])
    return {
        "pairs": len(taus),
        "mean_kendall_tau": float(np.mean(taus)) if taus else None,
        "leader_repeats": same_leader,
        "leaders": dict(Counter(min(r.members, key=r.rel().get) for r in rounds)),
    }


def _kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += int(np.sign(a[i] - a[j]) * np.sign(b[i] - b[j]))
    return 2.0 * s / (n * (n - 1))


def domain_leader_counts(rounds: list[Round]) -> dict[str, dict[str, int]]:
    """Per domain: how often each member was that round's leader there."""
    out: dict[str, Counter] = {}
    for r in rounds:
        for d in np.unique(r.domains):
            table = r.domain_rel(d)
            if table:
                leader = min(r.members, key=lambda m: table.get(m, np.inf))
                out.setdefault(str(d), Counter())[leader] += 1
    return {d: dict(c.most_common()) for d, c in sorted(out.items())}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _table_md(
    table: list[dict], keep: Callable[[str], bool] = lambda _: True, metric: str = "crps"
) -> list[str]:
    lines = [
        f"| row | rounds | mean {metric}_rel | vs ensemble | W/L | wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in table:
        if not keep(r["row"]):
            continue
        p = "" if r["wilcoxon_p"] is None else f"{r['wilcoxon_p']:.3f}"
        lines.append(
            f"| `{r['row']}` | {r['n_rounds']} | {r['mean_rel']:.4f} | "
            f"{r['mean_diff_vs_ensemble']:+.4f} | {r['wins_vs_ensemble']}/{r['losses_vs_ensemble']} | {p} |"
        )
    return lines


def _policy_rows(name: str) -> bool:
    """The rows worth repeating in the per-gap and per-cadence sections."""
    return name in PSEUDO or name.startswith(("source_leader", "domain_leader", "hindsight"))


def to_markdown(
    summaries: dict[int, dict], rounds: list[Round], skipped: list[str]
) -> str:
    base = summaries[min(summaries)]
    lines = [
        "# Ensemble calibration replay",
        "",
        (
            f"{len(rounds)} rounds replayed ({rounds[0].label} .. {rounds[-1].label})"
            + (f"; {len(skipped)} skipped." if skipped else ".")
        )
        if rounds
        else "no rounds",
        "",
        "Every policy is scored on round *i* using only rounds before *i*. "
        "`hindsight_*` rows use the round's own results: a ceiling, not a policy. "
        "Twice-daily runs are not independent rounds, so read p-values as indicative.",
        "",
        "Policies choose by CRPS; MASE is the same choices scored on the point forecast.",
    ]
    for m in METRICS:
        lines += ["", f"## All rows, {m.upper()}, gap {min(summaries)}", "", *_table_md(base[m]["table"], metric=m)]
    for gap, summary in sorted(summaries.items()):
        for m in METRICS:
            lines += [
                "",
                f"## Gap {gap}, {m.upper()}: policies see nothing from the last {gap} rounds",
                "",
                *_table_md(summary[m]["table"], _policy_rows, metric=m),
            ]
            for bucket, table in summary[m]["by_cadence"].items():
                lines += [
                    "",
                    f"### gap {gap}, {m.upper()}, sources updating `{bucket}`",
                    "",
                    *_table_md(table, _policy_rows, metric=m),
                ]
    st = base["stability"]
    lines += [
        "",
        "## Rank stability (members, consecutive rounds)",
        "",
        f"- pairs: {st['pairs']}, mean Kendall tau: "
        + ("n/a" if st["mean_kendall_tau"] is None else f"{st['mean_kendall_tau']:.3f}"),
        f"- same leader as previous round: {st['leader_repeats']} of {st['pairs']}",
        f"- round leaders: {st['leaders']}",
        "",
        "## Domain leaders (member leading each domain, count of rounds)",
        "",
    ]
    for d, counts in base["domain_leaders"].items():
        lines.append(f"- **{d}**: {counts}")
    return "\n".join(lines) + "\n"


def parse_windows(text: str) -> list[int | None]:
    out: list[int | None] = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            out.append(None if tok == "all" else int(tok))
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds-dir", type=Path, required=True)
    ap.add_argument("--windows", default="1,4,14,all",
                    help="trailing windows in rounds (two rounds per day); 'all' = full history")
    ap.add_argument("--gaps", default="0,2,8,20",
                    help="rounds withheld from each policy before the scored round, comma-separated")
    ap.add_argument("--out", type=Path, default=Path("replay_report.json"))
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args(argv)

    rounds, skipped = load_rounds(args.rounds_dir, load_domain_map(), load_cadence_map())
    if len(rounds) < 2:
        print(f"error: need at least 2 replayable rounds, found {len(rounds)}", file=sys.stderr)
        return 1
    windows = parse_windows(args.windows)
    results, summaries = {}, {}
    for gap in sorted({int(g) for g in args.gaps.split(",") if g.strip()}):
        if gap + 1 >= len(rounds):
            continue
        results[gap] = replay(rounds, windows, gap=gap)
        summaries[gap] = summarize(results[gap], rounds)
    args.out.write_text(
        json.dumps({"by_gap": results, "summaries": summaries, "skipped": skipped}, indent=1)
    )
    md = to_markdown(summaries, rounds, skipped)
    if args.markdown:
        args.markdown.write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
