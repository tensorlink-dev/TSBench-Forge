"""Statistical comparison of forecasters on the live benchmark.

The leaderboard (:func:`evaluate.leaderboard`) ranks models by a single
seasonal-naive-relative shifted-gmean number. That answers "who is ahead" but
not "is the gap real". This module closes that gap: it collects **per-challenge**
error scores for each model on an identical challenge set, then runs paired
significance tests so a ranking can be reported with p-values instead of vibes.

Design
------
* One inference pass per model. :func:`score_models` runs each forecaster once
  over the shared challenge list and returns aligned per-challenge MASE/CRPS
  arrays (same challenge order for every model → the samples are *paired*).
* Paired tests, not unpaired. The same challenges are forecast by every model,
  so the correct comparison is a **paired** one (Wilcoxon signed-rank — the
  robust default for heavy-tailed forecast errors — and a paired t-test on the
  differences as a parametric companion). Win-rate (fraction of challenges where
  A beats B) is reported alongside because it is interpretable and distribution
  free.
* Multiple comparisons are corrected. Comparing k models pairwise is k(k-1)/2
  tests; :func:`pairwise_significance` applies Benjamini-Hochberg FDR across the
  family. Vs-baseline comparisons use Holm.
* Friedman omnibus first. Before reading pairwise cells, :func:`friedman_omnibus`
  checks whether *any* model differs — if it can't reject, the pairwise table is
  noise.

Caveat the caller must respect
------------------------------
Challenges drawn from a small number of source series are **not independent** —
several windows share a parent series. The tests below treat challenges as the
unit; with few sources this overstates significance. Report the source count and,
when it is small, lean on the per-source-relative leaderboard for ranking and use
these tests as corroboration, not proof. :func:`source_clustered_note` emits the
warning automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from evaluate import _per_challenge_scores, friedman_test, probabilistic
from score import seasonal_naive


@dataclass
class ModelScores:
    """Per-challenge error arrays for one model, aligned to a shared challenge list."""

    name: str
    mase: np.ndarray
    crps: np.ndarray
    weight: np.ndarray
    source_ids: np.ndarray = field(default=None)

    def metric(self, which: str) -> np.ndarray:
        return self.crps if which == "crps" else self.mase


def score_models(
    models: dict[str, "callable"],
    challenges: list,
    *,
    seasonality: int | None = None,
    include_seasonal_naive: bool = True,
) -> dict[str, ModelScores]:
    """Run every model once over ``challenges`` → aligned per-challenge scores.

    Returns an insertion-ordered dict ``{name: ModelScores}``. The seasonal-naive
    baseline is added under key ``"seasonal_naive"`` unless already present, so it
    is always available as the reference rung for :func:`compare_to_baseline`.
    """
    source_ids = np.array(
        [str((getattr(ch, "meta", None) or {}).get("source_id")) for ch in challenges]
    )
    out: dict[str, ModelScores] = {}
    named = dict(models)
    if include_seasonal_naive and "seasonal_naive" not in named:
        named = {"seasonal_naive": probabilistic(seasonal_naive), **named}
    for name, fc in named.items():
        mase, crps, w = _per_challenge_scores(fc, challenges, seasonality)
        out[name] = ModelScores(name, mase, crps, w, source_ids)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Paired tests
# ──────────────────────────────────────────────────────────────────────────


def paired_test(a: np.ndarray, b: np.ndarray, *, alternative: str = "two-sided") -> dict:
    """Paired comparison of two per-challenge error arrays (lower = better).

    Differences are ``d = a - b`` so a *negative* median/mean means ``a`` has the
    lower error (a is better). ``alternative='less'`` tests H1: a < b (a better).
    Returns Wilcoxon signed-rank and paired-t p-values, the effect sizes, and the
    win-rate (fraction of challenges where a strictly beats b).
    """
    from scipy import stats

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    wins = float(np.mean(a < b)) if n else float("nan")
    nonzero = d[d != 0]
    # Wilcoxon needs at least one non-zero difference.
    if len(nonzero) >= 1:
        try:
            w_stat, w_p = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
        except ValueError:
            w_stat, w_p = float("nan"), 1.0
    else:
        w_stat, w_p = float("nan"), 1.0
    if n >= 2 and np.std(d) > 0:
        t_stat, t_p = stats.ttest_rel(a, b, alternative=alternative)
    else:
        t_stat, t_p = float("nan"), 1.0
    return {
        "n": n,
        "median_diff": float(np.median(d)),
        "mean_diff": float(np.mean(d)),
        "win_rate": wins,           # P(a beats b)
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        "t_stat": float(t_stat),
        "t_p": float(t_p),
    }


def compare_to_baseline(
    scores: dict[str, ModelScores],
    *,
    metric: str = "crps",
    baseline: str = "seasonal_naive",
    alpha: float = 0.05,
) -> list[dict]:
    """Each model vs the baseline: is it *significantly better* (one-sided)?

    Holm-corrected across the family of models. Rows are sorted by median
    improvement (most-improved first). ``significant`` uses the Holm-adjusted
    Wilcoxon p at ``alpha``.
    """
    base = scores[baseline].metric(metric)
    rows = []
    for name, s in scores.items():
        if name == baseline:
            continue
        r = paired_test(s.metric(metric), base, alternative="less")
        rows.append({"model": name, **r})
    _holm(rows, key="wilcoxon_p", out="wilcoxon_p_holm")
    _holm(rows, key="t_p", out="t_p_holm")
    for r in rows:
        r["significant"] = bool(r["wilcoxon_p_holm"] < alpha)
    rows.sort(key=lambda r: r["median_diff"])   # most negative = best improvement
    return rows


def pairwise_significance(
    scores: dict[str, ModelScores], *, metric: str = "crps", alpha: float = 0.05
) -> dict:
    """All-pairs paired Wilcoxon with Benjamini-Hochberg FDR across the family.

    Returns ``names`` (row/col order, best→worst by weighted-mean metric),
    ``p`` (symmetric matrix of BH-adjusted two-sided Wilcoxon p-values, NaN on the
    diagonal), ``win`` (win-rate matrix; ``win[i,j]`` = P(model i beats model j)),
    and ``better`` (matrix of the sign of median(i)-median(j): -1 if i better).
    """
    names = sorted(
        scores,
        key=lambda n: float(np.average(scores[n].metric(metric), weights=scores[n].weight)),
    )
    k = len(names)
    p = np.full((k, k), np.nan)
    win = np.full((k, k), np.nan)
    better = np.zeros((k, k))
    raw = []
    idx = []
    for i in range(k):
        for j in range(i + 1, k):
            r = paired_test(scores[names[i]].metric(metric), scores[names[j]].metric(metric))
            raw.append(r["wilcoxon_p"])
            idx.append((i, j))
            win[i, j] = r["win_rate"]
            win[j, i] = 1.0 - r["win_rate"]
            better[i, j] = np.sign(r["median_diff"])
            better[j, i] = -better[i, j]
    adj = _benjamini_hochberg(raw)
    for (i, j), pa in zip(idx, adj):
        p[i, j] = p[j, i] = pa
    return {"names": names, "p": p, "win": win, "better": better, "alpha": alpha}


def friedman_omnibus(scores: dict[str, ModelScores], *, metric: str = "crps") -> dict:
    """Friedman test across all models on the paired per-challenge metric.

    Reject → at least one model differs; only then is the pairwise table meaningful.
    """
    names = list(scores)
    mat = np.column_stack([scores[n].metric(metric) for n in names])  # n_challenges x k
    res = friedman_test(mat)
    res["models"] = names
    return res


# ──────────────────────────────────────────────────────────────────────────
# multiple-comparison corrections (no statsmodels dependency)
# ──────────────────────────────────────────────────────────────────────────


def _holm(rows: list[dict], *, key: str, out: str) -> None:
    """In-place Holm-Bonferroni step-down adjustment of ``rows[i][key]`` → ``out``."""
    m = len(rows)
    order = sorted(range(m), key=lambda i: rows[i][key])
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * rows[i][key])
        running = max(running, adj)   # enforce monotonicity
        rows[i][out] = running


def _benjamini_hochberg(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR adjustment; returns adjusted p-values in input order."""
    try:
        from scipy.stats import false_discovery_control

        return list(false_discovery_control(np.asarray(pvals, dtype=float), method="bh"))
    except Exception:
        pass
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        adj[i] = prev
    return list(adj)


def source_clustered_note(scores: dict[str, ModelScores]) -> str:
    """Warn when challenges cluster on few source series (tests overstate power)."""
    any_scores = next(iter(scores.values()))
    if any_scores.source_ids is None:
        return ""
    n_src = len(set(any_scores.source_ids.tolist()))
    n_ch = len(any_scores.source_ids)
    msg = f"{n_ch} challenges drawn from {n_src} source series"
    if n_src < 12:
        msg += (
            f" — LOW BREADTH: challenges are not independent (≈{n_ch / max(n_src,1):.0f} "
            "per source). Treat pairwise p-values as corroboration, not proof; the "
            "per-source-relative leaderboard is the primary ranking."
        )
    return msg
