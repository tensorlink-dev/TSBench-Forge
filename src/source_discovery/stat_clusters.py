"""Statistical archetype clusters — a third balance axis next to domain and cadence.

Why this exists
---------------
Domain and cadence are labels about where a series came from, not about what
kind of forecasting problem it is. Two hourly nature series can be a smooth
diurnal curve and an intermittent spike train. When the pool's statistical mix
differs from what external benchmarks (GIFT-Eval, BOOM, TIME) contain,
optimising for the pool pulls models away from those benchmarks: the pool
rewards the wrong behaviour. This was observed as miners who win cascade rounds
while their GIFT/BOOM/TIME scores fall.

What it does
------------
1. ``fit`` clusters series in the 15-feature space of ``tsfeatures`` with
   k-means++ (numpy only). Fit on the UNION of the pool and the reference
   benchmarks: a region the pool lacks still gets its own cluster, so the gap
   shows up as a cluster with near-zero pool share. Fitted on the pool alone,
   that region would have no centroid and the gap would be invisible.
2. ``alignment`` compares the pool's share of each archetype with a reference
   mixture and reports divergence, coverage and per-cluster ratios.
3. ``importance_weights`` turns those ratios into per-archetype sampling
   weights (clipped), so the eval draw can follow the reference mix without
   any new data, and ``sourcing_targets`` ranks the archetypes to look for.

Standardisation is robust (median / IQR, clipped at ±5) because several
features (spikiness, lumpiness, kurtosis) have very heavy tails. Plain
z-scores would let a handful of extreme series own whole clusters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from source_discovery.tsfeatures import FEATURES

CLIP = 5.0


@dataclass
class Archetypes:
    center: np.ndarray            # per-feature median used for scaling
    scale: np.ndarray             # per-feature IQR (floored)
    centroids: np.ndarray         # (k, n_features) in scaled space
    features: list[str] = field(default_factory=lambda: list(FEATURES))

    @property
    def k(self) -> int:
        return int(self.centroids.shape[0])

    def scale_matrix(self, mat: np.ndarray) -> np.ndarray:
        z = (np.asarray(mat, dtype=float) - self.center) / self.scale
        return np.clip(np.nan_to_num(z), -CLIP, CLIP)

    def assign(self, mat: np.ndarray) -> np.ndarray:
        z = self.scale_matrix(mat)
        d = ((z[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return d.argmin(axis=1)

    def describe(self, top: int = 3) -> list[str]:
        """Name each centroid by its most extreme features (scaled units)."""
        out = []
        for c in self.centroids:
            idx = np.argsort(-np.abs(c))[:top]
            out.append(", ".join(f"{'+' if c[i] > 0 else '-'}{self.features[i]}" for i in idx))
        return out

    def to_dict(self) -> dict:
        return {"features": self.features, "center": self.center.tolist(),
                "scale": self.scale.tolist(), "centroids": self.centroids.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Archetypes":
        return cls(np.array(d["center"]), np.array(d["scale"]),
                   np.array(d["centroids"]), list(d["features"]))


def _kmeans(z: np.ndarray, k: int, rng: np.random.Generator, iters: int = 100):
    n = z.shape[0]
    cent = [z[rng.integers(n)]]
    d2 = ((z - cent[0]) ** 2).sum(axis=1)
    for _ in range(1, k):                                   # k-means++ seeding
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        cent.append(z[rng.choice(n, p=p)])
        d2 = np.minimum(d2, ((z - cent[-1]) ** 2).sum(axis=1))
    cent = np.array(cent)
    for _ in range(iters):
        lab = ((z[:, None, :] - cent[None]) ** 2).sum(axis=2).argmin(axis=1)
        new = np.array([z[lab == j].mean(axis=0) if (lab == j).any() else cent[j]
                        for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    inertia = float(((z - cent[lab]) ** 2).sum())
    return cent, lab, inertia


def fit(mat: np.ndarray, k: int = 12, seed: int = 0, n_init: int = 5) -> Archetypes:
    """Fit k archetypes. ``mat`` should hold pool AND reference series."""
    mat = np.asarray(mat, dtype=float)
    center = np.nanmedian(mat, axis=0)
    q75, q25 = np.nanpercentile(mat, 75, axis=0), np.nanpercentile(mat, 25, axis=0)
    scale = np.maximum(q75 - q25, 1e-3)
    arch = Archetypes(center, scale, np.zeros((k, mat.shape[1])))
    z = arch.scale_matrix(mat)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        cent, _, inertia = _kmeans(z, k, rng)
        if best is None or inertia < best[1]:
            best = (cent, inertia)
    arch.centroids = best[0]
    return arch


def shares(labels: np.ndarray, k: int, weights: np.ndarray | None = None) -> np.ndarray:
    w = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=float)
    s = np.bincount(np.asarray(labels, dtype=int), weights=w, minlength=k).astype(float)
    return s / s.sum() if s.sum() > 0 else s


def alignment(pool_labels: np.ndarray, ref_labels: np.ndarray, k: int,
              ref_weights: np.ndarray | None = None, eps: float = 1e-4) -> dict:
    """How far the pool's archetype mix is from the reference mix.

    ``ref_weights`` lets a benchmark count by dataset rather than by series, so
    one giant dataset (GIFT's 370-series electricity) does not set the target.
    """
    p = shares(pool_labels, k)
    q = shares(ref_labels, k, ref_weights)
    ps, qs = p + eps, q + eps
    ps, qs = ps / ps.sum(), qs / qs.sum()
    m = 0.5 * (ps + qs)
    js = float(0.5 * (ps * np.log(ps / m)).sum() + 0.5 * (qs * np.log(qs / m)).sum())
    return {
        "pool_share": p, "ref_share": q,
        "ratio": qs / ps,                                    # >1: pool under-covers
        "js_divergence": js,                                 # 0 = identical, ln2 = disjoint
        "overlap": float(np.minimum(p, q).sum()),            # 1 = identical mixes
        "uncovered_ref_mass": float(q[p < 0.25 * q].sum()),  # ref mass the pool barely has
    }


def importance_weights(al: dict, lo: float = 0.25, hi: float = 4.0) -> np.ndarray:
    """Per-archetype draw weights that tilt the eval toward the reference mix.

    Clipped: an archetype the pool holds 3 series of must not be up-weighted
    100x, since that would hand a few series (and their feeds) the verdict.
    That case is a SOURCING gap (see ``sourcing_targets``), not a weighting one.
    """
    return np.clip(al["ratio"], lo, hi)


def sourcing_targets(al: dict, pool_size: int) -> list[tuple[int, int, float]]:
    """(cluster, series_needed_to_match_ref, ref_share), largest gap first."""
    need = np.maximum(0, np.round(al["ref_share"] * pool_size - al["pool_share"] * pool_size))
    order = np.argsort(-need)
    return [(int(c), int(need[c]), float(al["ref_share"][c])) for c in order if need[c] > 0]
