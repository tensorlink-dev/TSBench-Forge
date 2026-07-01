"""Equal-weight-per-DGP-class cross-domain sampler for the eval pool.

The goal: force *generalisation-across-domains* to be the primary axis of the
eval-pool composition, so a source-count-heavy domain like `nature` (20 sources,
32k rows/day) doesn't drown out a source-count-light one like `healthcare`
(5 sources, 58 rows/day).

Two orthogonal weightings compose:

1. **Per-domain equal weight.** N_domains = 7 (GIFT-Eval taxonomy); each domain
   contributes ``1/N_domains`` of the pool by default. A domain with 3 available
   series still occupies the same pool slice as a domain with 300; empty domains
   are dropped and their weight redistributed uniformly across the rest.
2. **Per-DGP-class equal weight *within* domain.** A domain with 8 DGP classes
   allocates ``1/8`` of its domain slice to each class. Prevents the
   "20-weather-stations = 20 votes for nature" trap.

Optional third weighting:

3. **Per-cadence equal weight.** Same logic across `{sub-min, few-min, 30-min,
   hourly, daily, weekly, monthly, quarterly, yearly}`. Enable to force cadence
   generalisation as an eval axis.

Also computes the two headline generalisation metrics (called from the scorer,
after per-window scores exist):

- ``generalisation_gap(scores_by_domain)`` -> best_domain - worst_domain
- ``lodo_score(scores_by_domain, held_out)``  -> the model's held-out-domain score

Pure stdlib + numpy. No dependency on the catalog file format — the caller
passes in a list of ``SeriesMeta`` dicts.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


FREQ_BAND = {
    "PT30S": "sub-min", "PT1M": "sub-min", "PT2M30S": "sub-min",
    "PT5M": "few-min", "PT6M": "few-min", "PT10M": "few-min", "PT15M": "few-min",
    "PT30M": "half-hour",
    "PT1H": "hourly", "PT8H": "hourly",
    "P1D": "daily", "P1W": "weekly", "P1M": "monthly", "P1Q": "quarterly", "P1Y": "yearly",
}


@dataclass(frozen=True)
class SeriesMeta:
    """One univariate series available to the pool sampler.

    Panel-expanded (a 50-station METAR source contributes 50 SeriesMeta rows).
    """
    series_id: str
    domain: str
    dgp_class: str
    freq: str
    source_id: str          # the sources.yaml `id`; kept for dedup + audit
    length: int             # number of available observations
    context_length: int = 0
    horizon: int = 0

    @property
    def cadence_band(self) -> str:
        return FREQ_BAND.get(self.freq, "other")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _distinct(values: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for v in values:
        seen.setdefault(v, None)
    return list(seen)


def sample_pool(
    catalog: Sequence[SeriesMeta],
    n_windows: int,
    *,
    rng: np.random.Generator | None = None,
    enforce_cadence_balance: bool = False,
    domains: Sequence[str] | None = None,
) -> list[SeriesMeta]:
    """Draw ``n_windows`` series (with replacement) using the equal-weight scheme.

    Args:
        catalog: every available (panel-expanded) series.
        n_windows: total pool slots to fill.
        rng: numpy Generator; defaults to a fresh ``default_rng()``.
        enforce_cadence_balance: if True, apply per-cadence-band equal weight on
            top of per-domain and per-DGP-class weighting.
        domains: restrict to this ordered list of domains (drop others). Useful
            for the leave-one-domain-out ablation.

    Returns:
        A list of SeriesMeta, one per pool slot. Order is deterministic under
        the same rng seed.

    Behavior guarantees:
        - If a domain has zero catalog entries, its slot is redistributed.
        - If a DGP class within a domain has zero entries, its slot is
          redistributed within the domain.
        - Series may repeat across slots (n_windows can exceed catalog size).
    """
    if not catalog:
        raise ValueError("catalog is empty; cannot sample a pool")
    if n_windows <= 0:
        raise ValueError(f"n_windows must be positive, got {n_windows}")
    rng = rng or np.random.default_rng()

    # Restrict domains if asked; filter empties.
    if domains is not None:
        catalog = [m for m in catalog if m.domain in set(domains)]
    domain_names = _distinct(m.domain for m in catalog)
    if not domain_names:
        raise ValueError("no catalog entries survive the domain filter")

    # Per-domain weight. Uniform across surviving (non-empty) domains.
    n_dom = len(domain_names)
    per_domain_slots = _distribute_slots(n_windows, n_dom)

    picks: list[SeriesMeta] = []
    for domain, dom_slots in zip(domain_names, per_domain_slots):
        dom_series = [m for m in catalog if m.domain == domain]
        # DGP classes present in this domain — uniform within.
        classes = _distinct(m.dgp_class for m in dom_series)
        per_class_slots = _distribute_slots(dom_slots, len(classes))

        for cls, cls_slots in zip(classes, per_class_slots):
            cls_series = [m for m in dom_series if m.dgp_class == cls]

            if enforce_cadence_balance:
                bands = _distinct(m.cadence_band for m in cls_series)
                per_band_slots = _distribute_slots(cls_slots, len(bands))
                for band, band_slots in zip(bands, per_band_slots):
                    pool = [m for m in cls_series if m.cadence_band == band]
                    picks.extend(_sample_with_replacement(pool, band_slots, rng))
            else:
                picks.extend(_sample_with_replacement(cls_series, cls_slots, rng))

    assert len(picks) == n_windows, (len(picks), n_windows)
    return picks


def _distribute_slots(total: int, k: int) -> list[int]:
    """Split ``total`` slots into ``k`` groups as evenly as possible.

    Extras go to the first groups. If k is 0 (empty domain / class), returns [].
    """
    if k == 0:
        return []
    base, extra = divmod(total, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


def _sample_with_replacement(
    pool: Sequence[SeriesMeta],
    n: int,
    rng: np.random.Generator,
) -> list[SeriesMeta]:
    if n == 0:
        return []
    if not pool:
        # An empty leaf shouldn't happen because empty classes/bands are already
        # dropped upstream, but guard anyway.
        raise ValueError("empty leaf pool during sampling")
    idx = rng.integers(0, len(pool), size=n)
    return [pool[i] for i in idx]


# ---------------------------------------------------------------------------
# Post-scoring metrics
# ---------------------------------------------------------------------------


def generalisation_gap(scores_by_domain: dict[str, float]) -> float:
    """Best-domain minus worst-domain score. Small = generalist, large = specialist.

    Assumes 'lower is better' (MASE-style). Flip the sign for accuracy-style
    metrics.

    A model that scores {0.7, 0.72, 0.68} across domains has a gap of 0.04 (tight
    generalist). One that scores {0.5, 0.9, 1.4} has a gap of 0.9 — good at one,
    bad at another.
    """
    if not scores_by_domain:
        raise ValueError("scores_by_domain is empty")
    vals = [v for v in scores_by_domain.values() if not np.isnan(v)]
    if not vals:
        return float("nan")
    return float(max(vals) - min(vals))


def equal_domain_aggregate(scores_by_domain: dict[str, float]) -> float:
    """Aggregate scores with 1/N-per-domain weight (not row-weighted).

    Same rationale as the sampler: prevents a domain with 30 series from
    dominating the aggregate over one with 5.
    """
    if not scores_by_domain:
        raise ValueError("scores_by_domain is empty")
    vals = [v for v in scores_by_domain.values() if not np.isnan(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def lodo_score(
    scores_by_domain: dict[str, float],
    held_out: str,
) -> tuple[float, float]:
    """Leave-one-domain-out generalisation test.

    Returns ``(held_out_score, in_distribution_mean)``. A model that has been
    over-tuned to the training domains shows a large gap between the two.

    The pool for the round should have been built with
    ``sample_pool(..., domains=[all except held_out])``, then the trained model
    scored on `held_out` at eval time. The scorer stores per-domain means; this
    function reads them out.
    """
    if held_out not in scores_by_domain:
        raise KeyError(f"held_out domain {held_out!r} not in scores_by_domain")
    ho = scores_by_domain[held_out]
    id_mean = float(np.mean([v for k, v in scores_by_domain.items() if k != held_out]))
    return float(ho), id_mean


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------


def coverage_report(catalog: Sequence[SeriesMeta]) -> dict:
    """Snapshot of the DGP-class × domain × cadence coverage — for pool audit.

    Emit this into the pool's provenance.json so validators can see how the
    equal-weight sampling actually distributed slots on a given day.
    """
    per_domain: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_class: dict[str, int] = defaultdict(int)
    per_band: dict[str, int] = defaultdict(int)
    for m in catalog:
        per_domain[m.domain][m.dgp_class] += 1
        per_class[m.dgp_class] += 1
        per_band[m.cadence_band] += 1
    return {
        "n_series": len(catalog),
        "n_domains": len(per_domain),
        "n_dgp_classes": len(per_class),
        "n_cadence_bands": len(per_band),
        "series_per_domain_class": {d: dict(v) for d, v in per_domain.items()},
        "series_per_dgp_class": dict(per_class),
        "series_per_cadence_band": dict(per_band),
    }
