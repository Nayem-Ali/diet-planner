"""Aggregation and simple statistics (pure standard library)."""
from __future__ import annotations

import random
from typing import List, Tuple

_SEVERITY_WEIGHT = {"low": 1.0, "medium": 2.0, "high": 3.0}


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(xs: List[float], n: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean. Deterministic given seed."""
    if not xs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    m = len(xs)
    for _ in range(n):
        sample = [xs[rng.randrange(m)] for _ in range(m)]
        means.append(sum(sample) / m)
    means.sort()
    lo = means[int((alpha / 2) * n)]
    hi = means[int((1 - alpha / 2) * n) - 1]
    return (lo, hi)


def paired_bootstrap_pvalue(a: List[float], b: List[float],
                            n: int = 2000, seed: int = 0) -> float:
    """Two-sided p-value that mean(a) != mean(b) for paired samples, via
    bootstrap on the per-pair differences. Pure stdlib (no scipy needed).
    For the camera-ready, a paired Wilcoxon (scipy.stats.wilcoxon) is also fine.
    """
    if len(a) != len(b) or not a:
        return 1.0
    diffs = [x - y for x, y in zip(a, b)]
    obs = sum(diffs) / len(diffs)
    if obs == 0:
        return 1.0
    rng = random.Random(seed)
    centered = [d - obs for d in diffs]  # null: mean diff = 0
    m = len(centered)
    count = 0
    for _ in range(n):
        s = sum(centered[rng.randrange(m)] for _ in range(m)) / m
        if abs(s) >= abs(obs):
            count += 1
    return count / n


def severity_weighted_violation_rate(violations: List[bool],
                                     severities: List[str]) -> float:
    """Weighted fraction of severity-weighted exposure that was violated."""
    total = sum(_SEVERITY_WEIGHT.get(s, 1.0) for s in severities)
    if total == 0:
        return 0.0
    hit = sum(_SEVERITY_WEIGHT.get(s, 1.0)
              for v, s in zip(violations, severities) if v)
    return hit / total
