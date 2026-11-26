#!/usr/bin/env python3
"""Statistics for WEAR cells (standard library only).

A cell maps each task id to its per-rollout values. Tasks are the unit of analysis:
the point estimate is the mean of task means, so every task has equal weight.
  two_stage_bootstrap_ci   percentile interval; resamples tasks, then rollouts within each task
  wilson                   Wilson score interval for k of n tasks
  paired_permutation       paired sign-flip test over tasks, Monte Carlo
  paired_sign_flip_exact   the same test by full enumeration, with an exact p
Library module; no command line.
"""
from __future__ import annotations

import math
import random
from fractions import Fraction
from typing import Mapping, Sequence


def task_means(cell: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Map {task_id: [per-rollout values]} to {task_id: task mean}; tasks without values
    are dropped."""
    return {t: sum(v) / len(v) for t, v in cell.items() if len(v) > 0}


def cell_mean(cell: Mapping[str, Sequence[float]]) -> float:
    """Cell point estimate: the mean of task means. Raises ValueError for an empty cell."""
    tm = task_means(cell)
    if not tm:
        raise ValueError("empty cell")
    return sum(tm.values()) / len(tm)


def two_stage_bootstrap_ci(
    cell: Mapping[str, Sequence[float]],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Two-stage bootstrap over tasks and rollouts; returns (point, lo, hi).

    Each of the n_boot draws resamples tasks with replacement, then rollouts within
    each drawn task, and takes the mean of task means. lo and hi are the alpha/2
    and 1 - alpha/2 quantiles of the draws; the result is fixed by seed.
    """
    tasks = [t for t, v in cell.items() if len(v) > 0]
    if not tasks:
        raise ValueError("empty cell")
    rng = random.Random(seed)
    point = cell_mean(cell)
    stats = []
    for _ in range(n_boot):
        picked = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        means = []
        for t in picked:
            vals = cell[t]
            res = [vals[rng.randrange(len(vals))] for _ in vals]
            means.append(sum(res) / len(res))
        stats.append(sum(means) / len(means))
    stats.sort()
    lo_i = int((alpha / 2) * n_boot)
    hi_i = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return point, stats[lo_i], stats[hi_i]


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval (95% by default) for k of n tasks; returns (lo, hi) clipped to [0, 1]."""
    if not (0 <= k <= n) or n <= 0:
        raise ValueError(f"bad k/n: {k}/{n}")
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def paired_permutation(
    pairs: Mapping[str, tuple[float, float]],
    n_perm: int = 10000,
    seed: int = 0,
) -> tuple[float, float]:
    """Paired sign-flip test with n_perm random sign patterns.

    pairs maps task id to (a, b). Returns (mean of b - a, two-sided p); p uses
    add-one smoothing, so it is never 0.
    """
    diffs = [b_p[1] - b_p[0] for b_p in pairs.values()]
    n = len(diffs)
    if n == 0:
        raise ValueError("no pairs")
    obs = sum(diffs) / n
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_perm):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs) / n
        if abs(s) >= abs(obs) - 1e-12:
            extreme += 1
    return obs, (extreme + 1) / (n_perm + 1)


def paired_sign_flip_exact(
    pairs: Mapping[str, tuple[float, float]],
) -> tuple[float, Fraction]:
    """Exact two-sided paired sign-flip p by enumerating all 2^n sign patterns.

    Same statistic as paired_permutation (mean of b - a); patterns whose |mean|
    ties the observed one count as extreme. Returns (observed_diff, p) with p an
    exact Fraction. Enumeration is limited to n <= 22 pairs.
    """
    diffs = [b - a for a, b in pairs.values()]
    n = len(diffs)
    if n == 0:
        raise ValueError("no pairs")
    if n > 22:
        raise ValueError("too many pairs for exact enumeration")
    obs = abs(sum(diffs))
    sums = [0.0]
    for d in diffs:
        sums = [x + d for x in sums] + [x - d for x in sums]
    extreme = sum(1 for x in sums if abs(x) >= obs - 1e-12)
    return sum(diffs) / n, Fraction(extreme, 2 ** n)
