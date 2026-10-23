# Copyright 2024 InstaDeep Ltd. All rights reserved.

"""Experiment 1 metrics: early-collapse depth, censored recovery, paired tests.

Definitions (design section 2.3):

  Drop                max(0, WR_off - min_{t<=150k} WR_t) / WR_off
  recovery steps      first t after the minimum where 3 consecutive evaluations
                      are >= WR_off; censored when never recovered
  WR@1M               mean of the last 5 evaluations

Paired differences are computed seed-by-seed with a bootstrap confidence
interval.  The paired sign-rank test (Holm-corrected) needs at least 2 seeds;
with a single seed only the descriptive values are reported and the test column
is left empty, which the analysis script states explicitly.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Metric field in the online.jsonl rows: win rate on SMAC, mean episode return
# on maps without a win condition.
METRIC_FIELD = {
    "win_rate": "evaluation/win_rate",
    "return": "evaluation/mean_episode_return",
}


def read_curve(path: str, field: str) -> List[Tuple[int, float]]:
    """(t_env, metric) pairs from one run's online.jsonl, sorted by t_env."""
    rows: List[Tuple[int, float]] = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if field not in payload or "t_env" not in payload:
                continue
            rows.append((int(float(payload["t_env"])), float(payload[field])))
    rows.sort(key=lambda item: item[0])
    return rows


def drop(curve: Sequence[Tuple[int, float]], t_off: int, window: int = 150_000) -> float:
    """Relative depth of the early collapse below the offline starting value."""
    if t_off is None or not curve:
        return float("nan")
    base = curve[0][1]
    if not math.isfinite(base) or base == 0.0:
        return float("nan")
    early = [value for t, value in curve if t <= window]
    if not early:
        return float("nan")
    # A negative baseline (MuJoCo) flips the sign of a relative drop, so the
    # magnitude is normalised by |baseline| instead.
    return max(0.0, (base - min(early)) / abs(base))


def recovery_steps(
    curve: Sequence[Tuple[int, float]], t_off: int, consecutive: int = 3
) -> Optional[int]:
    """First t after the minimum with `consecutive` evaluations at or above baseline.

    Returns None when the run never recovers (a censored observation).
    """
    if t_off is None or len(curve) < 1:
        return None
    base = curve[0][1]
    values = [value for _, value in curve]
    if not values:
        return None
    trough = int(np.argmin(values))
    run = 0
    for idx in range(trough, len(curve)):
        if curve[idx][1] >= base:
            run += 1
            if run >= consecutive:
                return curve[idx - consecutive + 1][0]
        else:
            run = 0
    return None


def tail_mean(curve: Sequence[Tuple[int, float]], n: int = 5) -> float:
    if not curve:
        return float("nan")
    values = [value for _, value in curve[-n:]]
    return float(np.mean(values))


def bootstrap_ci(
    diffs: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired differences."""
    arr = np.asarray([d for d in diffs if math.isfinite(d)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def holm(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        value = (m - rank) * pvalues[i]
        running = max(running, value)
        adjusted[i] = min(1.0, running)
    return adjusted


def paired_sign_rank_p(diffs: Sequence[float]) -> float:
    """One-sided sign-rank p-value for the mean paired difference being > 0.

    Exact enumeration over all 2^n sign assignments is used for n <= 12, the
    normal approximation beyond that.  The smallest p-value this test can return
    with n seeds is 2^-n, so a single seed bottoms out at 0.5 and can never pass
    a Holm-corrected threshold -- which is why the design asks for >= 6 seeds.
    """
    arr = np.asarray([d for d in diffs if math.isfinite(d)], dtype=np.float64)
    arr = arr[arr != 0.0]
    n = arr.size
    if n == 0:
        return 1.0
    ranks = np.abs(arr)
    order = np.argsort(ranks, kind="mergesort")
    sorted_ranks = ranks[order]
    signed = arr[order]
    w_plus = float(np.sum(np.where(signed > 0, sorted_ranks, 0.0)))
    total = float(np.sum(sorted_ranks))
    if total == 0.0:
        return 1.0
    if n <= 12:
        # Exact null distribution of W+ over all 2^n sign assignments.
        counts: Dict[float, int] = {0.0: 1}
        for rank in sorted_ranks:
            nxt: Dict[float, int] = {}
            for key, cnt in counts.items():
                nxt[key] = nxt.get(key, 0) + cnt
                nxt[key + float(rank)] = nxt.get(key + float(rank), 0) + cnt
            counts = nxt
        tail = sum(cnt for key, cnt in counts.items() if key >= w_plus)
        return tail / (2 ** n)
    mean = total / 2.0
    var = math.sqrt(sum(float(r) ** 2 for r in sorted_ranks) / 4.0)
    if var == 0.0:
        return 1.0
    z = (w_plus - mean) / var
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def kaplan_meier_median(times: Sequence[Optional[float]]) -> Optional[float]:
    """Median of possibly censored recovery times (None = never recovered).

    Returns None when more than half the runs are censored, i.e. the median
    recovery time is beyond the observation window.
    """
    events = sorted(t for t in times if t is not None)
    if not events:
        return None
    at_risk = len(times)
    survival = 1.0
    for t in events:
        n_events = sum(1 for e in events if e == t)
        survival *= 1.0 - n_events / at_risk
        if survival <= 0.5:
            return t
        at_risk -= n_events
    return None


@dataclass
class ArmSummary:
    """Per-arm descriptive summary over its seeds."""

    arm: str
    n_seeds: int
    drops: List[float] = field(default_factory=list)
    recoveries: List[Optional[int]] = field(default_factory=list)
    finals: List[float] = field(default_factory=list)
    extra: Dict[str, List[float]] = field(default_factory=dict)

    def median_iqr(self, values: Sequence[float]) -> str:
        arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
        if arr.size == 0:
            return "n/a"
        med = float(np.median(arr))
        if arr.size == 1:
            return f"{med:.3f}"
        q1, q3 = np.quantile(arr, [0.25, 0.75])
        return f"{med:.3f} [{q1:.3f}, {q3:.3f}]"

    def row(self) -> Dict[str, str]:
        out = {
            "arm": self.arm,
            "n_seeds": str(self.n_seeds),
            "drop": self.median_iqr(self.drops),
            "recovery_k": self._recovery_cell(),
            "final": self.median_iqr(self.finals),
        }
        for key, values in self.extra.items():
            out[key] = self.median_iqr(values)
        return out

    def _recovery_cell(self) -> str:
        observed = [r for r in self.recoveries if r is not None]
        censored = len(self.recoveries) - len(observed)
        if not observed:
            return f"censored ({censored}/{len(self.recoveries)})"
        km = kaplan_meier_median(self.recoveries)
        cell = f"{km / 1000:.0f}k" if km is not None else ">budget"
        if censored:
            cell += f" ({censored} censored)"
        return cell
