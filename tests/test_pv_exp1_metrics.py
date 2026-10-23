# Copyright 2024 InstaDeep Ltd. All rights reserved.

"""Unit tests for the experiment 1 metrics and statistics."""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pv.exp1_metrics import (  # noqa: E402
    ArmSummary,
    bootstrap_ci,
    drop,
    holm,
    kaplan_meier_median,
    paired_sign_rank_p,
    read_curve,
    recovery_steps,
    tail_mean,
)


def _write_curve(path, rows):
    import json

    with open(path, "w", encoding="utf-8") as handle:
        for t, value in rows:
            handle.write(json.dumps({"t_env": float(t), "evaluation/win_rate": value}) + "\n")
    return path


def test_drop_uses_absolute_baseline(tmp_path=None):
    path = os.path.join(tmp_path or "/tmp", "pv_curve.jsonl")
    # Baseline 0.5, trough 0.25 -> Drop = 0.5.
    _write_curve(path, [(0, 0.5), (25_000, 0.4), (50_000, 0.25), (100_000, 0.3)])
    curve = read_curve(path, "evaluation/win_rate")
    assert curve[0][1] == 0.5
    assert abs(drop(curve, curve[0][1]) - 0.5) < 1e-9


def test_drop_negative_baseline_is_sign_stable(tmp_path=None):
    """A negative baseline (MuJoCo) must not flip the sign of the drop."""
    path = os.path.join(tmp_path or "/tmp", "pv_curve_neg.jsonl")
    _write_curve(path, [(0, -100.0), (25_000, -200.0), (50_000, -150.0)])
    curve = read_curve(path, "evaluation/win_rate")
    value = drop(curve, curve[0][1])
    assert value >= 0.0
    assert abs(value - 1.0) < 1e-9


def test_recovery_requires_three_consecutive(tmp_path=None):
    path = os.path.join(tmp_path or "/tmp", "pv_curve_rec.jsonl")
    _write_curve(
        path,
        [
            (0, 0.5),
            (25_000, 0.2),   # trough
            (50_000, 0.4),   # not recovered
            (75_000, 0.55),
            (100_000, 0.6),
            (125_000, 0.62),  # third consecutive >= baseline
        ],
    )
    curve = read_curve(path, "evaluation/win_rate")
    assert recovery_steps(curve, curve[0][1]) == 75_000


def test_recovery_censored_when_never(tmp_path=None):
    path = os.path.join(tmp_path or "/tmp", "pv_curve_cens.jsonl")
    _write_curve(path, [(0, 0.5), (25_000, 0.2), (50_000, 0.3), (75_000, 0.45)])
    curve = read_curve(path, "evaluation/win_rate")
    assert recovery_steps(curve, curve[0][1]) is None


def test_tail_mean_of_last_five():
    curve = [(i * 1000, float(i)) for i in range(10)]
    assert tail_mean(curve, 5) == float(np.mean([5, 6, 7, 8, 9]))


def test_holm_monotone_and_bounded():
    raw = [0.01, 0.04, 0.03, 0.20]
    adjusted = holm(raw)
    assert len(adjusted) == len(raw)
    assert all(0.0 <= p <= 1.0 for p in adjusted)
    # The smallest raw p-value is multiplied by the largest factor.
    assert abs(adjusted[0] - 4 * 0.01) < 1e-12
    # Adjusted p-values are non-decreasing in the rank order of the raw values.
    order = sorted(range(len(raw)), key=lambda i: raw[i])
    assert all(
        adjusted[order[i]] <= adjusted[order[i + 1]] + 1e-12 for i in range(len(order) - 1)
    )


def test_sign_rank_single_seed_cannot_reach_significance():
    """One seed bottoms out at p = 0.5, so a Holm threshold is unreachable."""
    assert paired_sign_rank_p([1.0]) == 0.5
    # A single negative difference is evidence against the one-sided hypothesis.
    assert paired_sign_rank_p([-1.0]) == 1.0
    assert paired_sign_rank_p([0.0]) == 1.0
    # One positive and one negative difference of equal magnitude: W+ takes either
    # value with equal probability, so the one-sided tail is 3/4.
    assert paired_sign_rank_p([1.0, -1.0]) == 0.75


def test_sign_rank_consistent_positive_is_significant():
    p = paired_sign_rank_p([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert p < 0.05
    # With n = 6 the smallest attainable exact p-value is 1/64.
    assert abs(p - 1.0 / 64.0) < 1e-12


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    diffs = list(rng.normal(0.5, 0.1, size=12))
    lo, hi = bootstrap_ci(diffs, n_boot=2000, seed=0)
    assert lo < np.mean(diffs) < hi


def test_kaplan_meier_median_with_censoring():
    # Half the runs recover at 10k, the rest never do.
    times = [10_000, 10_000, None, None]
    assert kaplan_meier_median(times) == 10_000
    # A majority censored -> the median is beyond the window.
    assert kaplan_meier_median([10_000, None, None, None]) is None
    assert kaplan_meier_median([None, None]) is None


def test_arm_summary_cells():
    summary = ArmSummary(arm="A", n_seeds=3)
    summary.drops.extend([0.1, 0.2, 0.3])
    summary.recoveries.extend([10_000, 20_000, None])
    summary.finals.extend([0.4, 0.5, 0.6])
    row = summary.row()
    assert row["arm"] == "A"
    assert row["n_seeds"] == "3"
    assert "0.200" in row["drop"]
    assert "0.500" in row["final"]
    assert "1 censored" in row["recovery_k"]
