"""Paper-protocol Q diagnostics: sliding-window tau and signed bias."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.q_metrics import (
    summarize_paper_diagnostics,
    windowed_kendall_taus,
)


def test_windowed_tau_follows_monotone_relation():
    q = np.arange(10, dtype=np.float64)
    g = 5.0 * q + 7.0
    taus = windowed_kendall_taus(q, g, window=5, stride=1)
    assert len(taus) == 6
    assert all(t is not None and abs(t - 1.0) < 1e-9 for t in taus)


def test_windowed_tau_detects_reversal_and_stride():
    g = np.arange(10, dtype=np.float64)
    taus = windowed_kendall_taus(-g, g, window=5, stride=5)
    assert len(taus) == 2
    assert all(t is not None and abs(t + 1.0) < 1e-9 for t in taus)


def test_paper_summary_bias_and_over_under_rates():
    q = np.array([3.0, 3.0, 3.0, 3.0])
    g = np.array([1.0, 2.0, 4.0, 5.0])
    summary = summarize_paper_diagnostics(
        [{"q": q, "reward": g.copy()}],
        gamma=0.0,
        window=4,
    )
    assert abs(summary["bias_mean"] - 0.0) < 1e-9
    assert summary["overestimate_rate"] == 0.5
    assert summary["underestimate_rate"] == 0.5
    assert abs(summary["normalized_diff_literal"] - 0.4625) < 1e-9
    assert summary["window_tau_mean"] is None  # constant Q has no defined rank


def test_paper_summary_window_count_and_cluster_se():
    rewards = [np.arange(10, dtype=np.float64), np.arange(10, dtype=np.float64)]
    q = [np.arange(10, dtype=np.float64), np.arange(10, dtype=np.float64)]
    episodes = [
        {"q": q[i], "reward": rewards[i], "win": i == 0} for i in range(2)
    ]
    summary = summarize_paper_diagnostics(episodes, gamma=0.0, window=10, stride=1)
    assert summary["n_windows"] == 2
    assert abs(summary["window_tau_mean"] - 1.0) < 1e-9
    assert summary["window_tau_episode_se"] == 0.0
    assert summary["win_rate"] == 0.5
