"""Q-trace summaries stay defined when win rate would not show a bad Q."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.q_metrics import (
    _residual_sign_disagree_frac,
    discounted_returns,
    summarize_q_traces,
)


def test_calibrated_q_has_low_bias_and_high_tau():
    rewards = [0.0, 0.0, 1.0, 10.0]
    returns = discounted_returns(rewards, gamma=1.0)
    logs = summarize_q_traces([{"q": returns, "reward": np.asarray(rewards)}], gamma=1.0)
    assert abs(logs["evaluation/q_bias"]) < 1e-8
    assert logs["evaluation/q_kendall_tau"] > 0.9
    assert logs["evaluation/q_rank_defined_frac"] == 1.0


def test_residual_sign_disagree_frac():
    frac = _residual_sign_disagree_frac(
        np.array([2.0, 2.0], dtype=np.float64),
        np.array([0.0, 10.0], dtype=np.float64),
    )
    assert frac == 0.5


def test_flat_q_hides_from_rank_but_shows_up_as_bias():
    rewards = [0.0, 0.0, 0.0, 20.0]
    logs = summarize_q_traces(
        [{"q": np.full(4, 30.0), "reward": np.asarray(rewards)}],
        gamma=0.5,
    )
    assert logs["evaluation/q_bias"] > 0.0
    assert logs["evaluation/q_overestimate_rate"] == 1.0
    assert logs["evaluation/q_rank_defined_frac"] == 0.0
    assert "evaluation/q_kendall_tau" not in logs
    assert logs["evaluation/q_std"] < 1e-8
    assert logs["evaluation/return_to_go_std"] > 1.0
