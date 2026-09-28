#!/usr/bin/env python3
"""MaCal-QL unit tests: MC return-to-go, calibrated CQL loss, dual alpha."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from macal.learner import MaCalQLLearner
from macal.returns import compute_mc_returns


def _dummy_batch(
    batch_size: int,
    seq: int,
    n_agents: int,
    obs_dim: int,
    state_dim: int,
    n_actions: int,
    *,
    terminal: bool = True,
    reward_scale: float = 1.0,
):
    device = torch.device("cpu")
    torch.manual_seed(0)
    obs = torch.randn(batch_size, seq, n_agents, obs_dim, device=device)
    actions = torch.randint(0, n_actions, (batch_size, seq, n_agents, 1), device=device)
    state = torch.randn(batch_size, seq, state_dim, device=device)
    avail = torch.ones(batch_size, seq, n_agents, n_actions, dtype=torch.bool, device=device)
    reward = reward_scale * torch.ones(batch_size, seq, 1, device=device)
    term = torch.zeros(batch_size, seq, 1, device=device)
    filled = torch.ones(batch_size, seq, 1, device=device)
    if terminal:
        term[:, -2, 0] = 1.0
    filled[:, -1, 0] = 0.0
    return {
        "obs": obs,
        "actions": actions,
        "state": state,
        "avail_actions": avail,
        "reward": reward,
        "terminated": term,
        "filled": filled,
    }


def _learner(**kwargs) -> MaCalQLLearner:
    torch.manual_seed(0)
    defaults = dict(
        n_agents=3,
        n_actions=5,
        obs_dim=8,
        state_dim=16,
        device="cpu",
        num_ood_actions=10,
    )
    defaults.update(kwargs)
    return MaCalQLLearner(**defaults)


def test_mc_returns_terminal_episode():
    reward = torch.tensor([[[1.0], [2.0], [3.0]]])
    terminated = torch.tensor([[[0.0], [0.0], [1.0]]])
    filled = torch.ones(1, 3, 1)
    mc, valid = compute_mc_returns(reward, terminated, filled, gamma=0.99)
    assert torch.allclose(valid, torch.ones(1, 3, 1))
    assert abs(mc[0, 2, 0].item() - 3.0) < 1e-6
    assert abs(mc[0, 1, 0].item() - (2.0 + 0.99 * 3.0)) < 1e-6
    assert abs(mc[0, 0, 0].item() - (1.0 + 0.99 * (2.0 + 0.99 * 3.0))) < 1e-6


def test_mc_returns_truncated_episode_flagged_invalid():
    # Timeout truncation: slot step at t=3 is terminated but unfilled; no true terminal.
    reward = torch.tensor([[[1.0], [2.0], [3.0], [0.0]]])
    terminated = torch.tensor([[[0.0], [0.0], [0.0], [1.0]]])
    filled = torch.tensor([[[1.0], [1.0], [1.0], [0.0]]])
    mc, valid = compute_mc_returns(reward, terminated, filled, gamma=0.99)
    assert torch.allclose(valid, torch.zeros(1, 4, 1))
    # The recursion stops at the unfilled slot, but the anchor must be dropped.
    assert abs(mc[0, 2, 0].item() - 3.0) < 1e-6


def test_mc_returns_terminal_before_slot():
    reward = torch.tensor([[[0.0], [1.0], [2.0], [0.0]]])
    terminated = torch.tensor([[[0.0], [0.0], [1.0], [1.0]]])
    filled = torch.tensor([[[1.0], [1.0], [1.0], [0.0]]])
    mc, valid = compute_mc_returns(reward, terminated, filled, gamma=0.99)
    assert torch.allclose(valid, torch.ones(1, 4, 1))
    assert abs(mc[0, 2, 0].item() - 2.0) < 1e-6
    assert abs(mc[0, 1, 0].item() - (1.0 + 0.99 * 2.0)) < 1e-6


def test_train_on_batch_finite_and_logs():
    batch = _dummy_batch(4, 6, 3, 8, 16, 5)
    learner = _learner()
    logs = learner.train_on_batch(batch, t_env=0, eps=0.1)
    for key in ("loss", "td_loss", "cql_loss", "cql_diff", "alpha", "mc_return_mean"):
        assert key in logs, key
        assert np.isfinite(logs[key]), (key, logs[key])
    assert logs["alpha"] > 0.0

    obs = {f"agent_{i}": np.random.randn(8).astype(np.float32) for i in range(3)}
    legals = {f"agent_{i}": np.ones(5, dtype=np.float32) for i in range(3)}
    acts = learner.select_actions_eval(obs, legals, eps=0.0)
    assert len(acts) == 3
    q_tot = learner.mix_agent_qs([0.1, -0.2, 0.3], np.zeros(16, dtype=np.float32))
    assert np.isfinite(q_tot)


def test_dual_alpha_moves_toward_gap_constraint():
    batch = _dummy_batch(4, 6, 3, 8, 16, 5)

    learner_low_gap = _learner(target_action_gap=-1e6)
    before = float(learner_low_gap.log_alpha.exp().item())
    learner_low_gap.train_on_batch(batch, t_env=0, eps=0.1)
    after = float(learner_low_gap.log_alpha.exp().item())
    # push-down far above the (negative) gap -> alpha must grow.
    assert after > before

    learner_high_gap = _learner(target_action_gap=1e6)
    before = float(learner_high_gap.log_alpha.exp().item())
    learner_high_gap.train_on_batch(batch, t_env=0, eps=0.1)
    after = float(learner_high_gap.log_alpha.exp().item())
    assert after < before


def test_calibration_clamp_binds_on_high_mc_return():
    batch_zero = _dummy_batch(4, 6, 3, 8, 16, 5, reward_scale=0.0)
    batch_high = _dummy_batch(4, 6, 3, 8, 16, 5, reward_scale=100.0)

    learner_zero = _learner(dual_alpha=False, cql_weight=1.0)
    logs_zero = learner_zero.train_on_batch(batch_zero, t_env=0, eps=0.1)

    learner_high = _learner(dual_alpha=False, cql_weight=1.0)
    logs_high = learner_high.train_on_batch(batch_high, t_env=0, eps=0.1)

    # With G >= 100 the clamp dominates the proposal group: push-down >> dataset Q.
    assert logs_high["cql_loss"] > 50.0
    assert logs_high["cql_loss"] > logs_zero["cql_loss"] + 10.0


def test_fixed_alpha_mode():
    batch = _dummy_batch(4, 6, 3, 8, 16, 5)
    learner = _learner(dual_alpha=False, cql_weight=2.0)
    assert learner.log_alpha is None
    logs = learner.train_on_batch(batch, t_env=0, eps=0.1)
    assert np.isfinite(logs["loss"])
    assert logs["alpha"] == 2.0


if __name__ == "__main__":
    test_mc_returns_terminal_episode()
    test_mc_returns_truncated_episode_flagged_invalid()
    test_mc_returns_terminal_before_slot()
    test_train_on_batch_finite_and_logs()
    test_dual_alpha_moves_toward_gap_constraint()
    test_calibration_clamp_binds_on_high_mc_return()
    test_fixed_alpha_mode()
    print("macal tests ok")
