#!/usr/bin/env python3
"""CPU smoke test: one train_on_batch step and action selection."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ovmse.learner import PaperQMIXLearner


def _dummy_batch(batch_size: int, seq: int, n_agents: int, obs_dim: int, state_dim: int, n_actions: int):
    device = torch.device("cpu")
    obs = torch.randn(batch_size, seq, n_agents, obs_dim, device=device)
    actions = torch.randint(0, n_actions, (batch_size, seq, n_agents, 1), device=device)
    state = torch.randn(batch_size, seq, state_dim, device=device)
    avail = torch.ones(batch_size, seq, n_agents, n_actions, dtype=torch.bool, device=device)
    reward = torch.randn(batch_size, seq, 1, device=device)
    term = torch.zeros(batch_size, seq, 1, device=device)
    filled = torch.ones(batch_size, seq, 1, device=device)
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


def test_train_on_batch_and_policy():
    torch.manual_seed(0)
    n_agents, n_actions, obs_dim, state_dim = 3, 5, 8, 16
    learner = PaperQMIXLearner(
        n_agents=n_agents,
        n_actions=n_actions,
        obs_dim=obs_dim,
        state_dim=state_dim,
        device="cpu",
        cql_weight=0.05,
        raw_cql=False,
    )
    batch = _dummy_batch(4, 6, n_agents, obs_dim, state_dim, n_actions)
    logs = learner.train_on_batch(batch, t_env=0)
    assert "loss" in logs and np.isfinite(logs["loss"])

    obs = {f"agent_{i}": np.random.randn(obs_dim).astype(np.float32) for i in range(n_agents)}
    legals = {f"agent_{i}": np.ones(n_actions, dtype=np.float32) for i in range(n_agents)}
    acts = learner.select_actions_eval(obs, legals, eps=0.0)
    assert len(acts) == n_agents


if __name__ == "__main__":
    test_train_on_batch_and_policy()
    print("smoke ok")
