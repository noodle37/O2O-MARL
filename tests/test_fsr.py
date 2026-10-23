#!/usr/bin/env python3
"""FSR unit tests: flow utilities, support weights, learner offline/online step."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from fsr.flow import flow_matching_loss, log_prob_ode, score_from_velocity
from fsr.learner import FSRLearner
from fsr.networks import teammate_onehot
from fsr.support import mix_fraction, reliance_weight


def _dummy_batch(batch_size, seq, n_agents, obs_dim, state_dim, n_actions):
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


def test_teammate_onehot_drops_self():
    actions = torch.tensor([[[[0], [1], [2]]]])
    out = teammate_onehot(actions, 1, 4)
    assert out.shape == (1, 1, 8)
    assert out[0, 0, 0] == 1 and out[0, 0, 4 + 2] == 1


def test_reliance_weight_rises_with_logp():
    low = reliance_weight(torch.tensor([-5.0]), threshold=0.0)
    high = reliance_weight(torch.tensor([5.0]), threshold=0.0)
    assert float(high) > float(low)
    assert float(mix_fraction(torch.tensor(1.0), "linear", 0.5)) == 0.5
    assert float(mix_fraction(torch.tensor(0.0), "linear", 0.5)) == 0.0


def test_flow_matching_and_logp_finite():
    cond = torch.randn(16, 8)
    x1 = torch.randn(16, 6)

    def vfn(c, x, t):
        return x1[: x.shape[0]] - 0.1 * x

    loss = float(flow_matching_loss(vfn, cond, x1))
    assert np.isfinite(loss)
    score = score_from_velocity(vfn, cond, x1, t_eval=0.9)
    assert score.shape == x1.shape
    logp = log_prob_ode(vfn, cond[:4], x1[:4], steps=3)
    assert logp.shape == (4,)
    assert torch.isfinite(logp).all()


def _learner(**kwargs) -> FSRLearner:
    defaults = dict(
        n_agents=3,
        n_actions=5,
        obs_dim=8,
        state_dim=16,
        device="cpu",
        cql_weight=0.0,
        raw_cql=True,
        shape_critic=True,
        shape_weight=0.05,
        flow_hidden=(32, 32),
        encoder_hidden=16,
        flow_steps=3,
        logp_max_rows=64,
        rotate_explore=True,
    )
    defaults.update(kwargs)
    return FSRLearner(**defaults)


def test_offline_batch_trains_flow_and_critic():
    torch.manual_seed(0)
    learner = _learner()
    batch = _dummy_batch(4, 6, 3, 8, 16, 5)
    logs = learner.train_on_batch(batch, t_env=0)
    assert np.isfinite(logs["loss"])
    assert "fsr/flow_loss" in logs and np.isfinite(logs["fsr/flow_loss"])
    assert logs.get("lambda_memory", 0.0) == 0.0


def test_online_batch_uses_support_not_ovm():
    torch.manual_seed(1)
    learner = _learner()
    batch = _dummy_batch(4, 6, 3, 8, 16, 5)
    learner.train_on_batch(batch, t_env=0)
    learner.offline_q_network = learner.q_network
    learner.offline_mixer = learner.mixer
    learner.freeze_flow()
    learner.support_threshold = 0.0
    logs = learner.train_on_batch(batch, t_env=10, offline_bs=1, support_w_mean=0.4)
    assert "fsr/w_mean" in logs
    assert logs["lambda_memory"] == 0.0
    assert "fsr/prior_strength" in logs
    info = learner.support_mix_ratio(batch)
    assert 0.0 <= info["fsr/offline_frac"] <= 0.5


def test_rotation_picks_one_explorer():
    learner = _learner()
    learner.offline_q_network = learner.q_network
    learner.freeze_flow()
    explorers = []
    for _ in range(6):
        state = learner.new_episode_state()
        explorers.append(state["explorer"])
        obs = {f"agent_{i}": np.random.randn(8).astype(np.float32) for i in range(3)}
        legal = {f"agent_{i}": np.ones(5, dtype=np.float32) for i in range(3)}
        acts = learner.select_actions_eval(obs, legal, eps=0.0, state=state)
        assert len(acts) == 3
    assert set(explorers) == {0, 1, 2}


def test_eval_does_not_rotate():
    learner = _learner()
    learner.offline_q_network = learner.q_network
    learner.freeze_flow()
    obs = {f"agent_{i}": np.random.randn(8).astype(np.float32) for i in range(3)}
    legal = {f"agent_{i}": np.ones(5, dtype=np.float32) for i in range(3)}
    acts = learner.select_actions_eval(obs, legal, eps=0.0)
    assert len(acts) == 3


if __name__ == "__main__":
    test_teammate_onehot_drops_self()
    test_reliance_weight_rises_with_logp()
    test_flow_matching_and_logp_finite()
    test_offline_batch_trains_flow_and_critic()
    test_online_batch_uses_support_not_ovm()
    test_rotation_picks_one_explorer()
    test_eval_does_not_rotate()
    print("fsr tests ok")
