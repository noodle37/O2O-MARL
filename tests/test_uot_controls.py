#!/usr/bin/env python3
"""Unit tests for the simple UOT controls on 5m_vs_6m."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from uot.learner import UOTLearner
from uot.neighborhood import build_recorded_neighborhood
from uot.support import RecordedJoints, StateSupport


def _batch(batch_size=4, seq=6, n_agents=3, obs_dim=8, state_dim=16, n_actions=4):
    obs = torch.randn(batch_size, seq, n_agents, obs_dim)
    actions = torch.randint(0, n_actions, (batch_size, seq, n_agents, 1))
    state = torch.randn(batch_size, seq, state_dim)
    avail = torch.ones(batch_size, seq, n_agents, n_actions, dtype=torch.bool)
    reward = torch.randn(batch_size, seq, 1)
    term = torch.zeros(batch_size, seq, 1)
    filled = torch.ones(batch_size, seq, 1)
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


def _support(n_agents=3, n_actions=4):
    # Four recorded joint tuples over actions {0,1}; action 0/1 is marginally
    # seen for every agent, but [0,0,0] was never recorded jointly, so the joint
    # and marginal supports differ.
    joints = np.array(
        [[0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 0]], dtype=np.int64
    )[:n_agents]
    codes = joints @ (n_actions ** np.arange(n_agents, dtype=np.int64))
    marginal = np.zeros((n_agents, n_actions), dtype=bool)
    marginal[:, :2] = True
    return RecordedJoints(n_agents, n_actions, codes, marginal_mask=marginal)


def _learner(**kwargs):
    defaults = dict(
        n_agents=3,
        n_actions=4,
        obs_dim=8,
        state_dim=16,
        support=_support(),
        eta=0.1,
        delta=0.1,
        margin=0.1,
        n_candidates=4,
        max_hamming=1,
        device="cpu",
    )
    defaults.update(kwargs)
    learner = UOTLearner(**defaults)
    learner.freeze_oracle()
    return learner


def test_candidate_support_modes():
    torch.manual_seed(0)
    support = _support()
    q = torch.randn(2, 5, 3, 4)
    avail = torch.ones(2, 5, 3, 4, dtype=torch.bool)
    executed = torch.zeros(2, 5, 3, dtype=torch.long)
    for mode in ("joint", "marginal", "state", "none"):
        cand, valid = build_recorded_neighborhood(
            q, q, avail, executed, support, k=4, max_hamming=1, candidate_support=mode
        )
        assert cand.shape == (4, 2, 5, 3)
        assert valid.dtype == torch.bool
        if mode == "joint":
            assert support.contains(cand[valid]).all()
        if mode == "marginal":
            assert support.contains_marginal(cand[valid]).all()
    # Marginal support admits combinations that were never recorded jointly.
    near = torch.tensor([[[0, 0, 0]]])
    far = torch.tensor([[[1, 1, 1]]])
    assert not bool(support.contains(near).item())
    assert bool(support.contains_marginal(near).item())
    assert bool(support.contains(far).item())
    unseen = torch.tensor([[[2, 2, 2]]])
    assert not bool(support.contains_marginal(unseen).item())


def test_agent_ranking_loss_runs():
    torch.manual_seed(1)
    learner = _learner(ranking_scope="agent")
    batch = _batch()
    logs = learner.train_on_batch(batch, t_env=0)
    assert np.isfinite(logs["loss"])
    assert logs["pad_n_pairs"] > 0
    assert 0.0 <= logs["pad_hinge_rate"] <= 1.0
    assert logs["uot_ranking_scope_agent"] == 1.0


def test_state_support_weights_and_training():
    torch.manual_seed(2)
    states = torch.randn(32, 5, 16) * 0.1
    filled = torch.ones(32, 5, 1)
    ss = StateSupport(16, n_proj=4, n_bins=4, table_size=1024)
    ss.fit(states, filled)
    stats = ss.calibrate(0.1)
    assert np.isfinite(stats["uot/state_threshold"])
    w = ss.weights(torch.randn(3, 7, 16) * 0.1)
    assert w.shape == (3, 7)
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0
    learner = _learner(support_mode="state", state_support=ss)
    logs = learner.train_on_batch(_batch(), t_env=0)
    assert np.isfinite(logs["loss"])
    assert "uot_state_support_mean" in logs


def test_each_support_mode_trains():
    torch.manual_seed(3)
    ss = StateSupport(16, n_proj=4, n_bins=4, table_size=1024)
    ss.fit(torch.randn(16, 5, 16), torch.ones(16, 5, 1))
    ss.calibrate(0.1)
    for mode in ("joint", "marginal", "state", "none"):
        learner = _learner(support_mode=mode, state_support=ss if mode == "state" else None)
        logs = learner.train_on_batch(_batch(), t_env=0)
        assert np.isfinite(logs["loss"]), mode
        assert np.isfinite(logs["loss_ord"]), mode


def test_eta_decay_schedule():
    learner = _learner(eta=0.1, eta_decay_start=0, eta_decay_steps=1000)
    assert abs(learner.eta_at(0) - 0.1) < 1e-9
    assert abs(learner.eta_at(500) - 0.05) < 1e-9
    assert learner.eta_at(1000) == 0.0
    assert learner.eta_at(2000) == 0.0


if __name__ == "__main__":
    test_candidate_support_modes()
    test_agent_ranking_loss_runs()
    test_state_support_weights_and_training()
    test_each_support_mode_trains()
    test_eta_decay_schedule()
    print("uot control tests ok")
