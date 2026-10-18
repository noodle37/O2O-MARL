#!/usr/bin/env python3
"""OVM mechanism diagnostics: loss invariance and metric sanity."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ovmse.diagnostics import ovm_fixed_sample_diag, ovm_target_stats
from ovmse.learner import PaperQMIXLearner


def _dummy_batch(batch_size, seq, n_agents, obs_dim, state_dim, n_actions, seed=0):
    g = torch.Generator().manual_seed(seed)
    device = torch.device("cpu")
    obs = torch.randn(batch_size, seq, n_agents, obs_dim, generator=g, device=device)
    actions = torch.randint(0, n_actions, (batch_size, seq, n_agents, 1), generator=g)
    state = torch.randn(batch_size, seq, state_dim, generator=g)
    avail = torch.ones(batch_size, seq, n_agents, n_actions, dtype=torch.bool)
    reward = torch.randn(batch_size, seq, 1, generator=g)
    term = torch.zeros(batch_size, seq, 1)
    term[:, -1, 0] = 1.0
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


def _learner(**kwargs):
    torch.manual_seed(0)
    learner = PaperQMIXLearner(
        n_agents=3,
        n_actions=5,
        obs_dim=8,
        state_dim=16,
        device="cpu",
        cql_weight=0.0,
        raw_cql=False,
        **kwargs,
    )
    # Stand in for the frozen offline nets (same shapes, frozen).
    learner.offline_q_network = copy.deepcopy(learner.q_network)
    learner.offline_mixer = copy.deepcopy(learner.mixer)
    for p in learner.offline_q_network.parameters():
        p.requires_grad_(False)
    for p in learner.offline_mixer.parameters():
        p.requires_grad_(False)
    return learner


def test_diagnostics_do_not_change_the_loss():
    """The diagnostic path must be a pure observer of the same update."""
    batch = _dummy_batch(4, 6, 3, 8, 16, 5)
    runs = []
    for stats in (False, True):
        learner = _learner(use_ovm=True, ovm_diag_train_stats=stats, ovm_diag_stride=1)
        logs = learner.train_on_batch(batch, use_ovm=True, t_env=1000)
        runs.append((logs["loss"], copy.deepcopy(learner.q_network.state_dict()), logs))
    assert runs[0][0] == runs[1][0], (runs[0][0], runs[1][0])
    for key in runs[0][1]:
        assert torch.equal(runs[0][1][key], runs[1][1][key]), key
    assert any(k.startswith("train_ovm/") for k in runs[1][2])
    assert not any(k.startswith("train_ovm/") for k in runs[0][2])


def test_ovm_target_stats_activation_and_lift():
    torch.manual_seed(1)
    off = torch.tensor([[[5.0], [1.0], [3.0]]])
    td = torch.tensor([[[2.0], [4.0], [3.0]]])
    mask = torch.ones_like(off)
    # off > td at index 0 (5 > 2) and index 2 (3 > 3 is False), so 1 of 3 active.
    stats = ovm_target_stats(off, td, mask, lam=0.5, prefix="p/")
    assert abs(stats["p/activation_rate"] - 1.0 / 3.0) < 1e-6
    assert abs(stats["p/lift_raw_mean"] - 3.0 / 3.0) < 1e-6
    assert abs(stats["p/lift_active_mean"] - 3.0) < 1e-6
    assert abs(stats["p/lift_effective_mean"] - 0.5 * 3.0 / 3.0) < 1e-6
    assert abs(stats["p/lambda_memory"] - 0.5) < 1e-9
    assert abs(stats["p/ovm_target_mean"] - (5.0 + 4.0 + 3.0) / 3.0) < 1e-6


def test_fixed_sample_diag_runs_for_both_arms():
    batch = _dummy_batch(3, 5, 3, 8, 16, 5, seed=7)
    for use_ovm in (True, False):
        learner = _learner(use_ovm=use_ovm)
        diag = ovm_fixed_sample_diag(learner, batch, t_env=500)
        assert diag["ovm_diag/ovm_applied"] == float(use_ovm)
        for key in (
            "ovm_diag/off_q_mean",
            "ovm_diag/td_target_mean",
            "ovm_diag/ovm_target_mean",
            "ovm_diag/activation_rate",
            "ovm_diag/lift_raw_mean",
            "ovm_diag/lambda_memory",
            "ovm_diag/online_q_mean",
            "ovm_diag/online_minus_offline_q",
            "ovm_diag/greedy_agent_agree_rate",
        ):
            assert key in diag, (use_ovm, key)
            assert np.isfinite(diag[key]), (use_ovm, key, diag[key])
        assert 0.0 <= diag["ovm_diag/activation_rate"] <= 1.0
        assert diag["ovm_diag/n_transitions"] > 0
        # The OVM target can never sit below the plain TD target.
        assert diag["ovm_diag/lift_raw_mean"] >= -1e-9


if __name__ == "__main__":
    test_diagnostics_do_not_change_the_loss()
    test_ovm_target_stats_activation_and_lift()
    test_fixed_sample_diag_runs_for_both_arms()
    print("ovm diagnostics ok")
