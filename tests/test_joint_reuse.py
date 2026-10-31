#!/usr/bin/env python3

"""Unit checks for the per-agent retention rollout policy."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from joint_reuse.adaptive import AdaptiveLocalReusePolicy
from joint_reuse.policy import HybridRetentionPolicy
from joint_reuse.runner import parse_offline_agents

N_AGENTS = 2
N_ACTIONS = 3
OBS_DIM = 4
AGENTS = [f"agent_{i}" for i in range(N_AGENTS)]


class ConstantNet(torch.nn.Module):
    def __init__(self, q):
        super().__init__()
        self.register_buffer("q", torch.tensor(q, dtype=torch.float32))
        self.last_inputs = []

    def initial_state(self, batch_size, device):
        return None

    def forward(self, obs_in, hidden):
        self.last_inputs.append(int(obs_in[0, OBS_DIM:].argmax().item()))
        return self.q.unsqueeze(0), hidden


class FakeLearner:
    def __init__(self, offline_q, online_q, sequential=True):
        self.n_agents = N_AGENTS
        self.n_actions = N_ACTIONS
        self.device = torch.device("cpu")
        self.add_agent_id_to_obs = False
        self.sequential_explore = sequential
        self.offline_q_network = ConstantNet(offline_q)
        self.q_network = ConstantNet(online_q)

    def new_episode_state(self):
        return {
            "rnn": {
                a: self.q_network.initial_state(1, self.device)
                for a in AGENTS
            },
            "last": {},
        }


def obs():
    return {a: np.zeros(OBS_DIM, dtype=np.float32) for a in AGENTS}


def legals():
    return {a: np.ones(N_ACTIONS, dtype=np.float32) for a in AGENTS}


def test_per_agent_source_and_asymmetric_histories():
    learner = FakeLearner([3.0, 0.0, 0.0], [0.0, 4.0, 0.0])
    policy = HybridRetentionPolicy(learner, (0,), rng=np.random.default_rng(1))
    state = policy.new_episode_state()
    assert "offline" in state

    trace = {}
    first = policy.select_actions_eval(obs(), legals(), 0.0, state, trace)
    assert first == {"agent_0": 0, "agent_1": 1}
    assert trace["retention_mask"] == [True, False]
    assert state["last"] == {"agent_0": 0, "agent_1": 1}
    assert state["offline"]["last"] == {"agent_0": 0}

    policy.select_actions_eval(obs(), legals(), 0.0, state, None)
    # Offline agent history follows its own greedy action; the online net sees the
    # executed history for both agents.
    assert learner.offline_q_network.last_inputs[-1] == 0
    assert learner.q_network.last_inputs[-2:] == [0, 1]
    assert state["offline"]["last"] == {"agent_0": 0}

    stats = policy.stats(reset=True)
    assert stats["hybrid/steps"] == 4.0
    assert stats["hybrid/offline_agent_steps"] == 2.0
    assert abs(stats["hybrid/offline_agent_share"] - 0.5) < 1e-9


def test_no_retention_does_not_use_or_store_offline_policy():
    learner = FakeLearner([3.0, 0.0, 0.0], [0.0, 4.0, 0.0])
    policy = HybridRetentionPolicy(learner, (), rng=np.random.default_rng(2))
    state = policy.new_episode_state()
    assert "offline" not in state
    actions = policy.select_actions_eval(obs(), legals(), 0.0, state, None)
    assert actions == {"agent_0": 1, "agent_1": 1}
    assert len(learner.offline_q_network.last_inputs) == 0
    assert policy.stats()["hybrid/offline_agent_share"] == 0.0


def test_sequential_exploration_changes_exactly_one_agent():
    learner = FakeLearner([3.0, 0.0, 0.0], [0.0, 4.0, 0.0], sequential=True)
    policy = HybridRetentionPolicy(learner, (0,), rng=np.random.default_rng(3))
    state = policy.new_episode_state()
    actions = policy.select_actions_eval(obs(), legals(), 1.0, state, None)
    assert all(0 <= a < N_ACTIONS for a in actions.values())
    assert sum(state["explored"]) == 1
    assert state["last"] == actions
    # The online history follows the executed action; the offline history follows
    # the offline candidate even when another agent explored.
    assert state["offline"]["last"]["agent_0"] == 0


def test_parse_offline_agents():
    assert parse_offline_agents("") == ()
    assert parse_offline_agents("0, 4") == (0, 4)
    try:
        HybridRetentionPolicy(
            FakeLearner([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), (2,)
        )
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range offline agent must be rejected")


def _adaptive_policy(tmp_path, weight, offline_q=None):
    import pickle

    class FakeMixer(torch.nn.Module):
        def forward(self, qs, state):
            return qs.sum(dim=-1)

    learner = FakeLearner(
        [3.0, 0.0, 0.0] if offline_q is None else offline_q,
        [0.0, 4.0, 0.0],
    )
    learner.mixer = FakeMixer()
    learner.offline_mixer = FakeMixer()
    path = tmp_path / "adaptive.pkl"
    artifact = {
        "treatment_agents": (0,),
        "feature_names": (),
        "layout": {"n_allies": 2, "nf_ally": 1, "n_enemies": 0, "nf_enemy": 1},
        "ensemble": (),
        "episode_limit": 70,
    }
    with path.open("wb") as handle:
        pickle.dump(artifact, handle)
    policy = AdaptiveLocalReusePolicy(
        learner,
        str(path),
        rng=np.random.default_rng(7),
        weight_fn=lambda features: weight,
    )
    return learner, policy


def test_adaptive_weight_controls_only_treatment_agent(tmp_path):
    env_state = np.ones(2, dtype=np.float32)
    for weight, expected in ((0.0, {"agent_0": 1, "agent_1": 1}), (1.0, {"agent_0": 0, "agent_1": 1})):
        learner, policy = _adaptive_policy(tmp_path, weight)
        state = policy.new_episode_state()
        trace = {}
        actions = policy.select_actions_eval(obs(), legals(), 0.0, state, trace, env_state)
        assert actions == expected
        assert state["online"]["last"] == actions
        assert state["offline"]["last"] == actions
        assert trace["adaptive_weight"] == weight
        stats = policy.stats()
        assert stats["adaptive/conflict_share"] == 1.0
        assert stats["adaptive/intervention_share"] == weight


def test_adaptive_no_conflict_stays_online(tmp_path):
    learner, policy = _adaptive_policy(tmp_path, 1.0, offline_q=[0.0, 4.0, 0.0])
    state = policy.new_episode_state()
    trace = {}
    actions = policy.select_actions_eval(
        obs(), legals(), 0.0, state, trace, np.ones(2, dtype=np.float32)
    )
    assert actions == {"agent_0": 1, "agent_1": 1}
    assert trace["adaptive_weight"] == 0.0
    assert policy.stats()["adaptive/conflict_share"] == 0.0


if __name__ == "__main__":
    test_per_agent_source_and_asymmetric_histories()
    test_no_retention_does_not_use_or_store_offline_policy()
    test_sequential_exploration_changes_exactly_one_agent()
    test_parse_offline_agents()
    print("joint_reuse tests ok")
