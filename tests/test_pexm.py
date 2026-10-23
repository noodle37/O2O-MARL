#!/usr/bin/env python3
"""Unit tests for the PEX-M selector, action selection, and arm bookkeeping."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from training.pexm import (
    ThompsonReturnSelector,
    _prefix,
    collect_pexm_round,
    select_actions_with_net,
)

N_AGENTS = 2
N_ACTIONS = 3
AGENTS = ["agent_0", "agent_1"]


class ConstantNet(torch.nn.Module):
    def __init__(self, q):
        super().__init__()
        self.register_buffer("q", torch.tensor(q, dtype=torch.float32))

    def initial_state(self, batch_size, device):
        return None

    def forward(self, obs_in, hidden):
        return self.q.unsqueeze(0), hidden


class FakeLearner:
    def __init__(self, q):
        self.n_agents = N_AGENTS
        self.n_actions = N_ACTIONS
        self.device = torch.device("cpu")
        self.add_agent_id_to_obs = False
        self.q_network = ConstantNet(q)


class FakeEnv:
    def __init__(self, horizon=2, seed=0):
        self.horizon = horizon
        self.t = 0
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.t = 0
        return self._obs(), self._info()

    def _obs(self):
        return {agent: np.zeros(4, dtype=np.float32) for agent in AGENTS}

    def _info(self):
        return {
            "state": np.zeros(3, dtype=np.float32),
            "legals": {agent: np.ones(N_ACTIONS, dtype=bool) for agent in AGENTS},
        }

    def step(self, actions):
        assert set(actions) == set(AGENTS)
        self.t += 1
        terminal = {agent: self.t >= self.horizon for agent in AGENTS}
        trunc = {agent: False for agent in AGENTS}
        rewards = {agent: 1.0 for agent in AGENTS}
        info = self._info()
        info["battle_won"] = self.t >= self.horizon
        return self._obs(), rewards, terminal, trunc, info


def _fit_selector(offline=10.0, online=0.0, n=100, min_online=0.0):
    sel = ThompsonReturnSelector(min_online=min_online)
    for _ in range(n):
        sel.observe("offline", offline)
        sel.observe("online", online)
    return sel


def test_selector_prior_and_posterior():
    sel = ThompsonReturnSelector(min_online=0.25, mu0=1.0, kappa0=2.0, alpha0=2.0, beta0=9.0)
    p = sel.params("offline")
    assert p["n"] == 0
    assert p["mu"] == 1.0
    assert p["kappa"] == 2.0
    assert p["alpha"] == 2.0
    assert p["beta"] == 9.0
    for value in (5.0, 7.0, 9.0):
        sel.observe("offline", value)
    p = sel.params("offline")
    assert p["n"] == 3
    assert abs(p["mean"] - 7.0) < 1e-12
    assert p["kappa"] == 5.0
    assert abs(p["alpha"] - 3.5) < 1e-12
    assert p["sd"] > 0.0


def test_selector_min_online_floor_and_roundtrip():
    sel = ThompsonReturnSelector(min_online=1.0)
    for _ in range(10):
        arm, forced = sel.choose(np.random.default_rng(0))
        assert arm == "online" and forced
    assert sel.forced_online == 10
    payload = sel.to_dict()
    clone = ThompsonReturnSelector.from_dict(payload)
    assert clone.stats()["params"] == sel.stats()["params"]
    assert clone.counts == sel.counts


def test_selector_prefers_high_return_arm():
    sel = _fit_selector(offline=0.0, online=20.0)
    rng = np.random.default_rng(1)
    picks = [sel.choose(rng)[0] for _ in range(200)]
    assert picks.count("online") > 190
    assert sel.stats()["online_share"] > 0.9


def test_select_actions_masks_illegal_and_explores_legal_only():
    learner = FakeLearner([0.0, 100.0, 1.0])
    state = {"rnn": {"agent_0": None, "agent_1": None}, "last": {}}
    legals = {
        "agent_0": np.array([True, False, True]),
        "agent_1": np.array([True, True, True]),
    }
    obs = {agent: np.zeros(4, dtype=np.float32) for agent in AGENTS}
    actions = select_actions_with_net(
        learner.q_network, learner, obs, legals, 0.0, state, np.random.default_rng(0)
    )
    assert actions["agent_0"] == 2  # action 1 is illegal
    assert actions["agent_1"] == 1
    assert state["last"] == actions

    rng = np.random.default_rng(2)
    for _ in range(20):
        state = {"rnn": {"agent_0": None, "agent_1": None}, "last": {}}
        actions = select_actions_with_net(
            learner.q_network, learner, obs, legals, 1.0, state, rng
        )
        assert actions["agent_0"] in (0, 2)


def test_collect_round_bookkeeping_and_returns():
    envs = [FakeEnv(horizon=2, seed=0), FakeEnv(horizon=3, seed=1)]
    online = FakeLearner([0.0, 0.0, 1.0])
    offline = FakeLearner([1.0, 0.0, 0.0])
    selector = _fit_selector(offline=10.0, online=0.0, n=100, min_online=0.0)
    episodes = collect_pexm_round(
        envs, online, offline, selector, eps=0.0, act_rng=np.random.default_rng(3)
    )
    assert len(episodes) == 2
    assert all(ep["arm"] == "offline" and not ep["forced"] for ep in episodes)
    assert [ep["length"] for ep in episodes] == [2, 3]
    assert [ep["return"] for ep in episodes] == [2.0, 3.0]
    assert all(ep["won"] for ep in episodes)
    assert all(ep["steps"][0]["actions"].tolist() == [0, 0] for ep in episodes)
    assert selector.counts["offline"] == 2


def test_collect_round_forced_online():
    envs = [FakeEnv(horizon=2, seed=0)]
    online = FakeLearner([0.0, 0.0, 1.0])
    offline = FakeLearner([1.0, 0.0, 0.0])
    selector = ThompsonReturnSelector(min_online=1.0)
    episodes = collect_pexm_round(
        envs, online, offline, selector, eps=0.0, act_rng=np.random.default_rng(4)
    )
    assert episodes[0]["arm"] == "online" and episodes[0]["forced"]
    assert episodes[0]["steps"][0]["actions"].tolist() == [2, 2]


def test_prefix_keeps_eval_keys():
    logs = {"evaluation/win_rate": 0.5, "other": 1.0, "evaluation/x": 2.0}
    assert _prefix("eval_online/", logs) == {
        "eval_online/win_rate": 0.5,
        "eval_online/x": 2.0,
    }
