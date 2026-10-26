#!/usr/bin/env python3
# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Unit tests for the joint behaviour proposer and the paired evaluation arms.

Everything runs on CPU against constant-Q fake networks and a fake environment,
so the tests pin the protocol (0.5/0.5 selection, shared evaluation seeds, no
behaviour propensity, no training during the arms) rather than any learning
outcome.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from fhjp.controller import ProposerController
from fhjp.eval_arms import (
    arm_return,
    one_sided_upper_bound,
    paired_arm_estimate,
    should_exit,
)
from fhjp.proposer import JointProposalPolicy

N_AGENTS = 2
N_ACTIONS = 3
OBS_DIM = 4
AGENTS = [f"agent_{i}" for i in range(N_AGENTS)]
GAMMA = 0.99


class ConstantNet(torch.nn.Module):
    """Q network that always returns the same per-agent action values."""

    def __init__(self, q):
        super().__init__()
        self.register_buffer("q", torch.tensor(q, dtype=torch.float32))
        self.seen_last_actions: list[int] = []

    def initial_state(self, batch_size, device):
        return None

    def forward(self, obs_in, hidden):
        # obs_in is [obs, last action one-hot, (agent id)]; record what history
        # this net was conditioned on.
        self.seen_last_actions.append(int(obs_in[0, OBS_DIM:].argmax().item()))
        return self.q.unsqueeze(0), hidden


class RecordingEnv:
    """Deterministic env whose reward depends on the joint action."""

    def __init__(self, seed=0, horizon=3, reward_fn=None):
        self.seed = int(seed)
        self.horizon = int(horizon)
        self.reward_fn = reward_fn or (lambda actions, t: 0.0)
        self.t = 0
        self.resets = 0
        self.closed = False

    def reset(self):
        self.resets += 1
        self.t = 0
        return self._obs(), self._info()

    def _obs(self):
        return {a: np.zeros(OBS_DIM, dtype=np.float32) for a in AGENTS}

    def _info(self):
        return {
            "legals": {a: np.ones(N_ACTIONS, dtype=np.float32) for a in AGENTS},
            "state": np.zeros(8, dtype=np.float32),
        }

    def step(self, actions):
        reward = float(self.reward_fn(actions, self.t))
        self.t += 1
        terminal = self.t >= self.horizon
        info = self._info()
        return (
            self._obs(),
            {a: np.array(reward, dtype=np.float32) for a in AGENTS},
            {a: np.array(terminal) for a in AGENTS},
            {a: np.array(False) for a in AGENTS},
            info,
        )

    def close(self):
        self.closed = True


class FakeLearner:
    """Minimal learner surface the proposer and the arms need."""

    def __init__(self, offline_q, online_q, gamma=GAMMA):
        self.n_agents = N_AGENTS
        self.n_actions = N_ACTIONS
        self.device = torch.device("cpu")
        self.add_agent_id_to_obs = False
        self.sequential_explore = False
        self.gamma = gamma
        self.q_network = ConstantNet(online_q)
        self.offline_q_network = ConstantNet(offline_q)
        self.train_calls = 0

    def new_episode_state(self):
        return {
            "rnn": {a: self.q_network.initial_state(1, self.device) for a in AGENTS},
            "last": {},
        }

    def select_actions_eval(self, observations, legals, eps, state=None, trace=None, env_state=None):
        """Plain online greedy, closed loop on the executed actions."""
        actions = {a: int(np.argmax(self.q_network.q.numpy())) for a in observations}
        if state is not None:
            for a in actions:
                state["last"][a] = actions[a]
        return actions

    def train_on_batch(self, *args, **kwargs):
        self.train_calls += 1
        raise AssertionError("evaluation arms must not perform gradient updates")


def _obs():
    return {a: np.zeros(OBS_DIM, dtype=np.float32) for a in AGENTS}


def _legals():
    return {a: np.ones(N_ACTIONS, dtype=np.float32) for a in AGENTS}


def reward_if_all(action_value: float):
    """Reward 1 while every agent plays `action_value`, else 0."""

    def fn(actions, t):
        return 1.0 if all(int(a) == int(action_value) for a in actions.values()) else 0.0

    return fn


# --------------------------------------------------------------------- proposer


def test_conflict_source_and_p_zero_one():
    offline_q, online_q = [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]
    for p, expected_source in ((1.0, "offline"), (0.0, "online")):
        learner = FakeLearner(offline_q, online_q)
        policy = JointProposalPolicy(learner, p=p)
        state = policy.new_episode_state()
        trace: dict = {}
        actions = policy.select_actions_eval(_obs(), _legals(), 0.0, state, trace)
        assert actions == {a: int(np.argmax(offline_q if p else online_q)) for a in AGENTS}
        assert trace["proposal_conflict"] is True
        assert trace["proposal_source"] == expected_source
        stats = policy.stats(reset=True)
        assert stats["proposer/conflict_rate"] == 1.0
        assert stats["proposer/offline_select_rate"] == float(p == 1.0)


def test_no_conflict_means_no_randomisation():
    learner = FakeLearner([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    policy = JointProposalPolicy(learner, p=0.5)
    state = policy.new_episode_state()
    trace: dict = {}
    actions = policy.select_actions_eval(_obs(), _legals(), 0.0, state, trace)
    assert actions == {a: 0 for a in AGENTS}
    assert trace["proposal_conflict"] is False
    assert trace["proposal_source"] == "none"


def test_selection_rate_matches_p():
    np.random.seed(0)
    learner = FakeLearner([3.0, 0.0, 0.0], [0.0, 4.0, 0.0])
    policy = JointProposalPolicy(learner, p=0.5)
    state = policy.new_episode_state()
    for _ in range(400):
        policy.select_actions_eval(_obs(), _legals(), 0.0, state, None)
    stats = policy.stats(reset=True)
    # Conditioned on conflicting decisions, the only set the 0.5 protocol covers.
    assert stats["proposer/conflicts"] == 400
    assert abs(stats["proposer/offline_select_rate"] - 0.5) < 0.05
    assert abs(stats["proposer/offline_select_rate_all_steps"] - 0.5) < 0.05


def test_no_behaviour_propensity_is_recorded():
    learner = FakeLearner([3.0, 0.0, 0.0], [0.0, 4.0, 0.0])
    policy = JointProposalPolicy(learner, p=0.5)
    state = policy.new_episode_state()
    trace: dict = {}
    policy.select_actions_eval(_obs(), _legals(), 0.0, state, trace)
    assert set(trace) == {"proposal_source", "proposal_conflict"}
    assert not any("mu" in key for key in trace)


def test_action_histories_are_asymmetric():
    # Offline prefers action 0, online prefers action 1. Two steps are needed:
    # the first forward of each net always sees an empty history.
    offline_q, online_q = [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]
    for p, offline_history, online_history in ((1.0, 0, 0), (0.0, 0, 1)):
        learner = FakeLearner(offline_q, online_q)
        policy = JointProposalPolicy(learner, p=p)
        state = policy.new_episode_state()
        for _ in range(2):
            policy.select_actions_eval(_obs(), _legals(), 0.0, state, None)
        # Offline net is conditioned on its own greedy action (closed loop).
        assert learner.offline_q_network.seen_last_actions[-1] == offline_history
        # Online net is conditioned on the executed action, which under p = 0 is
        # its own greedy and under p = 1 is the offline candidate.
        assert learner.q_network.seen_last_actions[-1] == online_history
        assert state["offline"]["last"][AGENTS[0]] == 0
        assert state["last"][AGENTS[0]] == online_history


def test_exploration_is_applied_on_top_of_the_candidate():
    np.random.seed(3)
    learner = FakeLearner([3.0, 0.0, 0.0], [0.0, 4.0, 0.0])
    policy = JointProposalPolicy(learner, p=1.0)
    state = policy.new_episode_state()
    actions = policy.select_actions_eval(_obs(), _legals(), 1.0, state, None)
    # eps = 1 with the independent branch explores every agent.
    assert all(0 <= a < N_ACTIONS for a in actions.values())


def test_disabled_policy_delegates_to_the_learner():
    learner = FakeLearner([3.0, 0.0, 0.0], [0.0, 4.0, 0.0])
    policy = JointProposalPolicy(learner, p=0.5)
    policy.disable()
    assert policy.proposer_on is False
    actions = policy.select_actions_eval(_obs(), _legals(), 0.0, None, None)
    assert actions == {a: 1 for a in AGENTS}  # plain online greedy


# ------------------------------------------------------------------ eval arms


def test_arm_return_is_discounted_and_closed_loop():
    env = RecordingEnv(horizon=3, reward_fn=lambda actions, t: 1.0)
    learner = FakeLearner([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ret, steps = arm_return(env, learner.q_network, learner, GAMMA)
    assert steps == 3
    assert abs(ret - (1 + GAMMA + GAMMA**2)) < 1e-6
    # Every agent saw its own previous action.
    assert learner.q_network.seen_last_actions[0] == 0  # first step: no history
    assert all(a == 0 for a in learner.q_network.seen_last_actions[1:])


def test_paired_arms_share_one_seed_per_pair():
    built: list[int] = []

    def factory(seed):
        built.append(seed)
        return RecordingEnv(seed=seed, horizon=2, reward_fn=lambda a, t: 0.0)

    learner = FakeLearner([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    paired_arm_estimate(factory, learner, n_episodes=4, seed_base=7, gamma=GAMMA)
    assert len(built) == 8  # two envs per pair
    assert built[0::2] == built[1::2]  # shared seed inside each pair
    assert len(set(built[0::2])) == 4  # distinct seeds across pairs


def test_paired_estimate_matches_the_arm_returns():
    def factory(seed):
        return RecordingEnv(seed=seed, horizon=3, reward_fn=reward_if_all(0))

    learner = FakeLearner([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    est = paired_arm_estimate(factory, learner, n_episodes=5, seed_base=11, gamma=GAMMA)
    expected = 1 + GAMMA + GAMMA**2
    assert all(abs(r - expected) < 1e-6 for r in est["offline_returns"])
    assert all(abs(r) < 1e-9 for r in est["online_returns"])
    for d, off, on in zip(est["deltas"], est["offline_returns"], est["online_returns"]):
        assert abs(d - (off - on)) < 1e-9
    assert abs(est["tau_v"] - expected) < 1e-6
    assert est["eval_env_steps"] == 5 * 2 * 3


def test_arms_never_train_and_close_their_envs():
    learner = FakeLearner([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    envs: list[RecordingEnv] = []

    def factory(seed):
        env = RecordingEnv(seed=seed, horizon=2, reward_fn=lambda a, t: 0.0)
        envs.append(env)
        return env

    paired_arm_estimate(factory, learner, n_episodes=3, seed_base=5, gamma=GAMMA)
    assert learner.train_calls == 0
    assert all(env.closed for env in envs)


def test_exit_requires_no_measurable_positive_value():
    def factory(seed):
        return RecordingEnv(seed=seed, horizon=3, reward_fn=reward_if_all(0))

    learner = FakeLearner([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    good = paired_arm_estimate(factory, learner, n_episodes=8, seed_base=3, gamma=GAMMA)
    exit_now, logs = should_exit(good, min_eval_episodes=4, alpha=0.05)
    assert exit_now is False
    assert logs["proposer/tau_v"] > 0
    assert logs["proposer/tau_v_ub"] > 0

    def bad_factory(seed):
        return RecordingEnv(seed=seed, horizon=3, reward_fn=reward_if_all(1))

    bad = paired_arm_estimate(bad_factory, learner, n_episodes=8, seed_base=3, gamma=GAMMA)
    exit_now, logs = should_exit(bad, min_eval_episodes=4, alpha=0.05)
    assert exit_now is True
    assert logs["proposer/tau_v"] < 0
    assert logs["proposer/tau_v_ub"] <= 0


def test_exit_blocked_below_min_eval_episodes():
    def factory(seed):
        return RecordingEnv(seed=seed, horizon=2, reward_fn=reward_if_all(1))

    learner = FakeLearner([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    est = paired_arm_estimate(factory, learner, n_episodes=3, seed_base=1, gamma=GAMMA)
    exit_now, _ = should_exit(est, min_eval_episodes=8, alpha=0.05)
    assert exit_now is False


def test_one_sided_upper_bound():
    tau, std, n = 0.5, 1.0, 100
    ub = one_sided_upper_bound(tau, std, n, 0.05)
    assert abs(ub - (tau + 1.6448536269514722 * std / math.sqrt(n))) < 1e-9
    assert one_sided_upper_bound(0.2, 0.0, 10, 0.05) == 0.2
    assert one_sided_upper_bound(-0.2, 1.0, 1, 0.05) == -0.2


# ------------------------------------------------------------------ controller


def _controller(learner, reward_value, **block):
    def factory(seed):
        return RecordingEnv(seed=seed, horizon=3, reward_fn=reward_if_all(reward_value))

    cfg_block = {
        "p": 0.5,
        "version_len": 10,
        "eval_episodes_per_arm": 6,
        "min_eval_episodes": 4,
        "alpha": 0.05,
        "eval_seed_base": 100,
        "count_eval_in_budget": True,
    }
    cfg_block.update(block)
    return ProposerController(learner, cfg_block, factory)


def test_controller_exits_and_never_restarts():
    # Offline joint behaviour is worse, so the proposer must retire.
    learner = FakeLearner([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    controller = _controller(learner, reward_value=1)
    assert controller.due(10) is True
    logs, eval_steps = controller.boundary_logs(10)
    assert logs is not None
    assert logs["proposer/exited"] == 1.0
    assert controller.exited is True
    assert controller.policy.proposer_on is False
    assert eval_steps == 6 * 2 * 3
    assert controller.due(1000) is False
    # No further boundaries after the exit.
    assert controller.boundary_logs(1000) == (None, 0)
    assert controller.version == 1
    # Post-exit behaviour is the plain online policy.
    actions = controller.policy.select_actions_eval(_obs(), _legals(), 0.0, None, None)
    assert actions == {a: 1 for a in AGENTS}


def test_controller_keeps_proposer_while_value_remains():
    learner = FakeLearner([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    controller = _controller(learner, reward_value=0)
    logs, _ = controller.boundary_logs(10)
    assert logs["proposer/exited"] == 0.0
    assert controller.exited is False
    assert controller.policy.proposer_on is True
    # A later version is judged again from scratch.
    controller.last_version_t = 10
    assert controller.due(20) is True
    logs2, _ = controller.boundary_logs(20)
    assert logs2["proposer/version"] == 2.0
    assert logs2["proposer/tau_v"] > 0


def test_controller_seed_base_is_version_offset():
    learner = FakeLearner([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    seen: list[int] = []
    base = {"p": 0.5, "version_len": 10, "eval_episodes_per_arm": 2, "min_eval_episodes": 1,
            "alpha": 0.05, "eval_seed_base": 500, "count_eval_in_budget": False}

    def factory(seed):
        seen.append(seed)
        return RecordingEnv(seed=seed, horizon=2, reward_fn=reward_if_all(0))

    controller = ProposerController(learner, base, factory)
    controller.boundary_logs(10)
    controller.last_version_t = 10
    controller.boundary_logs(20)
    assert seen[0] != seen[4]  # version 2 draws different evaluation seeds
    assert len(seen) == 8  # two boundaries, two envs per pair, two pairs each


def test_controller_rejects_degenerate_version_len():
    learner = FakeLearner([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    for bad in ({"version_len": 0}, {"version_len": 100, "min_eval_episodes": 200}):
        try:
            ProposerController(learner, {"p": 0.5, **bad}, lambda seed: None)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_budget_accounting_can_be_disabled():
    learner = FakeLearner([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    controller = _controller(learner, reward_value=0, count_eval_in_budget=False)
    logs, eval_steps = controller.boundary_logs(10)
    assert logs is not None
    assert eval_steps == 0
    assert logs["proposer/eval_env_steps"] > 0


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} ok")
    print("fhjp tests ok")
