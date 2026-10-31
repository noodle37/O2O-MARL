#!/usr/bin/env python3
# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Unit tests for MARL-ROAD: bandit, surrogate reward, controller wiring.

Everything runs on CPU against fake Q networks and fake episode batches, so the
tests pin the mechanism (window UCB, masked Delta, period-fixed ratio, RNG
isolation, mutual exclusion) rather than any learning outcome.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from road.bandit import SlidingWindowUCB
from road.controller import RoadController, build_road_controller
from road.surrogate import joint_advantage_gap, surrogate_reward
from training.rollout import stack_episodes

N_AGENTS = 2
N_ACTIONS = 3
OBS_DIM = 2
STATE_DIM = 3
DEVICE = torch.device("cpu")


# ------------------------------------------------------------------ fakes


class FakeQMIX:
    """Returns a fixed mac_out; VDN-style sum mixer preserves monotonicity."""

    def __init__(self, mac_out: torch.Tensor):
        self.mac_out = mac_out
        self.device = DEVICE

    def _forward_qs(self, batch, target=False):
        return self.mac_out

    @staticmethod
    def mixer(agent_qs, states):
        return agent_qs.sum(dim=-1, keepdim=True)


class ShapeQMIX:
    """Per-agent Q = [0, 1, 2] regardless of the batch contents."""

    def __init__(self):
        self.device = DEVICE
        self.q = torch.tensor([0.0, 1.0, 2.0])

    def _forward_qs(self, batch, target=False):
        b, t, n, _ = batch["obs"].shape
        return self.q.view(1, 1, 1, -1).expand(b, t, n, -1)

    @staticmethod
    def mixer(agent_qs, states):
        return agent_qs.sum(dim=-1, keepdim=True)


class FakeOffline:
    def __init__(self, batch):
        self.batch = batch
        self.stats = SimpleNamespace(max_seq_length=4)

    def sample(self, batch_size, rng, device=None):
        return self.batch


def _batch(mac, actions, avail=None, filled=None, terminated=None, explored=None):
    b, t, n, a = mac.shape
    if avail is None:
        avail = torch.ones(b, t, n, a, dtype=torch.bool)
    if filled is None:
        filled = torch.ones(b, t, 1)
    if terminated is None:
        terminated = torch.zeros(b, t, 1)
    batch = {
        "obs": torch.zeros(b, t, n, OBS_DIM),
        "actions": torch.as_tensor(actions, dtype=torch.long).reshape(b, t, n, 1),
        "state": torch.zeros(b, t, STATE_DIM),
        "avail_actions": avail,
        "reward": torch.zeros(b, t, 1),
        "terminated": terminated,
        "filled": filled,
    }
    if explored is not None:
        batch["explored"] = explored
    return batch


def _episode(length=3):
    steps = []
    for _ in range(length):
        steps.append(
            {
                "obs": np.zeros((N_AGENTS, OBS_DIM), dtype=np.float32),
                "actions": np.zeros(N_AGENTS, dtype=np.int64),
                "state": np.zeros(STATE_DIM, dtype=np.float32),
                "reward": 0.0,
                "terminated": 0.0,
                "legals": np.ones((N_AGENTS, N_ACTIONS), dtype=np.float32),
                "explored": np.zeros(N_AGENTS, dtype=bool),
            }
        )
    return {"steps": steps, "length": length}


def _offline_batch(b=2, t=4):
    return _batch(torch.zeros(b, t, N_AGENTS, N_ACTIONS), np.zeros((b, t, N_AGENTS)))


# -------------------------------------------------------------------- bandit


def test_bandit_round_robin_first_pass():
    bandit = SlidingWindowUCB([0.1, 0.3, 0.5], c=2.0, window=10, seed=0)
    first = [bandit.select() for _ in range(3)]
    assert sorted(first) == [0.1, 0.3, 0.5]


def test_bandit_prefers_high_reward_arm():
    bandit = SlidingWindowUCB([0.0, 1.0, 2.0], c=2.0, window=100, seed=0)
    rewards = {0.0: 0.0, 1.0: 0.0, 2.0: 1.0}
    picks = []
    for _ in range(400):
        arm = bandit.select()
        picks.append(arm)
        bandit.update(arm, rewards[arm])
    assert picks.count(2.0) / len(picks) > 0.5


def test_bandit_window_forgets_regime_change():
    bandit = SlidingWindowUCB([0.0, 1.0], c=2.0, window=50, seed=0)

    def run(rounds, best):
        picks = []
        for _ in range(rounds):
            arm = bandit.select()
            picks.append(arm)
            bandit.update(arm, 1.0 if arm == best else 0.0)
        return picks

    run(300, 0.0)
    picks = run(300, 1.0)
    tail = picks[-100:]
    assert tail.count(1.0) > tail.count(0.0)


def test_bandit_state_roundtrip():
    bandit = SlidingWindowUCB([0.1, 0.2], c=1.5, window=7, seed=3)
    for _ in range(5):
        arm = bandit.select()
        bandit.update(arm, float(arm))
    clone = SlidingWindowUCB([0.1, 0.2], c=1.5, window=7, seed=99)
    clone.load_state_dict(bandit.state_dict())
    assert clone.total_counts() == bandit.total_counts()
    assert clone.window_counts() == bandit.window_counts()


# ------------------------------------------------------------------ surrogate


def test_gap_is_nonnegative():
    torch.manual_seed(0)
    mac = torch.randn(4, 6, N_AGENTS, N_ACTIONS)
    avail = torch.rand(4, 6, N_AGENTS, N_ACTIONS) > 0.3
    avail[..., 0] = True  # every agent has at least one legal action
    actions = torch.randint(0, N_ACTIONS, (4, 6, N_AGENTS))
    # Force the data action to be legal so the sample is meaningful.
    for b in range(4):
        for t in range(6):
            for i in range(N_AGENTS):
                legal = torch.nonzero(avail[b, t, i]).squeeze(-1)
                actions[b, t, i] = legal[torch.randint(legal.numel(), (1,))]
    batch = _batch(mac, actions, avail=avail)
    assert joint_advantage_gap(FakeQMIX(mac), batch) >= 0.0


def test_gap_zero_when_data_action_is_greedy():
    mac = torch.randn(2, 5, N_AGENTS, N_ACTIONS)
    avail = torch.ones(2, 5, N_AGENTS, N_ACTIONS, dtype=torch.bool)
    greedy = mac.argmax(dim=-1)
    batch = _batch(mac, greedy, avail=avail)
    assert abs(joint_advantage_gap(FakeQMIX(mac), batch)) < 1e-6


def test_illegal_actions_are_masked():
    mac = torch.tensor([[[[0.5, 100.0, 0.3]]]])
    avail = torch.tensor([[[[True, False, True]]]])
    batch = _batch(mac, np.array([[[2]]]), avail=avail)
    gap = joint_advantage_gap(FakeQMIX(mac), batch)
    assert abs(gap - 0.2) < 1e-6


def test_explored_steps_are_masked_out():
    mac = torch.tensor([1.0, 3.0, 1.0]).view(1, 1, 1, 3).repeat(1, 3, 1, 1)
    actions = np.array([[[1], [0], [1]]])  # greedy everywhere except t=1
    explored = torch.zeros(1, 3, 1)
    explored[0, 1, 0] = 1.0
    batch = _batch(mac, actions, explored=explored)
    mask = 1.0 - explored
    assert abs(joint_advantage_gap(FakeQMIX(mac), batch, extra_mask=mask)) < 1e-6
    # Without masking the gap at t=1 survives.
    assert joint_advantage_gap(FakeQMIX(mac), batch) > 0.5


def test_surrogate_reward_sign():
    assert abs(surrogate_reward(2.0, 0.5, 1.0) - 1.5) < 1e-9
    assert abs(surrogate_reward(2.0, 0.5, 0.0) - 2.0) < 1e-9
    assert abs(surrogate_reward(2.0, 0.5, 1e6) + 499998.0) < 1e-3


# ------------------------------------------------------------------ rollout


def test_stack_episodes_carries_any_agent_explored():
    episode = _episode(length=3)
    episode["steps"][1]["explored"] = np.array([False, True])
    batch = stack_episodes([episode], max_seq=5, device=DEVICE)
    assert batch["explored"].shape == (1, 5, 1)
    assert batch["explored"][0, 0, 0].item() == 0.0
    assert batch["explored"][0, 1, 0].item() == 1.0
    assert batch["explored"][0, 4, 0].item() == 0.0


# ---------------------------------------------------------------- controller


def _controller(learner=None, **block):
    learner = learner or ShapeQMIX()
    cfg_block = {
        "arms": [0.1, 0.2],
        "kappa": 1.0,
        "ucb_c": 2.0,
        "window": 10,
        "period_steps": 3,
        "delta_batch": 2,
        "normalize_reward": True,
        "mask_explored": True,
        "seed_offset": 7,
    }
    cfg_block.update(block)
    return RoadController(learner, cfg_block, seed=1)


def test_ratio_fixed_within_period_and_row_at_boundary():
    learner = ShapeQMIX()
    offline = FakeOffline(_offline_batch())
    buffer = deque(_episode() for _ in range(4))
    road = _controller(learner)
    ratio = road.ratio
    assert road.after_train_step(offline, buffer, 10) is None
    assert road.after_train_step(offline, buffer, 20) is None
    assert road.ratio == ratio
    row = road.after_train_step(offline, buffer, 30)
    assert row is not None
    assert row["road/period"] == 1.0
    assert road.ratio in road.arms
    assert road.period == 1
    assert road.steps_in_period == 0
    assert "road/delta_off" in row and "road/delta_on" in row


def test_delta_on_is_zero_when_every_step_explored():
    learner = ShapeQMIX()
    offline = FakeOffline(_offline_batch())
    episodes = []
    for _ in range(4):
        episode = _episode()
        for step in episode["steps"]:
            step["explored"] = np.array([True, False])
        episodes.append(episode)
    buffer = deque(episodes)
    road = _controller(learner)
    rows = [road.after_train_step(offline, buffer, t) for t in (10, 20, 30)]
    assert rows[-1]["road/delta_on"] == 0.0
    assert rows[-1]["road/explored_frac_on"] == 1.0


def test_controller_does_not_touch_the_main_rng():
    learner = ShapeQMIX()
    offline = FakeOffline(_offline_batch())
    buffer = deque(_episode() for _ in range(4))
    road = _controller(learner)
    main_rng = np.random.default_rng(123)
    before = main_rng.bit_generator.state
    for t in (10, 20, 30, 40, 50, 60):
        road.after_train_step(offline, buffer, t)
    assert main_rng.bit_generator.state == before


def test_controller_summary_counts_per_arm():
    learner = ShapeQMIX()
    offline = FakeOffline(_offline_batch())
    buffer = deque(_episode() for _ in range(4))
    road = _controller(learner)
    for t in range(1, 3 * 7 + 1):
        road.after_train_step(offline, buffer, t)
    summary = road.summary()
    assert set(summary) == {"road/count_total_0.1", "road/count_total_0.2"}
    # The initial arm selection plus one selection per completed period.
    assert sum(summary.values()) == 8


def test_build_controller_is_none_when_disabled():
    assert build_road_controller({"road": {"enabled": False}}, ShapeQMIX()) is None
    assert build_road_controller({}, ShapeQMIX()) is None


def test_build_controller_rejects_incompatible_configs():
    learner = ShapeQMIX()
    cases = [
        {"road": {"enabled": True}, "learner_type": "fsr"},
        {"road": {"enabled": True}, "fork_at": 100},
        {"road": {"enabled": True}, "resume_fork": "snapshot.pt"},
        {"road": {"enabled": True}, "learner_type": "mac_flow"},
    ]
    for cfg in cases:
        try:
            build_road_controller(cfg, learner if cfg.get("learner_type") != "mac_flow" else object())
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {cfg}")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} ok")
    print("road tests ok")
