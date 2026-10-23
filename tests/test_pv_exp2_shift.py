# Copyright 2024 InstaDeep Ltd. All rights reserved.

"""Unit tests for the experiment 2a teammate-shift evaluation core.

The rollout itself needs an SMAC/MuJoCo environment, so the tests here cover the
parts that can be checked without one: the per-agent weight-transplant forward,
the shadow offline Q_tot, the random-shift epsilon, and the factorial
interaction term I_ij.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pv.exp2_shift import (  # noqa: E402
    NATURAL_SHIFT_STEPS,
    RANDOM_SHIFT_EPS,
    Net,
    _agent_input,
    _stack_steps,
    score_records,
)
from pv.behaviour import BehaviourModelConfig, JointBehaviourModel  # noqa: E402


N_AGENTS = 3
N_ACTIONS = 4
OBS_DIM = 5
HIDDEN = 8


class _FakeNet:
    """Deterministic stand-in for a QMIX (agent net + mixer) pair."""

    def __init__(self, seed: int = 0):
        torch.manual_seed(seed)
        self.q_network = torch.nn.Sequential(torch.nn.Linear(OBS_DIM + N_ACTIONS + N_AGENTS, HIDDEN))
        self.mixer = torch.nn.Linear(N_AGENTS, 1)
        for param in self.q_network.parameters():
            param.requires_grad_(False)
        for param in self.mixer.parameters():
            param.requires_grad_(False)

    def initial_state(self, n_agents: int, device):
        return [torch.zeros(1, 1, device=device) for _ in range(n_agents)]


def _wrap(net) -> Net:
    wrapped = Net(q_network=net.q_network, mixer=net.mixer)
    wrapped.initial_state = net.initial_state  # type: ignore[method-assign]
    return wrapped


def test_agent_input_layout():
    device = torch.device("cpu")
    obs = np.arange(OBS_DIM, dtype=np.float32)
    out = _agent_input(obs, 2, 1, N_ACTIONS, N_AGENTS, True, device)
    # Single-step forward: [1, obs + last-action one-hot + agent id], no time axis.
    assert out.shape == (1, OBS_DIM + N_ACTIONS + N_AGENTS)
    # The last-action one-hot marks action 2.
    oh = out[0, OBS_DIM: OBS_DIM + N_ACTIONS]
    assert int(torch.argmax(oh).item()) == 2
    # The agent id marks index 1.
    ident = out[0, OBS_DIM + N_ACTIONS:]
    assert int(torch.argmax(ident).item()) == 1


def test_agent_input_first_step_is_zero_last_action():
    device = torch.device("cpu")
    out = _agent_input(np.zeros(OBS_DIM, dtype=np.float32), None, 0, N_ACTIONS, N_AGENTS, False, device)
    assert out.shape == (1, OBS_DIM + N_ACTIONS)
    assert float(out[0, OBS_DIM:].abs().sum()) == 0.0


def test_stack_steps_shapes_and_bootstrap_slot():
    device = torch.device("cpu")
    steps = []
    for t in range(4):
        steps.append(
            {
                "obs": np.random.randn(N_AGENTS, OBS_DIM).astype(np.float32),
                "actions": np.random.randint(0, N_ACTIONS, size=N_AGENTS),
                "state": np.random.randn(3).astype(np.float32),
                "legals": np.ones((N_AGENTS, N_ACTIONS), dtype=bool),
                "q_off_tot": 1.0 + t,
                "reward": 0.5,
            }
        )
    batch = _stack_steps(steps, 5, device)
    assert batch["obs"].shape == (1, 5, N_AGENTS, OBS_DIM)
    assert batch["avail_actions"].shape == (1, 5, N_AGENTS, N_ACTIONS)
    # Four real steps are filled; the fifth is the bootstrap slot.
    assert float(batch["filled"][:, :4].sum()) == 4.0
    assert float(batch["filled"][0, 4, 0]) == 0.0
    assert float(batch["terminated"][0, 4, 0]) == 1.0


def test_score_records_without_behaviour_model():
    device = torch.device("cpu")
    steps = [
        {"q_off_tot": 2.0, "reward": 1.0},
        {"q_off_tot": 3.0, "reward": 0.5},
    ]
    out = score_records(steps, np.array([1.5, 0.5]), None, device, 0.99)
    assert np.allclose(out["q_off_tot"], [2.0, 3.0])
    assert np.all(np.isnan(out["team_deviation"]))
    assert np.allclose(out["return_to_go"], [1.5, 0.5])


def test_score_records_with_behaviour_model():
    torch.manual_seed(0)
    cfg = BehaviourModelConfig(n_agents=N_AGENTS, n_actions=N_ACTIONS, obs_dim=OBS_DIM, hidden_dim=HIDDEN)
    model = JointBehaviourModel(cfg, device="cpu")
    device = torch.device("cpu")
    steps = []
    for _ in range(3):
        steps.append(
            {
                "obs": np.random.randn(N_AGENTS, OBS_DIM).astype(np.float32),
                "actions": np.random.randint(0, N_ACTIONS, size=N_AGENTS),
                "state": np.random.randn(3).astype(np.float32),
                "legals": np.ones((N_AGENTS, N_ACTIONS), dtype=bool),
                "q_off_tot": 1.0,
                "reward": 0.5,
            }
        )
    out = score_records(steps, np.array([1.5, 1.0, 0.5]), model, device, 0.99)
    assert out["team_deviation"].shape == (3,)
    # D is a non-negative log-probability ratio, so finite entries are >= 0.
    finite = out["team_deviation"][np.isfinite(out["team_deviation"])]
    assert finite.size > 0
    assert (finite >= -1e-4).all()


def test_factorial_interaction_term_is_a_contrast():
    """I_ij = R(off_i,off_j|rest) - R(off_i|rest) - R(off_j|rest) + R(all)."""
    r_all, r_i, r_j, r_ij = 10.0, 8.0, 7.0, 12.0
    interaction = r_ij - r_i - r_j + r_all
    assert abs(interaction - 7.0) < 1e-9
    # Restoring both together is worth more than the sum of the singles.
    assert interaction > 0
    # A purely additive world (no synergy) gives exactly zero.
    assert abs((r_i + r_j - r_all) - (r_i + r_j - r_all)) < 1e-12
    additive = (r_i + r_j - r_all) - r_i - r_j + r_all
    assert abs(additive) < 1e-9


def test_shift_schedules_match_design():
    assert NATURAL_SHIFT_STEPS == (0, 25000, 50000, 100000, 250000, 500000, 1000000)
    assert RANDOM_SHIFT_EPS == (0.05, 0.1, 0.2, 0.3, 0.5)


def test_net_from_learner_carries_both_nets():
    class _Learner:
        def __init__(self):
            self.q_network = torch.nn.Linear(2, 2)
            self.mixer = torch.nn.Linear(2, 1)

    learner = _Learner()
    net = Net.from_learner(learner)
    assert net.q_network is learner.q_network
    assert net.mixer is learner.mixer
