#!/usr/bin/env python3

"""Unit checks for VCKR sub-team knowledge, validity and gated execution."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from vckr.knowledge import VCKRParams
from vckr.learner import VCKRLearner
from vckr.validity import validity_inference

N_AGENTS = 3
N_ACTIONS = 4
OBS_DIM = 5
STATE_DIM = 6
N_EPISODES = 16
SEQ = 8


class FakeDataset:
    def __init__(self, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.obs = torch.randn(N_EPISODES, SEQ, N_AGENTS, OBS_DIM, generator=g)
        self.actions = torch.randint(0, N_ACTIONS, (N_EPISODES, SEQ, N_AGENTS, 1), generator=g)
        self.state = torch.randn(N_EPISODES, SEQ, STATE_DIM, generator=g)
        self.avail_actions = torch.ones(
            N_EPISODES, SEQ, N_AGENTS, N_ACTIONS, dtype=torch.bool
        )
        self.reward = torch.randn(N_EPISODES, SEQ, 1, generator=g)
        self.terminated = torch.zeros(N_EPISODES, SEQ, 1)
        self.filled = torch.ones(N_EPISODES, SEQ, 1)
        self.filled[:, -1] = 0.0

    def sample(self, batch_size, rng, device=None):
        idx = rng.choice(N_EPISODES, size=min(batch_size, N_EPISODES), replace=False)
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        batch = {
            "obs": self.obs.index_select(0, idx_t),
            "actions": self.actions.index_select(0, idx_t),
            "state": self.state.index_select(0, idx_t),
            "avail_actions": self.avail_actions.index_select(0, idx_t),
            "reward": self.reward.index_select(0, idx_t),
            "terminated": self.terminated.index_select(0, idx_t),
            "filled": self.filled.index_select(0, idx_t),
        }
        if device is not None:
            target = torch.device(device)
            batch = {k: v.to(target) for k, v in batch.items()}
        return batch


def make_params(**kwargs) -> VCKRParams:
    base = dict(
        horizon=3,
        max_units=4,
        unit_sizes=(1, 2),
        coupling_batches=2,
        coupling_batch_size=4,
        coupling_stride=2,
        gate_hidden=8,
        lock_horizon=2,
        delta_clip=5.0,
        max_selected_units=2,
    )
    base.update(kwargs)
    return VCKRParams(**base)


def make_learner(params: VCKRParams) -> VCKRLearner:
    learner = VCKRLearner(
        n_agents=N_AGENTS,
        n_actions=N_ACTIONS,
        obs_dim=OBS_DIM,
        state_dim=STATE_DIM,
        recurrent_layer_dim=8,
        mixer_embed_dim=8,
        mixer_hyper_dim=8,
        device="cpu",
        vckr_params=params,
    )
    learner.offline_q_network = copy.deepcopy(learner.q_network)
    learner.offline_mixer = copy.deepcopy(learner.mixer)
    learner.offline_q_network.eval()
    learner.offline_mixer.eval()
    for p in learner.offline_q_network.parameters():
        p.requires_grad_(False)
    for p in learner.offline_mixer.parameters():
        p.requires_grad_(False)
    return learner


def prepared(**kwargs):
    learner = make_learner(make_params(**kwargs))
    dataset = FakeDataset()
    learner.setup_vckr(dataset, {"seed": 1})
    return learner, dataset


def test_knowledge_and_validity_shapes():
    learner, dataset = prepared()
    knowledge = learner.knowledge
    assert knowledge is not None and knowledge.units
    assert knowledge.coupling.shape == (N_AGENTS, N_AGENTS)
    assert knowledge.membership.shape == (len(knowledge.units), N_AGENTS)
    assert knowledge.delta_temp >= 0.1

    batch = dataset.sample(4, np.random.default_rng(0), device=learner.device)
    mac_out = learner._forward_qs(batch)
    off_out = learner._offline_unroll(batch)
    validity = validity_inference(learner, batch, mac_out, off_out, knowledge, learner.vckr_params)

    b, t = batch["obs"].shape[:2]
    u = len(knowledge.units)
    for key in ("delta_on", "delta_off", "delta_hat", "alpha", "tv", "selected"):
        assert validity[key].shape == (u, b, t), key
        assert torch.isfinite(validity[key]).all(), key
    assert validity["member"].shape == (b, t, N_AGENTS)
    assert validity["bonus"].shape == (b, t)
    assert torch.all(validity["bonus"] >= 0.0)
    assert torch.all(validity["bonus"] <= learner.vckr_params.delta_clip + 1e-6)
    assert torch.all(validity["alpha"] >= 0.0) and torch.all(validity["alpha"] <= 1.0)
    assert torch.all(validity["tv"] >= 0.0) and torch.all(validity["tv"] <= 1.0 + 1e-5)

    # The greedy partition must not assign an agent to two selected units.
    cover = (
        validity["selected"].float().unsqueeze(-1) * knowledge.membership.unsqueeze(1).unsqueeze(2)
    ).sum(dim=0)
    assert torch.all(cover <= 1.0 + 1e-6)


def test_tv_penalty_blocks_expired_knowledge():
    learner, dataset = prepared(beta_tv=1000.0)
    batch = dataset.sample(4, np.random.default_rng(1), device=learner.device)
    mac_out = learner._forward_qs(batch)
    off_out = learner._offline_unroll(batch)
    validity = validity_inference(learner, batch, mac_out, off_out, learner.knowledge, learner.vckr_params)
    assert not bool(validity["selected"].any())
    assert torch.all(validity["bonus"] == 0.0)


def test_train_on_batch_runs_both_channels():
    learner, dataset = prepared()
    batch = dataset.sample(4, np.random.default_rng(2), device=learner.device)
    before = [p.detach().clone() for p in learner.gate_head.parameters()]
    logs = learner.train_on_batch(batch, t_env=0)
    assert np.isfinite(logs["loss"])
    for key in ("vckr/bonus_mean", "vckr/selected_units", "vckr/gate_loss", "vckr/gate_acc"):
        assert key in logs and np.isfinite(logs[key]), key
    changed = any(not torch.equal(a, b) for a, b in zip(before, learner.gate_head.parameters()))
    assert changed, "gate head should receive a distillation update"


class ConstantNet(torch.nn.Module):
    def __init__(self, preferred: int):
        super().__init__()
        self.register_buffer("q", torch.tensor([0.0, 0.0, 5.0, 0.0]))
        self.preferred = preferred

    def initial_state(self, batch_size, device):
        return torch.zeros(1, batch_size, 1, device=device)

    def forward(self, x, hidden):
        return self.q.unsqueeze(0), hidden


def _obs():
    return {f"agent_{i}": np.zeros(OBS_DIM, dtype=np.float32) for i in range(N_AGENTS)}


def _legals():
    return {f"agent_{i}": np.ones(N_ACTIONS, dtype=np.float32) for i in range(N_AGENTS)}


def test_gate_locks_agent_to_offline_snippet():
    learner = make_learner(make_params())
    learner.offline_q_network = ConstantNet(preferred=2)
    with torch.no_grad():
        learner.gate_head.net[-1].bias.fill_(10.0)
    learner.reset_eval_hidden()

    actions = learner.select_actions_eval(_obs(), _legals(), 0.0, state=None)
    assert set(actions.values()) == {2}
    assert all(v == 1 for v in learner.eval_lock_left.values())  # lock_horizon=2 -> one left

    state = learner.new_episode_state()
    trace = {}
    actions = learner.select_actions_eval(_obs(), _legals(), 0.3, state=state, trace=trace)
    assert set(actions.values()) == {2}
    assert set(state.keys()) >= {"rnn", "last", "offline_rnn", "lock_left", "explored"}
    assert len(state["explored"]) == N_AGENTS
    assert trace["vckr_locked"] == N_AGENTS


def test_gate_is_saved_and_restored():
    learner = make_learner(make_params())
    payload = {"q_network": learner.q_network.state_dict(), **learner.state_dicts()}
    assert "vckr_gate" in payload
    other = make_learner(make_params())
    other.load_offline_state(payload)
    for a, b in zip(learner.gate_head.parameters(), other.gate_head.parameters()):
        assert torch.equal(a, b)
