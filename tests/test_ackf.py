#!/usr/bin/env python3
# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""CPU tests for ACKF: gates, frozen knowledge, fusion, targets, checkpoints."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from ackf.learner import ACKFLearner
from training.online import _load_learner_weights
from utils.checkpoint import save_qmix_checkpoint

N_AGENTS = 3
N_ACTIONS = 5
OBS_DIM = 8
STATE_DIM = 16


def _learner(**overrides) -> ACKFLearner:
    kwargs = dict(
        n_agents=N_AGENTS,
        n_actions=N_ACTIONS,
        obs_dim=OBS_DIM,
        state_dim=STATE_DIM,
        device="cpu",
        learning_rate=1e-2,
        target_update_interval=2,
    )
    kwargs.update(overrides)
    learner = ACKFLearner(**kwargs)
    learner.set_offline_encoder(
        copy.deepcopy(learner.q_network), copy.deepcopy(learner.mixer)
    )
    learner.prepare_online_start()
    return learner


def _dummy_batch(batch_size=4, seq=6) -> dict:
    device = torch.device("cpu")
    obs = torch.randn(batch_size, seq, N_AGENTS, OBS_DIM, device=device)
    actions = torch.randint(0, N_ACTIONS, (batch_size, seq, N_AGENTS, 1), device=device)
    state = torch.randn(batch_size, seq, STATE_DIM, device=device)
    avail = torch.ones(
        batch_size, seq, N_AGENTS, N_ACTIONS, dtype=torch.bool, device=device
    )
    reward = torch.randn(batch_size, seq, 1, device=device)
    terminated = torch.zeros(batch_size, seq, 1, device=device)
    filled = torch.ones(batch_size, seq, 1, device=device)
    filled[:, -1, 0] = 0.0
    return {
        "obs": obs,
        "actions": actions,
        "state": state,
        "avail_actions": avail,
        "reward": reward,
        "terminated": terminated,
        "filled": filled,
    }


def _obs_dict(seed=0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    return {
        f"agent_{i}": torch.randn(OBS_DIM, generator=gen).numpy()
        for i in range(N_AGENTS)
    }


def _legal_dict() -> dict:
    return {f"agent_{i}": torch.ones(N_ACTIONS).numpy() for i in range(N_AGENTS)}


def test_gate_initialisation_and_frozen_encoder():
    torch.manual_seed(0)
    learner = _learner(gate_init=0.1)
    gates = torch.sigmoid(learner.gate_psi).detach()
    assert torch.allclose(gates, torch.full((N_AGENTS,), 0.1), atol=1e-6)
    expected_logit = math.log(0.1 / 0.9)
    assert torch.allclose(
        learner.gate_psi.detach(), torch.full((N_AGENTS,), expected_logit), atol=1e-6
    )

    assert learner.offline_q_network is not None
    assert all(not p.requires_grad for p in learner.offline_q_network.parameters())
    assert all(not p.requires_grad for p in learner.offline_mixer.parameters())
    # The offline encoder has the same initial parameters as the online learner.
    for key, value in learner.q_network.state_dict().items():
        assert torch.equal(value, learner.offline_q_network.state_dict()[key])
    # Targets are initialised from the online modules.
    assert torch.equal(learner.target_gate_psi, learner.gate_psi)


def test_train_step_updates_trainables_and_freezes_encoder():
    torch.manual_seed(1)
    learner = _learner()
    offline_before = {
        k: v.clone() for k, v in learner.offline_q_network.state_dict().items()
    }
    q_before = {
        k: v.clone() for k, v in learner.q_network.state_dict().items()
    }
    adapter_before = {
        k: v.clone() for k, v in learner.knowledge_adapter.state_dict().items()
    }
    gate_before = learner.gate_psi.detach().clone()

    logs = learner.train_on_batch(_dummy_batch(), t_env=0)

    assert math.isfinite(logs["loss"])
    assert "ackf/gate_mean" in logs and "ackf/delta_abs_mean" in logs
    assert learner.gate_psi.grad is not None
    assert float(learner.gate_psi.grad.abs().sum()) > 0.0
    assert learner.knowledge_adapter.net[0].weight.grad is not None
    assert float(learner.knowledge_adapter.net[0].weight.grad.abs().sum()) > 0.0
    assert any(
        not torch.equal(v, learner.q_network.state_dict()[k])
        for k, v in q_before.items()
    )
    assert any(
        not torch.equal(v, learner.knowledge_adapter.state_dict()[k])
        for k, v in adapter_before.items()
    )
    assert not torch.equal(gate_before, learner.gate_psi.detach())
    for key, value in learner.offline_q_network.state_dict().items():
        assert torch.equal(value, offline_before[key])


def test_fused_utility_matches_equation():
    torch.manual_seed(2)
    learner = _learner()
    batch = _dummy_batch()
    obs_aug = learner._augment_obs(batch["obs"], batch["actions"])
    resets = learner._resets_from_batch(batch)
    knowledge = learner._offline_knowledge(obs_aug, resets)
    with torch.no_grad():
        fused, q_on, delta_q, gate = learner._fused_utilities(
            obs_aug, resets, knowledge, target=False
        )
    expected = q_on + torch.sigmoid(learner.gate_psi).view(1, 1, N_AGENTS, 1) * delta_q
    assert torch.allclose(fused, expected, atol=1e-6)
    assert torch.allclose(
        gate, torch.sigmoid(learner.gate_psi).view(1, 1, N_AGENTS, 1), atol=1e-6
    )


def test_action_selection_uses_fused_utility():
    torch.manual_seed(3)
    learner = _learner()
    obs = _obs_dict(seed=4)
    legal = _legal_dict()
    state = learner.new_episode_state()
    manual_rnn = {key: value.clone() for key, value in state["rnn"].items()}
    manual_offline_rnn = {
        key: value.clone() for key, value in state["offline_rnn"].items()
    }
    trace = {}
    actions = learner.select_actions_eval(obs, legal, eps=0.0, state=state, trace=trace)

    expected_actions = {}
    expected_chosen = []
    gate = torch.sigmoid(learner.gate_psi).detach()
    for i, agent in enumerate(sorted(obs.keys())):
        obs_in = torch.cat(
            [
                torch.from_numpy(obs[agent]).float(),
                torch.zeros(learner.n_actions),
                F.one_hot(torch.tensor(i), N_AGENTS).float(),
            ]
        ).unsqueeze(0)
        with torch.no_grad():
            q_on, _ = learner.q_network(obs_in, manual_rnn[agent])
            q_off, h_off = learner.offline_q_network(
                obs_in, manual_offline_rnn[agent]
            )
            knowledge = learner.knowledge_norm(
                torch.cat([h_off.reshape(1, -1), q_off.reshape(1, -1)], dim=-1)
            )
            delta_q = learner.knowledge_adapter(
                knowledge, F.one_hot(torch.tensor(i), N_AGENTS).float().unsqueeze(0)
            )
            fused = q_on + gate[i].view(1) * delta_q
        expected_actions[agent] = int(torch.argmax(fused).item())
        expected_chosen.append(float(fused[0, expected_actions[agent]].item()))

    assert actions == expected_actions
    assert torch.allclose(
        torch.tensor(trace["chosen_q"]), torch.tensor(expected_chosen), atol=1e-6
    )
    assert len(trace["offline_chosen_q"]) == N_AGENTS
    assert "offline_rnn" in state and state["explored"] == [False] * N_AGENTS


def test_target_sync_covers_adapter_and_gates():
    torch.manual_seed(5)
    learner = _learner(target_update_interval=1)
    for step in range(2):
        learner.train_on_batch(_dummy_batch(), t_env=step)
    for key, value in learner.knowledge_adapter.state_dict().items():
        assert torch.equal(value, learner.target_knowledge_adapter.state_dict()[key])
    assert torch.equal(learner.gate_psi.detach(), learner.target_gate_psi.detach())
    for key, value in learner.mixer.state_dict().items():
        assert torch.equal(value, learner.target_mixer.state_dict()[key])


def test_checkpoint_roundtrip(tmp_path: Path):
    torch.manual_seed(6)
    learner = _learner()
    learner.train_on_batch(_dummy_batch(), t_env=0)
    path = save_qmix_checkpoint(
        str(tmp_path / "ackf.pt"), learner, include_optimizer=True
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert "ackf_adapter" in payload and "ackf_gate_psi" in payload

    restored = ACKFLearner(
        n_agents=N_AGENTS,
        n_actions=N_ACTIONS,
        obs_dim=OBS_DIM,
        state_dim=STATE_DIM,
        device="cpu",
    )
    _load_learner_weights(restored, payload)
    for key, value in learner.knowledge_adapter.state_dict().items():
        assert torch.equal(value, restored.knowledge_adapter.state_dict()[key])
    assert torch.equal(learner.gate_psi.detach(), restored.gate_psi.detach())
    assert restored.offline_q_network is not None
    assert all(not p.requires_grad for p in restored.offline_q_network.parameters())


def test_factory_registers_ackf():
    from training.factory import build_learner

    stats = SimpleNamespace(
        n_agents=N_AGENTS,
        n_actions=N_ACTIONS,
        obs_dim=OBS_DIM,
        state_dim=STATE_DIM,
        action_continuous=False,
    )
    dataset = SimpleNamespace(stats=stats)
    cfg = {
        "learner_type": "ackf",
        "seed": 1,
        "batch_size_run": 1,
        "task": {"action_mode": "discrete"},
        "system": {
            "device": "cpu",
            "learning_rate": 1e-3,
            "recurrent_layer_dim": 32,
            "mixer_embed_dim": 16,
            "mixer_hyper_dim": 16,
            "target_update_interval": 10,
        },
        "ackf": {"adapter_hidden_dim": 16, "gate_init": 0.2},
    }
    learner = build_learner(cfg, dataset)
    assert isinstance(learner, ACKFLearner)
    assert torch.allclose(
        torch.sigmoid(learner.gate_psi),
        torch.full((N_AGENTS,), 0.2),
        atol=1e-6,
    )


if __name__ == "__main__":
    test_gate_initialisation_and_frozen_encoder()
    test_train_step_updates_trainables_and_freezes_encoder()
    test_fused_utility_matches_equation()
    test_action_selection_uses_fused_utility()
    test_target_sync_covers_adapter_and_gates()
    test_factory_registers_ackf()
    print("ackf tests ok")
