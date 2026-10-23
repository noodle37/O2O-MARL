# Copyright 2024 InstaDeep Ltd. All rights reserved.

"""Unit tests for the joint behaviour model and the deviation measure D."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pv.behaviour import (  # noqa: E402
    BehaviourModelConfig,
    JointBehaviourModel,
    NoveltyEnsemble,
    fit_behaviour_model,
    gather_episodes,
    load_behaviour_model,
    random_order,
    save_behaviour_model,
    split_episodes,
)


class _FakeDataset:
    """Minimal stand-in for EpisodeDataset (only what the BC code touches)."""

    def __init__(
        self,
        n_episodes=8,
        length=6,
        n_agents=3,
        n_actions=4,
        obs_dim=5,
        seed=0,
        teacher: bool = False,
    ):
        rng = np.random.default_rng(seed)
        self.n_episodes = n_episodes
        self.device = torch.device("cpu")
        self.obs = torch.zeros(n_episodes, length + 1, n_agents, obs_dim)
        self.actions = torch.zeros(n_episodes, length + 1, n_agents, 1, dtype=torch.long)
        self.avail_actions = torch.ones(n_episodes, length + 1, n_agents, n_actions, dtype=torch.bool)
        self.state = torch.zeros(n_episodes, length + 1, 3)
        self.reward = torch.zeros(n_episodes, length + 1, 1)
        self.terminated = torch.zeros(n_episodes, length + 1, 1)
        self.filled = torch.zeros(n_episodes, length + 1, 1)
        for e in range(n_episodes):
            ep_len = rng.integers(2, length + 1)
            for t in range(ep_len):
                self.obs[e, t] = torch.from_numpy(rng.normal(size=(n_agents, obs_dim))).float()
                if teacher:
                    # Learnable rule: action follows the sign of the first obs dim.
                    base = (self.obs[e, t, :, 0] > 0).long()
                    acts = (base + t) % n_actions
                else:
                    acts = torch.from_numpy(rng.integers(0, n_actions, size=n_agents)).long()
                self.actions[e, t, :, 0] = acts
                self.reward[e, t, 0] = float(rng.normal())
                self.filled[e, t, 0] = 1.0
            self.obs[e, ep_len] = self.obs[e, ep_len - 1]
            self.avail_actions[e, ep_len] = self.avail_actions[e, ep_len - 1]
            self.terminated[e, ep_len, 0] = 1.0
            # Bootstrap slot copied, filled = 0.
        self.n_agents = n_agents
        self.n_actions = n_actions


def _batch(dataset, indices):
    return gather_episodes(dataset, indices, device="cpu")


def _model(seed=0):
    torch.manual_seed(seed)
    cfg = BehaviourModelConfig(n_agents=3, n_actions=4, obs_dim=5, hidden_dim=8)
    return JointBehaviourModel(cfg, device="cpu")


def test_order_with_first_and_marginal_independence():
    model = _model()
    dataset = _FakeDataset()
    batch = _batch(dataset, [0, 1, 2])
    order = JointBehaviourModel.order_with_first(3, 1, batch["obs"].shape[0], "cpu")
    assert order.shape == (3, 3)
    assert torch.all(order[:, 0] == 1)
    # The position-0 (marginal) log-prob must not move when the other agents'
    # current-step actions change.
    logp_a, _ = model.forward_logp(batch, order)
    changed = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    changed["actions"] = batch["actions"].clone()
    changed["actions"][:, :, 2, 0] = (changed["actions"][:, :, 2, 0] + 1) % 4
    changed["actions"][:, :, 0, 0] = (changed["actions"][:, :, 0, 0] + 2) % 4
    logp_b, _ = model.forward_logp(changed, order)
    assert torch.allclose(logp_a[:, :, 0], logp_b[:, :, 0], atol=1e-6)


def test_autoregressive_conditioning_direction():
    """Position k conditions only on the actions of positions < k, at the same step.

    Changing the action at position p must leave every earlier position untouched
    (their inputs do not include it) and must move every later position (they
    condition on it).  Position p itself moves because its target changed.
    """
    model = _model()
    dataset = _FakeDataset()
    batch = _batch(dataset, [0, 1, 2])
    order = JointBehaviourModel.ascending_order(batch)
    logp_a, _ = model.forward_logp(batch, order)
    t = 2  # compare one step so previous-action inputs stay fixed
    n = batch["obs"].shape[2]
    for pos in range(n):
        changed = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        changed["actions"] = batch["actions"].clone()
        changed["actions"][:, t, pos, 0] = (changed["actions"][:, t, pos, 0] + 1) % 4
        logp_b, _ = model.forward_logp(changed, order)
        for q in range(n):
            same = torch.allclose(logp_a[:, t, q], logp_b[:, t, q], atol=1e-6)
            if q < pos:
                assert same, f"position {q} must not depend on position {pos}"
            else:
                assert not same, f"position {q} must depend on position {pos}"


def test_deviation_non_negative_and_masked():
    model = _model()
    dataset = _FakeDataset()
    batch = _batch(dataset, [0, 1, 2])
    dev = model.deviations(batch)
    mask = dev["mask"]
    assert torch.isfinite(dev["team"][mask]).all()
    assert torch.isnan(dev["team"][~mask]).all()
    assert (dev["team"][mask] >= -1e-4).all()
    finite = dev["per_agent"][mask]
    assert torch.isfinite(finite).all()
    assert (finite >= -1e-4).all()


def test_illegal_action_is_masked_out():
    model = _model()
    dataset = _FakeDataset()
    batch = _batch(dataset, [0])
    batch = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    batch["avail_actions"] = batch["avail_actions"].clone()
    batch["avail_actions"][:, :, :, 3] = False
    batch["actions"] = batch["actions"].clone()
    batch["actions"][:, :, :, 0] = 3  # illegal target
    order = JointBehaviourModel.ascending_order(batch)
    logp, _ = model.forward_logp(batch, order)
    mask = model._filled_mask(batch)
    assert (logp[mask] <= -1e8).all()


def test_save_load_roundtrip(tmp_path=None):
    model = _model()
    dataset = _FakeDataset()
    batch = _batch(dataset, [0, 1])
    dev_a = model.deviations(batch)
    path = os.path.join(tmp_path or "/tmp", "pv_bc_roundtrip.pt")
    save_behaviour_model(path, model, meta={"test": True})
    reloaded = load_behaviour_model(path, device="cpu")
    dev_b = reloaded.deviations(batch)
    assert torch.allclose(dev_a["team"], dev_b["team"], equal_nan=True, atol=1e-6)
    assert torch.allclose(dev_a["per_agent"], dev_b["per_agent"], equal_nan=True, atol=1e-6)


def test_split_episodes_deterministic_and_disjoint():
    a_train, a_val = split_episodes(50, 0.1, seed=3)
    b_train, b_val = split_episodes(50, 0.1, seed=3)
    assert a_train == b_train and a_val == b_val
    assert not set(a_train) & set(a_val)
    assert len(a_train) + len(a_val) == 50


def test_random_order_is_permutation():
    gen = torch.Generator().manual_seed(0)
    order = random_order(16, 5, gen)
    assert order.shape == (16, 5)
    for row in order:
        assert sorted(row.tolist()) == [0, 1, 2, 3, 4]


def test_fit_behaviour_model_reduces_loss():
    dataset = _FakeDataset(
        n_episodes=24, length=8, n_agents=3, n_actions=4, obs_dim=5, seed=1, teacher=True
    )
    train_idx, val_idx = split_episodes(dataset.n_episodes, 0.25, seed=0)
    model, report = fit_behaviour_model(
        dataset,
        train_indices=train_idx,
        val_indices=val_idx,
        n_agents=3,
        n_actions=4,
        obs_dim=5,
        hidden_dim=16,
        max_updates=60,
        batch_size=8,
        eval_every=10,
        patience=5,
        device="cpu",
        seed=0,
        log_every=1000,
    )
    assert report.updates > 0
    assert len(report.train_nll) >= 2
    # Training loss must fall, and on a learnable teacher the held-out NLL must
    # beat the uniform-over-legal reference.
    assert report.train_nll[-1] < report.train_nll[0]
    val_batch = _batch(dataset, val_idx)
    assert model.nll(val_batch) <= model.uniform_nll(val_batch) + 1e-6


def test_ensemble_novelty_non_negative():
    models = [_model(seed=s) for s in range(3)]
    dataset = _FakeDataset()
    batch = _batch(dataset, [0, 1])
    nov = NoveltyEnsemble(models).novelty(batch)
    mask = models[0]._filled_mask(batch)
    assert torch.isfinite(nov[mask]).all()
    assert (nov[mask] >= -1e-6).all()
