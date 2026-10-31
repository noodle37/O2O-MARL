#!/usr/bin/env python3
# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""CPU tests for MU-APR: controller, objective, history, meta-gradient.

The central test differentiates through K inner Adam steps and compares the
analytic derivative of M(z) with a central finite difference.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from muapr.controller import (
    MUAPRSettings,
    MetaLambdaController,
    build_muapr_controller,
    lambda_from_z,
)
from muapr.history import build_mixed_batch, mixed_sizes, split_recent_episodes
from muapr.meta import MetaGradientEngine
from muapr.objective import (
    joint_log_ratio,
    leave_one_out_baseline,
    mc_team_returns,
    validation_objective,
)
from ovmse.learner import PaperQMIXLearner
from sapr.regularizer import masked_policy

N_AGENTS = 2
N_ACTIONS = 2
OBS_DIM = 3
STATE_DIM = 4
SEQ = 4


# --------------------------------------------------------------------- helpers


def make_batch(
    *,
    b: int = 2,
    t: int = SEQ,
    n: int = N_AGENTS,
    a: int = N_ACTIONS,
    obs_dim: int = OBS_DIM,
    state_dim: int = STATE_DIM,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> dict:
    gen = torch.Generator().manual_seed(seed)
    obs = torch.randn(b, t, n, obs_dim, generator=gen, dtype=dtype)
    state = torch.randn(b, t, state_dim, generator=gen, dtype=dtype)
    actions = torch.randint(0, a, (b, t, n, 1), generator=gen)
    avail = torch.ones(b, t, n, a, dtype=torch.bool)
    reward = torch.randn(b, t, 1, generator=gen, dtype=dtype)
    terminated = torch.zeros(b, t, 1, dtype=dtype)
    terminated[:, -1, 0] = 1.0
    filled = torch.ones(b, t, 1, dtype=dtype)
    return {
        "obs": obs,
        "actions": actions,
        "state": state,
        "avail_actions": avail,
        "reward": reward,
        "terminated": terminated,
        "filled": filled,
    }


def make_episode(length: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    steps = []
    for i in range(length):
        steps.append(
            {
                "obs": rng.normal(size=(N_AGENTS, OBS_DIM)).astype(np.float32),
                "actions": rng.integers(0, N_ACTIONS, size=N_AGENTS).astype(np.int64),
                "state": rng.normal(size=(STATE_DIM,)).astype(np.float32),
                "reward": float(rng.normal()),
                "terminated": 1.0 if i == length - 1 else 0.0,
                "legals": np.ones((N_AGENTS, N_ACTIONS), dtype=bool),
            }
        )
    return {"steps": steps, "length": length}


class FakeOffline:
    def __init__(self, batch: dict, max_seq: int = SEQ):
        self.batch = batch
        self.stats = SimpleNamespace(max_seq_length=max_seq)

    def sample(self, batch_size, rng, device=None):
        n = self.batch["obs"].shape[0]
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        return {k: v.index_select(0, idx_t) for k, v in self.batch.items()}


def tiny_learner(dtype: torch.dtype = torch.float32) -> PaperQMIXLearner:
    torch.manual_seed(0)
    learner = PaperQMIXLearner(
        n_agents=N_AGENTS,
        n_actions=N_ACTIONS,
        obs_dim=OBS_DIM,
        state_dim=STATE_DIM,
        linear_layer_dim=8,
        recurrent_layer_dim=8,
        mixer_embed_dim=4,
        mixer_hyper_dim=8,
        learning_rate=1e-2,
        gamma=0.9,
        td_lambda=0.6,
        grad_norm_clip=0.0,
        target_update_interval=1000,
        batch_size_run=1,
        add_agent_id_to_obs=True,
        moderate_lambda=False,
        device="cpu",
    )
    learner.offline_q_network = copy.deepcopy(learner.q_network)
    learner.offline_mixer = copy.deepcopy(learner.mixer)
    if dtype != torch.float32:
        for module in (
            learner.q_network,
            learner.target_q_network,
            learner.mixer,
            learner.target_mixer,
            learner.offline_q_network,
            learner.offline_mixer,
        ):
            module.to(dtype)
    return learner


def tiny_settings(**overrides) -> MUAPRSettings:
    values = dict(
        enabled=True,
        lambda_ref=0.1,
        temperature=1.0,
        support_floor=1e-8,
        interval_steps=10,
        start_after_steps=0,
        inner_steps=2,
        inner_batch_size=4,
        kl_offline_frac=0.5,
        val_recent_episodes=6,
        val_episodes=2,
        ratio_clip=0.5,
        advantage_normalize=False,
        gamma=0.9,
        kappa=0.0,
        meta_lr=1e-3,
        meta_beta1=0.9,
        meta_beta2=0.999,
        meta_eps=1e-8,
        z_clip=2.0,
        inner_grad_clip=0.0,
        count_in_budget=False,
    )
    values.update(overrides)
    return MUAPRSettings(**values)


def meta_fixture(dtype: torch.dtype = torch.float64):
    learner = tiny_learner(dtype)
    offline_batch = make_batch(b=4, dtype=dtype, seed=11)
    offline = FakeOffline(offline_batch, max_seq=SEQ)
    inner_pool = [make_episode(SEQ, seed=100 + i) for i in range(6)]
    val_eps = [make_episode(SEQ, seed=200 + i) for i in range(2)]
    from training.rollout import stack_episodes

    val_batch = stack_episodes(val_eps, SEQ, torch.device("cpu"))
    if dtype != torch.float32:
        val_batch = {k: v.to(dtype) if v.is_floating_point() else v for k, v in val_batch.items()}
    rng = np.random.default_rng(7)
    inner_batches = [
        build_mixed_batch(
            offline,
            inner_pool,
            rng,
            batch_size=4,
            offline_frac=0.5,
            max_seq_length=SEQ,
            device=torch.device("cpu"),
        )[:2]
        for _ in range(2)
    ]
    if dtype != torch.float32:
        inner_batches = [
            (
                {k: v.to(dtype) if v.is_floating_point() else v for k, v in batch.items()},
                mask,
            )
            for batch, mask in inner_batches
        ]
    return learner, offline, inner_pool, val_batch, inner_batches


# ------------------------------------------------------------------- settings


def test_settings_and_mutual_exclusion():
    settings = MUAPRSettings.from_cfg({"muapr": {"enabled": True}})
    assert settings.enabled and settings.lambda_ref == 0.1

    try:
        MUAPRSettings.from_cfg({"muapr": {"enabled": True, "kl_offline_frac": 1.0}})
    except ValueError:
        pass
    else:
        raise AssertionError("kl_offline_frac=1.0 must be rejected")

    try:
        build_muapr_controller(
            {"muapr": {"enabled": True}, "sapr": {"enabled": True}}
        )
    except ValueError:
        pass
    else:
        raise AssertionError("muapr and sapr must be mutually exclusive")

    assert build_muapr_controller({"muapr": {"enabled": False}}) is None


def test_lambda_mapping_and_clip():
    s = tiny_settings(z_clip=2.0)
    assert math.isclose(lambda_from_z(s, 0.0), 0.1)
    assert math.isclose(lambda_from_z(s, 1.0), 0.1 * math.exp(1.0))
    assert math.isclose(lambda_from_z(s, 5.0), 0.1 * math.exp(2.0))
    assert math.isclose(lambda_from_z(s, -5.0), 0.1 * math.exp(-2.0))


def test_controller_adam_matches_torch():
    s = tiny_settings(meta_lr=0.05, meta_beta1=0.9, meta_beta2=0.999, z_clip=10.0)
    ctrl = MetaLambdaController(s)
    z_ref = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([z_ref], lr=s.meta_lr, betas=(s.meta_beta1, s.meta_beta2), eps=s.meta_eps)
    for step in range(5):
        z = torch.tensor(ctrl.z, dtype=torch.float64, requires_grad=True)
        loss = 0.5 * (z - 3.0) ** 2
        grad = torch.autograd.grad(loss, z)[0]
        ctrl.update(float(grad.item()), t_env=step)
        opt.zero_grad()
        loss_ref = 0.5 * (z_ref - 3.0) ** 2
        loss_ref.backward()
        opt.step()
        assert abs(ctrl.z - float(z_ref.item())) < 1e-8
    assert ctrl.adam_step == 5


def test_controller_state_roundtrip_and_no_update():
    s = tiny_settings()
    ctrl = MetaLambdaController(s)
    ctrl.update(0.3, t_env=10, meta_loss=1.0, kappa_penalty=0.1, J_val=0.2)
    ctrl.update(-0.2, t_env=20)
    state = ctrl.state_dict()

    restored = MetaLambdaController(s)
    restored.load_state_dict(state)
    assert math.isclose(restored.z, ctrl.z)
    assert math.isclose(restored.m, ctrl.m)
    assert math.isclose(restored.v, ctrl.v)
    assert restored.adam_step == ctrl.adam_step
    assert restored.version == ctrl.version
    assert restored.last_meta_t == ctrl.last_meta_t

    z_before = restored.z
    logs = restored.no_update("insufficient_buffer", t_env=25)
    assert logs["muapr/updated"] == 0.0
    assert math.isclose(restored.z, z_before)
    assert restored.last_meta_t == 25


def test_due_respects_start_and_interval():
    s = tiny_settings(interval_steps=10, start_after_steps=50)
    ctrl = MetaLambdaController(s)
    assert not ctrl.due(49)
    assert ctrl.due(50)
    ctrl.no_update("x", t_env=50)
    assert not ctrl.due(59)
    assert ctrl.due(60)


# ------------------------------------------------------------------ objective


def test_mc_team_returns_hand_computed():
    reward = torch.tensor([[[1.0], [2.0], [9.0], [9.0]]])
    terminated = torch.tensor([[[0.0], [1.0], [1.0], [1.0]]])
    filled = torch.tensor([[[1.0], [1.0], [0.0], [0.0]]])
    returns = mc_team_returns(reward, terminated, filled, gamma=0.9)
    assert math.isclose(float(returns[0, 0]), 1.0 + 0.9 * 2.0, rel_tol=1e-6)
    assert math.isclose(float(returns[0, 1]), 2.0, rel_tol=1e-6)
    assert float(returns[0, 2]) == 0.0 and float(returns[0, 3]) == 0.0


def test_loo_baseline_hand_computed():
    q_i = torch.tensor([[[1.0, 2.0]]])
    q_tot = torch.tensor([[[5.0]]])
    baseline = leave_one_out_baseline(q_i, q_tot)
    assert torch.allclose(baseline, torch.tensor([[[4.0, 3.0]]]))


def test_joint_ratio_and_clipping_hand_computed():
    q_ref = torch.tensor([[[[0.0, 1.0]]]])  # one agent, two actions
    q_theta = torch.tensor([[[[1.0, 0.0]]]])
    avail = torch.ones(1, 1, 1, 2, dtype=torch.bool)
    actions = torch.tensor([[[[0]]]])
    log_ratio, agent_valid = joint_log_ratio(q_theta, q_ref, avail, actions, 1.0, 0.0)
    p_theta = masked_policy(q_theta, avail, 1.0, 0.0)[0, 0, 0, 0]
    p_ref = masked_policy(q_ref, avail, 1.0, 0.0)[0, 0, 0, 0]
    expected = float(torch.log(p_theta) - torch.log(p_ref))
    assert math.isclose(float(log_ratio[0, 0]), expected, rel_tol=1e-6)
    assert bool(agent_valid[0, 0])

    # Two agents multiply their ratios.
    q_ref2 = q_ref.expand(1, 1, 2, 2)
    q_theta2 = q_theta.expand(1, 1, 2, 2)
    avail2 = avail.expand(1, 1, 2, 2)
    actions2 = actions.expand(1, 1, 2, 1)
    log_ratio2, _ = joint_log_ratio(q_theta2, q_ref2, avail2, actions2, 1.0, 0.0)
    assert math.isclose(float(log_ratio2[0, 0]), 2.0 * expected, rel_tol=1e-6)


def test_validation_objective_hand_computed():
    batch = make_batch(b=1, t=2)
    q = torch.zeros(1, 2, N_AGENTS, N_ACTIONS)
    q[..., 1] = 1.0  # action 0 has Q=0, action 1 has Q=1
    actions = torch.zeros(1, 2, N_AGENTS, 1, dtype=torch.long)
    q_tot = torch.ones(1, 2, 1)
    returns = torch.tensor([[1.0, 2.0]])
    filled = torch.ones(1, 2, 1)
    J, stats = validation_objective(
        q,
        q,
        q_tot,
        batch["avail_actions"],
        actions,
        returns,
        filled,
        temperature=1.0,
        support_floor=0.0,
        ratio_clip=0.2,
        advantage_normalize=False,
    )
    # q_i = 0, baseline = 1 - 0 = 1, A = [0, 1]; mean over 2 steps x 2 agents = 0.5.
    assert math.isclose(float(J), 0.5, rel_tol=1e-6)
    assert math.isclose(stats["muapr/ratio_mean"], 1.0, rel_tol=1e-6)
    assert stats["muapr/clip_frac"] == 0.0

    # A changed reference policy raises the ratio above one for the chosen action.
    q_ref = torch.zeros_like(q)
    q_ref[..., 0] = 1.0
    _, stats2 = validation_objective(
        q,
        q_ref,
        q_tot,
        batch["avail_actions"],
        actions,
        returns,
        filled,
        temperature=1.0,
        support_floor=0.0,
        ratio_clip=0.5,
        advantage_normalize=False,
    )
    p = masked_policy(q, batch["avail_actions"], 1.0, 0.0)[0, 0, 0, 0]
    p_ref = masked_policy(q_ref, batch["avail_actions"], 1.0, 0.0)[0, 0, 0, 0]
    expected_ratio = float((p / p_ref) ** N_AGENTS)
    assert math.isclose(stats2["muapr/ratio_mean"], expected_ratio, rel_tol=1e-6)


# -------------------------------------------------------------------- history


def test_mixed_sizes_and_split_disjoint():
    assert mixed_sizes(32, 0.5) == (16, 16)
    assert mixed_sizes(3, 0.5) == (2, 1)
    assert mixed_sizes(2, 0.5) == (1, 1)

    buffer = [make_episode(SEQ, seed=i) for i in range(10)]
    split = split_recent_episodes(buffer, recent_n=6, val_n=2, rng=np.random.default_rng(0))
    assert split is not None
    assert len(split["validation"]) == 2
    assert len(split["inner"]) == 4
    assert not (set(map(id, split["validation"])) & set(map(id, split["inner"])))
    assert split_recent_episodes([make_episode(SEQ, 0)], recent_n=6, val_n=2, rng=np.random.default_rng(0)) is None


# --------------------------------------------------------------- meta-gradient


def test_meta_gradient_matches_finite_difference():
    learner, offline, _, val_batch, inner_batches = meta_fixture(torch.float64)
    engine = MetaGradientEngine(learner, tiny_settings(), offline)
    z0 = 0.1

    result = engine.run_inner_and_meta(
        z0, inner_batches, val_batch, create_graph=True
    )
    assert result["meta_grad"] is not None
    assert math.isfinite(result["meta_grad"])
    # No ratio clipping: keep the objective smooth for the finite difference.
    assert result["stats"]["muapr/clip_frac"] == 0.0

    h = 1e-3
    plus = engine.run_inner_and_meta(
        z0 + h, inner_batches, val_batch, create_graph=False
    )["meta_loss"]
    minus = engine.run_inner_and_meta(
        z0 - h, inner_batches, val_batch, create_graph=False
    )["meta_loss"]
    fd = (plus - minus) / (2.0 * h)
    assert abs(fd) > 1e-8, "finite difference is identically zero; test is vacuous"
    assert abs(fd - result["meta_grad"]) <= 2e-4 * max(1.0, abs(fd))


def test_meta_gradient_depends_on_optimizer_state():
    learner, offline, _, val_batch, inner_batches = meta_fixture(torch.float32)
    engine = MetaGradientEngine(learner, tiny_settings(), offline)
    z0 = 0.1
    grad_plain = engine.run_inner_and_meta(
        z0, inner_batches, val_batch, create_graph=True
    )["meta_grad"]

    for param in list(learner.q_network.parameters()) + list(learner.mixer.parameters()):
        state = learner.optimizer.state[param]
        state["step"] = torch.tensor(7.0)
        state["exp_avg"] = torch.full_like(param, 0.2)
        state["exp_avg_sq"] = torch.full_like(param, 0.7)

    grad_state = engine.run_inner_and_meta(
        z0, inner_batches, val_batch, create_graph=True
    )["meta_grad"]
    assert math.isfinite(grad_state)
    assert abs(grad_state - grad_plain) > 1e-9


def test_compute_meta_gradient_rejects_small_buffer():
    learner, offline, _, _, _ = meta_fixture(torch.float32)
    engine = MetaGradientEngine(learner, tiny_settings(), offline)
    result = engine.compute_meta_gradient(
        [make_episode(SEQ, 0)], np.random.default_rng(0), t_env=0, z_value=0.0
    )
    assert result["valid"] is False
    assert result["reason"] in ("insufficient_buffer", "inner_pool_too_small")


def test_inner_loss_ignores_offline_rows_for_td():
    learner, offline, _, _, inner_batches = meta_fixture(torch.float32)
    engine = MetaGradientEngine(learner, tiny_settings(), offline)
    batch, row_mask = inner_batches[0]
    params = engine._snapshot_params()
    lam = torch.tensor(0.0, device=torch.device("cpu"))
    loss, _ = engine._inner_loss(params, batch, row_mask, lam)
    only_online = row_mask.view(-1, 1, 1)
    # Recompute the TD error using only online rows and compare to the loss the
    # engine reports (KL is disabled via lam=0).
    with torch.no_grad():
        target_mac = learner._forward_qs(batch, target=True)
    mac = engine._functional_qs(params, batch)
    chosen = torch.gather(mac, 3, batch["actions"].long())
    mac_sel = torch.where(
        batch["avail_actions"], mac, torch.tensor(-9999999.0)
    )
    max_actions = torch.argmax(mac_sel, dim=3, keepdim=True)
    target_max = torch.gather(target_mac, 3, max_actions)
    chosen_tot = engine._functional_mix(params, chosen, batch["state"])
    with torch.no_grad():
        target_tot = learner.target_mixer(target_max, batch["state"])
    from ovmse.targets import build_td_lambda_targets

    term = batch["terminated"][:, :-1].float()
    mask = batch["filled"][:, :-1].float()
    mask[:, 1:] = mask[:, 1:] * (1.0 - term[:, :-1])
    td_targets = build_td_lambda_targets(
        batch["reward"][:, :-1],
        batch["terminated"][:, :-1].max(dim=2, keepdim=True)[0],
        mask,
        target_tot,
        learner.gamma,
        learner.td_lambda,
    ).detach()
    td_err = 0.5 * (chosen_tot[:, :-1] - td_targets) ** 2
    td_mask = mask * only_online
    expected = (td_err * td_mask).sum() / td_mask.sum()
    assert math.isclose(float(loss.detach()), float(expected.detach()), rel_tol=1e-6)
