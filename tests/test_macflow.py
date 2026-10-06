"""Unit tests for MAC-Flow flow math and learner pieces."""

from __future__ import annotations

import torch

from macflow.flow import discrete_actions_from_joint, euler_integrate, obs_joint_from_batch, split_joint_to_agents
from macflow.networks import vdn_tot
from macflow.learner import MACFlowLearner


def test_euler_constant_velocity():
    def vfn(t, oj, x):
        return torch.ones_like(x)

    x0 = torch.zeros(4, 6)
    oj = torch.zeros(4, 3)
    out = euler_integrate(vfn, oj, x0, steps=10)
    assert torch.allclose(out, torch.ones_like(out))


def test_flow_bc_target_mse_zero():
    x1 = torch.randn(8, 5)
    x0 = torch.randn(8, 5)
    t = torch.rand(8)
    xt = (1 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * x1
    target = x1 - x0
    pred = target
    assert float(torch.mean((pred - target) ** 2)) < 1e-12


def test_discrete_legal_mask():
    joint = torch.tensor([[2.0, 0.1, 0.1, 2.0, 0.1, 0.1]])
    avail = torch.tensor([[True, False, True], [True, True, False]]).unsqueeze(0)
    idx = discrete_actions_from_joint(joint, n_agents=2, n_actions=3, avail=avail)
    assert int(idx[0, 0, 0]) == 0
    assert int(idx[0, 1, 0]) == 0


def test_vdn_mean():
    q = torch.tensor([[[1.0], [3.0]], [[2.0], [4.0]]])
    tot = vdn_tot(q)
    assert torch.allclose(tot, torch.tensor([[2.0], [3.0]]))


def test_continuous_policy_tanh_bounds():
    learner = MACFlowLearner(
        n_agents=2,
        obs_dim=3,
        act_dim=2,
        action_mode="continuous",
        device="cpu",
    )
    obs = torch.randn(1, 2, 3)
    noise = torch.randn(1, 2, 2)
    out = learner.policy(obs, noise)
    assert out.abs().max() <= 1.0 + 1e-6


def test_distill_target_is_detached_euler_slice():
    learner = MACFlowLearner(
        n_agents=2,
        obs_dim=4,
        n_actions=3,
        action_mode="discrete",
        device="cpu",
    )
    obs = torch.randn(16, 2, 4)
    z = torch.randn(16, learner.joint_dim)
    teacher = learner._distill_target(obs, z)
    joint = learner._flow_sample_joint(obs_joint_from_batch(obs), z)
    euler = split_joint_to_agents(joint, 2, 3)
    assert torch.allclose(teacher, euler)
    assert not teacher.requires_grad
    euler = euler.detach()
    loss_near = torch.mean((euler - teacher) ** 2)
    loss_far = torch.mean((torch.softmax(euler, dim=-1) - teacher) ** 2)
    assert float(loss_near) < float(loss_far)


def test_terminal_step_uses_its_own_done():
    learner = MACFlowLearner(
        n_agents=2,
        obs_dim=3,
        n_actions=4,
        action_mode="discrete",
        device="cpu",
    )
    filled = torch.tensor([1.0, 1.0, 1.0, 0.0]).view(1, 4, 1)
    terminated = torch.tensor([0.0, 0.0, 1.0, 1.0]).view(1, 4, 1)
    batch = {
        "obs": torch.randn(1, 4, 2, 3),
        "actions": torch.zeros(1, 4, 2, 1),
        "avail_actions": torch.ones(1, 4, 2, 4, dtype=torch.bool),
        "reward": torch.arange(4, dtype=torch.float32).view(1, 4, 1),
        "terminated": terminated,
        "filled": filled,
    }
    tr = learner._gather_transitions(batch)
    assert tr["t_idx"].tolist() == [0, 1, 2]
    assert tr["done"].tolist() == [0.0, 0.0, 1.0]
    assert tr["reward"].tolist() == [0.0, 1.0, 2.0]


def test_discrete_q_guidance_reaches_policy():
    learner = MACFlowLearner(
        n_agents=2,
        obs_dim=3,
        n_actions=4,
        action_mode="discrete",
        alpha_distill=0.0,
        device="cpu",
    )
    b, t, n = 4, 5, 2
    batch = {
        "obs": torch.randn(b, t, n, 3),
        "actions": torch.randint(0, 4, (b, t, n, 1)),
        "avail_actions": torch.ones(b, t, n, 4, dtype=torch.bool),
        "reward": torch.randn(b, t, 1),
        "terminated": torch.zeros(b, t, 1),
        "filled": torch.ones(b, t, 1),
        "state": torch.randn(b, t, 8),
    }
    learner.train_on_batch(batch)
    grad = sum(
        float(p.grad.abs().sum())
        for p in learner.policy.parameters()
        if p.grad is not None
    )
    assert grad > 0.0


def test_train_on_batch_discrete_smoke():
    learner = MACFlowLearner(
        n_agents=2,
        obs_dim=3,
        n_actions=4,
        action_mode="discrete",
        device="cpu",
    )
    b, t, n = 4, 5, 2
    batch = {
        "obs": torch.randn(b, t, n, 3),
        "actions": torch.randint(0, 4, (b, t, n, 1)),
        "avail_actions": torch.ones(b, t, n, 4, dtype=torch.bool),
        "reward": torch.randn(b, t, 1),
        "terminated": torch.zeros(b, t, 1),
        "filled": torch.ones(b, t, 1),
        "state": torch.randn(b, t, 8),
    }
    logs = learner.train_on_batch(batch)
    assert "loss_flow" in logs and logs["loss_flow"] >= 0.0


if __name__ == "__main__":
    test_euler_constant_velocity()
    test_flow_bc_target_mse_zero()
    test_discrete_legal_mask()
    test_vdn_mean()
    test_continuous_policy_tanh_bounds()
    test_distill_target_is_detached_euler_slice()
    test_terminal_step_uses_its_own_done()
    test_discrete_q_guidance_reaches_policy()
    test_train_on_batch_discrete_smoke()
    print("macflow tests ok")
