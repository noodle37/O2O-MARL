"""Unit tests for the MAC-Flow per-agent flow stack (repo `discrete_macflow` style)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from macflow.flow import euler_integrate, flow_bc_loss
from macflow.learner import MACFlowLearner
from macflow.networks import add_agent_id, mix_mean


def test_euler_constant_velocity():
    def vfn(pair, x, t):
        return torch.ones_like(x)

    x0 = torch.zeros(4, 6)
    pair = torch.zeros(4, 3)
    out = euler_integrate(vfn, pair, x0, steps=10)
    assert torch.allclose(out, torch.ones_like(out))


def test_flow_bc_loss_mask_excludes_rows():
    pair = torch.zeros(8, 4)
    x1 = torch.zeros(8, 3)
    x1[4:] = 1e6  # masked-out rows would dominate the loss if included
    mask = torch.tensor([True] * 4 + [False] * 4)

    def vfn(pair, x, t):
        return torch.zeros_like(x)

    loss = float(flow_bc_loss(vfn, pair, x1, mask=mask))
    assert loss < 1e3


def test_add_agent_id_prepends_onehot():
    obs = torch.randn(2, 3, 5)
    pair = add_agent_id(obs)
    assert pair.shape == (2, 3, 8)
    eye = torch.eye(3)
    for i in range(3):
        assert torch.allclose(pair[..., i, :3], eye[i].expand(2, 3))


def test_mix_mean():
    q = torch.tensor([[[1.0], [3.0]], [[2.0], [4.0]]])
    assert torch.allclose(mix_mean(q), torch.tensor([[2.0], [3.0]]))


def _make_batch(b, s, n, o, fill_lens, n_actions=4):
    obs = torch.randn(b, s, n, o)
    actions = torch.randint(0, n_actions, (b, s, n, 1))
    filled = torch.zeros(b, s, 1)
    term = torch.zeros(b, s, 1)
    for i, ln in enumerate(fill_lens):
        filled[i, :ln, 0] = 1.0
        term[i, ln - 1, 0] = 1.0
        term[i, ln, 0] = 1.0
        obs[i, ln] = obs[i, ln - 1]
    return {
        "obs": obs,
        "actions": actions,
        "avail_actions": torch.ones(b, s, n, n_actions, dtype=torch.bool),
        "reward": torch.randn(b, s, 1),
        "terminated": term,
        "filled": filled,
    }


def test_window_sampling_respects_episode_bounds():
    learner = MACFlowLearner(
        n_agents=2, obs_dim=3, n_actions=4, action_mode="discrete", device="cpu",
        sequence_length=20,
    )
    batch = _make_batch(4, 21, 2, 3, [20, 5, 2, 17])
    wb = learner._window_batch(batch)
    assert wb["obs"].shape == (4, 20, 2, 3)
    for i, ln in enumerate([20, 5, 2, 17]):
        valid = wb["filled"][i] > 0.5
        # window covers real steps only, in order
        assert valid.sum() == min(20, ln)
        if ln >= 20:
            assert bool(valid.all())


def test_train_on_batch_discrete_smoke():
    learner = MACFlowLearner(
        n_agents=2, obs_dim=3, n_actions=4, action_mode="discrete", device="cpu",
        sequence_length=10,
    )
    logs = learner.train_on_batch(_make_batch(4, 21, 2, 3, [20, 18, 6, 9]), t_env=0)
    for key in ("loss_flow", "td_loss", "distill_loss", "q_loss", "q_tot_mean", "loss"):
        assert key in logs and np.isfinite(logs[key])
    assert logs["loss_flow"] >= 0.0 and logs["td_loss"] >= 0.0 and logs["distill_loss"] >= 0.0


def test_discrete_q_term_is_inert_for_policy():
    """Repo semantics: BC trains the flow, CE trains the policy, TD/q-loss touch the critic."""
    learner = MACFlowLearner(
        n_agents=2, obs_dim=3, n_actions=4, action_mode="discrete", device="cpu",
        sequence_length=10, alpha_distill=0.0,
    )
    learner.train_on_batch(_make_batch(4, 11, 2, 3, [10, 5, 2, 7]), t_env=0)

    def gnorm(mod):
        return sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)

    assert gnorm(learner.flow) > 0.0
    assert gnorm(learner.critic) > 0.0
    assert gnorm(learner.policy) == 0.0  # only the CE distillation can reach the policy


def test_distill_reaches_policy():
    learner = MACFlowLearner(
        n_agents=2, obs_dim=3, n_actions=4, action_mode="discrete", device="cpu",
        sequence_length=10, alpha_distill=3.0,
    )
    learner.train_on_batch(_make_batch(4, 11, 2, 3, [10, 5, 2, 7]), t_env=0)
    grad = sum(
        float(p.grad.abs().sum()) for p in learner.policy.parameters() if p.grad is not None
    )
    assert grad > 0.0


def test_target_polyak_updates():
    learner = MACFlowLearner(
        n_agents=2, obs_dim=3, n_actions=4, action_mode="discrete", device="cpu",
        sequence_length=10,
    )
    before = [p.clone() for p in learner.target_critic.parameters()]
    learner.train_on_batch(_make_batch(4, 11, 2, 3, [10, 5, 2, 7]), t_env=0)
    moved = sum(
        float((a - b).abs().sum())
        for a, b in zip(learner.target_critic.parameters(), before)
    )
    assert moved > 0.0


def test_eval_actions_are_legal():
    learner = MACFlowLearner(
        n_agents=2, obs_dim=4, n_actions=5, action_mode="discrete", device="cpu",
        sequence_length=10,
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        legals = {
            "agent_0": rng.random(5) > 0.5,
            "agent_1": rng.random(5) > 0.5,
        }
        for key in legals:
            if not legals[key].any():
                legals[key][int(rng.integers(0, 5))] = True
        obs = {a: rng.standard_normal(4).astype(np.float32) for a in legals}
        trace = {}
        actions = learner.select_actions_eval(obs, legals, 0.0, trace=trace)
        for a in legals:
            assert bool(legals[a][int(actions[a])])
        assert np.isfinite(trace["q_tot"])


def test_export_load_roundtrip():
    learner = MACFlowLearner(
        n_agents=2, obs_dim=4, n_actions=5, action_mode="discrete", device="cpu",
        sequence_length=10,
    )
    learner.train_on_batch(_make_batch(4, 11, 2, 4, [10, 5, 2, 7], n_actions=5), t_env=0)
    clone = MACFlowLearner(
        n_agents=2, obs_dim=4, n_actions=5, action_mode="discrete", device="cpu",
        sequence_length=10,
    )
    clone.load_state(learner.export_state())
    for p, q in zip(learner.flow.parameters(), clone.flow.parameters()):
        assert torch.allclose(p, q)
    for p, q in zip(learner.critic.parameters(), clone.critic.parameters()):
        assert torch.allclose(p, q)
    for p, q in zip(learner.policy.parameters(), clone.policy.parameters()):
        assert torch.allclose(p, q)


def test_continuous_mode_bounds_and_smoke():
    learner = MACFlowLearner(
        n_agents=2, obs_dim=3, act_dim=2, action_mode="continuous", device="cpu",
        sequence_length=8,
    )
    batch = _make_batch(4, 9, 2, 3, [8, 5, 2, 7])
    batch["actions"] = torch.rand(4, 9, 2, 2) * 2 - 1
    logs = learner.train_on_batch(batch, t_env=0)
    assert all(np.isfinite(v) for v in logs.values())
    obs = {"a0": np.zeros(3, dtype=np.float32), "a1": np.zeros(3, dtype=np.float32)}
    acts = learner.select_actions_eval(obs, None, 0.0)
    for v in acts.values():
        assert np.abs(v).max() <= 1.0 + 1e-6


def test_agent_id_in_flow_and_policy_paths():
    """Pair dim must be obs+n_agents everywhere (flow, policy, critic)."""
    learner = MACFlowLearner(
        n_agents=3, obs_dim=6, n_actions=4, action_mode="discrete", device="cpu",
        sequence_length=6,
    )
    batch = _make_batch(2, 7, 3, 6, [6, 4])
    logs = learner.train_on_batch(batch, t_env=0)
    assert logs["loss_flow"] >= 0.0
    # one-hot action x1 built from indices keeps CE shapes consistent
    assert learner.pair_dim == 9


if __name__ == "__main__":
    test_euler_constant_velocity()
    test_flow_bc_loss_mask_excludes_rows()
    test_add_agent_id_prepends_onehot()
    test_mix_mean()
    test_window_sampling_respects_episode_bounds()
    test_train_on_batch_discrete_smoke()
    test_discrete_q_term_is_inert_for_policy()
    test_distill_reaches_policy()
    test_target_polyak_updates()
    test_eval_actions_are_legal()
    test_export_load_roundtrip()
    test_continuous_mode_bounds_and_smoke()
    test_agent_id_in_flow_and_policy_paths()
    print("macflow tests ok")
