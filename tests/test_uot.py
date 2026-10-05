#!/usr/bin/env python3
"""UOT: recorded neighborhood, ordinal hinge, frozen oracle."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from uot.hinge import ordinal_hinge
from uot.learner import UOTLearner
from uot.neighborhood import build_recorded_neighborhood
from uot.support import RecordedJoints


def _code(n_actions: int, joint) -> int:
    joint = np.asarray(joint, dtype=np.int64)
    mults = n_actions ** np.arange(joint.shape[0], dtype=np.int64)
    return int(joint @ mults)


def _support(n_agents: int, n_actions: int, joints) -> RecordedJoints:
    return RecordedJoints(
        n_agents, n_actions, np.array([_code(n_actions, u) for u in joints], dtype=np.int64)
    )


def _joint_set(cand, valid, b=0, t=0):
    out = set()
    for k in range(cand.shape[0]):
        if bool(valid[k, b, t]):
            out.add(tuple(int(x) for x in cand[k, b, t].tolist()))
    return out


def test_recorded_hamming1_includes_any_agent():
    n_agents, n_actions = 3, 5
    support = _support(n_agents, n_actions, [[0, 0, 0], [0, 0, 1]])
    b, t = 2, 3
    live_q = torch.zeros(b, t, n_agents, n_actions)
    live_q[..., 0] = 4.0
    avail = torch.ones(b, t, n_agents, n_actions, dtype=torch.bool)
    executed = torch.zeros(b, t, n_agents, dtype=torch.long)
    cand, valid = build_recorded_neighborhood(
        live_q, live_q.clone(), avail, executed, support, k=8, max_hamming=2
    )
    have = _joint_set(cand, valid)
    assert (0, 0, 0) in have
    assert (0, 0, 1) in have
    assert have <= {(0, 0, 0), (0, 0, 1)}


def test_unrecorded_flip_is_absent():
    n_agents, n_actions = 3, 5
    support = _support(n_agents, n_actions, [[0, 0, 0]])
    b, t = 2, 3
    live_q = torch.zeros(b, t, n_agents, n_actions)
    live_q[..., 0] = 4.0
    avail = torch.ones(b, t, n_agents, n_actions, dtype=torch.bool)
    executed = torch.zeros(b, t, n_agents, dtype=torch.long)
    cand, valid = build_recorded_neighborhood(
        live_q, live_q.clone(), avail, executed, support, k=8, max_hamming=2
    )
    assert _joint_set(cand, valid) == {(0, 0, 0)}


def test_hinge_zero_when_live_order_already_holds():
    q_bar = torch.tensor([1.0, 0.0]).view(2, 1, 1, 1)
    q_live = torch.tensor([1.0, 0.0]).view(2, 1, 1, 1)
    cand = torch.tensor([[[0]], [[1]]])
    valid = torch.ones(2, 1, 1, dtype=torch.bool)
    mask = torch.ones(1, 1, dtype=torch.bool)
    loss, stats = ordinal_hinge(q_live, q_bar, cand, valid, mask, delta=0.1, margin=0.1)
    assert float(loss) == 0.0
    assert stats["uot_n_pairs"] == 1.0
    assert stats["uot_hinge_rate"] == 0.0
    assert stats["uot_sign_disagree"] == 0.0


def test_hinge_active_when_live_order_flips():
    q_bar = torch.tensor([1.0, 0.0]).view(2, 1, 1, 1)
    q_live = torch.tensor([0.0, 1.0]).view(2, 1, 1, 1)
    cand = torch.tensor([[[0]], [[1]]])
    valid = torch.ones(2, 1, 1, dtype=torch.bool)
    mask = torch.ones(1, 1, dtype=torch.bool)
    loss, stats = ordinal_hinge(q_live, q_bar, cand, valid, mask, delta=0.1, margin=0.1)
    assert abs(float(loss) - 1.1) < 1e-5
    assert stats["uot_hinge_rate"] == 1.0
    assert stats["uot_sign_disagree"] == 1.0


def test_near_tie_pair_is_dropped():
    q_bar = torch.tensor([0.05, 0.0]).view(2, 1, 1, 1)
    q_live = torch.tensor([0.0, 1.0]).view(2, 1, 1, 1)
    cand = torch.tensor([[[0]], [[1]]])
    valid = torch.ones(2, 1, 1, dtype=torch.bool)
    mask = torch.ones(1, 1, dtype=torch.bool)
    loss, stats = ordinal_hinge(q_live, q_bar, cand, valid, mask, delta=0.1, margin=0.1)
    assert float(loss) == 0.0
    assert stats["uot_n_pairs"] == 0.0


def _batch(n_agents=3, n_actions=4, seq=4):
    b = 2
    obs = torch.randn(b, seq, n_agents, 6)
    actions = torch.zeros(b, seq, n_agents, 1, dtype=torch.long)
    state = torch.randn(b, seq, 5)
    avail = torch.ones(b, seq, n_agents, n_actions, dtype=torch.bool)
    reward = torch.ones(b, seq, 1)
    term = torch.zeros(b, seq, 1)
    term[:, -2] = 1.0
    filled = torch.ones(b, seq, 1)
    filled[:, -1] = 0.0
    return {
        "obs": obs,
        "actions": actions,
        "state": state,
        "avail_actions": avail,
        "reward": reward,
        "terminated": term,
        "filled": filled,
    }


def _learner(joints) -> UOTLearner:
    n_agents, n_actions = 3, 4
    support = _support(n_agents, n_actions, joints)
    torch.manual_seed(0)
    return UOTLearner(
        n_agents=n_agents,
        n_actions=n_actions,
        obs_dim=6,
        state_dim=5,
        support=support,
        recurrent_layer_dim=8,
        mixer_embed_dim=4,
        mixer_hyper_dim=8,
        device="cpu",
    )


def test_empty_support_has_no_ordinal_term():
    learner = _learner([])
    learner.freeze_oracle()
    frozen = learner.frozen_q_network.fc2.bias.detach().clone()
    logs = learner.train_on_batch(_batch())
    assert logs["loss_ord"] == 0.0
    assert logs["uot_n_pairs"] == 0.0
    assert torch.equal(frozen, learner.frozen_q_network.fc2.bias)


def test_oracle_does_not_receive_gradients():
    learner = _learner([[0, 0, 0], [0, 0, 1]])
    learner.freeze_oracle()
    with torch.no_grad():
        learner.frozen_q_network.fc2.bias.zero_()
        learner.frozen_q_network.fc2.bias[1] = 5.0
        learner.q_network.fc2.weight.zero_()
        learner.q_network.fc2.bias.zero_()
        learner.q_network.fc2.bias[0] = 8.0
    frozen_bias = learner.frozen_q_network.fc2.bias.detach().clone()
    frozen_mix = next(learner.frozen_mixer.parameters()).detach().clone()
    live_mix = next(learner.mixer.parameters()).detach().clone()
    logs = learner.train_on_batch(_batch())
    assert logs["loss_ord"] > 0.0
    assert logs["uot_n_pairs"] > 0.0
    assert torch.equal(frozen_bias, learner.frozen_q_network.fc2.bias)
    assert torch.equal(frozen_mix, next(learner.frozen_mixer.parameters()).detach())
    assert not torch.equal(live_mix, next(learner.mixer.parameters()).detach())
    for param in learner.frozen_mixer.parameters():
        assert not param.requires_grad


def test_support_from_filled_episodes_ignores_padding():
    actions = torch.zeros(1, 3, 2, 1, dtype=torch.long)
    actions[0, 0, 1, 0] = 1
    actions[0, 2, :, 0] = 3
    filled = torch.tensor([[[1.0], [1.0], [0.0]]])
    support = RecordedJoints.from_episode_actions(actions, filled, n_actions=4)
    assert len(support) == 2
    joints = torch.tensor([[0, 1], [3, 3], [0, 0]])
    inside = support.contains(joints)
    assert bool(inside[0]) and not bool(inside[1]) and bool(inside[2])


if __name__ == "__main__":
    test_recorded_hamming1_includes_any_agent()
    test_unrecorded_flip_is_absent()
    test_hinge_zero_when_live_order_already_holds()
    test_hinge_active_when_live_order_flips()
    test_near_tie_pair_is_dropped()
    test_empty_support_has_no_ordinal_term()
    test_oracle_does_not_receive_gradients()
    test_support_from_filled_episodes_ignores_padding()
    print("uot tests ok")
