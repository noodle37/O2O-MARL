"""Transfer ablations preserve or break greedy argmax as intended."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.discrete_mujoco_actions import (
    continuous_to_index,
    index_to_continuous,
    n_discrete_actions,
)
from qmix.networks import MarlcRNN, QMixer
from ovmse.transfer import (
    affine_rescale_fc2_preserving_argmax,
    apply_transfer,
    permute_fc2_outputs,
)


def _rnn_cpu() -> MarlcRNN:
    torch.manual_seed(0)
    return MarlcRNN(4, 8, 5)


def test_affine_preserves_argmax():
    net = _rnn_cpu()
    h = torch.randn(3, net.hidden_dim)
    with torch.no_grad():
        before = net.fc2(h)
    affine_rescale_fc2_preserving_argmax(net, mu=1.5, sigma=2.0)
    with torch.no_grad():
        after = net.fc2(h)
    assert torch.equal(before.argmax(dim=-1), after.argmax(dim=-1))


def test_permute_changes_argmax_sometimes():
    net = _rnn_cpu()
    h = torch.randn(8, net.hidden_dim)
    with torch.no_grad():
        before = net.fc2(h).argmax(dim=-1)
    permute_fc2_outputs(net, seed=1)
    with torch.no_grad():
        after = net.fc2(h).argmax(dim=-1)
    assert not torch.equal(before, after)


def test_rank_only_rescales_fc2():
    class _Stub:
        def __init__(self):
            self.n_agents = 2
            self.device = torch.device("cpu")
            self.q_network = _rnn_cpu()
            self.mixer = QMixer(2, 8, 32, 64)
            self.target_mixer = QMixer(2, 8, 32, 64)
            self.target_q_network = _rnn_cpu()

    learner = _Stub()
    old_b = learner.q_network.fc2.bias.data.clone()
    apply_transfer(learner, "rank_only", mixer_reinit_seed=3)
    assert not torch.allclose(old_b, learner.q_network.fc2.bias.data)


def test_discrete_mujoco_roundtrip():
    scenario = "2halfcheetah"
    assert n_discrete_actions(scenario) == 27
    for idx in (0, 13, 26):
        vec = index_to_continuous(idx, scenario)
        assert continuous_to_index(vec, scenario) == idx
    vec = np.array([0.9, -0.1, 0.0], dtype=np.float32)
    back = index_to_continuous(continuous_to_index(vec, scenario), scenario)
    assert np.allclose(back, np.array([1.0, 0.0, 0.0]))


if __name__ == "__main__":
    test_affine_preserves_argmax()
    test_permute_changes_argmax_sometimes()
    test_rank_only_rescales_fc2()
    test_discrete_mujoco_roundtrip()
    print("transfer tests ok")
