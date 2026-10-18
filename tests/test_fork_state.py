#!/usr/bin/env python3
"""Fork snapshot fidelity: a resumed branch must continue the identical trainer.

If any of these round-trips is lossy, the three branches would differ for reasons
that have nothing to do with the ranking loss, and the whole fork experiment would
be uninterpretable.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from uot.learner import UOTLearner
from uot.support import RecordedJoints
from utils.checkpoint import load_trainer_snapshot, save_trainer_snapshot


def _code(n_actions: int, joint) -> int:
    joint = np.asarray(joint, dtype=np.int64)
    mults = n_actions ** np.arange(joint.shape[0], dtype=np.int64)
    return int(joint @ mults)


def _support(n_agents, n_actions, joints) -> RecordedJoints:
    return RecordedJoints(
        n_agents, n_actions, np.array([_code(n_actions, u) for u in joints], dtype=np.int64)
    )


def _batch(n_agents=3, n_actions=4, seq=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    b = 2
    return {
        "obs": torch.randn(b, seq, n_agents, 6, generator=g),
        "actions": torch.zeros(b, seq, n_agents, 1, dtype=torch.long),
        "state": torch.randn(b, seq, 5, generator=g),
        "avail_actions": torch.ones(b, seq, n_agents, n_actions, dtype=torch.bool),
        "reward": torch.ones(b, seq, 1),
        "terminated": torch.zeros(b, seq, 1),
        "filled": torch.ones(b, seq, 1),
    }


def _learner(joints, **kwargs) -> UOTLearner:
    n_agents, n_actions = 3, 4
    torch.manual_seed(0)
    learner = UOTLearner(
        n_agents=n_agents,
        n_actions=n_actions,
        obs_dim=6,
        state_dim=5,
        support=_support(n_agents, n_actions, joints),
        recurrent_layer_dim=8,
        mixer_embed_dim=4,
        mixer_hyper_dim=8,
        device="cpu",
        **kwargs,
    )
    learner.freeze_oracle()
    return learner


def _episode(seed: int) -> dict:
    g = np.random.default_rng(seed)
    return {
        "steps": [
            {
                "obs": g.normal(size=(3, 6)),
                "actions": np.zeros((3, 1), dtype=np.int64),
                "state": g.normal(size=5),
                "reward": 1.0,
                "terminated": 0.0,
                "legals": np.ones((3, 4), dtype=bool),
            }
        ],
        "length": 1,
    }


def _param_digest(learner) -> list[torch.Tensor]:
    return [p.detach().clone() for p in learner.q_network.parameters()] + [
        p.detach().clone() for p in learner.mixer.parameters()
    ]


def test_snapshot_resumes_the_identical_update(tmp_path=None):
    """Weights + optimizer + counters must round-trip so the next step is identical."""
    import tempfile

    joints = [[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1]]
    live = _learner(joints)
    batch = _batch()
    for _ in range(3):
        live.train_on_batch(batch, t_env=0, episode_delta=8)

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "snap.pt")
        buffer = deque([_episode(i) for i in range(4)], maxlen=5000)
        rng = np.random.default_rng(7)
        save_trainer_snapshot(
            path,
            live,
            online_buffer=buffer,
            buffer_maxlen=5000,
            t_env=1234,
            batch_rng=rng,
            extra={"seed": 1},
        )

        # A fresh learner at a different point in training must be pulled back exactly.
        resumed = _learner(joints)
        for _ in range(2):
            resumed.train_on_batch(_batch(seed=99), t_env=0, episode_delta=8)
        state = load_trainer_snapshot(path, resumed)

        assert state["t_env"] == 1234
        assert resumed.training_step_ctr == live.training_step_ctr
        assert resumed.episode_counter == live.episode_counter
        assert resumed.last_target_update_episode == live.last_target_update_episode
        assert len(state["online_buffer"]) == 4
        assert state["online_buffer"].maxlen == 5000

        # Q-bar must come from the snapshot, not from a re-freeze of the live net.
        for a, b in zip(live.frozen_q_network.parameters(), resumed.frozen_q_network.parameters()):
            assert torch.equal(a, b)
        assert not torch.equal(
            next(iter(live.frozen_q_network.parameters())),
            next(iter(resumed.q_network.parameters())),
        )

        # Same state + same batch => bit-identical gradients and parameters.
        live.train_on_batch(batch, t_env=1234, episode_delta=8)
        resumed.train_on_batch(batch, t_env=1234, episode_delta=8)
        for a, b in zip(_param_digest(live), _param_digest(resumed)):
            assert torch.equal(a, b), "resumed branch diverged from the parent update"
    print("snapshot round-trip ok")


def test_snapshot_restores_every_random_source():
    import tempfile

    joints = [[0, 0, 0], [0, 0, 1]]
    live = _learner(joints)

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "snap.pt")
        np.random.seed(11)
        torch.manual_seed(12)
        rng = np.random.default_rng(5)

        # Snapshot first, then record what each source produces next: that is what a
        # resumed run must reproduce after its state has been disturbed.
        save_trainer_snapshot(path, live, t_env=0, batch_rng=rng)
        expected_np = np.random.random()
        expected_torch = torch.randn(3)
        expected_rng = rng.random(4)

        # Disturb every source.
        np.random.seed(999)
        torch.manual_seed(999)
        other = np.random.default_rng(999)
        _ = other.random(4)
        _ = np.random.random(7)
        _ = torch.randn(7)

        resumed = _learner(joints)
        state = load_trainer_snapshot(path, resumed)
        assert np.allclose(np.random.random(), expected_np)
        assert torch.allclose(torch.randn(3), expected_torch)
        assert np.allclose(state["batch_rng"].random(4), expected_rng)
    print("rng round-trip ok")


def test_eta_schedule_continue_off_decay():
    cont = _learner([[0, 0, 0], [0, 0, 1]], eta=0.1)
    assert cont.eta_at(0) == 0.1
    assert cont.eta_at(1_000_000) == 0.1

    off = _learner([[0, 0, 0], [0, 0, 1]], eta=0.1, eta_drop_at=500_000)
    assert off.eta_at(499_999) == 0.1
    assert off.eta_at(500_000) == 0.0
    assert off.eta_at(2_000_000) == 0.0

    dec = _learner(
        [[0, 0, 0], [0, 0, 1]],
        eta=0.1,
        eta_decay_start=1_000_000,
        eta_decay_steps=2_000_000,
    )
    assert dec.eta_at(999_999) == 0.1
    assert abs(dec.eta_at(2_000_000) - 0.05) < 1e-9
    assert dec.eta_at(3_000_000) == 0.0

    zero = _learner([[0, 0, 0], [0, 0, 1]], eta=0.0)
    assert zero.eta_at(0) == 0.0
    print("eta schedule ok")


def test_zero_weight_keeps_the_counterfactual_diagnostic():
    """Branch OFF must still log what the constraint would have been."""
    joints = [[0, 0, 0], [0, 0, 1]]
    learner = _learner(joints, eta=0.1, eta_drop_at=0)
    logs = learner.train_on_batch(_batch(), t_env=10, episode_delta=8)
    assert logs["uot_eta_now"] == 0.0
    assert logs["reg_strength"] == 0.0
    # The hinge is still measured (it is just not backpropagated).
    assert "uot_n_pairs" in logs and "uot_hinge_rate" in logs
    print("counterfactual diagnostic ok")


def test_rng_states_survive_a_cuda_map_location():
    """Regression: torch.load(map_location=cuda) moves the RNG tensors to GPU, and
    torch.set_rng_state rejects non-CPU ByteTensors. A branch on cuda must still
    restore the CPU/CUDA generators."""
    if not torch.cuda.is_available():
        print("cuda unavailable, skipping")
        return
    import tempfile

    joints = [[0, 0, 0], [0, 0, 1]]
    live = _learner(joints)
    np.random.seed(3)
    torch.manual_seed(4)

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "snap.pt")
        save_trainer_snapshot(path, live, t_env=0)
        expected_np = np.random.random()
        expected_torch = torch.randn(3)

        torch.manual_seed(77)
        np.random.seed(77)
        _ = torch.randn(5)

        resumed = _learner(joints)
        load_trainer_snapshot(path, resumed, map_location=resumed.device)
        assert np.allclose(np.random.random(), expected_np)
        assert torch.allclose(torch.randn(3), expected_torch)
    print("cuda map_location rng ok")


def test_resume_without_freeze_oracle_restores_qbar():
    """Regression: a resumed branch skips freeze_oracle(), so the snapshot loader must
    rebuild the frozen oracle itself. Otherwise the ranking term silently no-ops and
    every branch trains as if the ranking loss were already off."""
    import tempfile

    joints = [[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1]]
    live = _learner(joints)
    batch = _batch()
    live.train_on_batch(batch, t_env=0, episode_delta=8)

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "snap.pt")
        save_trainer_snapshot(path, live, t_env=100)

        # Mirror the real resume path: a freshly built learner has frozen=None because
        # freeze_oracle() is deliberately not called.
        torch.manual_seed(123)
        resumed = UOTLearner(
            n_agents=3,
            n_actions=4,
            obs_dim=6,
            state_dim=5,
            support=_support(3, 4, joints),
            recurrent_layer_dim=8,
            mixer_embed_dim=4,
            mixer_hyper_dim=8,
            device="cpu",
        )
        assert resumed.frozen_q_network is None and resumed.frozen_mixer is None
        load_trainer_snapshot(path, resumed)

        assert resumed.frozen_q_network is not None
        assert resumed.frozen_mixer is not None
        for a, b in zip(live.frozen_q_network.parameters(), resumed.frozen_q_network.parameters()):
            assert torch.equal(a, b)
        for a, b in zip(live.frozen_mixer.parameters(), resumed.frozen_mixer.parameters()):
            assert torch.equal(a, b)
        for param in resumed.frozen_q_network.parameters():
            assert not param.requires_grad

        # The ordinal term must actually engage, not fall through to the zero default.
        logs = resumed.train_on_batch(batch, t_env=100, episode_delta=8)
        assert logs["uot_n_pairs"] > 0, "ranking term did not engage after resume"
        # The hinge consumes Q-bar's forward, so check that directly rather than
        # relying on the hinge being active (a satisfied hinge has loss 0).
        with torch.no_grad():
            bar_ref = live._forward_net(live.frozen_q_network, batch)
            bar_new = resumed._forward_net(resumed.frozen_q_network, batch)
        assert torch.allclose(bar_ref, bar_new), "restored Q-bar differs from the snapshot's"
    print("qbar restore without freeze_oracle ok")


def test_fork_branch_name_survives_yaml_boolean_parsing():
    """Regression: YAML 1.1 reads a bare `off` as False, which used to silently
    relabel the branch as "main" and make two branches share a checkpoint file."""
    from omegaconf import OmegaConf

    from training.online import FORK_BRANCHES, _fork_branch

    # The driver must quote the value so the branch name survives as a string.
    merged = OmegaConf.merge(
        OmegaConf.create({"fork_branch": "continue"}),
        OmegaConf.from_dotlist(["fork_branch='off'"]),
    )
    assert OmegaConf.to_container(merged, resolve=True)["fork_branch"] == "off"
    # Unquoted, the same override is a boolean.
    bare = OmegaConf.merge(
        OmegaConf.create({"fork_branch": "continue"}),
        OmegaConf.from_dotlist(["fork_branch=off"]),
    )
    assert OmegaConf.to_container(bare, resolve=True)["fork_branch"] is False

    for name in FORK_BRANCHES:
        assert _fork_branch({"fork_branch": name}) == name
    assert _fork_branch({}) == "main"
    assert _fork_branch({"fork_branch": None}) == "main"

    # An unquoted `off` becomes False; that must fail loudly, not silently relabel.
    try:
        _fork_branch({"fork_branch": False})
    except ValueError as exc:
        assert "boolean" in str(exc)
    else:
        raise AssertionError("boolean fork_branch did not raise")
    try:
        _fork_branch({"fork_branch": "typo"})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown fork_branch did not raise")
    print("fork branch naming ok")


if __name__ == "__main__":
    test_snapshot_resumes_the_identical_update()
    test_snapshot_restores_every_random_source()
    test_rng_states_survive_a_cuda_map_location()
    test_resume_without_freeze_oracle_restores_qbar()
    test_fork_branch_name_survives_yaml_boolean_parsing()
    test_eta_schedule_continue_off_decay()
    test_zero_weight_keeps_the_counterfactual_diagnostic()
    print("fork state tests ok")