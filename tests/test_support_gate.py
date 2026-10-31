"""Unit tests for the teammate-support gate on the offline value memory."""

from __future__ import annotations

import torch

from ovmse.learner import PaperQMIXLearner


def make_learner(**kwargs) -> PaperQMIXLearner:
    learner = PaperQMIXLearner(
        n_agents=3,
        n_actions=4,
        obs_dim=5,
        state_dim=6,
        recurrent_layer_dim=8,
        mixer_embed_dim=8,
        mixer_hyper_dim=8,
        device="cpu",
        **kwargs,
    )
    learner.offline_q_network = learner.q_network
    learner.offline_mixer = learner.mixer
    return learner


def make_batch(b: int = 2, t: int = 5, n: int = 3, a: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    batch = {
        "obs": torch.randn(b, t, n, 5, generator=g),
        "state": torch.randn(b, t, 6, generator=g),
        "actions": torch.randint(0, a, (b, t, n, 1), generator=g),
        "avail_actions": torch.ones(b, t, n, a, dtype=torch.bool),
        "reward": torch.randn(b, t, 1, generator=g),
        "terminated": torch.zeros(b, t, 1),
        "filled": torch.ones(b, t, 1),
    }
    return batch


def test_gate_is_bounded_and_shaped():
    learner = make_learner(support_gate_mode="on", support_gate_theta=1.0, support_gate_temp=0.2)
    batch = make_batch()
    gate = learner.support_gate(batch)
    assert gate.shape == (2, 4, 1)
    assert torch.all(gate >= 0.0) and torch.all(gate <= 1.0)


def test_gate_decreases_with_deviation():
    """Raising theta trusts more steps; the gate must be monotone in theta."""
    batch = make_batch()
    low = make_learner(support_gate_mode="on", support_gate_theta=0.0, support_gate_temp=0.2)
    high = make_learner(support_gate_mode="on", support_gate_theta=5.0, support_gate_temp=0.2)
    high.q_network.load_state_dict(low.q_network.state_dict())
    high.offline_q_network = high.q_network
    assert high.support_gate(batch).mean() > low.support_gate(batch).mean()


def test_anti_mode_is_the_exact_complement():
    batch = make_batch()
    on = make_learner(support_gate_mode="on", support_gate_theta=1.0, support_gate_temp=0.3)
    anti = make_learner(support_gate_mode="anti", support_gate_theta=1.0, support_gate_temp=0.3)
    anti.q_network.load_state_dict(on.q_network.state_dict())
    anti.offline_q_network = anti.q_network
    torch.testing.assert_close(on.support_gate(batch) + anti.support_gate(batch), torch.ones(2, 4, 1))


def test_deviation_ignores_unavailable_actions():
    """Masked actions must not leak probability mass out of the support measure."""
    learner = make_learner(support_gate_mode="on")
    batch = make_batch()
    dev_full = learner._offline_support_dev(batch)

    masked = {key: value.clone() for key, value in batch.items()}
    masked["avail_actions"][:, :, :, 3] = False
    masked["actions"] = masked["actions"].clamp(max=2)
    batch2 = {key: value.clone() for key, value in batch.items()}
    batch2["actions"] = masked["actions"]
    dev_masked = learner._offline_support_dev(masked)
    dev_unmasked_same_actions = learner._offline_support_dev(batch2)

    assert torch.isfinite(dev_masked).all()
    # Removing an action concentrates the remaining mass, so every executed
    # action becomes at least as likely as it was with the full action set.
    assert torch.all(dev_masked <= dev_unmasked_same_actions + 1e-5)
    assert dev_full.shape == dev_masked.shape


def test_zero_gate_recovers_plain_td_loss():
    learner = make_learner(support_gate_mode="on")
    batch = make_batch()
    actions, states = batch["actions"], batch["state"]
    chosen = torch.randn(2, 5, 1)
    targets = torch.randn(2, 4, 1)
    mask = torch.ones(2, 4, 1)
    gated = learner._ovm_td_loss(
        batch, actions, states, chosen, targets, mask, torch.zeros(2, 4, 1)
    )
    plain = 0.5 * (((chosen[:, :-1] - targets) ** 2) * mask).sum() / mask.sum()
    torch.testing.assert_close(gated, plain)


def test_full_gate_recovers_ungated_ovm_loss():
    learner = make_learner(support_gate_mode="on")
    batch = make_batch()
    actions, states = batch["actions"], batch["state"]
    chosen = torch.randn(2, 5, 1)
    targets = torch.randn(2, 4, 1)
    mask = torch.ones(2, 4, 1)
    tensor_one = learner._ovm_td_loss(
        batch, actions, states, chosen, targets, mask, torch.ones(2, 4, 1)
    )
    scalar_one = learner._ovm_td_loss(batch, actions, states, chosen, targets, mask, 1.0)
    torch.testing.assert_close(tensor_one, scalar_one)


def test_gate_is_per_step_not_batch_constant():
    learner = make_learner(support_gate_mode="on", support_gate_theta=1.2, support_gate_temp=0.15)
    gate = learner.support_gate(make_batch(b=4, t=8, seed=3))
    assert gate.std() > 1e-3


def test_calibration_sets_theta_to_offline_quantile():
    learner = make_learner(support_gate_mode="on")
    batches = [make_batch(b=4, t=6, seed=s) for s in range(3)]
    info = learner.calibrate_support_gate(batches, quantile=0.5, temp_frac=0.25)
    devs = torch.cat([learner._offline_support_dev(b).flatten() for b in batches])
    torch.testing.assert_close(
        torch.tensor(learner.support_gate_theta), torch.quantile(devs, 0.5), atol=1e-5, rtol=1e-4
    )
    assert learner.support_gate_temp > 0
    assert info["support/n"] == devs.numel()
    # Half the offline data sits on each side of a median threshold, so the
    # gate opens on about half of the old team's own steps by construction.
    gate = learner.support_gate(batches[0])
    assert 0.0 < float(gate.mean()) < 1.0


def test_calibration_respects_the_filled_mask():
    learner = make_learner(support_gate_mode="on")
    batch = make_batch(b=2, t=6, seed=7)
    batch["filled"][:, 3:] = 0.0
    info = learner.calibrate_support_gate([batch], quantile=0.5)
    assert info["support/n"] == 2 * 3


def test_gate_off_leaves_training_step_unchanged():
    """The mode switch must be the only difference between OVM and gated OVM."""
    batch = make_batch(b=4, t=6, seed=11)
    base = make_learner(support_gate_mode="off")
    logs_off = base.train_on_batch(batch, use_ovm=True, cql_weight=0.0, t_env=0)
    assert "support/gate_mean" not in logs_off

    gated = make_learner(support_gate_mode="on", support_gate_theta=1.0, support_gate_temp=0.2)
    logs_on = gated.train_on_batch(batch, use_ovm=True, cql_weight=0.0, t_env=0)
    assert 0.0 <= logs_on["support/gate_mean"] <= 1.0
    assert 0.0 <= logs_on["support/gate_frac_above_half"] <= 1.0


def test_gate_does_not_touch_the_non_ovm_path():
    batch = make_batch(b=4, t=6, seed=13)
    gated = make_learner(support_gate_mode="on")
    logs = gated.train_on_batch(batch, use_ovm=False, cql_weight=0.0, t_env=0)
    assert "support/gate_mean" not in logs


def test_cql_weight_constant_one_matches_unweighted():
    """A uniform weight must reduce to the plain CQL loss, so the weighted arm
    differs from a tuned global cql_weight only in where the mass sits."""
    learner = make_learner(support_cql_mode="on")
    batch = make_batch(seed=3)
    mac_out = learner._forward_qs(batch, target=False)
    actions = batch["actions"]
    chosen_q = torch.gather(mac_out, 3, actions.long())
    chosen_tot = learner.mixer(chosen_q.squeeze(-1), batch["state"])
    mask = batch["filled"][:, :-1].float()
    plain = learner._compute_cfcql_loss(mac_out, batch, chosen_tot, mask)
    weighted = learner._compute_cfcql_loss(
        mac_out, batch, chosen_tot, mask, weight=torch.ones_like(mask)
    )
    torch.testing.assert_close(plain, weighted)


def test_cql_weight_rescaling_leaves_loss_unchanged():
    """Renormalisation means only the relative placement of the weight matters."""
    learner = make_learner(support_cql_mode="on")
    batch = make_batch(seed=4)
    mac_out = learner._forward_qs(batch, target=False)
    chosen_q = torch.gather(mac_out, 3, batch["actions"].long())
    chosen_tot = learner.mixer(chosen_q.squeeze(-1), batch["state"])
    mask = batch["filled"][:, :-1].float()
    w = learner.support_gate(batch)
    a = learner._compute_cfcql_loss(mac_out, batch, chosen_tot, mask, weight=w)
    b = learner._compute_cfcql_loss(mac_out, batch, chosen_tot, mask, weight=w * 0.5)
    torch.testing.assert_close(a, b)


def test_cql_weight_reweights_steps():
    """A non-uniform weight must actually change the loss."""
    learner = make_learner(support_cql_mode="on", support_gate_theta=1.0, support_gate_temp=0.2)
    batch = make_batch(seed=5)
    mac_out = learner._forward_qs(batch, target=False)
    chosen_q = torch.gather(mac_out, 3, batch["actions"].long())
    chosen_tot = learner.mixer(chosen_q.squeeze(-1), batch["state"])
    mask = batch["filled"][:, :-1].float()
    gate = learner.support_gate(batch)
    assert gate.std() > 1e-4
    plain = learner._compute_cfcql_loss(mac_out, batch, chosen_tot, mask)
    weighted = learner._compute_cfcql_loss(mac_out, batch, chosen_tot, mask, weight=gate)
    assert not torch.allclose(plain, weighted)


def test_cql_mode_override_does_not_read_value_channel_mode():
    """The two channels carry independent arm assignments."""
    learner = make_learner(support_gate_mode="anti", support_cql_mode="on")
    batch = make_batch(seed=6)
    value_gate = learner.support_gate(batch)
    cql_gate = learner.support_gate(batch, mode="on")
    torch.testing.assert_close(value_gate + cql_gate, torch.ones_like(cql_gate))


def test_cql_mode_off_leaves_training_step_unchanged():
    learner = make_learner(support_cql_mode="off", cql_weight=0.05)
    batch = make_batch(seed=7)
    stats = learner.train_on_batch(batch, t_env=0)
    assert "support/cql_weight_mean" not in stats


def test_cql_mode_on_reports_weight_mean():
    learner = make_learner(support_cql_mode="on", cql_weight=0.05, support_gate_theta=1.0)
    batch = make_batch(seed=8)
    stats = learner.train_on_batch(batch, t_env=0)
    assert 0.0 <= stats["support/cql_weight_mean"] <= 1.0


def test_cql_normalize_mass_makes_mean_one():
    """Per-batch mass normalization forces mean weight == 1, matching uniform."""
    learner = make_learner(
        support_cql_mode="on",
        support_cql_normalize_mass=True,
        cql_weight=0.05,
        support_gate_theta=1.0,
    )
    batch = make_batch(seed=9)
    stats = learner.train_on_batch(batch, t_env=0)
    assert abs(stats["support/cql_weight_mean"] - 1.0) < 1e-4


def test_cql_normalize_mass_anti_also_mean_one():
    """Anti mode with mass normalization also has mean weight == 1."""
    learner = make_learner(
        support_cql_mode="anti",
        support_cql_normalize_mass=True,
        cql_weight=0.05,
        support_gate_theta=1.0,
    )
    batch = make_batch(seed=10)
    stats = learner.train_on_batch(batch, t_env=0)
    assert abs(stats["support/cql_weight_mean"] - 1.0) < 1e-4


def test_yaml_off_token_parsed_as_bool_is_not_treated_as_enabled():
    """OmegaConf turns `support_gate_mode=off` into False. If that were coerced
    with str() it would read as "False" != "off" and silently enable the gate in
    a control arm, which is how an invalid Stage C run was produced."""
    assert make_learner(support_gate_mode=False).support_gate_mode == "off"
    assert make_learner(support_gate_mode=True).support_gate_mode == "on"
    assert make_learner(support_cql_mode=False).support_cql_mode == "off"


def test_unknown_gate_mode_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        make_learner(support_gate_mode="disabled")
    with pytest.raises(ValueError):
        make_learner(support_cql_mode="yes_please")


def test_lift_diagnostics_track_the_applied_gate():
    """train_ovm lift stats must be computed from the gated lambda, otherwise
    two gate arms log identical diagnostics and cannot be audited."""
    from ovmse.diagnostics import ovm_target_stats

    off_tot = torch.full((2, 4, 1), 3.0)
    td = torch.zeros(2, 4, 1)
    mask = torch.ones(2, 4, 1)
    ungated = ovm_target_stats(off_tot, td, mask, 1.0)
    gated = ovm_target_stats(off_tot, td, mask, torch.full((2, 4, 1), 0.25))
    assert ungated["lift_effective_mean"] > gated["lift_effective_mean"]
    assert gated["lambda_memory"] == 0.25


def test_gate_stats_survive_the_training_log_filter():
    """run_online only forwards keys with known prefixes into the interval mean."""
    from training.online import _mean_stats

    rows = [
        {"train_ovm/lift_raw_mean": 1.0},
        {"train_ovm/lift_raw_mean": 3.0, "support/gate_mean": 0.4},
    ]
    out = _mean_stats(rows)
    assert out["support/gate_mean_interval_mean"] == 0.4
    assert out["train_ovm/lift_raw_mean_interval_mean"] == 2.0
