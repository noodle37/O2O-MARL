# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Offline-to-online transfer ablations: preserve or break Q ranking vs magnitude."""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn

from ovmse.networks import QMixer, MarlcRNN


TRANSFER_MODES = frozenset(
    {
        "ovmse",
        "full_td",
        "rank_only",
        "rank_break",
        "shift_plus",
        "shift_minus",
        "scratch",
    }
)


def _fc2_layers(net: MarlcRNN) -> nn.Linear:
    return net.fc2


def sync_targets(learner) -> None:
    learner.target_q_network.load_state_dict(learner.q_network.state_dict())
    learner.target_mixer.load_state_dict(learner.mixer.state_dict())


def reinit_mixer(learner, *, seed: int = 0) -> None:
    n_agents = learner.n_agents
    state_dim = learner.mixer.hyper_w_1[0].in_features
    embed = learner.mixer.embed_dim
    hyper = learner.mixer.hypernet_embed
    device = learner.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    learner.mixer = QMixer(n_agents, state_dim, embed, hyper).to(device)
    learner.target_mixer = copy.deepcopy(learner.mixer).to(device)


def affine_rescale_fc2_preserving_argmax(net: MarlcRNN, mu: float, sigma: float) -> None:
    """Apply q' = (q - mu) / sigma on every agent head (strictly increasing if sigma > 0)."""
    fc2 = _fc2_layers(net)
    if sigma <= 0:
        sigma = 1.0
    scale = 1.0 / sigma
    fc2.weight.data.mul_(scale)
    fc2.bias.data.sub_(mu).mul_(scale)


def estimate_fc2_affine_stats(net: MarlcRNN) -> tuple[float, float]:
    """Global mean/std of fc2 biases and row norms (used for rank_only)."""
    fc2 = _fc2_layers(net)
    with torch.no_grad():
        b = fc2.bias.data.reshape(-1)
        row_norm = fc2.weight.data.norm(dim=1)
        vals = torch.cat([b, row_norm])
        mu = float(vals.mean().item())
        sigma = float(vals.std().item())
        if sigma < 1e-6:
            sigma = 1.0
    return mu, sigma


def permute_fc2_outputs(net: MarlcRNN, seed: int = 0) -> None:
    """Permute output action rows of fc2; multiset of Q rows preserved per hidden dim."""
    fc2 = _fc2_layers(net)
    n_out = fc2.weight.shape[0]
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(n_out, generator=gen)
    fc2.weight.data = fc2.weight.data[perm]
    fc2.bias.data = fc2.bias.data[perm]


def shift_fc2_bias(net: MarlcRNN, delta: float) -> None:
    _fc2_layers(net).bias.data.add_(delta)


def apply_transfer(
    learner,
    mode: str,
    *,
    shift: float = 0.0,
    perm_seed: int = 0,
    mixer_reinit_seed: int = 0,
) -> str:
    """Mutate learner Q heads / mixer for an online arm. Returns normalized mode name."""
    mode = str(mode).lower()
    if mode not in TRANSFER_MODES:
        raise ValueError(f"Unknown transfer mode {mode}; expected one of {sorted(TRANSFER_MODES)}")

    if mode == "scratch":
        return mode

    if mode == "rank_only":
        mu, sigma = estimate_fc2_affine_stats(learner.q_network)
        affine_rescale_fc2_preserving_argmax(learner.q_network, mu, sigma)
        reinit_mixer(learner, seed=mixer_reinit_seed)
        sync_targets(learner)
        return mode

    if mode == "rank_break":
        permute_fc2_outputs(learner.q_network, seed=perm_seed)
        sync_targets(learner)
        return mode

    if mode in ("shift_plus", "shift_minus"):
        if shift <= 0:
            raise ValueError(f"shift must be positive for {mode}, got {shift}")
        delta = shift if mode == "shift_plus" else -shift
        shift_fc2_bias(learner.q_network, delta)
        sync_targets(learner)
        return mode

    # ovmse, full_td: weights unchanged
    return mode
