# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Surrogate reward R_q = Delta_off - kappa * Delta_on for MARL-ROAD."""

from __future__ import annotations

from typing import Dict, Optional

import torch


def transition_mask(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """[B, T, 1] mask of steps that belong to a non-terminal transition.

    Mirrors ``PaperQMIXLearner.train_on_batch`` but keeps the whole sequence:
    padded slots are dropped by ``filled``, and every step after a true terminal
    is dropped as well.
    """
    mask = batch["filled"].float().clone()
    if mask.shape[1] > 1:
        term = batch["terminated"].float()
        if term.dim() == 3 and term.shape[-1] > 1:
            term = term.max(dim=2, keepdim=True)[0]
        mask[:, 1:] = mask[:, 1:] * (1.0 - term[:, :-1])
    return mask


@torch.no_grad()
def joint_advantage_gap(
    learner,
    batch: Dict[str, torch.Tensor],
    extra_mask: Optional[torch.Tensor] = None,
) -> float:
    """Masked mean of Q_tot(joint greedy) - Q_tot(data joint action).

    The RNN history is the one in the batch (single-step deviation): only the
    current action is replaced by the greedy action, which is exact because the
    monotone QMixer's joint argmax factorises into per-agent argmaxes.
    """
    mac_out = learner._forward_qs(batch, target=False)
    avail = batch["avail_actions"].bool()
    if avail.shape[-1] == 1 and mac_out.shape[-1] > 1:
        avail = avail.expand_as(mac_out)
    masked_q = mac_out.masked_fill(~avail, -9999999.0)
    greedy = masked_q.argmax(dim=3, keepdim=True)

    actions = batch["actions"].long()
    if actions.shape[-1] != 1:
        raise ValueError(
            "joint_advantage_gap expects discrete actions [B,T,N,1], got "
            f"{tuple(actions.shape)}"
        )
    actions = actions.clamp(0, mac_out.shape[-1] - 1)

    q_star = torch.gather(mac_out, 3, greedy).squeeze(3)
    q_data = torch.gather(mac_out, 3, actions).squeeze(3)
    gap = learner.mixer(q_star, batch["state"]) - learner.mixer(q_data, batch["state"])

    mask = transition_mask(batch)
    if extra_mask is not None:
        mask = mask * extra_mask.float()
    return float((gap * mask).sum() / mask.sum().clamp(min=1.0))


def surrogate_reward(delta_off: float, delta_on: float, kappa: float) -> float:
    return float(delta_off) - float(kappa) * float(delta_on)
