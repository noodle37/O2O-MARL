# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Prior-retention losses for TCPA.

The offline collaboration prior is a *behaviour*, not a value level, so it is
retained through an ordering constraint on the online individual utilities
rather than through an absolute Q value:

    L_IGM = E[ g_t * sum_i ( max_{a in avail_i} q_on(tau_i, a)
                              - q_on(tau_i, a_off_i) ) ].

The bracket is non-negative by construction (the offline action is legal), and
it is zero whenever the offline action already is the local greedy action, so
the term only bites once the online value network starts to move the offline
action out of first place.  Because QMIX mixes monotonically, keeping every
offline action locally greedy keeps the offline joint action globally greedy
as well -- the IGM property is preserved without enumerating joint actions.

``abs_q_retention_loss`` is the control arm that keeps the *absolute* value of
the offline action instead of its rank; it is used in the ablation that
separates "relative ordering retention" from "absolute Q preservation".
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from tcpa.regret import NEG_INF, gather_agent_q


def _agent_step_weights(
    avail: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """``[B, T, N]`` weights of the valid (step, agent) decisions.

    ``mask`` follows the platform layout ``[B, T, 1]`` (filled and
    non-terminal); a 2-D ``[B, T]`` mask is accepted as well.
    """
    if mask.dim() == 3:
        mask = mask[..., 0]
    agent_valid = avail.bool().any(dim=-1).to(mask.dtype)
    return mask.to(agent_valid.dtype).unsqueeze(-1) * agent_valid


def igm_ranking_loss(
    q_on: torch.Tensor,
    a_off: torch.Tensor,
    avail: torch.Tensor,
    gate: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Gated IGM ranking loss on the offline joint action.

    Args:
        q_on: ``[B, T, N, A]`` online utilities (with gradient).
        a_off: ``[B, T, N]`` offline greedy action per agent.
        avail: ``[B, T, N, A]`` legal-action mask.
        gate: ``[B, T, 1]`` team-level retention coefficient.
        mask: ``[B, T]`` valid-step mask.

    Returns:
        ``(loss, stats)``. The loss is averaged over valid agent steps, so
        ``eta_prior`` keeps the same meaning for teams of different size.
    """
    avail = avail.bool()
    q_masked = q_on.masked_fill(~avail, NEG_INF)
    q_max = q_masked.max(dim=3).values  # [B, T, N]
    q_off_action = gather_agent_q(q_on, a_off)  # [B, T, N]
    hinge = (q_max - q_off_action).clamp(min=0.0)

    weight = _agent_step_weights(avail, mask)
    denom = weight.sum().clamp(min=1.0)
    loss = (gate * hinge * weight).sum() / denom
    stats = {
        "hinge_mean": (hinge * weight).sum() / denom,
        "offline_is_greedy_rate": (
            (hinge.detach() <= 1e-6).to(hinge.dtype) * weight
        ).sum()
        / denom,
    }
    return loss, stats


def abs_q_retention_loss(
    q_on: torch.Tensor,
    q_off: torch.Tensor,
    a_off: torch.Tensor,
    gate: torch.Tensor,
    mask: torch.Tensor,
    avail: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Control arm: keep the absolute value of the offline action.

    ``L = E[ g_t * sum_i ( q_on(tau_i, a_off_i) - q_off(tau_i, a_off_i) )^2 ]``.
    Unlike the ranking loss this pins the *level* of the offline action, so
    the online value network can no longer rescale its own utilities freely.
    """
    avail = avail.bool()
    diff = gather_agent_q(q_on, a_off) - gather_agent_q(q_off, a_off)
    weight = _agent_step_weights(avail, mask)
    denom = weight.sum().clamp(min=1.0)
    loss = (gate * (diff ** 2) * weight).sum() / denom
    return loss, {"abs_q_diff_rms": ((diff ** 2) * weight).sum().sqrt() / denom.sqrt()}


def value_anchor_loss(
    q_on: torch.Tensor,
    q_off: torch.Tensor,
    avail: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Generic trust-region anchor of the online utilities to the offline ones.

    This is the *control* arm for the "a constraint is needed early in training"
    hypothesis.  It constrains the online value function towards the frozen
    offline value function, but it never selects an action: it cannot say
    anything about which joint action the team should play.

        L_anchor = E[ mean_{a in avail_i} | q_on(tau_i, a) - q_off(tau_i, a) | ]

    The L1 norm is used on purpose: the term lives in the same Q units as the
    IGM ranking hinge, so both constraints can be compared at the same weight
    ``eta_prior`` without a hidden rescaling.  The anchor is *not* gated: it is
    the ungated, team-independent version of the constraint.
    """
    avail = avail.bool()
    diff = (q_on - q_off).abs()
    legal = avail.to(diff.dtype)
    n_legal = legal.sum(dim=-1).clamp(min=1.0)
    per_agent = (diff * legal).sum(dim=-1) / n_legal  # [B, T, N]

    weight = _agent_step_weights(avail, mask)
    denom = weight.sum().clamp(min=1.0)
    loss = (per_agent * weight).sum() / denom
    return loss, {"anchor_dev_mean": (per_agent * weight).sum() / denom}
