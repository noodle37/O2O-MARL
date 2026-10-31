# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Validation objective for MU-APR.

The validation split is a set of recent online episodes. The objective is a
PPO-style surrogate of the team return:

    G_t          Monte-Carlo team return (mean agent reward, discounted)
    b_i,t        leave-one-out baseline: Q_tot(s_t, a_t) - Q_i(h_t, a_i,t)
    A_i,t        G_t - b_i,t
    rho_t        prod_i pi_theta(a_i,t | h_t) / pi_ref(a_i,t | h_t)
    J_val         masked mean of min(rho * A, clip(rho, 1-eps, 1+eps) * A)

``q_ref`` is the policy at the start of the inner rollout (detached), so the
ratio measures the K-step change only.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from sapr.regularizer import masked_policy


def _step_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3 and x.shape[-1] == 1:
        return x[..., 0]
    return x


def mc_team_returns(
    reward: torch.Tensor,
    terminated: torch.Tensor,
    filled: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Discounted team return for every step of every episode. [B, T]."""
    r = _step_tensor(reward).float()
    term = _step_tensor(terminated).float()
    fill = _step_tensor(filled).float()
    b, t = r.shape
    returns = torch.zeros_like(r)
    running = torch.zeros(b, device=r.device, dtype=r.dtype)
    for step in range(t - 1, -1, -1):
        running = r[:, step] + float(gamma) * (1.0 - term[:, step]) * running
        returns[:, step] = running
    return returns * fill


def leave_one_out_baseline(
    q_i_chosen: torch.Tensor, q_tot: torch.Tensor
) -> torch.Tensor:
    """Team value without agent i's contribution: Q_tot - Q_i. [B, T, N]."""
    return q_tot - q_i_chosen


def joint_log_ratio(
    q_theta: torch.Tensor,
    q_ref: torch.Tensor,
    avail: torch.Tensor,
    actions: torch.Tensor,
    temperature: float,
    support_floor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-step joint log probability ratio and the per-agent validity mask."""
    a = actions.long()
    if a.dim() == 4:
        a = a[..., 0]
    a = a.clamp(0, q_theta.shape[-1] - 1)
    p = masked_policy(q_theta, avail, temperature, support_floor)
    p_ref = masked_policy(q_ref.detach(), avail, temperature, support_floor)
    tiny = torch.finfo(p.dtype).tiny
    chosen = torch.gather(p, -1, a.unsqueeze(-1)).squeeze(-1).clamp_min(tiny)
    chosen_ref = torch.gather(p_ref, -1, a.unsqueeze(-1)).squeeze(-1).clamp_min(tiny)
    agent_valid = avail.bool().any(dim=-1) & torch.gather(
        avail.bool(), -1, a.unsqueeze(-1)
    ).squeeze(-1)
    log_ratio = ((chosen.log() - chosen_ref.log()) * agent_valid).sum(dim=-1)
    return log_ratio, agent_valid


def validation_objective(
    q_theta: torch.Tensor,
    q_ref: torch.Tensor,
    q_tot: torch.Tensor,
    avail: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    filled: torch.Tensor,
    *,
    temperature: float,
    support_floor: float,
    ratio_clip: float,
    advantage_normalize: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return (J_val, detached diagnostics)."""
    a = actions.long()
    if a.dim() == 4:
        a = a[..., 0]
    a = a.clamp(0, q_theta.shape[-1] - 1)
    q_tot_step = _step_tensor(q_tot)  # [B,T]
    # PPO-style constant advantage: both the team value and the local value are
    # taken from the frozen meta-start reference and detached, so only the
    # probability ratio carries the z gradient.
    q_i_ref = torch.gather(q_ref, -1, a.unsqueeze(-1)).squeeze(-1)  # [B,T,N]
    baseline = (q_tot_step.unsqueeze(-1) - q_i_ref).detach()
    advantage = returns.unsqueeze(-1) - baseline
    filled_step = _step_tensor(filled).bool()  # [B,T]

    log_ratio, agent_valid = joint_log_ratio(
        q_theta, q_ref, avail, actions, temperature, support_floor
    )
    valid = filled_step.unsqueeze(-1) & agent_valid
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        zeros = {
            "muapr/valid_steps": 0.0,
            "muapr/J_val": 0.0,
            "muapr/A_mean": 0.0,
            "muapr/A_std": 0.0,
            "muapr/G_mean": 0.0,
            "muapr/baseline_mean": 0.0,
            "muapr/ratio_mean": 1.0,
            "muapr/ratio_min": 1.0,
            "muapr/ratio_max": 1.0,
            "muapr/clip_frac": 0.0,
            "muapr/ratio_gt_clip_frac": 0.0,
            "muapr/ratio_lt_clip_frac": 0.0,
            "muapr/val_return_mean": 0.0,
        }
        return q_theta.sum() * 0.0, zeros

    adv = advantage[valid]
    adv_mean = adv.mean()
    adv_std = adv.std(unbiased=False)
    adv_norm = (advantage - adv_mean) / adv_std.clamp_min(1e-6) if advantage_normalize else advantage

    ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
    ratio_clipped = ratio.clamp(1.0 - ratio_clip, 1.0 + ratio_clip)
    surrogate = torch.minimum(ratio.unsqueeze(-1) * adv_norm, ratio_clipped.unsqueeze(-1) * adv_norm)
    J = (surrogate * valid).sum() / valid.sum().clamp(min=1)

    step_valid = filled_step & agent_valid.any(dim=-1)
    ratios = ratio[step_valid]
    if ratios.numel() == 0:
        ratios = torch.ones(1, device=ratio.device, dtype=ratio.dtype)
    gt = (ratios > 1.0 + ratio_clip).float().mean()
    lt = (ratios < 1.0 - ratio_clip).float().mean()
    first_t = filled_step[:, 0]
    val_returns = returns[:, 0][first_t]
    stats = {
        "muapr/valid_steps": float(n_valid),
        "muapr/J_val": float(J.detach().item()),
        "muapr/A_mean": float(adv_mean.item()),
        "muapr/A_std": float(adv_std.item()),
        "muapr/G_mean": float(returns[filled_step].mean().item()),
        "muapr/baseline_mean": float(baseline[valid].mean().item()),
        "muapr/ratio_mean": float(ratios.mean().item()),
        "muapr/ratio_min": float(ratios.min().item()),
        "muapr/ratio_max": float(ratios.max().item()),
        "muapr/clip_frac": float((gt + lt).item()),
        "muapr/ratio_gt_clip_frac": float(gt.item()),
        "muapr/ratio_lt_clip_frac": float(lt.item()),
        "muapr/val_return_mean": float(val_returns.mean().item()) if val_returns.numel() else 0.0,
    }
    return J, stats
