"""Ordinal coordination prior: teacher ranking only, never a numerical floor."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def ordinal_q_hinge(
    live_q: torch.Tensor,
    teacher_q: torch.Tensor,
    avail: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float = 0.1,
    delta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Pairwise hinge that asks live Q to keep the frozen teacher's action order.

    live_q, teacher_q: [B, T, N, A]; avail [B, T, N, A] bool; mask [B, T, 1] or [B, T].
    Teacher values are detached. The live Q enters only through pairwise differences,
    so this cannot be abused as a value floor.
    """
    teacher = teacher_q.detach()
    if mask.dim() == 3:
        mask_bt = mask[..., 0] > 0.5
    else:
        mask_bt = mask > 0.5
    avail_b = avail.bool()
    n_actions = live_q.shape[-1]
    zeros = live_q.new_zeros(())
    loss_sum = zeros
    keep_sum = zeros
    active_sum = zeros
    for i in range(n_actions):
        for j in range(i + 1, n_actions):
            legal = avail_b[..., i] & avail_b[..., j] & mask_bt.unsqueeze(-1)
            d_teacher = teacher[..., i] - teacher[..., j]
            keep = legal & (d_teacher.abs() >= margin)
            keep_f = keep.float()
            n_keep = keep_f.sum()
            if float(n_keep.item()) < 1:
                continue
            sign = torch.sign(d_teacher).detach()
            d_live = live_q[..., i] - live_q[..., j]
            hinge = torch.relu(delta - sign * d_live)
            loss_sum = loss_sum + (hinge * keep_f).sum()
            keep_sum = keep_sum + n_keep
            active_sum = active_sum + ((hinge > 0) & keep).float().sum()
    stats = {
        "fsr/prior_pairs": float(keep_sum.detach().item()),
        "fsr/prior_hinge_rate": 0.0,
    }
    if float(keep_sum.item()) < 1:
        return zeros, stats
    stats["fsr/prior_hinge_rate"] = float((active_sum / keep_sum).detach().item())
    return loss_sum / keep_sum, stats
