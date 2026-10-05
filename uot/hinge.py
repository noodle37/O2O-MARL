"""Ordinal hinge. Q_bar enters only through its sign."""

from __future__ import annotations

from typing import Dict

import torch


def ordinal_hinge(
    q_live: torch.Tensor,
    q_bar: torch.Tensor,
    cand: torch.Tensor,
    valid: torch.Tensor,
    mask: torch.Tensor,
    *,
    delta: float,
    margin: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Hinge over unordered pairs of valid recorded candidates.

    q_live, q_bar: [K,B,T,1]. cand [K,B,T,N]. valid [K,B,T]. mask [B,T] bool.
    """
    zeros = q_live.new_zeros(())
    k = cand.shape[0]
    mask_bt = mask.bool()
    stats = {
        "uot_cand_in_frac": 0.0,
        "uot_n_pairs": 0.0,
        "uot_hinge_rate": 0.0,
        "uot_hinge_rate_std": 0.0,
        "uot_sign_disagree": 0.0,
    }
    cand_in = (valid.float() * mask_bt.unsqueeze(0).float()).sum()
    cand_den = (mask_bt.float().sum() * k).clamp(min=1.0)
    stats["uot_cand_in_frac"] = float((cand_in / cand_den).item())

    loss_sum = zeros
    keep_sum = zeros
    act_sum = zeros
    disagree_sum = zeros
    for i in range(k):
        for j in range(i + 1, k):
            same = (cand[i] == cand[j]).all(dim=-1)
            dbar = (q_bar[i] - q_bar[j]).squeeze(-1)
            keep = valid[i] & valid[j] & (~same) & mask_bt & (dbar.abs() >= margin)
            keep_f = keep.float()
            if float(keep_f.sum().item()) < 1:
                continue
            sign = torch.sign(dbar).detach()
            dlive = (q_live[i] - q_live[j]).squeeze(-1)
            hinge = torch.relu(delta - sign * dlive)
            loss_sum = loss_sum + (hinge * keep_f).sum()
            keep_sum = keep_sum + keep_f.sum()
            act_sum = act_sum + ((hinge > 0) & keep).float().sum()
            disagree_sum = disagree_sum + (keep & (torch.sign(dlive) != sign)).float().sum()

    stats["uot_n_pairs"] = float(keep_sum.item())
    if float(keep_sum.item()) < 1:
        return zeros, stats
    rate = act_sum / keep_sum
    stats["uot_hinge_rate"] = float(rate.item())
    stats["uot_sign_disagree"] = float((disagree_sum / keep_sum).item())
    p = rate.detach()
    stats["uot_hinge_rate_std"] = float(torch.sqrt(p * (1.0 - p) + 1e-8).item())
    return loss_sum / keep_sum, stats
