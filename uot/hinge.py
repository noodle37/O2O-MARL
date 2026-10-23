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
    weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Hinge over unordered pairs of valid recorded candidates.

    q_live, q_bar: [K,B,T,1]. cand [K,B,T,N]. valid [K,B,T]. mask [B,T] bool.
    ``weight`` [B,T] optionally down-weights a transition's pairs (state-only
    support). The loss is the weighted mean hinge over the kept pairs.
    """
    zeros = q_live.new_zeros(())
    k = cand.shape[0]
    mask_bt = mask.bool()
    if weight is None:
        weight_bt = torch.ones_like(mask_bt, dtype=q_live.dtype)
    else:
        weight_bt = weight.reshape(mask_bt.shape).to(dtype=q_live.dtype).clamp(0.0, 1.0)
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
            keep_f = keep.float() * weight_bt.unsqueeze(0)
            if float(keep_f.sum().item()) <= 1e-12:
                continue
            sign = torch.sign(dbar).detach()
            dlive = (q_live[i] - q_live[j]).squeeze(-1)
            hinge = torch.relu(delta - sign * dlive)
            loss_sum = loss_sum + (hinge * keep_f).sum()
            keep_sum = keep_sum + keep_f.sum()
            # Weighted like the loss, otherwise the state-support arm can report
            # hinge rates above 1 (unweighted active pairs over weighted keep mass).
            act_sum = act_sum + (((hinge > 0) & keep).float() * weight_bt.unsqueeze(0)).sum()
            disagree_sum = disagree_sum + (
                (keep & (torch.sign(dlive) != sign)).float() * weight_bt.unsqueeze(0)
            ).sum()

    stats["uot_n_pairs"] = float(keep_sum.item())
    if float(keep_sum.item()) <= 1e-12:
        return zeros, stats
    rate = act_sum / keep_sum
    stats["uot_hinge_rate"] = float(rate.item())
    stats["uot_sign_disagree"] = float((disagree_sum / keep_sum).item())
    p = rate.detach()
    stats["uot_hinge_rate_std"] = float(torch.sqrt(p * (1.0 - p) + 1e-8).item())
    return loss_sum / keep_sum, stats
