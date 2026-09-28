# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Monte-Carlo return-to-go for the Cal-QL calibration anchor V^mu."""

from __future__ import annotations

import torch


def compute_mc_returns(
    reward: torch.Tensor,
    terminated: torch.Tensor,
    filled: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse-pass return-to-go G_t = r_t + gamma * (1 - done_t) * G_{t+1}.

    Args:
        reward: [B, T, 1] (or [B, T, N]) shared team rewards.
        terminated: [B, T, 1] (or [B, T, N]) true-terminal flags (not timeout truncation).
        filled: [B, T, 1] step validity mask (bootstrap slots are unfilled).

    Returns:
        mc_return: [B, T, 1] G_t per step.
        mc_valid: [B, T, 1] 1.0 where the episode ended in a true terminal step.
            Timeout-truncated episodes are flagged 0: their G omits the tail
            after the time limit, so the calibration anchor would be biased.
    """
    r = reward.amax(dim=-1)
    term = terminated.amax(dim=-1)
    fill = filled.amax(dim=-1)
    b, t = r.shape
    mc = torch.zeros_like(r)
    g = torch.zeros(b, device=r.device, dtype=r.dtype)
    for i in range(t - 1, -1, -1):
        g = r[:, i] + gamma * (1.0 - term[:, i]) * g
        mc[:, i] = g
    valid_ep = ((term * fill).amax(dim=1) > 0.5).float().unsqueeze(1)
    valid = valid_ep.expand(b, t)
    return mc.unsqueeze(-1), valid.unsqueeze(-1)
