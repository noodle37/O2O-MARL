# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""TD-lambda target construction (PyMARL-style, backward pass along time)."""

import torch


def build_td_lambda_targets(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    mask: torch.Tensor,
    target_qs: torch.Tensor,
    gamma: float,
    td_lambda: float,
) -> torch.Tensor:
    """Compute λ-returns from team rewards and bootstrapped target Q_tot (time reversed)."""
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1.0 - terminated.sum(dim=1).clamp(max=1.0))
    for t in range(ret.shape[1] - 2, -1, -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] * (
            rewards[:, t]
            + (1.0 - td_lambda) * gamma * target_qs[:, t + 1] * (1.0 - terminated[:, t])
        )
    return ret[:, :-1]
