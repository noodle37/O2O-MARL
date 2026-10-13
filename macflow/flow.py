"""Flow matching utilities for MAC-Flow (per-agent, repo `discrete_macflow` style)."""

from __future__ import annotations

from typing import Optional

import torch


def flow_bc_loss(
    velocity_fn,
    obs_pair: torch.Tensor,
    x1: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE between v(o, x_t, t) and (x1 - x0); x0 ~ N(0, I), t ~ U[0, 1].

    obs_pair [R, P], x1 [R, A]; mask [R] bool selects rows (padded steps dropped).
    """
    n = x1.shape[0]
    x0 = torch.randn_like(x1)
    if t is None:
        t = torch.rand(n, device=x1.device, dtype=x1.dtype)
    xt = (1.0 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * x1
    target = x1 - x0
    pred = velocity_fn(obs_pair, xt, t)
    per_row = ((pred - target) ** 2).mean(dim=-1)
    if mask is None:
        return per_row.mean()
    denom = mask.sum().clamp(min=1.0)
    return (per_row * mask).sum() / denom


def euler_integrate(
    velocity_fn,
    obs_pair: torch.Tensor,
    x0: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Integrate from t=0 to 1; t is i/steps, held constant within each step (repo style)."""
    x = x0
    for i in range(steps):
        t = torch.full((x.shape[0],), i / float(steps), device=x.device, dtype=x.dtype)
        x = x + velocity_fn(obs_pair, x, t) / float(steps)
    return x
