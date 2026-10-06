"""Flow matching utilities for MAC-Flow."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def joint_action_from_batch(
    actions: torch.Tensor,
    n_agents: int,
    n_actions: int,
    continuous: bool,
) -> torch.Tensor:
    """Build joint action vector x1 for flow target.

    discrete: actions [B,T,N,1] int -> [B,T, N*n_actions] one-hot concat
    continuous: actions [B,T,N,D] -> [B,T, N*D] flatten
    """
    if continuous:
        b, t, n, d = actions.shape
        return actions.reshape(b, t, n * d)
    idx = actions[..., 0].long().clamp(0, n_actions - 1)
    b, t, n = idx.shape
    oh = F.one_hot(idx, n_actions).float()
    return oh.reshape(b, t, n * n_actions)


def flow_bc_loss(
    velocity_fn,
    obs_joint: torch.Tensor,
    x1: torch.Tensor,
    *,
    n_samples: int = 1,
) -> torch.Tensor:
    """MSE between v(t, o, x_t) and (x1 - x0). obs_joint, x1: [B, D]."""
    b, d = x1.shape
    x0 = torch.randn(b, d, device=x1.device, dtype=x1.dtype)
    t = torch.rand(b, device=x1.device, dtype=x1.dtype)
    xt = (1.0 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * x1
    target = x1 - x0
    pred = velocity_fn(t, obs_joint, xt)
    return F.mse_loss(pred, target)


def euler_integrate(
    velocity_fn,
    obs_joint: torch.Tensor,
    x0: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Integrate from t=0 to 1. obs_joint [B,Dobs], x0 [B,Dact] -> x [B,Dact]."""
    x = x0
    dt = 1.0 / float(steps)
    t = torch.zeros(x0.shape[0], device=x0.device, dtype=x0.dtype)
    for _ in range(steps):
        v = velocity_fn(t, obs_joint, x)
        x = x + v * dt
        t = t + dt
    return x


def split_joint_to_agents(
    joint: torch.Tensor, n_agents: int, per_dim: int
) -> torch.Tensor:
    """[B, n_agents*per_dim] -> [B, n_agents, per_dim]."""
    return joint.reshape(joint.shape[0], n_agents, per_dim)


def discrete_actions_from_joint(
    joint: torch.Tensor,
    n_agents: int,
    n_actions: int,
    avail: Optional[torch.Tensor],
) -> torch.Tensor:
    """joint [B, N*A] -> [B, N, 1] int indices with legal masking."""
    per = joint.reshape(joint.shape[0], n_agents, n_actions)
    if avail is not None:
        per = per.masked_fill(~avail, -1e9)
    idx = per.argmax(dim=-1, keepdim=True)
    return idx


def obs_joint_from_batch(obs: torch.Tensor) -> torch.Tensor:
    """obs [B,T,N,D] or [B,N,D] -> concat agents on last dim."""
    if obs.dim() == 4:
        b, t, n, d = obs.shape
        return obs.reshape(b, t, n * d)
    b, n, d = obs.shape
    return obs.reshape(b, n * d)
