"""MLP building blocks for MAC-Flow (512x4, LayerNorm)."""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: [...] in [0,1] -> [..., dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half, 1)
    )
    args = t.unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
    return emb


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int, layer_norm: bool = True):
        super().__init__()
        dims = [in_dim] + hidden + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JointFlowField(nn.Module):
    """v_phi(t, o_joint, x_t) -> velocity."""

    def __init__(self, obs_joint_dim: int, joint_action_dim: int, time_dim: int = 64):
        super().__init__()
        hidden = [512, 512, 512, 512]
        self.time_dim = time_dim
        self.mlp = MLP(obs_joint_dim + joint_action_dim + time_dim, hidden, joint_action_dim)

    def forward(self, t: torch.Tensor, obs_joint: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        if t_emb.dim() == x.dim() - 1:
            t_emb = t_emb.unsqueeze(-2).expand(*x.shape[:-1], self.time_dim)
        inp = torch.cat([obs_joint, x, t_emb], dim=-1)
        return self.mlp(inp)


class TwinQCritic(nn.Module):
    """Per-agent Q(o, a) with twin heads and LayerNorm MLPs."""

    def __init__(self, obs_dim: int, action_dim: int, n_agents: int, hidden: List[int] | None = None):
        super().__init__()
        hidden = hidden or [512, 512, 512, 512]
        in_dim = obs_dim + action_dim + n_agents
        self.q1 = MLP(in_dim, hidden, 1, layer_norm=True)
        self.q2 = MLP(in_dim, hidden, 1, layer_norm=True)
        self.n_agents = n_agents

    def _cat_obs(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]
        n = self.n_agents
        eye = torch.eye(n, device=obs.device, dtype=obs.dtype).unsqueeze(0).expand(b, -1, -1)
        return torch.cat([obs, actions, eye], dim=-1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """obs [B,N,obs_dim], actions [B,N,act_dim] -> [B,N,1] mean of twin Q."""
        x = self._cat_obs(obs, actions)
        return (self.q1(x) + self.q2(x)) * 0.5

    def both(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._cat_obs(obs, actions)
        return self.q1(x), self.q2(x)


class OneStepPolicy(nn.Module):
    """mu_w(o, z) -> logits (discrete) or tanh actions (continuous)."""

    def __init__(
        self,
        obs_dim: int,
        noise_dim: int,
        out_dim: int,
        n_agents: int,
        continuous: bool,
        hidden: List[int] | None = None,
    ):
        super().__init__()
        hidden = hidden or [512, 512, 512, 512]
        in_dim = obs_dim + noise_dim + n_agents
        self.continuous = continuous
        self.out_dim = out_dim
        self.n_agents = n_agents
        self.mlp = MLP(in_dim, hidden, out_dim, layer_norm=True)

    def forward(self, obs: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """obs [B,N,obs], noise [B,N,noise_dim] -> [B,N,out_dim]."""
        b, n, _ = obs.shape
        eye = torch.eye(n, device=obs.device, dtype=obs.dtype).unsqueeze(0).expand(b, -1, -1)
        x = torch.cat([obs, noise, eye], dim=-1)
        out = self.mlp(x)
        if self.continuous:
            return torch.tanh(out)
        return out


def vdn_tot(agent_q: torch.Tensor) -> torch.Tensor:
    """agent_q [B,N,1] -> [B,1] team value."""
    return agent_q.mean(dim=1)
