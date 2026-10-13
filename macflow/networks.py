"""Per-agent MLP building blocks for MAC-Flow (repo `discrete_macflow` style).

Layout mirrors the original repo:
- agent-id one-hot is prepended to each agent's observation: [id, obs]
- flow field / one-step policy use hidden 256x4 MLPs with GELU and optional LayerNorm
  (actor_layer_norm=False in the original discrete agent)
- critic is a twin ensemble with LayerNorm on (mean-aggregated, q_agg='mean')
- time embedding: [sin(t * 2^i * pi), cos(t * 2^i * pi)] for i < kfreq (repo `_time_sin_embed`)
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


def _default_init(layer: nn.Linear) -> None:
    """variance_scaling(1.0, 'fan_avg', 'uniform') as in the original repo."""
    fan_in, fan_out = layer.weight.shape[1], layer.weight.shape[0]
    var = 1.0 / (0.5 * (fan_in + fan_out))
    bound = math.sqrt(3.0 * var)
    nn.init.uniform_(layer.weight, -bound, bound)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    """Dense stack, GELU activation, LayerNorm after activation on hidden layers."""

    def __init__(self, in_dim: int, hidden: List[int], out_dim: int, layer_norm: bool = False):
        super().__init__()
        dims = [in_dim, *hidden, out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            _default_init(linear)
            layers.append(linear)
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def time_sin_embedding(t: torch.Tensor, kfreq: int = 8) -> torch.Tensor:
    """t: [..., 1] -> [..., 2*kfreq]; freqs are 2^i * pi (repo `_time_sin_embed`)."""
    freqs = (2.0 ** torch.arange(kfreq, device=t.device, dtype=t.dtype)) * math.pi
    ang = t * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def add_agent_id(obs: torch.Tensor) -> torch.Tensor:
    """[..., N, O] -> [..., N, N + O]; one-hot agent id prepended to each obs."""
    n = obs.shape[-2]
    eye = torch.eye(n, device=obs.device, dtype=obs.dtype)
    ids = eye.expand(*obs.shape[:-2], n, n)
    return torch.cat([ids, obs], dim=-1)


class FlowField(nn.Module):
    """v_phi(o_i, x_t, t) -> velocity; shared per-agent MLP."""

    def __init__(
        self,
        pair_dim: int,
        act_dim: int,
        hidden: List[int],
        layer_norm: bool = False,
        t_freqs: int = 8,
    ):
        super().__init__()
        self.t_freqs = int(t_freqs)
        self.t_dim = 2 * self.t_freqs
        self.mlp = MLP(pair_dim + act_dim + self.t_dim, hidden, act_dim, layer_norm=layer_norm)

    def forward(self, obs_pair: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = time_sin_embedding(t, self.t_freqs)
        return self.mlp(torch.cat([obs_pair, x, t_emb], dim=-1))


class OneStepPolicy(nn.Module):
    """mu_w(o_i, z) -> raw output (logits for discrete, action vector for continuous)."""

    def __init__(
        self,
        pair_dim: int,
        noise_dim: int,
        out_dim: int,
        hidden: List[int],
        layer_norm: bool = False,
    ):
        super().__init__()
        self.mlp = MLP(pair_dim + noise_dim, hidden, out_dim, layer_norm=layer_norm)

    def forward(self, obs_pair: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([obs_pair, noise], dim=-1))


class TwinQ(nn.Module):
    """Q(o_i, a_i) twin ensemble; forward returns mean(Q1, Q2) (repo q_agg='mean')."""

    def __init__(self, pair_dim: int, act_dim: int, hidden: List[int], layer_norm: bool = True):
        super().__init__()
        self.q1 = MLP(pair_dim + act_dim, hidden, 1, layer_norm=layer_norm)
        self.q2 = MLP(pair_dim + act_dim, hidden, 1, layer_norm=layer_norm)

    def both(self, obs_pair: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs_pair, actions], dim=-1)
        return self.q1(x), self.q2(x)

    def forward(self, obs_pair: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.both(obs_pair, actions)
        return (q1 + q2) * 0.5


def mix_mean(agent_q: torch.Tensor) -> torch.Tensor:
    """agent_q [B, N, 1] -> [B, 1] average mixer (used for the policy Q objective/trace)."""
    return agent_q.mean(dim=1)
