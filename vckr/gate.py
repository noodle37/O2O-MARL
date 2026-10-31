# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Decentralized validity-gate head.

The centralized training step knows which agents belong to a selected valid
sub-team at each step.  This head distils that indicator into a local function
of (own observation, agent id), so at execution each agent can decide on its
own whether to hand over to its offline snippet for H steps.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GateHead(nn.Module):
    def __init__(self, obs_dim: int, n_agents: int, hidden: int = 64, bias_init: float = -4.0):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_agents = int(n_agents)
        self.net = nn.Sequential(
            nn.Linear(int(obs_dim) + int(n_agents), int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), 1),
        )
        with torch.no_grad():
            self.net[-1].bias.fill_(float(bias_init))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: [..., N, obs_dim] -> logits [..., N]."""
        n = obs.shape[-2]
        eye = torch.eye(self.n_agents, device=obs.device, dtype=obs.dtype)
        shape = obs.shape[:-2]
        ids = eye.view(*([1] * len(shape)), self.n_agents, self.n_agents).expand(
            *shape, self.n_agents, self.n_agents
        )
        x = torch.cat([obs, ids], dim=-1)
        return self.net(x).squeeze(-1)
