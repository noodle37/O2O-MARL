# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""ACKF knowledge adapter."""

from __future__ import annotations

import torch
import torch.nn as nn


class KnowledgeAdapter(nn.Module):
    """Shared residual adapter f_xi(e_off, ID_i) -> delta q.

    One adapter is shared by all agents; the explicit agent-id one-hot lets it
    specialise the residual per agent.  The small initialisation keeps the
    fused utility close to the online utility at the start of online training.
    """

    def __init__(
        self,
        knowledge_dim: int,
        n_agents: int,
        n_actions: int,
        hidden_dim: int = 64,
        init_std: float = 1e-2,
    ):
        super().__init__()
        self.n_agents = int(n_agents)
        self.net = nn.Sequential(
            nn.Linear(knowledge_dim + self.n_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=float(init_std))
                nn.init.zeros_(layer.bias)

    def forward(self, knowledge: torch.Tensor, agent_ids: torch.Tensor) -> torch.Tensor:
        """Map [..., knowledge_dim] + [..., n_agents] to [..., n_actions]."""
        return self.net(torch.cat([knowledge, agent_ids], dim=-1))
