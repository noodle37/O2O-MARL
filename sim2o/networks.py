# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""OMIGA value-decomposition mixer shared by Q_tot and V_tot."""

from __future__ import annotations

import torch
import torch.nn as nn


class OMIGAMixer(nn.Module):
    """Q_tot = sum_i w_i(s) Q_i + b(s), V_tot = sum_i w_i(s) V_i + b(s)."""

    def __init__(self, num_agents: int, state_dim: int, hyper_dim: int = 64):
        super().__init__()
        self.num_agents = num_agents
        self.hyper_dim = hyper_dim
        self.fc = nn.Linear(state_dim, hyper_dim)
        self.w_head = nn.Linear(hyper_dim, num_agents)
        self.b_head = nn.Linear(hyper_dim, 1)

    def _params(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.relu(self.fc(states))
        weights = torch.abs(self.w_head(hidden))
        offset = self.b_head(hidden)
        return weights, offset

    def weights(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return non-negative agent weights and the state offset.

        states: [B, T, S] -> weights [B, T, N], offset [B, T, 1]
        states: [B, S]    -> weights [B, N],    offset [B, 1]
        """
        if states.dim() == 2:
            return self._params(states)
        if states.dim() != 3:
            raise ValueError(f"states must be [B,T,S] or [B,S], got {tuple(states.shape)}")
        batch, time, state_dim = states.shape
        weights, offset = self._params(states.reshape(batch * time, state_dim))
        return (
            weights.reshape(batch, time, self.num_agents),
            offset.reshape(batch, time, 1),
        )

    def forward(self, agent_values: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Map per-agent values to the joint value.

        agent_values: [B, T, N] -> [B, T, 1]; [B, N] -> [B, 1]
        """
        if agent_values.dim() == 2:
            if states.dim() != 2:
                raise ValueError(
                    f"states must be [B,S] with [B,N] values, got {tuple(states.shape)}"
                )
            weights, offset = self._params(states)
            return (weights * agent_values).sum(dim=-1, keepdim=True) + offset
        if agent_values.dim() != 3 or states.dim() != 3:
            raise ValueError(
                f"expected [B,T,N] values and [B,T,S] states, got "
                f"{tuple(agent_values.shape)} and {tuple(states.shape)}"
            )
        batch, time, n_agents = agent_values.shape
        weights, offset = self.weights(states)
        values = agent_values.reshape(batch * time, n_agents)
        return (weights.reshape(batch * time, n_agents) * values).sum(
            dim=-1, keepdim=True
        ).reshape(batch, time, 1) + offset
