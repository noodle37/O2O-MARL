# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Agent RNN (PyMARL2 n_rnn) and QMIX mixing network."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarlcRNN(torch.nn.Module):
    """PyMARL2 n_rnn: ReLU-FC, GRUCell, linear Q head. No activation after the GRU."""

    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.rnn = torch.nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, n_actions)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def forward(
        self, x: torch.Tensor, hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        squeeze_time = x.dim() == 3
        if squeeze_time:
            x = x.squeeze(0)
        if hidden_state is None:
            hidden_state = self.initial_state(x.shape[0], x.device)
        hidden = hidden_state.reshape(x.shape[0], self.hidden_dim)
        hidden = self.rnn(F.relu(self.fc1(x)), hidden)
        q = self.fc2(hidden)
        new_hidden = hidden.unsqueeze(0)
        if squeeze_time:
            q = q.unsqueeze(0)
        return q, new_hidden


class QMixer(nn.Module):
    """QMIX mixing network (monotonic hypernetwork)."""

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        embed_dim: int = 32,
        hypernet_embed: int = 64,
        non_monotonic: bool = False,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim
        self.hypernet_embed = hypernet_embed
        self.non_monotonic = non_monotonic

        self.hyper_w_1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, embed_dim * num_agents),
        )
        self.hyper_w_final = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, embed_dim),
        )
        self.hyper_b_1 = nn.Linear(state_dim, embed_dim)
        self.V = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Mix per-agent Q-values into Q_tot.

        Args:
            agent_qs: [B, T, N] per-agent Q-values for chosen actions.
            states: [B, T, S] global environment states.

        Returns:
            q_tot: [B, T, 1] mixed team Q-value.
        """
        batch_size = agent_qs.shape[0]
        state_dim = states.shape[2:]

        agent_qs = agent_qs.reshape(-1, 1, self.num_agents)
        states = states.reshape(-1, *state_dim)

        w1 = self.hyper_w_1(states)
        if not self.non_monotonic:
            w1 = torch.abs(w1)
        b1 = self.hyper_b_1(states)
        w1 = w1.reshape(-1, self.num_agents, self.embed_dim)
        b1 = b1.reshape(-1, 1, self.embed_dim)
        hidden = F.elu(torch.matmul(agent_qs, w1) + b1)

        w_final = self.hyper_w_final(states)
        if not self.non_monotonic:
            w_final = torch.abs(w_final)
        w_final = w_final.reshape(-1, self.embed_dim, 1)
        v = self.V(states).reshape(-1, 1, 1)

        y = torch.matmul(hidden, w_final) + v
        return y.reshape(batch_size, -1, 1)
