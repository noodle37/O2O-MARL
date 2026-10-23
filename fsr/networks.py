"""History encoder, teammate-action encoding, and a per-state scale network."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from macflow.networks import FlowField


class HistoryEncoder(nn.Module):
    """GRU over augmented local observations; output is the hidden state (τ_i)."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)

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
        new_hidden = hidden.unsqueeze(0)
        out = hidden.unsqueeze(0) if squeeze_time else hidden
        return out, new_hidden


class ScaleNet(nn.Module):
    """Per-state positive scale α(s) that aligns ∇Q with the flow score."""

    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(state_dim), int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), 1),
            nn.Softplus(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class SupportFlowBank(nn.Module):
    """Per-agent (or shared) conditional flow field v_θ(x_t, t | τ_i)."""

    def __init__(
        self,
        n_agents: int,
        cond_dim: int,
        teammate_dim: int,
        hidden: List[int],
        *,
        shared: bool = False,
        t_freqs: int = 8,
    ):
        super().__init__()
        self.n_agents = int(n_agents)
        self.shared = bool(shared)
        self.teammate_dim = int(teammate_dim)
        n_fields = 1 if self.shared else self.n_agents
        self.fields = nn.ModuleList(
            [
                FlowField(cond_dim, self.teammate_dim, hidden, layer_norm=False, t_freqs=t_freqs)
                for _ in range(n_fields)
            ]
        )

    def velocity(
        self, agent_i: int, cond: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        idx = 0 if self.shared else int(agent_i)
        return self.fields[idx](cond, x, t)


def teammate_onehot(actions: torch.Tensor, agent_i: int, n_actions: int) -> torch.Tensor:
    """Concatenate one-hot teammate actions. actions [..., N, 1] or [..., N]."""
    if actions.dim() >= 1 and actions.shape[-1] == 1:
        idx = actions.squeeze(-1)
    else:
        idx = actions
    oh = F.one_hot(idx.long().clamp(0, n_actions - 1), n_actions).float()
    others = torch.cat([oh[..., :agent_i, :], oh[..., agent_i + 1 :, :]], dim=-2)
    return others.reshape(*others.shape[:-2], -1)


def splice_teammate_onehot(
    own_onehot: torch.Tensor,
    teammate_flat: torch.Tensor,
    agent_i: int,
) -> torch.Tensor:
    """Put a flat teammate one-hot back into a [..., N, A] joint one-hot tensor."""
    n_actions = own_onehot.shape[-1]
    n_agents = own_onehot.shape[-2]
    others = teammate_flat.reshape(*teammate_flat.shape[:-1], n_agents - 1, n_actions)
    return torch.cat(
        [others[..., :agent_i, :], own_onehot[..., agent_i : agent_i + 1, :], others[..., agent_i:, :]],
        dim=-2,
    )
