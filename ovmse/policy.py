# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Epsilon-greedy action selection during environment rollouts."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


class EvalPolicyMixin:
    """Requires host: n_agents, n_actions, add_agent_id_to_obs, device, q_network."""

    def new_episode_state(self) -> Dict[str, Any]:
        return {
            "rnn": {
                f"agent_{i}": self.q_network.initial_state(1, self.device)
                for i in range(self.n_agents)
            },
            "last": {},
        }

    def reset_eval_hidden(self) -> None:
        state = self.new_episode_state()
        self.eval_rnn = state["rnn"]
        self.eval_last_actions = state["last"]

    def select_actions_eval(
        self,
        observations: Dict[str, np.ndarray],
        legals: Dict[str, np.ndarray],
        eps: float,
        state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        if state is None:
            rnn_state = self.eval_rnn
            last_actions = self.eval_last_actions
        else:
            rnn_state = state["rnn"]
            last_actions = state["last"]
        epsilon_dec = 0.0 if eps <= 0.0 else eps / self.n_agents
        actions = {}
        for i, agent in enumerate(sorted(observations.keys())):
            obs = torch.from_numpy(observations[agent]).float().to(self.device)
            last_idx = last_actions.get(agent)
            if last_idx is None:
                last_oh = torch.zeros(self.n_actions, device=self.device)
            else:
                last_oh = F.one_hot(
                    torch.tensor(int(last_idx), device=self.device), self.n_actions
                ).float()
            pieces = [obs, last_oh]
            if self.add_agent_id_to_obs:
                pieces.append(F.one_hot(torch.tensor(i, device=self.device), self.n_agents).float())
            obs_in = torch.cat(pieces, dim=-1).unsqueeze(0)
            q, rnn_state[agent] = self.q_network(obs_in, rnn_state.get(agent))
            legal = torch.from_numpy(legals[agent]).bool().to(self.device)
            q = q[0].masked_fill(~legal, -1e9)
            if np.random.rand() < epsilon_dec:
                legal_idx = torch.nonzero(legal, as_tuple=False).squeeze(-1)
                a = int(legal_idx[torch.randint(legal_idx.numel(), (1,))].item())
            else:
                a = int(torch.argmax(q).item())
            last_actions[agent] = a
            actions[agent] = a
        return actions
