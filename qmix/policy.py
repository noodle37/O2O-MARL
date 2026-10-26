# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Epsilon-greedy action selection during environment rollouts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


class EvalPolicyMixin:
    """Requires host: n_agents, n_actions, add_agent_id_to_obs, device, q_network."""

    @staticmethod
    def linear_epsilon(t_env: int, eps_start: float, eps_min: float, decay_steps: int) -> float:
        """Linear decay eps_start -> eps_min over decay_steps, clamped at eps_min."""
        if decay_steps <= 0 or t_env >= decay_steps:
            return eps_min
        return eps_start + (eps_min - eps_start) * (t_env / decay_steps)


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
        self.eval_offline_rnn = None
        self.eval_offline_last_actions = None
        offline_net = getattr(self, "offline_q_network", None)
        if offline_net is not None:
            self.eval_offline_rnn = {
                f"agent_{i}": offline_net.initial_state(1, self.device)
                for i in range(self.n_agents)
            }
            self.eval_offline_last_actions = {}

    def select_actions_eval(
        self,
        observations: Dict[str, np.ndarray],
        legals: Dict[str, np.ndarray],
        eps: float,
        state: Optional[Dict[str, Any]] = None,
        trace: Optional[Dict[str, Any]] = None,
        env_state=None,
    ) -> Dict[str, int]:
        del env_state
        if state is None:
            rnn_state = self.eval_rnn
            last_actions = self.eval_last_actions
        else:
            rnn_state = state["rnn"]
            last_actions = state["last"]
        epsilon_dec = 0.0 if eps <= 0.0 else eps / self.n_agents
        # Sequential exploration (OVMSE paper style): at each step exactly one
        # agent explores with the scheduled epsilon, all others act greedily.
        # With eps <= 0 this degenerates to the plain greedy policy.
        sequential = bool(getattr(self, "sequential_explore", False)) and eps > 0.0
        explorer = int(np.random.randint(self.n_agents)) if sequential else -1
        actions = {}
        chosen_q: List[float] = []
        offline_chosen_q: List[float] = []
        explored_flags: List[bool] = []
        n_disagree = 0
        score_offline = (
            trace is not None
            and state is None
            and getattr(self, "offline_q_network", None) is not None
            and getattr(self, "eval_offline_rnn", None) is not None
        )
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
            if sequential:
                explore_here = i == explorer and np.random.rand() < eps
            else:
                explore_here = np.random.rand() < epsilon_dec
            if explore_here:
                legal_idx = torch.nonzero(legal, as_tuple=False).squeeze(-1)
                a = int(legal_idx[torch.randint(legal_idx.numel(), (1,))].item())
            else:
                a = int(torch.argmax(q).item())
            explored_flags.append(bool(explore_here))
            last_actions[agent] = a
            actions[agent] = a
            if trace is not None:
                chosen_q.append(float(q[a].item()))
                if score_offline:
                    off_q, off_a = self._offline_q_and_greedy(
                        agent, i, observations[agent], legal, a
                    )
                    offline_chosen_q.append(off_q)
                    n_disagree += int(off_a != a)
        if trace is not None:
            trace["chosen_q"] = chosen_q
            trace["n_disagree"] = n_disagree
            trace["n_agents"] = len(chosen_q)
            if score_offline:
                trace["offline_chosen_q"] = offline_chosen_q
        if state is not None:
            # Consumed by collect_parallel; marks steps whose online action was
            # exploratory so ROAD can keep them out of Delta_on.
            state["explored"] = explored_flags
        return actions

    def _offline_q_and_greedy(
        self,
        agent: str,
        agent_index: int,
        observation: np.ndarray,
        legal: torch.Tensor,
        executed_action: int,
    ) -> tuple[float, int]:
        """Q and greedy action of the frozen offline net on the executed trajectory."""
        last_actions = self.eval_offline_last_actions
        last_idx = last_actions.get(agent)
        if last_idx is None:
            last_oh = torch.zeros(self.n_actions, device=self.device)
        else:
            last_oh = F.one_hot(
                torch.tensor(int(last_idx), device=self.device), self.n_actions
            ).float()
        pieces = [torch.from_numpy(observation).float().to(self.device), last_oh]
        if self.add_agent_id_to_obs:
            pieces.append(
                F.one_hot(torch.tensor(agent_index, device=self.device), self.n_agents).float()
            )
        obs_in = torch.cat(pieces, dim=-1).unsqueeze(0)
        q, self.eval_offline_rnn[agent] = self.offline_q_network(
            obs_in, self.eval_offline_rnn.get(agent)
        )
        q = q[0].masked_fill(~legal, -1e9)
        greedy = int(torch.argmax(q).item())
        last_actions[agent] = executed_action
        return float(q[executed_action].item()), greedy
