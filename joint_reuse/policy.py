"""Per-agent offline/online behaviour mixing for joint-reuse continuation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class HybridRetentionPolicy:
    """Rollout policy whose per-agent source is fixed for the whole run.

    Agents in ``offline_agents`` act greedily from the frozen offline network and
    update that network's recurrent history with its own candidate action. All
    other agents act greedily from the online network and update its history with
    the executed action, including exploration noise.
    """

    def __init__(
        self,
        learner,
        offline_agents,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        agents = tuple(sorted(set(int(a) for a in offline_agents)))
        if any(a < 0 or a >= learner.n_agents for a in agents):
            raise ValueError(f"offline agents {agents} outside [0, {learner.n_agents})")
        self.learner = learner
        self.offline_agents = frozenset(agents)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._steps = 0
        self._offline_agent_steps = 0

    def _offline_net(self):
        net = getattr(self.learner, "offline_q_network", None)
        if net is None:
            raise RuntimeError("retention branch requires learner.offline_q_network")
        return net

    def _fresh_source_state(self, net) -> Dict[str, Any]:
        return {
            "rnn": {
                f"agent_{i}": net.initial_state(1, self.learner.device)
                for i in range(self.learner.n_agents)
            },
            "last": {},
        }

    def new_episode_state(self) -> Dict[str, Any]:
        state = self.learner.new_episode_state()
        if self.offline_agents:
            state["offline"] = self._fresh_source_state(self._offline_net())
        return state

    def _legal(self, legals, agent: str) -> torch.Tensor:
        if legals is None or agent not in legals:
            return torch.ones(self.learner.n_actions, dtype=torch.bool, device=self.learner.device)
        return torch.from_numpy(np.asarray(legals[agent])).bool().to(self.learner.device)

    def _greedy(
        self,
        net,
        agent_index: int,
        agent: str,
        observation: np.ndarray,
        legal: torch.Tensor,
        last_actions: Dict[str, int],
        hidden,
    ) -> Tuple[int, Any]:
        last = last_actions.get(agent)
        if last is None:
            last_oh = torch.zeros(self.learner.n_actions, device=self.learner.device)
        else:
            last_oh = F.one_hot(
                torch.tensor(int(last), device=self.learner.device), self.learner.n_actions
            ).float()
        pieces = [
            torch.from_numpy(np.asarray(observation)).float().to(self.learner.device),
            last_oh,
        ]
        if self.learner.add_agent_id_to_obs:
            pieces.append(
                F.one_hot(
                    torch.tensor(agent_index, device=self.learner.device), self.learner.n_agents
                ).float()
            )
        obs_in = torch.cat(pieces, dim=-1).unsqueeze(0)
        with torch.inference_mode():
            q, next_hidden = net(obs_in, hidden)
            q = q[0].masked_fill(~legal, -1e9)
        return int(torch.argmax(q).item()), next_hidden

    def select_actions_eval(
        self,
        observations: Dict[str, np.ndarray],
        legals: Optional[Dict[str, np.ndarray]],
        eps: float,
        state: Optional[Dict[str, Any]] = None,
        trace: Optional[Dict[str, Any]] = None,
        env_state=None,
    ) -> Dict[str, int]:
        del env_state
        if state is None:
            raise ValueError("hybrid retention collection requires an episode state")
        online_state = state
        offline_state = state.get("offline")
        if self.offline_agents and offline_state is None:
            raise ValueError("episode state is missing its offline policy history")

        agents = sorted(observations.keys())
        sequential = bool(getattr(self.learner, "sequential_explore", False)) and eps > 0.0
        explorer = int(self.rng.integers(self.learner.n_agents)) if sequential else -1
        candidates: Dict[str, int] = {}
        offline_candidates: Dict[str, int] = {}
        explored = [False] * len(agents)

        for agent_index, agent in enumerate(agents):
            legal = self._legal(legals, agent)
            if agent_index in self.offline_agents:
                candidate, offline_state["rnn"][agent] = self._greedy(
                    self._offline_net(),
                    agent_index,
                    agent,
                    observations[agent],
                    legal,
                    offline_state["last"],
                    offline_state["rnn"].get(agent),
                )
            else:
                candidate, online_state["rnn"][agent] = self._greedy(
                    self.learner.q_network,
                    agent_index,
                    agent,
                    observations[agent],
                    legal,
                    online_state["last"],
                    online_state["rnn"].get(agent),
                )
            candidates[agent] = candidate
            if agent_index in self.offline_agents:
                offline_candidates[agent] = candidate

            if sequential:
                explore_here = agent_index == explorer and float(self.rng.random()) < eps
            else:
                explore_here = float(self.rng.random()) < eps / self.learner.n_agents
            if explore_here:
                legal_idx = torch.nonzero(legal, as_tuple=False).squeeze(-1)
                candidates[agent] = int(
                    legal_idx[int(self.rng.integers(legal_idx.numel()))].item()
                )
                explored[agent_index] = True

        for agent_index, agent in enumerate(agents):
            executed = candidates[agent]
            online_state["last"][agent] = executed
            if agent_index in self.offline_agents:
                offline_state["last"][agent] = offline_candidates[agent]
                self._offline_agent_steps += 1
        self._steps += len(agents)
        online_state["explored"] = explored

        if trace is not None:
            trace["retention_mask"] = [i in self.offline_agents for i in range(len(agents))]
        return candidates

    def stats(self, *, reset: bool = False) -> Dict[str, float]:
        steps = max(self._steps, 1)
        out = {
            "hybrid/steps": float(self._steps),
            "hybrid/offline_agent_steps": float(self._offline_agent_steps),
            "hybrid/offline_agent_share": float(self._offline_agent_steps / steps),
        }
        if reset:
            self._steps = 0
            self._offline_agent_steps = 0
        return out

    def __getattr__(self, name: str):
        return getattr(self.learner, name)
