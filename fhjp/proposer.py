# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Joint behaviour proposal: two complete joint candidates, one executed at p.

The offline Q network only ranks actions: its values never enter a training
target, and no behaviour propensity is computed or stored. Each step proposes the
offline joint greedy action and the online joint greedy action; when they differ
one of them is executed with probability p (0.5), so the online joint behaviour
always remains as a control branch.

Action histories are deliberately asymmetric, matching the algorithm spec:

* the online net is conditioned on the **executed** actions, because that is the
  history the deployed policy actually faces;
* the offline net is conditioned on **its own** greedy actions, i.e. the closed
  loop of pi_off, which is also what the offline evaluation arm runs.

Because QMIX's mixer is monotonic in every agent Q, the per-agent argmax is the
joint argmax, so no joint-action search is needed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def legal_mask(learner, legals: Optional[Dict[str, Any]], agent: str) -> torch.Tensor:
    """Boolean legal-action mask; all-legal when the env exposes no legals."""
    if legals is None or agent not in legals:
        return torch.ones(learner.n_actions, dtype=torch.bool, device=learner.device)
    return torch.from_numpy(np.asarray(legals[agent])).bool().to(learner.device)


def agent_q(
    net: torch.nn.Module,
    learner,
    agent_index: int,
    agent: str,
    observation: Any,
    legal: torch.Tensor,
    last_action: Optional[int],
    hidden,
) -> Tuple[torch.Tensor, Any]:
    """One agent's masked Q values plus the next hidden state.

    Input order matches PyMARL: observation, previous action one-hot, agent id.
    """
    if last_action is None:
        last_oh = torch.zeros(learner.n_actions, device=learner.device)
    else:
        last_oh = F.one_hot(
            torch.tensor(int(last_action), device=learner.device), learner.n_actions
        ).float()
    pieces = [torch.from_numpy(np.asarray(observation)).float().to(learner.device), last_oh]
    if learner.add_agent_id_to_obs:
        pieces.append(
            F.one_hot(torch.tensor(agent_index, device=learner.device), learner.n_agents).float()
        )
    obs_in = torch.cat(pieces, dim=-1).unsqueeze(0)
    with torch.inference_mode():
        q, hidden = net(obs_in, hidden)
        q = q[0].masked_fill(~legal, -1e9)
    return q, hidden


def joint_greedy(
    net: torch.nn.Module,
    learner,
    observations: Dict[str, Any],
    legals: Optional[Dict[str, Any]],
    policy_state: Dict[str, Any],
) -> Dict[str, int]:
    """Joint greedy action of one net.

    Advances ``policy_state["rnn"]`` but deliberately leaves
    ``policy_state["last"]`` untouched: the caller decides which action history
    each net is conditioned on (executed actions for the online net, its own
    greedy actions for the offline net).
    """
    actions: Dict[str, int] = {}
    for i, agent in enumerate(sorted(observations.keys())):
        legal = legal_mask(learner, legals, agent)
        q, policy_state["rnn"][agent] = agent_q(
            net,
            learner,
            i,
            agent,
            observations[agent],
            legal,
            policy_state["last"].get(agent),
            policy_state["rnn"].get(agent),
        )
        action = int(q.argmax().item())
        actions[agent] = action
    return actions


def fresh_policy_state(net: torch.nn.Module, learner) -> Dict[str, Any]:
    """Initial RNN hidden states and an empty action history for one net."""
    return {
        "rnn": {
            f"agent_{i}": net.initial_state(1, learner.device)
            for i in range(learner.n_agents)
        },
        "last": {},
    }


class JointProposalPolicy:
    """Drop-in behaviour policy for ``collect_parallel``.

    Implements exactly the two methods the rollout needs (``new_episode_state``
    and ``select_actions_eval``) and forwards everything else to the learner, so
    the standard evaluation path, which calls the learner directly, keeps using
    the plain epsilon-greedy policy with no proposal randomisation.
    """

    def __init__(self, learner, *, p: float = 0.5) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"proposal probability p must be in [0, 1], got {p}")
        if getattr(learner, "offline_q_network", None) is None:
            raise RuntimeError(
                "joint proposal needs a frozen offline_q_network on the learner; "
                "enable transfer.load_offline_nets or disable the proposer"
            )
        self.learner = learner
        self.p = float(p)
        self.proposer_on = True
        self._fallback_state: Optional[Dict[str, Any]] = None
        self._steps = 0
        self._conflicts = 0
        self._offline_selected = 0
        self._online_selected = 0

    # ------------------------------------------------------------------ setup

    def _online_net(self) -> torch.nn.Module:
        return self.learner.q_network

    def _offline_net(self) -> torch.nn.Module:
        net = getattr(self.learner, "offline_q_network", None)
        if net is None:
            raise RuntimeError(
                "joint proposal needs a frozen offline_q_network on the learner; "
                "enable transfer.load_offline_nets or disable the proposer"
            )
        return net

    def new_episode_state(self) -> Dict[str, Any]:
        state = self.learner.new_episode_state()
        state["offline"] = fresh_policy_state(self._offline_net(), self.learner)
        return state

    def _policy_states(self, state: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if state is not None:
            if "offline" not in state:
                state["offline"] = fresh_policy_state(self._offline_net(), self.learner)
            return state, state["offline"]
        if self._fallback_state is None:
            self._fallback_state = fresh_policy_state(self._online_net(), self.learner)
            self._fallback_state["offline"] = fresh_policy_state(
                self._offline_net(), self.learner
            )
        return self._fallback_state, self._fallback_state["offline"]

    # ------------------------------------------------------------- behaviour

    def select_actions_eval(
        self,
        observations: Dict[str, Any],
        legals: Optional[Dict[str, Any]],
        eps: float,
        state: Optional[Dict[str, Any]] = None,
        trace: Optional[Dict[str, Any]] = None,
        env_state=None,
    ) -> Dict[str, int]:
        if not self.proposer_on:
            return self.learner.select_actions_eval(
                observations, legals, eps, state=state, trace=trace, env_state=env_state
            )

        online_state, offline_state = self._policy_states(state)
        a_off = joint_greedy(
            self._offline_net(), self.learner, observations, legals, offline_state
        )
        a_on = joint_greedy(
            self._online_net(), self.learner, observations, legals, online_state
        )

        conflict = a_off != a_on
        if conflict:
            take_offline = bool(np.random.random() < self.p)
            candidate = a_off if take_offline else a_on
            source = "offline" if take_offline else "online"
        else:
            candidate = a_on
            source = "none"

        executed = self._explore(candidate, legals, eps)

        # Online history follows what was executed; offline history follows the
        # closed loop of pi_off.
        for agent in candidate:
            online_state["last"][agent] = executed[agent]
            offline_state["last"][agent] = a_off[agent]

        self._record(source, conflict)
        if trace is not None:
            trace["proposal_source"] = source
            trace["proposal_conflict"] = bool(conflict)
        return executed

    def _explore(
        self, candidate: Dict[str, int], legals: Optional[Dict[str, Any]], eps: float
    ) -> Dict[str, int]:
        """Existing exploration mechanism, greedy baseline = chosen candidate.

        Mirrors ``EvalPolicyMixin.select_actions_eval``: sequential exploration
        picks one agent per step, the independent branch uses eps / n_agents.
        """
        if eps <= 0.0:
            return dict(candidate)
        actions = dict(candidate)
        sequential = bool(getattr(self.learner, "sequential_explore", False))
        explorer = int(np.random.randint(self.learner.n_agents)) if sequential else -1
        for i, agent in enumerate(sorted(candidate.keys())):
            if sequential:
                explore_here = i == explorer and np.random.random() < eps
            else:
                explore_here = np.random.random() < eps / self.learner.n_agents
            if not explore_here:
                continue
            legal = legal_mask(self.learner, legals, agent)
            legal_idx = torch.nonzero(legal, as_tuple=False).squeeze(-1)
            actions[agent] = int(legal_idx[torch.randint(legal_idx.numel(), (1,))].item())
        return actions

    # ---------------------------------------------------------- diagnostics

    def _record(self, source: str, conflict: bool) -> None:
        self._steps += 1
        self._conflicts += int(bool(conflict))
        if source == "offline":
            self._offline_selected += 1
        elif source == "online":
            self._online_selected += 1

    def stats(self, *, reset: bool = False) -> Dict[str, float]:
        """Proposal diagnostics.

        ``offline_select_rate`` is conditioned on conflicting decisions, the only
        set where the 0.5 protocol applies; the all-steps variant additionally
        depends on the conflict rate and is reported separately.
        """
        steps = max(self._steps, 1)
        conflicts = max(self._conflicts, 1)
        out = {
            "proposer/steps": float(self._steps),
            "proposer/conflicts": float(self._conflicts),
            "proposer/conflict_rate": float(self._conflicts / steps),
            "proposer/offline_select_rate": float(self._offline_selected / conflicts),
            "proposer/online_select_rate": float(self._online_selected / conflicts),
            "proposer/offline_select_rate_all_steps": float(self._offline_selected / steps),
        }
        if reset:
            self._steps = 0
            self._conflicts = 0
            self._offline_selected = 0
            self._online_selected = 0
        return out

    def disable(self) -> None:
        """Permanent exit: fall back to the plain online behaviour policy."""
        self.proposer_on = False

    def __getattr__(self, name: str):
        # Only called for attributes not defined here, so the learner's own
        # methods and attributes stay reachable through the proxy.
        return getattr(self.learner, name)
