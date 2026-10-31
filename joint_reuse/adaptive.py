"""Conflict-local offline action proposals driven by a learned response model."""

from __future__ import annotations

import pickle
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from scripts import run_verified_prefs as vp


class AdaptiveLocalReusePolicy:
    def __init__(
        self,
        learner,
        model_path: str,
        *,
        rng: Optional[np.random.Generator] = None,
        weight_fn=None,
    ) -> None:
        with open(model_path, "rb") as handle:
            artifact = pickle.load(handle)
        self.learner = learner
        self.treatment_agents = tuple(int(a) for a in artifact["treatment_agents"])
        self.feature_names = tuple(artifact["feature_names"])
        self.layout = artifact["layout"]
        self.ensemble = artifact["ensemble"]
        self.episode_limit = int(artifact["episode_limit"])
        self.rng = rng if rng is not None else np.random.default_rng()
        self.weight_fn = weight_fn
        self._episode_step = 0
        self._steps = 0
        self._conflicts = 0
        self._interventions = 0
        self._offline_agent_steps = 0
        self._weight_sum = 0.0
        self._forward_seconds = 0.0

    def _fresh_state(self, net):
        return {
            "rnn": {
                f"agent_{i}": net.initial_state(1, self.learner.device)
                for i in range(self.learner.n_agents)
            },
            "last": {},
        }

    def new_episode_state(self):
        self._episode_step = 0
        return {
            "online": self._fresh_state(self.learner.q_network),
            "offline": self._fresh_state(self.learner.offline_q_network),
        }

    def _input(self, observation, last_action, agent_index):
        pieces = [torch.from_numpy(np.asarray(observation)).float().to(self.learner.device)]
        if last_action is None:
            pieces.append(torch.zeros(self.learner.n_actions, device=self.learner.device))
        else:
            pieces.append(
                F.one_hot(
                    torch.tensor(int(last_action), device=self.learner.device),
                    self.learner.n_actions,
                ).float()
            )
        if self.learner.add_agent_id_to_obs:
            pieces.append(
                F.one_hot(
                    torch.tensor(agent_index, device=self.learner.device),
                    self.learner.n_agents,
                ).float()
            )
        return torch.cat(pieces, dim=-1).unsqueeze(0)

    def _weight(self, features):
        if self.weight_fn is not None:
            return float(self.weight_fn(features))
        x = np.asarray([[features[k] for k in self.feature_names]], dtype=np.float64)
        predictions = [float(model.predict(scaler.transform(x))[0]) for scaler, model in self.ensemble]
        return float(np.mean(np.asarray(predictions) > 0.0))

    def select_actions_eval(
        self,
        observations,
        legals,
        eps,
        state=None,
        trace=None,
        env_state=None,
    ):
        if state is None or env_state is None:
            raise ValueError("adaptive collection requires episode state and global env state")
        agents = sorted(observations.keys())
        online_state = state["online"]
        offline_state = state["offline"]
        online_actions = []
        offline_actions = []
        online_qs = []
        offline_qs = []
        started = time.perf_counter()
        with torch.inference_mode():
            for i, agent in enumerate(agents):
                legal = None
                if legals is not None:
                    legal = torch.from_numpy(np.asarray(legals[agent])).bool().to(self.learner.device)
                online_q, online_state["rnn"][agent] = self.learner.q_network(
                    self._input(observations[agent], online_state["last"].get(agent), i),
                    online_state["rnn"].get(agent),
                )
                offline_q, offline_state["rnn"][agent] = self.learner.offline_q_network(
                    self._input(observations[agent], offline_state["last"].get(agent), i),
                    offline_state["rnn"].get(agent),
                )
                online_q = online_q.reshape(-1)
                offline_q = offline_q.reshape(-1)
                online_qs.append(online_q)
                offline_qs.append(offline_q)
                online_actions.append(vp._greedy(online_q, legal))
                offline_actions.append(vp._greedy(offline_q, legal))
        self._forward_seconds += time.perf_counter() - started

        executed = list(online_actions)
        conflict = any(online_actions[i] != offline_actions[i] for i in self.treatment_agents)
        intervened = False
        weight = 0.0
        if conflict:
            global_state = np.asarray(env_state, dtype=np.float32)
            allies, enemies = vp.prf.alive_counts(global_state, self.layout)

            def gather(qs, actions):
                return [float(qs[i][actions[i]].item()) for i in range(self.learner.n_agents)]

            with torch.inference_mode():
                on_live = vp._mix(
                    self.learner.mixer, gather(online_qs, online_actions), global_state, self.learner.device
                )
                on_off = vp._mix(
                    self.learner.mixer, gather(online_qs, offline_actions), global_state, self.learner.device
                )
                off_live = vp._mix(
                    self.learner.offline_mixer,
                    gather(offline_qs, online_actions),
                    global_state,
                    self.learner.device,
                )
                off_off = vp._mix(
                    self.learner.offline_mixer,
                    gather(offline_qs, offline_actions),
                    global_state,
                    self.learner.device,
                )
            features = {
                "allies": int(allies),
                "enemies": int(enemies),
                "phase": float(self._episode_step) / float(max(1, self.episode_limit)),
                "n_disagree": int(sum(a != b for a, b in zip(online_actions, offline_actions))),
                "q_off_margin": off_off - off_live,
                "q_on_margin": on_live - on_off,
                "abs_q_off_margin": abs(off_off - off_live),
                "abs_q_on_margin": abs(on_live - on_off),
                "q_off_live": off_live,
                "q_off_off": off_off,
                "q_on_live": on_live,
                "q_on_off": on_off,
            }
            weight = self._weight(features)
            self._conflicts += 1
            self._weight_sum += weight
            if self.rng.random() < weight:
                for i in self.treatment_agents:
                    executed[i] = offline_actions[i]
                intervened = True
                self._interventions += 1
                self._offline_agent_steps += len(self.treatment_agents)

        sequential = bool(getattr(self.learner, "sequential_explore", False)) and eps > 0.0
        explorer = int(self.rng.integers(self.learner.n_agents)) if sequential else -1
        explored = [False] * len(agents)
        for i, agent in enumerate(agents):
            legal = None
            if legals is not None:
                legal = torch.from_numpy(np.asarray(legals[agent])).bool().to(self.learner.device)
            explore_here = (
                i == explorer if sequential else True
            ) and float(self.rng.random()) < (eps if sequential else eps / self.learner.n_agents)
            if explore_here:
                legal_idx = torch.nonzero(legal, as_tuple=False).squeeze(-1)
                executed[i] = int(legal_idx[int(self.rng.integers(legal_idx.numel()))].item())
                explored[i] = True
            online_state["last"][agent] = int(executed[i])
            offline_state["last"][agent] = int(executed[i])

        self._steps += 1
        self._episode_step += 1
        online_state["explored"] = explored
        state["explored"] = explored
        if trace is not None:
            trace["retention_mask"] = [
                intervened and i in self.treatment_agents for i in range(len(agents))
            ]
            trace["adaptive_weight"] = weight
        return {agent: int(executed[i]) for i, agent in enumerate(agents)}

    def stats(self, *, reset=False):
        out = {
            "adaptive/steps": float(self._steps),
            "adaptive/conflicts": float(self._conflicts),
            "adaptive/interventions": float(self._interventions),
            "adaptive/offline_agent_steps": float(self._offline_agent_steps),
            "adaptive/weight_sum": float(self._weight_sum),
            "adaptive/conflict_share": self._conflicts / max(1, self._steps),
            "adaptive/mean_conflict_weight": self._weight_sum / max(1, self._conflicts),
            "adaptive/intervention_share": self._interventions / max(1, self._steps),
            "adaptive/offline_agent_share": self._offline_agent_steps
            / max(1, self._steps * self.learner.n_agents),
            "adaptive/forward_seconds": self._forward_seconds,
        }
        if reset:
            self._steps = 0
            self._conflicts = 0
            self._interventions = 0
            self._offline_agent_steps = 0
            self._weight_sum = 0.0
            self._forward_seconds = 0.0
        return out

    def __getattr__(self, name):
        return getattr(self.learner, name)
