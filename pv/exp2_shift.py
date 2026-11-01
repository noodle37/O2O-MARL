# Copyright 2024 InstaDeep Ltd. All rights reserved.

"""Experiment 2a: controlled teammate shift (evaluation only, no training).

One examined agent (or pair) keeps the offline parameters and acts greedily; the
remaining teammates are shifted in one of two ways:

  N  natural shift    teammates use arm A's checkpoint at t in
                      {0, 25k, 50k, 100k, 250k, 500k, 1M}.  This is the shift a
                      real fine-tuning run produces, and the teammates get
                      stronger.
  R  random shift     teammates keep the offline parameters but take a uniformly
                      random legal action with probability eps in
                      {0.05, 0.1, 0.2, 0.3, 0.5}.  They only change; they do not
                      get stronger.

R separates "deviates from the data" from "teammate got stronger": a regularity
that appears under both N and R is attributed to the deviation itself.

Per-step records (both families):
  * team deviation D(s) under the joint behaviour model;
  * Q^off_tot, the frozen offline Q_tot of the *executed* joint action, computed
    as a shadow forward with its own RNN state (so it is the offline value of
    what actually happened, not of the offline greedy action);
  * G, the Monte-Carlo return-to-go along the episode.

Behavioural validity (N only), with R(.) the mean normalised episode return:
  * single-agent transplant gap   g_i(t) = R(all@t) - R(i@off, rest@t);
  * pairwise interaction          I_ij(t) = R(off_i, off_j | rest@t)
                                           - R(off_i | rest@t) - R(off_j | rest@t)
                                           + R(all@t),
    the interaction term of a 2x2 factorial design.  I_ij > 0 means restoring two
    offline agents together is worth more than the sum of restoring each alone,
    i.e. synergy that cannot be reduced to individual preferences.

Per-agent weight transplantation: both the live net (from the shifted
checkpoint) and the frozen offline net are forwarded every step, each with its
own RNN state, and each agent's Q is taken from the source its configuration
names.  The mixer is the one of the configuration's "rest" group, matching the
design (the examined agents are transplanted, the team's value function is not).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pv.behaviour import JointBehaviourModel, load_behaviour_model
from pv.probe import mc_return_to_go
from training.rollout_helpers import legals_from_info
from utils.paths import resolve_path


# Shift schedule shared by both maps (design section 4.1).
NATURAL_SHIFT_STEPS = (0, 25000, 50000, 100000, 250000, 500000, 1000000)
RANDOM_SHIFT_EPS = (0.05, 0.1, 0.2, 0.3, 0.5)


@dataclass
class Net:
    """One QMIX (agent net + mixer) pair with its own RNN state per agent."""

    q_network: torch.nn.Module
    mixer: torch.nn.Module

    @classmethod
    def from_learner(cls, learner) -> "Net":
        return cls(q_network=learner.q_network, mixer=learner.mixer)

    def initial_state(self, n_agents: int, device) -> List[torch.Tensor]:
        return [self.q_network.initial_state(1, device) for _ in range(n_agents)]


def _agent_input(obs: np.ndarray, last_action: Optional[int], agent_index: int,
                 n_actions: int, n_agents: int, add_agent_id: bool, device) -> torch.Tensor:
    """Mirror qmix/policy.py select_actions_eval's per-agent input construction."""
    pieces = [torch.from_numpy(np.asarray(obs)).float().to(device)]
    if last_action is None:
        pieces.append(torch.zeros(n_actions, device=device))
    else:
        pieces.append(
            F.one_hot(torch.tensor(int(last_action), device=device), n_actions).float()
        )
    if add_agent_id:
        pieces.append(F.one_hot(torch.tensor(agent_index, device=device), n_agents).float())
    return torch.cat(pieces, dim=-1).unsqueeze(0)


def _forward_agent(net: Net, obs_in: torch.Tensor, hidden: torch.Tensor):
    q, new_hidden = net.q_network(obs_in, hidden)
    return q, new_hidden


@dataclass
class EpisodeRecord:
    """Per-step arrays for one evaluated episode."""

    steps: List[dict] = field(default_factory=list)
    episode_return: float = 0.0
    won: Optional[bool] = None
    length: int = 0
    perturb_trials: Dict[int, int] = field(default_factory=dict)
    perturb_events: Dict[int, int] = field(default_factory=dict)
    perturb_changes: Dict[int, int] = field(default_factory=dict)


def _hash_uniform(*values: int) -> float:
    """FNV-1a hash mapped to [0, 1), so noise is common across conditions."""
    h = 2166136261
    for value in values:
        h ^= int(value) & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
    return h / 4294967296.0


def run_episode(
    env,
    *,
    live: Net,
    offline: Net,
    sources: Dict[int, str],
    examined: set,
    eps_partner: float,
    n_actions: int,
    add_agent_id: bool,
    device,
    rng: np.random.Generator,
    noise_seed: Optional[int] = None,
    episode_index: int = 0,
) -> EpisodeRecord:
    """Roll out one episode with the given per-agent weight sources.

    sources maps agent index -> "live" or "offline" and decides which network
    supplies each agent's Q.  Agents in `examined` always act greedily; the rest
    (partners) take a uniformly random legal action with probability
    eps_partner.  Under the natural shift eps_partner is 0, under the random
    shift it is the eps being probed.

    Deviation D(s) is not computed here: it needs the whole episode's actions
    under one agent order, so it is scored afterwards by score_records().
    """
    n_agents = len(sources)
    hidden_live = live.initial_state(n_agents, device)
    hidden_off = offline.initial_state(n_agents, device)
    last_live: List[Optional[int]] = [None] * n_agents
    last_off: List[Optional[int]] = [None] * n_agents
    # Shadow forward keeps its own RNN state, independent of the policy's.
    shadow_hidden = offline.initial_state(n_agents, device)
    shadow_last: List[Optional[int]] = [None] * n_agents

    obs, info = env.reset()
    record = EpisodeRecord()
    done = False
    step = 0
    while not done:
        agents = sorted(obs.keys())
        legals = legals_from_info(info, agents)
        actions: Dict[str, int] = {}
        for i, agent in enumerate(agents):
            legal = None
            if legals is not None:
                legal = torch.from_numpy(np.asarray(legals[agent])).bool().to(device)
            use_offline = sources.get(i, "live") == "offline"
            if use_offline:
                obs_in = _agent_input(
                    obs[agent], last_off[i], i, n_actions, n_agents, add_agent_id, device
                )
                q, hidden_off[i] = _forward_agent(offline, obs_in, hidden_off[i])
            else:
                obs_in = _agent_input(
                    obs[agent], last_live[i], i, n_actions, n_agents, add_agent_id, device
                )
                q, hidden_live[i] = _forward_agent(live, obs_in, hidden_live[i])
            if legal is not None:
                q = q[0].masked_fill(~legal, -1e9)
            greedy_action = int(torch.argmax(q).item())
            action = greedy_action
            if i not in examined and eps_partner > 0.0:
                record.perturb_trials[i] = record.perturb_trials.get(i, 0) + 1
                if noise_seed is None:
                    perturbed = rng.random() < eps_partner
                else:
                    perturbed = (
                        _hash_uniform(noise_seed, episode_index, step, i, 0) < eps_partner
                    )
                if perturbed:
                    record.perturb_events[i] = record.perturb_events.get(i, 0) + 1
                    if legal is not None:
                        legal_idx = torch.nonzero(legal, as_tuple=False).squeeze(-1)
                        if noise_seed is None:
                            pick = int(torch.randint(legal_idx.numel(), (1,)).item())
                        else:
                            pick = int(
                                _hash_uniform(noise_seed, episode_index, step, i, 1)
                                * legal_idx.numel()
                            )
                            pick = min(pick, legal_idx.numel() - 1)
                        action = int(legal_idx[pick].item())
                    elif noise_seed is None:
                        action = int(rng.integers(n_actions))
                    else:
                        action = int(
                            _hash_uniform(noise_seed, episode_index, step, i, 1) * n_actions
                        )
                        action = min(action, n_actions - 1)
                    if action != greedy_action:
                        record.perturb_changes[i] = record.perturb_changes.get(i, 0) + 1
            actions[agent] = action
            if use_offline:
                last_off[i] = action
            else:
                last_live[i] = action

        state = info.get("state")
        # Shadow forward: the frozen offline Q_tot of the executed joint action.
        off_qtot = None
        if state is not None:
            agent_qs = []
            for i, agent in enumerate(agents):
                legal = None
                if legals is not None:
                    legal = torch.from_numpy(np.asarray(legals[agent])).bool().to(device)
                obs_in = _agent_input(
                    obs[agent], shadow_last[i], i, n_actions, n_agents, add_agent_id, device
                )
                q, shadow_hidden[i] = _forward_agent(offline, obs_in, shadow_hidden[i])
                if legal is not None:
                    q = q[0].masked_fill(~legal, -1e9)
                # Single-step forward returns [1, n_actions]; index the flat vector.
                agent_qs.append(float(q.reshape(-1)[actions[agent]].item()))
                shadow_last[i] = actions[agent]
            qs = torch.tensor(agent_qs, dtype=torch.float32, device=device).view(1, 1, n_agents)
            st = torch.from_numpy(np.asarray(state)).float().to(device).view(1, 1, -1)
            off_qtot = float(offline.mixer(qs, st).reshape(-1)[0].item())

        next_obs, rewards, terminal, trunc, next_info = env.step(actions)
        reward = float(np.mean(list(rewards.values())))
        record.steps.append(
            {
                "obs": np.stack([obs[a] for a in agents]),
                "actions": np.array([actions[a] for a in agents], dtype=np.int64),
                "state": np.asarray(state) if state is not None else None,
                "legals": (
                    np.stack([np.asarray(legals[a]) for a in agents])
                    if legals is not None
                    else None
                ),
                "q_off_tot": off_qtot,
                "reward": reward,
            }
        )
        record.episode_return += reward
        if "battle_won" in next_info:
            record.won = bool(next_info["battle_won"])
        obs, info = next_obs, next_info
        done = all(terminal.values()) or all(trunc.values())
        step += 1

    record.length = len(record.steps)
    return record


def _stack_steps(steps: Sequence[dict], max_seq: int, device) -> Dict[str, torch.Tensor]:
    """PyMARL-layout batch from raw step dicts (single episode)."""
    n = 1
    n_agents = steps[0]["obs"].shape[0]
    obs_dim = steps[0]["obs"].shape[1]
    state_dim = steps[0]["state"].shape[0] if steps[0]["state"] is not None else 1
    has_legals = steps[0]["legals"] is not None
    n_actions = steps[0]["legals"].shape[1] if has_legals else 1
    obs = torch.zeros(n, max_seq, n_agents, obs_dim, device=device)
    actions = torch.zeros(n, max_seq, n_agents, 1, dtype=torch.long, device=device)
    state = torch.zeros(n, max_seq, state_dim, device=device)
    avail = (
        torch.zeros(n, max_seq, n_agents, n_actions, dtype=torch.bool, device=device)
        if has_legals
        else None
    )
    reward = torch.zeros(n, max_seq, 1, device=device)
    term = torch.zeros(n, max_seq, 1, device=device)
    filled = torch.zeros(n, max_seq, 1, device=device)
    length = len(steps)
    for t, st in enumerate(steps):
        obs[0, t] = torch.from_numpy(st["obs"]).float()
        actions[0, t, :, 0] = torch.from_numpy(st["actions"]).long()
        if st["state"] is not None:
            state[0, t] = torch.from_numpy(st["state"]).float()
        if avail is not None and st["legals"] is not None:
            avail[0, t] = torch.from_numpy(st["legals"]).bool()
    # A single-episode record has no per-step reward stored; the caller supplies
    # returns through `returns`, so reward stays zero here.
    filled[0, :length, 0] = 1.0
    if length < max_seq:
        obs[0, length] = obs[0, length - 1]
        state[0, length] = state[0, length - 1]
        if avail is not None:
            avail[0, length] = avail[0, length - 1]
        term[0, length, 0] = 1.0
    batch = {
        "obs": obs,
        "actions": actions,
        "state": state,
        "reward": reward,
        "terminated": term,
        "filled": filled,
    }
    batch["avail_actions"] = (
        avail if avail is not None else torch.ones(n, max_seq, n_agents, 1, dtype=torch.bool, device=device)
    )
    return batch


def score_records(
    steps: Sequence[dict],
    returns: np.ndarray,
    behaviour: Optional[JointBehaviourModel],
    device,
    gamma: float,
) -> Dict[str, np.ndarray]:
    """Attach D(s) and align Q^off_tot with the MC return-to-go."""
    n_steps = len(steps)
    team = np.full(n_steps, np.nan)
    if behaviour is not None:
        max_seq = n_steps + 1
        batch = _stack_steps(steps, max_seq, device)
        dev = behaviour.deviations(batch)
        mask = dev["mask"][0].detach().cpu().numpy()
        team_vals = dev["team"][0].detach().cpu().numpy()
        team[mask[:n_steps]] = team_vals[:n_steps][mask[:n_steps]]
    q_off = np.array(
        [float(s["q_off_tot"]) if s["q_off_tot"] is not None else np.nan for s in steps]
    )
    return {
        "team_deviation": team,
        "q_off_tot": q_off,
        "return_to_go": returns,
    }
