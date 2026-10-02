# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Online episode collection and batch stacking (PyMARL tensor layout)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import numpy as np
import torch

from qmix.core import QMIXBase


def td_terminated_flag(terminal: Dict[str, np.ndarray], info: Dict[str, Any]) -> float:
    """1.0 only for true episode termination; 0.0 on SMAC timeout (episode_limit)."""
    if not all(terminal.values()):
        return 0.0
    if info.get("episode_limit", False):
        return 0.0
    return 1.0


def collect_parallel(envs, learner: QMIXBase, eps: float) -> List[Dict[str, Any]]:
    n = len(envs)
    states = [learner.new_episode_state() for _ in range(n)]
    episodes = [{"steps": [], "length": 0} for _ in range(n)]
    obs_info = [env.reset() for env in envs]
    done = [False] * n
    pool = ThreadPoolExecutor(max_workers=n)

    def step_one(i: int, actions):
        if done[i]:
            return None
        return envs[i].step(actions)

    try:
        while not all(done):
            actions = [None] * n
            for i in range(n):
                if done[i]:
                    continue
                obs, info = obs_info[i]
                actions[i] = learner.select_actions_eval(obs, info["legals"], eps, states[i])
            results = list(pool.map(lambda i: step_one(i, actions[i]), range(n)))
            for i, result in enumerate(results):
                if result is None:
                    continue
                obs, info = obs_info[i]
                next_obs, rewards, terminal, trunc, next_info = result
                episodes[i]["steps"].append(
                    {
                        "obs": np.stack([obs[a] for a in sorted(obs.keys())]),
                        "actions": np.array([actions[i][a] for a in sorted(obs.keys())], dtype=np.int64),
                        "state": info["state"],
                        "legals": np.stack([info["legals"][a] for a in sorted(info["legals"].keys())]),
                        "reward": float(np.mean(list(rewards.values()))),
                        "terminated": td_terminated_flag(terminal, next_info),
                    }
                )
                obs_info[i] = (next_obs, next_info)
                if all(terminal.values()) or all(trunc.values()):
                    done[i] = True
                    episodes[i]["length"] = len(episodes[i]["steps"])
    finally:
        pool.shutdown(wait=True)
    return episodes


def collect_episode(env, learner: QMIXBase, eps: float) -> Dict[str, Any]:
    learner.reset_eval_hidden()
    obs, info = env.reset()
    steps = []
    done = False
    while not done:
        actions = learner.select_actions_eval(obs, info["legals"], eps)
        next_obs, rewards, terminal, trunc, next_info = env.step(actions)
        steps.append(
            {
                "obs": np.stack([obs[a] for a in sorted(obs.keys())]),
                "actions": np.array([actions[a] for a in sorted(obs.keys())], dtype=np.int64),
                "state": info["state"],
                "legals": np.stack([info["legals"][a] for a in sorted(info["legals"].keys())]),
                "reward": float(np.mean(list(rewards.values()))),
                "terminated": td_terminated_flag(terminal, next_info),
            }
        )
        obs, info = next_obs, next_info
        done = all(terminal.values()) or all(trunc.values())
    return {"steps": steps, "length": len(steps)}


def stack_episodes(episodes, max_seq: int, device: torch.device) -> Dict[str, torch.Tensor]:
    n = len(episodes)
    if n == 0:
        raise ValueError("empty episode list")
    n_agents = episodes[0]["steps"][0]["obs"].shape[0]
    obs_dim = episodes[0]["steps"][0]["obs"].shape[1]
    state_dim = episodes[0]["steps"][0]["state"].shape[0]
    n_actions = episodes[0]["steps"][0]["legals"].shape[1]
    obs = torch.zeros(n, max_seq, n_agents, obs_dim, device=device)
    actions = torch.zeros(n, max_seq, n_agents, 1, dtype=torch.long, device=device)
    state = torch.zeros(n, max_seq, state_dim, device=device)
    avail = torch.zeros(n, max_seq, n_agents, n_actions, dtype=torch.bool, device=device)
    reward = torch.zeros(n, max_seq, 1, device=device)
    term = torch.zeros(n, max_seq, 1, device=device)
    filled = torch.zeros(n, max_seq, 1, device=device)
    for i, ep in enumerate(episodes):
        length = len(ep["steps"])
        for t, st in enumerate(ep["steps"]):
            obs[i, t] = torch.from_numpy(st["obs"]).float()
            actions[i, t, :, 0] = torch.from_numpy(st["actions"]).long()
            state[i, t] = torch.from_numpy(st["state"]).float()
            avail[i, t] = torch.from_numpy(st["legals"]).bool()
            reward[i, t, 0] = st["reward"]
            term[i, t, 0] = st["terminated"]
            filled[i, t, 0] = 1.0
        if length < max_seq:
            # Bootstrap slot: copy last obs/state, mark terminated, zero filled on pad step.
            obs[i, length] = obs[i, length - 1]
            state[i, length] = state[i, length - 1]
            avail[i, length] = avail[i, length - 1]
            term[i, length, 0] = 1.0
            filled[i, length, 0] = 0.0
    return {
        "obs": obs,
        "actions": actions,
        "state": state,
        "avail_actions": avail,
        "reward": reward,
        "terminated": term,
        "filled": filled,
    }


def merge_offline_online(
    off: Dict[str, torch.Tensor], on: Dict[str, torch.Tensor], mixing_ratio: float
) -> Dict[str, torch.Tensor]:
    max_t = min(off["obs"].shape[1], on["obs"].shape[1])
    merged = {}
    for key in off:
        if key not in on:
            continue
        o = off[key][:, :max_t]
        n = on[key][:, :max_t]
        merged[key] = torch.cat([o, n], dim=0)
    return merged
