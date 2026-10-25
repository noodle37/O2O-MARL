# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Fixed-policy evaluation rollouts (epsilon=0)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from qmix.core import QMIXBase
from training.rollout_helpers import legals_from_info
from training.q_metrics import summarize_q_traces


def evaluate(
    env,
    learner: QMIXBase,
    n_episodes: int,
    t_env: int,
    cfg: Dict[str, Any],
    return_episodes: bool = False,
) -> Dict[str, Any]:
    eps = 0.0
    returns = []
    wins = []
    traces: List[Dict[str, Any]] = []
    for _ in range(n_episodes):
        learner.reset_eval_hidden()
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        won = None
        q_steps: List[float] = []
        offline_q_steps: List[float] = []
        reward_steps: List[float] = []
        n_disagree = 0
        n_decisions = 0
        while not done:
            trace: Dict[str, Any] = {}
            state = info.get("state")
            agents = sorted(obs.keys())
            actions = learner.select_actions_eval(
                obs,
                legals_from_info(info, agents),
                eps,
                trace=trace,
                env_state=state,
            )
            if trace.get("q_tot") is not None:
                q_steps.append(float(trace["q_tot"]))
            elif state is not None and trace.get("chosen_q"):
                q_steps.append(learner.mix_agent_qs(trace["chosen_q"], state))
                if trace.get("offline_chosen_q"):
                    offline_q_steps.append(
                        learner.mix_agent_qs(trace["offline_chosen_q"], state, offline=True)
                    )
                n_disagree += int(trace.get("n_disagree", 0))
                n_decisions += int(trace.get("n_agents", 0))
            obs, rewards, terminal, trunc, info = env.step(actions)
            reward = float(np.mean(list(rewards.values())))
            reward_steps.append(reward)
            ep_ret += reward
            if "battle_won" in info:
                won = bool(info["battle_won"])
            done = all(terminal.values()) or all(trunc.values())
        returns.append(ep_ret)
        if won is not None:
            wins.append(float(won))
        if q_steps:
            ep_trace: Dict[str, Any] = {
                "q": np.asarray(q_steps, dtype=np.float64),
                "reward": np.asarray(reward_steps[: len(q_steps)], dtype=np.float64),
                "n_disagree": n_disagree,
                "n_decisions": n_decisions,
            }
            if offline_q_steps:
                ep_trace["offline_q"] = np.asarray(offline_q_steps, dtype=np.float64)
            traces.append(ep_trace)
    logs = {
        "evaluation/mean_episode_return": float(np.mean(returns)),
        "evaluation/max_episode_return": float(np.max(returns)),
        "evaluation/min_episode_return": float(np.min(returns)),
    }
    if wins:
        logs["evaluation/win_rate"] = float(np.mean(wins))
    if return_episodes:
        logs["evaluation/episode_returns"] = [float(x) for x in returns]
    if traces:
        logs.update(summarize_q_traces(traces, gamma=float(learner.gamma)))
    return logs
