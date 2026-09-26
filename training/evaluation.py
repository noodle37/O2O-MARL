# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Fixed-policy evaluation rollouts (epsilon=0)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ovmse.learner import PaperQMIXLearner


def evaluate(
    env,
    learner: PaperQMIXLearner,
    n_episodes: int,
    t_env: int,
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    eps = 0.0
    returns = []
    wins = []
    for _ in range(n_episodes):
        learner.reset_eval_hidden()
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        won = None
        while not done:
            actions = learner.select_actions_eval(obs, info["legals"], eps)
            obs, rewards, terminal, trunc, info = env.step(actions)
            ep_ret += float(np.mean(list(rewards.values())))
            if "battle_won" in info:
                won = bool(info["battle_won"])
            done = all(terminal.values()) or all(trunc.values())
        returns.append(ep_ret)
        if won is not None:
            wins.append(float(won))
    logs = {
        "evaluation/mean_episode_return": float(np.mean(returns)),
        "evaluation/max_episode_return": float(np.max(returns)),
        "evaluation/min_episode_return": float(np.min(returns)),
    }
    if wins:
        logs["evaluation/win_rate"] = float(np.mean(wins))
    return logs
