# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Version-boundary judgement: two paired on-policy evaluation arms.

Each version boundary freezes the current online parameters and, for every
evaluation seed, runs one full episode with the offline greedy policy and one
with the online greedy policy from the same seed. Both arms are epsilon = 0, use
no proposal randomisation, write nothing to the replay buffer and perform no
gradient update.

Importance sampling is deliberately not used: with deterministic target policies
under the 0.5/0.5 mixture behaviour, the importance weights vanish on long
trajectories, so self-normalisation or weight clipping cannot recover the missing
support. Each arm samples its own target-policy trajectory distribution, so both
single-arm estimates are unbiased and their difference stays unbiased; the shared
seed only reduces the variance of the paired difference.
"""

from __future__ import annotations

import copy
import math
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from fhjp.proposer import fresh_policy_state, joint_greedy
from training.rollout_helpers import legals_from_info


def arm_return(
    env,
    net: torch.nn.Module,
    learner,
    gamma: float,
) -> Tuple[float, int]:
    """One epsilon-greedy episode: discounted return and env steps.

    The net is conditioned on its own previous actions (closed loop) and starts
    from the initial RNN hidden state.
    """
    state = fresh_policy_state(net, learner)
    obs, info = env.reset()
    done = False
    total = 0.0
    steps = 0
    while not done:
        agents = sorted(obs.keys())
        actions = joint_greedy(net, learner, obs, legals_from_info(info, agents), state)
        obs, rewards, terminal, trunc, info = env.step(actions)
        reward = float(np.mean(list(rewards.values())))
        total += (gamma**steps) * reward
        steps += 1
        for agent in agents:
            state["last"][agent] = actions[agent]
        done = all(terminal.values()) or all(trunc.values())
    return total, steps


def _close_env(env) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass


def paired_arm_estimate(
    env_factory: Callable[[int], Any],
    learner,
    *,
    n_episodes: int,
    seed_base: int,
    gamma: Optional[float] = None,
    offline_net: Optional[torch.nn.Module] = None,
    frozen_online_net: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    """Paired on-policy estimate of tau_v = J(pi_off) - J(pi_greedy).

    ``env_factory(seed)`` must build a fresh environment for that seed, so the
    two arms of a pair share the same construction seed and the same reset count
    and therefore see the same initial conditions.
    """
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")
    gamma = float(learner.gamma if gamma is None else gamma)
    offline_net = offline_net if offline_net is not None else learner.offline_q_network
    if offline_net is None:
        raise RuntimeError("paired arms need the frozen offline_q_network")
    if frozen_online_net is None:
        # Freeze the current online parameters for the whole boundary: the arms
        # must all be scored against the same version of the online policy.
        frozen_online_net = copy.deepcopy(learner.q_network)
        frozen_online_net.eval()
    offline_net.eval()

    rng = np.random.default_rng(int(seed_base))
    offline_returns: List[float] = []
    online_returns: List[float] = []
    deltas: List[float] = []
    eval_steps = 0

    for _ in range(n_episodes):
        seed = int(rng.integers(0, 2**31 - 1))
        env_off = env_factory(seed)
        try:
            g_off, n_off = arm_return(env_off, offline_net, learner, gamma)
        finally:
            _close_env(env_off)
        env_on = env_factory(seed)
        try:
            g_on, n_on = arm_return(env_on, frozen_online_net, learner, gamma)
        finally:
            _close_env(env_on)
        offline_returns.append(g_off)
        online_returns.append(g_on)
        deltas.append(g_off - g_on)
        eval_steps += n_off + n_on

    tau = float(np.mean(deltas))
    std = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
    return {
        "tau_v": tau,
        "std_delta": std,
        "n_pairs": len(deltas),
        "offline_returns": offline_returns,
        "online_returns": online_returns,
        "deltas": deltas,
        "eval_env_steps": int(eval_steps),
    }


def one_sided_upper_bound(tau: float, std: float, n: int, alpha: float) -> float:
    """Upper bound of the one-sided confidence interval for the mean difference."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n == 1 or std == 0.0:
        return tau
    z = statistics.NormalDist().inv_cdf(1.0 - alpha)
    return tau + z * std / math.sqrt(n)


def should_exit(
    estimate: Dict[str, Any], *, min_eval_episodes: int, alpha: float
) -> Tuple[bool, Dict[str, float]]:
    """Exit when the offline joint behaviour has no measurable positive value."""
    n = int(estimate["n_pairs"])
    tau = float(estimate["tau_v"])
    std = float(estimate["std_delta"])
    ub = one_sided_upper_bound(tau, std, n, alpha)
    logs = {
        "proposer/tau_v": tau,
        "proposer/tau_v_std": std,
        "proposer/tau_v_ub": ub,
        "proposer/n_eval_episodes_per_arm": float(n),
        "proposer/offline_arm_return": float(np.mean(estimate["offline_returns"])),
        "proposer/online_arm_return": float(np.mean(estimate["online_returns"])),
    }
    exit_now = bool(n >= int(min_eval_episodes) and ub <= 0.0)
    logs["proposer/exited"] = float(exit_now)
    return exit_now, logs
