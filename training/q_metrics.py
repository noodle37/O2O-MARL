"""Q calibration diagnostics along greedy eval trajectories.

Win rate can stay high while Q_tot is a biased or flat estimate of the
discounted return. These summaries are the episode-level analogues of the
normalized Q error and Kendall rank correlation used in single-agent O2O.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import kendalltau


def discounted_returns(rewards: Sequence[float], gamma: float) -> np.ndarray:
    """Return-to-go G_t = r_t + gamma r_{t+1} + ... for one episode."""
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    out = np.zeros_like(rewards_arr)
    acc = 0.0
    for t in range(len(rewards_arr) - 1, -1, -1):
        acc = float(rewards_arr[t]) + gamma * acc
        out[t] = acc
    return out


def _kendall(q_values: np.ndarray, returns: np.ndarray) -> Optional[float]:
    if q_values.size < 3:
        return None
    if float(np.std(q_values)) < 1e-6 or float(np.std(returns)) < 1e-6:
        return None
    tau = kendalltau(q_values, returns).statistic
    if tau is None or not np.isfinite(tau):
        return None
    return float(tau)


def summarize_q_traces(
    episodes: List[Dict[str, np.ndarray]],
    gamma: float,
) -> Dict[str, float]:
    """Aggregate per-episode traces of Q_tot, reward, and optional offline Q_tot.

    Each episode dict has ``q`` and ``reward`` arrays of equal length. Optional
    ``offline_q`` is the frozen offline Q_tot on the same executed actions.
    ``n_disagree`` / ``n_decisions`` count agent-steps whose offline greedy
    action differs from the executed greedy action.
    """
    if not episodes:
        return {}

    biases: List[float] = []
    norm_diffs: List[float] = []
    over_flags: List[float] = []
    taus: List[float] = []
    q_stds: List[float] = []
    g_stds: List[float] = []
    mean_qs: List[float] = []
    mean_gs: List[float] = []
    off_biases: List[float] = []
    off_taus: List[float] = []
    gaps: List[float] = []
    disagree = 0.0
    decisions = 0.0
    n_rank = 0

    for ep in episodes:
        q_values = np.asarray(ep["q"], dtype=np.float64)
        rewards = np.asarray(ep["reward"], dtype=np.float64)
        returns = discounted_returns(rewards, gamma)
        if q_values.size == 0:
            continue
        biases.append(float(np.mean(q_values - returns)))
        denom = np.maximum(np.abs(returns), 1.0)
        norm_diffs.append(float(np.mean((q_values - returns) / denom)))
        over_flags.append(float(np.mean(q_values > returns)))
        q_stds.append(float(np.std(q_values)))
        g_stds.append(float(np.std(returns)))
        mean_qs.append(float(np.mean(q_values)))
        mean_gs.append(float(np.mean(returns)))
        n_rank += 1
        tau = _kendall(q_values, returns)
        if tau is not None:
            taus.append(tau)

        if "offline_q" in ep:
            offline_q = np.asarray(ep["offline_q"], dtype=np.float64)
            off_biases.append(float(np.mean(offline_q - returns)))
            gaps.append(float(np.mean(q_values - offline_q)))
            off_tau = _kendall(offline_q, returns)
            if off_tau is not None:
                off_taus.append(off_tau)
        disagree += float(ep.get("n_disagree", 0.0))
        decisions += float(ep.get("n_decisions", 0.0))

    logs: Dict[str, float] = {
        "evaluation/mean_q_tot": float(np.mean(mean_qs)),
        "evaluation/mean_return_to_go": float(np.mean(mean_gs)),
        "evaluation/q_bias": float(np.mean(biases)),
        "evaluation/q_normalized_diff": float(np.mean(norm_diffs)),
        "evaluation/q_overestimate_rate": float(np.mean(over_flags)),
        "evaluation/q_std": float(np.mean(q_stds)),
        "evaluation/return_to_go_std": float(np.mean(g_stds)),
        "evaluation/q_rank_defined_frac": float(len(taus) / max(n_rank, 1)),
    }
    if taus:
        logs["evaluation/q_kendall_tau"] = float(np.mean(taus))
    if off_biases:
        logs["evaluation/offline_q_bias"] = float(np.mean(off_biases))
        logs["evaluation/live_minus_offline_q"] = float(np.mean(gaps))
    if off_taus:
        logs["evaluation/offline_q_kendall_tau"] = float(np.mean(off_taus))
    if decisions > 0:
        logs["evaluation/greedy_action_disagree_rate"] = disagree / decisions
    return logs
