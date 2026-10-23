"""Q calibration diagnostics along greedy eval trajectories.

Win rate can stay high while Q_tot is a biased or flat estimate of the
discounted return. These summaries are the episode-level analogues of the
normalized Q error and Kendall rank correlation used in single-agent O2O.

The paper-style protocol (Zhang et al. 2023, arXiv:2312.07685) compares
Q_estimated with the Monte Carlo return-to-go Q_true on a rollout of the
current policy. Rank quality is the mean Kendall tau over sliding windows of
``window`` consecutive decisions (the paper reports window 10). Bias is the
signed residual and its normalized forms.
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


def _residual_sign_disagree_frac(q_values: np.ndarray, returns: np.ndarray) -> float:
    if q_values.size == 0:
        return 0.0
    bias = float(np.mean(q_values - returns))
    if abs(bias) < 1e-12:
        return 0.0
    sign_bias = 1.0 if bias > 0 else -1.0
    residual = q_values - returns
    signs = np.sign(residual)
    signs[signs == 0] = sign_bias
    disagree = np.mean(signs != sign_bias)
    return float(disagree)


def _fit_affine_calibration(q: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    """Least-squares g ≈ scale * q + shift (scalar)."""
    if q.size < 2:
        return 1.0, 0.0
    qm = float(np.mean(q))
    gm = float(np.mean(g))
    var_q = float(np.mean((q - qm) ** 2))
    if var_q < 1e-12:
        return 1.0, gm - qm
    cov = float(np.mean((q - qm) * (g - gm)))
    scale = cov / var_q
    shift = gm - scale * qm
    return scale, shift


def _cluster_mean_se(values: Sequence[float]) -> Optional[float]:
    """SE of the mean across independent episodes (not across steps)."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size < 2:
        return None
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def _window_starts(length: int, window: int, stride: int) -> List[int]:
    if window < 3 or stride < 1 or length < window:
        return []
    return list(range(0, length - window + 1, stride))


def windowed_kendall_taus(
    q_values: np.ndarray,
    returns: np.ndarray,
    window: int = 10,
    stride: int = 1,
) -> List[Optional[float]]:
    """Kendall tau inside every sliding window of the same episode."""
    q_values = np.asarray(q_values, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    if q_values.shape != returns.shape:
        raise ValueError("q_values and returns must be aligned")
    out: List[Optional[float]] = []
    for start in _window_starts(q_values.size, window, stride):
        sl = slice(start, start + window)
        out.append(_kendall(q_values[sl], returns[sl]))
    return out


def summarize_paper_diagnostics(
    episodes: List[Dict[str, np.ndarray]],
    gamma: float,
    window: int = 10,
    stride: int = 1,
) -> Dict[str, float]:
    """Paper-style Q bias and sliding-window rank diagnostics.

    Each episode dict must contain ``q`` (Q_tot of the executed joint action)
    and ``reward`` arrays of equal length. Every quantile/SE is computed over
    episodes, so the sample unit is the episode, not the decision step.
    """
    if not episodes:
        return {}
    episode_bias: List[float] = []
    episode_over: List[float] = []
    episode_window_tau: List[float] = []
    episode_full_tau: List[float] = []
    all_q: List[np.ndarray] = []
    all_g: List[np.ndarray] = []
    all_window_tau: List[float] = []
    n_windows = 0
    n_defined_windows = 0
    n_literal = 0
    n_points = 0
    returns_episode: List[float] = []
    wins: List[float] = []

    for ep in episodes:
        q_values = np.asarray(ep["q"], dtype=np.float64)
        rewards = np.asarray(ep["reward"], dtype=np.float64)
        if q_values.size == 0:
            continue
        returns = discounted_returns(rewards, gamma)[: q_values.size]
        residual = q_values - returns
        episode_bias.append(float(np.mean(residual)))
        episode_over.append(float(np.mean(q_values > returns)))
        full_tau = _kendall(q_values, returns)
        if full_tau is not None:
            episode_full_tau.append(full_tau)
        taus = windowed_kendall_taus(q_values, returns, window=window, stride=stride)
        n_windows += len(taus)
        defined = [t for t in taus if t is not None]
        n_defined_windows += len(defined)
        if defined:
            all_window_tau.extend(defined)
            episode_window_tau.append(float(np.mean(defined)))
        all_q.append(q_values)
        all_g.append(returns)
        n_points += int(q_values.size)
        mask_literal = np.abs(returns) >= 1.0
        n_literal += int(mask_literal.sum())
        returns_episode.append(float(np.sum(rewards)))
        win = ep.get("win")
        if win is not None:
            wins.append(float(bool(win)))

    if n_points == 0:
        return {}
    q_all = np.concatenate(all_q)
    g_all = np.concatenate(all_g)
    residual = q_all - g_all
    over = q_all > g_all
    under = q_all < g_all
    tie = ~(over | under)
    denom_abs = np.maximum(np.abs(g_all), 1.0)
    literal_mask = np.abs(g_all) >= 1.0
    literal = residual[literal_mask] / g_all[literal_mask]
    bias_mean = float(np.mean(residual))
    q_mean = float(np.mean(q_all))
    g_mean = float(np.mean(g_all))
    var_q = float(np.mean((q_all - q_mean) ** 2))
    cov = float(np.mean((q_all - q_mean) * (g_all - g_mean)))
    slope = cov / var_q if var_q > 1e-12 else 1.0
    shift = g_mean - slope * q_mean
    out = {
        "n_episodes": len(all_q),
        "n_steps": int(n_points),
        "mean_episode_return": float(np.mean(returns_episode)),
        "mean_episode_length": float(n_points / len(all_q)),
        "q_mean": q_mean,
        "q_std": float(np.std(q_all)),
        "g_mean": g_mean,
        "g_std": float(np.std(g_all)),
        "bias_mean": bias_mean,
        "bias_median": float(np.median(residual)),
        "bias_std": float(np.std(residual)),
        "bias_mean_se_episode": _cluster_mean_se(episode_bias),
        "bias_quantiles": {
            str(p): float(np.quantile(residual, p / 100)) for p in (5, 25, 50, 75, 95)
        },
        "overestimate_rate": float(np.mean(over)),
        "underestimate_rate": float(np.mean(under)),
        "tie_rate": float(np.mean(tie)),
        "overestimate_rate_se_episode": _cluster_mean_se(episode_over),
        "normalized_diff_abs": float(np.mean(residual / denom_abs)),
        "normalized_diff_literal": (
            float(np.mean(literal)) if literal.size else None
        ),
        "literal_denominator_frac": float(n_literal / n_points),
        "affine_fit_g_on_q": {"slope": float(slope), "shift": float(shift)},
        "kendall_tau_full_episode": (
            float(np.mean(episode_full_tau)) if episode_full_tau else None
        ),
        "kendall_tau_full_se_episode": _cluster_mean_se(episode_full_tau),
        "window": window,
        "window_stride": stride,
        "n_windows": n_windows,
        "n_windows_tau_defined": n_defined_windows,
        "window_tau_mean": (
            float(np.mean(all_window_tau)) if all_window_tau else None
        ),
        "window_tau_median": (
            float(np.median(all_window_tau)) if all_window_tau else None
        ),
        "window_tau_std": (
            float(np.std(all_window_tau)) if all_window_tau else None
        ),
        "window_tau_negative_frac": (
            float(np.mean(np.asarray(all_window_tau) < 0.0)) if all_window_tau else None
        ),
        "window_tau_episode_mean": (
            float(np.mean(episode_window_tau)) if episode_window_tau else None
        ),
        "window_tau_episode_se": _cluster_mean_se(episode_window_tau),
    }
    if wins:
        out["win_rate"] = float(np.mean(wins))
    return out


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
    sign_disagree: List[float] = []
    cal_train_q: List[float] = []
    cal_train_g: List[float] = []
    cal_hold_abs: List[float] = []
    split = max(1, len(episodes) // 2)

    for ep_idx, ep in enumerate(episodes):
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
        sign_disagree.append(_residual_sign_disagree_frac(q_values, returns))
        if ep_idx < split:
            cal_train_q.append(q_values)
            cal_train_g.append(returns)
        else:
            if cal_train_q:
                q_tr = np.concatenate(cal_train_q)
                g_tr = np.concatenate(cal_train_g)
                scale, shift = _fit_affine_calibration(q_tr, g_tr)
                pred = scale * q_values + shift
                cal_hold_abs.append(float(np.mean(np.abs(pred - returns))))

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
    if sign_disagree:
        logs["evaluation/residual_sign_disagree_frac"] = float(np.mean(sign_disagree))
    if cal_hold_abs:
        logs["evaluation/calibrated_holdout_mae"] = float(np.mean(cal_hold_abs))

    paper = summarize_paper_diagnostics(episodes, gamma, window=10)
    optional = {
        "evaluation/q_bias_median": paper.get("bias_median"),
        "evaluation/q_bias_se_episode": paper.get("bias_mean_se_episode"),
        "evaluation/q_underestimate_rate": paper.get("underestimate_rate"),
        "evaluation/q_normalized_diff_literal": paper.get("normalized_diff_literal"),
        "evaluation/q_window_kendall_tau_w10": paper.get("window_tau_mean"),
        "evaluation/q_window_kendall_tau_se_w10": paper.get("window_tau_episode_se"),
        "evaluation/q_window_tau_negative_frac_w10": paper.get("window_tau_negative_frac"),
        "evaluation/q_window_tau_median_w10": paper.get("window_tau_median"),
        "evaluation/q_window_count_w10": paper.get("n_windows"),
    }
    logs.update({k: float(v) for k, v in optional.items() if v is not None})
    return logs
