#!/usr/bin/env python3
"""Windowed Kendall tau of offline Q_tot against return-to-go.

On greedy rollouts of a frozen checkpoint, each timestep is one visited
state-joint-action pair. Estimated value is Q_tot of the executed joint
action. True value is the discounted return-to-go. Episodes are cut into
contiguous windows of 10 steps. Kendall's tau is computed inside each window.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SC2PATH", "/home/jxy/StarCraftII")

import numpy as np
from omegaconf import OmegaConf
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.factory import build_learner, load_vault_dataset, make_env
from training.q_metrics import discounted_returns
from utils.checkpoint import load_qmix_checkpoint
from utils.paths import ensure_project_root, resolve_path

WINDOW = 10
N_EPISODES = 32
RETURN_TIE = 1e-8


def window_starts(n: int, length: int = WINDOW) -> list[int]:
    """Non-overlapping windows. The trailing partial window is dropped."""
    return list(range(0, n - length + 1, length))


def kendall(q_values: np.ndarray, returns: np.ndarray) -> float | None:
    if q_values.size < 3:
        return None
    if float(np.std(q_values)) < 1e-8 or float(np.std(returns)) < 1e-8:
        return None
    tau = kendalltau(q_values, returns).statistic
    if tau is None or not np.isfinite(tau):
        return None
    return float(tau)


def discordant_rate(q_values: np.ndarray, returns: np.ndarray) -> tuple[int, int]:
    """Pairs whose Q order and return order disagree. Ties are not pairs."""
    decided = 0
    discordant = 0
    n = int(q_values.size)
    for i in range(n):
        for j in range(i + 1, n):
            dg = float(returns[i] - returns[j])
            dq = float(q_values[i] - q_values[j])
            if abs(dg) <= RETURN_TIE or abs(dq) <= RETURN_TIE:
                continue
            decided += 1
            if (dq > 0) != (dg > 0):
                discordant += 1
    return discordant, decided


def misrank_counts(q_values: np.ndarray, returns: np.ndarray) -> tuple[int, int, int]:
    """Suppressed high-return points, inflated low-return points, and window size."""
    n = int(q_values.size)
    g_hi = float(np.percentile(returns, 75))
    g_lo = float(np.percentile(returns, 25))
    q_hi = float(np.percentile(q_values, 75))
    q_mid = float(np.percentile(q_values, 50))
    suppressed = 0
    inflated = 0
    for i in range(n):
        if returns[i] >= g_hi and q_values[i] <= q_mid:
            suppressed += 1
        if returns[i] <= g_lo and q_values[i] >= q_hi:
            inflated += 1
    return suppressed, inflated, n


def gap_medians(q_values: np.ndarray, returns: np.ndarray) -> tuple[float, float] | None:
    """Median |dQ| on the largest and smallest |dG| quartiles of pairs."""
    gaps_g = []
    gaps_q = []
    n = int(q_values.size)
    for i in range(n):
        for j in range(i + 1, n):
            gaps_g.append(abs(float(returns[i] - returns[j])))
            gaps_q.append(abs(float(q_values[i] - q_values[j])))
    gaps_g = np.asarray(gaps_g, dtype=np.float64)
    gaps_q = np.asarray(gaps_q, dtype=np.float64)
    if float(np.std(gaps_g)) < 1e-8:
        return None
    hi = float(np.percentile(gaps_g, 75))
    lo = float(np.percentile(gaps_g, 25))
    large = gaps_q[gaps_g >= hi]
    small = gaps_q[gaps_g <= lo]
    if large.size == 0 or small.size == 0:
        return None
    return float(np.median(large)), float(np.median(small))


def _self_check() -> None:
    if window_starts(25) != [0, 10]:
        raise RuntimeError("window cut failed")
    if window_starts(9) != []:
        raise RuntimeError("short episode should yield no window")
    perfect_q = np.arange(10, dtype=np.float64)
    perfect_g = perfect_q * 2
    if abs(kendall(perfect_q, perfect_g) - 1.0) > 1e-8:
        raise RuntimeError("perfect order should have tau 1")
    disc, decided = discordant_rate(perfect_q, perfect_g)
    if disc != 0 or decided == 0:
        raise RuntimeError("perfect order should have no discordant pair")
    rev_disc, rev_decided = discordant_rate(perfect_q, -perfect_g)
    if rev_disc != rev_decided:
        raise RuntimeError("reversed order should be entirely discordant")
    large, small = gap_medians(perfect_q, perfect_g)
    if not large > small:
        raise RuntimeError("aligned gaps should keep the large-gap quartile larger")


def collect_episodes(env, learner, n_episodes: int) -> list[dict]:
    episodes = []
    for episode in range(n_episodes):
        learner.reset_eval_hidden()
        obs, info = env.reset()
        done = False
        q_steps: list[float] = []
        reward_steps: list[float] = []
        won = None
        while not done:
            trace: dict = {}
            state = info.get("state")
            actions = learner.select_actions_eval(obs, info["legals"], 0.0, trace=trace, env_state=state)
            if state is None or not trace.get("chosen_q"):
                raise RuntimeError("evaluation step did not produce a joint Q")
            q_steps.append(learner.mix_agent_qs(trace["chosen_q"], state))
            obs, rewards, terminal, trunc, info = env.step(actions)
            reward_steps.append(float(np.mean(list(rewards.values()))))
            if "battle_won" in info:
                won = bool(info["battle_won"])
            done = all(bool(np.asarray(v)) for v in terminal.values()) or all(
                bool(np.asarray(v)) for v in trunc.values()
            )
        episodes.append(
            {
                "q": np.asarray(q_steps, dtype=np.float64),
                "reward": np.asarray(reward_steps, dtype=np.float64),
                "win": won,
            }
        )
        print(
            f"episode {episode + 1}/{n_episodes} steps={len(q_steps)} "
            f"return={float(np.sum(reward_steps)):.3f} win={won}",
            flush=True,
        )
    return episodes


def summarize(episodes: list[dict], gamma: float, window: int) -> dict:
    window_rows = []
    episode_taus = []
    for ep_i, episode in enumerate(episodes):
        returns = discounted_returns(episode["reward"], gamma)
        q_values = episode["q"]
        ep_tau = kendall(q_values, returns)
        if ep_tau is not None:
            episode_taus.append(ep_tau)
        for start in window_starts(int(q_values.size), window):
            sl = slice(start, start + window)
            q_w = q_values[sl]
            g_w = returns[sl]
            tau = kendall(q_w, g_w)
            disc, decided = discordant_rate(q_w, g_w)
            suppressed, inflated, n_points = misrank_counts(q_w, g_w)
            gaps = gap_medians(q_w, g_w)
            row = {
                "episode": ep_i,
                "start": start,
                "tau": tau,
                "n_discordant": disc,
                "n_decided_pairs": decided,
                "n_suppressed": suppressed,
                "n_inflated": inflated,
                "n_points": n_points,
                "median_abs_dq_large_dg": None if gaps is None else gaps[0],
                "median_abs_dq_small_dg": None if gaps is None else gaps[1],
            }
            window_rows.append(row)

    defined = [row for row in window_rows if row["tau"] is not None]
    decided_pairs = sum(row["n_decided_pairs"] for row in window_rows)
    discordant_pairs = sum(row["n_discordant"] for row in window_rows)
    points = sum(row["n_points"] for row in window_rows)
    gap_rows = [
        row
        for row in window_rows
        if row["median_abs_dq_large_dg"] is not None and row["median_abs_dq_small_dg"] is not None
    ]
    n_gap_flat = sum(
        row["median_abs_dq_large_dg"] <= row["median_abs_dq_small_dg"] for row in gap_rows
    )
    returns = [float(np.sum(ep["reward"])) for ep in episodes]
    wins = [float(ep["win"]) for ep in episodes if ep["win"] is not None]
    return {
        "n_episodes": len(episodes),
        "mean_return": float(np.mean(returns)) if returns else None,
        "win_rate": float(np.mean(wins)) if wins else None,
        "n_windows": len(window_rows),
        "n_windows_tau_defined": len(defined),
        "window_tau_mean": None if not defined else float(np.mean([row["tau"] for row in defined])),
        "episode_tau_mean": None if not episode_taus else float(np.mean(episode_taus)),
        "n_episode_tau_defined": len(episode_taus),
        "n_decided_pairs": decided_pairs,
        "n_discordant_pairs": discordant_pairs,
        "discordant_rate": None if decided_pairs == 0 else discordant_pairs / decided_pairs,
        "n_points": points,
        "suppressed_rate": None if points == 0 else sum(row["n_suppressed"] for row in window_rows) / points,
        "inflated_rate": None if points == 0 else sum(row["n_inflated"] for row in window_rows) / points,
        "n_gap_windows": len(gap_rows),
        "gap_not_reflected_rate": None if not gap_rows else n_gap_flat / len(gap_rows),
        "window": window,
    }


def half_summary(episodes: list[dict], gamma: float, first: bool) -> dict:
    """Kendall tau on the first or second half of each episode."""
    taus = []
    discordant = 0
    decided = 0
    used = 0
    for episode in episodes:
        returns = discounted_returns(episode["reward"], gamma)
        n = int(episode["q"].size)
        mid = n // 2
        sl = slice(0, mid) if first else slice(mid, n)
        if int(episode["q"][sl].size) < 8:
            continue
        used += 1
        tau = kendall(episode["q"][sl], returns[sl])
        if tau is not None:
            taus.append(tau)
        disc, n_decided = discordant_rate(episode["q"][sl], returns[sl])
        discordant += disc
        decided += n_decided
    return {
        "n_episodes": used,
        "tau_mean": None if not taus else float(np.mean(taus)),
        "discordant_rate": None if decided == 0 else discordant / decided,
    }


def run_one(config_path: str) -> dict:
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    cfg["system"]["data_device"] = "cpu"
    ckpt = resolve_path(cfg["checkpoint_path"])
    gamma = float(cfg["system"]["discount"])
    scenario = cfg["task"]["scenario"]
    print(
        f"scenario={scenario} checkpoint={ckpt} episodes={N_EPISODES} gamma={gamma} updates=0",
        flush=True,
    )
    dataset = load_vault_dataset(cfg)
    learner = build_learner(cfg, dataset)
    del dataset
    load_qmix_checkpoint(ckpt, learner, load_optimizer=False)
    learner.q_network.eval()
    learner.mixer.eval()
    env = make_env(cfg, seed=int(cfg["seed"]))
    try:
        episodes = collect_episodes(env, learner, N_EPISODES)
    finally:
        inner = getattr(env, "_environment", None)
        close = getattr(inner, "close", None) if inner is not None else None
        if close is not None:
            close()
    lengths = [int(ep["q"].size) for ep in episodes]
    by_window = {}
    for window in (5, 10, 20):
        summary = summarize(episodes, gamma, window)
        by_window[str(window)] = {
            "n_windows": summary["n_windows"],
            "window_tau_mean": summary["window_tau_mean"],
            "discordant_rate": summary["discordant_rate"],
            "n_decided_pairs": summary["n_decided_pairs"],
            "n_discordant_pairs": summary["n_discordant_pairs"],
        }
        print(
            f"[{scenario}] window={window} n={summary['n_windows']} "
            f"tau={summary['window_tau_mean']} discordant={summary['discordant_rate']}",
            flush=True,
        )
    episode_summary = summarize(episodes, gamma, 10)
    return {
        "scenario": scenario,
        "checkpoint": ckpt,
        "n_episodes": N_EPISODES,
        "mean_return": episode_summary["mean_return"],
        "win_rate": episode_summary["win_rate"],
        "median_length": float(np.median(lengths)),
        "episode_tau_mean": episode_summary["episode_tau_mean"],
        "by_window": by_window,
        "first_half": half_summary(episodes, gamma, first=True),
        "second_half": half_summary(episodes, gamma, first=False),
    }


def main() -> None:
    _self_check()
    ensure_project_root()
    results = [run_one(path) for path in ("configs/5m_vs_6m.yaml", "configs/2s3z.yaml")]
    payload = {"parameter_updates": 0, "gamma": 0.99, "results": results}
    out = Path(resolve_path("logs")) / "verify_window_kendall_more.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
