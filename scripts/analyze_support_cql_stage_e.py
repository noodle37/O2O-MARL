#!/usr/bin/env python3
"""Stage E: verify anti-placement on support-binned overestimation (existing checkpoints, CPU eval).

Loads the final checkpoint of each Stage D arm, runs greedy evaluation episodes,
records per-step Q_tot, return-to-go G, and offline support deviation D, then bins
steps by D to compare q_overestimate_rate and q_bias across arms.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.factory import build_learner, load_vault_dataset, make_env  # noqa: E402
from training.online import resolve_path  # noqa: E402
from training.q_metrics import discounted_returns  # noqa: E402
from training.rollout_helpers import legals_from_info  # noqa: E402
from utils.checkpoint import load_qmix_checkpoint  # noqa: E402
from utils.paths import ensure_project_root  # noqa: E402


ARMS = ["nocql", "uniform", "gated", "anti"]
ARM_OVERRIDES = {
    "nocql": ["system.online_cql_weight=0", "system.support_cql_mode=off"],
    "uniform": ["system.online_cql_weight=50.0", "system.support_cql_mode=off"],
    "gated": ["system.online_cql_weight=50.0", "system.support_cql_mode=on"],
    "anti": ["system.online_cql_weight=50.0", "system.support_cql_mode=anti"],
}


def _select_action_q_tot(
    learner,
    observations: Dict[str, np.ndarray],
    legals: Dict[str, np.ndarray],
    state: Optional[np.ndarray],
) -> Tuple[Dict[str, int], float]:
    """Greedy action selection returning actions and mixed Q_tot of executed action."""
    trace: Dict[str, any] = {}
    actions = learner.select_actions_eval(
        observations, legals, eps=0.0, trace=trace, env_state=state
    )
    q_tot = None
    if trace.get("q_tot") is not None:
        q_tot = float(trace["q_tot"])
    elif state is not None and trace.get("chosen_q"):
        q_tot = learner.mix_agent_qs(trace["chosen_q"], state, offline=False)
    return actions, q_tot


def _offline_support_dev_step(
    learner,
    observations: Dict[str, np.ndarray],
    legals: Dict[str, np.ndarray],
    actions: Dict[str, int],
    offline_rnn: Dict[str, torch.Tensor],
    offline_last: Dict[str, int],
    softmax_temp: float = 1.0,
) -> Tuple[float, Dict[str, torch.Tensor], Dict[str, int]]:
    """Compute D_t for one step and advance offline RNN state.

    D_t = -(1/N) sum_i log pi_off,i(a_i | tau_t).
    Returns (D, next_offline_rnn, next_offline_last).
    """
    logps: List[float] = []
    next_rnn = {}
    next_last = {}
    for i, agent in enumerate(sorted(observations.keys())):
        obs = torch.from_numpy(observations[agent]).float().to(learner.device)
        last_idx = offline_last.get(agent)
        if last_idx is None:
            last_oh = torch.zeros(learner.n_actions, device=learner.device)
        else:
            last_oh = F.one_hot(
                torch.tensor(int(last_idx), device=learner.device), learner.n_actions
            ).float()
        pieces = [obs, last_oh]
        if learner.add_agent_id_to_obs:
            pieces.append(
                F.one_hot(torch.tensor(i, device=learner.device), learner.n_agents).float()
            )
        obs_in = torch.cat(pieces, dim=-1).unsqueeze(0)
        q, next_rnn[agent] = learner.offline_q_network(obs_in, offline_rnn.get(agent))
        q = q[0]
        legal = torch.from_numpy(legals[agent]).bool().to(learner.device)
        q_masked = q.masked_fill(~legal, -1e9)
        temp = max(float(softmax_temp), 1e-6)
        logp = torch.log_softmax(q_masked / temp, dim=-1)
        a = actions[agent]
        logps.append(float(logp[a].item()))
        next_last[agent] = a
    d = -float(np.mean(logps))
    return d, next_rnn, next_last


def evaluate_one_episode(
    learner, env, cfg: Dict[str, any]
) -> Optional[Dict[str, np.ndarray]]:
    """Return episode dict with q, g, d, reward arrays; None on failure."""
    learner.reset_eval_hidden()
    obs, info = env.reset()
    done = False
    q_steps: List[float] = []
    d_steps: List[float] = []
    reward_steps: List[float] = []

    # Mirror the offline RNN initialization done in reset_eval_hidden.
    offline_rnn: Dict[str, torch.Tensor] = {
        f"agent_{i}": learner.offline_q_network.initial_state(1, learner.device)
        for i in range(learner.n_agents)
    }
    offline_last: Dict[str, int] = {}
    softmax_temp = float(getattr(learner, "support_softmax_temp", 1.0))

    while not done:
        state = info.get("state")
        agents = sorted(obs.keys())
        legals = legals_from_info(info, agents)
        actions, q_tot = _select_action_q_tot(learner, obs, legals, state)
        if q_tot is None:
            return None

        d, offline_rnn, offline_last = _offline_support_dev_step(
            learner, obs, legals, actions, offline_rnn, offline_last, softmax_temp
        )

        q_steps.append(q_tot)
        d_steps.append(d)

        obs, rewards, terminal, trunc, info = env.step(actions)
        reward = float(np.mean(list(rewards.values())))
        reward_steps.append(reward)
        done = all(terminal.values()) or all(trunc.values())

    q = np.asarray(q_steps, dtype=np.float64)
    d = np.asarray(d_steps, dtype=np.float64)
    rewards = np.asarray(reward_steps[: len(q_steps)], dtype=np.float64)
    returns = discounted_returns(rewards, float(learner.gamma))[: q.size]
    if q.size == 0 or returns.size == 0:
        return None
    return {
        "q": q,
        "g": returns,
        "d": d,
        "reward": rewards,
        "bias": q - returns,
        "over": q > returns,
    }


def bin_steps(
    episodes: List[Dict[str, np.ndarray]], n_bins: int = 3
) -> Tuple[np.ndarray, Dict[int, Dict[str, float]]]:
    """Pool all steps across episodes and bin by D."""
    all_d = np.concatenate([ep["d"] for ep in episodes])
    all_bias = np.concatenate([ep["bias"] for ep in episodes])
    all_over = np.concatenate([ep["over"] for ep in episodes])

    # Use quantile edges so each bin has equal mass (more stable for skewed D).
    edges = np.quantile(all_d, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-6  # include max
    out: Dict[int, Dict[str, float]] = {}
    for b in range(n_bins):
        lo = edges[b]
        hi = edges[b + 1]
        if b < n_bins - 1:
            mask = (all_d >= lo) & (all_d < hi)
        else:
            mask = (all_d >= lo) & (all_d <= hi)
        n = int(mask.sum())
        out[b] = {
            "lo": float(lo),
            "hi": float(hi),
            "n": n,
            "bias_mean": float(all_bias[mask].mean()) if n else None,
            "bias_std": float(all_bias[mask].std()) if n > 1 else None,
            "overestimate_rate": float(all_over[mask].mean()) if n else None,
        }
    return edges, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="checkpoints/support_cql_stage_d/5m_vs_6m")
    parser.add_argument("--config", default="configs/support_gate_5m_vs_6m.yaml")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--n-bins", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    ensure_project_root()
    from omegaconf import OmegaConf

    base_cfg = OmegaConf.load(args.config)

    summary: Dict[str, any] = {
        "root": args.root,
        "episodes": args.episodes,
        "n_bins": args.n_bins,
        "device": args.device,
        "arms": {},
    }

    for arm in ARMS:
        print(f"\n=== arm {arm} ===")
        arm_rows = []
        for seed_dir in sorted(glob.glob(os.path.join(args.root, f"{arm}_seed*"))):
            seed = int(os.path.basename(seed_dir).split("_seed")[1])
            ckpt = os.path.join(seed_dir, f"ovmse_paper_online_full_td_seed{seed}.pt")
            if not os.path.exists(ckpt):
                print(f"  skip seed {seed}: checkpoint missing")
                continue

            cfg = OmegaConf.merge(
                base_cfg,
                OmegaConf.from_dotlist(
                    [f"seed={seed}", f"system.device={args.device}"] + ARM_OVERRIDES[arm]
                ),
            )
            cfg = OmegaConf.to_container(cfg, resolve=True)

            dataset = load_vault_dataset(cfg)
            learner = build_learner(cfg, dataset)
            load_qmix_checkpoint(ckpt, learner, map_location=learner.device)
            learner.q_network.eval()
            learner.mixer.eval()
            # Stage E needs the frozen offline network for D on ALL arms, including nocql.
            if not (hasattr(learner, "offline_q_network") and learner.offline_q_network is not None):
                offline_ckpt = cfg.get("checkpoint_path") or cfg.get("system", {}).get("checkpoint_path")
                if offline_ckpt and os.path.exists(offline_ckpt):
                    learner.load_offline_for_ovm(offline_ckpt)
                else:
                    raise FileNotFoundError(f"offline checkpoint not found for D computation: {offline_ckpt}")
            learner.offline_q_network.eval()
            learner.offline_mixer.eval()

            env = make_env(cfg, seed=int(cfg["eval_env_seed"]))
            episodes = []
            for ep_idx in range(args.episodes):
                ep = evaluate_one_episode(learner, env, cfg)
                if ep is not None:
                    episodes.append(ep)
                if (ep_idx + 1) % 32 == 0:
                    print(f"  seed {seed}: {ep_idx + 1}/{args.episodes} episodes", flush=True)

            edges, bins = bin_steps(episodes, n_bins=args.n_bins)
            pooled_bias = float(np.concatenate([ep["bias"] for ep in episodes]).mean())
            pooled_over = float(np.concatenate([ep["over"] for ep in episodes]).mean())
            mean_d = float(np.concatenate([ep["d"] for ep in episodes]).mean())
            row = {
                "seed": seed,
                "n_episodes": len(episodes),
                "mean_d": mean_d,
                "pooled_bias": pooled_bias,
                "pooled_overestimate_rate": pooled_over,
                "bin_edges": [float(e) for e in edges],
                "bins": bins,
            }
            arm_rows.append(row)
            print(f"  seed {seed}: n_eps={len(episodes)}, mean_D={mean_d:.3f}, "
                  f"pooled_over={pooled_over:.3f}, pooled_bias={pooled_bias:.3f}")
            for b, stats in bins.items():
                print(
                    f"    bin {b} D [{stats['lo']:.3f}, {stats['hi']:.3f}) n={stats['n']} "
                    f"over={stats['overestimate_rate']:.3f} bias={stats['bias_mean']:+.3f}"
                )

        summary["arms"][arm] = arm_rows

    # Aggregate across seeds per arm.
    print("\n=== across seeds (mean per bin) ===")
    agg: Dict[str, Dict[int, Dict[str, any]]] = defaultdict(lambda: defaultdict(dict))
    for arm in ARMS:
        rows = summary["arms"][arm]
        if not rows:
            continue
        n_bins = len(rows[0]["bins"])
        for b in range(n_bins):
            overs = [r["bins"][b]["overestimate_rate"] for r in rows if r["bins"][b]["overestimate_rate"] is not None]
            biases = [r["bins"][b]["bias_mean"] for r in rows if r["bins"][b]["bias_mean"] is not None]
            ns = [r["bins"][b]["n"] for r in rows]
            if overs:
                agg[arm][b] = {
                    "over_mean": float(np.mean(overs)),
                    "over_se": float(np.std(overs) / np.sqrt(len(overs))) if len(overs) > 1 else None,
                    "bias_mean": float(np.mean(biases)) if biases else None,
                    "bias_se": float(np.std(biases) / np.sqrt(len(biases))) if len(biases) > 1 else None,
                    "n_total": int(sum(ns)),
                    "n_seeds": len(rows),
                }
                print(
                    f"{arm:8s} bin {b}: over={agg[arm][b]['over_mean']:.3f} "
                    f"bias={agg[arm][b]['bias_mean']:+.3f} n_total={agg[arm][b]['n_total']}"
                )

    summary["aggregate"] = {arm: dict(bins) for arm, bins in agg.items()}

    # Pre-registered contrast: anti vs uniform in high-D bin (last bin) and low-D bin (first bin).
    print("\n=== pre-registered contrasts ===")
    contrasts = []
    for bin_label, b in [("low-D", 0), ("high-D", args.n_bins - 1)]:
        anti_rows = summary["arms"].get("anti", [])
        uniform_rows = summary["arms"].get("uniform", [])
        if not anti_rows or not uniform_rows:
            continue
        anti_over = [r["bins"][b]["overestimate_rate"] for r in anti_rows if r["bins"][b]["overestimate_rate"] is not None]
        uniform_over = [r["bins"][b]["overestimate_rate"] for r in uniform_rows if r["bins"][b]["overestimate_rate"] is not None]
        if len(anti_over) == len(uniform_over) and anti_over:
            diffs = [a - u for a, u in zip(anti_over, uniform_over)]
            mean_diff = float(np.mean(diffs))
            n_pos = int(np.sum(np.array(diffs) > 0))
            print(
                f"{bin_label}: anti_over - uniform_over = {mean_diff:+.3f} "
                f"per-seed={np.round(diffs, 3).tolist()} positive {n_pos}/{len(diffs)}"
            )
            contrasts.append({
                "bin": bin_label,
                "metric": "overestimate_rate",
                "anti_minus_uniform": {
                    "mean": mean_diff,
                    "per_seed": diffs,
                    "n_positive": n_pos,
                    "n": len(diffs),
                },
            })
    summary["contrasts"] = contrasts

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
