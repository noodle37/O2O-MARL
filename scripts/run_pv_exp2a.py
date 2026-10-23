#!/usr/bin/env python3
# Copyright 2024 InstaDeep Ltd. All rights reserved.

"""Experiment 2a: controlled teammate-shift evaluation (no gradient updates).

One examined agent (or pair) keeps the offline parameters and acts greedily; the
remaining teammates are shifted either naturally (arm A's checkpoint at t) or
randomly (offline parameters plus eps uniform noise).  For every episode the
frozen offline Q_tot is scored as a shadow forward on the executed joint action,
and the Monte-Carlo return-to-go is recorded, so the value knowledge can be
binned by teammate deviation D(s).

Behavioural validity (natural shift only):
  g_i(t) = R(all@t) - R(i@off, rest@t)
  I_ij(t) = R(off_i, off_j | rest@t) - R(off_i | rest@t) - R(off_j | rest@t) + R(all@t)

Usage:
  python scripts/run_pv_exp2a.py --map 5m_vs_6m --seed 1 \
      --t-steps 0 25000 50000 --episodes 100 \
      --out checkpoints/pv_exp2a/5m_vs_6m/seed1.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pv.behaviour import load_behaviour_model
from pv.exp2_shift import (
    NATURAL_SHIFT_STEPS,
    RANDOM_SHIFT_EPS,
    Net,
    run_episode,
    score_records,
)
from pv.probe import mc_return_to_go
from training.factory import build_learner, load_vault_dataset, make_env
from utils.paths import ensure_project_root, resolve_path

MAP_CONFIG = {
    "5m_vs_6m": "configs/pv_base_5m_vs_6m.yaml",
    "2halfcheetah": "configs/pv_base_2halfcheetah.yaml",
}
MAP_OFFLINE = {
    "5m_vs_6m": "checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt",
    "2halfcheetah": "checkpoints/pv_exp1_offline/2halfcheetah_medium/ovmse_paper_seed1.pt",
}
# Normalisation of the value bias (Q^off_tot - G) / R_max.  SMAC's max episode
# return is 20 (win bonus); MuJoCo returns are negative and of order 1e3, so the
# sign-stable max(|G|, 1) form is used there instead.
R_MAX = {"5m_vs_6m": 20.0, "2halfcheetah": None}


def load_cfg(path: str, overrides: list) -> dict:
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


def checkpoint_at(map_name: str, seed: int, t: int) -> str:
    """Arm A's checkpoint after t environment steps (t=0 is the offline start)."""
    if t <= 0:
        return MAP_OFFLINE[map_name]
    path = os.path.join(
        "checkpoints", "pv_exp1", map_name, f"A_seed{seed}", "periodic", f"ovm_diag_ckpt_t{t}.pt"
    )
    return path


def configs_for(n_agents: int) -> List[dict]:
    """all@t, every single transplant, every pair transplant."""
    out = [{"name": "all", "offline_agents": []}]
    for i in range(n_agents):
        out.append({"name": f"single_{i}", "offline_agents": [i]})
    for i, j in itertools.combinations(range(n_agents), 2):
        out.append({"name": f"pair_{i}_{j}", "offline_agents": [i, j]})
    return out


def evaluate_config(
    env,
    *,
    live: Net,
    offline: Net,
    offline_agents: List[int],
    n_agents: int,
    eps_partner: float,
    examined: set,
    behaviour,
    gamma: float,
    n_actions: int,
    add_agent_id: bool,
    device,
    n_episodes: int,
    rng: np.random.Generator,
    r_max,
) -> dict:
    """Run one configuration and return its summary plus per-state arrays.

    `env` is created once per configuration with the shared evaluation seed, so
    every configuration is scored on the same episode sequence (SMAC v1 derives
    its episode sequence from the reset count, so a fresh env per episode would
    replay the same first episode).
    """
    sources = {i: ("offline" if i in offline_agents else "live") for i in range(n_agents)}
    returns, wins, states = [], [], []
    for _ in range(n_episodes):
        record = run_episode(
            env,
            live=live,
            offline=offline,
            sources=sources,
            examined=examined,
            eps_partner=eps_partner,
            n_actions=n_actions,
            add_agent_id=add_agent_id,
            device=device,
            rng=rng,
        )
        returns.append(record.episode_return)
        if record.won is not None:
            wins.append(float(record.won))
        rewards = np.array([s["reward"] for s in record.steps], dtype=np.float64)
        ret = mc_return_to_go(rewards, gamma)
        scored = score_records(record.steps, ret, behaviour, device, gamma)
        states.append(scored)

    team = np.concatenate([s["team_deviation"] for s in states]) if states else np.array([])
    q_off = np.concatenate([s["q_off_tot"] for s in states]) if states else np.array([])
    ret_go = np.concatenate([s["return_to_go"] for s in states]) if states else np.array([])
    finite = np.isfinite(team) & np.isfinite(q_off) & np.isfinite(ret_go)
    summary = {
        "n_episodes": n_episodes,
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "win_rate": float(np.mean(wins)) if wins else None,
        "n_states": int(finite.sum()),
        "mean_team_deviation": float(np.nanmean(team)) if team.size else float("nan"),
    }
    if finite.sum() >= 8:
        from scipy.stats import spearmanr

        rho = spearmanr(q_off[finite], ret_go[finite]).statistic
        bias = q_off[finite] - ret_go[finite]
        denom = r_max if r_max is not None else np.maximum(np.abs(ret_go[finite]), 1.0)
        summary["spearman_rho"] = float(rho)
        summary["bias_mean"] = float(bias.mean())
        summary["bias_normalized_mean"] = float((bias / denom).mean())
    return summary, {"team": team, "q_off": q_off, "return_to_go": ret_go}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", required=True, choices=sorted(MAP_CONFIG))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--t-steps", nargs="+", type=int, default=list(NATURAL_SHIFT_STEPS))
    ap.add_argument("--eps-values", nargs="+", type=float, default=list(RANDOM_SHIFT_EPS))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--behaviour-ckpt", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-seed", type=int, default=424242)
    ap.add_argument("--smoke", action="store_true", help="2 episodes per config")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    ensure_project_root()
    cfg = load_cfg(MAP_CONFIG[args.map], list(args.overrides))
    dataset = load_vault_dataset(cfg)
    learner = build_learner(cfg, dataset)
    device = learner.device
    gamma = float(cfg["system"]["discount"])
    n_agents = int(dataset.stats.n_agents)
    n_actions = int(dataset.stats.n_actions)
    add_agent_id = bool(cfg["system"].get("add_agent_id_to_obs", True))

    offline_ckpt = resolve_path(MAP_OFFLINE[args.map])
    if not os.path.isfile(offline_ckpt):
        raise SystemExit(f"FATAL: offline checkpoint missing: {offline_ckpt}")
    if hasattr(learner, "load_offline_for_ovm"):
        learner.load_offline_for_ovm(str(offline_ckpt))
    offline = Net(q_network=learner.offline_q_network, mixer=learner.offline_mixer)
    live = Net(q_network=learner.q_network, mixer=learner.mixer)

    behaviour = None
    bc_path = args.behaviour_ckpt or os.path.join(
        "checkpoints", "pv_behaviour", args.map, "model.pt"
    )
    if os.path.isfile(str(resolve_path(bc_path))):
        behaviour = load_behaviour_model(str(resolve_path(bc_path)), device=str(device))
        print(f"[2a] behaviour model {bc_path}", flush=True)
    else:
        print(f"[2a] WARNING: no behaviour model at {bc_path}; D(s) disabled", flush=True)

    r_max = R_MAX.get(args.map)
    n_episodes = 2 if args.smoke else args.episodes
    rng = np.random.default_rng(args.eval_seed)
    configs = configs_for(n_agents)
    if args.smoke:
        configs = [c for c in configs if c["name"] in ("all", "single_0", "pair_0_1")][:2]

    results = {
        "map": args.map,
        "seed": args.seed,
        "episodes_per_config": n_episodes,
        "eval_seed": args.eval_seed,
        "n_agents": n_agents,
        "natural": {},
        "random": {},
    }

    for t in args.t_steps:
        ckpt = checkpoint_at(args.map, args.seed, t)
        if not os.path.isfile(ckpt):
            print(f"[2a] skip t={t}: checkpoint missing {ckpt}", flush=True)
            continue
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        learner.q_network.load_state_dict(payload["q_network"])
        learner.mixer.load_state_dict(payload["mixer"])
        if "target_q_network" in payload:
            learner.target_q_network.load_state_dict(payload["target_q_network"])
        if "target_mixer" in payload:
            learner.target_mixer.load_state_dict(payload["target_mixer"])
        print(f"[2a] t={t} loaded {ckpt}", flush=True)

        per_config = {}
        for cfg_spec in configs:
            # One env per configuration with the shared evaluation seed, so every
            # configuration is scored on the same episode sequence.
            env = make_env(cfg, seed=args.eval_seed)
            summary, _ = evaluate_config(
                env,
                live=live,
                offline=offline,
                offline_agents=cfg_spec["offline_agents"],
                n_agents=n_agents,
                eps_partner=0.0,
                examined=set(cfg_spec["offline_agents"]),
                behaviour=behaviour,
                gamma=gamma,
                n_actions=n_actions,
                add_agent_id=add_agent_id,
                device=device,
                n_episodes=n_episodes,
                rng=rng,
                r_max=r_max,
            )
            per_config[cfg_spec["name"]] = summary
            print(f"[2a]   t={t} {cfg_spec['name']}: {summary}", flush=True)
        results["natural"][str(t)] = per_config

    # Random shift: everyone keeps the offline parameters, partners add eps noise.
    for eps in args.eps_values:
        env = make_env(cfg, seed=args.eval_seed)
        summary, _ = evaluate_config(
            env,
            live=live,
            offline=offline,
            offline_agents=list(range(n_agents)),
            n_agents=n_agents,
            eps_partner=eps,
            examined=set(),
            behaviour=behaviour,
            gamma=gamma,
            n_actions=n_actions,
            add_agent_id=add_agent_id,
            device=device,
            n_episodes=n_episodes,
            rng=rng,
            r_max=r_max,
        )
        results["random"][str(eps)] = summary
        print(f"[2a]   eps={eps}: {summary}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, default=str)
    print(f"[2a] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
