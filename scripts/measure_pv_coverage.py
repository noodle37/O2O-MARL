#!/usr/bin/env python3
# Copyright 2024 InstaDeep Ltd. All rights reserved.

"""Coverage ratio rho_cov for one online checkpoint (experiment 1 metric).

Roll out episodes with the checkpoint's *behaviour* policy -- the same epsilon
the online run had at that step, recomputed from the linear schedule -- and score
every executed joint action under the joint behaviour model.  rho_cov is the
fraction of those joint actions whose deviation D(s) is at or below the held-out
q95 threshold fixed before any online data was seen.

The design defines rho_cov over "the last 10k steps of online joint actions".
This script operationalises it as a fresh behaviour rollout at the same epsilon,
which is reproducible from the periodic checkpoints alone; the training-time
alternative would require logging every executed joint action.

Usage:
  python scripts/measure_pv_coverage.py --map 5m_vs_6m \
      --checkpoint checkpoints/pv_exp1/5m_vs_6m/C_seed1/periodic/ovm_diag_ckpt_t50000.pt \
      --episodes 32 --out /tmp/cov.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pv.behaviour import load_behaviour_model
from pv.exp2_shift import _stack_steps
from training.factory import build_learner, load_vault_dataset, make_env
from training.rollout import collect_episode
from utils.paths import ensure_project_root, resolve_path

MAP_CONFIG = {
    "5m_vs_6m": "configs/pv_base_5m_vs_6m.yaml",
    "2halfcheetah": "configs/pv_base_2halfcheetah.yaml",
}


def load_cfg(path: str, overrides: list) -> dict:
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", required=True, choices=sorted(MAP_CONFIG))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--behaviour-dir", default=None)
    ap.add_argument("--env-seed", type=int, default=424242)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    ensure_project_root()
    cfg = load_cfg(MAP_CONFIG[args.map], list(args.overrides))
    dataset = load_vault_dataset(cfg)
    learner = build_learner(cfg, dataset)
    device = learner.device
    learner.sequential_explore = bool(cfg["system"].get("sequential_explore", False))

    bdir = args.behaviour_dir or os.path.join("checkpoints", "pv_behaviour", args.map)
    model = load_behaviour_model(str(resolve_path(os.path.join(bdir, "model.pt"))), device=str(device))
    with open(str(resolve_path(os.path.join(bdir, "thresholds.json"))), encoding="utf-8") as handle:
        thresholds = json.load(handle)
    q95 = float(thresholds["q95"])

    payload = torch.load(str(resolve_path(args.checkpoint)), map_location=device, weights_only=False)
    learner.q_network.load_state_dict(payload["q_network"])
    learner.mixer.load_state_dict(payload["mixer"])

    # Recompute the epsilon this checkpoint was collected at.
    sys_cfg = cfg["system"]
    t_env = int(payload.get("extra", {}).get("t_env", 0))
    eps = learner.linear_epsilon(
        t_env,
        float(sys_cfg.get("eps_start", 1.0)),
        float(sys_cfg.get("eps_min", 0.05)),
        int(sys_cfg.get("eps_decay_steps", 100000)),
    )

    env = make_env(cfg, seed=args.env_seed)
    all_dev, all_nll, lengths = [], [], []
    for ep_idx in range(args.episodes):
        record = collect_episode(env, learner, eps)
        lengths.append(record["length"])
        n = int(record["length"])
        # _stack_steps appends one bootstrap slot, so slice to the real steps
        # before applying the filled mask.
        batch = _stack_steps(record["steps"], n + 1, device)
        dev = model.deviations(batch)
        mask = dev["mask"][0].detach().cpu().numpy()[:n]
        team = dev["team"][0].detach().cpu().numpy()[:n]
        all_dev.append(team[mask])
        all_nll.append(team[mask] * model.n_agents)  # D = -log p / n

    dev = np.concatenate(all_dev) if all_dev else np.array([])
    nll = np.concatenate(all_nll) if all_nll else np.array([])
    finite = np.isfinite(dev)
    result = {
        "map": args.map,
        "checkpoint": str(args.checkpoint),
        "t_env": t_env,
        "epsilon": float(eps),
        "episodes": int(args.episodes),
        "env_seed": int(args.env_seed),
        "q95_threshold": q95,
        "n_states": int(finite.sum()),
        "rho_cov": float(np.mean(dev[finite] <= q95)) if finite.any() else float("nan"),
        "mean_deviation": float(np.mean(dev[finite])) if finite.any() else float("nan"),
        "mean_joint_nll": float(np.mean(nll[finite])) if finite.any() else float("nan"),
        "median_episode_length": float(np.median(lengths)) if lengths else float("nan"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(
        f"[cov] rho_cov={result['rho_cov']:.4f} (q95={q95:.3f}, eps={eps:.3f}, "
        f"states={result['n_states']}) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
