#!/usr/bin/env python3
"""E1: is the reuse benefit of a fixed focal policy conditioned on teammates?

The existing transplant evidence (pv exp2a) changes the teammate and the examined
agent at the same time, so it cannot separate "the offline policy is better here"
from "the teammate composition changed". This experiment holds the focal
checkpoint fixed and changes only partner behaviour:

  focal agents  act greedily from the offline net ("off") or from the live
                checkpoint ("live"); partners always use the live checkpoint and
                take a uniformly random legal action with probability eps.

Every configuration starts from the same env seed with its own identically seeded
RNG, so the episode sequence and the partner noise draws are common random
numbers across the two focal branches. The paired difference

  Delta(eps) = R(focal=offline, partners=live+eps) - R(focal=live, partners=live+eps)

is the reuse benefit at teammate-noise level eps. E1 holds only if Delta changes
with eps with a consistent sign across checkpoints; a flat Delta means the benefit
does not depend on the teammate condition probed here.

Usage:
  python scripts/run_e1_teammate_causal.py \
    --checkpoint C_s2_250k=checkpoints/pv_exp1/5m_vs_6m/C_seed2/periodic/ovm_diag_ckpt_t250000.pt \
    --episodes 64 --eps 0 0.1 0.2 0.3 \
    --focal single_0=0 pair_1_4=1,4 all=0,1,2,3,4 \
    --device cuda --out checkpoints/e1_e7_supplement/e1/C_s2_250k.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pv.exp2_shift import Net, run_episode
from training.factory import build_learner, load_vault_dataset, make_env
from utils.checkpoint import load_qmix_checkpoint
from utils.paths import ensure_project_root, resolve_path

MAP_CONFIG = "configs/pv_base_5m_vs_6m.yaml"
MAP_OFFLINE = "checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt"


def parse_focal(specs: List[str]) -> Dict[str, Tuple[int, ...]]:
    out: Dict[str, Tuple[int, ...]] = {}
    for spec in specs:
        name, _, agents = spec.partition("=")
        if not name or not agents:
            raise SystemExit(f"bad --focal '{spec}' (expected name=idx,idx)")
        out[name] = tuple(int(x) for x in agents.split(",") if x.strip())
    return out


def parse_checkpoints(specs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if not name or not path:
            raise SystemExit(f"bad --checkpoint '{spec}' (expected name=path)")
        resolved = str(resolve_path(path))
        if not os.path.isfile(resolved):
            raise SystemExit(f"checkpoint missing: {resolved}")
        out[name] = resolved
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="label=path; may repeat")
    ap.add_argument("--focal", nargs="+", default=["single_0=0", "pair_1_4=1,4",
                                                  "all=0,1,2,3,4"])
    ap.add_argument("--eps", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--eval-seed", type=int, default=424242)
    ap.add_argument("--offline-ckpt", default=MAP_OFFLINE)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    ensure_project_root()
    cfg = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(MAP_CONFIG), OmegaConf.from_dotlist(args.overrides)),
        resolve=True,
    )
    dataset = load_vault_dataset(cfg)
    learner = build_learner(cfg, dataset)
    device = learner.device
    n_agents = int(dataset.stats.n_agents)
    n_actions = int(dataset.stats.n_actions)
    add_agent_id = bool(cfg["system"].get("add_agent_id_to_obs", True))

    offline_ckpt = str(resolve_path(args.offline_ckpt))
    if not os.path.isfile(offline_ckpt):
        raise SystemExit(f"offline checkpoint missing: {offline_ckpt}")
    learner.load_offline_for_ovm(offline_ckpt)
    offline = Net(q_network=learner.offline_q_network, mixer=learner.offline_mixer)
    live = Net(q_network=learner.q_network, mixer=learner.mixer)

    checkpoints = parse_checkpoints(args.checkpoint)
    focal_sets = parse_focal(args.focal)
    for name, agents in focal_sets.items():
        if not agents or max(agents) >= n_agents:
            raise SystemExit(f"focal '{name}' out of range 0..{n_agents - 1}")

    episodes = 2 if args.smoke else args.episodes
    eps_values = [0.0] if args.smoke else list(args.eps)
    focal_order = list(focal_sets)[:1] if args.smoke else list(focal_sets)

    payload: Dict = {
        "config": MAP_CONFIG,
        "offline_ckpt": offline_ckpt,
        "eval_seed": args.eval_seed,
        "episodes": episodes,
        "n_agents": n_agents,
        "eps": eps_values,
        "checkpoints": checkpoints,
        "runs": [],
    }

    for ckpt_name, ckpt_path in checkpoints.items():
        load_qmix_checkpoint(ckpt_path, learner, load_optimizer=False)
        print(f"[e1] {ckpt_name} loaded {ckpt_path}", flush=True)

        for focal_name in focal_order:
            focal = focal_sets[focal_name]
            is_full_team = set(focal) == set(range(n_agents))
            for eps in eps_values:
                if is_full_team and eps > 0.0:
                    continue  # no partners left to perturb
                for branch in ("live", "off"):
                    sources = {
                        i: ("offline" if (branch == "off" and i in focal) else "live")
                        for i in range(n_agents)
                    }
                    env = make_env(cfg, seed=args.eval_seed)
                    rng = np.random.default_rng(args.eval_seed)
                    returns: List[float] = []
                    wins: List[float] = []
                    lengths: List[int] = []
                    try:
                        for _ in range(episodes):
                            rec = run_episode(
                                env,
                                live=live,
                                offline=offline,
                                sources=sources,
                                examined=set(focal),
                                eps_partner=eps,
                                n_actions=n_actions,
                                add_agent_id=add_agent_id,
                                device=device,
                                rng=rng,
                            )
                            returns.append(float(rec.episode_return))
                            lengths.append(int(rec.length))
                            if rec.won is not None:
                                wins.append(float(rec.won))
                    finally:
                        try:
                            env.close()
                        except Exception:
                            pass
                    row = {
                        "checkpoint": ckpt_name,
                        "focal": focal_name,
                        "focal_agents": list(focal),
                        "eps": float(eps),
                        "branch": branch,
                        "mean_return": float(np.mean(returns)),
                        "win_rate": float(np.mean(wins)) if wins else None,
                        "mean_length": float(np.mean(lengths)),
                        "returns": returns,
                    }
                    payload["runs"].append(row)
                    print(
                        f"[e1] {ckpt_name} focal={focal_name} eps={eps:g} "
                        f"{branch:4s} return={row['mean_return']:.3f} "
                        f"win={row['win_rate']}",
                        flush=True,
                    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[e1] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
