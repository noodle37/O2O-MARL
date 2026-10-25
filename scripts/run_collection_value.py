#!/usr/bin/env python3
"""Insight F: collection value B_v(z) vs teaching value T_v(z).

Collect an equal episode budget from several collectors z (offline teacher,
online policy at 250k, online policy at 1M) on the same scenario sequence.  Then
train identical student snapshots for a fixed update budget on each collector's
data and evaluate the student on the same fixed evaluation sequence:

    B_v(z) = J_v(pi^z)              (collector's own greedy performance)
    T_v(z) = J_v(student after z) - J_v(student before)

The experiment repeats the student with the offline checkpoint and the 250k
checkpoint, and repeats the offline-student budget at 100/300 updates, so the
teaching value can be compared at two timescales.  If the ranking by B differs
from the ranking by T, collector quality and teaching quality are distinct.

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/run_collection_value.py \
      --collect-episodes 48 --updates 100 300 \
      --out checkpoints/collection_value/5m_vs_6m.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.evaluation import evaluate
from training.factory import build_learner, load_vault_dataset, make_env
from training.rollout import collect_episode, stack_episodes
from utils.checkpoint import load_qmix_checkpoint
from utils.paths import ensure_project_root, resolve_path


COLLECTORS = {
    "offline": "checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt",
    "online_250k": "checkpoints/pv_exp1/5m_vs_6m/{arm}_seed{seed}/periodic/ovm_diag_ckpt_t250000.pt",
    "online_1m": "checkpoints/pv_exp1/5m_vs_6m/{arm}_seed{seed}/periodic/ovm_diag_ckpt_t1000000.pt",
}
STUDENTS = {
    "offline": "checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt",
    "online_250k": "checkpoints/pv_exp1/5m_vs_6m/{arm}_seed{seed}/periodic/ovm_diag_ckpt_t250000.pt",
    "online_1m": "checkpoints/pv_exp1/5m_vs_6m/{arm}_seed{seed}/periodic/ovm_diag_ckpt_t1000000.pt",
}


def snapshot_learner(learner) -> Dict[str, Any]:
    return {
        "q": copy.deepcopy(learner.q_network.state_dict()),
        "mixer": copy.deepcopy(learner.mixer.state_dict()),
        "target_q": copy.deepcopy(learner.target_q_network.state_dict()),
        "target_mixer": copy.deepcopy(learner.target_mixer.state_dict()),
        "optimizer": copy.deepcopy(learner.optimizer.state_dict()),
    }


def restore_learner(learner, snap: Dict[str, Any]) -> None:
    learner.q_network.load_state_dict(snap["q"])
    learner.mixer.load_state_dict(snap["mixer"])
    learner.target_q_network.load_state_dict(snap["target_q"])
    learner.target_mixer.load_state_dict(snap["target_mixer"])
    learner.optimizer.load_state_dict(snap["optimizer"])
    learner.episode_counter = 0
    learner.last_target_update_episode = 0


def fresh_eval(cfg, learner, n_episodes: int, t_env: int, seed: int,
               return_episodes: bool = False) -> Dict[str, Any]:
    env = make_env(cfg, seed=seed)
    try:
        logs = evaluate(env, learner, n_episodes, t_env, cfg,
                        return_episodes=return_episodes)
    finally:
        try:
            env._environment.close()
        except Exception:
            pass
    out = {
        "return": float(logs["evaluation/mean_episode_return"]),
        "win_rate": float(logs.get("evaluation/win_rate", float("nan"))),
    }
    if return_episodes:
        out["episode_returns"] = logs.get("evaluation/episode_returns", [])
    return out


def collect(cfg, learner, n_episodes: int, seed: int, eps: float) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    env = make_env(cfg, seed=seed)
    episodes = []
    try:
        for _ in range(n_episodes):
            episodes.append(collect_episode(env, learner, eps))
    finally:
        try:
            env._environment.close()
        except Exception:
            pass
    returns = [float(sum(s["reward"] for s in ep["steps"])) for ep in episodes]
    lengths = [int(ep["length"]) for ep in episodes]
    return episodes, {
        "n_episodes": n_episodes,
        "behavior_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "episode_returns": returns,
        "episode_lengths": lengths,
    }


def train_student(
    learner, snap, stacked, *, updates: int, batch_eps: int, seed: int, lr: float | None = None
) -> Dict[str, float]:
    restore_learner(learner, snap)
    if lr is not None:
        for group in learner.optimizer.param_groups:
            group["lr"] = lr
    rng = np.random.default_rng(seed)
    n = stacked["obs"].shape[0]
    losses = []
    for _ in range(updates):
        idx = torch.as_tensor(
            rng.choice(n, size=min(batch_eps, n), replace=False), device=stacked["obs"].device
        )
        batch = {k: v.index_select(0, idx) for k, v in stacked.items()}
        logs = learner.train_on_batch(
            batch, cql_weight=0.0, use_ovm=False, t_env=0, episode_delta=1
        )
        losses.append(float(logs["loss"]))
    return {
        "mean_loss_first50": float(np.mean(losses[:50])),
        "mean_loss_last50": float(np.mean(losses[-50:])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/pv_base_5m_vs_6m.yaml")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arm", default="A", help="PV exp1 arm prefix, e.g. A/B/C/D")
    ap.add_argument("--students", nargs="+", default=["offline", "online_250k"],
                    choices=sorted(STUDENTS), help="student snapshots to train")
    ap.add_argument("--collect-episodes", type=int, default=48)
    ap.add_argument("--collect-eps", type=float, default=0.05)
    ap.add_argument("--updates", nargs="+", type=int, default=[100, 300])
    ap.add_argument("--batch-eps", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None, help="student learning rate override")
    ap.add_argument("--eval-episodes", type=int, default=64)
    ap.add_argument("--eval-seed", type=int, default=424242)
    ap.add_argument("--collect-seed", type=int, default=515151)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    ensure_project_root()
    cfg = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides)),
        resolve=True,
    )
    dataset = load_vault_dataset(cfg)
    learner = build_learner(cfg, dataset)
    train_seed = args.seed + 401

    # Student baselines on the fixed evaluation sequence.
    student_snaps = {}
    base_evals = {}
    for name in args.students:
        path_t = STUDENTS[name]
        path = str(resolve_path(path_t.format(seed=args.seed, arm=args.arm)))
        load_qmix_checkpoint(path, learner, load_optimizer=True)
        student_snaps[name] = snapshot_learner(learner)
        learner.episode_counter = 0
        learner.last_target_update_episode = 0
        base_evals[name] = fresh_eval(
            cfg, learner, args.eval_episodes, 0, args.eval_seed, return_episodes=True
        )
        print(f"[F] student={name} base {base_evals[name]}", flush=True)

    results = {
        "map": cfg["task"]["scenario"],
        "arm": args.arm,
        "collect_episodes": args.collect_episodes,
        "collect_eps": args.collect_eps,
        "eval_episodes": args.eval_episodes,
        "eval_seed": args.eval_seed,
        "collect_seed": args.collect_seed,
        "batch_eps": args.batch_eps,
        "student_lr": args.lr,
        "student_names": list(args.students),
        "students": {name: {"checkpoint": STUDENTS[name], "base_eval": base_evals[name]}
                     for name in args.students},
        "collectors": {},
    }
    max_seq = int(dataset.stats.max_seq_length)
    for name, path_t in COLLECTORS.items():
        path = str(resolve_path(path_t.format(seed=args.seed, arm=args.arm)))
        if not os.path.isfile(path):
            print(f"[F] skip collector {name}: missing {path}", flush=True)
            continue
        load_qmix_checkpoint(path, learner, load_optimizer=False)
        learner.episode_counter = 0
        learner.last_target_update_episode = 0
        B = fresh_eval(
            cfg, learner, args.eval_episodes, 0, args.eval_seed, return_episodes=True
        )
        t0 = time.time()
        episodes, data_stats = collect(
            cfg, learner, args.collect_episodes, args.collect_seed, args.collect_eps
        )
        stacked = stack_episodes(episodes, max_seq, learner.device)
        collector = {
            "checkpoint": path,
            "B_own_policy": B,
            "data": data_stats,
            "collection_seconds": round(time.time() - t0, 1),
            "students": {},
        }
        print(f"[F] collector={name} B={B['return']:.3f} data={data_stats}", flush=True)
        for student_name, snap in student_snaps.items():
            budgets = args.updates if student_name == "offline" else [max(args.updates)]
            for budget in budgets:
                t1 = time.time()
                train_logs = train_student(
                    learner, snap, stacked, updates=budget, batch_eps=args.batch_eps,
                    seed=train_seed, lr=args.lr,
                )
                final = fresh_eval(
                    cfg, learner, args.eval_episodes, 0, args.eval_seed,
                    return_episodes=True,
                )
                key = f"u{budget}"
                collector["students"].setdefault(student_name, {})[key] = {
                    "train": train_logs,
                    "eval": final,
                    "T_return": float(final["return"] - base_evals[student_name]["return"]),
                    "T_win": float(final["win_rate"] - base_evals[student_name]["win_rate"]),
                    "seconds": round(time.time() - t1, 1),
                }
                print(
                    f"[F]   student={student_name} u={budget} "
                    f"T_return={collector['students'][student_name][key]['T_return']:+.3f} "
                    f"loss={train_logs['mean_loss_last50']:.4f}",
                    flush=True,
                )
        results["collectors"][name] = collector

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"[F] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
