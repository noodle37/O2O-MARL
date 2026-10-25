#!/usr/bin/env python3
"""Insight E: does replay selection by short-horizon learning gain differ from
density / TD-error / return proxies?

For a fixed learner snapshot theta_0 and policy version v:

    U_v(j) = J(theta_j^+) - J(theta_0)

where theta_j^+ is produced by exactly M updates from theta_0 on experience group
j, with the same batch size, mixing ratio, optimizer state and update count.
Groups are the top quartile by TD-error, state density, episode return, plus a
random quartile.  J is the mean return of 64 greedy episodes on a freshly
launched evaluation environment with the same seed for every policy, so all
groups are scored on the same episode sequence.

The experiment is run at two policy versions.  It reports the U table, proxy
rank agreement, and whether the U ranking is stable across versions.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/run_replay_gain.py \
      --t-steps 250000 1000000 --updates 500 --out checkpoints/replay_gain/5m_vs_6m.json
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
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ovmse.targets import build_td_lambda_targets
from training.evaluation import evaluate
from training.factory import build_learner, load_vault_dataset, make_env
from utils.checkpoint import load_qmix_checkpoint
from utils.paths import ensure_project_root, resolve_path


def select_episodes(dataset, idx: torch.Tensor, device) -> Dict[str, torch.Tensor]:
    idx_t = idx.to(dataset.device).long()
    return {
        "obs": dataset.obs.index_select(0, idx_t).to(device),
        "actions": dataset.actions.index_select(0, idx_t).to(device),
        "state": dataset.state.index_select(0, idx_t).to(device),
        "avail_actions": dataset.avail_actions.index_select(0, idx_t).to(device),
        "reward": dataset.reward.index_select(0, idx_t).to(device),
        "terminated": dataset.terminated.index_select(0, idx_t).to(device),
        "filled": dataset.filled.index_select(0, idx_t).to(device),
    }


def parse_state(state: torch.Tensor, n_agents: int, n_actions: int):
    nf_al, nf_en = 4, 3
    n_en = (state.shape[-1] - n_agents * nf_al - n_agents * n_actions) // nf_en
    allies = state[..., : n_agents * nf_al].reshape(*state.shape[:-1], n_agents, nf_al)
    base = n_agents * nf_al
    enemies = state[..., base : base + n_en * nf_en].reshape(*state.shape[:-1], n_en, nf_en)
    return allies, enemies


@torch.no_grad()
def episode_proxies(dataset, n_agents: int, n_actions: int) -> Dict[str, np.ndarray]:
    state = dataset.state
    filled = dataset.filled.squeeze(-1) > 0.5
    rewards = dataset.reward.squeeze(-1)
    ret = (rewards * filled.float()).sum(dim=1).cpu().numpy()
    allies, enemies = parse_state(state, n_agents, n_actions)
    phi = torch.cat(
        [
            allies[..., 0],
            allies[..., 1:3].reshape(*state.shape[:-1], -1),
            enemies[..., 0],
            enemies[..., 1:3].reshape(*state.shape[:-1], -1),
        ],
        dim=-1,
    )
    mean_phi = (phi * filled.unsqueeze(-1).float()).sum(dim=1) / filled.sum(
        dim=1, keepdim=True
    ).clamp_min(1)
    z = mean_phi.cpu().numpy()
    z = (z - z.mean(0)) / z.std(0).clip(1e-6)
    nn = NearestNeighbors(n_neighbors=11).fit(z)
    dist, _ = nn.kneighbors(z)
    density = -dist[:, 1:].mean(axis=1)
    return {"return": ret, "density": density, "mean_phi": z}


@torch.no_grad()
def td_error_scores(learner, dataset, chunk: int = 128) -> np.ndarray:
    n = dataset.n_episodes
    out = np.zeros(n, dtype=np.float64)
    for start in range(0, n, chunk):
        idx = torch.arange(start, min(start + chunk, n), device=dataset.device)
        batch = select_episodes(dataset, idx, learner.device)
        actions = batch["actions"]
        avail = batch["avail_actions"]
        chosen = torch.gather(learner._forward_qs(batch), 3, actions.long())
        chosen_tot = learner.mixer(chosen, batch["state"])
        target_out = learner._forward_qs(batch, target=True)
        mac_sel = torch.where(avail, target_out, torch.tensor(-1e9, device=learner.device))
        max_actions = torch.argmax(mac_sel, dim=3, keepdim=True)
        target_max = torch.gather(target_out, 3, max_actions)
        target_tot = learner.target_mixer(target_max, batch["state"])
        term = batch["terminated"][:, :-1].float()
        if term.dim() == 3 and term.shape[2] > 1:
            term = term.max(dim=2, keepdim=True)[0]
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1.0 - term[:, :-1])
        targets = build_td_lambda_targets(
            batch["reward"][:, :-1],
            term,
            mask,
            target_tot,
            learner.gamma,
            learner.td_lambda,
        )
        err = 0.5 * (chosen_tot[:, :-1] - targets) ** 2
        per_ep = (err * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)
        out[start : start + len(idx)] = per_ep.cpu().numpy()
    return out


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


def snapshot_learner(learner) -> Dict[str, Any]:
    return {
        "q": copy.deepcopy(learner.q_network.state_dict()),
        "mixer": copy.deepcopy(learner.mixer.state_dict()),
        "target_q": copy.deepcopy(learner.target_q_network.state_dict()),
        "target_mixer": copy.deepcopy(learner.target_mixer.state_dict()),
        "optimizer": copy.deepcopy(learner.optimizer.state_dict()),
        "episode_counter": learner.episode_counter,
        "last_target": learner.last_target_update_episode,
        "step": learner.training_step_ctr,
    }


def restore_learner(learner, snap: Dict[str, Any]) -> None:
    learner.q_network.load_state_dict(snap["q"])
    learner.mixer.load_state_dict(snap["mixer"])
    learner.target_q_network.load_state_dict(snap["target_q"])
    learner.target_mixer.load_state_dict(snap["target_mixer"])
    learner.optimizer.load_state_dict(snap["optimizer"])
    learner.episode_counter = snap["episode_counter"]
    learner.last_target_update_episode = snap["last_target"]
    learner.training_step_ctr = snap["step"]


def train_group(learner, dataset, group_idx: np.ndarray, *, updates: int, batch_eps: int,
                seed: int, t_env: int = 0, lr: float | None = None) -> Dict[str, float]:
    if lr is not None:
        for group in learner.optimizer.param_groups:
            group["lr"] = lr
    rng = np.random.default_rng(seed)
    losses = []
    t0 = time.time()
    for _ in range(updates):
        size = min(batch_eps, len(group_idx))
        picked = rng.choice(group_idx, size=size, replace=False)
        batch = select_episodes(
            dataset, torch.as_tensor(picked, device=dataset.device), learner.device
        )
        logs = learner.train_on_batch(
            batch, cql_weight=0.0, use_ovm=False, t_env=t_env, episode_delta=1
        )
        losses.append(float(logs["loss"]))
    return {
        "seconds": round(time.time() - t0, 1),
        "mean_loss_last50": float(np.mean(losses[-50:])),
        "mean_loss_first50": float(np.mean(losses[:50])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/pv_base_5m_vs_6m.yaml")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arm", default="A", help="PV exp1 arm prefix, e.g. A/B/C/D")
    ap.add_argument("--t-steps", nargs="+", type=int, default=[250000, 1000000])
    ap.add_argument("--updates", type=int, default=500)
    ap.add_argument("--batch-eps", type=int, default=32)
    ap.add_argument("--eval-episodes", type=int, default=64)
    ap.add_argument("--eval-seed", type=int, default=424242)
    ap.add_argument("--lr", type=float, default=None, help="update learning rate override")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-episodes", action="store_true", help="dump per-episode eval returns")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    ensure_project_root()
    cfg = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides)),
        resolve=True,
    )
    dataset = load_vault_dataset(cfg)
    learner = build_learner(cfg, dataset)
    n_agents = int(dataset.stats.n_agents)
    n_actions = int(dataset.stats.n_actions)
    proxies = episode_proxies(dataset, n_agents, n_actions)
    n = dataset.n_episodes
    k = int(round(0.25 * n))
    rng = np.random.default_rng(args.seed)
    groups = {}
    for name in ("density", "return"):
        order = np.argsort(-proxies[name])
        groups[name] = order[:k]
    groups["random"] = rng.choice(n, size=k, replace=False)
    print(
        "[E] groups "
        + " ".join(f"{name}:{len(idx)}" for name, idx in groups.items()),
        flush=True,
    )

    results = {
        "map": cfg["task"]["scenario"],
        "seed": args.seed,
        "arm": args.arm,
        "save_episodes": bool(args.save_episodes),
        "updates": args.updates,
        "batch_eps": args.batch_eps,
        "eval_episodes": args.eval_episodes,
        "eval_seed": args.eval_seed,
        "lr": args.lr,
        "n_episodes": int(n),
        "group_sizes": {**{name: int(len(idx)) for name, idx in groups.items()}, "td": k},
        "versions": {},
    }
    for t in args.t_steps:
        ckpt = str(
            resolve_path(
                os.path.join(
                    "checkpoints", "pv_exp1", "5m_vs_6m", f"{args.arm}_seed{args.seed}",
                    "periodic", f"ovm_diag_ckpt_t{t}.pt",
                )
            )
        )
        if not os.path.isfile(ckpt):
            print(f"[E] skip t={t}: no checkpoint {ckpt}", flush=True)
            continue
        load_qmix_checkpoint(ckpt, learner, load_optimizer=True)
        learner.episode_counter = 0
        learner.last_target_update_episode = 0
        learner.q_network.eval()
        learner.mixer.eval()
        snap = snapshot_learner(learner)
        t0 = time.time()
        base_eval = fresh_eval(
            cfg, learner, args.eval_episodes, t, args.eval_seed,
            return_episodes=args.save_episodes,
        )
        print(f"[E] t={t} theta0 {base_eval} ({time.time()-t0:.0f}s)", flush=True)
        td_scores = td_error_scores(learner, dataset)
        version = {
            "checkpoint": ckpt,
            "theta0": base_eval,
            "groups": {},
        }
        # Re-rank TD with the snapshot-specific scores; density/return are static.
        order_td = np.argsort(-td_scores)
        version_groups = {
            "td": order_td[:k],
            "density": groups["density"],
            "return": groups["return"],
            "random": groups["random"],
        }
        for name, idx in version_groups.items():
            restore_learner(learner, snap)
            train_logs = train_group(
                learner, dataset, idx, updates=args.updates, batch_eps=args.batch_eps,
                seed=args.seed + 17, t_env=0, lr=args.lr,
            )
            t1 = time.time()
            final = fresh_eval(
                cfg, learner, args.eval_episodes, t, args.eval_seed,
                return_episodes=args.save_episodes,
            )
            version["groups"][name] = {
                "n_episodes": int(len(idx)),
                "train": train_logs,
                "eval": final,
                "U_return": float(final["return"] - base_eval["return"]),
                "U_win": float(final["win_rate"] - base_eval["win_rate"]),
                "eval_seconds": round(time.time() - t1, 1),
            }
            print(
                f"[E] t={t} {name} U_return={version['groups'][name]['U_return']:+.3f} "
                f"U_win={version['groups'][name]['U_win']:+.3f} "
                f"loss={train_logs['mean_loss_last50']:.4f}",
                flush=True,
            )
        results["versions"][str(t)] = version

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"[E] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
