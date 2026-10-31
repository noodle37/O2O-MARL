#!/usr/bin/env python3
"""Premise measurement for teammate-conditioned validity of the offline value memory.

The offline value memory (OVM) replaces the TD target y by max(q_off, y), where
q_off is the frozen offline QMIX value of the executed joint action.  That floor
is a claim about the *continuation*: it asserts the current team can still earn
q_off from here.  This script measures, per sample, whether the floor moves the
target toward or away from the return the current team actually realised, and
whether a teammate-behaviour-agreement signal predicts which of the two happens.

Per filled step of a collected episode it records
    y        TD(lambda) target of the loaded online learner,
    q_off    frozen offline Q_tot of the executed joint action,
    G        realised discounted return-to-go under the current team,
and per agent
    off_margin  max_a Q_off(tau_j,a) - Q_off(tau_j,a_exec)   (0 == full agreement)
    off_logp    log softmax(Q_off(tau_j,.)/T)[a_exec]
    off_greedy  1 if a_exec is the offline greedy action
    on_margin   same margin under the live online network (comparator signal).

Nothing is updated; the script only runs forward passes.  Aggregation, forward
windows and AUCs live in scripts/analyze_tcvm_premise.py so that the stored
records stay raw.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_tcvm_premise.py \
    --config configs/pv_base_5m_vs_6m.yaml \
    --policy-ckpt checkpoints/pv_exp1/5m_vs_6m/A_seed1/periodic/ovm_diag_ckpt_t250000.pt \
    --tag A_seed1_t250000 --episodes 200 \
    --out checkpoints/tcvm_premise/5m_vs_6m
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ovmse.targets import build_td_lambda_targets
from training.factory import build_learner, load_vault_dataset, make_env
from training.rollout import collect_parallel, stack_episodes
from utils.checkpoint import load_qmix_checkpoint
from utils.paths import ensure_project_root, resolve_path

OFFLINE_REFERENCE = (
    "checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt"
)
NEG_INF = -9999999.0


def load_cfg(path: str, overrides: List[str]) -> Dict[str, Any]:
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


def td_targets_and_mask(learner, batch: Dict[str, torch.Tensor]):
    """Exactly the target/mask construction of PaperQMIXLearner.train_on_batch."""
    actions = batch["actions"]
    avail = batch["avail_actions"]
    states = batch["state"]
    rewards = batch["reward"]
    terminated = batch["terminated"]
    filled = batch["filled"]

    term = terminated[:, :-1].float()
    if term.dim() == 3 and term.shape[2] > 1:
        term = term.max(dim=2, keepdim=True)[0]
    mask = filled[:, :-1].float()
    mask[:, 1:] = mask[:, 1:] * (1.0 - term[:, :-1])

    with torch.no_grad():
        target_mac_out = learner._forward_qs(batch, target=True)
        mac_out = learner._forward_qs(batch, target=False)
        mac_sel = torch.where(avail, mac_out, torch.tensor(NEG_INF, device=learner.device))
        max_actions = torch.argmax(mac_sel, dim=3, keepdim=True)
        target_max_q = torch.gather(target_mac_out, 3, max_actions)
        chosen_q = torch.gather(mac_out, 3, actions.long())
        chosen_tot, target_max_tot, mixed_rew = learner._mixing(
            chosen_q, target_max_q, states, rewards
        )
        y = build_td_lambda_targets(
            mixed_rew[:, :-1],
            terminated[:, :-1].max(dim=2, keepdim=True)[0],
            mask,
            target_max_tot,
            learner.gamma,
            learner.td_lambda,
        )
    return y, mask, mac_out, chosen_tot


def offline_quantities(learner, batch: Dict[str, torch.Tensor], temp: float):
    """Frozen offline Q_tot of the executed joint action plus per-agent agreement."""
    actions = batch["actions"]
    avail = batch["avail_actions"]
    states = batch["state"]
    with torch.no_grad():
        off_out = learner._unroll(
            learner.offline_q_network,
            learner._augment_obs(batch["obs"], actions),
            learner._resets_from_batch(batch),
        )
        off_chosen = torch.gather(off_out[:, :-1], 3, actions[:, :-1].long())
        off_tot = learner.offline_mixer(off_chosen, states[:, :-1])
        masked = torch.where(avail, off_out, torch.tensor(NEG_INF, device=learner.device))
        best = masked.max(dim=3, keepdim=True)[0]
        chosen = torch.gather(off_out, 3, actions.long())
        margin = (best - chosen).squeeze(-1)
        greedy_eq = (masked.argmax(dim=3) == actions.long().squeeze(-1)).float()
        logp = torch.log_softmax(masked / max(float(temp), 1e-6), dim=3)
        logp_chosen = torch.gather(logp, 3, actions.long()).squeeze(-1)
    return off_tot, margin, greedy_eq, logp_chosen


def online_margin(learner, batch: Dict[str, torch.Tensor], mac_out: torch.Tensor):
    avail = batch["avail_actions"]
    actions = batch["actions"]
    with torch.no_grad():
        masked = torch.where(avail, mac_out, torch.tensor(NEG_INF, device=learner.device))
        best = masked.max(dim=3, keepdim=True)[0]
        chosen = torch.gather(mac_out, 3, actions.long())
        return (best - chosen).squeeze(-1)


def return_to_go(reward: torch.Tensor, mask: torch.Tensor, gamma: float) -> torch.Tensor:
    """G_t over the filled prefix of each episode (reward [B,T,1], mask [B,T-1,1])."""
    r = (reward[:, :-1, 0] * mask[:, :, 0]).clone()
    out = torch.zeros_like(r)
    acc = torch.zeros(r.shape[0], device=r.device)
    for t in range(r.shape[1] - 1, -1, -1):
        acc = r[:, t] + gamma * acc
        out[:, t] = acc
    return out


def records_from_batch(learner, batch: Dict[str, torch.Tensor], temp: float, offset: int):
    y, mask, mac_out, chosen_tot = td_targets_and_mask(learner, batch)
    off_tot, off_margin, off_greedy, off_logp = offline_quantities(learner, batch, temp)
    on_margin = online_margin(learner, batch, mac_out)
    g = return_to_go(batch["reward"], mask, learner.gamma)

    keep = mask[:, :, 0] > 0.5
    b_idx, t_idx = torch.nonzero(keep, as_tuple=True)
    lengths = keep.float().sum(dim=1)
    rows = {
        "episode": (b_idx + offset).cpu().numpy().astype(np.int32),
        "t": t_idx.cpu().numpy().astype(np.int32),
        "length": lengths[b_idx].cpu().numpy().astype(np.int32),
        "y": y[:, :, 0][keep].cpu().numpy().astype(np.float32),
        "q_off": off_tot[:, :, 0][keep].cpu().numpy().astype(np.float32),
        "q_on": chosen_tot[:, :-1, 0][keep].cpu().numpy().astype(np.float32),
        "g": g[keep].cpu().numpy().astype(np.float32),
        "reward": batch["reward"][:, :-1, 0][keep].cpu().numpy().astype(np.float32),
        "off_margin": off_margin[:, :-1][keep].cpu().numpy().astype(np.float32),
        "off_greedy": off_greedy[:, :-1][keep].cpu().numpy().astype(np.float32),
        "off_logp": off_logp[:, :-1][keep].cpu().numpy().astype(np.float32),
        "on_margin": on_margin[:, :-1][keep].cpu().numpy().astype(np.float32),
    }
    return rows, int(batch["obs"].shape[0])


def collect_online_episodes(cfg, learner, n_episodes: int, eps: float, seed: int):
    n_parallel = int(cfg.get("batch_size_run", 8))
    envs = [make_env(cfg, seed=seed + i) for i in range(n_parallel)]
    episodes: List[Dict[str, Any]] = []
    try:
        while len(episodes) < n_episodes:
            episodes.extend(collect_parallel(envs, learner, eps))
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
    return episodes[:n_episodes]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy-ckpt", default="")
    parser.add_argument("--offline-ckpt", default=OFFLINE_REFERENCE)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source", choices=("online", "offline_data"), default="online")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--env-seed", type=int, default=424242)
    parser.add_argument("--out", default="checkpoints/tcvm_premise/5m_vs_6m")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    ensure_project_root()
    cfg = load_cfg(args.config, list(args.overrides))
    dataset = load_vault_dataset(cfg)
    learner = build_learner(cfg, dataset)
    learner.use_ovm = False
    if hasattr(learner, "ovm_diag_train_stats"):
        learner.ovm_diag_train_stats = False

    if args.source == "online":
        if not args.policy_ckpt:
            raise ValueError("--policy-ckpt is required for --source online")
        load_qmix_checkpoint(
            str(resolve_path(args.policy_ckpt)), learner, map_location=learner.device
        )
    elif args.policy_ckpt:
        load_qmix_checkpoint(
            str(resolve_path(args.policy_ckpt)), learner, map_location=learner.device
        )
    learner.load_offline_for_ovm(str(resolve_path(args.offline_ckpt)))

    chunk = 32
    rows: List[Dict[str, np.ndarray]] = []
    offset = 0
    if args.source == "online":
        episodes = collect_online_episodes(
            cfg, learner, int(args.episodes), float(args.eps), int(args.env_seed)
        )
        for start in range(0, len(episodes), chunk):
            batch = stack_episodes(
                episodes[start : start + chunk],
                dataset.stats.max_seq_length,
                learner.device,
            )
            part, n = records_from_batch(learner, batch, args.temp, offset)
            rows.append(part)
            offset += n
    else:
        rng = np.random.default_rng(int(args.env_seed))
        taken = 0
        while taken < int(args.episodes):
            take = min(chunk, int(args.episodes) - taken)
            batch = dataset.sample(take, rng, device=learner.device)
            part, n = records_from_batch(learner, batch, args.temp, offset)
            rows.append(part)
            offset += n
            taken += take

    merged = {key: np.concatenate([part[key] for part in rows]) for key in rows[0]}
    os.makedirs(str(resolve_path(args.out)), exist_ok=True)
    npz_path = os.path.join(str(resolve_path(args.out)), f"{args.tag}.npz")
    np.savez_compressed(npz_path, **merged)

    floor_active = merged["q_off"] > merged["y"]
    target_ovm = np.maximum(merged["q_off"], merged["y"])
    gain = np.abs(merged["y"] - merged["g"]) - np.abs(target_ovm - merged["g"])
    meta = {
        "tag": args.tag,
        "source": args.source,
        "policy_ckpt": args.policy_ckpt,
        "offline_ckpt": args.offline_ckpt,
        "episodes": int(offset),
        "samples": int(merged["y"].size),
        "eps": float(args.eps),
        "temp": float(args.temp),
        "env_seed": int(args.env_seed),
        "mean_y_bias": float(np.mean(merged["y"] - merged["g"])),
        "mean_q_off_bias": float(np.mean(merged["q_off"] - merged["g"])),
        "floor_active_rate": float(np.mean(floor_active)),
        "mean_uplift": float(np.mean(np.maximum(merged["q_off"] - merged["y"], 0.0))),
        "mean_floor_gain": float(np.mean(gain)),
        "mean_floor_gain_active": float(np.mean(gain[floor_active])) if floor_active.any() else 0.0,
        "mean_return_g": float(np.mean(merged["g"])),
        "npz": npz_path,
    }
    with open(os.path.join(str(resolve_path(args.out)), f"{args.tag}.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
