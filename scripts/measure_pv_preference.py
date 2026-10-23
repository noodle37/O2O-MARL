#!/usr/bin/env python3
# Copyright 2024 InstaDeep Ltd. All rights reserved.

"""Individual preference retention on held-out data (experiment 1 metric).

For a given online checkpoint, replay the held-out episodes of the vault split
(the same 10% the behaviour model never saw) through both the live network and
the frozen offline network, take each agent's greedy action under the legal-action
mask, and report the fraction where they agree, averaged over agents.

The offline reference defaults to the run's own start checkpoint; arm D starts
from an affine-rescaled copy whose greedy policy is identical (a > 0 keeps the
mixer monotonic), so either reference gives the same reading.

Usage:
  python scripts/measure_pv_preference.py --map 5m_vs_6m \
      --checkpoint checkpoints/pv_exp1/5m_vs_6m/C_seed1/periodic/ovm_diag_ckpt_t50000.pt \
      --out /tmp/pref.json
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

from pv.behaviour import gather_episodes
from training.factory import build_learner, load_vault_dataset
from utils.paths import ensure_project_root, resolve_path

MAP_CONFIG = {
    "5m_vs_6m": "configs/pv_base_5m_vs_6m.yaml",
    "2halfcheetah": "configs/pv_base_2halfcheetah.yaml",
}
MAP_OFFLINE = {
    "5m_vs_6m": "checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt",
    "2halfcheetah": "checkpoints/pv_exp1_offline/2halfcheetah_medium/ovmse_paper_seed1.pt",
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
    ap.add_argument("--behaviour-dir", default=None, help="dir holding split.json")
    ap.add_argument("--offline-ckpt", default=None, help="override the offline reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    ensure_project_root()
    cfg = load_cfg(MAP_CONFIG[args.map], list(args.overrides))
    dataset = load_vault_dataset(cfg)
    learner = build_learner(cfg, dataset)
    device = learner.device

    split_path = os.path.join(
        args.behaviour_dir or os.path.join("checkpoints", "pv_behaviour", args.map), "split.json"
    )
    if not os.path.isfile(str(resolve_path(split_path))):
        raise SystemExit(f"FATAL: split not found at {split_path}; run train_pv_behaviour.py first")
    with open(str(resolve_path(split_path)), encoding="utf-8") as handle:
        val_idx = json.load(handle)["val_episodes"]

    offline_ckpt = args.offline_ckpt or MAP_OFFLINE[args.map]
    if hasattr(learner, "load_offline_for_ovm"):
        learner.load_offline_for_ovm(str(resolve_path(offline_ckpt)))

    payload = torch.load(str(resolve_path(args.checkpoint)), map_location=device, weights_only=False)
    learner.q_network.load_state_dict(payload["q_network"])
    learner.mixer.load_state_dict(payload["mixer"])

    batch = gather_episodes(dataset, val_idx, device=device)
    with torch.no_grad():
        live_q = learner._forward_qs(batch, target=False)
        obs_aug = learner._augment_obs(batch["obs"], batch["actions"])
        resets = learner._resets_from_batch(batch)
        off_q = learner._unroll(learner.offline_q_network, obs_aug, resets)

    avail = batch["avail_actions"]
    # Masked argmax per agent; illegal actions get -1e9 like the rollout path.
    live_masked = torch.where(avail, live_q, torch.full_like(live_q, -1e9))
    off_masked = torch.where(avail, off_q, torch.full_like(off_q, -1e9))
    live_greedy = torch.argmax(live_masked, dim=-1)
    off_greedy = torch.argmax(off_masked, dim=-1)

    filled = batch["filled"]
    mask = (filled[..., 0] if filled.shape[-1] == 1 else filled.squeeze(-1)) > 0.5
    agree = (live_greedy == off_greedy).float()
    # Per-agent retention, then averaged over agents (design section 2.3).
    per_agent = (agree * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
    retention = float(per_agent.mean().item())
    n_decisions = int((mask.unsqueeze(-1).expand_as(agree)).sum().item())

    result = {
        "map": args.map,
        "checkpoint": str(args.checkpoint),
        "offline_ckpt": str(offline_ckpt),
        "split": str(split_path),
        # The analysis script bins these measurements by training step, so record
        # the step this checkpoint was taken at (the periodic checkpoints carry it
        # in their `extra` block).
        "t_env": int(payload.get("extra", {}).get("t_env", -1)),
        "n_heldout_episodes": len(val_idx),
        "n_agent_decisions": n_decisions,
        "preference_retention": retention,
        "per_agent_retention": [float(x) for x in per_agent.mean(dim=0).tolist()],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"[pref] retention={retention:.4f} over {n_decisions} agent decisions -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
