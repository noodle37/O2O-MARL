#!/usr/bin/env python3
"""Build the fixed state-action sample every OVM diagnostic is scored on.

One sample per map, drawn once with a fixed seed from the same vault the online
run trains on, so an OVM-on arm, an OVM-off arm and every time point inside a run
are scored on byte-identical transitions.

Usage:
  python scripts/make_ovm_diag_sample.py --config configs/5m_vs_6m.yaml \
      --out logs/ovm_diag/fixed_sample_5m_vs_6m.pt --n-episodes 128
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data.vault_dataset import load_episode_dataset_from_vault
from utils.paths import VAULT_DIR, ensure_project_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Map config yaml (task block is used)")
    parser.add_argument("--out", required=True, help="Output .pt path")
    parser.add_argument("--n-episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()

    ensure_project_root()
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    task = cfg["task"]
    print(
        f"[sample] source={task['source']} env={task['env']} scenario={task['scenario']} "
        f"dataset={task['dataset']} action_mode={task.get('action_mode', 'discrete')}"
    )

    dataset = load_episode_dataset_from_vault(
        task["source"],
        task["env"],
        task["scenario"],
        str(task["dataset"]),
        rel_dir=str(VAULT_DIR),
        device="cpu",
        action_mode=str(task.get("action_mode", "discrete")),
    )
    stats = dataset.stats
    print(
        f"[sample] vault episodes={dataset.n_episodes} max_seq={stats.max_seq_length} "
        f"agents={stats.n_agents} obs={stats.obs_dim} state={stats.state_dim} "
        f"actions={stats.n_actions}"
    )

    rng = np.random.default_rng(args.seed)
    n = min(args.n_episodes, dataset.n_episodes)
    idx = np.sort(rng.choice(dataset.n_episodes, size=n, replace=False))
    idx_t = torch.as_tensor(idx, dtype=torch.long)
    batch = {
        "obs": dataset.obs.index_select(0, idx_t).clone(),
        "actions": dataset.actions.index_select(0, idx_t).clone(),
        "state": dataset.state.index_select(0, idx_t).clone(),
        "avail_actions": dataset.avail_actions.index_select(0, idx_t).clone(),
        "reward": dataset.reward.index_select(0, idx_t).clone(),
        "terminated": dataset.terminated.index_select(0, idx_t).clone(),
        "filled": dataset.filled.index_select(0, idx_t).clone(),
    }
    meta = {
        "scenario": str(task["scenario"]),
        "source": str(task["source"]),
        "env": str(task["env"]),
        "dataset": str(task["dataset"]),
        "seed": int(args.seed),
        "n_episodes": int(n),
        "n_available": int(dataset.n_episodes),
        "max_seq_length": int(stats.max_seq_length),
        "n_agents": int(stats.n_agents),
        "n_actions": int(stats.n_actions),
        "n_transitions": int(batch["filled"][:, :-1].sum().item()),
        "config": str(args.config),
    }
    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({"batch": batch, "meta": meta}, out)
    size_mb = os.path.getsize(out) / 1e6
    print(f"[sample] wrote {out} ({size_mb:.1f} MB) meta={meta}")


if __name__ == "__main__":
    main()
