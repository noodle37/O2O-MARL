#!/usr/bin/env python3
# Copyright 2024 InstaDeep Ltd. All rights reserved.

"""Fit the autoregressive joint behaviour model used by the deviation measure D.

Outputs (under --out, default checkpoints/pv_behaviour/<map>):
  model.pt            single behaviour model (or member_<k>.pt with --ensemble K)
  ensemble.json       member list when --ensemble K > 1
  split.json          fixed 90/10 episode split; every later measurement reuses it
  report.json         train/val NLL, uniform-over-legal reference, calibration bins
  thresholds.json     D quantiles (q50/q90/q95/q99) on the held-out split, fixed
                      before any online data is seen

Usage:
  python scripts/train_pv_behaviour.py --config configs/pv_base_5m_vs_6m.yaml \
      --map 5m_vs_6m --ensemble 5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from data.vault_dataset import print_dataset_stats
from pv.behaviour import (
    FitReport,
    JointBehaviourModel,
    NoveltyEnsemble,
    fit_behaviour_model,
    gather_episodes,
    load_behaviour_model,
    random_order,
    save_behaviour_model,
    save_json,
    split_episodes,
)
from training.factory import load_vault_dataset
from utils.paths import ensure_project_root


def load_cfg(path: str, overrides: list[str]) -> dict:
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


def calibration_bins(model: JointBehaviourModel, batch, n_bins: int = 10) -> list[dict]:
    """Confidence calibration of the argmax action over the held-out split."""
    conf, correct = model.confidence_and_correct(batch)
    if conf.numel() == 0:
        return []
    edges = torch.quantile(conf, torch.linspace(0, 1, n_bins + 1))
    edges[0] = 0.0
    edges[-1] = 1.0
    rows = []
    for k in range(n_bins):
        lo, hi = float(edges[k]), float(edges[k + 1])
        sel = (conf >= lo) & (conf <= hi if k == n_bins - 1 else conf < hi)
        if sel.sum() < 1:
            continue
        rows.append(
            {
                "bin": k,
                "conf_low": lo,
                "conf_high": hi,
                "n": int(sel.sum().item()),
                "mean_conf": float(conf[sel].mean().item()),
                "accuracy": float(correct[sel].mean().item()),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="config carrying the task block")
    ap.add_argument("--map", required=True, help="scenario name, used for the output dir")
    ap.add_argument("--out", default=None, help="output dir (default checkpoints/pv_behaviour/<map>)")
    ap.add_argument("--ensemble", type=int, default=5, help="number of members (1 = single model)")
    ap.add_argument("--max-updates", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("overrides", nargs="*", help="extra Hydra-style overrides")
    args = ap.parse_args()

    ensure_project_root()
    cfg = load_cfg(args.config, list(args.overrides))
    out_dir = args.out or os.path.join("checkpoints", "pv_behaviour", args.map)
    os.makedirs(out_dir, exist_ok=True)

    dataset = load_vault_dataset(cfg)
    print_dataset_stats(dataset, label="behaviour_vault")
    stats = dataset.stats
    train_idx, val_idx = split_episodes(dataset.n_episodes, args.val_frac, seed=args.seed)
    save_json(
        os.path.join(out_dir, "split.json"),
        {
            "map": args.map,
            "n_episodes": int(dataset.n_episodes),
            "val_frac": float(args.val_frac),
            "split_seed": int(args.seed),
            "train_episodes": train_idx,
            "val_episodes": val_idx,
        },
    )
    print(f"[bc] split: {len(train_idx)} train / {len(val_idx)} val episodes", flush=True)

    device = args.device
    members: list[str] = []
    fit_reports: list[dict] = []
    for member in range(max(1, args.ensemble)):
        seed = args.seed + 1000 * member
        model, report = fit_behaviour_model(
            dataset,
            train_indices=train_idx,
            val_indices=val_idx,
            n_agents=int(stats.n_agents),
            n_actions=int(stats.n_actions),
            obs_dim=int(stats.obs_dim),
            hidden_dim=args.hidden_dim,
            max_updates=args.max_updates,
            batch_size=args.batch_size,
            lr=args.lr,
            eval_every=args.eval_every,
            patience=args.patience,
            device=device,
            seed=seed,
        )
        # In ensemble mode member_0.pt is the primary model; model.pt is a stable
        # alias so downstream consumers (pv_probe, 2a, coverage) can point at one
        # path regardless of how many members were trained.
        name = "model.pt" if args.ensemble <= 1 else f"member_{member}.pt"
        path = os.path.join(out_dir, name)
        save_behaviour_model(
            path,
            model,
            meta={
                "map": args.map,
                "member": member,
                "seed": seed,
                "n_agents": int(stats.n_agents),
                "n_actions": int(stats.n_actions),
                "obs_dim": int(stats.obs_dim),
                "max_updates": args.max_updates,
                "batch_size": args.batch_size,
            },
        )
        members.append(path)
        if args.ensemble > 1 and member == 0:
            # Stable alias so pv_probe / 2a / coverage can point at model.pt
            # regardless of the member count.
            import shutil

            shutil.copyfile(path, os.path.join(out_dir, "model.pt"))
        fit_reports.append(
            {
                "member": member,
                "seed": seed,
                "updates": report.updates,
                "best_update": report.best_update,
                "best_val_nll": report.best_val_nll,
                "stopped_early": report.stopped_early,
                "val_nll_curve": report.val_nll,
            }
        )
        print(
            f"[bc] member {member}: updates={report.updates} best_val_nll="
            f"{report.best_val_nll:.4f} early_stop={report.stopped_early}",
            flush=True,
        )

    primary = load_behaviour_model(members[0], device=device)
    val_batch = gather_episodes(dataset, val_idx, device=primary.device)
    uniform_nll = primary.uniform_nll(val_batch)
    model_nll = primary.nll(val_batch)
    calib = calibration_bins(primary, val_batch)

    # Team deviation quantiles on the held-out split, fixed before online data.
    dev = primary.deviations(val_batch)
    team = dev["team"][dev["mask"]].detach().cpu().numpy()
    quantiles = {}
    if team.size:
        for q in (50, 90, 95, 99):
            quantiles[f"q{q}"] = float(np.quantile(team, q / 100.0))
        quantiles["mean"] = float(team.mean())
        quantiles["std"] = float(team.std())
        quantiles["n_states"] = int(team.size)

    novelty = {}
    if len(members) > 1:
        ensemble = NoveltyEnsemble([load_behaviour_model(p, device=device) for p in members])
        nov = ensemble.novelty(val_batch)
        nov_vals = nov[dev["mask"]].detach().cpu().numpy()
        if nov_vals.size:
            novelty = {
                "mean": float(nov_vals.mean()),
                "std": float(nov_vals.std()),
                "q95": float(np.quantile(nov_vals, 0.95)),
                "n_states": int(nov_vals.size),
            }
        save_json(os.path.join(out_dir, "ensemble.json"), {"members": members, "novelty": novelty})

    report = {
        "map": args.map,
        "config": args.config,
        "n_agents": int(stats.n_agents),
        "n_actions": int(stats.n_actions),
        "obs_dim": int(stats.obs_dim),
        "hidden_dim": args.hidden_dim,
        "max_updates": args.max_updates,
        "batch_size": args.batch_size,
        "members": members,
        "fits": fit_reports,
        "heldout": {
            "model_nll": model_nll,
            "uniform_over_legal_nll": uniform_nll,
            "nll_gain_vs_uniform": uniform_nll - model_nll,
            "calibration_bins": calib,
        },
        "team_deviation_quantiles": quantiles,
        "ensemble_novelty": novelty,
    }
    save_json(os.path.join(out_dir, "report.json"), report)
    save_json(os.path.join(out_dir, "thresholds.json"), quantiles)
    print(
        f"[bc] held-out NLL={model_nll:.4f} (uniform {uniform_nll:.4f}, "
        f"gain {uniform_nll - model_nll:.4f}); D quantiles="
        + ", ".join(f"{k}={v:.3f}" for k, v in quantiles.items() if k.startswith("q")),
        flush=True,
    )
    print(f"[bc] wrote {out_dir}/{{model.pt|member_*.pt,split.json,report.json,thresholds.json}}")


if __name__ == "__main__":
    main()
