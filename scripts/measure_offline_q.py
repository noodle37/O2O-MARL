#!/usr/bin/env python3
"""Measure offline Q_tot accuracy on vault dataset samples (SO2-style protocol).

Frozen checkpoint, no parameter updates, no environment interaction. For every
filled transition (s_t, u_t) of the offline dataset:

- bias: distribution of Q_tot(s_t, u_t) - G_t, where G_t is the discounted
  MC return-to-go computed from the recorded rewards of that episode
- global rank: pairwise concordance of (Q_tot, G) over random transition
  pairs, plus Kendall tau on a subsample

Outputs a single JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SC2PATH", "/home/jxy/StarCraftII")

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.factory import build_learner, load_vault_dataset
from utils.checkpoint import load_qmix_checkpoint
from utils.paths import ensure_project_root, resolve_path

Q_TIE = 1e-8
G_TIE = 1e-4


def _self_check() -> None:
    q = torch.tensor([1.0, 2.0, 3.0, 3.0])
    g = torch.tensor([1.0, 0.0, 5.0, 5.0 + 1e-9])
    counts = pairwise_outcomes(q, g, pairs=torch.tensor([[0, 1], [1, 2], [2, 3], [0, 0]]))
    # (0,1): dq>0 dg>0 -> agree; (1,2): dq>0 dg<0 -> invert;
    # (2,3): dg tie -> tie; (0,0): dq=0 -> tie
    if counts != {"agree": 1, "invert": 1, "tie": 2}:
        raise RuntimeError(f"pairwise self-check failed: {counts}")


def pairwise_outcomes(
    q: torch.Tensor, g: torch.Tensor, pairs: torch.Tensor
) -> dict[str, int]:
    dq = (q[pairs[:, 0]] - q[pairs[:, 1]]).double()
    dg = (g[pairs[:, 0]] - g[pairs[:, 1]]).double()
    tie = (dq.abs() <= Q_TIE) | (dg.abs() <= G_TIE)
    same_sign = ((dq > 0) & (dg > 0)) | ((dq < 0) & (dg < 0))
    agree = int((~tie & same_sign).sum().item())
    invert = int((~tie & ~same_sign).sum().item())
    return {"agree": agree, "invert": invert, "tie": int(tie.sum().item())}


def return_to_go(reward: torch.Tensor, mask: torch.Tensor, gamma: float) -> torch.Tensor:
    """G_t = r_t + gamma*G_{t+1} over filled steps; padded rewards are zeroed."""
    r = reward * mask
    b, t = r.shape
    out = torch.zeros_like(r)
    acc = torch.zeros(b, device=r.device, dtype=r.dtype)
    for i in reversed(range(t)):
        acc = r[:, i] + gamma * acc
        out[:, i] = acc
    return out


@torch.inference_mode()
def collect_q_and_g(learner, dataset, gamma: float, episodes_per_batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    q_parts: list[torch.Tensor] = []
    g_parts: list[torch.Tensor] = []
    n = dataset.n_episodes
    for start in range(0, n, episodes_per_batch):
        idx = torch.arange(start, min(start + episodes_per_batch, n), device=dataset.device)
        batch = {
            key: getattr(dataset, key).index_select(0, idx).to(learner.device)
            for key in ("obs", "actions", "state", "reward", "terminated", "filled")
        }
        mask = batch["filled"].squeeze(-1) > 0.5
        mac_out = learner._forward_qs(batch)
        chosen_q = torch.gather(mac_out, 3, batch["actions"].long())
        q_tot = learner.mixer(chosen_q, batch["state"]).view(chosen_q.shape[0], chosen_q.shape[1])
        g = return_to_go(batch["reward"].squeeze(-1), mask.float(), gamma)
        q_parts.append(q_tot[mask].float())
        g_parts.append(g[mask].float())
        print(f"episodes {start + len(idx)}/{n} transitions={sum(p.numel() for p in q_parts)}", flush=True)
    return torch.cat(q_parts), torch.cat(g_parts)


def main() -> None:
    _self_check()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="default: checkpoint_path in config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-pairs", type=int, default=1_000_000)
    parser.add_argument("--kendall-subsample", type=int, default=20_000)
    parser.add_argument("--episodes-per-batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0, help="rng seed for pair sampling")
    args = parser.parse_args()

    ensure_project_root()
    t0 = time.time()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    ckpt = resolve_path(args.checkpoint or cfg["checkpoint_path"])
    gamma = float(cfg["system"]["discount"])

    dataset = load_vault_dataset(cfg)
    learner = build_learner(cfg, dataset)
    loaded = load_qmix_checkpoint(ckpt, learner, load_optimizer=False)
    learner.q_network.eval()
    learner.mixer.eval()

    q, g = collect_q_and_g(learner, dataset, gamma, args.episodes_per_batch)
    n = int(q.numel())
    if n < 2:
        raise RuntimeError("not enough filled transitions to measure")
    residual = (q.double() - g.double()).cpu().numpy()
    q_np = q.double().cpu().numpy()
    g_np = g.double().cpu().numpy()

    denom = np.maximum(np.abs(g_np), 1.0)
    qm, gm = float(q_np.mean()), float(g_np.mean())
    var_q = float(np.mean((q_np - qm) ** 2))
    cov = float(np.mean((q_np - qm) * (g_np - gm)))
    slope = cov / var_q if var_q > 1e-12 else 1.0
    shift = gm - slope * qm

    rng = np.random.default_rng(args.seed)
    pairs = torch.from_numpy(
        rng.integers(0, n, size=(args.n_pairs, 2))
    ).to(q.device)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    counts = pairwise_outcomes(q, g, pairs)
    decided = counts["agree"] + counts["invert"]
    concordance = counts["agree"] / decided if decided else None

    m = int(min(args.kendall_subsample, n))
    sub = rng.choice(n, size=m, replace=False)
    tau = kendalltau(q_np[sub], g_np[sub]).statistic
    tau = float(tau) if np.isfinite(tau) else None

    extra = loaded.get("extra", {}) if isinstance(loaded, dict) else {}
    payload = {
        "config": args.config,
        "checkpoint": str(ckpt),
        "ckpt_t_env": extra.get("t_env"),
        "ckpt_seed": extra.get("seed"),
        "ckpt_best_eval_return": extra.get("best_eval_return"),
        "scenario": cfg["task"]["scenario"],
        "dataset": cfg["task"].get("dataset"),
        "gamma": gamma,
        "n_episodes": dataset.n_episodes,
        "n_transitions": n,
        "q_mean": qm,
        "q_std": float(q_np.std()),
        "g_mean": gm,
        "g_std": float(g_np.std()),
        "bias_mean": float(residual.mean()),
        "bias_std": float(residual.std()),
        "bias_normalized_mean": float(np.mean(residual / denom)),
        "overestimate_rate": float(np.mean(q_np > g_np)),
        "residual_quantiles": {
            str(p): float(np.quantile(residual, p / 100)) for p in (5, 25, 50, 75, 95)
        },
        "affine_calibration": {"slope": float(slope), "shift": float(shift)},
        "pairwise": {
            "n_pairs": int(pairs.shape[0]),
            **counts,
            "concordance": concordance,
            "q_tie": Q_TIE,
            "g_tie": G_TIE,
        },
        "kendall_tau": tau,
        "kendall_subsample": m,
        "elapsed_s": round(time.time() - t0, 1),
    }

    out = Path(resolve_path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
