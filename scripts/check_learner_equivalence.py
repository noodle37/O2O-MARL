#!/usr/bin/env python3
"""One-step equivalence check: UOT eta=0 vs PaperQMIX with OVM off.

Both learners are loaded from the same checkpoint (weights, targets, optimizer
state, training_step_ctr) and updated on the same episode batch. The TD-only
path must match to float tolerance; otherwise arm B is not a clean vanilla O2O
control for arm A.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

os.environ.setdefault("SC2PATH", "/home/jxy/StarCraftII")

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.factory import build_learner, load_vault_dataset  # noqa: E402
from utils.checkpoint import load_qmix_checkpoint  # noqa: E402
from utils.paths import ensure_project_root  # noqa: E402


def max_param_diff(a, b, names) -> tuple[float, str]:
    worst = 0.0
    worst_name = ""
    for name in names:
        pa = dict(a.named_parameters())[name].detach()
        pb = dict(b.named_parameters())[name].detach()
        d = float((pa - pb).abs().max().item())
        if d > worst:
            worst, worst_name = d, name
    return worst, worst_name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/matched_protocol_5m_vs_6m.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--batch-seed", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    args = ap.parse_args()

    ensure_project_root()
    torch.manual_seed(0)
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    dataset = load_vault_dataset(cfg)

    cfg_uot = copy.deepcopy(cfg)
    cfg_uot["learner_type"] = "uot"
    cfg_uot["system"]["uot_eta"] = 0.0
    cfg_paper = copy.deepcopy(cfg)
    cfg_paper["learner_type"] = "ovmse_paper"
    cfg_paper["system"]["use_ovm"] = False

    uot = build_learner(cfg_uot, dataset)
    paper = build_learner(cfg_paper, dataset)

    load_qmix_checkpoint(args.checkpoint, uot, load_optimizer=True, map_location=uot.device)
    load_qmix_checkpoint(args.checkpoint, paper, load_optimizer=True, map_location=paper.device)
    assert uot.training_step_ctr == paper.training_step_ctr

    rng = np.random.default_rng(args.batch_seed)
    batch = dataset.sample(args.batch_size, rng, device=uot.device)

    logs_uot = uot.train_on_batch(batch, cql_weight=0.0, use_ovm=False, t_env=0, episode_delta=8)
    logs_paper = paper.train_on_batch(
        batch, cql_weight=0.0, use_ovm=False, t_env=0, episode_delta=8
    )

    q_diff, q_name = max_param_diff(uot.q_network, paper.q_network, [n for n, _ in uot.q_network.named_parameters()])
    m_diff, m_name = max_param_diff(uot.mixer, paper.mixer, [n for n, _ in uot.mixer.named_parameters()])
    print("uot  ", {k: logs_uot[k] for k in ("loss", "loss_td", "grad_norm")})
    print("paper", {k: logs_paper[k] for k in ("loss", "grad_norm") if k in logs_paper})
    print(f"param max diff: q={q_diff:.3e} ({q_name}), mixer={m_diff:.3e} ({m_name})")
    loss_diff = abs(float(logs_uot["loss_td"]) - float(logs_paper["loss"]))
    grad_diff = abs(float(logs_uot["grad_norm"]) - float(logs_paper["grad_norm"]))
    print(f"TD loss diff={loss_diff:.3e} grad_norm diff={grad_diff:.3e}")

    ok = max(q_diff, m_diff, loss_diff, grad_diff) <= args.tolerance
    print("EQUIVALENCE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
