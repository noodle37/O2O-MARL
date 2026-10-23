#!/usr/bin/env python3
"""Gate check: real teammate combinations should score higher than shuffled ones."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from fsr.learner import FSRLearner
from training.factory import build_learner, load_vault_dataset
from train import _load_cfg
from utils.checkpoint import load_qmix_checkpoint
from utils.paths import ensure_project_root


def _shuffle_teammates(actions: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    out = actions.clone()
    b, t, n, _ = out.shape
    for i in range(b):
        for s in range(t):
            perm = rng.permutation(n)
            out[i, s] = out[i, s, perm]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    ensure_project_root()
    cfg = _load_cfg(args.config, args.overrides)
    dataset = load_vault_dataset(cfg)
    learner = build_learner(cfg, dataset)
    assert isinstance(learner, FSRLearner)
    load_qmix_checkpoint(args.ckpt, learner, load_optimizer=False)
    learner.freeze_flow()
    rng = np.random.default_rng(0)
    real, fake = [], []
    for _ in range(args.batches):
        batch = dataset.sample(min(16, dataset.n_episodes), rng, device=learner.device)
        hidden = learner._history_hidden(batch)
        filled = batch["filled"][:, :-1, 0]
        logp, scored = learner._agent_logp(hidden, batch["actions"], filled)
        real.extend(logp[scored].detach().cpu().tolist())
        shuffled = dict(batch)
        shuffled["actions"] = _shuffle_teammates(batch["actions"], rng)
        logp_f, scored_f = learner._agent_logp(hidden, shuffled["actions"], filled)
        fake.extend(logp_f[scored_f].detach().cpu().tolist())
    real_t = torch.tensor(real)
    fake_t = torch.tensor(fake)
    gap = float(real_t.mean() - fake_t.mean())
    print(
        f"FSR bucket gate real_mean={float(real_t.mean()):.4f} "
        f"shuffled_mean={float(fake_t.mean()):.4f} gap={gap:.4f} "
        f"n={len(real)} threshold={learner.support_threshold:.4f}"
    )
    if gap <= 0:
        raise SystemExit("FSR gate failed: shuffled teammate actions are not lower-density")


if __name__ == "__main__":
    main()
