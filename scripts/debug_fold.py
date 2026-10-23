#!/usr/bin/env python3
# Debug: is the affine fold into the QMIX mixer exact?

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from scripts.make_scale_corrected_ckpt import (
    episode_q_tot,
    episode_return,
    fold_affine_into_mixer,
    load_cfg,
)
from training.factory import build_learner, load_vault_dataset, make_env
from training.online import _load_learner_weights
from training.rollout import collect_episode
from utils.paths import ensure_project_root


def main() -> None:
    ensure_project_root()
    cfg = load_cfg("configs/pv_base_5m_vs_6m.yaml", ["seed=1", "phase=online"])
    dataset = load_vault_dataset(cfg)
    learner = build_learner(cfg, dataset)
    ckpt = "checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt"
    payload = torch.load(ckpt, map_location=learner.device, weights_only=False)
    _load_learner_weights(learner, payload)
    if hasattr(learner, "freeze_oracle"):
        learner.freeze_oracle()
    if hasattr(learner, "load_offline_for_ovm"):
        learner.load_offline_for_ovm(ckpt)

    env = make_env(cfg, seed=1001)
    episodes = [collect_episode(env, learner, 0.0) for _ in range(8)]
    Q = np.concatenate([episode_q_tot(learner, ep) for ep in episodes])
    G = np.concatenate([episode_return(ep["steps"], float(cfg["system"]["discount"])) for ep in episodes])
    print(f"n={Q.size} Q mean={Q.mean():.3f} std={Q.std():.3f} min={Q.min():.3f} max={Q.max():.3f}")
    print(f"G mean={G.mean():.3f} std={G.std():.3f}")

    # What the script would fit.
    q_mean, g_mean = float(Q.mean()), float(G.mean())
    slope = float(((Q - q_mean) * (G - g_mean)).sum() / ((Q - q_mean) ** 2).sum())
    a = 1.0 if slope <= 0 else slope
    b = float(g_mean - a * q_mean)
    print(f"slope={slope:.4f} a={a:.4f} b={b:.4f}")

    # Intermediate magnitudes inside the mixer, to see whether cancellation is at play.
    with torch.no_grad():
        ep = episodes[0]
        st = torch.from_numpy(np.asarray(ep["steps"][0]["state"])).float().to(learner.device).view(1, 1, -1)
        n = learner.n_agents
        qs = torch.zeros(1, 1, n, device=learner.device)
        w1 = torch.abs(learner.mixer.hyper_w_1(st))
        w_final = torch.abs(learner.mixer.hyper_w_final(st))
        v = learner.mixer.V(st)
        b1 = learner.mixer.hyper_b_1(st)
        hidden = torch.nn.functional.elu(torch.matmul(qs, w1.reshape(-1, n, 32)) + b1)
        term_w = float((torch.matmul(hidden, w_final.reshape(-1, 32, 1))).abs().max())
        print(f"|state|max={float(st.abs().max()):.2f} |w_final|max={float(w_final.max()):.4f} "
              f"|V(s)|max={float(v.abs().max()):.2f} |hidden@w_final|max={term_w:.2f}")

    for test_a, test_b in ((a, b), (1.5, 2.0), (2.0, -3.0)):
        fold_affine_into_mixer(learner.mixer, test_a, test_b)
        fold_affine_into_mixer(learner.target_mixer, test_a, test_b)
        q_new = np.concatenate([episode_q_tot(learner, ep) for ep in episodes])
        expect = test_a * Q + test_b
        err = float(np.max(np.abs(q_new - expect)))
        rel = err / max(float(np.abs(expect).max()), 1e-9)
        print(f"a={test_a:.3f} b={test_b:.3f} -> max_abs_err={err:.3e} rel={rel:.3e} "
              f"|Q|max={float(np.abs(expect).max()):.2f}")


if __name__ == "__main__":
    main()
