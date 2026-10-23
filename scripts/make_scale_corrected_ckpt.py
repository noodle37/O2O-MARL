#!/usr/bin/env python3
"""Arm D (scale correction) starting checkpoint for problem-validation exp 1.

Pipeline (design section 3.1, arm D):
  1. Load the shared offline checkpoint into the paper learner (the same path
     the online trainer uses), so D starts from exactly what arm A starts from.
  2. Roll out 64 episodes with the offline greedy policy (eps = 0). These
     environment steps count toward arm D's budget and are recorded in the meta.
  3. Fit an affine map Q~ = a*Q_tot + b (a > 0) from the offline Q_tot of the
     executed joint actions to the Monte-Carlo return-to-go G.
  4. Fold (a, b) into the QMIX mixer: the last layer of hyper_w_final is scaled
     by a and the V stream becomes a*V(s) + b, on both the online and the target
     mixer. Per-agent Q values are untouched, and because a > 0 keeps the mixer
     monotonic, the greedy joint policy at t = 0 is identical to arm A's. Only
     the scale agreement between the TD target and the current estimate changes.
  5. Verify the fold numerically on the collected episodes (Q'_tot == a*Q_tot+b
     for both the online and the target mixer; q_network weights bit-identical),
     then save as arm D's starting checkpoint.

Usage:
  python scripts/make_scale_corrected_ckpt.py \
    --config configs/pv_base_5m_vs_6m.yaml --seed 1 \
    --out checkpoints/pv_exp1_D_ckpt/D_seed1_scaled.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.factory import build_learner, load_vault_dataset, make_env
from training.online import _load_learner_weights
from training.rollout import collect_episode
from utils.checkpoint import save_qmix_checkpoint
from utils.paths import ensure_project_root, resolve_path


def load_cfg(path: str, overrides: list[str]) -> dict:
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


def fold_affine_into_mixer(mixer: torch.nn.Module, a: float, b: float) -> None:
    """Q_tot' = a*Q_tot + b by scaling hyper_w_final's last layer and the V stream.

    The mixer forward computes w_final = |hyper_w_final(s)| and v = V(s), then
    y = hidden . w_final + v. With a > 0, |a*x| = a*|x|, so scaling the last
    linear weights of hyper_w_final by a scales w_final by a, and turning V's
    last layer into a*V(s) + b gives y' = a*y + b for every input.

    Both the weight AND the bias of hyper_w_final's last layer must be scaled:
    that layer computes w@relu(...) + bias, so scaling only the weight leaves
    w_final' = a*w_final - (a-1)*bias, which is not a*times w_final and breaks
    the identity (measured: ~0.3 absolute error on Q_tot of order 10).
    """
    w_last = mixer.hyper_w_final[-1]
    v_last = mixer.V[-1]
    if not isinstance(w_last, torch.nn.Linear) or not isinstance(v_last, torch.nn.Linear):
        raise TypeError("unexpected mixer structure; expected Linear final layers")
    with torch.no_grad():
        w_last.weight.data.mul_(a)
        w_last.bias.data.mul_(a)
        v_last.weight.data.mul_(a)
        v_last.bias.data.mul_(a).add_(b)


def episode_q_tot(learner, ep: dict, *, target: bool = False) -> np.ndarray:
    """Q_tot of the executed joint action at every step of one episode.

    Mirrors qmix/policy.py select_actions_eval: per-agent forward with the GRU
    hidden state carried across steps, executed action scored, then mixed with
    the mixer. Avoids the training batch layout entirely.
    """
    steps = ep["steps"]
    T = len(steps)
    n = learner.n_agents
    device = learner.device
    net = learner.target_q_network if target else learner.q_network
    mixer = learner.target_mixer if target else learner.mixer
    with torch.no_grad():
        hidden = [net.initial_state(1, device) for _ in range(n)]
        last_actions: list = [None] * n
        qtot = np.zeros(T, dtype=np.float64)
        for t, s in enumerate(steps):
            obs_arr = np.asarray(s["obs"])  # [n, obs_dim], sorted agent order
            agent_qs = []
            for i in range(n):
                obs = torch.from_numpy(obs_arr[i]).float().to(device)
                last_idx = last_actions[i]
                if last_idx is None:
                    last_oh = torch.zeros(learner.n_actions, device=device)
                else:
                    last_oh = F.one_hot(
                        torch.tensor(int(last_idx), device=device), learner.n_actions
                    ).float()
                pieces = [obs, last_oh]
                if learner.add_agent_id_to_obs:
                    pieces.append(F.one_hot(torch.tensor(i, device=device), n).float())
                obs_in = torch.cat(pieces, dim=-1).unsqueeze(0)
                q, hidden[i] = net(obs_in, hidden[i])
                a_exec = int(np.asarray(s["actions"]).reshape(-1)[i])
                agent_qs.append(float(q[0, a_exec].item()))
                last_actions[i] = a_exec
            qs = torch.tensor(agent_qs, dtype=torch.float32, device=device).view(1, 1, n)
            st = torch.from_numpy(np.asarray(s["state"])).float().to(device).view(1, 1, -1)
            qtot[t] = float(mixer(qs, st).reshape(-1)[0].item())
    return qtot


def episode_return(steps: list, gamma: float) -> np.ndarray:
    """Monte-Carlo discounted return-to-go per step (no bootstrap at timeout)."""
    T = len(steps)
    rewards = np.array([s["reward"] for s in steps], dtype=np.float64)
    ret = np.zeros(T, dtype=np.float64)
    g = 0.0
    for t in range(T - 1, -1, -1):
        g = rewards[t] + gamma * g
        ret[t] = g
    return ret


def _state_dict_fingerprint(module: torch.nn.Module) -> list:
    return [(k, v.detach().clone()) for k, v in module.state_dict().items()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/pv_base_5m_vs_6m.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="output checkpoint path (*.pt)")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument(
        "--collect-seed",
        type=int,
        default=None,
        help="env seed for the collection episodes (default: seed + 1000)",
    )
    ap.add_argument("--offline-ckpt", default=None, help="override checkpoint_path")
    args = ap.parse_args()

    ensure_project_root()
    collect_seed = args.collect_seed if args.collect_seed is not None else args.seed + 1000
    cfg = load_cfg(args.config, [f"seed={args.seed}", "phase=online"])
    gamma = float(cfg["system"]["discount"])

    dataset = load_vault_dataset(cfg)
    learner = build_learner(cfg, dataset)
    ckpt = resolve_path(args.offline_ckpt or cfg["checkpoint_path"])
    if not os.path.isfile(ckpt):
        raise SystemExit(f"FATAL: offline checkpoint not found: {ckpt}")
    payload = torch.load(ckpt, map_location=learner.device, weights_only=False)
    _load_learner_weights(learner, payload)
    if hasattr(learner, "freeze_oracle"):
        learner.freeze_oracle()
    if hasattr(learner, "load_offline_for_ovm"):
        learner.load_offline_for_ovm(ckpt)

    qnet_before = _state_dict_fingerprint(learner.q_network)
    target_qnet_before = _state_dict_fingerprint(learner.target_q_network)

    env = make_env(cfg, seed=collect_seed)
    episodes = []
    env_steps = 0
    for ep_idx in range(args.episodes):
        ep = collect_episode(env, learner, 0.0)  # eps = 0 -> offline greedy
        env_steps += int(ep["length"])
        episodes.append(ep)
        if (ep_idx + 1) % 16 == 0:
            print(f"collected {ep_idx + 1}/{args.episodes} episodes ({env_steps} env steps)")

    Q = np.concatenate([episode_q_tot(learner, ep) for ep in episodes])
    Q_target = np.concatenate([episode_q_tot(learner, ep, target=True) for ep in episodes])
    G = np.concatenate([episode_return(ep["steps"], gamma) for ep in episodes])
    if Q.size < 10:
        raise SystemExit(f"FATAL: too few transitions collected ({Q.size})")

    q_mean, g_mean = float(Q.mean()), float(G.mean())
    denom = float(((Q - q_mean) ** 2).sum())
    slope = float(((Q - q_mean) * (G - g_mean)).sum() / denom) if denom > 0 else 0.0
    clamped = slope <= 0.0
    a = 1.0 if clamped else slope
    b = float(g_mean - a * q_mean)
    ss_tot = float(((G - g_mean) ** 2).sum())
    r2 = 1.0 - float(((G - (a * Q + b)) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    fold_affine_into_mixer(learner.mixer, a, b)
    fold_affine_into_mixer(learner.target_mixer, a, b)

    # Verify on the collected episodes: the folded mixer must reproduce
    # a*Q_tot + b exactly (input-independent affine map), and the per-agent
    # networks must be bit-identical, so the greedy policy is unchanged.
    q_new = np.concatenate([episode_q_tot(learner, ep) for ep in episodes])
    err_online = float(np.max(np.abs(q_new - (a * Q + b)))) if Q.size else 0.0
    q_target = np.concatenate([episode_q_tot(learner, ep, target=True) for ep in episodes])
    err_target = float(np.max(np.abs(q_target - (a * Q_target + b)))) if Q_target.size else 0.0
    qnet_same = all(
        torch.equal(old, new)
        for (k, old), (_, new) in zip(qnet_before, _state_dict_fingerprint(learner.q_network))
    )
    target_qnet_same = all(
        torch.equal(old, new)
        for (k, old), (_, new) in zip(
            target_qnet_before, _state_dict_fingerprint(learner.target_q_network)
        )
    )
    tol = 1e-3
    if err_online > tol or err_target > tol or not qnet_same or not target_qnet_same:
        raise SystemExit(
            "FATAL: arm D fold verification failed: "
            f"err_online={err_online:.3e} err_target={err_target:.3e} "
            f"qnet_same={qnet_same} target_qnet_same={target_qnet_same}"
        )

    meta = {
        "seed": int(args.seed),
        "collect_seed": int(collect_seed),
        "offline_ckpt": str(ckpt),
        "gamma": gamma,
        "n_episodes": int(args.episodes),
        "n_transitions": int(Q.size),
        "env_steps": int(env_steps),
        "slope": slope,
        "a": a,
        "b": b,
        "a_clamped_to_1": bool(clamped),
        "r2": r2,
        "q_mean": q_mean,
        "g_mean": g_mean,
        "max_abs_err_online": err_online,
        "max_abs_err_target": err_target,
        "qnet_unchanged": bool(qnet_same and target_qnet_same),
        "created": datetime.now().isoformat(timespec="seconds"),
    }

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    save_qmix_checkpoint(
        out_dir,
        learner,
        include_optimizer=True,
        extra={
            "seed": int(args.seed),
            "t_env": 0,
            "phase": "online",
            "transfer_mode": "scale_corrected",
            "pv_arm": "D",
            "pv_affine": meta,
        },
        default_name=os.path.basename(args.out),
    )
    meta_path = os.path.splitext(args.out)[0] + "_affine_meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print(
        f"arm D checkpoint written: {args.out}\n"
        f"  fit: Q~ = {a:.6f} * Q_tot + {b:.6f}  (unconstrained slope {slope:.6f}"
        f"{', CLAMPED to a=1' if clamped else ''}; R^2 = {r2:.4f})\n"
        f"  data: {args.episodes} episodes, {Q.size} transitions, {env_steps} env steps\n"
        f"  verify: max|err| online={err_online:.2e} target={err_target:.2e}, "
        f"q_network unchanged={qnet_same and target_qnet_same}\n"
        f"  meta: {meta_path}"
    )


if __name__ == "__main__":
    main()
