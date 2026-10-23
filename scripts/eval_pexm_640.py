#!/usr/bin/env python3
"""640-episode greedy evaluation of a PEX-M run.

Modes:
  online   : trainable arm only
  offline  : frozen teacher only
  thompson : per-episode arm sampled from the saved posterior (no forced floor)
  mean     : per-episode arm = larger posterior mean (deterministic)

Reports win rate with a Wilson 95% interval, matching scripts/eval_640.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.factory import build_learner, load_vault_dataset, make_env  # noqa: E402
from training.pexm import ThompsonReturnSelector  # noqa: E402
from training.rollout_helpers import legals_from_info  # noqa: E402
from utils.checkpoint import load_qmix_checkpoint  # noqa: E402
from utils.paths import ensure_project_root, resolve_path  # noqa: E402

MODES = ("online", "offline", "thompson", "mean")


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def episode_outcome(env, learner) -> tuple[float, float]:
    learner.reset_eval_hidden()
    obs, info = env.reset()
    ep_ret = 0.0
    won = 0.0
    done = False
    while not done:
        agents = sorted(obs.keys())
        actions = learner.select_actions_eval(
            obs, legals_from_info(info, agents), 0.0
        )
        obs, rewards, terminal, trunc, info = env.step(actions)
        ep_ret += float(np.mean(list(rewards.values())))
        if "battle_won" in info:
            won = 1.0 if bool(info["battle_won"]) else 0.0
        done = all(terminal.values()) or all(trunc.values())
    return ep_ret, won


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="Hydra-style overrides")
    parser.add_argument("--online-checkpoint", required=True)
    parser.add_argument("--offline-checkpoint", default=None)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--episodes", type=int, default=640)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--modes", default="online,offline,thompson,mean")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ensure_project_root()
    cfg = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides)),
        resolve=True,
    )
    if args.device:
        cfg["system"]["device"] = args.device
    cfg["system"]["data_device"] = "cpu"
    cfg["system"]["moderate_lambda"] = False
    cfg["system"]["use_ovm"] = False
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = set(modes) - set(MODES)
    if unknown:
        raise ValueError(f"unknown modes: {sorted(unknown)}")
    if any(m in ("thompson", "mean") for m in modes) and not os.path.isfile(args.selector):
        raise FileNotFoundError(f"selector posterior missing: {args.selector}")

    dataset = load_vault_dataset(cfg)
    online = build_learner(cfg, dataset)
    offline = build_learner(cfg, dataset)
    del dataset
    load_qmix_checkpoint(
        args.online_checkpoint, online, load_optimizer=False, map_location=online.device
    )
    offline_path = resolve_path(
        args.offline_checkpoint
        or (cfg.get("pexm") or {}).get("offline_checkpoint")
        or cfg.get("checkpoint_path")
    )
    load_qmix_checkpoint(offline_path, offline, load_optimizer=False,
                         map_location=offline.device)
    online.q_network.eval()
    online.mixer.eval()
    offline.q_network.eval()
    offline.mixer.eval()

    selector = None
    if any(m in ("thompson", "mean") for m in modes):
        payload = json.loads(open(args.selector, encoding="utf-8").read())
        selector = ThompsonReturnSelector.from_dict(payload)
        selector.min_online = 0.0  # deployment samples the trained posterior
        print(
            f"selector offline={selector.params('offline')} "
            f"online={selector.params('online')}",
            flush=True,
        )

    env = make_env(cfg, seed=int(cfg["seed"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rng = np.random.default_rng(int(cfg["seed"]) + 99)
    n_chunks = (args.episodes + args.chunk - 1) // args.chunk
    with open(args.out, "w", encoding="utf-8") as out:
        for mode in modes:
            wins_total = 0.0
            ret_total = 0.0
            online_episodes = 0
            eps_total = 0
            t0 = time.time()
            for i in range(n_chunks):
                n = min(args.chunk, args.episodes - eps_total)
                chunk_wins = 0.0
                chunk_ret = 0.0
                chunk_online = 0
                for _ in range(n):
                    if mode == "online":
                        arm = "online"
                    elif mode == "offline":
                        arm = "offline"
                    elif mode == "thompson":
                        arm, _ = selector.choose(rng)
                    else:
                        p_off = selector.params("offline")["mu"]
                        p_on = selector.params("online")["mu"]
                        arm = "online" if p_on >= p_off else "offline"
                    learner = online if arm == "online" else offline
                    ret, won = episode_outcome(env, learner)
                    chunk_wins += won
                    chunk_ret += ret
                    chunk_online += int(arm == "online")
                wins_total += chunk_wins
                ret_total += chunk_ret
                online_episodes += chunk_online
                eps_total += n
                row = {
                    "mode": mode,
                    "chunk": i,
                    "episodes_done": eps_total,
                    "chunk_win_rate": chunk_wins / n,
                    "cum_win_rate": wins_total / eps_total,
                    "chunk_mean_return": chunk_ret / n,
                    "cum_mean_return": ret_total / eps_total,
                    "chunk_online_share": chunk_online / n,
                    "cum_online_share": online_episodes / eps_total,
                    "elapsed_s": round(time.time() - t0, 1),
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(
                    f"[pexm_eval] {mode} chunk {i + 1}/{n_chunks} "
                    f"cum_win={row['cum_win_rate']:.4f} ({eps_total} eps)",
                    flush=True,
                )

            p = wins_total / eps_total
            lo, hi = wilson_ci(p, eps_total)
            summary = {
                "mode": mode,
                "online_checkpoint": args.online_checkpoint,
                "offline_checkpoint": offline_path,
                "selector": args.selector if mode in ("thompson", "mean") else None,
                "episodes": eps_total,
                "win_rate": p,
                "win_rate_ci95": [lo, hi],
                "mean_return": ret_total / eps_total,
                "online_share": online_episodes / eps_total,
            }
            out.write(json.dumps(summary) + "\n")
            out.flush()
            print(f"[pexm_eval] SUMMARY {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
