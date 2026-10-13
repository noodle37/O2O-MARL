#!/usr/bin/env python3
"""Evaluate a checkpoint with N episodes in chunks; report win rate with Wilson CI."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.evaluation import evaluate  # noqa: E402
from training.factory import build_learner, load_vault_dataset, make_env  # noqa: E402
from utils.checkpoint import load_qmix_checkpoint  # noqa: E402
from utils.paths import ensure_project_root  # noqa: E402


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="Hydra-style overrides")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=640)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="eval")
    args = parser.parse_args()

    ensure_project_root()
    cfg = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides)),
        resolve=True,
    )

    dataset = load_vault_dataset(cfg)
    learner = build_learner(cfg, dataset)
    payload = load_qmix_checkpoint(args.checkpoint, learner, map_location=learner.device)
    if hasattr(learner, "load_state"):
        learner.flow.eval()
        learner.policy.eval()
        learner.critic.eval()
    else:
        learner.q_network.eval()
        learner.mixer.eval()
    meta = payload.get("extra", {})
    print(f"[eval640] tag={args.tag} ckpt={args.checkpoint} meta={meta}", flush=True)

    env = make_env(cfg, seed=int(cfg["seed"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_chunks = (args.episodes + args.chunk - 1) // args.chunk
    wins_total = 0.0
    eps_total = 0
    ret_means = []
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as out:
        for i in range(n_chunks):
            n = min(args.chunk, args.episodes - eps_total)
            logs = evaluate(env, learner, n, eps_total, cfg)
            wins_total += logs["evaluation/win_rate"] * n
            eps_total += n
            ret_means.append(logs["evaluation/mean_episode_return"])
            row = {
                "tag": args.tag,
                "chunk": i,
                "episodes_done": eps_total,
                "chunk_win_rate": logs["evaluation/win_rate"],
                "cum_win_rate": wins_total / eps_total,
                "chunk_mean_return": logs["evaluation/mean_episode_return"],
                "chunk_q_bias": logs.get("evaluation/q_bias"),
                "ckpt_t_env": meta.get("t_env"),
                "elapsed_s": round(time.time() - t0, 1),
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(
                f"[eval640] {args.tag} chunk {i + 1}/{n_chunks} cum_win={row['cum_win_rate']:.4f} ({eps_total} eps)",
                flush=True,
            )

        p = wins_total / eps_total
        lo, hi = wilson_ci(p, eps_total)
        summary = {
            "tag": args.tag,
            "checkpoint": args.checkpoint,
            "ckpt_t_env": meta.get("t_env"),
            "episodes": eps_total,
            "win_rate": p,
            "win_rate_ci95": [lo, hi],
            "mean_return": sum(ret_means) / len(ret_means),
        }
        out.write(json.dumps(summary) + "\n")
        out.flush()
        print(f"[eval640] SUMMARY {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
