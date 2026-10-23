#!/usr/bin/env python3
"""Measure Q-value bias and rank accuracy with the O2O paper protocol.

Frozen checkpoint, epsilon=0 rollouts, no parameter updates. For every decision
step of every episode:

- Q_estimated = Q_tot(s_t, u_t) of the executed joint action.
- Q_true = discounted return-to-go G_t of the same rollout, which is the paper's
  Monte Carlo target ("actual returns along an adequately extended trajectory").

Bias is Q_estimated - Q_true; the normalized forms divide by G_t (paper) and by
max(|G_t|, 1) (sign-stable, used when the policy return is negative). Rank
accuracy is Kendall's tau inside sliding windows of consecutive decisions
(window 10, stride 1 in the primary reading), averaged over windows and
episodes. Episodes are generated from fixed environment seeds so checkpoints
see the same initial states.

Output is one JSON file; optional per-episode Q/return traces are stored as
.npz next to it for plotting and audit.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.factory import build_learner, load_vault_dataset, make_env
from training.q_metrics import discounted_returns, summarize_paper_diagnostics
from training.rollout_helpers import legals_from_info
from utils.checkpoint import load_qmix_checkpoint
from utils.paths import ensure_project_root, resolve_path


def _done(terminals, truncations) -> bool:
    return all(bool(np.asarray(v)) for v in terminals.values()) or all(
        bool(np.asarray(v)) for v in truncations.values()
    )


def collect_episode(env, learner, max_steps: int | None = None) -> dict:
    learner.reset_eval_hidden()
    obs, info = env.reset()
    done = False
    q_steps: list[float] = []
    reward_steps: list[float] = []
    won = None
    while not done:
        trace: dict = {}
        agents = sorted(obs.keys())
        state = info.get("state")
        actions = learner.select_actions_eval(
            obs,
            legals_from_info(info, agents),
            0.0,
            trace=trace,
            env_state=state,
        )
        if trace.get("q_tot") is not None:
            q_steps.append(float(trace["q_tot"]))
        elif trace.get("chosen_q") and state is not None:
            q_steps.append(learner.mix_agent_qs(trace["chosen_q"], state))
        else:
            raise RuntimeError("evaluation step did not produce a joint Q")
        obs, rewards, terminals, truncations, info = env.step(actions)
        reward_steps.append(float(np.mean(list(rewards.values()))))
        if "battle_won" in info:
            won = bool(info["battle_won"])
        done = _done(terminals, truncations)
        if max_steps is not None and len(q_steps) >= max_steps:
            break
    return {
        "q": np.asarray(q_steps, dtype=np.float64),
        "reward": np.asarray(reward_steps[: len(q_steps)], dtype=np.float64),
        "win": won,
    }


def _close_env(env) -> None:
    inner = getattr(env, "_environment", None)
    close = getattr(inner, "close", None) if inner is not None else None
    if close is not None:
        try:
            close()
        except NotImplementedError:
            pass


def run(args) -> dict:
    ensure_project_root()
    t0 = time.time()
    cfg = OmegaConf.load(args.config)
    overrides = list(args.override or [])
    overrides += [f"system.data_device=cpu"]
    if args.device:
        overrides += [f"system.device={args.device}"]
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    cfg = OmegaConf.to_container(cfg, resolve=True)
    ckpt = resolve_path(args.checkpoint or cfg.get("checkpoint_path"))
    gamma = float(cfg["system"]["discount"])
    scenario = cfg["task"]["scenario"]

    dataset = load_vault_dataset(cfg)
    learner = build_learner(cfg, dataset)
    del dataset
    load_qmix_checkpoint(ckpt, learner, load_optimizer=False)
    learner.q_network.eval()
    learner.mixer.eval()

    print(
        f"[paper-q] scenario={scenario} label={args.label} ckpt={ckpt} "
        f"episodes={args.episodes} window={args.window} gamma={gamma}",
        flush=True,
    )
    episodes: list[dict] = []
    for ep in range(args.episodes):
        env = make_env(cfg, seed=args.seed + ep)
        try:
            episode = collect_episode(env, learner, max_steps=args.max_steps)
        finally:
            _close_env(env)
        episode["seed"] = args.seed + ep
        episodes.append(episode)
        print(
            f"[paper-q] episode {ep + 1}/{args.episodes} seed={args.seed + ep} "
            f"steps={episode['q'].size} return={float(episode['reward'].sum()):.3f} "
            f"win={episode['win']}",
            flush=True,
        )

    results = {}
    for window in sorted({5, args.window, 20}):
        summary = summarize_paper_diagnostics(
            episodes, gamma, window=int(window), stride=args.stride
        )
        results[str(window)] = summary

    payload = {
        "label": args.label,
        "config": args.config,
        "checkpoint": str(ckpt),
        "scenario": scenario,
        "gamma": gamma,
        "n_episodes": args.episodes,
        "seed": args.seed,
        "window": args.window,
        "stride": args.stride,
        "primary_window": str(args.window),
        "results": results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if args.save_traces:
        trace_path = Path(resolve_path(args.out)).with_suffix(".npz")
        arrays = {}
        for i, ep in enumerate(episodes):
            arrays[f"q_{i}"] = ep["q"]
            arrays[f"reward_{i}"] = ep["reward"]
            arrays[f"returns_{i}"] = discounted_returns(ep["reward"], gamma)
        np.savez_compressed(trace_path, **arrays)
        payload["traces"] = str(trace_path)

    out = Path(resolve_path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    primary = results[str(args.window)]
    print(
        f"[paper-q] {args.label} mean_return={primary.get('mean_episode_return')} "
        f"bias={primary.get('bias_mean')} over={primary.get('overestimate_rate')} "
        f"tau_full={primary.get('kendall_tau_full_episode')} "
        f"tau_w{args.window}={primary.get('window_tau_mean')}",
        flush=True,
    )
    print(f"wrote {out}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--label", default="q")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12_345)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="OmegaConf dotlist override, e.g. task.dataset=Medium-Replay",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
