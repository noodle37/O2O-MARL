# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Online fine-tuning with optional OVM and offline/online batch mixing."""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from data.vault_dataset import print_dataset_stats
from training.evaluation import evaluate
from training.factory import build_learner, load_vault_dataset, make_env, make_logger
from training.rollout import collect_parallel, merge_offline_online, stack_episodes
from utils.checkpoint import save_qmix_checkpoint
from utils.paths import resolve_path


def _resolve_offline_checkpoint(cfg: Dict[str, Any], offline_ckpt: Optional[str] = None) -> str:
    candidates: List[Optional[str]] = [resolve_path(offline_ckpt or cfg.get("checkpoint_path"))]
    ckpt_dir = resolve_path(cfg.get("checkpoint_dir"))
    seed = cfg["seed"]
    if ckpt_dir:
        candidates.append(os.path.join(ckpt_dir, f"ovmse_paper_seed{seed}_best.pt"))
        candidates.append(os.path.join(ckpt_dir, f"ovmse_paper_seed{seed}.pt"))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Online requires offline checkpoint, tried: {candidates}")


def run_online(cfg: Dict[str, Any], offline_ckpt: Optional[str] = None) -> None:
    offline_data = load_vault_dataset(cfg)
    print_dataset_stats(offline_data, label="online_offline_vault")
    learner = build_learner(cfg, offline_data)
    ckpt = _resolve_offline_checkpoint(cfg, offline_ckpt)
    payload = torch.load(ckpt, map_location=learner.device, weights_only=False)
    learner.q_network.load_state_dict(payload["q_network"])
    learner.mixer.load_state_dict(payload["mixer"])
    learner.target_q_network.load_state_dict(payload.get("target_q_network", payload["q_network"]))
    learner.target_mixer.load_state_dict(payload.get("target_mixer", payload["mixer"]))
    if "optimizer" in payload:
        learner.optimizer.load_state_dict(payload["optimizer"])
    use_ovm = bool(cfg["system"].get("use_ovm", True))
    if use_ovm:
        learner.load_offline_for_ovm(ckpt)
    learner.use_ovm = use_ovm
    print(f"online use_ovm={use_ovm} loaded={ckpt}")

    batch_size_run = int(cfg.get("batch_size_run", 1))
    envs = [
        make_env(cfg, seed=int(cfg["seed"]) + i)
        for i in range(batch_size_run)
    ]
    logger = make_logger(cfg)

    t_max = int(cfg["t_max"])
    test_interval = int(cfg.get("test_interval", 10_000))
    num_eval = int(cfg.get("num_eval_episodes", 32))
    eps_start = float(cfg["system"].get("eps_start", 1.0))
    eps_min = float(cfg["system"].get("eps_min", 0.05))
    eps_decay = int(cfg["system"].get("eps_decay_steps", 100_000))
    mixing_ratio = float(cfg["system"].get("mixing_ratio", 0.0))
    online_cql = float(cfg["system"].get("online_cql_weight", 0.0))
    total_batch = int(cfg.get("batch_size", 128))
    buffer_size = int(cfg.get("buffer_size", 5000))
    rng = np.random.default_rng(int(cfg["seed"]) + 1)

    online_buffer: deque = deque(maxlen=buffer_size)
    t_env = 0
    learner.t_env = 0
    learner.batch_size_run = batch_size_run
    last_test_t = -test_interval

    def eps_at(t: int) -> float:
        if t >= eps_decay:
            return eps_min
        return eps_start + (eps_min - eps_start) * (t / eps_decay)

    while t_env < t_max:
        if test_interval > 0 and t_env - last_test_t >= test_interval:
            last_test_t = t_env
            eval_logs = evaluate(envs[0], learner, num_eval, t_env, cfg)
            logger.write(eval_logs | {"evaluation/training_steps": t_env, "t_env": t_env}, force=True)

        for ep in collect_parallel(envs, learner, eps_at(t_env)):
            online_buffer.append(ep)
            t_env += ep["length"]
            learner.t_env = t_env

        if len(online_buffer) >= total_batch:
            if mixing_ratio > 0:
                offline_bs = int(round(total_batch * mixing_ratio))
                online_bs = total_batch - offline_bs
                if online_bs < 1:
                    online_bs = 1
                    offline_bs = total_batch - 1
            else:
                offline_bs = 0
                online_bs = total_batch
            picked = rng.choice(len(online_buffer), size=online_bs, replace=False)
            online_batch = stack_episodes(
                [online_buffer[int(i)] for i in picked],
                offline_data.stats.max_seq_length,
                learner.device,
            )
            if offline_bs > 0:
                train_batch = merge_offline_online(
                    offline_data.sample(offline_bs, rng), online_batch, mixing_ratio
                )
            else:
                train_batch = online_batch
            logs = learner.train_on_batch(
                train_batch,
                cql_weight=online_cql,
                use_ovm=use_ovm,
                t_env=t_env,
                episode_delta=batch_size_run,
            )
            logger.write({**logs, "t_env": t_env, "online_buffer": len(online_buffer)})

    ckpt_dir = resolve_path(cfg.get("checkpoint_dir"))
    if ckpt_dir:
        save_qmix_checkpoint(
            ckpt_dir,
            learner,
            include_optimizer=True,
            extra={"seed": int(cfg["seed"]), "t_env": t_env, "phase": "online"},
            default_name=f"ovmse_paper_online_seed{cfg['seed']}.pt",
        )
