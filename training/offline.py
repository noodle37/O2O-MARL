# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Offline training on vault episode batches."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from data.vault_dataset import print_dataset_stats
from macal.learner import MaCalQLLearner
from training.evaluation import evaluate
from training.factory import build_learner, load_vault_dataset, make_env, make_logger
from utils.checkpoint import save_qmix_checkpoint
from utils.paths import resolve_path


def run_offline(cfg: Dict[str, Any]) -> str:
    dataset = load_vault_dataset(cfg)
    print_dataset_stats(dataset, label="offline")
    learner = build_learner(cfg, dataset)
    if learner.moderate_lambda:
        cache = resolve_path(cfg.get("behaviour_cache_dir"))
        learner.train_behaviour(
            dataset, cache_path=cache, batch_size=int(cfg.get("batch_size", 128))
        )

    env = make_env(cfg, seed=int(cfg["seed"]))
    logger = make_logger(cfg)

    t_max = int(cfg["t_max"])
    batch_size = int(cfg.get("batch_size", 128))
    test_interval = int(cfg.get("test_interval", 10_000))
    num_eval = int(cfg.get("num_eval_episodes", 32))
    ckpt_dir = resolve_path(cfg.get("checkpoint_dir"))
    save_best = bool(cfg.get("save_best_checkpoint", True))
    rng = np.random.default_rng(int(cfg["seed"]))

    best_return = float("-inf")
    best_path: Optional[str] = None
    t_env = 0
    learner.t_env = 0
    last_test_t = -test_interval
    episode_delta = int(cfg.get("batch_size_run", 1))
    n_updates = 0
    sys_cfg = cfg["system"]
    eps_start = float(sys_cfg.get("eps_start", 1.0))
    eps_min = float(sys_cfg.get("eps_min", 0.05))
    eps_decay_steps = int(sys_cfg.get("eps_decay_steps", 0))
    ckpt_prefix = "macal_ql" if isinstance(learner, MaCalQLLearner) else "ovmse_paper"

    while t_env < t_max:
        if test_interval > 0 and t_env - last_test_t >= test_interval:
            last_test_t = t_env
            eval_logs = evaluate(env, learner, num_eval, t_env, cfg)
            logger.write(
                eval_logs | {"evaluation/training_steps": t_env, "t_env": t_env}, force=True
            )
            if save_best and ckpt_dir:
                mean_ret = float(eval_logs.get("evaluation/mean_episode_return", float("-inf")))
                if mean_ret > best_return:
                    best_return = mean_ret
                    best_path = save_qmix_checkpoint(
                        ckpt_dir,
                        learner,
                        include_optimizer=True,
                        extra={
                            "seed": int(cfg["seed"]),
                            "best_eval_return": mean_ret,
                            "t_env": t_env,
                            "phase": "offline",
                        },
                        default_name=f"{ckpt_prefix}_seed{cfg['seed']}_best.pt",
                    )

        batch = dataset.sample(batch_size, rng, device=learner.device)
        filled = int(batch["filled"][:, :-1].sum().item())
        if isinstance(learner, MaCalQLLearner):
            logs = learner.train_on_batch(
                batch,
                t_env=t_env,
                episode_delta=episode_delta,
                eps=learner.linear_epsilon(t_env, eps_start, eps_min, eps_decay_steps),
            )
        else:
            logs = learner.train_on_batch(
                batch,
                cql_weight=learner.cql_weight,
                use_ovm=False,
                t_env=t_env,
                episode_delta=episode_delta,
            )
        t_env += filled
        learner.t_env = t_env
        n_updates += 1
        if n_updates % 50 == 0:
            print(f"offline t_env={t_env} updates={n_updates}", flush=True)
        if t_env >= t_max:
            logger.write({**logs, "t_env": t_env, "training_sps": 0.0})
            break
        logger.write({**logs, "t_env": t_env, "training_sps": 0.0})

    if test_interval > 0:
        eval_logs = evaluate(env, learner, num_eval, t_env, cfg)
        logger.write(eval_logs | {"evaluation/training_steps": t_env, "t_env": t_env}, force=True)
        if save_best and ckpt_dir:
            mean_ret = float(eval_logs.get("evaluation/mean_episode_return", float("-inf")))
            if mean_ret > best_return:
                best_return = mean_ret
                best_path = save_qmix_checkpoint(
                    ckpt_dir,
                    learner,
                    include_optimizer=True,
                    extra={
                        "seed": int(cfg["seed"]),
                        "best_eval_return": mean_ret,
                        "t_env": t_env,
                        "phase": "offline",
                    },
                    default_name=f"{ckpt_prefix}_seed{cfg['seed']}_best.pt",
                )

    final_path = ""
    if ckpt_dir:
        final_path = save_qmix_checkpoint(
            ckpt_dir,
            learner,
            include_optimizer=True,
            extra={"seed": int(cfg["seed"]), "t_env": t_env, "phase": "offline_final"},
            default_name=f"{ckpt_prefix}_seed{cfg['seed']}.pt",
        )
    print(f"Offline complete t_env={t_env} best={best_path} final={final_path}")
    return best_path or final_path
