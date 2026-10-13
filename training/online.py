# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Online fine-tuning with optional OVM and offline/online batch mixing."""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from data.vault_dataset import print_dataset_stats
from macal.learner import MaCalQLLearner
from macflow.learner import MACFlowLearner
from ovmse.transfer import apply_transfer
from training.evaluation import evaluate
from training.factory import build_learner, load_vault_dataset, make_env, make_logger
from training.rollout import collect_parallel, merge_offline_online, stack_episodes
from utils.checkpoint import save_qmix_checkpoint
from utils.paths import resolve_path


def offline_checkpoint_names(learner_type: str, seed: Any) -> List[str]:
    """Filenames written by this learner's own offline phase."""
    lt = str(learner_type).lower()
    if lt == "macal_ql":
        stem = "macal_ql"
    elif lt == "mac_flow":
        stem = "mac_flow"
    else:
        stem = "ovmse_paper"
    return [f"{stem}_seed{seed}_best.pt", f"{stem}_seed{seed}.pt"]


def _resolve_offline_checkpoint(cfg: Dict[str, Any], offline_ckpt: Optional[str] = None) -> str:
    candidates: List[Optional[str]] = [resolve_path(offline_ckpt or cfg.get("checkpoint_path"))]
    ckpt_dir = resolve_path(cfg.get("checkpoint_dir"))
    if ckpt_dir:
        for name in offline_checkpoint_names(str(cfg.get("learner_type", "ovmse_paper")), cfg["seed"]):
            candidates.append(os.path.join(ckpt_dir, name))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Online requires this run's offline pretraining checkpoint, tried: {candidates}"
    )


def _transfer_mode(cfg: Dict[str, Any]) -> str:
    block = cfg.get("transfer") or {}
    return str(block.get("mode", "ovmse")).lower()


def _load_learner_weights(learner, payload: Dict[str, Any]) -> None:
    learner.q_network.load_state_dict(payload["q_network"])
    learner.mixer.load_state_dict(payload["mixer"])
    learner.target_q_network.load_state_dict(payload.get("target_q_network", payload["q_network"]))
    learner.target_mixer.load_state_dict(payload.get("target_mixer", payload["mixer"]))
    if "optimizer" in payload and getattr(learner, "load_offline_optimizer", True):
        learner.optimizer.load_state_dict(payload["optimizer"])
    if "training_step_ctr" in payload:
        learner.training_step_ctr = int(payload["training_step_ctr"])


def run_online(cfg: Dict[str, Any], offline_ckpt: Optional[str] = None) -> None:
    offline_data = load_vault_dataset(cfg)
    print_dataset_stats(offline_data, label="online_offline_vault")
    learner = build_learner(cfg, offline_data)

    mode = _transfer_mode(cfg)
    transfer_block = cfg.get("transfer") or {}
    shift = float(transfer_block.get("shift", 0.0))
    perm_seed = int(transfer_block.get("perm_seed", int(cfg["seed"]) + 17))
    mixer_seed = int(transfer_block.get("mixer_reinit_seed", int(cfg["seed"]) + 23))

    ckpt = _resolve_offline_checkpoint(cfg, offline_ckpt)
    payload = torch.load(ckpt, map_location=learner.device, weights_only=False)

    if isinstance(learner, MACFlowLearner):
        if "flow" in payload:
            learner.load_state(payload)
        if "training_step_ctr" in payload:
            learner.training_step_ctr = int(payload["training_step_ctr"])
        use_ovm = False
        mode = "mac_flow"
    else:
        if mode != "scratch":
            _load_learner_weights(learner, payload)
            if hasattr(learner, "freeze_oracle"):
                learner.freeze_oracle()

        # Frozen offline nets for diagnostics (and OVM targets); always from the offline checkpoint file.
        if hasattr(learner, "load_offline_for_ovm"):
            learner.load_offline_for_ovm(ckpt)

        use_ovm = mode == "ovmse"
        learner.use_ovm = use_ovm

    batch_size_run = int(cfg.get("batch_size_run", 1))
    envs = [make_env(cfg, seed=int(cfg["seed"]) + i) for i in range(batch_size_run)]
    logger = make_logger(cfg)

    if mode in ("shift_plus", "shift_minus") and shift <= 0.0:
        pre = evaluate(envs[0], learner, int(cfg.get("num_eval_episodes", 32)), 0, cfg)
        shift = abs(float(pre.get("evaluation/q_bias", 1.0)))
        if shift < 1e-6:
            shift = 1.0
        logger.write(
            pre
            | {
                "evaluation/training_steps": 0,
                "t_env": 0,
                "transfer/inferred_shift": shift,
                "transfer/mode": mode,
            },
            force=True,
        )
    else:
        pre = evaluate(envs[0], learner, int(cfg.get("num_eval_episodes", 32)), 0, cfg)
        logger.write(
            pre
            | {"evaluation/training_steps": 0, "t_env": 0, "transfer/mode": mode},
            force=True,
        )

    if not isinstance(learner, MACFlowLearner):
        apply_transfer(
            learner,
            mode,
            shift=shift,
            perm_seed=perm_seed,
            mixer_reinit_seed=mixer_seed,
        )
    print(
        f"online learner={type(learner).__name__} transfer_mode={mode} "
        f"use_ovm={use_ovm} mixing_ratio={float(cfg['system'].get('mixing_ratio', 0.0))} "
        f"shift={shift} offline_ckpt={ckpt}"
    )

    if isinstance(learner, MACFlowLearner):
        from macflow.online_runner import run_macflow_online

        run_macflow_online(cfg, learner, envs[0], logger)
        return

    count_updates = bool(cfg.get("count_updates", False))
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
        return learner.linear_epsilon(t, eps_start, eps_min, eps_decay)

    while t_env < t_max:
        if test_interval > 0 and t_env - last_test_t >= test_interval:
            last_test_t = t_env
            eval_logs = evaluate(envs[0], learner, num_eval, t_env, cfg)
            eval_row = eval_logs | dict(getattr(learner, "last_diag", {}) or {}) | {
                "evaluation/training_steps": t_env,
                "t_env": t_env,
                "transfer/mode": mode,
            }
            logger.write(eval_row, force=True)

        for ep in collect_parallel(envs, learner, eps_at(t_env)):
            online_buffer.append(ep)
            if not count_updates:
                t_env += int(ep["length"])
            learner.t_env = t_env

        if len(online_buffer) < total_batch:
            continue

        # One gradient step per parallel rollout.
        episode_delta = batch_size_run
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
                offline_data.sample(offline_bs, rng, device=learner.device),
                online_batch,
                mixing_ratio,
            )
        else:
            train_batch = online_batch
        if isinstance(learner, MaCalQLLearner):
            logs = learner.train_on_batch(
                train_batch,
                t_env=t_env,
                episode_delta=episode_delta,
                eps=eps_at(t_env),
            )
        elif isinstance(learner, MACFlowLearner):
            logs = learner.train_on_batch(train_batch, t_env=t_env)
            if count_updates:
                t_env += 1
                learner.t_env = t_env
        else:
            logs = learner.train_on_batch(
                train_batch,
                cql_weight=online_cql,
                use_ovm=use_ovm,
                t_env=t_env,
                episode_delta=episode_delta,
            )
        train_logs = {
            **logs,
            "t_env": t_env,
            "online_buffer": len(online_buffer),
            "transfer/mode": mode,
        }
        logger.write(train_logs)

    ckpt_dir = resolve_path(cfg.get("checkpoint_dir"))
    if ckpt_dir:
        save_qmix_checkpoint(
            ckpt_dir,
            learner,
            include_optimizer=True,
            extra={
                "seed": int(cfg["seed"]),
                "t_env": t_env,
                "phase": "online",
                "transfer_mode": mode,
            },
            default_name=(
                f"{getattr(learner, 'checkpoint_stem', None) or ('mac_flow' if isinstance(learner, MACFlowLearner) else 'macal' if isinstance(learner, MaCalQLLearner) else 'ovmse_paper')}"
                f"_online_{mode}_seed{cfg['seed']}.pt"
            ),
        )
