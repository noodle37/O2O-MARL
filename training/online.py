# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Online fine-tuning with optional OVM and offline/online batch mixing."""

from __future__ import annotations

import json
import os
import shutil
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from data.vault_dataset import print_dataset_stats
from macal.learner import MaCalQLLearner
from macflow.learner import MACFlowLearner
from ovmse.diagnostics import ovm_fixed_sample_diag
from ovmse.transfer import apply_transfer
from training.evaluation import evaluate
from training.factory import build_learner, load_vault_dataset, make_env, make_logger
from training.rollout import collect_parallel, merge_offline_online, stack_episodes
from utils.checkpoint import load_trainer_snapshot, save_qmix_checkpoint, save_trainer_snapshot
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


FORK_BRANCHES = ("continue", "off", "decay")


def _fork_branch(cfg: Dict[str, Any]) -> str:
    """Branch label, used for log tags and checkpoint filenames.

    "main" is the parent run; the three branches are FORK_BRANCHES. YAML 1.1 reads a
    bare `off`/`no`/`false` as a boolean, so an unquoted `fork_branch=off` override
    silently becomes False and would fall through to a default — mislabelling the
    branch and making two branches share a checkpoint filename. Quote the override
    (`fork_branch='off'`) and treat a bool as an error rather than silently
    relabelling the run.
    """
    value = cfg.get("fork_branch")
    if value is None or value == "":
        return "main"
    if isinstance(value, bool):
        raise ValueError(
            f"fork_branch parsed as {value!r}: YAML reads a bare 'off'/'no'/'false' "
            "as a boolean. Quote it, e.g. fork_branch='off'."
        )
    name = str(value)
    if name != "main" and name not in FORK_BRANCHES:
        raise ValueError(f"unknown fork_branch {name!r}; expected 'main' or {FORK_BRANCHES}")
    return name


def _fork_snapshot_path(cfg: Dict[str, Any], t_env: int) -> str:
    ckpt_dir = resolve_path(cfg.get("checkpoint_dir"))
    name = f"fork_snapshot_{_fork_branch(cfg)}_t{t_env}.pt"
    return os.path.join(ckpt_dir, name) if ckpt_dir else name


def _fork_meta_path(cfg: Dict[str, Any], t_env: int) -> str:
    return _fork_snapshot_path(cfg, t_env).replace(".pt", ".json")


def _seed_branch_log(cfg: Dict[str, Any]) -> None:
    """Prepend the parent's pre-fork rows so each branch log is a full curve.

    The parent run wrote the shared history up to the fork point; a branch starts
    from that same point, so its log file should contain those rows too. Appending
    is not enough: the parent rows must come first and be written only once.
    """
    parent_log = cfg.get("fork_parent_log")
    log_path = cfg.get("log_path")
    if not parent_log or not log_path:
        return
    parent_log = str(resolve_path(parent_log))
    log_path = str(resolve_path(log_path))
    if not os.path.isfile(parent_log) or parent_log == log_path:
        return
    if os.path.isfile(log_path) and os.path.getsize(log_path) > 0:
        # This branch log already has rows (a re-run); leave it alone rather than
        # duplicating the parent prefix.
        return
    parent_rows = open(parent_log, encoding="utf-8").read()
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(parent_rows)
    print(f"[fork] seeded branch log {log_path} with {len(parent_rows.splitlines())} parent rows")


def _write_fork_snapshot(
    cfg: Dict[str, Any],
    learner,
    t_env: int,
    online_buffer: deque,
    batch_rng,
) -> None:
    """Freeze the shared state every branch resumes from, plus a manifest."""
    path = _fork_snapshot_path(cfg, t_env)
    save_trainer_snapshot(
        path,
        learner,
        online_buffer=online_buffer,
        buffer_maxlen=int(cfg.get("buffer_size", 5000)),
        t_env=t_env,
        batch_rng=batch_rng,
        extra={
            "seed": int(cfg["seed"]),
            "branch": _fork_branch(cfg),
            "fork_at": int(cfg.get("fork_at", -1)),
            "phase": "fork",
        },
    )
    meta = {
        "seed": int(cfg["seed"]),
        "t_env": int(t_env),
        "fork_branch": _fork_branch(cfg),
        "snapshot": path,
        "offline_checkpoint": str(cfg.get("checkpoint_path", "")),
        "cfg": cfg,
    }
    meta_path = _fork_meta_path(cfg, t_env)
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, default=str)
    print(f"[fork] snapshot t_env={t_env} -> {path}")


def _load_fork_snapshot(cfg: Dict[str, Any], learner) -> tuple[int, deque, Any]:
    """Restore the shared trainer state and return (t_env, online_buffer, batch_rng)."""
    path = str(resolve_path(cfg["resume_fork"]))
    state = load_trainer_snapshot(path, learner, map_location=learner.device)
    t_env = int(state["t_env"])
    online_buffer = state.get("online_buffer")
    if online_buffer is None:
        online_buffer = deque(maxlen=int(cfg.get("buffer_size", 5000)))
    batch_rng = state.get("batch_rng")
    if batch_rng is None:
        batch_rng = np.random.default_rng(int(cfg["seed"]) + 1)
    print(
        f"[fork] resumed branch={_fork_branch(cfg)} t_env={t_env} "
        f"buffer={len(online_buffer)} snapshot={path}"
    )
    return t_env, online_buffer, batch_rng


def _load_ovm_diag_sample(cfg: Dict[str, Any], device) -> Optional[Dict[str, Any]]:
    """Fixed batch of state-action samples re-scored at every eval point."""
    block = cfg.get("ovm_diag") or {}
    path = block.get("sample_path")
    if not path:
        return None
    path = str(resolve_path(path))
    if not os.path.isfile(path):
        print(f"[ovm_diag] sample not found at {path}; fixed-sample diagnostics disabled")
        return None
    payload = torch.load(path, map_location=device, weights_only=False)
    batch = payload["batch"] if "batch" in payload else payload
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    print(
        f"[ovm_diag] fixed sample {path} episodes={batch['obs'].shape[0]} "
        f"seq={batch['obs'].shape[1]} (meta={payload.get('meta', {})})"
    )
    return batch


def _mean_stats(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Running mean of per-step training diagnostics over one eval interval."""
    if not rows:
        return {}
    keys = rows[0].keys()
    out: Dict[str, float] = {}
    for key in keys:
        vals = [float(r[key]) for r in rows if key in r and r[key] == r[key]]
        if vals:
            out[f"{key}_interval_mean"] = float(sum(vals) / len(vals))
    out["train_ovm/n_steps"] = float(len(rows))
    return out


def run_online(cfg: Dict[str, Any], offline_ckpt: Optional[str] = None) -> None:
    offline_data = load_vault_dataset(cfg)
    print_dataset_stats(offline_data, label="online_offline_vault")
    learner = build_learner(cfg, offline_data)

    mode = _transfer_mode(cfg)
    transfer_block = cfg.get("transfer") or {}
    shift = float(transfer_block.get("shift", 0.0))
    perm_seed = int(transfer_block.get("perm_seed", int(cfg["seed"]) + 17))
    mixer_seed = int(transfer_block.get("mixer_reinit_seed", int(cfg["seed"]) + 23))

    fork_at = int(cfg.get("fork_at", -1))
    resume_fork = cfg.get("resume_fork")
    branch = _fork_branch(cfg)

    batch_size_run = int(cfg.get("batch_size_run", 1))
    envs = [make_env(cfg, seed=int(cfg["seed"]) + i) for i in range(batch_size_run)]
    logger = make_logger(cfg)
    num_eval = int(cfg.get("num_eval_episodes", 32))

    if resume_fork:
        # A branch: nets/optimizer/buffer/RNG all come from the snapshot, so the
        # offline checkpoint load, freeze_oracle and apply_transfer must be skipped.
        # Q-bar in particular has to be the snapshot's frozen oracle, not a re-freeze
        # of the fork-time live net.
        _seed_branch_log(cfg)
        t_env, online_buffer, rng = _load_fork_snapshot(cfg, learner)
        learner.t_env = t_env
        learner.batch_size_run = batch_size_run
        last_test_t = t_env
        ckpt = str(resume_fork)
        use_ovm = False
        mode = "fork_resume"
        pre = evaluate(envs[0], learner, num_eval, t_env, cfg)
        logger.write(
            pre
            | dict(getattr(learner, "last_diag", {}) or {})
            | {
                "evaluation/training_steps": t_env,
                "t_env": t_env,
                "transfer/mode": mode,
                "fork/branch": branch,
                "fork/resumed": 1.0,
            },
            force=True,
        )
    else:
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

        if mode in ("shift_plus", "shift_minus") and shift <= 0.0:
            pre = evaluate(envs[0], learner, num_eval, 0, cfg)
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
            pre = evaluate(envs[0], learner, num_eval, 0, cfg)
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
        f"shift={shift} offline_ckpt={ckpt} fork_at={fork_at} branch={branch} "
        f"resume_fork={resume_fork}"
    )

    if isinstance(learner, MACFlowLearner):
        from macflow.online_runner import run_macflow_online

        run_macflow_online(cfg, learner, envs[0], logger)
        return

    count_updates = bool(cfg.get("count_updates", False))
    t_max = int(cfg["t_max"])
    test_interval = int(cfg.get("test_interval", 10_000))
    eps_start = float(cfg["system"].get("eps_start", 1.0))
    eps_min = float(cfg["system"].get("eps_min", 0.05))
    eps_decay = int(cfg["system"].get("eps_decay_steps", 100_000))
    mixing_ratio = float(cfg["system"].get("mixing_ratio", 0.0))
    online_cql = float(cfg["system"].get("online_cql_weight", 0.0))
    total_batch = int(cfg.get("batch_size", 128))
    buffer_size = int(cfg.get("buffer_size", 5000))

    if resume_fork:
        # t_env / online_buffer / rng already restored from the snapshot above.
        pass
    else:
        rng = np.random.default_rng(int(cfg["seed"]) + 1)
        online_buffer = deque(maxlen=buffer_size)
        t_env = 0
        learner.t_env = 0
        last_test_t = -test_interval
    learner.batch_size_run = batch_size_run

    def eps_at(t: int) -> float:
        return learner.linear_epsilon(t, eps_start, eps_min, eps_decay)

    diag_sample = _load_ovm_diag_sample(cfg, learner.device)
    train_stats: List[Dict[str, float]] = []
    ckpt_every = int((cfg.get("ovm_diag") or {}).get("ckpt_every", 0))

    while t_env < t_max:
        if test_interval > 0 and t_env - last_test_t >= test_interval:
            last_test_t = t_env
            eval_logs = evaluate(envs[0], learner, num_eval, t_env, cfg)
            diag_logs: Dict[str, Any] = {}
            if diag_sample is not None:
                try:
                    diag_logs = ovm_fixed_sample_diag(learner, diag_sample, t_env)
                except Exception as exc:  # keep the run alive; report and continue
                    print(f"[ovm_diag] fixed-sample diagnostic failed: {exc!r}")
                    diag_logs = {}
            eval_row = (
                eval_logs
                | dict(getattr(learner, "last_diag", {}) or {})
                | diag_logs
                | _mean_stats(train_stats)
                | {
                    "evaluation/training_steps": t_env,
                    "t_env": t_env,
                    "transfer/mode": mode,
                    "fork/branch": branch,
                    "ovm_diag/use_ovm": float(use_ovm),
                }
            )
            logger.write(eval_row, force=True)
            train_stats = []
            if fork_at > 0 and t_env >= fork_at:
                # Fork point reached: freeze the shared state and stop. The branches
                # resume from this snapshot and only differ in the ranking weight.
                _write_fork_snapshot(cfg, learner, t_env, online_buffer, rng)
                return

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
        if any(k.startswith("train_ovm/") for k in logs):
            train_stats.append(
                {k: v for k, v in logs.items() if k.startswith("train_ovm/")}
            )
        if ckpt_every > 0 and t_env // ckpt_every != (t_env - 1) // ckpt_every:
            ckpt_dir_now = resolve_path(cfg.get("checkpoint_dir"))
            if ckpt_dir_now:
                save_qmix_checkpoint(
                    os.path.join(ckpt_dir_now, "periodic"),
                    learner,
                    include_optimizer=True,
                    extra={
                        "seed": int(cfg["seed"]),
                        "t_env": t_env,
                        "phase": "online",
                        "transfer_mode": mode,
                        "fork_branch": branch,
                    },
                    default_name=f"ovm_diag_ckpt_t{t_env}.pt",
                )

    ckpt_dir = resolve_path(cfg.get("checkpoint_dir"))
    if ckpt_dir:
        stem = getattr(learner, "checkpoint_stem", None) or (
            "mac_flow"
            if isinstance(learner, MACFlowLearner)
            else "macal"
            if isinstance(learner, MaCalQLLearner)
            else "ovmse_paper"
        )
        # Fork branches share one directory, so the branch name must be in the file.
        suffix = "" if branch in ("", "main") else f"_{branch}"
        save_qmix_checkpoint(
            ckpt_dir,
            learner,
            include_optimizer=True,
            extra={
                "seed": int(cfg["seed"]),
                "t_env": t_env,
                "phase": "online",
                "transfer_mode": mode,
                "fork_branch": branch,
            },
            default_name=f"{stem}_online_{mode}{suffix}_seed{cfg['seed']}.pt",
        )

    # Archive the online batch distribution this arm actually trained on, so the
    # activation rate can be re-scored on the arm's own data later. The cap keeps
    # long-episode maps (2halfcheetah: 1000 steps/episode) to a sane file size.
    snapshot_eps = int((cfg.get("ovm_diag") or {}).get("buffer_snapshot_episodes", 0))
    if snapshot_eps != 0 and ckpt_dir:
        eps_to_save = (
            list(online_buffer)
            if snapshot_eps < 0
            else list(online_buffer)[-snapshot_eps:]
        )
        save_trainer_snapshot(
            os.path.join(ckpt_dir, f"ovm_diag_final_snapshot_t{t_env}.pt"),
            learner,
            online_buffer=eps_to_save,
            buffer_maxlen=int(cfg.get("buffer_size", 5000)),
            t_env=t_env,
            batch_rng=rng,
            extra={
                "seed": int(cfg["seed"]),
                "branch": branch,
                "phase": "online_final",
                "transfer_mode": mode,
                "use_ovm": bool(use_ovm),
                "n_buffer_episodes_saved": len(eps_to_save),
            },
        )
