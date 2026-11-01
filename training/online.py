# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Online fine-tuning with optional OVM and offline/online batch mixing."""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from data.vault_dataset import print_dataset_stats
from fsr.learner import FSRLearner
from macal.learner import MaCalQLLearner
from macflow.learner import MACFlowLearner
from muapr.controller import build_muapr_controller
from muapr.meta import MetaGradientEngine
from ovmse.diagnostics import ovm_fixed_sample_diag
from ovmse.learner import PaperQMIXLearner
from ovmse.transfer import apply_transfer
from road.controller import build_road_controller
from sapr.controller import build_sapr_controller
from sapr.probe import probe_prior_utility
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
    elif lt == "fsr":
        stem = "fsr"
    elif lt == "sim2o":
        stem = "sim2o"
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
    if hasattr(learner, "load_fsr_state"):
        learner.load_fsr_state(payload)
    if hasattr(learner, "load_offline_state"):
        learner.load_offline_state(payload)


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


def _periodic_boundary(t_env: int, ckpt_every: int, last_multiple: int) -> Optional[int]:
    """Return the new checkpoint multiple once t_env has crossed one, else None.

    t_env advances by whole episodes, so on variable-length maps (SMAC) it almost
    never lands exactly on a multiple of ckpt_every; comparing consecutive
    divisions therefore never fires and no periodic checkpoint is ever written.
    Track the last saved multiple instead and fire on crossing. The returned
    multiple names the checkpoint after the boundary it covers, while the true
    t_env stays in the checkpoint's `extra`.
    """
    if ckpt_every <= 0:
        return None
    multiple = t_env // ckpt_every
    return multiple if multiple > last_multiple else None


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
    sapr=None,
    muapr=None,
) -> None:
    """Freeze the shared state every branch resumes from, plus a manifest."""
    path = _fork_snapshot_path(cfg, t_env)
    extra = {
        "seed": int(cfg["seed"]),
        "branch": _fork_branch(cfg),
        "fork_at": int(cfg.get("fork_at", -1)),
        "phase": "fork",
    }
    if sapr is not None:
        extra["sapr"] = sapr.state_dict()
    if muapr is not None:
        extra["muapr"] = muapr.state_dict()
    save_trainer_snapshot(
        path,
        learner,
        online_buffer=online_buffer,
        buffer_maxlen=int(cfg.get("buffer_size", 5000)),
        t_env=t_env,
        batch_rng=batch_rng,
        extra=extra,
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


def _load_fork_snapshot(cfg: Dict[str, Any], learner) -> tuple[int, deque, Any, Dict[str, Any]]:
    """Restore the shared trainer state and return (t_env, online_buffer, batch_rng, extra)."""
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
    return t_env, online_buffer, batch_rng, dict(state.get("extra") or {})


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
    keys = {k for r in rows for k in r}
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
    learner.sequential_explore = bool(cfg["system"].get("sequential_explore", False))
    # A conservative term in the online phase needs the CFCQL behaviour prior.  The
    # offline phase trains and caches it, so load the cache here instead of letting
    # the loss run against an untrained network.
    if (
        float(cfg["system"].get("online_cql_weight", 0.0)) > 0
        and getattr(learner, "moderate_lambda", False)
        and not getattr(learner, "_behaviour_trained", False)
    ):
        cache = cfg.get("behaviour_cache_dir")
        print(
            f"[online] preparing behaviour prior for the online conservative term "
            f"(cache={cache})",
            flush=True,
        )
        learner.train_behaviour(
            offline_data,
            cache_path=str(cache) if cache else None,
            batch_size=int(cfg.get("batch_size", 128)),
        )

    mode = _transfer_mode(cfg)
    transfer_block = cfg.get("transfer") or {}
    shift = float(transfer_block.get("shift", 0.0))
    perm_seed = int(transfer_block.get("perm_seed", int(cfg["seed"]) + 17))
    mixer_seed = int(transfer_block.get("mixer_reinit_seed", int(cfg["seed"]) + 23))

    fork_at = int(cfg.get("fork_at", -1))
    resume_fork = cfg.get("resume_fork")
    branch = _fork_branch(cfg)
    fork_extra: Dict[str, Any] = {}

    batch_size_run = int(cfg.get("batch_size_run", 1))
    envs = [make_env(cfg, seed=int(cfg["seed"]) + i) for i in range(batch_size_run)]
    # Dedicated evaluation environment.  SMAC v1 derives its episode sequence from
    # how many times reset() has been called, so scoring on a collection env would
    # put every arm on a different episode set at the same t_env.  A separate env
    # with a fixed seed gives all arms identical evaluation episodes.
    eval_env_seed = int(cfg.get("eval_env_seed", 0) or 0)
    eval_env = make_env(cfg, seed=eval_env_seed) if eval_env_seed > 0 else None
    eval_target = eval_env if eval_env is not None else envs[0]
    logger = make_logger(cfg)
    num_eval = int(cfg.get("num_eval_episodes", 32))

    if resume_fork:
        # A branch: nets/optimizer/buffer/RNG all come from the snapshot, so the
        # offline checkpoint load, freeze_oracle and apply_transfer must be skipped.
        # Q-bar in particular has to be the snapshot's frozen oracle, not a re-freeze
        # of the fork-time live net.
        _seed_branch_log(cfg)
        t_env, online_buffer, rng, fork_extra = _load_fork_snapshot(cfg, learner)
        learner.t_env = t_env
        learner.batch_size_run = batch_size_run
        last_test_t = t_env
        ckpt = str(resume_fork)
        use_ovm = False
        mode = "fork_resume"
        pre = evaluate(eval_target, learner, num_eval, t_env, cfg)
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

            # Frozen offline nets for diagnostics (and OVM targets).  They come from
            # the offline reference checkpoint, which defaults to this run's start
            # but can name a different file: arm D starts from an affine-rescaled
            # copy and must keep the unscaled net as its reference.  Arm Z (scratch)
            # opts out entirely so a from-scratch run carries no offline network.
            reference_ckpt = str(transfer_block.get("offline_reference_ckpt") or ckpt)
            if hasattr(learner, "load_offline_for_ovm") and bool(
                transfer_block.get("load_offline_nets", True)
            ):
                learner.load_offline_for_ovm(reference_ckpt)

            use_ovm = mode == "ovmse"
            learner.use_ovm = use_ovm
            if isinstance(learner, FSRLearner):
                learner.use_ovm = False
                use_ovm = False
                learner.freeze_flow()

        if mode in ("shift_plus", "shift_minus") and shift <= 0.0:
            pre = evaluate(eval_target, learner, num_eval, 0, cfg)
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
            pre = evaluate(eval_target, learner, num_eval, 0, cfg)
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

    rollout_policy = learner
    # ROAD outer controller: picks the offline episode fraction for each period.
    # The inner learner, batch construction and mixer are unchanged.
    road = build_road_controller(cfg, learner)
    # SAPR outer loop: a frozen-prior KL weight adapted from random
    # local-intervention probes. Only the plain QMIX training step computes the
    # KL term, so other learners reject the config instead of silently ignoring
    # it.
    sapr = build_sapr_controller(cfg)
    if sapr is not None:
        if type(learner).train_on_batch is not PaperQMIXLearner.train_on_batch:
            raise RuntimeError(
                "sapr requires the PaperQMIXLearner train_on_batch implementation"
            )
        if getattr(learner, "offline_q_network", None) is None:
            raise RuntimeError(
                "sapr requires a frozen offline policy; enable transfer.load_offline_nets"
            )
        if int(learner.n_actions) <= 1:
            raise RuntimeError("sapr random-intervention probes require discrete actions")
        if fork_extra.get("sapr"):
            sapr.load_state_dict(fork_extra["sapr"])
        learner.set_prior(
            weight=sapr.lambda_value,
            temperature=sapr.settings.temperature,
            support_floor=sapr.settings.support_floor,
        )
    # MU-APR: same frozen prior, but the weight is driven by a meta-gradient on
    # a validation split of recent episodes instead of random-intervention probes.
    muapr = build_muapr_controller(cfg)
    muapr_engine = None
    if muapr is not None:
        if sapr is not None:
            raise RuntimeError("muapr and sapr cannot both be enabled")
        if road is not None:
            raise RuntimeError("muapr cannot be combined with the road controller")
        if type(learner).train_on_batch is not PaperQMIXLearner.train_on_batch:
            raise RuntimeError(
                "muapr requires the PaperQMIXLearner train_on_batch implementation"
            )
        if getattr(learner, "offline_q_network", None) is None:
            raise RuntimeError(
                "muapr requires a frozen offline policy; enable transfer.load_offline_nets"
            )
        if int(learner.n_actions) <= 1:
            raise RuntimeError("muapr requires discrete actions")
        if fork_extra.get("muapr"):
            muapr.load_state_dict(fork_extra["muapr"])
        learner.set_prior(
            weight=muapr.lambda_value,
            temperature=muapr.settings.temperature,
            support_floor=muapr.settings.support_floor,
        )
        muapr_engine = MetaGradientEngine(learner, muapr.settings, offline_data)
    # Teammate-support gate: theta comes from the quantile of the old team's own
    # deviation distribution on the offline dataset, so "supported" is defined by
    # the offline data rather than by a tuned constant.
    gate_info: Dict[str, float] = {}
    # Read the normalised modes off the learner, never the raw config: OmegaConf
    # turns the YAML tokens on/off into booleans, so str(cfg[...]) == "False"
    # here and a raw comparison against "off" is always true.
    _gate_needed = (use_ovm and getattr(learner, "support_gate_mode", "off") != "off") or (
        getattr(learner, "support_cql_mode", "off") != "off"
    )
    if (
        _gate_needed
        and hasattr(learner, "calibrate_support_gate")
        and getattr(learner, "offline_q_network", None) is not None
    ):
        gate_cfg = cfg["system"]
        cal_rng = np.random.default_rng(int(cfg["seed"]) + 7)
        n_batches = int(gate_cfg.get("support_calib_batches", 8))
        cal_bs = int(gate_cfg.get("support_calib_batch_size", 64))
        gate_info = learner.calibrate_support_gate(
            [offline_data.sample(cal_bs, cal_rng, device=learner.device) for _ in range(n_batches)],
            quantile=float(gate_cfg.get("support_quantile", 0.5)),
            temp_frac=float(gate_cfg.get("support_temp_frac", 0.25)),
        )
        print(
            f"support_gate mode={learner.support_gate_mode} "
            f"cql_mode={learner.support_cql_mode} "
            f"theta={gate_info['support/theta']:.4f} temp={gate_info['support/temp']:.4f} "
            f"offline_dev_mean={gate_info['support/offline_dev_mean']:.4f} "
            f"n={gate_info['support/n']}",
            flush=True,
        )
    print(
        f"online learner={type(learner).__name__} transfer_mode={mode} "
        f"use_ovm={use_ovm} mixing_ratio={float(cfg['system'].get('mixing_ratio', 0.0))} "
        f"seq_explore={bool(getattr(learner, 'sequential_explore', False))} "
        f"shift={shift} offline_ckpt={ckpt} fork_at={fork_at} branch={branch} "
        f"resume_fork={resume_fork} "
        f"road={'on' if road is not None else 'off'} "
        f"sapr={'on' if sapr is not None else 'off'}"
    )
    if road is not None:
        print(
            f"[road] arms={road.arms} kappa={road.kappa} c={road.c} "
            f"window={road.window} period_steps={road.period_steps} "
            f"delta_batch={road.delta_batch} normalize={road.normalize_reward} "
            f"mask_explored={road.mask_explored} seed_offset={road.seed_offset} "
            f"initial_ratio={road.ratio}",
            flush=True,
        )
    if sapr is not None:
        s = sapr.settings
        print(
            f"[sapr] lambda={sapr.lambda_value:.6f} bounds=[{s.lambda_min}, {s.lambda_max}] "
            f"base={s.lambda_base} delta={s.delta_lambda} alpha={s.alpha} "
            f"beta={s.beta} rho={s.rho} temperature={s.temperature} "
            f"interval_steps={s.interval_steps} probes={s.probe_episodes} "
            f"behaviour={s.probe_behaviour} se_mode={s.se_mode} "
            f"count_in_budget={s.count_probe_in_budget}",
            flush=True,
        )
    if muapr is not None:
        s = muapr.settings
        print(
            f"[muapr] lambda_ref={s.lambda_ref:.6f} lambda0={muapr.lambda_value:.6f} "
            f"z={muapr.z:.6f} interval={s.interval_steps} start_after={s.start_after_steps} "
            f"K={s.inner_steps} inner_bs={s.inner_batch_size} "
            f"offline_frac={s.kl_offline_frac} val_recent={s.val_recent_episodes} "
            f"val_eps={s.val_episodes} ratio_clip={s.ratio_clip} kappa={s.kappa} "
            f"meta_lr={s.meta_lr} z_clip={s.z_clip}",
            flush=True,
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

    eval_interval_early = int(cfg.get("eval_interval_early", 0) or 0)
    eval_interval_late = int(cfg.get("eval_interval_late", 0) or 0)
    eval_switch_t = int(cfg.get("eval_switch_t", 0) or 0)

    diag_sample = _load_ovm_diag_sample(cfg, learner.device)
    train_stats: List[Dict[str, float]] = []
    ckpt_every = int((cfg.get("ovm_diag") or {}).get("ckpt_every", 0))
    # Resume starts mid-interval, so seed the tracker with the interval already covered.
    last_ckpt_multiple = t_env // ckpt_every if ckpt_every > 0 else -1
    # Experiment 2b probe: its own cadence, independent of the evaluation cadence.
    pv_probe_block = cfg.get("pv_probe") or {}
    probe_every = int(pv_probe_block.get("every", 0) or 0) if pv_probe_block.get("enabled") else 0
    last_probe_t = -probe_every if probe_every > 0 else 0

    while t_env < t_max:
        # Dense early evaluation (eval_interval_early up to eval_switch_t),
        # then the sparser late interval. Falls back to test_interval when unset.
        interval = test_interval
        if eval_interval_early > 0 and t_env <= eval_switch_t:
            interval = eval_interval_early
        elif eval_interval_late > 0 and t_env > eval_switch_t:
            interval = eval_interval_late
        if interval > 0 and t_env - last_test_t >= interval:
            last_test_t = t_env
            eval_logs = evaluate(eval_target, learner, num_eval, t_env, cfg)
            dual_logs: Dict[str, Any] = {}
            if hasattr(learner, "after_evaluation"):
                dual_logs = learner.after_evaluation(eval_logs)
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
                | dual_logs
                | {
                    "evaluation/training_steps": t_env,
                    "t_env": t_env,
                    "transfer/mode": mode,
                    "fork/branch": branch,
                    "ovm_diag/use_ovm": float(use_ovm),
                }
                | gate_info
                | (
                    {
                        "sapr/lambda": float(sapr.lambda_value),
                        "sapr/version": float(sapr.version),
                        "sapr/m": float(sapr.m),
                        "sapr/v": float(sapr.v),
                    }
                    if sapr is not None
                    else {}
                )
                | (
                    {
                        "muapr/lambda": float(muapr.lambda_value),
                        "muapr/z": float(muapr.z),
                        "muapr/version": float(muapr.version),
                    }
                    if muapr is not None
                    else {}
                )
            )
            logger.write(eval_row, force=True)
            train_stats = []
            if fork_at > 0 and t_env >= fork_at:
                # Fork point reached: freeze the shared state and stop. The branches
                # resume from this snapshot and only differ in the ranking weight.
                _write_fork_snapshot(
                    cfg, learner, t_env, online_buffer, rng, sapr=sapr, muapr=muapr
                )
                return

        for ep in collect_parallel(envs, rollout_policy, eps_at(t_env)):
            online_buffer.append(ep)
            if not count_updates:
                t_env += int(ep["length"])
            learner.t_env = t_env

        # Experiment 2b: per-state records from the most recent complete episodes.
        if probe_every > 0 and t_env - last_probe_t >= probe_every and online_buffer:
            last_probe_t = t_env
            try:
                from pv.probe import run_probe

                run_probe(cfg, learner, list(online_buffer), t_env)
            except Exception as exc:  # keep the run alive; report and continue
                print(f"[pv_probe] failed at t_env={t_env}: {exc!r}", flush=True)

        if len(online_buffer) < total_batch:
            continue

        # One gradient step per parallel rollout.
        episode_delta = batch_size_run
        step_mix = mixing_ratio
        support_info: Dict[str, float] = {}
        if road is not None:
            step_mix = road.ratio
        if isinstance(learner, FSRLearner):
            probe_idx = rng.choice(len(online_buffer), size=min(total_batch, len(online_buffer)), replace=False)
            probe = stack_episodes(
                [online_buffer[int(i)] for i in probe_idx],
                offline_data.stats.max_seq_length,
                learner.device,
            )
            support_info = learner.support_mix_ratio(probe)
            step_mix = float(support_info.get("fsr/offline_frac", 0.0))
        if muapr is not None:
            # Mixed history nu_k = 1/2 d_off_ref + 1/2 d_on,k_recent. TD is
            # restricted to the online tail below; KL sees both halves.
            step_mix = float(muapr.settings.kl_offline_frac)
        if step_mix > 0:
            offline_bs = int(round(total_batch * step_mix))
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
        offline_batch = None
        if offline_bs > 0:
            offline_batch = offline_data.sample(offline_bs, rng, device=learner.device)
            train_batch = merge_offline_online(offline_batch, online_batch, step_mix)
        else:
            train_batch = online_batch
        td_row_mask = None
        kl_offline_frac = 0.0
        if muapr is not None:
            n_off = int(offline_batch["obs"].shape[0]) if offline_batch is not None else 0
            n_on = int(online_batch["obs"].shape[0])
            td_row_mask = torch.cat(
                [
                    torch.zeros(n_off, device=learner.device),
                    torch.ones(n_on, device=learner.device),
                ]
            )
            kl_offline_frac = float(n_off / max(n_off + n_on, 1))
        eff_offline_frac = 0.0
        if offline_bs > 0:
            all_filled = train_batch["filled"][..., 0].sum()
            off_filled = train_batch["filled"][:offline_bs, :, 0].sum()
            eff_offline_frac = float(off_filled / all_filled.clamp(min=1.0))
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
        elif isinstance(learner, FSRLearner):
            logs = learner.train_on_batch(
                train_batch,
                cql_weight=online_cql,
                use_ovm=False,
                t_env=t_env,
                episode_delta=episode_delta,
                offline_bs=offline_bs,
                support_w_mean=support_info.get("fsr/w_mean"),
            )
            logs.update(support_info)
        else:
            logs = learner.train_on_batch(
                train_batch,
                cql_weight=online_cql,
                use_ovm=use_ovm,
                t_env=t_env,
                episode_delta=episode_delta,
                td_row_mask=td_row_mask,
            )
        train_logs = {
            **logs,
            "t_env": t_env,
            "online_buffer": len(online_buffer),
            "transfer/mode": mode,
        }
        if muapr is not None:
            train_logs["muapr/kl_offline_frac"] = kl_offline_frac
        if road is not None:
            train_logs["road/ratio"] = float(step_mix)
            train_logs["road/eff_offline_frac"] = eff_offline_frac
        logger.write(train_logs)
        if road is not None:
            road_row = road.after_train_step(offline_data, online_buffer, t_env)
            if road_row is not None:
                logger.write(road_row, force=True)
        tracked_prefixes = ("train_ovm/", "support/", "sapr/", "muapr/")
        if any(k.startswith(tracked_prefixes) for k in logs):
            train_stats.append(
                {k: v for k, v in logs.items() if k.startswith(tracked_prefixes)}
            )

        # SAPR version boundary: freeze the current online policy, run random
        # local-intervention probes on fresh seeded environments, and move the
        # KL weight one smooth step. Probe rollouts count towards the
        # interaction budget; the probe seed advances with the version so
        # successive boundaries draw independent interventions.
        if sapr is not None and sapr.due(t_env):
            probe_stats = probe_prior_utility(
                lambda seed: make_env(cfg, seed=seed),
                learner,
                sapr.settings,
                seed_base=sapr.settings.probe_seed_base + 1000 * sapr.version,
                gamma=float(learner.gamma),
            )
            if probe_stats["valid"]:
                ctrl_logs = sapr.update(probe_stats["mu_hat"], probe_stats["s_hat"])
            else:
                ctrl_logs = sapr.no_update(reason="invalid_estimate")
            learner.set_prior_weight(sapr.lambda_value)
            if sapr.settings.count_probe_in_budget:
                t_env += int(probe_stats["probe_env_steps"])
                learner.t_env = t_env
            sapr.last_probe_t = t_env
            logger.write(
                {"t_env": t_env}
                | {
                    f"sapr/probe/{key}": float(value)
                    for key, value in probe_stats.items()
                    if isinstance(value, (int, float))
                }
                | ctrl_logs,
                force=True,
            )

        # MU-APR meta boundary: snapshot the live learner, run K differentiable
        # Adam steps on mixed-history batches, evaluate the validation split, and
        # move z by one Adam step. The virtual parameters are discarded.
        if muapr is not None and muapr.due(t_env):
            meta_start = time.perf_counter()
            try:
                result = muapr_engine.compute_meta_gradient(
                    online_buffer, rng, t_env, muapr.z
                )
            except Exception as exc:  # keep the run alive; report and continue
                print(f"[muapr] meta step failed at t_env={t_env}: {exc!r}", flush=True)
                result = {
                    "valid": False,
                    "reason": "exception",
                    "meta_grad": 0.0,
                    "meta_loss": 0.0,
                    "kappa_penalty": 0.0,
                    "J_val": 0.0,
                    "stats": {},
                    "inner_rows": [],
                    "engine_time_s": 0.0,
                }
            if result["valid"]:
                ctrl_logs = muapr.update(
                    result["meta_grad"],
                    t_env=t_env,
                    meta_loss=result["meta_loss"],
                    kappa_penalty=result["kappa_penalty"],
                    J_val=result["J_val"],
                )
            else:
                ctrl_logs = muapr.no_update(result["reason"], t_env=t_env)
            learner.set_prior_weight(muapr.lambda_value)
            meta_row = (
                {"t_env": t_env}
                | result["stats"]
                | ctrl_logs
                | {
                    "muapr/meta_time_s": float(time.perf_counter() - meta_start),
                    "muapr/engine_time_s": float(result.get("engine_time_s", 0.0)),
                    "muapr/val_episodes": float(result.get("val_episodes", 0)),
                    "muapr/inner_episodes": float(result.get("inner_episodes", 0)),
                }
            )
            logger.write(meta_row, force=True)
            for inner in result["inner_rows"]:
                logger.write({"t_env": t_env} | inner, force=True)
            print(
                f"[muapr] t_env={t_env} updated={int(bool(result['valid']))} "
                f"z={muapr.z:.4f} lambda={muapr.lambda_value:.4f} "
                f"J={result['J_val']:.4f} m={result['meta_grad']:.5f} "
                f"time={meta_row['muapr/meta_time_s']:.2f}s",
                flush=True,
            )

        boundary = _periodic_boundary(t_env, ckpt_every, last_ckpt_multiple)
        if boundary is not None:
            last_ckpt_multiple = boundary
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
                        **({"sapr": sapr.state_dict()} if sapr is not None else {}),
                        **({"muapr": muapr.state_dict()} if muapr is not None else {}),
                    },
                    default_name=f"ovm_diag_ckpt_t{boundary * ckpt_every}.pt",
                )

    ckpt_dir = resolve_path(cfg.get("checkpoint_dir"))
    if ckpt_dir:
        stem = getattr(learner, "checkpoint_stem", None) or (
            "mac_flow"
            if isinstance(learner, MACFlowLearner)
            else "macal"
            if isinstance(learner, MaCalQLLearner)
            else "fsr"
            if isinstance(learner, FSRLearner)
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
                **({"road": road.state_dict()} if road is not None else {}),
                **({"sapr": sapr.state_dict()} if sapr is not None else {}),
                **({"muapr": muapr.state_dict()} if muapr is not None else {}),
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
