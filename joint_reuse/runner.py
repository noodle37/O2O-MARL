"""Controlled continuation runs for the joint-reuse validation experiment."""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from joint_reuse.adaptive import AdaptiveLocalReusePolicy
from joint_reuse.policy import HybridRetentionPolicy
from training.evaluation import evaluate
from training.factory import build_learner, load_vault_dataset, make_env, make_logger
from training.rollout import collect_parallel, stack_episodes
from utils.checkpoint import load_qmix_checkpoint, save_qmix_checkpoint
from utils.paths import resolve_path


DEFAULT_OFFLINE_REFERENCE = (
    "checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt"
)


def load_cfg(path: str, overrides: list[str]) -> Dict[str, Any]:
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


def parse_offline_agents(value: str) -> tuple[int, ...]:
    if value is None or str(value).strip() == "":
        return ()
    return tuple(int(x) for x in str(value).split(",") if str(x).strip() != "")


def evaluate_online_student(env, learner, n_episodes: int, t_env: int, cfg: Dict[str, Any]):
    """Evaluate only the live online policy by removing the offline diagnostic nets."""
    offline_q = getattr(learner, "offline_q_network", None)
    offline_mixer = getattr(learner, "offline_mixer", None)
    learner.offline_q_network = None
    learner.offline_mixer = None
    try:
        return evaluate(env, learner, n_episodes, t_env, cfg)
    finally:
        learner.offline_q_network = offline_q
        learner.offline_mixer = offline_mixer


def run_branch(
    cfg: Dict[str, Any],
    *,
    seed: int,
    branch: str,
    offline_agents,
    start_checkpoint: str,
    run_dir: str,
    offline_reference: str = DEFAULT_OFFLINE_REFERENCE,
    steps: int = 250_000,
    eval_episodes: int = 32,
    test_interval: int = 50_000,
    save_every: int = 100_000,
    smoke: bool = False,
    adaptive_model: Optional[str] = None,
) -> str:
    cfg = dict(cfg)
    cfg["seed"] = int(seed)
    if smoke:
        cfg["batch_size"] = 2
        cfg["batch_size_run"] = 2
        eval_episodes = 1
        test_interval = 10**12
        save_every = 0
        steps = min(int(steps), 1_000)

    np.random.seed(int(seed) + 31)
    torch.manual_seed(int(seed) + 32)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) + 33)

    dataset = load_vault_dataset(cfg)
    learner = build_learner(cfg, dataset)
    learner.sequential_explore = bool(cfg["system"].get("sequential_explore", False))
    learner.use_ovm = False
    if hasattr(learner, "ovm_diag_train_stats"):
        learner.ovm_diag_train_stats = False

    start_checkpoint = str(resolve_path(start_checkpoint))
    payload = load_qmix_checkpoint(
        start_checkpoint, learner, load_optimizer=True, map_location=learner.device
    )
    start_t_env = int((payload.get("extra") or {}).get("t_env", payload.get("t_env", 0)))
    if start_t_env <= 0:
        raise ValueError(f"start checkpoint has no positive t_env: {start_checkpoint}")

    offline_reference = str(resolve_path(offline_reference))
    if hasattr(learner, "load_offline_for_ovm"):
        learner.load_offline_for_ovm(offline_reference)

    t_max = int(start_t_env + steps)
    batch_size_run = int(cfg.get("batch_size_run", 1))
    batch_size = int(cfg.get("batch_size", 128))
    learner.batch_size_run = batch_size_run

    envs = [make_env(cfg, seed=int(seed) + i) for i in range(batch_size_run)]
    eval_env_seed = int(cfg.get("eval_env_seed", 0) or 0)
    eval_env = make_env(cfg, seed=eval_env_seed) if eval_env_seed > 0 else envs[0]
    logger = make_logger({**cfg, "log_path": os.path.join(run_dir, "online.jsonl")})

    if branch == "adaptive":
        if not adaptive_model:
            raise ValueError("adaptive branch requires adaptive_model")
        policy = AdaptiveLocalReusePolicy(
            learner,
            str(resolve_path(adaptive_model)),
            rng=np.random.default_rng(int(seed) + 41),
        )
    else:
        policy = HybridRetentionPolicy(
            learner,
            offline_agents,
            rng=np.random.default_rng(int(seed) + 41),
        )
    batch_rng = np.random.default_rng(int(seed) + 43)
    online_buffer = deque(maxlen=int(cfg.get("buffer_size", 5000)))
    eps = float(cfg["system"].get("eps_min", 0.05))
    policy_stats_history = []

    def policy_summary():
        if not policy_stats_history or "adaptive/steps" not in policy_stats_history[0]:
            return {}
        totals = {
            key: sum(float(row[key]) for row in policy_stats_history)
            for key in (
                "adaptive/steps",
                "adaptive/conflicts",
                "adaptive/interventions",
                "adaptive/offline_agent_steps",
                "adaptive/weight_sum",
                "adaptive/forward_seconds",
            )
        }
        steps = max(1.0, totals["adaptive/steps"])
        conflicts = max(1.0, totals["adaptive/conflicts"])
        return totals | {
            "adaptive/conflict_share": totals["adaptive/conflicts"] / steps,
            "adaptive/mean_conflict_weight": totals["adaptive/weight_sum"] / conflicts,
            "adaptive/intervention_share": totals["adaptive/interventions"] / steps,
            "adaptive/offline_agent_share": totals["adaptive/offline_agent_steps"]
            / (steps * learner.n_agents),
        }

    print(
        f"[joint_reuse] seed={seed} branch={branch} retained={tuple(offline_agents)} "
        f"start_t_env={start_t_env} t_max={t_max} steps={steps} "
        f"batch={batch_size} parallel_envs={batch_size_run} "
        f"checkpoint={start_checkpoint}",
        flush=True,
    )

    def extra(t_env: int, phase: str) -> Dict[str, Any]:
        return {
            "seed": int(seed),
            "branch": branch,
            "phase": phase,
            "t_env": int(t_env),
            "start_t_env": int(start_t_env),
            "retained_offline_agents": list(offline_agents),
            "start_checkpoint": start_checkpoint,
            "offline_reference": offline_reference,
            "training_mode": "plain_online_qmix_td_lambda",
            "smoke": bool(smoke),
        }

    t_env = start_t_env
    learner.t_env = t_env
    last_test_t = t_env - test_interval
    last_save_t = t_env
    final_path = os.path.join(
        run_dir,
        f"joint_reuse_online_{branch}_seed{seed}.pt",
    )

    while t_env < t_max:
        if t_env - last_test_t >= test_interval:
            last_test_t = t_env
            eval_logs = evaluate_online_student(
                eval_env, learner, int(eval_episodes), t_env, cfg
            )
            logger.write(
                eval_logs
                | policy_summary()
                | {
                    "evaluation/training_steps": t_env,
                    "t_env": t_env,
                    "branch": branch,
                    "retained_offline_agents": list(offline_agents),
                    "eval/policy": "online_only",
                },
                force=True,
            )

        policy.stats(reset=True)
        episodes = collect_parallel(envs, policy, eps)
        hybrid_stats = policy.stats(reset=True)
        policy_stats_history.append(hybrid_stats)
        for episode in episodes:
            online_buffer.append(episode)
            t_env += int(episode["length"])
            learner.t_env = t_env

        if len(online_buffer) < batch_size:
            continue

        picked = batch_rng.choice(len(online_buffer), size=batch_size, replace=False)
        train_batch = stack_episodes(
            [online_buffer[int(i)] for i in picked],
            dataset.stats.max_seq_length,
            learner.device,
        )
        logs = learner.train_on_batch(
            train_batch,
            cql_weight=0.0,
            use_ovm=False,
            t_env=t_env,
            episode_delta=batch_size_run,
        )
        logger.write(
            {
                **logs,
                **hybrid_stats,
                "t_env": t_env,
                "online_buffer": len(online_buffer),
                "branch": branch,
                "retained_offline_agents": list(offline_agents),
            }
        )

        if save_every > 0 and t_env - last_save_t >= save_every:
            last_save_t = t_env
            save_qmix_checkpoint(
                os.path.join(run_dir, "periodic"),
                learner,
                include_optimizer=True,
                extra=extra(t_env, "online_periodic"),
                default_name=(
                    f"joint_reuse_online_{branch}_seed{seed}_t{t_env}.pt"
                ),
            )

    final_path = save_qmix_checkpoint(
        run_dir,
        learner,
        include_optimizer=True,
        extra=extra(t_env, "online_final"),
        default_name=os.path.basename(final_path),
    )
    eval_logs = evaluate_online_student(eval_env, learner, int(eval_episodes), t_env, cfg)
    logger.write(
        eval_logs
        | policy_summary()
        | {
            "evaluation/training_steps": t_env,
            "t_env": t_env,
            "branch": branch,
            "retained_offline_agents": list(offline_agents),
            "eval/policy": "online_only",
            "final_checkpoint": final_path,
        },
        force=True,
    )
    print(f"FINAL_CHECKPOINT={final_path}", flush=True)

    for env in envs:
        try:
            env.close()
        except Exception:
            pass
    if eval_env is not envs[0]:
        try:
            eval_env.close()
        except Exception:
            pass
    return final_path
