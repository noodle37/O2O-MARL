"""O2O online fine-tuning loop for MAC-Flow.

Mirrors `smac_o2o_main.py` of the original repo:
- single training env, greedy (z=0, masked argmax) collection
- fresh online buffer (no offline mixed in), timeouts count as terminal
- one gradient step per env step after a batch_size*sequence_length warmup
- eval every test_interval env steps over num_eval episodes
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from training.evaluation import evaluate
from training.factory import make_env
from training.rollout_helpers import actions_to_step_array, legals_from_info
from utils.checkpoint import save_qmix_checkpoint
from utils.paths import resolve_path


def _new_episode() -> Dict[str, List[np.ndarray]]:
    return {"obs": [], "actions": [], "reward": [], "done": [], "avail": []}


def _append_step(
    episode: Dict[str, List[np.ndarray]],
    obs: Dict[str, np.ndarray],
    actions: Dict[str, Any],
    info: Dict[str, Any],
    reward: float,
    done: bool,
) -> None:
    agents = sorted(obs.keys())
    episode["obs"].append(np.stack([obs[a] for a in agents]).astype(np.float32))
    episode["actions"].append(actions_to_step_array(actions, agents))
    episode["reward"].append(reward)
    episode["done"].append(done)
    legals = legals_from_info(info, agents)
    if legals is not None:
        episode["avail"].append(np.stack([legals[a] for a in agents]).astype(bool))


def _finalize_episode(episode: Dict[str, List[np.ndarray]], continuous: bool) -> Dict[str, torch.Tensor]:
    """Vault-style tensors with a bootstrap slot at index L (repo `terminals` convention)."""
    length = len(episode["obs"])
    obs = torch.from_numpy(np.stack(episode["obs"]))  # [L, N, O]
    if continuous:
        actions = torch.from_numpy(np.stack(episode["actions"])).float()  # [L, N, D]
    else:
        actions = torch.from_numpy(np.stack(episode["actions"]).astype(np.int64)).unsqueeze(-1)
    reward = torch.tensor(episode["reward"], dtype=torch.float32).unsqueeze(-1)  # [L, 1]
    terminated = torch.zeros(length, 1)
    terminated[length - 1, 0] = 1.0 if episode["done"][-1] else 0.0
    filled = torch.ones(length, 1)

    boot_obs = obs[-1:].clone()
    obs = torch.cat([obs, boot_obs], dim=0)
    if continuous:
        actions = torch.cat([actions, torch.zeros_like(actions[-1:])], dim=0)
    else:
        actions = torch.cat([actions, torch.zeros_like(actions[-1:])], dim=0)
    reward = torch.cat([reward, torch.zeros_like(reward[-1:])], dim=0)
    terminated = torch.cat([terminated, torch.ones(1, 1)], dim=0)
    filled = torch.cat([filled, torch.zeros(1, 1)], dim=0)

    record = {
        "obs": obs,
        "actions": actions,
        "reward": reward,
        "terminated": terminated,
        "filled": filled,
    }
    if episode["avail"]:
        avail = torch.from_numpy(np.stack(episode["avail"]))  # [L, N, A]
        record["avail_actions"] = torch.cat([avail, avail[-1:].clone()], dim=0)
    return record


def _stack_episodes(episodes: List[Dict[str, torch.Tensor]], device: torch.device) -> Dict[str, torch.Tensor]:
    b = len(episodes)
    s = max(int(ep["obs"].shape[0]) for ep in episodes)
    n, o = episodes[0]["obs"].shape[1:]
    has_avail = "avail_actions" in episodes[0]
    obs = torch.zeros(b, s, n, o)
    act_shape = episodes[0]["actions"].shape[1:]
    actions = torch.zeros(b, s, *act_shape, dtype=episodes[0]["actions"].dtype)
    reward = torch.zeros(b, s, 1)
    terminated = torch.zeros(b, s, 1)
    filled = torch.zeros(b, s, 1)
    avail = (
        torch.zeros(b, s, n, episodes[0]["avail_actions"].shape[-1], dtype=torch.bool)
        if has_avail
        else None
    )
    for i, ep in enumerate(episodes):
        ln = int(ep["obs"].shape[0])
        obs[i, :ln] = ep["obs"]
        actions[i, :ln] = ep["actions"]
        reward[i, :ln] = ep["reward"]
        terminated[i, :ln] = ep["terminated"]
        filled[i, :ln] = ep["filled"]
        if avail is not None:
            avail[i, :ln] = ep["avail_actions"]
    batch = {
        "obs": obs.to(device),
        "actions": actions.to(device),
        "reward": reward.to(device),
        "terminated": terminated.to(device),
        "filled": filled.to(device),
    }
    if avail is not None:
        batch["avail_actions"] = avail.to(device)
    return batch


def run_macflow_online(cfg: Dict[str, Any], learner, env, logger) -> None:
    seed = int(cfg["seed"])
    t_max = int(cfg["t_max"])
    test_interval = int(cfg.get("test_interval", 100_000))
    num_eval = int(cfg.get("num_eval_episodes", 10))
    batch_eps = int(cfg.get("batch_size", 32))
    seq_len = int(getattr(learner, "sequence_length", 20))
    warmup = batch_eps * seq_len
    buffer_size = int(cfg.get("buffer_size", 5_000))
    rng = np.random.default_rng(seed + 1)
    eval_env = make_env(cfg, seed=seed)

    print(
        f"MAC-Flow online (O2O): t_max={t_max} episodes_per_batch={batch_eps} "
        f"seq_len={seq_len} warmup={warmup} buffer={buffer_size} "
        f"eval every {test_interval} x {num_eval} eps"
    )

    online_buffer: deque = deque(maxlen=buffer_size)
    t_env = 0
    learner.t_env = 0
    last_test_t = 0
    n_updates = 0
    obs, info = env.reset()
    episode = _new_episode()

    while t_env < t_max:
        agents = sorted(obs.keys())
        actions = learner.select_actions_eval(obs, legals_from_info(info, agents), 0.0)
        next_obs, rewards, terminal, trunc, next_info = env.step(actions)
        done_step = bool(all(terminal.values()) or all(trunc.values()))
        _append_step(
            episode,
            obs,
            actions,
            info,
            float(np.mean(list(rewards.values()))),
            done_step,
        )
        t_env += 1
        learner.t_env = t_env

        if done_step:
            online_buffer.append(_finalize_episode(episode, learner.continuous))
            episode = _new_episode()
            obs, info = env.reset()
        else:
            obs, info = next_obs, next_info

        if t_env > warmup and len(online_buffer) > 0:
            replace = len(online_buffer) < batch_eps
            picked = rng.choice(len(online_buffer), size=batch_eps, replace=replace)
            batch = _stack_episodes(
                [online_buffer[int(i)] for i in picked], learner.device
            )
            logs = learner.train_on_batch(batch, t_env=t_env)
            n_updates += 1
            logger.write(
                {
                    **logs,
                    "t_env": t_env,
                    "online_buffer": len(online_buffer),
                    "transfer/mode": "mac_flow",
                }
            )

        if test_interval > 0 and t_env - last_test_t >= test_interval:
            last_test_t = t_env
            eval_logs = evaluate(eval_env, learner, num_eval, t_env, cfg)
            logger.write(
                eval_logs
                | dict(getattr(learner, "last_diag", {}) or {})
                | {
                    "evaluation/training_steps": t_env,
                    "t_env": t_env,
                    "transfer/mode": "mac_flow",
                },
                force=True,
            )

        if t_env % 50_000 == 0:
            print(
                f"online t_env={t_env} updates={n_updates} "
                f"buffer_eps={len(online_buffer)}",
                flush=True,
            )

    if test_interval > 0 and last_test_t < t_env:
        eval_logs = evaluate(eval_env, learner, num_eval, t_env, cfg)
        logger.write(
            eval_logs
            | dict(getattr(learner, "last_diag", {}) or {})
            | {
                "evaluation/training_steps": t_env,
                "t_env": t_env,
                "transfer/mode": "mac_flow",
            },
            force=True,
        )

    ckpt_dir = resolve_path(cfg.get("checkpoint_dir"))
    if ckpt_dir:
        stem = getattr(learner, "checkpoint_stem", None) or "mac_flow"
        save_qmix_checkpoint(
            ckpt_dir,
            learner,
            include_optimizer=True,
            extra={
                "seed": seed,
                "t_env": t_env,
                "phase": "online",
                "transfer_mode": "mac_flow",
            },
            default_name=f"{stem}_online_mac_flow_seed{seed}.pt",
        )
    print(f"MAC-Flow online complete t_env={t_env} updates={n_updates}")
