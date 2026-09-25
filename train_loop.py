# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Offline / online training loops for paper OVMSE (Vault episodes + PyMARL semantics)."""

from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from og_marl.baselines.torch_systems.checkpointing import resolve_path, save_qmix_checkpoint
from og_marl.environments import get_environment
from og_marl.loggers import WandbLogger
from og_marl.vault_utils.download_vault import download_and_unzip_vault
from og_marl.vault_utils.vault_episodes import (
    EpisodeDataset,
    load_episode_dataset_from_vault,
    print_dataset_stats,
)
from learner import PaperQMIXLearner


def _build_learner(cfg: Dict[str, Any], dataset: EpisodeDataset) -> PaperQMIXLearner:
    s = dataset.stats
    sys_cfg = cfg["system"]
    return PaperQMIXLearner(
        n_agents=s.n_agents,
        n_actions=s.n_actions,
        obs_dim=s.obs_dim,
        state_dim=s.state_dim,
        linear_layer_dim=int(sys_cfg.get("linear_layer_dim", 64)),
        recurrent_layer_dim=int(sys_cfg.get("recurrent_layer_dim", 64)),
        mixer_embed_dim=int(sys_cfg.get("mixer_embed_dim", 32)),
        mixer_hyper_dim=int(sys_cfg.get("mixer_hyper_dim", 64)),
        learning_rate=float(sys_cfg.get("learning_rate", 1e-3)),
        gamma=float(sys_cfg.get("discount", 0.99)),
        td_lambda=float(sys_cfg.get("td_lambda", 0.6)),
        grad_norm_clip=float(sys_cfg.get("grad_norm_clip", 10.0)),
        target_update_interval=int(sys_cfg.get("target_update_interval", 200)),
        batch_size_run=int(cfg.get("batch_size_run", 1)),
        add_agent_id_to_obs=bool(sys_cfg.get("add_agent_id_to_obs", True)),
        raw_cql=bool(sys_cfg.get("raw_cql", False)),
        moderate_lambda=bool(sys_cfg.get("moderate_lambda", False)),
        cql_weight=float(sys_cfg.get("cql_weight", 0.05)),
        online_cql_weight=float(sys_cfg.get("online_cql_weight", 0.0)),
        num_ood_actions=int(sys_cfg.get("num_ood_actions", 100)),
        softmax_temp=float(sys_cfg.get("softmax_temp", 5.0)),
        max_behaviour_filled_steps=int(sys_cfg.get("max_behaviour_filled_steps", 500_000)),
        use_ovm=bool(sys_cfg.get("use_ovm", False)),
        mixing_ratio=float(sys_cfg.get("mixing_ratio", 0.0)),
        lambda_memory_end=float(sys_cfg.get("lambda_memory_end", 1.0)),
        lambda_anneal_steps=int(sys_cfg.get("lambda_anneal_steps", 0)),
        device=str(sys_cfg.get("device", "cuda")),
    )


def load_vault_dataset(cfg: Dict[str, Any]) -> EpisodeDataset:
    task = cfg["task"]
    download_and_unzip_vault(task["source"], task["env"], task["scenario"])
    return load_episode_dataset_from_vault(
        task["source"],
        task["env"],
        task["scenario"],
        str(task["dataset"]),
        rel_dir="vaults",
        device=str(cfg["system"].get("device", "cuda")),
    )


def run_offline(cfg: Dict[str, Any]) -> str:
    dataset = load_vault_dataset(cfg)
    print_dataset_stats(dataset, label="offline")
    learner = _build_learner(cfg, dataset)
    if learner.moderate_lambda:
        cache = resolve_path(cfg.get("behaviour_cache_dir"))
        learner.train_behaviour(
            dataset, cache_path=cache, batch_size=int(cfg.get("batch_size", 128))
        )

    env = get_environment(
        cfg["task"]["source"], cfg["task"]["env"], cfg["task"]["scenario"], seed=int(cfg["seed"])
    )
    logger = WandbLogger(
        project=cfg.get("wandb_project", "og-marl"),
        config={**cfg},
    )

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

    while t_env < t_max:
        if test_interval > 0 and t_env - last_test_t >= test_interval:
            last_test_t = t_env
            eval_logs = _evaluate(env, learner, num_eval, t_env, cfg)
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
                        default_name=f"ovmse_paper_seed{cfg['seed']}_best.pt",
                    )

        batch = dataset.sample(batch_size, rng)
        filled = int(batch["filled"][:, :-1].sum().item())
        logs = learner.train_on_batch(
            batch,
            cql_weight=learner.cql_weight,
            use_ovm=False,
            t_env=t_env,
            episode_delta=episode_delta,
        )
        t_env += filled
        learner.t_env = t_env
        if t_env >= t_max:
            logger.write({**logs, "t_env": t_env, "training_sps": 0.0})
            break
        logger.write({**logs, "t_env": t_env, "training_sps": 0.0})

    final_path = ""
    if ckpt_dir:
        final_path = save_qmix_checkpoint(
            ckpt_dir,
            learner,
            include_optimizer=True,
            extra={"seed": int(cfg["seed"]), "t_env": t_env, "phase": "offline_final"},
            default_name=f"ovmse_paper_seed{cfg['seed']}.pt",
        )
    print(f"Offline complete t_env={t_env} best={best_path} final={final_path}")
    return best_path or final_path


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
    learner = _build_learner(cfg, offline_data)
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
        get_environment(
            cfg["task"]["source"],
            cfg["task"]["env"],
            cfg["task"]["scenario"],
            seed=int(cfg["seed"]) + i,
        )
        for i in range(batch_size_run)
    ]
    logger = WandbLogger(project=cfg.get("wandb_project", "og-marl"), config={**cfg})

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
            eval_logs = _evaluate(envs[0], learner, num_eval, t_env, cfg)
            logger.write(eval_logs | {"evaluation/training_steps": t_env, "t_env": t_env}, force=True)

        for ep in _collect_parallel(envs, learner, eps_at(t_env)):
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
            online_batch = _stack_episodes(
                [online_buffer[int(i)] for i in picked],
                offline_data.stats.max_seq_length,
                learner.device,
            )
            if offline_bs > 0:
                train_batch = _merge_offline_online(
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


def _evaluate(env, learner: PaperQMIXLearner, n_episodes: int, t_env: int, cfg: Dict[str, Any]) -> Dict[str, float]:
    eps = 0.0
    returns = []
    wins = []
    for _ in range(n_episodes):
        learner.reset_eval_hidden()
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        won = None
        while not done:
            actions = learner.select_actions_eval(obs, info["legals"], eps)
            obs, rewards, terminal, trunc, info = env.step(actions)
            ep_ret += float(np.mean(list(rewards.values())))
            if "battle_won" in info:
                won = bool(info["battle_won"])
            done = all(terminal.values()) or all(trunc.values())
        returns.append(ep_ret)
        if won is not None:
            wins.append(float(won))
    logs = {
        "evaluation/mean_episode_return": float(np.mean(returns)),
        "evaluation/max_episode_return": float(np.max(returns)),
        "evaluation/min_episode_return": float(np.min(returns)),
    }
    if wins:
        logs["evaluation/win_rate"] = float(np.mean(wins))
    return logs


def _collect_parallel(envs, learner: PaperQMIXLearner, eps: float) -> List[Dict[str, Any]]:
    n = len(envs)
    states = [learner.new_episode_state() for _ in range(n)]
    episodes = [{"steps": [], "length": 0} for _ in range(n)]
    obs_info = [env.reset() for env in envs]
    done = [False] * n
    pool = ThreadPoolExecutor(max_workers=n)

    def step_one(i: int, actions):
        if done[i]:
            return None
        return envs[i].step(actions)

    try:
        while not all(done):
            actions = [None] * n
            for i in range(n):
                if done[i]:
                    continue
                obs, info = obs_info[i]
                actions[i] = learner.select_actions_eval(obs, info["legals"], eps, states[i])
            results = list(pool.map(lambda i: step_one(i, actions[i]), range(n)))
            for i, result in enumerate(results):
                if result is None:
                    continue
                obs, info = obs_info[i]
                next_obs, rewards, terminal, trunc, next_info = result
                episodes[i]["steps"].append(
                    {
                        "obs": np.stack([obs[a] for a in sorted(obs.keys())]),
                        "actions": np.array([actions[i][a] for a in sorted(obs.keys())], dtype=np.int64),
                        "state": info["state"],
                        "legals": np.stack([info["legals"][a] for a in sorted(info["legals"].keys())]),
                        "reward": float(np.mean(list(rewards.values()))),
                        "terminated": float(all(terminal.values())),
                    }
                )
                obs_info[i] = (next_obs, next_info)
                if all(terminal.values()) or all(trunc.values()):
                    done[i] = True
                    episodes[i]["length"] = len(episodes[i]["steps"])
    finally:
        pool.shutdown(wait=True)
    return episodes


def _collect_episode(env, learner: PaperQMIXLearner, eps: float) -> Dict[str, Any]:
    learner.reset_eval_hidden()
    obs, info = env.reset()
    steps = []
    done = False
    while not done:
        actions = learner.select_actions_eval(obs, info["legals"], eps)
        next_obs, rewards, terminal, trunc, next_info = env.step(actions)
        steps.append(
            {
                "obs": np.stack([obs[a] for a in sorted(obs.keys())]),
                "actions": np.array([actions[a] for a in sorted(obs.keys())], dtype=np.int64),
                "state": info["state"],
                "legals": np.stack([info["legals"][a] for a in sorted(info["legals"].keys())]),
                "reward": float(np.mean(list(rewards.values()))),
                "terminated": float(all(terminal.values())),
            }
        )
        obs, info = next_obs, next_info
        done = all(terminal.values()) or all(trunc.values())
    return {"steps": steps, "length": len(steps)}


def _stack_episodes(episodes, max_seq: int, device: torch.device) -> Dict[str, torch.Tensor]:
    n = len(episodes)
    if n == 0:
        raise ValueError("empty episode list")
    n_agents = episodes[0]["steps"][0]["obs"].shape[0]
    obs_dim = episodes[0]["steps"][0]["obs"].shape[1]
    state_dim = episodes[0]["steps"][0]["state"].shape[0]
    n_actions = episodes[0]["steps"][0]["legals"].shape[1]
    obs = torch.zeros(n, max_seq, n_agents, obs_dim, device=device)
    actions = torch.zeros(n, max_seq, n_agents, 1, dtype=torch.long, device=device)
    state = torch.zeros(n, max_seq, state_dim, device=device)
    avail = torch.zeros(n, max_seq, n_agents, n_actions, dtype=torch.bool, device=device)
    reward = torch.zeros(n, max_seq, 1, device=device)
    term = torch.zeros(n, max_seq, 1, device=device)
    filled = torch.zeros(n, max_seq, 1, device=device)
    for i, ep in enumerate(episodes):
        L = len(ep["steps"])
        for t, st in enumerate(ep["steps"]):
            obs[i, t] = torch.from_numpy(st["obs"]).float()
            actions[i, t, :, 0] = torch.from_numpy(st["actions"]).long()
            state[i, t] = torch.from_numpy(st["state"]).float()
            avail[i, t] = torch.from_numpy(st["legals"]).bool()
            reward[i, t, 0] = st["reward"]
            term[i, t, 0] = st["terminated"]
            filled[i, t, 0] = 1.0
        if L < max_seq:
            obs[i, L] = obs[i, L - 1]
            state[i, L] = state[i, L - 1]
            avail[i, L] = avail[i, L - 1]
            term[i, L, 0] = 1.0
            filled[i, L, 0] = 0.0
    return {
        "obs": obs,
        "actions": actions,
        "state": state,
        "avail_actions": avail,
        "reward": reward,
        "terminated": term,
        "filled": filled,
    }


def _merge_offline_online(
    off: Dict[str, torch.Tensor], on: Dict[str, torch.Tensor], mixing_ratio: float
) -> Dict[str, torch.Tensor]:
    max_t = min(off["obs"].shape[1], on["obs"].shape[1])
    merged = {}
    for key in off:
        if key not in on:
            continue
        o = off[key][:, :max_t]
        n = on[key][:, :max_t]
        b_off = o.shape[0]
        b_on = n.shape[0]
        merged[key] = torch.cat([o, n], dim=0)
    return merged
