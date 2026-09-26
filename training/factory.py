# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Construct learner, dataset, environment, and logger from config."""

from __future__ import annotations

from typing import Any, Dict

from data.vault_dataset import EpisodeDataset, load_episode_dataset_from_vault
from data.vault_download import download_and_unzip_vault
from envs import get_environment
from ovmse.learner import PaperQMIXLearner
from utils.logger import JsonlLogger, WandbLogger
from utils.paths import VAULT_DIR


def build_learner(cfg: Dict[str, Any], dataset: EpisodeDataset) -> PaperQMIXLearner:
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
        rel_dir=str(VAULT_DIR),
        device=str(cfg["system"].get("device", "cuda")),
    )


def make_env(cfg: Dict[str, Any], seed: int):
    return get_environment(
        cfg["task"]["source"], cfg["task"]["env"], cfg["task"]["scenario"], seed=seed
    )


def make_logger(cfg: Dict[str, Any]):
    log_path = cfg.get("log_path")
    if log_path:
        return JsonlLogger(str(log_path))
    return WandbLogger(
        project=cfg.get("wandb_project", "og-marl"),
        config={**cfg},
    )
