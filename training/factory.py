# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Construct learner, dataset, environment, and logger from config."""

from __future__ import annotations

from typing import Any, Dict

import torch

from ackf.learner import ACKFLearner
from data.vault_dataset import EpisodeDataset, load_episode_dataset_from_vault
from data.vault_download import download_and_unzip_vault
from envs import get_environment
from fsr.learner import FSRLearner
from macal.learner import MaCalQLLearner
from macflow.learner import MACFlowLearner
from ovmse.learner import PaperQMIXLearner
from qmix.core import QMIXBase
from uot.learner import UOTLearner
from uot.support import RecordedJoints, StateSupport
from utils.logger import JsonlLogger, WandbLogger
from utils.paths import VAULT_DIR


def _build_qmix_cql(cfg: Dict[str, Any], dataset: EpisodeDataset) -> PaperQMIXLearner:
    """QMIX+CQL learner used by the OVMSE paper run."""
    s = dataset.stats
    sys_cfg = cfg["system"]
    use_ovm = bool(sys_cfg.get("use_ovm", False))
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
        use_ovm=use_ovm,
        mixing_ratio=float(sys_cfg.get("mixing_ratio", 0.0)),
        lambda_memory_end=float(sys_cfg.get("lambda_memory_end", 1.0)),
        lambda_anneal_steps=int(sys_cfg.get("lambda_anneal_steps", 0)),
        support_gate_mode=str(sys_cfg.get("support_gate_mode", "off")),
        support_cql_mode=str(sys_cfg.get("support_cql_mode", "off")),
        support_cql_normalize_mass=bool(sys_cfg.get("support_cql_normalize_mass", False)),
        support_gate_theta=float(sys_cfg.get("support_gate_theta", 1.0)),
        support_gate_temp=float(sys_cfg.get("support_gate_temp", 0.25)),
        support_softmax_temp=float(sys_cfg.get("support_softmax_temp", 1.0)),
        ovm_diag_train_stats=bool(sys_cfg.get("ovm_diag_train_stats", True)),
        ovm_diag_stride=int(sys_cfg.get("ovm_diag_stride", 4)),
        device=str(sys_cfg.get("device", "cuda")),
    )


def build_learner(cfg: Dict[str, Any], dataset: EpisodeDataset) -> QMIXBase:
    s = dataset.stats
    sys_cfg = cfg["system"]
    learner_type = str(cfg.get("learner_type", "ovmse_paper")).lower()
    if learner_type == "uot":
        support = RecordedJoints.from_episode_actions(
            dataset.actions, dataset.filled, s.n_actions
        )
        ranking_scope = str(sys_cfg.get("uot_ranking_scope", "joint")).lower()
        support_mode = str(sys_cfg.get("uot_support_mode", "joint")).lower()
        state_support = None
        if support_mode == "state":
            state_support = StateSupport(
                s.state_dim,
                seed=int(sys_cfg.get("uot_state_support_seed", cfg["seed"])),
            )
            state_support.to(torch.device(str(sys_cfg.get("device", "cuda"))))
            state_support.fit(
                dataset.state.to(state_support.device), dataset.filled.to(state_support.device)
            )
            state_stats = state_support.calibrate(
                quantile=float(sys_cfg.get("uot_state_support_quantile", 0.10))
            )
            print(f"UOT state support calibration: {state_stats}")
        print(
            f"UOT controls ranking_scope={ranking_scope} support_mode={support_mode} "
            f"marginal_mask={int(support.marginal_mask.sum().item())}/{support.marginal_mask.numel()}"
        )
        return UOTLearner(
            n_agents=s.n_agents,
            n_actions=s.n_actions,
            obs_dim=s.obs_dim,
            state_dim=s.state_dim,
            support=support,
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
            eta=float(sys_cfg.get("uot_eta", 0.1)),
            eta_drop_at=int(sys_cfg.get("uot_eta_drop_at", -1)),
            eta_decay_start=int(sys_cfg.get("uot_eta_decay_start", -1)),
            eta_decay_steps=int(sys_cfg.get("uot_eta_decay_steps", 0)),
            delta=float(sys_cfg.get("uot_delta", 0.1)),
            margin=float(sys_cfg.get("uot_margin", 0.1)),
            n_candidates=int(sys_cfg.get("uot_k", 8)),
            max_hamming=int(sys_cfg.get("uot_max_hamming", 2)),
            eps_min=float(sys_cfg.get("eps_min", 0.05)),
            ranking_scope=ranking_scope,
            support_mode=support_mode,
            state_support=state_support,
            device=str(sys_cfg.get("device", "cuda")),
        )
    if learner_type == "mac_flow":
        action_mode = str((cfg.get("task") or {}).get("action_mode", "discrete")).lower()
        continuous = action_mode == "continuous" or s.action_continuous
        act_dim = s.act_dim if continuous else s.n_actions
        return MACFlowLearner(
            n_agents=s.n_agents,
            obs_dim=s.obs_dim,
            n_actions=s.n_actions if not continuous else 0,
            act_dim=act_dim,
            action_mode="continuous" if continuous else "discrete",
            learning_rate=float(sys_cfg.get("learning_rate", 3e-4)),
            gamma=float(sys_cfg.get("discount", 0.99)),
            alpha_distill=float(sys_cfg.get("alpha_distill", 3.0)),
            flow_steps=int(sys_cfg.get("flow_steps", 10)),
            polyak_tau=float(sys_cfg.get("polyak_tau", 0.005)),
            grad_norm_clip=float(sys_cfg.get("grad_norm_clip", 0.0)),
            normalize_q=bool(sys_cfg.get("normalize_q", False)),
            device=str(sys_cfg.get("device", "cuda")),
            batch_size_run=int(cfg.get("batch_size_run", 1)),
            sequence_length=int(cfg.get("sequence_length", 20)),
            hidden_dims=tuple(int(h) for h in sys_cfg.get("hidden_dims", (256, 256, 256, 256))),
            actor_layer_norm=bool(sys_cfg.get("actor_layer_norm", False)),
            t_embed_frequencies=int(sys_cfg.get("t_embed_frequencies", 8)),
            adam_eps=float(sys_cfg.get("adam_eps", 1e-8)),
        )
    if learner_type == "ackf":
        action_mode = str((cfg.get("task") or {}).get("action_mode", "discrete")).lower()
        if action_mode != "discrete" or s.action_continuous:
            raise ValueError("ACKF supports discrete actions only")
        ackf = cfg.get("ackf") or {}
        lr = float(sys_cfg.get("learning_rate", 1e-3))
        return ACKFLearner(
            n_agents=s.n_agents,
            n_actions=s.n_actions,
            obs_dim=s.obs_dim,
            state_dim=s.state_dim,
            recurrent_layer_dim=int(sys_cfg.get("recurrent_layer_dim", 64)),
            mixer_embed_dim=int(sys_cfg.get("mixer_embed_dim", 32)),
            mixer_hyper_dim=int(sys_cfg.get("mixer_hyper_dim", 64)),
            learning_rate=lr,
            gamma=float(sys_cfg.get("discount", 0.99)),
            td_lambda=float(sys_cfg.get("td_lambda", 0.6)),
            grad_norm_clip=float(sys_cfg.get("grad_norm_clip", 10.0)),
            target_update_interval=int(sys_cfg.get("target_update_interval", 200)),
            batch_size_run=int(cfg.get("batch_size_run", 1)),
            add_agent_id_to_obs=bool(sys_cfg.get("add_agent_id_to_obs", True)),
            adapter_hidden_dim=int(ackf.get("adapter_hidden_dim", 64)),
            adapter_init_std=float(ackf.get("adapter_init_std", 1e-2)),
            gate_init=float(ackf.get("gate_init", 0.1)),
            gate_lr_scale=float(ackf.get("gate_lr_scale", 0.1)),
            norm_eps=float(ackf.get("norm_eps", 1e-5)),
            device=str(sys_cfg.get("device", "cuda")),
        )
    if learner_type == "fsr":
        fsr = cfg.get("fsr") or {}
        return FSRLearner(
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
            moderate_lambda=False,
            cql_weight=float(sys_cfg.get("cql_weight", 0.05)),
            online_cql_weight=float(sys_cfg.get("online_cql_weight", 0.0)),
            num_ood_actions=int(sys_cfg.get("num_ood_actions", 100)),
            softmax_temp=float(sys_cfg.get("softmax_temp", 5.0)),
            use_ovm=False,
            mixing_ratio=float(sys_cfg.get("mixing_ratio", 0.0)),
            lambda_memory_end=1.0,
            lambda_anneal_steps=0,
            ovm_diag_train_stats=bool(sys_cfg.get("ovm_diag_train_stats", True)),
            ovm_diag_stride=int(sys_cfg.get("ovm_diag_stride", 4)),
            device=str(sys_cfg.get("device", "cuda")),
            shared_flow=bool(fsr.get("shared_flow", False)),
            shape_critic=bool(fsr.get("shape_critic", True)),
            shape_weight=float(fsr.get("shape_weight", 0.1)),
            flow_hidden=tuple(int(h) for h in fsr.get("flow_hidden", (128, 128, 128))),
            encoder_hidden=int(fsr.get("encoder_hidden", 64)),
            flow_lr=float(fsr.get("flow_lr", 3e-4)),
            flow_steps=int(fsr.get("flow_steps", 8)),
            score_t=float(fsr.get("score_t", 0.9)),
            calib_quantile=float(fsr.get("calib_quantile", 0.10)),
            w_temperature=float(fsr.get("w_temperature", 1.0)),
            mix_fn=str(fsr.get("mix_fn", "linear")),
            mix_max=float(fsr.get("mix_max", 0.5)),
            prior_eta=float(fsr.get("prior_eta", 0.1)),
            prior_margin=float(fsr.get("prior_margin", 0.1)),
            prior_delta=float(fsr.get("prior_delta", 0.1)),
            visit_scale=float(fsr.get("visit_scale", 50.0)),
            visit_bins=int(fsr.get("visit_bins", 8)),
            rotate_explore=bool(fsr.get("rotate_explore", True)),
            rotate_period=int(fsr.get("rotate_period", 1)),
            logp_max_rows=int(fsr.get("logp_max_rows", 2048)),
            shape_max_rows=int(fsr.get("shape_max_rows", 256)),
            t_embed_frequencies=int(fsr.get("t_embed_frequencies", 8)),
        )
    if learner_type == "macal_ql":
        return MaCalQLLearner(
            n_agents=s.n_agents,
            n_actions=s.n_actions,
            obs_dim=s.obs_dim,
            state_dim=s.state_dim,
            recurrent_layer_dim=int(sys_cfg.get("recurrent_layer_dim", 64)),
            mixer_embed_dim=int(sys_cfg.get("mixer_embed_dim", 32)),
            mixer_hyper_dim=int(sys_cfg.get("mixer_hyper_dim", 64)),
            learning_rate=float(sys_cfg.get("learning_rate", 1e-3)),
            gamma=float(sys_cfg.get("discount", 0.99)),
            grad_norm_clip=float(sys_cfg.get("grad_norm_clip", 10.0)),
            target_update_interval=int(sys_cfg.get("target_update_interval", 200)),
            batch_size_run=int(cfg.get("batch_size_run", 1)),
            add_agent_id_to_obs=bool(sys_cfg.get("add_agent_id_to_obs", True)),
            num_ood_actions=int(sys_cfg.get("num_ood_actions", 10)),
            cql_weight=float(sys_cfg.get("cql_weight", 1.0)),
            dual_alpha=bool(sys_cfg.get("dual_alpha", True)),
            target_action_gap=float(sys_cfg.get("target_action_gap", 0.8)),
            alpha_lr=float(sys_cfg.get("alpha_lr", 1e-4)),
            alpha_init=float(sys_cfg.get("alpha_init", 1.0)),
            eps_min=float(sys_cfg.get("eps_min", 0.05)),
            device=str(sys_cfg.get("device", "cuda")),
        )
    if learner_type != "ovmse_paper":
        raise ValueError(f"Unknown learner_type {learner_type!r}")
    return _build_qmix_cql(cfg, dataset)


def load_vault_dataset(cfg: Dict[str, Any]) -> EpisodeDataset:
    task = cfg["task"]
    download_and_unzip_vault(task["source"], task["env"], task["scenario"])
    return load_episode_dataset_from_vault(
        task["source"],
        task["env"],
        task["scenario"],
        str(task["dataset"]),
        rel_dir=str(VAULT_DIR),
        device=str(cfg["system"].get("data_device", cfg["system"].get("device", "cuda"))),
        action_mode=str(task.get("action_mode", "discrete")),
    )


def make_env(cfg: Dict[str, Any], seed: int):
    task = cfg["task"]
    return get_environment(
        task["source"],
        task["env"],
        task["scenario"],
        seed=seed,
        action_mode=str(task.get("action_mode", "discrete")),
    )


def make_logger(cfg: Dict[str, Any]):
    log_path = cfg.get("log_path")
    if log_path:
        return JsonlLogger(str(log_path))
    return WandbLogger(
        project=cfg.get("wandb_project", "og-marl"),
        config={**cfg},
    )
