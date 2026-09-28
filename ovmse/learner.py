# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""QMIX learner with PyMARL-style episode batches (TD-lambda, masked losses)."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch

from ovmse.behaviour import BehaviourCloningMixin
from ovmse.cql import CQLLossMixin
from ovmse.ovm import OVMMixin
from ovmse.targets import build_td_lambda_targets
from qmix.core import QMIXBase
from qmix.networks import MarlcRNN, QMixer


class PaperQMIXLearner(
    QMIXBase,
    CQLLossMixin,
    BehaviourCloningMixin,
    OVMMixin,
):
    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        obs_dim: int,
        state_dim: int,
        *,
        linear_layer_dim: int = 64,
        recurrent_layer_dim: int = 64,
        mixer_embed_dim: int = 32,
        mixer_hyper_dim: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        td_lambda: float = 0.6,
        grad_norm_clip: float = 10.0,
        target_update_interval: int = 200,
        batch_size_run: int = 1,
        add_agent_id_to_obs: bool = True,
        raw_cql: bool = False,
        moderate_lambda: bool = False,
        cql_weight: float = 0.05,
        online_cql_weight: float = 0.0,
        num_ood_actions: int = 100,
        softmax_temp: float = 5.0,
        max_behaviour_filled_steps: int = 500_000,
        use_ovm: bool = False,
        mixing_ratio: float = 0.0,
        lambda_memory_end: float = 1.0,
        lambda_anneal_steps: int = 0,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.td_lambda = td_lambda
        self.grad_norm_clip = grad_norm_clip
        self.target_update_interval = target_update_interval
        self.batch_size_run = batch_size_run
        self.add_agent_id_to_obs = add_agent_id_to_obs
        self.raw_cql = raw_cql
        self.moderate_lambda = moderate_lambda
        self.cql_weight = cql_weight
        self.online_cql_weight = online_cql_weight
        self.num_ood_actions = num_ood_actions
        self.softmax_temp = softmax_temp
        self.max_behaviour_filled_steps = max_behaviour_filled_steps
        self.use_ovm = use_ovm
        self.mixing_ratio = mixing_ratio
        self.lambda_memory_end = lambda_memory_end
        self.lambda_anneal_steps = lambda_anneal_steps

        self.obs_dim = obs_dim
        self.hidden_dim = recurrent_layer_dim
        obs_in = obs_dim + n_actions
        if add_agent_id_to_obs:
            obs_in += n_agents

        self.q_network = MarlcRNN(obs_in, recurrent_layer_dim, n_actions).to(self.device)
        self.target_q_network = copy.deepcopy(self.q_network).to(self.device)
        self.mixer = QMixer(n_agents, state_dim, mixer_embed_dim, mixer_hyper_dim).to(
            self.device
        )
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)

        self.offline_q_network: Optional[MarlcRNN] = None
        self.offline_mixer: Optional[QMixer] = None

        params = list(self.q_network.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

        self.behaviour_network: Optional[MarlcRNN] = None
        self.behaviour_optimizer: Optional[torch.optim.Adam] = None
        if moderate_lambda:
            self.behaviour_network = MarlcRNN(obs_in, recurrent_layer_dim, n_actions).to(
                self.device
            )
            self.behaviour_optimizer = torch.optim.Adam(
                self.behaviour_network.parameters(), lr=learning_rate
            )

        self.episode_counter = 0
        self.last_target_update_episode = 0
        self.t_env = 0
        self._behaviour_trained = False
        self._bc_best_loss = 1e6
        self._bc_epochs_since_best = 0
        self.eval_rnn: Dict[str, torch.Tensor] = {}
        self.eval_last_actions: Dict[str, Optional[int]] = {}
        self.eval_offline_rnn = None
        self.eval_offline_last_actions = None
        self.training_step_ctr = 0

    def train_on_batch(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        cql_weight: Optional[float] = None,
        use_ovm: Optional[bool] = None,
        t_env: Optional[int] = None,
        episode_delta: int = 1,
    ) -> Dict[str, float]:
        cql_w = self.cql_weight if cql_weight is None else cql_weight
        ovm_on = self.use_ovm if use_ovm is None else use_ovm
        t_env = self.t_env if t_env is None else t_env

        actions = batch["actions"]
        avail = batch["avail_actions"]
        states = batch["state"]
        rewards = batch["reward"]
        terminated = batch["terminated"]
        filled = batch["filled"]

        term = terminated[:, :-1].float()
        if term.dim() == 3 and term.shape[2] > 1:
            term = term.max(dim=2, keepdim=True)[0]
        mask = filled[:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1.0 - term[:, :-1])
        mask_exp = mask

        with torch.no_grad():
            target_mac_out = self._forward_qs(batch, target=True)
        mac_out = self._forward_qs(batch, target=False)

        chosen_q = torch.gather(mac_out, 3, actions.long())
        mac_sel = torch.where(avail, mac_out, torch.tensor(-9999999.0, device=self.device))
        max_actions = torch.argmax(mac_sel, dim=3, keepdim=True)
        target_max_q = torch.gather(target_mac_out, 3, max_actions)

        chosen_tot, target_max_tot, mixed_rew = self._mixing(
            chosen_q, target_max_q, states, rewards
        )

        td_targets = build_td_lambda_targets(
            mixed_rew[:, :-1],
            terminated[:, :-1].max(dim=2, keepdim=True)[0],
            mask_exp,
            target_max_tot,
            self.gamma,
            self.td_lambda,
        ).detach()

        lam = self.get_lambda_memory(t_env)
        if ovm_on and self.offline_q_network is not None and self.offline_mixer is not None:
            loss = self._ovm_td_loss(
                batch, actions, states, chosen_tot, td_targets, mask_exp, lam
            )
        else:
            td_err = 0.5 * (chosen_tot[:, :-1] - td_targets) ** 2
            loss = (td_err * mask_exp).sum() / mask_exp.sum().clamp(min=1.0)

        cql_loss_val = 0.0
        if cql_w > 0:
            if self.raw_cql:
                cql_loss = self._compute_macql_loss(mac_out, batch, chosen_tot, mask_exp)
            else:
                cql_loss = self._compute_cfcql_loss(mac_out, batch, chosen_tot, mask_exp)
            loss = loss + cql_w * cql_loss
            cql_loss_val = float(cql_loss.detach().item())

        self.training_step_ctr += 1
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.tensor(0.0, device=self.device)
        if self.grad_norm_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.q_network.parameters()) + list(self.mixer.parameters()),
                self.grad_norm_clip,
            )
        self.optimizer.step()

        self.episode_counter += int(episode_delta)
        if (
            self.episode_counter - self.last_target_update_episode
        ) / self.target_update_interval >= 1.0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.last_target_update_episode = self.episode_counter

        mean_q = float(chosen_tot.detach().mean().item())
        return {
            "loss": float(loss.detach().item()),
            "cql_loss": cql_loss_val,
            "grad_norm": float(grad_norm.detach().item()),
            "mean_chosen_q_values": mean_q,
            "lambda_memory": lam,
        }

    def state_dicts(self) -> Dict[str, Any]:
        return {
            "q_network": self.q_network.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
        }
