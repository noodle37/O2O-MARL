# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""MaCal-QL learner: QMIX one-step TD + Cal-QL calibrated conservative loss on Q_tot."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch

from macal.calql import CalQLLossMixin
from qmix.core import QMIXBase
from qmix.networks import MarlcRNN, QMixer


class MaCalQLLearner(QMIXBase, CalQLLossMixin):
    """Cal-QL adapted to QMIX (MaCal-QL).

    TD loss is one-step Q-learning with the IGM-decomposed target max_u' Q_tot
    (per-agent argmax of the target agent nets). The conservative penalty is
    Cal-QL's calibrated push-down applied to the joint Q_tot; alpha is either
    fixed or learned by dual gradient descent against a target action gap.
    """

    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        obs_dim: int,
        state_dim: int,
        *,
        recurrent_layer_dim: int = 64,
        mixer_embed_dim: int = 32,
        mixer_hyper_dim: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        grad_norm_clip: float = 10.0,
        target_update_interval: int = 200,
        batch_size_run: int = 1,
        add_agent_id_to_obs: bool = True,
        num_ood_actions: int = 10,
        cql_weight: float = 1.0,
        dual_alpha: bool = True,
        target_action_gap: float = 0.8,
        alpha_lr: float = 1e-4,
        alpha_init: float = 1.0,
        eps_min: float = 0.05,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.grad_norm_clip = grad_norm_clip
        self.target_update_interval = target_update_interval
        self.batch_size_run = batch_size_run
        self.add_agent_id_to_obs = add_agent_id_to_obs
        self.num_ood_actions = num_ood_actions
        self.cql_weight = cql_weight
        self.dual_alpha = dual_alpha
        self.target_action_gap = target_action_gap
        self.eps_min = eps_min
        self.use_ovm = False
        self.moderate_lambda = False

        self.obs_dim = obs_dim
        obs_in = obs_dim + n_actions
        if add_agent_id_to_obs:
            obs_in += n_agents

        self.q_network = MarlcRNN(obs_in, recurrent_layer_dim, n_actions).to(self.device)
        self.target_q_network = copy.deepcopy(self.q_network).to(self.device)
        self.mixer = QMixer(n_agents, state_dim, mixer_embed_dim, mixer_hyper_dim).to(
            self.device
        )
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)

        self.offline_q_network = None
        self.offline_mixer = None

        params = list(self.q_network.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

        if dual_alpha:
            self.log_alpha = torch.tensor(
                [torch.log(torch.tensor(float(alpha_init)))],
                device=self.device,
                requires_grad=True,
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None

        self.episode_counter = 0
        self.last_target_update_episode = 0
        self.t_env = 0
        self.eval_rnn: Dict[str, torch.Tensor] = {}
        self.eval_last_actions: Dict[str, Optional[int]] = {}
        self.eval_offline_rnn = None
        self.eval_offline_last_actions = None
        self.training_step_ctr = 0

    def train_on_batch(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        t_env: Optional[int] = None,
        episode_delta: int = 1,
        eps: Optional[float] = None,
    ) -> Dict[str, float]:
        t_env = self.t_env if t_env is None else t_env
        eps = self.eps_min if eps is None else float(eps)

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

        with torch.no_grad():
            target_mac_out = self._forward_qs(batch, target=True)
        mac_out = self._forward_qs(batch, target=False)

        chosen_q = torch.gather(mac_out, 3, actions.long())
        chosen_tot = self.mixer(chosen_q, states)

        # One-step target: y = r + gamma (1 - done) max_u' Qbar_tot(tau', u'),
        # with the max IGM-decomposed into per-agent argmax of the target nets.
        with torch.no_grad():
            target_sel = torch.where(
                avail, target_mac_out, torch.tensor(-9999999.0, device=self.device)
            )
            target_max_q = torch.gather(
                target_mac_out, 3, target_sel.argmax(dim=3, keepdim=True)
            )
            target_max_tot = self.target_mixer(target_max_q, states)

        td_targets = rewards[:, :-1] + self.gamma * (1.0 - term) * target_max_tot[:, 1:]
        td_err = 0.5 * (chosen_tot[:, :-1] - td_targets) ** 2
        td_loss = (td_err * mask).sum() / mask.sum().clamp(min=1.0)

        cql_loss, cql_diff, mc_mean = self._compute_calql_loss(
            mac_out, batch, chosen_tot, mask, eps
        )

        if self.dual_alpha:
            alpha = self.log_alpha.exp().detach()
            loss = td_loss + alpha * cql_loss
            # Dual ascent: grow alpha while push-down minus push-up exceeds the gap.
            alpha_loss = -(self.log_alpha.exp() * (cql_diff - self.target_action_gap))
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha = torch.tensor(self.cql_weight, device=self.device)
            loss = td_loss + self.cql_weight * cql_loss

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

        return {
            "loss": float(loss.detach().item()),
            "td_loss": float(td_loss.detach().item()),
            "cql_loss": float(cql_loss.detach().item()),
            "cql_diff": float(cql_diff.item()),
            "alpha": float(alpha.item()),
            "grad_norm": float(grad_norm.detach().item()),
            "mean_chosen_q_values": float(chosen_tot.detach().mean().item()),
            "mc_return_mean": float(mc_mean.item()),
        }

    def state_dicts(self) -> Dict[str, Any]:
        return {
            "q_network": self.q_network.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
        }
