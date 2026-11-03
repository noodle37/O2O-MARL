# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""QMIX learner with PyMARL-style episode batches (TD-lambda, masked losses)."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch

from ovmse.behaviour import BehaviourCloningMixin
from ovmse.cql import CQLLossMixin
from ovmse.diagnostics import ovm_target_stats
from ovmse.ovm import OVMMixin, normalize_gate_mode
from ovmse.targets import build_td_lambda_targets
from qmix.core import QMIXBase
from qmix.networks import MarlcRNN, QMixer
from sapr.regularizer import SAPRPriorMixin


class PaperQMIXLearner(
    QMIXBase,
    CQLLossMixin,
    BehaviourCloningMixin,
    OVMMixin,
    SAPRPriorMixin,
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
        support_gate_mode: str = "off",
        support_cql_mode: str = "off",
        support_cql_normalize_mass: bool = False,
        support_gate_theta: float = 1.0,
        support_gate_temp: float = 0.25,
        support_softmax_temp: float = 1.0,
        ovm_diag_train_stats: bool = True,
        ovm_diag_stride: int = 4,
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
        self.support_gate_mode = normalize_gate_mode(support_gate_mode)
        self.support_cql_mode = normalize_gate_mode(support_cql_mode)
        self.support_cql_normalize_mass = bool(support_cql_normalize_mass)
        self.support_gate_theta = float(support_gate_theta)
        self.support_gate_temp = float(support_gate_temp)
        self.support_softmax_temp = float(support_softmax_temp)
        self.ovm_diag_train_stats = bool(ovm_diag_train_stats)
        self.ovm_diag_stride = max(1, int(ovm_diag_stride))

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
        self.init_sapr_prior()

    def train_on_batch(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        cql_weight: Optional[float] = None,
        use_ovm: Optional[bool] = None,
        t_env: Optional[int] = None,
        episode_delta: int = 1,
        td_row_mask: Optional[torch.Tensor] = None,
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
        # MU-APR composes each batch from offline reference rows and online rows;
        # only the online tail may contribute to the TD objective, while the KL
        # term still sees the full mixed history.
        if td_row_mask is not None:
            td_rows = td_row_mask.to(mask.device, mask.dtype).view(-1, 1, 1)
            td_mask = mask_exp * td_rows
        else:
            td_mask = mask_exp

        with torch.no_grad():
            target_mac_out = self._forward_qs(batch, target=True)
        mac_out = self._forward_qs(batch, target=False)

        chosen_q = torch.gather(mac_out, 3, actions.long())
        mac_sel = torch.where(avail, mac_out, torch.tensor(-9999999.0, device=self.device))
        max_actions = self._select_bootstrap_actions(mac_sel, avail)
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
        ovm_stats: Dict[str, float] = {}
        gate_stats: Dict[str, float] = {}
        offline_ready = self.offline_q_network is not None and self.offline_mixer is not None
        # OVM-on: off_tot is already computed for the loss, so the stats are free.
        # OVM-off: recompute on a stride so the extra frozen unroll stays cheap.
        stride_hit = (self.training_step_ctr % self.ovm_diag_stride) == 0
        off_out = None
        if ovm_on and offline_ready:
            off_out = self._offline_unroll(batch)
            off_tot = self._offline_chosen_tot(batch, actions, states, off_out=off_out)
            lam_eff = lam
            if self.support_gate_mode != "off":
                gate = self.support_gate(batch, off_out=off_out)
                lam_eff = lam * gate
                gate_mask = mask_exp > 0.5
                gate_stats["support/gate_mean"] = float(gate[gate_mask].mean().item())
                gate_stats["support/gate_frac_above_half"] = float(
                    (gate[gate_mask] > 0.5).float().mean().item()
                )
            loss = self._ovm_td_loss(
                batch, actions, states, chosen_tot, td_targets, td_mask, lam_eff, off_tot=off_tot
            )
            if self.ovm_diag_train_stats:
                ovm_stats = ovm_target_stats(
                    off_tot,
                    td_targets,
                    mask_exp,
                    lam_eff,
                    chosen_tot=chosen_tot,
                    prefix="train_ovm/",
                )
        else:
            td_err = 0.5 * (chosen_tot[:, :-1] - td_targets) ** 2
            loss = (td_err * td_mask).sum() / td_mask.sum().clamp(min=1.0)
            if self.ovm_diag_train_stats and offline_ready and stride_hit:
                off_tot = self._offline_chosen_tot(batch, actions, states)
                ovm_stats = ovm_target_stats(
                    off_tot,
                    td_targets,
                    mask_exp,
                    lam,
                    chosen_tot=chosen_tot,
                    prefix="train_ovm/",
                )

        cql_loss_val = 0.0
        if cql_w > 0:
            # Second reuse channel: keep the conservatism where the old team's
            # support still covers the executed joint action and release it
            # where it does not. The weighted loss is renormalised, so this arm
            # differs from a tuned global cql_weight only in WHERE it applies.
            cql_weight_map = None
            if self.support_cql_mode != "off" and offline_ready:
                cql_weight_map = self.support_gate(
                    batch, off_out=off_out, mode=self.support_cql_mode
                )
                if self.support_cql_normalize_mass:
                    # Per-batch normalization: keep the same total CQL mass as
                    # uniform (mean weight == 1), so any benefit can only come
                    # from WHERE the penalty is placed.
                    masked = cql_weight_map[mask_exp > 0.5]
                    mean_w = masked.mean().clamp(min=1e-6)
                    cql_weight_map = cql_weight_map / mean_w
                gate_stats["support/cql_weight_mean"] = float(
                    cql_weight_map[mask_exp > 0.5].mean().item()
                )
            if self.raw_cql:
                cql_loss = self._compute_macql_loss(
                    mac_out, batch, chosen_tot, mask_exp, weight=cql_weight_map
                )
            else:
                cql_loss = self._compute_cfcql_loss(
                    mac_out, batch, chosen_tot, mask_exp, weight=cql_weight_map
                )
            loss = loss + cql_w * cql_loss
            cql_loss_val = float(cql_loss.detach().item())

        prior_stats: Dict[str, float] = {}
        prior_w = float(getattr(self, "prior_kl_weight", 0.0))
        if prior_w > 0.0:
            if not offline_ready:
                raise RuntimeError(
                    "prior KL weight is positive but no frozen offline network is loaded"
                )
            if off_out is None:
                off_out = self._offline_unroll(batch)
            prior_kl_value = self.prior_kl_loss(batch, mac_out, offline_out=off_out)
            loss = loss + prior_w * prior_kl_value
            prior_stats = {
                "sapr/prior_kl": float(prior_kl_value.detach().item()),
                "sapr/prior_weight": prior_w,
                "muapr/kl_train": float(prior_kl_value.detach().item()),
                "muapr/lambda": prior_w,
            }

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
            **ovm_stats,
            **gate_stats,
            **prior_stats,
            **getattr(self, "_bootstrap_stats", {}),
        }

    def _select_bootstrap_actions(
        self, mac_sel: torch.Tensor, avail: torch.Tensor
    ) -> torch.Tensor:
        """Action whose target Q bootstraps the TD target. Greedy by default.

        MaSO2 overrides this with a perturbed legal action. Other learners keep
        the double-Q argmax, including subclasses that do not override it.
        """
        self._bootstrap_stats = {}
        return torch.argmax(mac_sel, dim=3, keepdim=True)

    def state_dicts(self) -> Dict[str, Any]:
        return {
            "q_network": self.q_network.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
        }
