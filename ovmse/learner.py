# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""QMIX learner with PyMARL-style episode batches (TD-lambda, masked losses)."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ovmse.behaviour import BehaviourCloningMixin
from ovmse.cql import CQLLossMixin
from ovmse.networks import MarlcRNN, QMixer
from ovmse.ovm import OVMMixin
from ovmse.policy import EvalPolicyMixin
from ovmse.targets import build_td_lambda_targets
from ovmse.tensor_utils import (
    expand_batch_and_agent_dim_of_time_major_sequence,
    merge_batch_and_agent_dim_of_time_major_sequence,
    switch_two_leading_dims,
)


class PaperQMIXLearner(
    CQLLossMixin,
    BehaviourCloningMixin,
    OVMMixin,
    EvalPolicyMixin,
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

    def _resets_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        filled = batch["filled"]
        term = batch["terminated"]
        if filled.dim() == 3:
            b, t, _ = filled.shape
        else:
            b, t, _, _ = filled.shape
        filled_1 = filled[..., 0] if filled.shape[-1] == 1 else filled.squeeze(-1)
        term_1 = term[..., 0] if term.shape[-1] == 1 else term.squeeze(-1)
        resets = torch.zeros(b, t, device=self.device, dtype=torch.bool)
        resets[:, 0] = True
        if t > 1:
            prev_filled = filled_1[:, :-1] > 0.5
            prev_term = term_1[:, :-1] > 0.5
            resets[:, 1:] = (~prev_filled) | prev_term
        return resets.unsqueeze(-1)

    def _augment_obs(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Match PyMARL order: obs, previous action one-hot, agent id."""
        b, t, n, _ = obs.shape
        last = obs.new_zeros(b, t, n, self.n_actions)
        if t > 1:
            prev = actions[:, :-1, :, 0].long().clamp(0, self.n_actions - 1)
            last[:, 1:] = F.one_hot(prev, self.n_actions).float()
        parts = [obs, last]
        if self.add_agent_id_to_obs:
            eye = torch.eye(n, device=obs.device, dtype=obs.dtype)
            parts.append(eye.view(1, 1, n, n).expand(b, t, n, n))
        return torch.cat(parts, dim=-1)

    def _unroll(self, net: torch.nn.Module, obs: torch.Tensor, resets: torch.Tensor) -> torch.Tensor:
        b, t, n, _ = obs.shape
        resets = resets.expand(b, t, n)
        obs_tm = merge_batch_and_agent_dim_of_time_major_sequence(switch_two_leading_dims(obs))
        resets_tm = merge_batch_and_agent_dim_of_time_major_sequence(
            switch_two_leading_dims(resets)
        )
        qs = self._unroll_reset_before_forward(net, obs_tm, resets_tm)
        return switch_two_leading_dims(expand_batch_and_agent_dim_of_time_major_sequence(qs, b, n))

    @staticmethod
    def _unroll_reset_before_forward(
        net: torch.nn.Module, inputs: torch.Tensor, resets: torch.Tensor
    ) -> torch.Tensor:
        """Zero the GRU hidden state before a reset step, then keep it for the next step.

        The shared MARLR unroll clears the hidden state after the forward, so a
        reset at t=0 discards that step and the next step starts from zero again.
        """
        time_steps, batch_size = inputs.shape[:2]
        device = inputs.device
        hidden_state = net.initial_state(batch_size, device)
        outputs = []
        for i in range(time_steps):
            reset_mask = resets[i].unsqueeze(0).unsqueeze(-1)
            hidden_state = torch.where(
                reset_mask, net.initial_state(batch_size, device), hidden_state
            )
            output, hidden_state = net(inputs[i], hidden_state)
            outputs.append(output)
        return torch.stack(outputs, dim=0)

    def _forward_qs(
        self, batch: Dict[str, torch.Tensor], target: bool = False
    ) -> torch.Tensor:
        net = self.target_q_network if target else self.q_network
        obs = self._augment_obs(batch["obs"], batch["actions"])
        resets = self._resets_from_batch(batch)
        return self._unroll(net, obs, resets)

    def _mixing(
        self,
        chosen_qs: torch.Tensor,
        target_max_qs: torch.Tensor,
        states: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen_tot = self.mixer(chosen_qs, states)
        with torch.no_grad():
            target_tot = self.target_mixer(target_max_qs, states)
        return chosen_tot, target_tot, rewards

    @torch.no_grad()
    def mix_agent_qs(
        self,
        agent_qs: List[float],
        state: np.ndarray,
        *,
        offline: bool = False,
    ) -> float:
        """Q_tot of one joint action. agent_qs follows sorted agent id order."""
        mixer = self.offline_mixer if offline else self.mixer
        if mixer is None:
            raise RuntimeError("offline mixer is not loaded")
        qs = torch.tensor(agent_qs, dtype=torch.float32, device=self.device).view(1, 1, -1)
        st = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, 1, -1)
        return float(mixer(qs, st).reshape(-1)[0].item())

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
