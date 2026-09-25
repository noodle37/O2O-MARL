# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""QMIX learner with PyMARL-style episode batches (TD-lambda, masked losses)."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from og_marl.baselines.torch_systems.networks import QMixer
from og_marl.baselines.torch_systems.utils import (
    batch_concat_agent_id_to_obs,
    concat_agent_id_to_obs,
    expand_batch_and_agent_dim_of_time_major_sequence,
    gather,
    merge_batch_and_agent_dim_of_time_major_sequence,
    switch_two_leading_dims,
)
from og_marl.vault_utils.vault_episodes import EpisodeDataset


class MarlcRNN(torch.nn.Module):
    """PyMARL2 n_rnn: ReLU-FC, GRUCell, linear Q head. No activation after the GRU."""

    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.rnn = torch.nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, n_actions)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def forward(
        self, x: torch.Tensor, hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        squeeze_time = x.dim() == 3
        if squeeze_time:
            x = x.squeeze(0)
        if hidden_state is None:
            hidden_state = self.initial_state(x.shape[0], x.device)
        hidden = hidden_state.reshape(x.shape[0], self.hidden_dim)
        hidden = self.rnn(F.relu(self.fc1(x)), hidden)
        q = self.fc2(hidden)
        new_hidden = hidden.unsqueeze(0)
        if squeeze_time:
            q = q.unsqueeze(0)
        return q, new_hidden


def build_td_lambda_targets(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    mask: torch.Tensor,
    target_qs: torch.Tensor,
    gamma: float,
    td_lambda: float,
) -> torch.Tensor:
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1.0 - terminated.sum(dim=1).clamp(max=1.0))
    for t in range(ret.shape[1] - 2, -1, -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] * (
            rewards[:, t]
            + (1.0 - td_lambda) * gamma * target_qs[:, t + 1] * (1.0 - terminated[:, t])
        )
    return ret[:, :-1]


class PaperQMIXLearner:
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

        self.offline_q_network: Optional[DeepRNN] = None
        self.offline_mixer: Optional[QMixer] = None

        params = list(self.q_network.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

        self.behaviour_network: Optional[DeepRNN] = None
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

    def load_offline_for_ovm(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.offline_q_network = copy.deepcopy(self.q_network)
        self.offline_mixer = copy.deepcopy(self.mixer)
        self.offline_q_network.load_state_dict(ckpt["q_network"])
        self.offline_mixer.load_state_dict(ckpt["mixer"])
        self.offline_q_network.eval()
        self.offline_mixer.eval()
        for p in self.offline_q_network.parameters():
            p.requires_grad = False
        for p in self.offline_mixer.parameters():
            p.requires_grad = False

    def get_lambda_memory(self, t_env: int) -> float:
        if self.lambda_anneal_steps <= 0 or self.lambda_memory_end >= 1.0:
            return 1.0
        t = min(t_env, self.lambda_anneal_steps)
        return max(
            self.lambda_memory_end,
            1.0 - (1.0 - self.lambda_memory_end) / self.lambda_anneal_steps * t,
        )

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

    def _compute_macql_loss(
        self,
        mac_out: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        chosen_qtotals: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        sample_n = self.num_ood_actions
        actions = batch["actions"]
        avail = batch["avail_actions"]
        states = batch["state"]
        bs, ts, n_agents, n_actions = mac_out.shape

        repeat_avail = avail[:, :-1].unsqueeze(0).expand(sample_n, -1, -1, -1, -1)
        chosen_if_avail = torch.gather(
            repeat_avail, dim=-1, index=torch.randint(
                0, n_actions, (sample_n, bs, ts - 1, n_agents, 1), device=self.device
            )
        ).min(-2)[0]

        repeat_mac = mac_out[:, :-1].unsqueeze(0).expand(sample_n, -1, -1, -1, -1)
        random_actions = torch.randint(
            0, n_actions, (sample_n, bs, ts - 1, n_agents, 1), device=self.device
        )
        random_q = torch.gather(repeat_mac, dim=-1, index=random_actions).squeeze(-1)
        repeat_state = states[:, :-1].unsqueeze(0).expand(sample_n, -1, -1, -1)
        random_q = random_q.reshape(sample_n * bs, ts - 1, n_agents)
        repeat_state = repeat_state.reshape(sample_n * bs, ts - 1, repeat_state.shape[-1])
        random_qtotal = self.mixer(random_q, repeat_state).view(sample_n, bs, ts - 1, 1)
        negative = torch.logsumexp(random_qtotal * chosen_if_avail, dim=0)
        dataset_expec = chosen_qtotals[:, :-1]
        cql_term = (negative - dataset_expec) * mask
        return cql_term.sum() / mask.sum().clamp(min=1.0)

    def _compute_cfcql_loss(
        self,
        mac_out: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        chosen_qtotals: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        actions = batch["actions"]
        avail = batch["avail_actions"]
        states = batch["state"]
        b, t, n, a = mac_out.shape
        qs = mac_out[:, :-1]
        st = states[:, :-1]
        dataset_actions = actions[:, :-1].long()
        legals = avail[:, :-1]

        negative_terms = []
        ratio_terms = []
        for agent_idx in range(n):
            dataset_q = torch.gather(qs, dim=3, index=dataset_actions).squeeze(-1)
            agent_q_cf = dataset_q.unsqueeze(2).expand(b, t - 1, a, n).clone()
            agent_q_cf[:, :, :, agent_idx] = qs[:, :, agent_idx, :]
            st_exp = st.unsqueeze(2).expand(b, t - 1, a, st.shape[-1])
            q_in = agent_q_cf.reshape(b * (t - 1) * a, n)
            s_in = st_exp.reshape(b * (t - 1) * a, st.shape[-1])
            q_tot = self.mixer(q_in.unsqueeze(1), s_in.unsqueeze(1)).reshape(b, t - 1, a)
            legal_agent = legals[:, :, agent_idx, :]
            q_tot = torch.where(legal_agent, q_tot, torch.tensor(-1e10, device=self.device))
            negative_terms.append(torch.logsumexp(q_tot, dim=2, keepdim=True))
            if self.moderate_lambda and self.behaviour_network is not None:
                mu_prob = F.softmax(q_tot, dim=2)
                beta_prob = self._behaviour_probs_for_agent(batch, agent_idx)[:, :-1]
                kl = F.kl_div(
                    torch.log(beta_prob + 1e-5),
                    mu_prob + 1e-5,
                    reduction="none",
                )
                kl = (kl * legal_agent.float()).sum(dim=2, keepdim=True)
                ratio_terms.append(kl)

        negative_sampling = torch.cat(negative_terms, dim=-1)
        if self.moderate_lambda and ratio_terms:
            ratio = torch.cat(ratio_terms, dim=-1)
            if self.softmax_temp >= 100:
                lambda_mask = F.one_hot(torch.argmax(ratio, dim=-1), num_classes=n).float()
            else:
                lambda_mask = F.softmax(ratio * self.softmax_temp, dim=-1).detach()
            negative_sampling = (negative_sampling * lambda_mask).sum(dim=-1, keepdim=True)
        else:
            negative_sampling = negative_sampling.mean(dim=-1, keepdim=True)
        cql_term = (negative_sampling - chosen_qtotals[:, :-1]) * mask
        return cql_term.sum() / mask.sum().clamp(min=1.0)

    def _behaviour_probs_for_agent(self, batch: Dict[str, torch.Tensor], agent_idx: int) -> torch.Tensor:
        assert self.behaviour_network is not None
        obs = self._augment_obs(batch["obs"], batch["actions"])
        with torch.no_grad():
            logits = self._unroll(self.behaviour_network, obs, self._resets_from_batch(batch))
            # Match MARLC behaviour_mac.forward: illegal logits are -1e5 before softmax.
            logits = logits.masked_fill(~batch["avail_actions"], -1e5)
            return F.softmax(logits[:, :, agent_idx, :], dim=-1)

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
            with torch.no_grad():
                off_out = self._unroll(
                    self.offline_q_network,
                    self._augment_obs(batch["obs"], batch["actions"]),
                    self._resets_from_batch(batch),
                )
                off_chosen = torch.gather(off_out[:, :-1], 3, actions[:, :-1].long())
                off_tot = self.offline_mixer(off_chosen, states[:, :-1])
            ovm_targets = torch.maximum(off_tot, td_targets)
            td_err = (chosen_tot[:, :-1] - td_targets) ** 2
            ovm_err = (chosen_tot[:, :-1] - ovm_targets) ** 2
            loss_td = ((1.0 - lam) * td_err + lam * ovm_err) * mask_exp
            loss = 0.5 * loss_td.sum() / mask_exp.sum().clamp(min=1.0)
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

    def train_behaviour(
        self,
        dataset: EpisodeDataset,
        cache_path: Optional[str] = None,
        batch_size: int = 128,
    ) -> None:
        if cache_path and os.path.isdir(cache_path):
            assert self.behaviour_network is not None
            self.behaviour_network.load_state_dict(
                torch.load(os.path.join(cache_path, "behaviour.pt"), map_location=self.device)
            )
            self.behaviour_network.eval()
            for p in self.behaviour_network.parameters():
                p.requires_grad = False
            self._behaviour_trained = True
            return

        assert self.behaviour_network is not None
        assert self.behaviour_optimizer is not None
        filled_budget = 0
        rng = np.random.default_rng(0)
        self.behaviour_network.train()
        self._bc_best_loss = 1e6
        self._bc_epochs_since_best = 0
        best_state = copy.deepcopy(self.behaviour_network.state_dict())
        while filled_budget < self.max_behaviour_filled_steps:
            batch = dataset.sample(batch_size, rng)
            actions = batch["actions"][:, :-1].long()
            avail = batch["avail_actions"][:, :-1]
            terminated = batch["terminated"][:, :-1].float()
            if terminated.dim() == 3 and terminated.shape[-1] > 1:
                terminated = terminated.max(dim=-1, keepdim=True)[0]
            mask = batch["filled"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1.0 - terminated[:, :-1])
            obs = self._augment_obs(batch["obs"], batch["actions"])
            logits = self._unroll(self.behaviour_network, obs, self._resets_from_batch(batch))[:, :-1]
            logits = logits.masked_fill(~avail, -1e5)
            pi = F.softmax(logits, dim=-1)
            pi = pi.masked_fill(~avail, 1e-10)
            log_pi = torch.log(pi.clamp(min=1e-10))
            action_lp = torch.gather(log_pi, 3, actions)
            # MARLC repeats the mask over agents. Broadcasting without that
            # puts N in the numerator only and scales the loss by n_agents.
            mask_agent = mask.unsqueeze(2).expand_as(action_lp)
            loss = -(action_lp * mask_agent).sum() / mask_agent.sum().clamp(min=1.0)
            self.behaviour_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.behaviour_network.parameters(), self.grad_norm_clip
            )
            self.behaviour_optimizer.step()
            filled_budget += int(batch["filled"][:, :-1].sum().item())
            loss_val = float(loss.detach().item())
            if loss_val < self._bc_best_loss:
                self._bc_best_loss = loss_val
                self._bc_epochs_since_best = 0
                best_state = copy.deepcopy(self.behaviour_network.state_dict())
            else:
                self._bc_epochs_since_best += 1
            if self._bc_epochs_since_best > 20:
                break
        self.behaviour_network.load_state_dict(best_state)
        self.behaviour_network.eval()
        for p in self.behaviour_network.parameters():
            p.requires_grad = False
        if cache_path:
            os.makedirs(cache_path, exist_ok=True)
            torch.save(
                self.behaviour_network.state_dict(),
                os.path.join(cache_path, "behaviour.pt"),
            )
        self._behaviour_trained = True

    def new_episode_state(self) -> Dict[str, Any]:
        return {
            "rnn": {
                f"agent_{i}": self.q_network.initial_state(1, self.device)
                for i in range(self.n_agents)
            },
            "last": {},
        }

    def reset_eval_hidden(self) -> None:
        state = self.new_episode_state()
        self.eval_rnn = state["rnn"]
        self.eval_last_actions = state["last"]

    def select_actions_eval(
        self,
        observations: Dict[str, np.ndarray],
        legals: Dict[str, np.ndarray],
        eps: float,
        state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        if state is None:
            rnn_state = self.eval_rnn
            last_actions = self.eval_last_actions
        else:
            rnn_state = state["rnn"]
            last_actions = state["last"]
        epsilon_dec = 0.0 if eps <= 0.0 else eps / self.n_agents
        actions = {}
        for i, agent in enumerate(sorted(observations.keys())):
            obs = torch.from_numpy(observations[agent]).float().to(self.device)
            last_idx = last_actions.get(agent)
            if last_idx is None:
                last_oh = torch.zeros(self.n_actions, device=self.device)
            else:
                last_oh = F.one_hot(
                    torch.tensor(int(last_idx), device=self.device), self.n_actions
                ).float()
            pieces = [obs, last_oh]
            if self.add_agent_id_to_obs:
                pieces.append(F.one_hot(torch.tensor(i, device=self.device), self.n_agents).float())
            obs_in = torch.cat(pieces, dim=-1).unsqueeze(0)
            q, rnn_state[agent] = self.q_network(obs_in, rnn_state.get(agent))
            legal = torch.from_numpy(legals[agent]).bool().to(self.device)
            q = q[0].masked_fill(~legal, -1e9)
            if np.random.rand() < epsilon_dec:
                legal_idx = torch.nonzero(legal, as_tuple=False).squeeze(-1)
                a = int(legal_idx[torch.randint(legal_idx.numel(), (1,))].item())
            else:
                a = int(torch.argmax(q).item())
            last_actions[agent] = a
            actions[agent] = a
        return actions

    def state_dicts(self) -> Dict[str, Any]:
        return {
            "q_network": self.q_network.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
        }
