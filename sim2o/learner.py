# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Sim2O: OMIGA offline backbone + Coordinated Beam Search online adaptation."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from qmix.core import QMIXBase
from qmix.networks import MarlcRNN
from sim2o.cbs import coordinated_beam_search
from sim2o.networks import OMIGAMixer


class Sim2OLearner(QMIXBase):
    """Offline OMIGA training and online joint-action composition.

    Offline: V, Q and policy are trained with OMIGA's in-sample objectives.
    Online: the offline policy is frozen as the prior; CBS composes its
    proposals with the online policy and scores them with the centralized
    mixer. Q/V/policy keep training on mixed offline and online batches.
    """

    checkpoint_stem = "sim2o"

    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        obs_dim: int,
        state_dim: int,
        *,
        recurrent_layer_dim: int = 64,
        mixer_hyper_dim: int = 64,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        grad_norm_clip: float = 10.0,
        target_update_interval: int = 200,
        batch_size_run: int = 1,
        add_agent_id_to_obs: bool = True,
        alpha: float = 10.0,
        beam_width: int = 5,
        temperature: float = 5.0,
        target_tau: float = 0.005,
        eval_greedy: bool = True,
        random_agent_order: bool = True,
        seed: int = 1,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.grad_norm_clip = grad_norm_clip
        self.target_update_interval = target_update_interval
        self.batch_size_run = batch_size_run
        self.add_agent_id_to_obs = add_agent_id_to_obs
        self.alpha = float(alpha)
        self.beam_width = int(beam_width)
        self.temperature = float(temperature)
        self.target_tau = float(target_tau)
        self.eval_greedy = bool(eval_greedy)
        self.random_agent_order = bool(random_agent_order)

        # Compatibility with the shared offline/online loops.
        self.cql_weight = 0.0
        self.online_cql_weight = 0.0
        self.moderate_lambda = False
        self.use_ovm = False
        self.mixing_ratio = 0.5
        self.hidden_dim = recurrent_layer_dim

        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        obs_in = obs_dim + n_actions
        if add_agent_id_to_obs:
            obs_in += n_agents

        self.q_network = MarlcRNN(obs_in, recurrent_layer_dim, n_actions).to(self.device)
        self.target_q_network = copy.deepcopy(self.q_network).to(self.device)
        self.v_network = MarlcRNN(obs_in, recurrent_layer_dim, 1).to(self.device)
        self.target_v_network = copy.deepcopy(self.v_network).to(self.device)
        self.policy_network = MarlcRNN(obs_in, recurrent_layer_dim, n_actions).to(self.device)
        self.frozen_policy_network: Optional[MarlcRNN] = None

        self.mixer = OMIGAMixer(n_agents, state_dim, mixer_hyper_dim).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.q_network.parameters()) + list(self.mixer.parameters()),
            lr=learning_rate,
        )
        self.v_optimizer = torch.optim.Adam(self.v_network.parameters(), lr=learning_rate)
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=learning_rate
        )

        self.episode_counter = 0
        self.last_target_update_episode = 0
        self.training_step_ctr = 0
        self.t_env = 0

        self.eval_rnn: Dict[str, torch.Tensor] = {}
        self.eval_last_actions: Dict[str, Optional[int]] = {}
        self.eval_offline_rnn = None
        self.eval_offline_last_actions = None
        self.eval_q_rnn: Dict[str, torch.Tensor] = {}
        self.eval_policy_rnn: Dict[str, torch.Tensor] = {}
        self.eval_prior_rnn: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ offline
    def train_on_batch(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        cql_weight: Optional[float] = None,
        use_ovm: Optional[bool] = None,
        t_env: Optional[int] = None,
        episode_delta: int = 1,
        **_ignored: Any,
    ) -> Dict[str, float]:
        if t_env is not None:
            self.t_env = int(t_env)
        actions = batch["actions"].long()
        avail = batch["avail_actions"]
        states = batch["state"]
        rewards = batch["reward"]
        terminated = batch["terminated"]
        filled = batch["filled"]

        mask = filled[:, :-1].float()
        term = terminated[:, :-1].float()
        if term.dim() == 3 and term.shape[2] > 1:
            term = term.max(dim=2, keepdim=True)[0]
        mask_n = mask.expand(-1, -1, self.n_agents)

        obs_aug = self._augment_obs(batch["obs"], actions)
        resets = self._resets_from_batch(batch)

        q_all = self._unroll(self.q_network, obs_aug, resets)
        q_chosen = torch.gather(q_all, 3, actions).squeeze(-1)
        w, b = self.mixer.weights(states)
        q_tot = (w * q_chosen).sum(dim=-1, keepdim=True) + b

        v_all = self._unroll(self.v_network, obs_aug, resets).squeeze(-1)
        v_tot = (w * v_all).sum(dim=-1, keepdim=True) + b

        with torch.no_grad():
            v_next_all = self._unroll(self.target_v_network, obs_aug, resets).squeeze(-1)
            w_next, b_next = self.target_mixer.weights(states)
            v_tot_next = (w_next[:, 1:] * v_next_all[:, 1:]).sum(
                dim=-1, keepdim=True
            ) + b_next[:, 1:]

            q_t_all = self._unroll(self.target_q_network, obs_aug, resets)
            q_t_chosen = torch.gather(q_t_all, 3, actions).squeeze(-1)
            w_t, _ = self.target_mixer.weights(states)

        td_target = rewards[:, :-1, 0] + self.gamma * (1.0 - term[:, :, 0]) * v_tot_next[:, :, 0]
        q_pred = q_tot[:, :-1, 0]
        q_loss = (
            ((q_pred - td_target.detach()) ** 2) * mask[:, :, 0]
        ).sum() / mask[:, :, 0].sum().clamp(min=1.0)

        z = (w_t / self.alpha) * (q_t_chosen - v_all)
        z = z.clamp(-10.0, 10.0)
        z_steps = z[:, :-1]
        mask_bool = mask_n.bool()
        if mask_bool.any():
            max_z = z_steps[mask_bool].max().clamp(min=-1.0).detach()
        else:
            max_z = torch.zeros((), device=self.device)

        v_loss_elem = torch.exp(z_steps - max_z)
        v_loss_elem = v_loss_elem + torch.exp(-max_z) * w_t[:, :-1] * v_all[:, :-1] / self.alpha
        v_loss = (v_loss_elem * mask_n).sum() / mask_n.sum().clamp(min=1.0)

        logits = self._unroll(self.policy_network, obs_aug, resets)
        logits = logits.masked_fill(~avail, -1e9)
        logp = torch.gather(F.log_softmax(logits, dim=-1), 3, actions).squeeze(-1)
        exp_a = torch.exp(z_steps.detach())
        policy_loss = (
            (-exp_a * logp[:, :-1]) * mask_n
        ).sum() / mask_n.sum().clamp(min=1.0)

        self.training_step_ctr += 1
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_gn = self._clip(self.policy_network.parameters())
        self.policy_optimizer.step()

        self.optimizer.zero_grad()
        q_loss.backward()
        q_gn = self._clip(
            list(self.q_network.parameters()) + list(self.mixer.parameters())
        )
        self.optimizer.step()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        v_gn = self._clip(self.v_network.parameters())
        self.v_optimizer.step()

        self.episode_counter += int(episode_delta)
        if self.target_tau > 0:
            self._soft_update(self.target_tau)
            self.last_target_update_episode = self.episode_counter
        elif (
            self.episode_counter - self.last_target_update_episode
        ) / max(1, self.target_update_interval) >= 1.0:
            self._hard_update()
            self.last_target_update_episode = self.episode_counter

        return {
            "loss": float((q_loss + v_loss + policy_loss).detach().item()),
            "q_loss": float(q_loss.detach().item()),
            "v_loss": float(v_loss.detach().item()),
            "policy_loss": float(policy_loss.detach().item()),
            "q_tot_mean": float(q_tot.detach().mean().item()),
            "v_tot_mean": float(v_tot.detach().mean().item()),
            "mixer_w_mean": float(w.detach().mean().item()),
            "adv_weight_mean": float(exp_a.detach().mean().item()),
            "grad_norm_q": q_gn,
            "grad_norm_v": v_gn,
            "grad_norm_policy": policy_gn,
            "sim2o/target_tau": self.target_tau,
        }

    def _clip(self, params) -> float:
        if self.grad_norm_clip <= 0:
            return 0.0
        return float(torch.nn.utils.clip_grad_norm_(params, self.grad_norm_clip).item())

    @torch.no_grad()
    def _soft_update(self, tau: float) -> None:
        for source, target in (
            (self.q_network, self.target_q_network),
            (self.mixer, self.target_mixer),
            (self.v_network, self.target_v_network),
        ):
            for param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    @torch.no_grad()
    def _hard_update(self) -> None:
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_v_network.load_state_dict(self.v_network.state_dict())

    # ------------------------------------------------------------------- online
    def _freeze_prior(self) -> None:
        if self.frozen_policy_network is None:
            return
        self.frozen_policy_network.eval()
        for param in self.frozen_policy_network.parameters():
            param.requires_grad_(False)

    def freeze_oracle(self) -> None:
        if self.frozen_policy_network is None:
            self.frozen_policy_network = copy.deepcopy(self.policy_network).to(self.device)
        self._freeze_prior()

    def load_offline_state(self, payload: Dict[str, Any]) -> None:
        """Restore Sim2O-only networks from an offline/online checkpoint payload."""
        if "sim2o_policy_network" not in payload:
            return
        self.policy_network.load_state_dict(payload["sim2o_policy_network"])
        if "sim2o_v_network" in payload:
            self.v_network.load_state_dict(payload["sim2o_v_network"])
            self.target_v_network.load_state_dict(
                payload.get("sim2o_target_v_network", payload["sim2o_v_network"])
            )
        if "sim2o_v_optimizer" in payload:
            self.v_optimizer.load_state_dict(payload["sim2o_v_optimizer"])
        if "sim2o_policy_optimizer" in payload:
            self.policy_optimizer.load_state_dict(payload["sim2o_policy_optimizer"])
        self.frozen_policy_network = copy.deepcopy(self.policy_network).to(self.device)
        if "sim2o_frozen_policy_network" in payload:
            self.frozen_policy_network.load_state_dict(
                payload["sim2o_frozen_policy_network"]
            )
        self._freeze_prior()

    def state_dicts(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "sim2o_v_network": self.v_network.state_dict(),
            "sim2o_target_v_network": self.target_v_network.state_dict(),
            "sim2o_policy_network": self.policy_network.state_dict(),
            "sim2o_v_optimizer": self.v_optimizer.state_dict(),
            "sim2o_policy_optimizer": self.policy_optimizer.state_dict(),
        }
        if self.frozen_policy_network is not None:
            out["sim2o_frozen_policy_network"] = self.frozen_policy_network.state_dict()
        return out

    def new_episode_state(self) -> Dict[str, Any]:
        state = {
            "q_rnn": {
                f"agent_{i}": self.q_network.initial_state(1, self.device)
                for i in range(self.n_agents)
            },
            "policy_rnn": {
                f"agent_{i}": self.policy_network.initial_state(1, self.device)
                for i in range(self.n_agents)
            },
            "prior_rnn": {},
            "last": {},
        }
        if self.frozen_policy_network is not None:
            state["prior_rnn"] = {
                f"agent_{i}": self.frozen_policy_network.initial_state(1, self.device)
                for i in range(self.n_agents)
            }
        return state

    def reset_eval_hidden(self) -> None:
        state = self.new_episode_state()
        self.eval_q_rnn = state["q_rnn"]
        self.eval_policy_rnn = state["policy_rnn"]
        self.eval_prior_rnn = state["prior_rnn"]
        self.eval_last_actions = state["last"]
        self.eval_rnn = self.eval_q_rnn
        self.eval_offline_rnn = None
        self.eval_offline_last_actions = None

    @staticmethod
    def _sample_actions(logits: torch.Tensor, deterministic: bool) -> List[int]:
        if deterministic:
            return [int(a) for a in logits.argmax(dim=1).tolist()]
        probs = torch.softmax(logits, dim=1)
        return [int(a) for a in torch.multinomial(probs, 1).squeeze(1).tolist()]

    def _joint_q(
        self,
        actions: Sequence[int],
        q_all: torch.Tensor,
        env_state: Optional[np.ndarray],
    ) -> Tuple[Optional[float], List[float]]:
        chosen_q = [float(q_all[i, int(a)].item()) for i, a in enumerate(actions)]
        if env_state is None:
            return None, chosen_q
        q_tensor = torch.tensor(chosen_q, dtype=torch.float32, device=self.device)
        state_tensor = torch.as_tensor(
            env_state, dtype=torch.float32, device=self.device
        )
        return float(self.mixer(q_tensor.view(1, 1, -1), state_tensor.view(1, 1, -1)).item()), chosen_q

    @torch.no_grad()
    def select_actions_eval(
        self,
        observations: Dict[str, np.ndarray],
        legals: Dict[str, np.ndarray],
        eps: float,
        state: Optional[Dict[str, Any]] = None,
        trace: Optional[Dict[str, Any]] = None,
        env_state: Optional[np.ndarray] = None,
    ) -> Dict[str, int]:
        if state is None:
            q_rnn = self.eval_q_rnn
            policy_rnn = self.eval_policy_rnn
            prior_rnn = self.eval_prior_rnn
            last_actions = self.eval_last_actions
        else:
            q_rnn = state["q_rnn"]
            policy_rnn = state["policy_rnn"]
            prior_rnn = state.setdefault("prior_rnn", {})
            last_actions = state["last"]

        agents = sorted(observations.keys())
        n_agents = len(agents)
        if not q_rnn:
            for i, agent in enumerate(agents):
                q_rnn[agent] = self.q_network.initial_state(1, self.device)
                policy_rnn[agent] = self.policy_network.initial_state(1, self.device)
                if self.frozen_policy_network is not None:
                    prior_rnn[agent] = self.frozen_policy_network.initial_state(1, self.device)
        if self.frozen_policy_network is not None and not prior_rnn:
            for agent in agents:
                prior_rnn[agent] = self.frozen_policy_network.initial_state(1, self.device)

        obs_list = []
        legal_list = []
        for i, agent in enumerate(agents):
            obs = torch.from_numpy(observations[agent]).float().to(self.device)
            last_idx = last_actions.get(agent)
            if last_idx is None:
                last_onehot = torch.zeros(self.n_actions, device=self.device)
            else:
                last_onehot = F.one_hot(
                    torch.tensor(int(last_idx), device=self.device), self.n_actions
                ).float()
            pieces = [obs, last_onehot]
            if self.add_agent_id_to_obs:
                pieces.append(
                    F.one_hot(torch.tensor(i, device=self.device), self.n_agents).float()
                )
            obs_list.append(torch.cat(pieces, dim=-1))
            if legals is not None:
                legal_list.append(
                    torch.from_numpy(np.asarray(legals[agent])).bool().to(self.device)
                )
            else:
                legal_list.append(
                    torch.ones(self.n_actions, dtype=torch.bool, device=self.device)
                )

        obs_batch = torch.stack(obs_list, dim=0)
        legal_mask = torch.stack(legal_list, dim=0)

        q_hidden = torch.cat([q_rnn[agent] for agent in agents], dim=1)
        q_all, q_hidden = self.q_network(obs_batch, q_hidden)
        for i, agent in enumerate(agents):
            q_rnn[agent] = q_hidden[:, i : i + 1]
        q_all = q_all.masked_fill(~legal_mask, -1e9)

        policy_hidden = torch.cat([policy_rnn[agent] for agent in agents], dim=1)
        logits, policy_hidden = self.policy_network(obs_batch, policy_hidden)
        for i, agent in enumerate(agents):
            policy_rnn[agent] = policy_hidden[:, i : i + 1]
        logits = logits.masked_fill(~legal_mask, -1e9)

        prior_logits: Optional[torch.Tensor] = None
        prior_actions: Optional[List[int]] = None
        if self.frozen_policy_network is not None:
            prior_hidden = torch.cat([prior_rnn[agent] for agent in agents], dim=1)
            prior_logits, prior_hidden = self.frozen_policy_network(obs_batch, prior_hidden)
            for i, agent in enumerate(agents):
                prior_rnn[agent] = prior_hidden[:, i : i + 1]
            prior_logits = prior_logits.masked_fill(~legal_mask, -1e9)

        deterministic = bool(self.eval_greedy and state is None and eps <= 0.0)
        online_actions = self._sample_actions(logits, deterministic)
        if prior_logits is not None:
            prior_actions = self._sample_actions(prior_logits, deterministic)

        random_action = (
            not deterministic and eps > 0.0 and float(np.random.rand()) < float(eps)
        )
        q_tot: Optional[float] = None
        chosen_q: List[float]
        if random_action:
            executed = []
            for i in range(n_agents):
                legal_idx = torch.nonzero(legal_mask[i], as_tuple=False).squeeze(-1)
                if legal_idx.numel() == 0:
                    legal_idx = torch.arange(self.n_actions, device=self.device)
                pick = torch.randint(legal_idx.numel(), (1,), device=self.device)
                executed.append(int(legal_idx[pick].item()))
            q_tot, chosen_q = self._joint_q(executed, q_all, env_state)
        elif prior_actions is None or env_state is None:
            executed = online_actions
            q_tot, chosen_q = self._joint_q(executed, q_all, env_state)
        else:
            if deterministic or not self.random_agent_order:
                order = list(range(n_agents))
            else:
                order = np.random.permutation(n_agents).tolist()
            executed, q_tot, chosen_q = coordinated_beam_search(
                q_all,
                online_actions,
                prior_actions,
                torch.as_tensor(env_state, dtype=torch.float32, device=self.device),
                self.mixer,
                beam_width=self.beam_width,
                temperature=self.temperature,
                agent_order=order,
                deterministic=deterministic,
            )

        actions: Dict[str, int] = {}
        for i, agent in enumerate(agents):
            actions[agent] = int(executed[i])
            last_actions[agent] = int(executed[i])

        if trace is not None:
            trace["q_tot"] = q_tot
            trace["chosen_q"] = chosen_q
            trace["n_agents"] = n_agents
            if prior_logits is not None:
                trace["n_disagree"] = int(
                    (prior_logits.argmax(dim=1) != logits.argmax(dim=1)).sum().item()
                )
            else:
                trace["n_disagree"] = 0
        return actions
