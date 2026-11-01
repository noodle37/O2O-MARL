# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Adaptive Cooperative Knowledge Fusion (ACKF).

A frozen offline QMIX encoder supplies a per-agent knowledge representation
``LN([h_off; q_off])``.  A shared adapter maps that representation to an
action-level residual, and one learnable scalar gate per agent controls how
much of the residual is fused with the online utility.  The online network,
mixer, adapter and gates are trained jointly with the team-level QMIX TD loss;
the offline encoder never receives gradients.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ackf.networks import KnowledgeAdapter
from ovmse.targets import build_td_lambda_targets
from qmix.core import QMIXBase
from qmix.networks import MarlcRNN, QMixer
from qmix.tensor_utils import (
    expand_batch_and_agent_dim_of_time_major_sequence,
    merge_batch_and_agent_dim_of_time_major_sequence,
    switch_two_leading_dims,
)


class ACKFLearner(QMIXBase):
    """QMIX online learner with frozen offline knowledge fusion."""

    checkpoint_stem = "ackf"

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
        td_lambda: float = 0.6,
        grad_norm_clip: float = 10.0,
        target_update_interval: int = 200,
        batch_size_run: int = 1,
        add_agent_id_to_obs: bool = True,
        adapter_hidden_dim: int = 64,
        adapter_init_std: float = 1e-2,
        gate_init: float = 0.1,
        gate_lr_scale: float = 0.1,
        norm_eps: float = 1e-5,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.gamma = float(gamma)
        self.td_lambda = float(td_lambda)
        self.grad_norm_clip = float(grad_norm_clip)
        self.target_update_interval = int(target_update_interval)
        self.batch_size_run = int(batch_size_run)
        self.add_agent_id_to_obs = bool(add_agent_id_to_obs)
        self.hidden_dim = int(recurrent_layer_dim)
        self.gate_init = float(gate_init)
        if not 0.0 < self.gate_init < 1.0:
            raise ValueError(f"gate_init must be in (0, 1), got {gate_init}")

        # Compatibility with the shared offline/online loops.  ACKF uses none
        # of the offline-specific terms: its only objective is team-level TD.
        self.cql_weight = 0.0
        self.online_cql_weight = 0.0
        self.moderate_lambda = False
        self.use_ovm = False
        self.mixing_ratio = 0.0
        self.support_gate_mode = "off"
        self.support_cql_mode = "off"
        self.sequential_explore = False

        obs_in = self.obs_dim + self.n_actions
        if self.add_agent_id_to_obs:
            obs_in += self.n_agents
        self.obs_input_dim = obs_in

        self.q_network = MarlcRNN(obs_in, self.hidden_dim, self.n_actions).to(self.device)
        self.target_q_network = copy.deepcopy(self.q_network).to(self.device)
        self.mixer = QMixer(
            self.n_agents, state_dim, mixer_embed_dim, mixer_hyper_dim
        ).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)

        knowledge_dim = self.hidden_dim + self.n_actions
        self.knowledge_norm = torch.nn.LayerNorm(knowledge_dim, eps=float(norm_eps)).to(
            self.device
        )
        # The representation itself belongs to the frozen encoder: normalise,
        # do not learn a second affine transform on top of it.
        for param in self.knowledge_norm.parameters():
            param.requires_grad_(False)

        self.knowledge_adapter = KnowledgeAdapter(
            knowledge_dim,
            self.n_agents,
            self.n_actions,
            hidden_dim=int(adapter_hidden_dim),
            init_std=float(adapter_init_std),
        ).to(self.device)
        self.target_knowledge_adapter = copy.deepcopy(self.knowledge_adapter).to(self.device)

        gate_logit = math.log(self.gate_init / (1.0 - self.gate_init))
        self.gate_psi = torch.nn.Parameter(
            torch.full((self.n_agents,), gate_logit, device=self.device)
        )
        self.target_gate_psi = torch.nn.Parameter(
            self.gate_psi.detach().clone().to(self.device), requires_grad=False
        )

        self.offline_q_network: Optional[MarlcRNN] = None
        self.offline_mixer: Optional[QMixer] = None

        # The online optimizer keeps the offline QMIX parameter order, so an
        # offline QMIX checkpoint's Adam state loads without adaptation.
        self.optimizer = torch.optim.Adam(
            list(self.q_network.parameters()) + list(self.mixer.parameters()),
            lr=learning_rate,
        )
        self.adapter_optimizer = torch.optim.Adam(
            self.knowledge_adapter.parameters(), lr=learning_rate
        )
        self.gate_optimizer = torch.optim.Adam(
            [self.gate_psi], lr=float(gate_lr_scale) * float(learning_rate)
        )

        self.episode_counter = 0
        self.last_target_update_episode = 0
        self.training_step_ctr = 0
        self.t_env = 0

        self.eval_rnn: Dict[str, torch.Tensor] = {}
        self.eval_last_actions: Dict[str, Optional[int]] = {}
        self.eval_offline_rnn: Dict[str, torch.Tensor] = {}
        self.eval_offline_last_actions = None

    # ------------------------------------------------------------- offline setup
    def set_offline_encoder(self, q_network: MarlcRNN, mixer: Optional[QMixer] = None) -> None:
        """Install and freeze the offline knowledge encoder."""
        self.offline_q_network = q_network.to(self.device)
        self.offline_mixer = mixer.to(self.device) if mixer is not None else None
        for net in (self.offline_q_network, self.offline_mixer):
            if net is None:
                continue
            net.eval()
            for param in net.parameters():
                param.requires_grad_(False)

    def load_offline_for_ovm(self, path: str) -> None:
        """Load the pretrained QMIX checkpoint as the frozen offline encoder."""
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if "q_network" not in payload:
            raise KeyError(f"offline encoder checkpoint '{path}' has no q_network")
        q_network = copy.deepcopy(self.q_network).to(self.device)
        q_network.load_state_dict(payload["q_network"])
        mixer = copy.deepcopy(self.mixer).to(self.device)
        if "mixer" in payload:
            mixer.load_state_dict(payload["mixer"])
        self.set_offline_encoder(q_network, mixer)

    def prepare_online_start(self) -> None:
        """Sync every target network after the offline checkpoint is loaded."""
        if self.offline_q_network is None:
            raise RuntimeError(
                "ACKF requires a frozen offline encoder; enable transfer.load_offline_nets"
            )
        self._sync_targets()

    def _sync_targets(self) -> None:
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_knowledge_adapter.load_state_dict(self.knowledge_adapter.state_dict())
        self.target_gate_psi.data.copy_(self.gate_psi.detach())

    def load_offline_state(self, payload: Dict[str, Any]) -> None:
        """Restore ACKF-only modules from a checkpoint, if present."""
        if "ackf_adapter" not in payload:
            return
        self.knowledge_adapter.load_state_dict(payload["ackf_adapter"])
        if "ackf_target_adapter" in payload:
            self.target_knowledge_adapter.load_state_dict(payload["ackf_target_adapter"])
        if "ackf_gate_psi" in payload:
            self.gate_psi.data.copy_(
                torch.as_tensor(payload["ackf_gate_psi"], device=self.device)
            )
        if "ackf_target_gate_psi" in payload:
            self.target_gate_psi.data.copy_(
                torch.as_tensor(payload["ackf_target_gate_psi"], device=self.device)
            )
        if "ackf_adapter_optimizer" in payload:
            self.adapter_optimizer.load_state_dict(payload["ackf_adapter_optimizer"])
        if "ackf_gate_optimizer" in payload:
            self.gate_optimizer.load_state_dict(payload["ackf_gate_optimizer"])
        if "ackf_offline_q_network" in payload:
            self.offline_q_network = copy.deepcopy(self.q_network).to(self.device)
            self.offline_q_network.load_state_dict(payload["ackf_offline_q_network"])
            mixer = copy.deepcopy(self.mixer).to(self.device)
            if "ackf_offline_mixer" in payload:
                mixer.load_state_dict(payload["ackf_offline_mixer"])
            self.set_offline_encoder(self.offline_q_network, mixer)

    def state_dicts(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "ackf_adapter": self.knowledge_adapter.state_dict(),
            "ackf_target_adapter": self.target_knowledge_adapter.state_dict(),
            "ackf_gate_psi": self.gate_psi.detach().cpu(),
            "ackf_target_gate_psi": self.target_gate_psi.detach().cpu(),
            "ackf_adapter_optimizer": self.adapter_optimizer.state_dict(),
            "ackf_gate_optimizer": self.gate_optimizer.state_dict(),
        }
        if self.offline_q_network is not None:
            out["ackf_offline_q_network"] = self.offline_q_network.state_dict()
        if self.offline_mixer is not None:
            out["ackf_offline_mixer"] = self.offline_mixer.state_dict()
        return out

    # ----------------------------------------------------------------- unrolling
    def _unroll_with_hidden(
        self, net: torch.nn.Module, obs: torch.Tensor, resets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (hidden, q) for every [B, T, N, ...] step of the batch."""
        b, t, n, _ = obs.shape
        resets = resets.expand(b, t, n)
        obs_tm = merge_batch_and_agent_dim_of_time_major_sequence(
            switch_two_leading_dims(obs)
        )
        resets_tm = merge_batch_and_agent_dim_of_time_major_sequence(
            switch_two_leading_dims(resets)
        )
        q_tm, h_tm = self._unroll_hidden_reset_before_forward(net, obs_tm, resets_tm)
        q = switch_two_leading_dims(
            expand_batch_and_agent_dim_of_time_major_sequence(q_tm, b, n)
        )
        h = switch_two_leading_dims(
            expand_batch_and_agent_dim_of_time_major_sequence(h_tm, b, n)
        )
        return h, q

    @staticmethod
    def _unroll_hidden_reset_before_forward(
        net: torch.nn.Module, inputs: torch.Tensor, resets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        time_steps, batch_size = inputs.shape[:2]
        hidden_state = net.initial_state(batch_size, inputs.device, dtype=inputs.dtype)
        outputs: List[torch.Tensor] = []
        hiddens: List[torch.Tensor] = []
        for i in range(time_steps):
            reset_mask = resets[i].unsqueeze(0).unsqueeze(-1)
            hidden_state = torch.where(
                reset_mask,
                net.initial_state(batch_size, inputs.device, dtype=inputs.dtype),
                hidden_state,
            )
            output, hidden_state = net(inputs[i], hidden_state)
            outputs.append(output)
            hiddens.append(hidden_state.squeeze(0))
        return torch.stack(outputs, dim=0), torch.stack(hiddens, dim=0)

    def _agent_ids_like(self, tensor: torch.Tensor) -> torch.Tensor:
        b, t = tensor.shape[:2]
        eye = torch.eye(self.n_agents, device=tensor.device, dtype=tensor.dtype)
        return eye.view(1, 1, self.n_agents, self.n_agents).expand(b, t, -1, -1)

    def _offline_knowledge(
        self, obs_aug: torch.Tensor, resets: torch.Tensor
    ) -> torch.Tensor:
        """e_off = LN([h_off; q_off]) under the frozen encoder."""
        if self.offline_q_network is None:
            raise RuntimeError(
                "ACKF requires a frozen offline encoder; enable transfer.load_offline_nets"
            )
        with torch.no_grad():
            h_off, q_off = self._unroll_with_hidden(
                self.offline_q_network, obs_aug, resets
            )
            knowledge = self.knowledge_norm(torch.cat([h_off, q_off], dim=-1))
        return knowledge.detach()

    def _fused_utilities(
        self,
        obs_aug: torch.Tensor,
        resets: torch.Tensor,
        knowledge: torch.Tensor,
        *,
        target: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """q_ACKF = q_on + g_i * delta_q_K for the whole [B, T, N, A] batch."""
        net = self.target_q_network if target else self.q_network
        adapter = (
            self.target_knowledge_adapter if target else self.knowledge_adapter
        )
        psi = self.target_gate_psi if target else self.gate_psi
        _, q_on = self._unroll_with_hidden(net, obs_aug, resets)
        delta_q = adapter(knowledge, self._agent_ids_like(knowledge))
        gate = torch.sigmoid(psi).view(1, 1, self.n_agents, 1)
        fused = q_on + gate * delta_q
        return fused, q_on, delta_q, gate

    # ------------------------------------------------------------------ training
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
        if cql_weight is not None and float(cql_weight) > 0.0:
            raise RuntimeError("ACKF does not use a CQL term; set online_cql_weight=0")
        if use_ovm:
            raise RuntimeError("ACKF does not use the OVM target; set use_ovm=false")
        if self.offline_q_network is None:
            raise RuntimeError(
                "ACKF requires a frozen offline encoder; enable transfer.load_offline_nets"
            )
        if t_env is not None:
            self.t_env = int(t_env)

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
        if td_row_mask is not None:
            row = td_row_mask.to(mask.device, mask.dtype).view(-1, 1, 1)
            td_mask = mask * row
        else:
            td_mask = mask

        obs_aug = self._augment_obs(batch["obs"], actions)
        resets = self._resets_from_batch(batch)
        knowledge = self._offline_knowledge(obs_aug, resets)

        fused, q_on, delta_q, gate = self._fused_utilities(
            obs_aug, resets, knowledge, target=False
        )
        with torch.no_grad():
            target_fused, _, _, _ = self._fused_utilities(
                obs_aug, resets, knowledge, target=True
            )

        chosen_q = torch.gather(fused, 3, actions.long())
        mac_sel = fused.masked_fill(~avail, -1e9)
        max_actions = torch.argmax(mac_sel, dim=3, keepdim=True)
        target_max_q = torch.gather(target_fused, 3, max_actions)

        chosen_tot = self.mixer(chosen_q, states)
        with torch.no_grad():
            target_max_tot = self.target_mixer(target_max_q, states)

        td_targets = build_td_lambda_targets(
            rewards[:, :-1],
            terminated[:, :-1].max(dim=2, keepdim=True)[0],
            mask,
            target_max_tot,
            self.gamma,
            self.td_lambda,
        ).detach()

        td_err = (chosen_tot[:, :-1] - td_targets) ** 2
        loss = (td_err * td_mask).sum() / td_mask.sum().clamp(min=1.0)

        self.training_step_ctr += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.adapter_optimizer.zero_grad(set_to_none=True)
        self.gate_optimizer.zero_grad(set_to_none=True)
        loss.backward()

        trainable = (
            list(self.q_network.parameters())
            + list(self.mixer.parameters())
            + list(self.knowledge_adapter.parameters())
            + [self.gate_psi]
        )
        grad_norm = torch.tensor(0.0, device=self.device)
        if self.grad_norm_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, self.grad_norm_clip)
        gate_grad = (
            0.0
            if self.gate_psi.grad is None
            else float(self.gate_psi.grad.detach().abs().mean().item())
        )
        self.optimizer.step()
        self.adapter_optimizer.step()
        self.gate_optimizer.step()

        self.episode_counter += int(episode_delta)
        if (
            self.episode_counter - self.last_target_update_episode
        ) / max(1, self.target_update_interval) >= 1.0:
            self._sync_targets()
            self.last_target_update_episode = self.episode_counter

        valid = (mask > 0.5).expand(-1, -1, self.n_agents)
        with torch.no_grad():
            delta_valid = delta_q[:, :-1][valid]
            fusion_valid = (gate * delta_q)[:, :-1][valid]
            q_on_valid = q_on[:, :-1][valid]
            gate_values = torch.sigmoid(self.gate_psi)
            stats = {
                "ackf/gate_mean": float(gate_values.mean().item()),
                "ackf/gate_min": float(gate_values.min().item()),
                "ackf/gate_max": float(gate_values.max().item()),
                "ackf/gate_grad_abs_mean": gate_grad,
                "ackf/delta_abs_mean": (
                    float(delta_valid.abs().mean().item())
                    if delta_valid.numel() > 0
                    else 0.0
                ),
                "ackf/fusion_abs_mean": (
                    float(fusion_valid.abs().mean().item())
                    if fusion_valid.numel() > 0
                    else 0.0
                ),
                "ackf/q_on_abs_mean": (
                    float(q_on_valid.abs().mean().item())
                    if q_on_valid.numel() > 0
                    else 0.0
                ),
            }

        return {
            "loss": float(loss.detach().item()),
            "grad_norm": float(grad_norm.detach().item()),
            "mean_chosen_q_values": float(chosen_tot.detach().mean().item()),
            "lambda_memory": 0.0,
            **stats,
        }

    # -------------------------------------------------------------------- online
    def new_episode_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "rnn": {
                f"agent_{i}": self.q_network.initial_state(1, self.device)
                for i in range(self.n_agents)
            },
            "last": {},
            "offline_rnn": {},
        }
        if self.offline_q_network is not None:
            state["offline_rnn"] = {
                f"agent_{i}": self.offline_q_network.initial_state(1, self.device)
                for i in range(self.n_agents)
            }
        return state

    def reset_eval_hidden(self) -> None:
        state = self.new_episode_state()
        self.eval_rnn = state["rnn"]
        self.eval_offline_rnn = state["offline_rnn"]
        self.eval_last_actions = state["last"]
        self.eval_offline_last_actions = None

    @torch.no_grad()
    def select_actions_eval(
        self,
        observations: Dict[str, np.ndarray],
        legals: Dict[str, np.ndarray],
        eps: float,
        state: Optional[Dict[str, Any]] = None,
        trace: Optional[Dict[str, Any]] = None,
        env_state=None,
    ) -> Dict[str, int]:
        del env_state
        if state is None:
            rnn_state = self.eval_rnn
            offline_rnn = self.eval_offline_rnn
            last_actions = self.eval_last_actions
        else:
            rnn_state = state["rnn"]
            offline_rnn = state.setdefault("offline_rnn", {})
            last_actions = state["last"]
        if self.offline_q_network is None:
            raise RuntimeError("ACKF requires a frozen offline knowledge encoder")
        for i, agent in enumerate(sorted(observations.keys())):
            rnn_state.setdefault(
                agent, self.q_network.initial_state(1, self.device)
            )
            offline_rnn.setdefault(
                agent, self.offline_q_network.initial_state(1, self.device)
            )

        epsilon_dec = 0.0 if eps <= 0.0 else eps / self.n_agents
        sequential = bool(getattr(self, "sequential_explore", False)) and eps > 0.0
        explorer = int(np.random.randint(self.n_agents)) if sequential else -1
        gate = torch.sigmoid(self.gate_psi)

        actions: Dict[str, int] = {}
        chosen_q: List[float] = []
        offline_chosen_q: List[float] = []
        explored_flags: List[bool] = []
        n_disagree = 0
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
                pieces.append(
                    F.one_hot(torch.tensor(i, device=self.device), self.n_agents).float()
                )
            obs_in = torch.cat(pieces, dim=-1).unsqueeze(0)

            q_on, rnn_state[agent] = self.q_network(obs_in, rnn_state.get(agent))
            q_off, h_off = self.offline_q_network(obs_in, offline_rnn.get(agent))
            knowledge = self.knowledge_norm(
                torch.cat([h_off.reshape(1, -1), q_off.reshape(1, -1)], dim=-1)
            )
            agent_id = F.one_hot(
                torch.tensor(i, device=self.device), self.n_agents
            ).float().unsqueeze(0)
            delta_q = self.knowledge_adapter(knowledge, agent_id)
            fused_q = q_on + gate[i].view(1) * delta_q
            legal = torch.from_numpy(legals[agent]).bool().to(self.device)
            masked = fused_q[0].masked_fill(~legal, -1e9)

            if sequential:
                explore_here = i == explorer and np.random.rand() < eps
            else:
                explore_here = np.random.rand() < epsilon_dec
            if explore_here:
                legal_idx = torch.nonzero(legal, as_tuple=False).squeeze(-1)
                a = int(
                    legal_idx[
                        torch.randint(legal_idx.numel(), (1,), device=legal_idx.device)
                    ].item()
                )
            else:
                a = int(torch.argmax(masked).item())
            explored_flags.append(bool(explore_here))
            last_actions[agent] = a
            actions[agent] = a

            if trace is not None:
                chosen_q.append(float(fused_q[0, a].item()))
                if self.offline_mixer is not None:
                    off_masked = q_off[0].masked_fill(~legal, -1e9)
                    offline_greedy = int(torch.argmax(off_masked).item())
                    offline_chosen_q.append(float(q_off[0, a].item()))
                    n_disagree += int(offline_greedy != a)

        if trace is not None:
            trace["chosen_q"] = chosen_q
            trace["offline_chosen_q"] = offline_chosen_q
            trace["n_disagree"] = n_disagree
            trace["n_agents"] = len(chosen_q)
        if state is not None:
            state["explored"] = explored_flags
        return actions
