# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Shared QMIX machinery: episode unroll, obs augmentation, mixing, scalar Q_tot queries."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from qmix.policy import EvalPolicyMixin
from qmix.tensor_utils import (
    expand_batch_and_agent_dim_of_time_major_sequence,
    merge_batch_and_agent_dim_of_time_major_sequence,
    switch_two_leading_dims,
)


class QMIXCoreMixin:
    """Requires host: device, n_agents, n_actions, add_agent_id_to_obs,
    q_network, target_q_network, mixer, target_mixer."""

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

        The shared MARLC unroll clears the hidden state after the forward, so a
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


class QMIXBase(QMIXCoreMixin, EvalPolicyMixin):
    """Common learner base for rollout/evaluation typing; both learners inherit it."""
