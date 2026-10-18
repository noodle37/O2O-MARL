# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Offline value memory (OVM): frozen offline QMIX for online TD targets."""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch


class OVMMixin:
    """Requires host: q_network, mixer, device, lambda_memory_end, lambda_anneal_steps, offline_q_network, offline_mixer."""

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

    def _offline_chosen_tot(
        self,
        batch: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Frozen offline Q_tot of the executed actions (read-only, no autograd)."""
        with torch.no_grad():
            off_out = self._unroll(
                self.offline_q_network,
                self._augment_obs(batch["obs"], batch["actions"]),
                self._resets_from_batch(batch),
            )
            off_chosen = torch.gather(off_out[:, :-1], 3, actions[:, :-1].long())
            return self.offline_mixer(off_chosen, states[:, :-1])

    def _ovm_td_loss(
        self,
        batch: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        states: torch.Tensor,
        chosen_tot: torch.Tensor,
        td_targets: torch.Tensor,
        mask_exp: torch.Tensor,
        lam: float,
        off_tot: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Blend TD error with max(offline Q_tot, TD target) when OVM is enabled."""
        if off_tot is None:
            off_tot = self._offline_chosen_tot(batch, actions, states)
        ovm_targets = torch.maximum(off_tot, td_targets)
        td_err = (chosen_tot[:, :-1] - td_targets) ** 2
        ovm_err = (chosen_tot[:, :-1] - ovm_targets) ** 2
        loss_td = ((1.0 - lam) * td_err + lam * ovm_err) * mask_exp
        return 0.5 * loss_td.sum() / mask_exp.sum().clamp(min=1.0)
