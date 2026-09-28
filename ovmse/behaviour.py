# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Behaviour-cloning network training for moderate-λ CFCQL."""

from __future__ import annotations

import copy
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from data.vault_dataset import EpisodeDataset


class BehaviourCloningMixin:
    """Requires host: behaviour_network, behaviour_optimizer, device, n_actions, grad_norm_clip, max_behaviour_filled_steps."""

    def _behaviour_probs_for_agent(self, batch: Dict[str, torch.Tensor], agent_idx: int) -> torch.Tensor:
        assert self.behaviour_network is not None
        obs = self._augment_obs(batch["obs"], batch["actions"])
        with torch.no_grad():
            logits = self._unroll(self.behaviour_network, obs, self._resets_from_batch(batch))
            # Match MARLC behaviour_mac.forward: illegal logits are -1e5 before softmax.
            logits = logits.masked_fill(~batch["avail_actions"], -1e5)
            return F.softmax(logits[:, :, agent_idx, :], dim=-1)

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
            batch = dataset.sample(batch_size, rng, device=self.device)
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
