# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Offline value memory (OVM): frozen offline QMIX for online TD targets."""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch

# OmegaConf parses the YAML 1.1 tokens `on`/`off` as booleans, so a CLI override
# of `system.support_gate_mode=off` arrives as False. Mapping those back and
# rejecting anything else keeps a typo from silently enabling the gate in a
# control arm, which would make an experiment look like a null result.
_GATE_MODES = {"off": "off", "false": "off", "on": "on", "true": "on", "anti": "anti"}


def normalize_gate_mode(value: object) -> str:
    key = str(value).strip().lower()
    if key not in _GATE_MODES:
        raise ValueError(
            f"unknown support gate mode {value!r}; expected one of off / on / anti"
        )
    return _GATE_MODES[key]


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

    def _offline_unroll(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Frozen offline per-agent utilities over the batch. [B, T, N, A]"""
        with torch.no_grad():
            return self._unroll(
                self.offline_q_network,
                self._augment_obs(batch["obs"], batch["actions"]),
                self._resets_from_batch(batch),
            )

    def _offline_chosen_tot(
        self,
        batch: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        states: torch.Tensor,
        off_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Frozen offline Q_tot of the executed actions (read-only, no autograd)."""
        with torch.no_grad():
            if off_out is None:
                off_out = self._offline_unroll(batch)
            off_chosen = torch.gather(off_out[:, :-1], 3, actions[:, :-1].long())
            return self.offline_mixer(off_chosen, states[:, :-1])

    def _offline_support_dev(
        self,
        batch: Dict[str, torch.Tensor],
        off_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-step deviation of the executed joint action from the old team's support.

        D_t = - (1/N) sum_i log pi_off_i(a_i,t | tau_t), with pi_off the softmax
        over available actions of the FROZEN offline utilities. D_t >= 0 and is
        measured entirely under the frozen network, so its scale does not move
        as the online policy changes and one threshold stays meaningful for the
        whole run. Returns [B, T-1, 1].
        """
        with torch.no_grad():
            if off_out is None:
                off_out = self._offline_unroll(batch)
            logits = off_out[:, :-1] / max(float(self.support_softmax_temp), 1e-6)
            avail = batch["avail_actions"][:, :-1].bool()
            logits = logits.masked_fill(~avail, -1e9)
            logp = torch.log_softmax(logits, dim=-1)
            chosen = torch.gather(logp, 3, batch["actions"][:, :-1].long())
            return (-chosen.squeeze(-1)).mean(dim=-1, keepdim=True)

    def support_gate(
        self,
        batch: Dict[str, torch.Tensor],
        off_out: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """g_t in [0, 1]: how far the offline value claim is trusted at step t.

        `mode` overrides the value-memory mode so the same gate can drive a
        second reuse channel with its own arm assignment.
        """
        dev = self._offline_support_dev(batch, off_out=off_out)
        temp = max(float(self.support_gate_temp), 1e-6)
        gate = torch.sigmoid((float(self.support_gate_theta) - dev) / temp)
        if (self.support_gate_mode if mode is None else normalize_gate_mode(mode)) == "anti":
            # Falsification arm: same gate mass, inverted ordering. If the
            # mechanism works because of WHERE it applies the floor and not
            # merely because it applies less of it, this arm must do worse.
            gate = 1.0 - gate
        return gate

    @torch.no_grad()
    def calibrate_support_gate(
        self,
        batches,
        quantile: float = 0.5,
        temp_frac: float = 0.25,
    ) -> Dict[str, float]:
        """Fix theta from the old team's own data, not from a tuned constant.

        A step is treated as supported when its deviation is no larger than the
        `quantile` of the deviations the offline policy assigns to the offline
        dataset itself, so the threshold inherits the old team's own notion of
        typical behaviour and transfers across tasks without retuning.
        """
        samples = []
        for batch in batches:
            dev = self._offline_support_dev(batch)
            mask = batch["filled"][:, :-1].float()
            if mask.shape[-1] != 1:
                mask = mask[..., :1]
            samples.append(dev[mask > 0.5].flatten())
        if not samples:
            raise ValueError("calibrate_support_gate received no batches")
        pooled = torch.cat(samples)
        self.support_gate_theta = float(torch.quantile(pooled, float(quantile)).item())
        self.support_gate_temp = max(float(pooled.std().item()) * float(temp_frac), 1e-3)
        return {
            "support/theta": self.support_gate_theta,
            "support/temp": self.support_gate_temp,
            "support/offline_dev_mean": float(pooled.mean().item()),
            "support/offline_dev_std": float(pooled.std().item()),
            "support/n": int(pooled.numel()),
        }

    def _ovm_td_loss(
        self,
        batch: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        states: torch.Tensor,
        chosen_tot: torch.Tensor,
        td_targets: torch.Tensor,
        mask_exp: torch.Tensor,
        lam: float | torch.Tensor,
        off_tot: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Blend TD error with max(offline Q_tot, TD target) when OVM is enabled.

        `lam` may be a scalar or a per-step tensor broadcastable to the target
        shape, which is what turns the global memory schedule into a per-sample
        validity gate.
        """
        if off_tot is None:
            off_tot = self._offline_chosen_tot(batch, actions, states)
        ovm_targets = torch.maximum(off_tot, td_targets)
        td_err = (chosen_tot[:, :-1] - td_targets) ** 2
        ovm_err = (chosen_tot[:, :-1] - ovm_targets) ** 2
        loss_td = ((1.0 - lam) * td_err + lam * ovm_err) * mask_exp
        return 0.5 * loss_td.sum() / mask_exp.sum().clamp(min=1.0)
