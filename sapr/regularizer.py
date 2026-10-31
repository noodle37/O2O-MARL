# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""KL regularization of the online Q-policy towards the frozen offline prior.

For value-based MARL the per-agent policy is the legal-action softmax of the
local Q values,

    pi_theta^i(a | h) = softmax_a(Q_theta^i(h, a) / tau),

and the regularizer is the forward KL

    R_prior = (1 / N) sum_i E_h [ KL(pi_theta^i(.|h) || pi_off^i(.|h)) ].

The offline network is frozen and only supplies the reference distribution.
``support_floor`` mixes a tiny uniform mass over the legal actions into both
distributions, which keeps log pi_off finite even when the offline Q values
assign an effectively zero score to an action the online policy visits.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def masked_policy(
    q: torch.Tensor,
    avail: torch.Tensor,
    temperature: float,
    support_floor: float = 0.0,
) -> torch.Tensor:
    """Legal-action policy probabilities with an optional uniform support floor.

    Args:
        q: ``[..., A]`` action values.
        avail: ``[..., A]`` boolean legal-action mask.
        temperature: positive softmax temperature.
        support_floor: mass in ``[0, 1)`` mixed uniformly over legal actions.

    Rows with no legal action are treated as all-legal (their policy is never
    used for a real decision; the environment ignores dead agents' actions).
    """
    if temperature <= 0.0 or not math.isfinite(float(temperature)):
        raise ValueError(f"temperature must be positive and finite, got {temperature}")
    if not 0.0 <= support_floor < 1.0:
        raise ValueError(f"support_floor must lie in [0, 1), got {support_floor}")

    avail = avail.bool()
    usable = torch.where(avail.any(dim=-1, keepdim=True), avail, torch.ones_like(avail))
    logits = q / float(temperature)
    logits = logits.masked_fill(~usable, -1e9)
    probs = torch.softmax(logits, dim=-1)
    if support_floor > 0.0:
        legal = usable.to(probs.dtype)
        legal_mass = legal / legal.sum(dim=-1, keepdim=True).clamp(min=1.0)
        probs = (1.0 - support_floor) * probs + support_floor * legal_mass
    return probs


def prior_kl(
    online_q: torch.Tensor,
    offline_q: torch.Tensor,
    avail: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    support_floor: float = 0.0,
) -> torch.Tensor:
    """KL(pi_online || pi_offline) averaged over valid agents and decision steps.

    Args:
        online_q, offline_q: ``[B, T, N, A]`` action values.
        avail: ``[B, T, N, A]`` legal-action mask.
        mask: ``[B, T]`` valid-step mask (filled and non-terminal).
    """
    if online_q.shape != offline_q.shape or online_q.shape != avail.shape:
        raise ValueError(
            "online_q, offline_q and avail must share shape "
            f"{tuple(online_q.shape)}"
        )
    if mask.shape != online_q.shape[:2]:
        raise ValueError(
            f"mask must have shape {tuple(online_q.shape[:2])}, got {tuple(mask.shape)}"
        )

    p_on = masked_policy(online_q, avail, temperature, support_floor)
    p_off = masked_policy(offline_q, avail, temperature, support_floor)
    tiny = torch.finfo(p_on.dtype).tiny
    log_on = torch.log(p_on.clamp_min(tiny))
    log_off = torch.log(p_off.clamp_min(tiny))
    kl = (p_on * (log_on - log_off)).sum(dim=-1)  # [B, T, N]

    # Dead agents (no legal action) carry no policy and are excluded.
    agent_valid = avail.bool().any(dim=-1).to(kl.dtype)
    weight = mask.to(kl.dtype).unsqueeze(-1) * agent_valid
    denom = weight.sum().clamp(min=1.0)
    return (kl * weight).sum() / denom


class SAPRPriorMixin:
    """Adds the frozen-prior KL term to a QMIX-style learner.

    Host requirements: ``offline_q_network`` and ``_offline_unroll`` from the
    OVM machinery, plus ``n_agents``/``device`` for shapes.
    """

    prior_kl_weight: float
    prior_temperature: float
    prior_support_floor: float

    def init_sapr_prior(self) -> None:
        self.prior_kl_weight = 0.0
        self.prior_temperature = 1.0
        self.prior_support_floor = 1e-8

    def set_prior(
        self,
        *,
        weight: Optional[float] = None,
        temperature: Optional[float] = None,
        support_floor: Optional[float] = None,
    ) -> None:
        if weight is not None:
            self.set_prior_weight(weight)
        if temperature is not None:
            value = float(temperature)
            if value <= 0.0 or not math.isfinite(value):
                raise ValueError(f"prior temperature must be positive, got {temperature}")
            self.prior_temperature = value
        if support_floor is not None:
            value = float(support_floor)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"prior support_floor must lie in [0, 1), got {value}")
            self.prior_support_floor = value

    def set_prior_weight(self, weight: float) -> None:
        value = float(weight)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"prior KL weight must be finite and non-negative, got {weight}")
        self.prior_kl_weight = value

    def prior_kl_loss(
        self,
        batch: dict,
        online_out: torch.Tensor,
        offline_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Regularizer on valid transitions; returns a scalar tensor."""
        if getattr(self, "offline_q_network", None) is None and offline_out is None:
            raise RuntimeError("prior KL needs a frozen offline_q_network")
        if offline_out is None:
            offline_out = self._offline_unroll(batch)

        if "avail_actions" not in batch:
            raise KeyError(
                "SAPR prior KL requires batch['avail_actions']; the environment "
                "must expose per-agent legal actions"
            )
        filled = batch["filled"][:, :-1]
        if filled.dim() == 3:
            filled = filled[..., 0]
        avail = batch["avail_actions"][:, :-1].bool()
        return prior_kl(
            online_out[:, :-1],
            offline_out[:, :-1],
            avail,
            filled.to(online_out.dtype),
            self.prior_temperature,
            self.prior_support_floor,
        )
