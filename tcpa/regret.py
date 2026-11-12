# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Team-conditioned regret and the dynamic prior-retention gate.

The frozen offline collaboration prior is a joint behaviour, so its validity
has to be judged at the *team* level: the offline joint action has to stay
competitive with the joint action the current online team would take.

Both joint actions are scored by the SAME online target mixer, so the
difference is an ordering statement made by one value model:

    R_t = Q_on(s_t, a_on*) - Q_on(s_t, a_off)

where ``a_on*`` is the greedy joint action of the online target network under
the QMIX monotonicity constraint (so ``R_t >= 0``), and ``a_off`` is the greedy
action of the frozen offline policy, restricted to the legal actions of the
current state.

The retention coefficient is then

    g_t = sg[ exp(-R_t / tau_g) ],

which is large when the two joint actions are judged equally good (keep the
prior) and decays to zero when the online team is clearly better (release the
prior).  Only two mixer evaluations are needed per step: no ``2^N`` joint
enumeration and no per-member substitution.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

NEG_INF = -1e9


def masked_argmax(q: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
    """Greedy action per agent over the legal actions only. [B, T, N]"""
    return torch.argmax(q.masked_fill(~avail.bool(), NEG_INF), dim=3)


def gather_agent_q(q: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Q value of one joint action. q: [B, T, N, A], actions: [B, T, N]."""
    return torch.gather(q, 3, actions.unsqueeze(-1)).squeeze(-1)


def team_regret(
    q_on_target: torch.Tensor,
    q_off: torch.Tensor,
    avail: torch.Tensor,
    states: torch.Tensor,
    target_mixer: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    """Team-conditioned regret of the offline joint action.

    Args:
        q_on_target: ``[B, T, N, A]`` online target utilities (base + residual).
        q_off: ``[B, T, N, A]`` frozen offline utilities.
        avail: ``[B, T, N, A]`` legal-action mask.
        states: ``[B, T, S]`` global states used by the mixer.
        target_mixer: the online target mixing network.

    Returns:
        dict with ``regret`` ``[B, T, 1]``, ``a_on`` and ``a_off`` ``[B, T, N]``
        and the two team values.
    """
    avail = avail.bool()
    a_off = masked_argmax(q_off, avail)
    a_on = masked_argmax(q_on_target, avail)

    # Both joint actions are evaluated by the same online target mixer.
    q_tot_on = target_mixer(gather_agent_q(q_on_target, a_on), states)
    q_tot_off = target_mixer(gather_agent_q(q_on_target, a_off), states)
    regret = q_tot_on - q_tot_off
    return {
        "regret": regret,
        "q_tot_on": q_tot_on,
        "q_tot_off": q_tot_off,
        "a_on": a_on,
        "a_off": a_off,
    }


def prior_gate(
    regret: torch.Tensor,
    tau_g: float,
    *,
    mode: str = "dynamic",
) -> torch.Tensor:
    """Retention coefficient ``g_t = sg[exp(-R_t / tau_g)]``.

    ``mode='fixed'`` returns ones, which is the fixed-retention control arm.
    The gradient is always stopped: the gate selects how strongly the prior is
    retained, it is never learned through the team regret.
    """
    if mode == "fixed":
        return torch.ones_like(regret)
    tau = max(float(tau_g), 1e-6)
    return torch.exp(-regret.detach().clamp(min=0.0) / tau).detach()


class RegretScaleTracker:
    """Running scale of the team regret, used by the ``auto`` temperature.

    The regret lives in Q units, which differ by an order of magnitude between
    maps and change while the online value network is still moving.  A fixed
    temperature would therefore either switch the prior off completely or keep
    it on unconditionally.  The tracker keeps an exponential moving average of
    the regret over valid steps and turns it into a temperature

        tau_g = EMA(R) / ln 2,

    so that a typical step yields ``g ~ 0.5`` on every map.  The value in use
    is logged, so an arm is never silently mis-scaled.
    """

    def __init__(self, decay: float = 0.1, floor: float = 1e-3):
        self.decay = float(decay)
        self.floor = float(floor)
        self.ema = 0.0
        self.n_updates = 0

    def update(self, regret: torch.Tensor, mask: torch.Tensor) -> float:
        """Fold the valid-step regrets into the EMA and return the new value."""
        with torch.no_grad():
            values = regret[..., 0]
            valid = mask.bool()
            if valid.dim() == 3:
                valid = valid[..., 0]
            if bool(valid.any()):
                mean_regret = float(values[valid].mean().item())
            else:
                mean_regret = 0.0
            if self.n_updates == 0:
                self.ema = mean_regret
            else:
                self.ema = (1.0 - self.decay) * self.ema + self.decay * mean_regret
            self.n_updates += 1
        return self.ema

    def temperature(self, mode: str, fixed: Optional[float]) -> float:
        """Temperature in force for the current step."""
        if mode == "fixed" and fixed is not None:
            return max(float(fixed), 1e-6)
        return max(self.ema, self.floor) / 0.6931471805599453
