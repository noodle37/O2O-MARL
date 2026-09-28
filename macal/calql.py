# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""MaCal-QL conservative loss: Cal-QL's calibrated push-down on joint Q_tot.

Adapts Cal-QL (Nakamoto et al., 2024) to QMIX. The calibration acts on the
joint Q_tot because that is the value the decentralized greedy policy actually
improves. Without an actor, candidate joint actions come from a decentralized
epsilon-greedy proposal built from per-agent argmax Q_i; importance weights
correct for the proposal.

Per transition (batch of k candidates per group):
    q_rand_is[m] = Q_tot(tau, u^r_m) + log |U|                    (uniform group, no clamp)
    q_prop_is[m] = Q_tot(tau, u^p_m) - log pi_prop(u^p_m | tau)   (epsilon-greedy group)
    q_prop_is[m] = max(q_prop_is[m], G_t)                         Cal-QL calibration clamp
    L_CQL = mean[ logsumexp({q_rand_is} U {q_prop_is}) - Q_tot(tau, u_data) ]
"""

from __future__ import annotations

from typing import Dict

import torch

from macal.returns import compute_mc_returns


class CalQLLossMixin:
    """Requires host: device, mixer, gamma, num_ood_actions plus QMIX core helpers."""

    def _compute_calql_loss(
        self,
        mac_out: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        chosen_qtotals: torch.Tensor,
        mask: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (cql_loss, cql_diff_detached, mc_return_mean).

        cql_diff is the masked-mean gap between the push-down (logsumexp of
        IS-corrected candidate values) and the push-up (dataset-action values);
        the dual-alpha variant constrains it below the target action gap.
        """
        k = self.num_ood_actions
        avail = batch["avail_actions"]
        states = batch["state"]
        b, t, n, a = mac_out.shape
        tp = t - 1
        qs = mac_out[:, :-1]
        st = states[:, :-1]
        legals = avail[:, :-1]

        n_legal = legals.float().sum(dim=-1).clamp(min=1.0)
        log_n_legal = n_legal.log()

        greedy = torch.where(
            legals, qs, torch.tensor(-1e9, device=self.device)
        ).argmax(dim=-1, keepdim=True)

        legals_k = legals.unsqueeze(0).expand(k, -1, -1, -1, -1)

        def sample_uniform_legal() -> torch.Tensor:
            scores = torch.rand(k, b, tp, n, a, device=self.device)
            return scores.masked_fill(~legals_k, -1e9).argmax(dim=-1, keepdim=True)

        u_rand = sample_uniform_legal()
        u_unif_prop = sample_uniform_legal()
        exploit = torch.rand(k, b, tp, n, 1, device=self.device) >= eps
        u_prop = torch.where(
            exploit, greedy.unsqueeze(0).expand(k, -1, -1, -1, -1), u_unif_prop
        )

        w_rand = log_n_legal.sum(dim=-1, keepdim=True)
        is_greedy = (u_prop.squeeze(-1) == greedy.squeeze(-1)).float()
        log_pi_prop = torch.log(
            (1.0 - eps) * is_greedy + eps / n_legal
        ).clamp(min=-20.0)
        w_prop = -log_pi_prop.sum(dim=-1, keepdim=True)

        qs_k = qs.unsqueeze(0).expand(k, -1, -1, -1, -1)
        st_k = st.unsqueeze(0).expand(k, -1, -1, -1)
        q_rand_tot = (
            self.mixer(
                torch.gather(qs_k, -1, u_rand).squeeze(-1).reshape(k * b, tp, n),
                st_k.reshape(k * b, tp, st.shape[-1]),
            )
            .reshape(k, b, tp, 1)
        )
        q_prop_tot = (
            self.mixer(
                torch.gather(qs_k, -1, u_prop).squeeze(-1).reshape(k * b, tp, n),
                st_k.reshape(k * b, tp, st.shape[-1]),
            )
            .reshape(k, b, tp, 1)
        )

        q_rand_is = q_rand_tot + w_rand
        q_prop_is = q_prop_tot + w_prop

        mc_return, mc_valid = compute_mc_returns(
            batch["reward"], batch["terminated"], batch["filled"], self.gamma
        )
        anchor = mc_return[:, :-1]
        anchor_valid = mc_valid[:, :-1] > 0.5
        q_prop_cal = torch.where(
            anchor_valid, torch.maximum(q_prop_is, anchor), q_prop_is
        )

        push_down = torch.logsumexp(torch.cat([q_rand_is, q_prop_cal], dim=0), dim=0)
        cql_term = push_down - chosen_qtotals[:, :-1]
        denom = mask.sum().clamp(min=1.0)
        cql_loss = (cql_term * mask).sum() / denom
        mc_mean = (anchor * anchor_valid.float() * mask).sum() / (
            anchor_valid.float() * mask
        ).sum().clamp(min=1.0)
        return cql_loss, cql_loss.detach(), mc_mean.detach()
