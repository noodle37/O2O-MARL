# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Conservative Q-learning losses (MACQL and CFCQL with optional moderate-λ)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


class CQLLossMixin:
    """Requires host: device, n_agents, num_ood_actions, mixer, raw_cql, moderate_lambda, softmax_temp."""

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
            repeat_avail,
            dim=-1,
            index=torch.randint(0, n_actions, (sample_n, bs, ts - 1, n_agents, 1), device=self.device),
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
