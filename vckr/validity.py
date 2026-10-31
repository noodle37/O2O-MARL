# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Counterfactual validity of sub-team knowledge under the current team context.

For a unit kappa with members C and offline snippet pi_off, the quantity we
need is

    Delta_k(kappa, h) = E[Q_k(h, a^C ~ pi_off^C, a^Cbar ~ pi_k^Cbar)]
                        - E[Q_k(h, a ~ pi_k)],

i.e. the advantage of switching only the sub-team to its offline snippet while
everyone else keeps the current online policy.  Evaluating the expectation over
joint actions is exponential, so we use the mixer's own input semantics: the
expected per-agent utility under a policy, then one monotone mixing pass.  This
is exact when the mixer is linear in the utilities and a controlled
approximation otherwise (the mixer is smooth and monotone).

The offline critic evaluates the same counterfactual in the old context, and
the two are blended by the start-domain score.  A total-variation penalty on
the complement policy subtracts validity when the surrounding teammates have
drifted away from the context the snippet was learned in.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch


def _mix(mixer: torch.nn.Module, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """Mixer output for [M, N] or [U, M, N] utilities.

    QMixer reads the state dimension from axis 2, so a flat [M, S] state is
    promoted to [M, 1, S] first; the result is folded back to the leading dims.
    """
    leading = agent_qs.shape[:-1]
    if agent_qs.dim() == 2:
        agent_qs = agent_qs.unsqueeze(1)
        states = states.unsqueeze(1)
    out = mixer(agent_qs, states)
    return out.reshape(*leading)


@torch.no_grad()
def validity_inference(
    learner: Any,
    batch: Dict[str, torch.Tensor],
    mac_out: torch.Tensor,
    off_out: torch.Tensor,
    knowledge: Any,
    params: Any,
) -> Dict[str, torch.Tensor]:
    device = mac_out.device
    b, t, n, _ = mac_out.shape
    m = b * t
    avail = batch["avail_actions"].bool()
    states = batch["state"].reshape(m, -1)
    obs_flat = batch["obs"].reshape(m, n, -1)

    temp = max(float(params.policy_temp), 1e-6)
    pi_on = torch.softmax(mac_out.masked_fill(~avail, -1e9) / temp, dim=-1)
    pi_off = torch.softmax(off_out.masked_fill(~avail, -1e9) / temp, dim=-1)

    qbar_on = (pi_on * mac_out).sum(dim=-1).reshape(m, n)
    qbar_off = (pi_off * mac_out).sum(dim=-1).reshape(m, n)
    qbar_on_off = (pi_on * off_out).sum(dim=-1).reshape(m, n)
    qbar_off_off = (pi_off * off_out).sum(dim=-1).reshape(m, n)

    mixer = learner.mixer
    off_mixer = learner.offline_mixer
    base_on = _mix(mixer, qbar_on, states)
    base_off = _mix(off_mixer, qbar_on_off, states)

    units: Sequence[Tuple[int, ...]] = knowledge.units
    u_count = len(units)
    q_cur = qbar_on.unsqueeze(0).expand(u_count, -1, -1).clone()
    q_old = qbar_on_off.unsqueeze(0).expand(u_count, -1, -1).clone()
    for u, members in enumerate(units):
        idx = list(members)
        q_cur[u, :, idx] = qbar_off[:, idx]
        q_old[u, :, idx] = qbar_off_off[:, idx]
    states_batch = states.unsqueeze(0).expand(u_count, -1, -1)
    mix_cur = _mix(mixer, q_cur, states_batch)  # [U, M]
    mix_old = _mix(off_mixer, q_old, states_batch)

    delta_on = mix_cur - base_on.unsqueeze(0)
    delta_off = mix_old - base_off.unsqueeze(0)

    alpha = knowledge.domain_alpha(obs_flat, params)  # [U, M]

    tv_agent = 0.5 * (pi_on - pi_off).abs().sum(dim=-1).reshape(m, n)
    comp = (~knowledge.membership).float()
    tv = torch.einsum("mn,un->um", tv_agent, comp) / comp.sum(dim=1, keepdim=True).clamp(min=1.0)

    delta_hat = alpha * delta_off + (1.0 - alpha) * delta_on - float(params.beta_tv) * tv
    clip = float(params.delta_clip)
    delta_hat = delta_hat.clamp(min=-clip, max=clip)

    selected = torch.zeros(u_count, m, dtype=torch.bool, device=device)
    occupied = torch.zeros(m, n, dtype=torch.bool, device=device)
    remaining = delta_hat.clone()
    membership = knowledge.membership
    for _ in range(max(1, int(params.max_selected_units))):
        allow = torch.ones(u_count, m, dtype=torch.bool, device=device)
        for u, members in enumerate(units):
            allow[u] = ~occupied[:, list(members)].any(dim=-1)
        score = remaining.masked_fill(~allow, -1e9)
        best_val, best_u = score.max(dim=0)
        can = best_val > 0.0
        if bool(params.require_domain):
            alpha_best = alpha.gather(0, best_u.unsqueeze(0)).squeeze(0)
            can = can & (alpha_best >= 0.5)
        if not bool(can.any()):
            break
        cols = can.nonzero(as_tuple=False).squeeze(-1)
        rows = best_u[cols]
        selected[rows, cols] = True
        occupied[cols] = occupied[cols] | membership[rows]
        remaining[rows, cols] = -1e9

    gated = torch.sigmoid(delta_hat / max(float(knowledge.delta_temp), 1e-6)) * delta_hat
    bonus = (selected.float() * gated).sum(dim=0).clamp(max=clip)

    def as_bt(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(u_count, b, t)

    return {
        "delta_on": as_bt(delta_on),
        "delta_off": as_bt(delta_off),
        "delta_hat": as_bt(delta_hat),
        "alpha": as_bt(alpha),
        "tv": as_bt(tv),
        "selected": as_bt(selected),
        "member": occupied.reshape(b, t, n),
        "bonus": bonus.reshape(b, t),
    }
