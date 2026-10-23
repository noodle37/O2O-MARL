# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Coordinated Beam Search over offline/online agent-level action proposals."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def _multinomial(probs: torch.Tensor, n: int, *, replacement: bool) -> List[int]:
    p = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    p = p.clamp_min(0.0)
    if float(p.sum()) <= 0.0:
        p = torch.ones_like(p)
    p = p / p.sum()
    return torch.multinomial(p, n, replacement=replacement).tolist()


def coordinated_beam_search(
    q_values: torch.Tensor,
    online_actions: Sequence[int],
    offline_actions: Sequence[int],
    state: torch.Tensor,
    mixer,
    *,
    beam_width: int,
    temperature: float,
    agent_order: Sequence[int],
    deterministic: bool,
) -> Tuple[List[int], float, List[float]]:
    """Compose a joint action, returning (actions, Q_tot, per-agent chosen Q).

    q_values: [N, A] local Q values, already masked for illegal actions.
    state: [S] global state used by the centralized mixer.
    """
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    n_agents = q_values.shape[0]
    device = q_values.device
    beam: List[Tuple[int, ...]] = [tuple(int(a) for a in online_actions)]

    def score(candidates: List[Tuple[int, ...]]):
        actions = torch.as_tensor(candidates, dtype=torch.long, device=device)
        local_q = q_values.gather(1, actions.t()).t()
        states = state.unsqueeze(0).expand(len(candidates), -1)
        scores = mixer(local_q, states).reshape(-1)
        return local_q, scores

    for agent in agent_order:
        expanded = list(beam)
        seen = set(beam)
        for candidate in beam:
            variant = list(candidate)
            variant[int(agent)] = int(offline_actions[int(agent)])
            variant_t = tuple(variant)
            if variant_t not in seen:
                seen.add(variant_t)
                expanded.append(variant_t)
        _, scores = score(expanded)
        if len(expanded) > beam_width:
            probs = torch.softmax(scores / temperature, dim=0)
            picked = _multinomial(probs, beam_width, replacement=False)
            beam = [expanded[i] for i in picked]
        else:
            beam = expanded

    local_q, scores = score(beam)
    if deterministic:
        idx = int(torch.argmax(scores).item())
    else:
        idx = _multinomial(torch.softmax(scores / temperature, dim=0), 1, replacement=True)[0]
    actions = [int(a) for a in beam[idx]]
    q_tot = float(scores[idx].detach().item())
    chosen_q = [float(x) for x in local_q[idx].detach().tolist()]
    if len(actions) != n_agents:
        raise RuntimeError("CBS produced an invalid joint action")
    return actions, q_tot, chosen_q
