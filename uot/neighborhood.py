"""Recorded-neighborhood candidates for Ungated Ordinal Transfer.

The pool is S_off. On each filled step the anchors are the frozen greedy
joint, the live greedy joint, and the executed joint. Candidates are
recorded tuples within Hamming distance d_max of any anchor. If more than
K survive, keep smaller distance first, then a larger frozen per-agent
utility gap on the flipped agents. That gap only ranks which joints enter
the extra sum. The loss still uses Q_bar and Q_theta.
"""

from __future__ import annotations

import torch

from uot.support import RecordedJoints


def product_greedy(qvals: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
    """qvals/avail [B,T,N,A] -> joint [B,T,N]."""
    legal = avail.bool() if avail.dtype != torch.bool else avail
    return qvals.masked_fill(~legal, -1e9).argmax(dim=-1)


def joints_available(joints: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
    """joints [..., N], avail [B,T,N,A] -> bool [...]."""
    n_act = avail.shape[-1]
    idx = joints.long().clamp(0, n_act - 1).unsqueeze(-1)
    av = avail
    while av.dim() < joints.dim() + 1:
        av = av.unsqueeze(0)
    av = av.expand(*joints.shape, n_act)
    gathered = torch.gather(av, dim=-1, index=idx).squeeze(-1)
    if gathered.dtype != torch.bool:
        gathered = gathered > 0
    return gathered.all(dim=-1)


def pack_unique_topk(
    joints: torch.Tensor,
    valid: torch.Tensor,
    scores: torch.Tensor,
    support: RecordedJoints,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep up to k unique joints per (B,T), highest score first.

    joints [P,B,T,N], valid/scores [P,B,T] -> sel [K,B,T,N], sel_valid [K,B,T], nfill [B,T].
    """
    p, b, t, n = joints.shape
    device = joints.device
    if p == 0 or k <= 0:
        return (
            joints.new_zeros(max(k, 0), b, t, n),
            torch.zeros(max(k, 0), b, t, dtype=torch.bool, device=device),
            torch.zeros(b, t, dtype=torch.long, device=device),
        )
    scores = scores.masked_fill(~valid, -1e18)
    order = torch.argsort(scores, dim=0, descending=True)
    codes = support.encode(joints)
    joints_s = joints.gather(0, order.unsqueeze(-1).expand_as(joints))
    codes_s = codes.gather(0, order)
    valid_s = valid.gather(0, order)
    selected = joints.new_zeros(k, b, t, n)
    sel_valid = torch.zeros(k, b, t, dtype=torch.bool, device=device)
    nfill = torch.zeros(b, t, dtype=torch.long, device=device)
    seen = torch.full((k, b, t), -1, dtype=torch.long, device=device)
    for pi in range(p):
        if pi > 0 and bool((nfill >= k).all().item()):
            break
        code = codes_s[pi]
        take = valid_s[pi] & (nfill < k)
        if k > 0:
            take = take & ~(seen == code.unsqueeze(0)).any(dim=0)
        if not bool(take.any().item()):
            continue
        for slot in range(k):
            write = take & (nfill == slot)
            if not bool(write.any().item()):
                continue
            selected[slot] = torch.where(write.unsqueeze(-1), joints_s[pi], selected[slot])
            sel_valid[slot] = sel_valid[slot] | write
            seen[slot] = torch.where(write, code, seen[slot])
        nfill = nfill + take.long()
    return selected, sel_valid, nfill


def _agent_q(bar_q: torch.Tensor, agent: int) -> torch.Tensor:
    return bar_q[:, :, agent, :]


def hamming1(
    anchors: torch.Tensor,
    avail: torch.Tensor,
    bar_q: torch.Tensor,
    support: RecordedJoints,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recorded Hamming-1 neighbors of each anchor. Any agent may differ."""
    n_anc, b, t, n = anchors.shape
    n_act = bar_q.shape[-1]
    pieces, valids, scores = [], [], []
    for i in range(n):
        cand = anchors.unsqueeze(1).expand(n_anc, n_act, b, t, n).contiguous().clone()
        acts = torch.arange(n_act, device=anchors.device, dtype=torch.long)
        cand[..., i] = acts.view(1, n_act, 1, 1)
        cand = cand.reshape(n_anc * n_act, b, t, n)
        old_i = anchors[..., i].unsqueeze(1).expand(n_anc, n_act, b, t).reshape(n_anc * n_act, b, t)
        changed = cand[..., i] != old_i
        valid = changed & joints_available(cand, avail) & support.contains(cand)
        q_all = _agent_q(bar_q, i).unsqueeze(0).expand(n_anc, -1, -1, -1)
        q_new = q_all.permute(0, 3, 1, 2).reshape(n_anc * n_act, b, t)
        q_old = torch.gather(
            q_all, dim=-1, index=anchors[..., i].unsqueeze(-1).clamp(0, n_act - 1)
        ).squeeze(-1)
        q_old = q_old.unsqueeze(1).expand(n_anc, n_act, b, t).reshape(n_anc * n_act, b, t)
        pieces.append(cand)
        valids.append(valid)
        scores.append((q_new - q_old).abs())
    return torch.cat(pieces, 0), torch.cat(valids, 0), torch.cat(scores, 0)


def hamming2(
    anchors: torch.Tensor,
    avail: torch.Tensor,
    bar_q: torch.Tensor,
    support: RecordedJoints,
    i: int,
    j: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recorded Hamming-2 neighbors that differ in agents i and j."""
    n_anc, b, t, n = anchors.shape
    n_act = bar_q.shape[-1]
    cand = anchors[:, None, None].expand(n_anc, n_act, n_act, b, t, n).contiguous().clone()
    ai = torch.arange(n_act, device=anchors.device, dtype=torch.long)
    cand[..., i] = ai.view(1, n_act, 1, 1, 1)
    cand[..., j] = ai.view(1, 1, n_act, 1, 1)
    cand = cand.reshape(n_anc * n_act * n_act, b, t, n)
    old_i = anchors[..., i].unsqueeze(1).unsqueeze(1).expand(n_anc, n_act, n_act, b, t)
    old_j = anchors[..., j].unsqueeze(1).unsqueeze(1).expand(n_anc, n_act, n_act, b, t)
    old_i = old_i.reshape(n_anc * n_act * n_act, b, t)
    old_j = old_j.reshape(n_anc * n_act * n_act, b, t)
    changed = (cand[..., i] != old_i) & (cand[..., j] != old_j)
    valid = changed & joints_available(cand, avail) & support.contains(cand)
    q_i = _agent_q(bar_q, i)
    q_j = _agent_q(bar_q, j)
    qi_a = q_i.permute(2, 0, 1)
    qj_a = q_j.permute(2, 0, 1)
    q_new_i = (
        qi_a[:, None]
        .expand(n_act, n_act, b, t)
        .unsqueeze(0)
        .expand(n_anc, n_act, n_act, b, t)
        .reshape(n_anc * n_act * n_act, b, t)
    )
    q_new_j = (
        qj_a[None, :]
        .expand(n_act, n_act, b, t)
        .unsqueeze(0)
        .expand(n_anc, n_act, n_act, b, t)
        .reshape(n_anc * n_act * n_act, b, t)
    )
    q_old_i = torch.gather(
        q_i.unsqueeze(0).expand(n_anc, -1, -1, -1),
        dim=-1,
        index=anchors[..., i].unsqueeze(-1).clamp(0, n_act - 1),
    ).squeeze(-1)
    q_old_j = torch.gather(
        q_j.unsqueeze(0).expand(n_anc, -1, -1, -1),
        dim=-1,
        index=anchors[..., j].unsqueeze(-1).clamp(0, n_act - 1),
    ).squeeze(-1)
    q_old_i = (
        q_old_i.unsqueeze(1)
        .unsqueeze(1)
        .expand(n_anc, n_act, n_act, b, t)
        .reshape(n_anc * n_act * n_act, b, t)
    )
    q_old_j = (
        q_old_j.unsqueeze(1)
        .unsqueeze(1)
        .expand(n_anc, n_act, n_act, b, t)
        .reshape(n_anc * n_act * n_act, b, t)
    )
    score = (q_new_i - q_old_i).abs() + (q_new_j - q_old_j).abs()
    return cand, valid, score


def build_recorded_neighborhood(
    live_q: torch.Tensor,
    bar_q: torch.Tensor,
    avail: torch.Tensor,
    executed: torch.Tensor,
    support: RecordedJoints,
    *,
    k: int,
    max_hamming: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cand [K,B,T,N] and valid [K,B,T]. Every valid joint is in S_off."""
    live_g = product_greedy(live_q, avail)
    bar_g = product_greedy(bar_q, avail)
    exec_u = executed.long()
    if exec_u.dim() == live_g.dim() + 1:
        exec_u = exec_u.squeeze(-1)
    anchors = torch.stack([bar_g, live_g, exec_u], dim=0)
    h0_valid = joints_available(anchors, avail) & support.contains(anchors)
    h0_score = torch.ones(anchors.shape[:-1], device=anchors.device) + 1.0e6
    joints, valid, scores = anchors, h0_valid, h0_score
    if max_hamming >= 1:
        h1_j, h1_v, h1_s = hamming1(anchors, avail, bar_q, support)
        joints = torch.cat([joints, h1_j], dim=0)
        valid = torch.cat([valid, h1_v], dim=0)
        scores = torch.cat([scores, h1_s], dim=0)
    selected, sel_valid, nfill = pack_unique_topk(joints, valid, scores, support, k)
    n_agents = anchors.shape[-1]
    if max_hamming >= 2 and n_agents >= 2 and bool((nfill < k).any().item()):
        for i in range(n_agents):
            if bool((nfill >= k).all().item()):
                break
            for j in range(i + 1, n_agents):
                if bool((nfill >= k).all().item()):
                    break
                h2_j, h2_v, h2_s = hamming2(anchors, avail, bar_q, support, i, j)
                selected, sel_valid, nfill = pack_unique_topk(
                    torch.cat([selected, h2_j], dim=0),
                    torch.cat([sel_valid, h2_v], dim=0),
                    torch.cat([sel_valid.float() * 1.0e6, h2_s], dim=0),
                    support,
                    k,
                )
    return selected, sel_valid
