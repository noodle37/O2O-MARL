# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Offline sub-team knowledge units.

A unit kappa = (C_kappa, H_kappa, pi_off^C_kappa) is discovered without
enumerating 2^N subsets:

* pair coupling  I_ij = || d^2 Q_tot^off / d q_i d q_j ||  is measured on the
  frozen offline mixer over offline states.  The derivative is taken with
  respect to the per-agent utilities, which is what the mixer actually
  consumes; a monotone mixer can still couple two agents through its
  nonlinear hidden layer.
* candidate sub-teams of size 1..3 are ranked by within-team coupling and a
  size-balanced quota keeps multi-agent units in the dictionary.
* each unit keeps a start-domain discriminator (diagonal Mahalanobis distance
  of the concatenated member observations, thresholded on offline data) and a
  local H-step return memory.

The offline policy snippet is the frozen offline critic's softmax policy.  The
unit does not store a replay buffer: the validity module re-evaluates the
snippet against the current critic at every online step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


@dataclass
class VCKRParams:
    horizon: int = 10
    max_units: int = 10
    unit_sizes: Tuple[int, ...] = (1, 2, 3)
    coupling_batches: int = 8
    coupling_batch_size: int = 64
    coupling_stride: int = 4
    min_cohesion_frac: float = 0.25
    domain_quantile: float = 0.9
    domain_alpha_temp: float = 0.25
    require_domain: bool = True
    policy_temp: float = 1.0
    exec_policy_temp: float = 1.0
    beta_tv: float = 0.2
    temp_frac: float = 0.5
    delta_clip: float = 5.0
    bonus_scale: float = 1.0
    max_selected_units: int = 2
    gate_hidden: int = 64
    gate_lr: float = 1e-3
    gate_threshold: float = 0.5
    gate_pos_weight: float = 2.0
    gate_bias_init: float = -4.0
    lock_horizon: int = 10

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "VCKRParams":
        block = cfg.get("vckr") or {}
        sizes = tuple(int(x) for x in block.get("unit_sizes", (1, 2, 3)))
        if not sizes or min(sizes) < 1:
            raise ValueError(f"vckr.unit_sizes must be positive ints, got {sizes}")
        return cls(
            horizon=int(block.get("horizon", 10)),
            max_units=int(block.get("max_units", 10)),
            unit_sizes=sizes,
            coupling_batches=int(block.get("coupling_batches", 8)),
            coupling_batch_size=int(block.get("coupling_batch_size", 64)),
            coupling_stride=max(1, int(block.get("coupling_stride", 4))),
            min_cohesion_frac=float(block.get("min_cohesion_frac", 0.25)),
            domain_quantile=float(block.get("domain_quantile", 0.9)),
            domain_alpha_temp=float(block.get("domain_alpha_temp", 0.25)),
            require_domain=bool(block.get("require_domain", True)),
            policy_temp=float(block.get("policy_temp", 1.0)),
            exec_policy_temp=float(block.get("exec_policy_temp", 1.0)),
            beta_tv=float(block.get("beta_tv", 0.2)),
            temp_frac=float(block.get("temp_frac", 0.5)),
            delta_clip=float(block.get("delta_clip", 5.0)),
            bonus_scale=float(block.get("bonus_scale", 1.0)),
            max_selected_units=int(block.get("max_selected_units", 2)),
            gate_hidden=int(block.get("gate_hidden", 64)),
            gate_lr=float(block.get("gate_lr", 1e-3)),
            gate_threshold=float(block.get("gate_threshold", 0.5)),
            gate_pos_weight=float(block.get("gate_pos_weight", 2.0)),
            gate_bias_init=float(block.get("gate_bias_init", -4.0)),
            lock_horizon=int(block.get("lock_horizon", 10)),
        )


@dataclass
class SubteamKnowledge:
    units: List[Tuple[int, ...]]
    horizon: int
    coupling: torch.Tensor  # [N, N], CPU, diagonal zero
    membership: torch.Tensor  # [U, N] bool on the learner device
    dom_mean: List[torch.Tensor]
    dom_var: List[torch.Tensor]
    dom_thr: torch.Tensor  # [U]
    cohesion: torch.Tensor  # [U]
    return_mean: float
    return_std: float
    delta_temp: float
    n_agents: int
    obs_dim: int
    info: Dict[str, Any] = field(default_factory=dict)

    def domain_alpha(self, obs_flat: torch.Tensor, params: VCKRParams) -> torch.Tensor:
        """Per-unit start-domain score in [0, 1] for flattened observations [M, N, obs]."""
        alphas = []
        for u, members in enumerate(self.units):
            x = obs_flat[:, list(members), :].reshape(obs_flat.shape[0], -1)
            d2 = ((x - self.dom_mean[u]) ** 2 / self.dom_var[u]).mean(dim=-1)
            thr = float(self.dom_thr[u].item())
            scale = max(params.domain_alpha_temp * thr, 1e-6)
            alphas.append(torch.sigmoid((thr - d2) / scale))
        return torch.stack(alphas, dim=0)


def _h_step_returns(
    reward: torch.Tensor,
    filled: torch.Tensor,
    horizon: int,
    gamma: float,
) -> torch.Tensor:
    """Undiscounted-mask H-step return per step, [B, T]. Termination is ignored
    inside the window; the memory is a coarse prior, not a Bellman target."""
    b, t = reward.shape
    out = torch.zeros_like(reward)
    for h in range(horizon):
        if h >= t:
            break
        disc = gamma**h
        if h == 0:
            out += reward
        else:
            out[:, : t - h] += disc * reward[:, h:]
    return out * filled


@torch.no_grad()
def _sample_features(
    learner: Any,
    dataset: Any,
    params: VCKRParams,
    rng: np.random.Generator,
) -> Dict[str, torch.Tensor]:
    """Flattened offline features for coupling, domain and return memory."""
    obs_rows: List[torch.Tensor] = []
    state_rows: List[torch.Tensor] = []
    qbar_rows: List[torch.Tensor] = []
    ret_rows: List[torch.Tensor] = []
    temp = max(float(params.policy_temp), 1e-6)
    for _ in range(max(1, int(params.coupling_batches))):
        batch = dataset.sample(
            int(params.coupling_batch_size), rng, device=learner.device
        )
        off_out = learner._offline_unroll(batch)  # [B, T, N, A]
        avail = batch["avail_actions"].bool()
        logits = off_out.masked_fill(~avail, -1e9) / temp
        pi = torch.softmax(logits, dim=-1)
        qbar = (pi * off_out).sum(dim=-1)  # [B, T, N]

        filled = batch["filled"][:, :, 0] > 0.5  # [B, T]
        t_axis = batch["obs"].shape[1]
        idx = torch.arange(0, t_axis, int(params.coupling_stride), device=filled.device)
        valid = filled[:, idx].reshape(-1)
        if not bool(valid.any()):
            continue
        obs_rows.append(batch["obs"][:, idx].reshape(-1, *batch["obs"].shape[2:])[valid])
        state_rows.append(batch["state"][:, idx].reshape(-1, *batch["state"].shape[2:])[valid])
        qbar_rows.append(qbar[:, idx].reshape(-1, qbar.shape[-1])[valid])

        hret = _h_step_returns(
            batch["reward"][:, :, 0], filled.float(), int(params.horizon), float(learner.gamma)
        )
        ret_rows.append(hret[:, idx].reshape(-1)[valid])

    if not obs_rows:
        raise RuntimeError("VCKR knowledge extraction found no valid offline steps")
    return {
        "obs": torch.cat(obs_rows, dim=0),
        "state": torch.cat(state_rows, dim=0),
        "qbar": torch.cat(qbar_rows, dim=0),
        "hret": torch.cat(ret_rows, dim=0),
    }


def _coupling_matrix(
    learner: Any,
    features: Dict[str, torch.Tensor],
    chunk: int = 2048,
) -> torch.Tensor:
    """Mean |d^2 Q_tot^off / d q_i d q_j| over offline states."""
    n = int(learner.n_agents)
    qbar = features["qbar"]
    states = features["state"]
    total = int(qbar.shape[0])
    acc = torch.zeros(n, n, device=qbar.device)
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        q = qbar[start:end].detach().clone().requires_grad_(True)
        st = states[start:end]
        out = learner.offline_mixer(q.unsqueeze(1), st.unsqueeze(1))
        grad = torch.autograd.grad(out.sum(), q, create_graph=True)[0]
        cols = []
        for j in range(n):
            cols.append(
                torch.autograd.grad(
                    grad[:, j].sum(),
                    q,
                    retain_graph=(j < n - 1),
                    create_graph=False,
                )[0]
            )
        hess = torch.stack(cols, dim=-1)  # [m, N, N]
        acc += hess.abs().sum(dim=0)
    acc = acc / max(total, 1)
    acc = 0.5 * (acc + acc.t())
    acc.fill_diagonal_(0.0)
    return acc.detach().cpu()


def _candidate_units(
    coupling: torch.Tensor, params: VCKRParams
) -> Tuple[List[Tuple[int, ...]], List[float]]:
    n = int(coupling.shape[0])
    cands: List[Tuple[float, Tuple[int, ...]]] = []
    for size in params.unit_sizes:
        if size > n:
            continue
        for combo in combinations(range(n), size):
            if size == 1:
                cohesion = float(coupling[combo[0]].mean().item())
            else:
                sub = coupling[list(combo)][:, list(combo)]
                cohesion = float(sub.sum().item() / (size * (size - 1)))
            cands.append((cohesion, tuple(int(i) for i in combo)))
    if not cands:
        raise ValueError("no candidate sub-teams for the configured sizes")
    cands.sort(key=lambda item: item[0], reverse=True)
    max_c = max(cands[0][0], 0.0)
    if max_c <= 1e-8:
        # Additive value: no coordination to discover; fall back to singletons.
        units = [(i,) for i in range(n)][: params.max_units]
        return units, [0.0] * len(units)

    threshold = float(params.min_cohesion_frac) * max_c
    kept = [c for c in cands if c[0] >= threshold]
    by_size: Dict[int, List[Tuple[float, Tuple[int, ...]]]] = {}
    for item in kept:
        by_size.setdefault(len(item[1]), []).append(item)
    quota = max(1, int(params.max_units) // max(1, len(by_size)))
    picked: List[Tuple[float, Tuple[int, ...]]] = []
    for size in sorted(by_size):
        picked.extend(by_size[size][:quota])
    seen = {item[1] for item in picked}
    for item in kept:
        if len(picked) >= params.max_units:
            break
        if item[1] in seen:
            continue
        picked.append(item)
        seen.add(item[1])
    picked = picked[: params.max_units]
    return [item[1] for item in picked], [float(item[0]) for item in picked]


def _fit_domain(
    features: Dict[str, torch.Tensor],
    units: Sequence[Tuple[int, ...]],
    params: VCKRParams,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    obs = features["obs"]
    means: List[torch.Tensor] = []
    vars_: List[torch.Tensor] = []
    thrs: List[float] = []
    for members in units:
        x = obs[:, list(members), :].reshape(obs.shape[0], -1)
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False).clamp(min=1e-6)
        d2 = ((x - mean) ** 2 / var).mean(dim=-1)
        thr = float(torch.quantile(d2, float(params.domain_quantile)).item())
        means.append(mean)
        vars_.append(var)
        thrs.append(max(thr, 1e-6))
    return means, vars_, torch.tensor(thrs, device=obs.device)


def build_subteam_knowledge(
    learner: Any,
    dataset: Any,
    params: VCKRParams,
    seed: int = 0,
) -> Tuple[SubteamKnowledge, Dict[str, Any]]:
    """Extract the offline dictionary once, before online interaction starts."""
    if learner.offline_q_network is None or learner.offline_mixer is None:
        raise RuntimeError("VCKR needs the frozen offline critic and mixer loaded")
    rng = np.random.default_rng(int(seed) + 101)
    features = _sample_features(learner, dataset, params, rng)
    coupling = _coupling_matrix(learner, features)
    units, cohesion = _candidate_units(coupling, params)
    dom_mean, dom_var, dom_thr = _fit_domain(features, units, params)

    hret = features["hret"]
    ret_mean = float(hret.mean().item())
    ret_std = float(hret.std().item()) if hret.numel() > 1 else 0.0
    delta_temp = max(float(params.temp_frac) * ret_std, 0.1)

    membership = torch.tensor(
        [[1.0 if i in members else 0.0 for i in range(learner.n_agents)] for members in units],
        device=learner.device,
    ).bool()
    knowledge = SubteamKnowledge(
        units=list(units),
        horizon=int(params.horizon),
        coupling=coupling,
        membership=membership,
        dom_mean=dom_mean,
        dom_var=dom_var,
        dom_thr=dom_thr,
        cohesion=torch.tensor(cohesion, device=learner.device),
        return_mean=ret_mean,
        return_std=ret_std,
        delta_temp=delta_temp,
        n_agents=int(learner.n_agents),
        obs_dim=int(learner.obs_dim),
    )

    info: Dict[str, Any] = {
        "vckr/n_units": float(len(units)),
        "vckr/coupling_mean": float(coupling.mean().item()),
        "vckr/coupling_max": float(coupling.max().item()),
        "vckr/return_mean": ret_mean,
        "vckr/return_std": ret_std,
        "vckr/delta_temp": delta_temp,
        "vckr/units": ";".join("-".join(str(i) for i in u) for u in units),
    }
    for u, members in enumerate(units):
        info[f"vckr/unit{u}_cohesion"] = float(cohesion[u])
        info[f"vckr/unit{u}_domain_thr"] = float(dom_thr[u].item())
    for i in range(learner.n_agents):
        for j in range(i + 1, learner.n_agents):
            info[f"vckr/coupling_{i}_{j}"] = float(coupling[i, j].item())
    knowledge.info = info
    return knowledge, info
