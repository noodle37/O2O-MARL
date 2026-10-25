#!/usr/bin/env python3
"""Insight C: is the next-state relation structure a joint (relational) function?

The online hypothesis is that the next state's relational features are not
decomposable into independent per-agent predictors q_i(z_i, a_i); they require a
centralized decoder g({z_i, a_i}, s) whose output feeds a decentralized q_i
through stop-gradients.  This script tests the premise directly on held-out
offline Vault episodes:

  additive  sum_i f_i(z_i, a_i, s)              (independent per-agent terms)
  central   g(s, {e_i}, {pair(e_i, e_j)})       (relational decoder)
  no_pair   g(s, {e_i})                         (ablation)

Targets are next-step relational features parsed from the global state:
enemy health, enemy position change, ally health, and team reward.  R^2 is in
standardized target units, so it is comparable across target groups.

The structural check perturbs one agent's action at evaluation time and measures
how much the *other* agents' predicted health changes.  The additive model has
zero cross-agent sensitivity by construction; the central model need not.

Usage:
  CUDA_VISIBLE_DEVICES=3 python scripts/run_relational_premise.py \
      --config configs/pv_base_5m_vs_6m.yaml --epochs 40 \
      --out checkpoints/relational_premise/5m_vs_6m.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.factory import load_vault_dataset
from utils.paths import ensure_project_root, resolve_path


TARGET_GROUPS = ("enemy_health", "enemy_xy_delta", "ally_health", "team_reward")
GROUP_SLICES = {
    "enemy_health": (0, 6),
    "enemy_xy_delta": (6, 18),
    "ally_health": (18, 23),
    "team_reward": (23, 24),
}


def mlp(sizes: List[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class AdditiveModel(nn.Module):
    """Sum of independent per-agent predictors, each conditioned on the state."""

    def __init__(self, obs_dim: int, state_dim: int, n_agents: int, n_actions: int, out_dim: int):
        super().__init__()
        self.enc = mlp([obs_dim + n_actions + n_agents + state_dim, 128, 128])
        self.head = nn.Linear(128, out_dim)

    def forward(self, obs, act, agent_id, state):
        b, a, _ = obs.shape
        s = state.unsqueeze(1).expand(b, a, state.shape[-1])
        e = self.enc(torch.cat([obs, act, agent_id, s], dim=-1))
        return self.head(e).sum(dim=1)


class CentralModel(nn.Module):
    """Permutation-invariant agent encoder plus explicit pairwise interactions."""

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_agents: int,
        n_actions: int,
        out_dim: int,
        use_pairs: bool = True,
    ):
        super().__init__()
        self.use_pairs = use_pairs
        self.n_agents = n_agents
        self.enc = mlp([obs_dim + n_actions + n_agents, 128, 128])
        self.state_enc = mlp([state_dim, 128, 128])
        if use_pairs:
            self.pair_enc = mlp([256, 128, 128])
            head_in = 128 * 3 + 128
        else:
            head_in = 128 * 3
        self.head = mlp([head_in, 128, out_dim])

    def forward(self, obs, act, agent_id, state):
        e = self.enc(torch.cat([obs, act, agent_id], dim=-1))
        pooled = torch.cat([e.sum(dim=1), e.max(dim=1).values], dim=-1)
        s = self.state_enc(state)
        pieces = [s, pooled]
        if self.use_pairs:
            pair = []
            for i, j in itertools.combinations(range(self.n_agents), 2):
                pair.append(self.pair_enc(torch.cat([e[:, i], e[:, j]], dim=-1)))
            pieces.append(torch.stack(pair, dim=1).sum(dim=1))
        return self.head(torch.cat(pieces, dim=-1))


def parse_state(state: torch.Tensor, n_agents: int, n_actions: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (allies [.., A, 4], enemies [.., E, 3]) for the 5m_vs_6m layout."""
    nf_al, nf_en = 4, 3
    n_enemies = (state.shape[-1] - n_agents * nf_al - n_agents * n_actions) // nf_en
    allies = state[..., : n_agents * nf_al].reshape(*state.shape[:-1], n_agents, nf_al)
    base = n_agents * nf_al
    enemies = state[..., base : base + n_enemies * nf_en].reshape(
        *state.shape[:-1], n_enemies, nf_en
    )
    return allies, enemies


def build_transitions(dataset, n_agents: int, n_actions: int, device):
    obs = dataset.obs.to(device)
    state = dataset.state.to(device)
    actions = dataset.actions.squeeze(-1).long().to(device)
    reward = dataset.reward.squeeze(-1).to(device)
    filled = dataset.filled.squeeze(-1).to(device)
    valid = (filled[:, :-1] > 0.5) & (filled[:, 1:] > 0.5)
    ep_idx, t_idx = valid.nonzero(as_tuple=True)
    next_state = state[ep_idx, t_idx + 1]
    cur_allies, cur_enemies = parse_state(state[ep_idx, t_idx], n_agents, n_actions)
    nxt_allies, nxt_enemies = parse_state(next_state, n_agents, n_actions)
    targets = torch.cat(
        [
            nxt_enemies[..., 0],
            (nxt_enemies[..., 1:] - cur_enemies[..., 1:]).reshape(-1, nxt_enemies.shape[1] * 2),
            nxt_allies[..., 0],
            reward[ep_idx, t_idx].unsqueeze(-1),
        ],
        dim=-1,
    )
    return {
        "obs": obs[ep_idx, t_idx],
        "state": state[ep_idx, t_idx],
        "actions": actions[ep_idx, t_idx],
        "targets": targets,
        "episodes": ep_idx,
        "t": t_idx,
    }


def one_hot_actions(actions: torch.Tensor, n_actions: int) -> torch.Tensor:
    return F.one_hot(actions, n_actions).float()


def agent_id_tensor(batch: int, n_agents: int, device) -> torch.Tensor:
    ids = torch.arange(n_agents, device=device)
    return F.one_hot(ids, n_agents).float().unsqueeze(0).expand(batch, -1, -1)


def evaluate(model, data, n_actions: int, n_agents: int) -> Dict[str, float]:
    model.eval()
    preds, targs = [], []
    with torch.inference_mode():
        for start in range(0, len(data["obs"]), 8192):
            sl = slice(start, start + 8192)
            pred = model(
                data["obs"][sl],
                one_hot_actions(data["actions"][sl], n_actions),
                agent_id_tensor(len(data["obs"][sl]), n_agents, data["obs"].device),
                data["state"][sl],
            )
            preds.append(pred)
            targs.append(data["targets"][sl])
    pred = torch.cat(preds)
    targ = torch.cat(targs)
    mse_n = ((pred - targ) ** 2).mean(dim=0)
    var = targ.var(dim=0, unbiased=False).clamp_min(1e-8)
    r2 = 1.0 - ((pred - targ) ** 2).mean(dim=0) / var
    out = {}
    for name in TARGET_GROUPS:
        lo, hi = GROUP_SLICES[name]
        out[name] = {
            "mse_standardized": float(mse_n[lo:hi].mean()),
            "r2": float(r2[lo:hi].mean()),
        }
    return out


def cross_agent_sensitivity(model, data, n_actions: int, n_agents: int, device, seed: int = 0):
    """Measure genuine pairwise action interaction.

    interaction(j,k) = g(a_j', a_k') - g(a_j') - g(a_k') + g(a)
    is exactly zero for the additive model (each agent contributes one term) and
    nonzero only when the decoder represents joint action interactions.
    """
    rng = np.random.default_rng(seed)
    n = min(512, len(data["obs"]))
    idx = torch.as_tensor(rng.choice(len(data["obs"]), size=n, replace=False), device=device)
    obs, state, act = data["obs"][idx], data["state"][idx], data["actions"][idx]
    ids = agent_id_tensor(n, n_agents, device)
    with torch.inference_mode():
        base = model(obs, one_hot_actions(act, n_actions), ids, state)
        single_effects = {g: [] for g in TARGET_GROUPS}
        interactions = {g: [] for g in TARGET_GROUPS}
        for _ in range(8):
            j, k = rng.choice(n_agents, size=2, replace=False)
            alt_j, alt_k = act.clone(), act.clone()
            alt_j[:, j] = torch.as_tensor(
                rng.integers(0, n_actions, size=n), device=device, dtype=torch.long
            )
            alt_k[:, k] = torch.as_tensor(
                rng.integers(0, n_actions, size=n), device=device, dtype=torch.long
            )
            alt_both = alt_j.clone()
            alt_both[:, k] = alt_k[:, k]
            pj = model(obs, one_hot_actions(alt_j, n_actions), ids, state)
            pk = model(obs, one_hot_actions(alt_k, n_actions), ids, state)
            pb = model(obs, one_hot_actions(alt_both, n_actions), ids, state)
            delta = pb - pj - pk + base
            for g in TARGET_GROUPS:
                lo, hi = GROUP_SLICES[g]
                interactions[g].append(float(delta[:, lo:hi].abs().mean()))
                single_effects[g].append(float((pj[:, lo:hi] - base[:, lo:hi]).abs().mean()))
    return {
        "single_action_effect": {
            g: float(np.mean(single_effects[g])) for g in TARGET_GROUPS
        },
        "pair_interaction": {g: float(np.mean(interactions[g])) for g in TARGET_GROUPS},
        "interaction_to_single": {
            g: float(np.mean(interactions[g]) / max(1e-8, np.mean(single_effects[g])))
            for g in TARGET_GROUPS
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/pv_base_5m_vs_6m.yaml")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--split-seed", type=int, default=0, help="episode-shuffle seed for the 80/20 split")
    ap.add_argument("--model-seed", type=int, default=0, help="torch init / batch-order seed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    ensure_project_root()
    cfg = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides)),
        resolve=True,
    )
    dataset = load_vault_dataset(cfg)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n_agents = int(dataset.stats.n_agents)
    n_actions = int(dataset.stats.n_actions)
    print(
        f"[C] episodes={dataset.n_episodes} transitions={int((dataset.filled[:, :-1] > 0.5).sum())}",
        flush=True,
    )
    data = build_transitions(dataset, n_agents, n_actions, device)

    rng = np.random.default_rng(args.split_seed)
    torch.manual_seed(args.model_seed)
    ep_ids = np.arange(dataset.n_episodes)
    rng.shuffle(ep_ids)
    n_train = int(0.8 * dataset.n_episodes)
    train_eps = torch.as_tensor(ep_ids[:n_train], device=device)
    test_eps = torch.as_tensor(ep_ids[n_train:], device=device)
    ep = data["episodes"]
    train_mask = torch.isin(ep, train_eps)
    test_mask = torch.isin(ep, test_eps)
    train = {k: v[train_mask] for k, v in data.items() if k not in ("episodes", "t")}
    test = {k: v[test_mask] for k, v in data.items() if k not in ("episodes", "t")}
    mean = train["targets"].mean(dim=0)
    std = train["targets"].std(dim=0).clamp_min(1e-6)
    train_n = dict(train)
    train_n["targets"] = (train["targets"] - mean) / std
    test_n = dict(test)
    test_n["targets"] = (test["targets"] - mean) / std
    print(f"[C] train={len(train['obs'])} test={len(test['obs'])}", flush=True)

    obs_dim = int(dataset.stats.obs_dim)
    state_dim = int(dataset.stats.state_dim)
    out_dim = int(train["targets"].shape[1])
    models = {
        "additive": AdditiveModel(obs_dim, state_dim, n_agents, n_actions, out_dim),
        "central_no_pair": CentralModel(obs_dim, state_dim, n_agents, n_actions, out_dim, use_pairs=False),
        "central": CentralModel(obs_dim, state_dim, n_agents, n_actions, out_dim, use_pairs=True),
    }
    results = {
        "n_episodes": int(dataset.n_episodes),
        "n_train": int(len(train["obs"])),
        "n_test": int(len(test["obs"])),
        "n_train_episodes": int(n_train),
        "n_test_episodes": int(dataset.n_episodes - n_train),
        "split_seed": int(args.split_seed),
        "model_seed": int(args.model_seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "target_groups": list(TARGET_GROUPS),
        "models": {},
        "cross_agent_sensitivity": {},
    }
    n_train = len(train["obs"])
    for name, model in models.items():
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        t0 = time.time()
        for epoch in range(args.epochs):
            perm = torch.as_tensor(rng.permutation(n_train), device=device)
            total = 0.0
            for start in range(0, n_train, args.batch_size):
                idx = perm[start : start + args.batch_size]
                pred = model(
                    train_n["obs"][idx],
                    one_hot_actions(train_n["actions"][idx], n_actions),
                    agent_id_tensor(len(idx), n_agents, device),
                    train_n["state"][idx],
                )
                loss = ((pred - train_n["targets"][idx]) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()
                total += float(loss.detach()) * len(idx)
            if (epoch + 1) % 10 == 0:
                print(f"[C] {name} epoch={epoch+1} loss={total/n_train:.5f}", flush=True)
        results["models"][name] = {
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "train_seconds": round(time.time() - t0, 1),
            "test": evaluate(model, test_n, n_actions, n_agents),
        }
        results["cross_agent_sensitivity"][name] = cross_agent_sensitivity(
            model, test, n_actions, n_agents, device
        )
        print(
            f"[C] {name} R2="
            + " ".join(
                f"{g}:{results['models'][name]['test'][g]['r2']:.3f}" for g in TARGET_GROUPS
            ),
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"[C] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
