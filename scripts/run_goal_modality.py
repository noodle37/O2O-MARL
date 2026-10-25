#!/usr/bin/env python3
"""Insight D: are H-step cooperative goals multi-modal in their implementation?

Offline premise check.  For a horizon H, quantize the state change

    g = Quantize(phi(s_{t+H}) - phi(s_t))

into K goal clusters, then ask, on held-out-style offline transitions:

  * support / reachability breadth of each goal
  * action entropy H(a_i | state_bin) vs H(a_i | state_bin, goal)
  * number of distinct joint-action implementations per (state_bin, goal)
  * share of H-step return variance that remains *within* a (state_bin, goal)
  * temporal persistence of the goal label (does it describe a plan?)
  * use-benefit: H-step return by goal

If the goal absorbs most of the action information and within-goal return
variance is small, a goal-conditioned decoder loses nothing.  If action entropy
and within-goal return variance stay high, goals are multi-modal and the decoder
must preserve implementation information.

Usage:
  python scripts/run_goal_modality.py --config configs/pv_base_5m_vs_6m.yaml \
      --out checkpoints/goal_modality/5m_vs_6m.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from omegaconf import OmegaConf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.factory import load_vault_dataset
from utils.paths import ensure_project_root


def parse_state(state: np.ndarray, n_agents: int, n_actions: int):
    """Allies [.., A, 4], enemies [.., E, 3] for the standard SMAC state layout."""
    nf_al, nf_en = 4, 3
    n_en = (state.shape[-1] - n_agents * nf_al - n_agents * n_actions) // nf_en
    allies = state[..., : n_agents * nf_al].reshape(*state.shape[:-1], n_agents, nf_al)
    base = n_agents * nf_al
    enemies = state[..., base : base + n_en * nf_en].reshape(*state.shape[:-1], n_en, nf_en)
    return allies, enemies


def phi_features(state: np.ndarray, n_agents: int, n_actions: int) -> np.ndarray:
    allies, enemies = parse_state(state, n_agents, n_actions)
    al_h, al_xy = allies[..., 0], allies[..., 1:3]
    en_h, en_xy = enemies[..., 0], enemies[..., 1:3]
    alive = np.stack([(al_h > 0).sum(-1), (en_h > 0).sum(-1)], axis=-1).astype(np.float32)
    # Alive-aware pairwise distances (zeros for dead units; alive flags are separate).
    d_ae = np.linalg.norm(al_xy[:, :, None, :] - en_xy[:, None, :, :], axis=-1)
    d_aa = np.linalg.norm(al_xy[:, :, None, :] - al_xy[:, None, :, :], axis=-1)
    aa_idx = np.triu_indices(n_agents, k=1)
    feats = [
        al_h,
        al_xy.reshape(len(state), -1),
        en_h,
        en_xy.reshape(len(state), -1),
        alive,
        d_ae.reshape(len(state), -1),
        d_aa[:, aa_idx[0], aa_idx[1]],
    ]
    return np.concatenate(feats, axis=-1).astype(np.float32)


def goal_slices(n_agents: int, n_enemies: int) -> List[Tuple[str, slice]]:
    # phi layout from phi_features
    out, off = [], 0
    sizes = {
        "ally_health": n_agents,
        "ally_xy": n_agents * 2,
        "enemy_health": n_enemies,
        "enemy_xy": n_enemies * 2,
        "alive": 2,
        "d_ally_enemy": n_agents * n_enemies,
        "d_ally_ally": n_agents * (n_agents - 1) // 2,
    }
    for name, size in sizes.items():
        out.append((name, slice(off, off + size)))
        off += size
    return out


def entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return float("nan")
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def build_horizon_data(dataset, H: int, gamma: float, n_agents: int, n_actions: int):
    filled = dataset.filled.squeeze(-1).cpu().numpy() > 0.5
    reward = dataset.reward.squeeze(-1).cpu().numpy()
    state = dataset.state.cpu().numpy()
    actions = dataset.actions.squeeze(-1).long().cpu().numpy()
    avail = dataset.avail_actions.cpu().numpy()
    N, T = filled.shape
    ep_idx, t_idx = np.nonzero(filled[:, : T - H])
    valid = filled[ep_idx, t_idx + H]
    ep_idx, t_idx = ep_idx[valid], t_idx[valid]
    # discounted H-step return
    ret = np.zeros(len(ep_idx), dtype=np.float32)
    for k in range(H):
        ret += (gamma**k) * reward[ep_idx, t_idx + k]
    start = state[ep_idx, t_idx]
    end = state[ep_idx, t_idx + H]
    phi0 = phi_features(start, n_agents, n_actions)
    phi1 = phi_features(end, n_agents, n_actions)
    delta = phi1 - phi0
    act = actions[ep_idx, t_idx]
    legal = avail[ep_idx, t_idx]
    alive = start[:, : n_agents * 4].reshape(-1, n_agents, 4)[:, :, 0] > 0
    allies0, enemies0 = parse_state(start, n_agents, n_actions)
    allies1, enemies1 = parse_state(end, n_agents, n_actions)
    deltas = {
        "ally_health": allies1[..., 0] - allies0[..., 0],
        "enemy_health": enemies1[..., 0] - enemies0[..., 0],
        "ally_alive_change": (allies1[..., 0] > 0).sum(-1) - (allies0[..., 0] > 0).sum(-1),
        "enemy_alive_change": (enemies1[..., 0] > 0).sum(-1) - (enemies0[..., 0] > 0).sum(-1),
    }
    return {
        "ep": ep_idx,
        "t": t_idx,
        "phi0": phi0,
        "delta": delta,
        "ret": ret,
        "actions": act,
        "legal": legal,
        "alive": alive,
        "deltas": deltas,
    }


def standardize(train: np.ndarray, other: np.ndarray):
    mean = train.mean(0)
    std = train.std(0).clip(1e-6)
    return (train - mean) / std, (other - mean) / std, mean, std


def analyse_horizon(data, H: int, Ks: List[int], n_agents: int, n_actions: int, seed: int = 0):
    n = len(data["delta"])
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(n, size=min(60000, n), replace=False)
    d_fit = data["delta"][fit_idx]
    d_mean, d_std = d_fit.mean(0), d_fit.std(0).clip(1e-6)
    d_z = (data["delta"] - d_mean) / d_std
    p_fit = data["phi0"][fit_idx]
    p_mean, p_std = p_fit.mean(0), p_fit.std(0).clip(1e-6)
    p_z = (data["phi0"] - p_mean) / p_std

    # Coarse state bins for conditional action entropy.
    n_bins = 64
    bin_model = KMeans(n_clusters=n_bins, n_init=5, random_state=seed).fit(p_z[fit_idx])
    bins = bin_model.predict(p_z)

    out = {"horizon": H, "n_transitions": int(n), "Ks": {}}
    for K in Ks:
        km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(d_z[fit_idx])
        goal = km.predict(d_z)
        sample = rng.choice(n, size=min(10000, n), replace=False)
        sil = float(silhouette_score(d_z[sample], goal[sample], sample_size=min(5000, len(sample))))
        support = np.bincount(goal, minlength=K) / n

        # Entropy of factorised actions given (bin) and (bin, goal).
        h_bin_all: Dict[int, np.ndarray] = {}
        groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for idx in range(n):
            groups[(int(bins[idx]), int(goal[idx]))].append(idx)
        for b in range(n_bins):
            h_bin_all[b] = np.zeros(n_agents, dtype=np.float64)
            for i in range(n_agents):
                mask = (bins == b) & data["alive"][:, i]
                counts = np.bincount(data["actions"][mask, i], minlength=n_actions)
                h_bin_all[b][i] = entropy(counts) if mask.any() else np.nan

        w_sum = 0.0
        h_bin_w = 0.0
        h_goal_w = 0.0
        mult = []
        within_var = []
        within_weight = []
        group_rets = []
        group_rows = []
        kept_idx = []
        for (b, g), idxs in groups.items():
            if len(idxs) < 20:
                continue
            idxs_np = np.asarray(idxs)
            kept_idx.append(idxs_np)
            # per-agent conditional entropy
            h_agents = []
            for i in range(n_agents):
                alive_idx = idxs_np[data["alive"][idxs_np, i]]
                if len(alive_idx) < 5:
                    continue
                counts = np.bincount(data["actions"][alive_idx, i], minlength=n_actions)
                h_agents.append(entropy(counts))
            if not h_agents:
                continue
            h_g = float(np.mean(h_agents))
            h_b = float(np.nanmean(h_bin_all[b]))
            h_bin_w += len(idxs) * h_b
            h_goal_w += len(idxs) * h_g
            w_sum += len(idxs)
            # implementation multiplicity: distinct joint actions with >=10% mass
            joint, counts = np.unique(data["actions"][idxs_np], axis=0, return_counts=True)
            mult.append(int((counts / len(idxs_np) >= 0.1).sum()))
            r = data["ret"][idxs_np]
            within_var.append(float(r.var()))
            within_weight.append(len(idxs))
            group_rets.append(float(r.mean()))
            group_rows.append(
                {
                    "state_bin": int(b),
                    "goal": int(g),
                    "n": int(len(idxs)),
                    "action_entropy": h_g,
                    "state_only_action_entropy": h_b,
                    "goal_action_info": h_b - h_g,
                    "implementations_ge10pct": int(mult[-1]),
                    "return_mean": float(r.mean()),
                    "return_std": float(r.std()),
                }
            )
        h_bin_weighted = h_bin_w / max(1e-9, w_sum)
        h_goal_weighted = h_goal_w / max(1e-9, w_sum)
        total_var = float(data["ret"].var())
        within_var_weighted = float(np.average(within_var, weights=within_weight)) if within_var else float("nan")
        # State-only baseline on the same transitions, to see whether the goal
        # label explains variance beyond the coarse state bin.
        kept = np.concatenate(kept_idx) if kept_idx else np.array([], dtype=int)
        state_within, state_weight = [], []
        for b in np.unique(bins[kept]):
            mask = bins[kept] == b
            if mask.sum() < 20:
                continue
            state_within.append(float(data["ret"][kept][mask].var()))
            state_weight.append(int(mask.sum()))
        state_only_within = (
            float(np.average(state_within, weights=state_weight)) if state_within else float("nan")
        )

        # Temporal persistence of the label.
        lookup = {(int(e), int(t)): int(g) for e, t, g in zip(data["ep"], data["t"], goal)}
        same, total_pairs = 0, 0
        for (e, t), g in lookup.items():
            nxt = lookup.get((e, t + 1))
            if nxt is not None:
                total_pairs += 1
                same += int(nxt == g)
        observed = same / max(1, total_pairs)
        chance = float((support**2).sum())
        kappa = (observed - chance) / max(1e-9, 1.0 - chance)

        # Goal-level diagnostics and reachability breadth.
        goal_rows = []
        for g in range(K):
            mask = goal == g
            idxs = np.where(mask)[0]
            bins_with_support = len(set(int(x) for x in bins[idxs]))
            deltas_g = {name: float(values[idxs].mean()) for name, values in data["deltas"].items()}
            goal_rows.append(
                {
                    "goal": g,
                    "support": float(support[g]),
                    "n": int(mask.sum()),
                    "state_bins_with_support_ge10": int(
                        sum(1 for b in set(int(x) for x in bins[idxs]) if ((bins == b) & mask).sum() >= 10)
                    ),
                    "mean_h_return": float(data["ret"][idxs].mean()),
                    "std_h_return": float(data["ret"][idxs].std()),
                    **{f"mean_delta_{k}": v for k, v in deltas_g.items()},
                }
            )
        out["Ks"][str(K)] = {
            "silhouette": sil,
            "goal_support": [float(x) for x in support],
            "weighted_action_entropy_state_only": float(h_bin_weighted),
            "weighted_action_entropy_state_plus_goal": float(h_goal_weighted),
            "goal_action_info": float(h_bin_weighted - h_goal_weighted),
            "mean_implementations_ge10pct": float(np.mean(mult)) if mult else float("nan"),
            "within_group_return_variance": within_var_weighted,
            "total_return_variance": total_var,
            "within_group_variance_share": float(within_var_weighted / max(1e-9, total_var)),
            "state_only_within_group_variance": state_only_within,
            "state_only_within_group_variance_share": float(
                state_only_within / max(1e-9, total_var)
            ),
            "goal_variance_reduction_vs_state": float(
                (state_only_within - within_var_weighted) / max(1e-9, state_only_within)
            ),
            "goal_persistence_observed": float(observed),
            "goal_persistence_chance": chance,
            "goal_persistence_kappa": float(kappa),
            "n_eval_groups": len(group_rows),
            "goals": goal_rows,
            "example_groups": sorted(group_rows, key=lambda r: -r["n"])[:30],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/pv_base_5m_vs_6m.yaml")
    ap.add_argument("--horizons", nargs="+", type=int, default=[5, 10])
    ap.add_argument("--ks", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0, help="KMeans / sampling seed")
    ap.add_argument("--out", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    ensure_project_root()
    cfg = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides)),
        resolve=True,
    )
    dataset = load_vault_dataset(cfg)
    n_agents = int(dataset.stats.n_agents)
    n_actions = int(dataset.stats.n_actions)
    results = {
        "map": cfg["task"]["scenario"],
        "n_episodes": int(dataset.n_episodes),
        "n_agents": n_agents,
        "n_actions": n_actions,
        "gamma": float(args.gamma),
        "seed": int(args.seed),
        "horizons": {},
    }
    for H in args.horizons:
        data = build_horizon_data(dataset, H, args.gamma, n_agents, n_actions)
        print(
            f"[D] H={H} transitions={len(data['delta'])} "
            f"return_mean={data['ret'].mean():.3f}",
            flush=True,
        )
        results["horizons"][str(H)] = analyse_horizon(
            data, H, args.ks, n_agents, n_actions, seed=args.seed
        )
        top = results["horizons"][str(H)]["Ks"][str(args.ks[1] if len(args.ks) > 1 else args.ks[0])]
        print(
            f"[D] H={H} K={args.ks[1] if len(args.ks) > 1 else args.ks[0]} "
            f"H(a|s)={top['weighted_action_entropy_state_only']:.3f} "
            f"H(a|s,g)={top['weighted_action_entropy_state_plus_goal']:.3f} "
            f"impl={top['mean_implementations_ge10pct']:.2f} "
            f"within_var_share={top['within_group_variance_share']:.3f} "
            f"kappa={top['goal_persistence_kappa']:.3f}",
            flush=True,
        )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"[D] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
