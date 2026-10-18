#!/usr/bin/env python3
"""Measure offline Q_tot against recorded trajectory returns (SO2-style protocol).

Frozen checkpoint, no parameter updates, no environment interaction. For every
filled transition (s_t, u_t) of the offline dataset:

- Q_tot(s_t, u_t): the frozen critic's value of the recorded joint action. Its
  own bootstrap target is r + gamma * mixer(max_i Q_i(s', .)), i.e. the value of
  the critic's *own* per-agent-greedy continuation, not of the behaviour policy.
- G_t: the discounted return actually realised by the *behaviour* policy along
  the one recorded trajectory from t onward.

So Q_tot - G_t is "critic value minus realised behaviour return". It is NOT a
value-estimation error: the two sides estimate different things (critic's
implicit policy vs the data-collecting policy), and a perfectly converged QMIX
critic still shows Q_tot >= G_t wherever the behaviour action is not jointly
achievable at the per-agent maxima. Read every residual statistic below as a
gap, never as a bias/overestimate.

Ranking note: the dataset stores exactly one joint action per (episode, step)
and each step is a distinct state, so random pairs of transitions almost never
share a state. The random-pair concordance therefore measures cross-state
value association, not action ranking. Only the within-episode pairs (reported
separately) and the branch-rollout protocol in verify_local_rank_inversion.py
come closer, and neither can compare two actions at one state.

Outputs a single JSON file. Keys ending in ``_deprecated`` are the previous
names, kept only so older result files stay comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SC2PATH", "/home/jxy/StarCraftII")

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.factory import build_learner, load_vault_dataset
from utils.checkpoint import load_qmix_checkpoint
from utils.paths import ensure_project_root, resolve_path

Q_TIE = 1e-8
G_TIE = 1e-4


def _self_check() -> None:
    q = torch.tensor([1.0, 2.0, 3.0, 3.0])
    g = torch.tensor([1.0, 0.0, 5.0, 5.0 + 1e-9])
    counts = pairwise_outcomes(q, g, pairs=torch.tensor([[0, 1], [1, 2], [2, 3], [0, 0]]))
    # (0,1): dq>0 dg>0 -> agree; (1,2): dq>0 dg<0 -> invert;
    # (2,3): dg tie -> tie; (0,0): dq=0 -> tie
    if counts != {"agree": 1, "invert": 1, "tie": 2}:
        raise RuntimeError(f"pairwise self-check failed: {counts}")
    # pair_concordance must agree with pairwise_outcomes on the same inputs
    q_np = q.numpy(); g_np = g.numpy()
    idx = np.array([[0, 1], [1, 2], [2, 3], [0, 0]])
    a, i, t = pair_concordance(q_np[idx[:, 0]], g_np[idx[:, 0]], q_np[idx[:, 1]], g_np[idx[:, 1]])
    if (a, i, t) != (1, 1, 2):
        raise RuntimeError(f"pair_concordance self-check failed: {(a, i, t)}")


def pairwise_outcomes(
    q: torch.Tensor, g: torch.Tensor, pairs: torch.Tensor
) -> dict[str, int]:
    dq = (q[pairs[:, 0]] - q[pairs[:, 1]]).double()
    dg = (g[pairs[:, 0]] - g[pairs[:, 1]]).double()
    tie = (dq.abs() <= Q_TIE) | (dg.abs() <= G_TIE)
    same_sign = ((dq > 0) & (dg > 0)) | ((dq < 0) & (dg < 0))
    agree = int((~tie & same_sign).sum().item())
    invert = int((~tie & ~same_sign).sum().item())
    return {"agree": agree, "invert": invert, "tie": int(tie.sum().item())}


def return_to_go(reward: torch.Tensor, mask: torch.Tensor, gamma: float) -> torch.Tensor:
    """G_t = r_t + gamma*G_{t+1} over filled steps; padded rewards are zeroed."""
    r = reward * mask
    b, t = r.shape
    out = torch.zeros_like(r)
    acc = torch.zeros(b, device=r.device, dtype=r.dtype)
    for i in reversed(range(t)):
        acc = r[:, i] + gamma * acc
        out[:, i] = acc
    return out


def pair_concordance(q_a, g_a, q_b, g_b) -> tuple[int, int, int]:
    """Vectorised (agree, invert, tie) over two aligned arrays of Q and G."""
    q_a = np.asarray(q_a, dtype=np.float64); q_b = np.asarray(q_b, dtype=np.float64)
    g_a = np.asarray(g_a, dtype=np.float64); g_b = np.asarray(g_b, dtype=np.float64)
    dq = q_a - q_b
    dg = g_a - g_b
    tie = (np.abs(dq) <= Q_TIE) | (np.abs(dg) <= G_TIE)
    same = ((dq > 0) & (dg > 0)) | ((dq < 0) & (dg < 0))
    return int((~tie & same).sum()), int((~tie & ~same).sum()), int(tie.sum())


def within_episode_pairs(q_ep: np.ndarray, g_ep: np.ndarray, filled: np.ndarray):
    """All unordered pairs (i<j) inside each episode. Still cross-state.

    q_ep/g_ep/filled are [n_episodes, max_seq]. Each timestep is a different
    state, so these pairs compare different states; they only remove the
    between-episode component that dominates random cross-transition pairs.
    """
    qa, ga, qb, gb = [], [], [], []
    for e in range(filled.shape[0]):
        idx = np.flatnonzero(filled[e])
        if idx.size < 2:
            continue
        iu = np.triu_indices(idx.size, k=1)
        qq = q_ep[e, idx]; gg = g_ep[e, idx]
        qa.append(qq[iu[0]]); qb.append(qq[iu[1]])
        ga.append(gg[iu[0]]); gb.append(gg[iu[1]])
    if not qa:
        return (np.array([]),) * 4
    return np.concatenate(qa), np.concatenate(ga), np.concatenate(qb), np.concatenate(gb)


def clustered_se(values: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    """Standard error of the mean with a cluster-robust (grouped) variance."""
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    order = np.argsort(labels, kind="stable")
    labels = labels[order]; values = values[order]
    bounds = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    sums = np.add.reduceat(values, bounds)
    counts = np.diff(np.r_[bounds, values.size])
    means = sums / counts
    k = means.size
    if k < 2:
        return float("nan"), k
    return float(means.std(ddof=1) / np.sqrt(k)), k


def residual_decomposition(
    q: np.ndarray, g: np.ndarray, group: np.ndarray
) -> dict:
    """Split Q_tot - G_t into a real offset and a mean-zero shrinkage term.

    resid = (Q - mean_G_in_group) + (mean_G_in_group - G)

    The second term is the within-group outcome spread and averages to zero by
    construction, so it can never produce a non-zero mean. Whatever mean the
    residual has comes entirely from the first term: how far Q sits above the
    average realised return of the same decision. That offset is a property of
    which policy the critic evaluates, not an estimation error.
    """
    q = np.asarray(q, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    group = np.asarray(group)
    order = np.argsort(group, kind="stable")
    g_s, q_s = g[order], q[order]
    bounds = np.flatnonzero(np.r_[True, group[order][1:] != group[order][:-1]])
    g_mean = np.repeat(
        np.add.reduceat(g_s, bounds) / np.diff(np.r_[bounds, g_s.size]),
        np.diff(np.r_[bounds, g_s.size]),
    )
    term1 = q_s - g_mean
    term2 = g_mean - g_s
    resid = term1 + term2
    v1, v2, vr = float(term1.var()), float(term2.var()), float(resid.var())
    return {
        "grouping": "distinct (state, joint action) decisions",
        "n_groups": int(bounds.size),
        "residual_mean": float(resid.mean()),
        "offset_mean": float(term1.mean()),
        "offset_std": float(term1.std()),
        "shrinkage_mean": float(term2.mean()),
        "shrinkage_std": float(term2.std()),
        "offset_variance_share": None if vr <= 0 else v1 / vr,
        "shrinkage_variance_share": None if vr <= 0 else v2 / vr,
        "note": "offset_mean is the whole residual mean; shrinkage_mean is zero "
                "by construction and cannot explain any non-zero mean",
    }


def decision_group_ids(dataset) -> np.ndarray:
    """Group id per filled transition, in row-major (episode, time) order.

    Two occurrences share an id only when both the quantised global state and
    the recorded joint action are byte-identical, i.e. they are literally the
    same decision. This matches the flattening order of q_grid[filled].
    """
    filled = dataset.filled.squeeze(-1) > 0.5
    actions = dataset.actions.squeeze(-1).long()
    states = dataset.state
    n_ep, t_len = filled.shape
    flat_states = states.reshape(n_ep * t_len, -1).cpu().numpy()
    flat_actions = actions.reshape(n_ep * t_len, actions.shape[-1]).cpu().numpy()
    flat_filled = filled.reshape(n_ep * t_len).cpu().numpy()
    s = flat_states[flat_filled]
    a = flat_actions[flat_filled]
    s_view = np.ascontiguousarray(s).view([("", s.dtype)] * s.shape[1]).ravel()
    a_view = np.ascontiguousarray(a).view([("", a.dtype)] * a.shape[1]).ravel()
    pair_key = np.empty(s.shape[0], dtype=[("s", s_view.dtype), ("a", a_view.dtype)])
    pair_key["s"] = s_view
    pair_key["a"] = a_view
    _, pair_inv = np.unique(pair_key, return_inverse=True)
    return pair_inv.reshape(-1)


def dataset_support_stats(dataset, group_ids: np.ndarray | None = None) -> dict:
    """How many distinct (state, action) decisions the dataset actually holds.

    Occurrences overstate the sample: the same quantised SMAC state recurs
    across episodes, and the behaviour policy is near-deterministic on it.
    """
    if group_ids is None:
        group_ids = decision_group_ids(dataset)
    filled = dataset.filled.squeeze(-1) > 0.5
    states = dataset.state
    n_ep, t_len = filled.shape
    flat_states = states.reshape(n_ep * t_len, -1).cpu().numpy()
    flat_filled = filled.reshape(n_ep * t_len).cpu().numpy()
    s_view = (
        np.ascontiguousarray(flat_states[flat_filled])
        .view([("", flat_states.dtype)] * flat_states.shape[1])
        .ravel()
    )
    _, state_inv = np.unique(s_view, return_inverse=True)
    n_states = int(state_inv.max()) + 1
    # count states that carry more than one distinct recorded joint action
    order = np.lexsort((group_ids, state_inv))
    st_sorted = state_inv[order]
    g_sorted = group_ids[order]
    new_group = np.ones(st_sorted.size, dtype=bool)
    new_group[1:] = (st_sorted[1:] != st_sorted[:-1]) | (g_sorted[1:] != g_sorted[:-1])
    distinct_pairs_per_state = np.bincount(st_sorted[new_group], minlength=n_states)
    multi = int((distinct_pairs_per_state > 1).sum())
    _, pair_cnt = np.unique(group_ids, return_counts=True)
    return {
        "n_occurrences": int(pair_cnt.sum()),
        "n_distinct_state_action_pairs": int(pair_cnt.size),
        "occurrences_per_pair_median": float(np.median(pair_cnt)),
        "occurrences_per_pair_mean": float(pair_cnt.mean()),
        "occurrences_per_pair_max": int(pair_cnt.max()),
        "n_distinct_states": n_states,
        "n_states_with_multiple_recorded_actions": multi,
        "note": "same-state action pairs are the only population that can measure "
                "action ranking from the dataset alone; if the count above is ~0 "
                "the dataset cannot measure action ranking at all",
    }


@torch.inference_mode()
def collect_q_and_g(learner, dataset, gamma: float, episodes_per_batch: int):
    """Return (Q, G, episode_id) flattened over filled steps, plus the padded grids.

    Q is Q_tot(s_t, u_t) from the frozen critic; G is the realised discounted
    return-to-go of the recorded trajectory.
    """
    n = dataset.n_episodes
    t_len = dataset.filled.shape[1]
    q_grid = torch.zeros(n, t_len)
    g_grid = torch.zeros(n, t_len)
    f_grid = dataset.filled.squeeze(-1) > 0.5
    for start in range(0, n, episodes_per_batch):
        idx = torch.arange(start, min(start + episodes_per_batch, n), device=dataset.device)
        batch = {
            key: getattr(dataset, key).index_select(0, idx).to(learner.device)
            for key in ("obs", "actions", "state", "reward", "terminated", "filled")
        }
        mask = batch["filled"].squeeze(-1) > 0.5
        mac_out = learner._forward_qs(batch)
        chosen_q = torch.gather(mac_out, 3, batch["actions"].long())
        q_tot = learner.mixer(chosen_q, batch["state"]).view(chosen_q.shape[0], chosen_q.shape[1])
        g = return_to_go(batch["reward"].squeeze(-1), mask.float(), gamma)
        for row, ep in enumerate(range(start, start + len(idx))):
            t_idx = torch.nonzero(mask[row], as_tuple=False).squeeze(-1)
            q_grid[ep, t_idx] = q_tot[row, t_idx].float().cpu()
            g_grid[ep, t_idx] = g[row, t_idx].float().cpu()
        print(f"episodes {start + len(idx)}/{n}", flush=True)
    filled = f_grid.cpu()
    episode_id = torch.arange(n).unsqueeze(1).expand(n, t_len)[filled]
    return q_grid[filled], g_grid[filled], episode_id, q_grid.numpy(), g_grid.numpy(), filled.numpy()


def main() -> None:
    _self_check()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="default: checkpoint_path in config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-pairs", type=int, default=1_000_000)
    parser.add_argument("--kendall-subsample", type=int, default=20_000)
    parser.add_argument("--episodes-per-batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0, help="rng seed for pair sampling")
    args = parser.parse_args()

    ensure_project_root()
    t0 = time.time()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    ckpt = resolve_path(args.checkpoint or cfg["checkpoint_path"])
    gamma = float(cfg["system"]["discount"])

    dataset = load_vault_dataset(cfg)
    learner = build_learner(cfg, dataset)
    loaded = load_qmix_checkpoint(ckpt, learner, load_optimizer=False)
    learner.q_network.eval()
    learner.mixer.eval()

    q, g, episode_id, q_grid, g_grid, filled_grid = collect_q_and_g(
        learner, dataset, gamma, args.episodes_per_batch
    )
    n = int(q.numel())
    if n < 2:
        raise RuntimeError("not enough filled transitions to measure")
    residual = (q.double() - g.double()).cpu().numpy()
    q_np = q.double().cpu().numpy()
    g_np = g.double().cpu().numpy()
    ep_np = episode_id.numpy()
    over = (q_np > g_np).astype(np.float64)

    denom = np.maximum(np.abs(g_np), 1.0)
    qm, gm = float(q_np.mean()), float(g_np.mean())
    var_q = float(np.mean((q_np - qm) ** 2))
    cov = float(np.mean((q_np - qm) * (g_np - gm)))
    slope = cov / var_q if var_q > 1e-12 else 1.0
    shift = gm - slope * qm

    rng = np.random.default_rng(args.seed)
    pairs = torch.from_numpy(
        rng.integers(0, n, size=(args.n_pairs, 2))
    ).to(q.device)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    counts = pairwise_outcomes(q, g, pairs)
    decided = counts["agree"] + counts["invert"]
    concordance = counts["agree"] / decided if decided else None
    n_distinct_pairs = int(pairs.shape[0]) // 2

    m = int(min(args.kendall_subsample, n))
    sub = rng.choice(n, size=m, replace=False)
    tau = kendalltau(q_np[sub], g_np[sub]).statistic
    tau = float(tau) if np.isfinite(tau) else None

    # Within-episode exhaustive pairs: removes the between-episode component
    # that dominates random cross-transition pairs. Still cross-state.
    wq_a, wg_a, wq_b, wg_b = within_episode_pairs(q_grid, g_grid, filled_grid)
    w_agree, w_invert, w_tie = pair_concordance(wq_a, wg_a, wq_b, wg_b)
    w_decided = w_agree + w_invert
    w_conc = w_agree / w_decided if w_decided else None
    w_gap = np.abs(wg_a - wg_b)
    w_by_gap = {}
    for lo, hi in ((0.0, 0.5), (0.5, 2.0), (2.0, float("inf"))):
        sel = (w_gap >= lo) & (w_gap < hi)
        a, i, t = pair_concordance(wq_a[sel], wg_a[sel], wq_b[sel], wg_b[sel])
        d = a + i
        w_by_gap[f"{lo}_{hi}"] = {
            "n_pairs": int(sel.sum()),
            "concordance": (a / d) if d else None,
            "n_tie": t,
        }

    se_resid_ep, k_ep = clustered_se(residual, ep_np)
    se_over_ep, _ = clustered_se(over, ep_np)
    group_ids = decision_group_ids(dataset)
    support = dataset_support_stats(dataset, group_ids)
    decomp = residual_decomposition(q_np, g_np, group_ids)
    # How often does Q_tot return literally the same number for the same decision?
    same_value = 0
    n_multi = 0
    order = np.argsort(group_ids, kind="stable")
    gs_sorted = group_ids[order]
    qs_sorted = q_np[order]
    bounds = np.flatnonzero(np.r_[True, gs_sorted[1:] != gs_sorted[:-1]])
    starts, ends = bounds, np.r_[bounds[1:], gs_sorted.size]
    spreads = []
    for a, b in zip(starts, ends):
        if b - a < 2:
            continue
        n_multi += 1
        spread = float(qs_sorted[a:b].max() - qs_sorted[a:b].min())
        spreads.append(spread)
        if spread <= 1e-9:
            same_value += 1
    spreads = np.asarray(spreads)
    decomp["n_decisions_visited_more_than_once"] = int(n_multi)
    decomp["frac_decisions_with_identical_q_every_visit"] = (
        None if n_multi == 0 else same_value / n_multi
    )
    decomp["max_within_decision_q_spread"] = (
        None if spreads.size == 0 else float(spreads.max())
    )
    decomp["identical_q_note"] = (
        "Q_tot is a deterministic function of (obs history, joint action), so the same "
        "decision gets the same value up to float rounding. The few decisions that do "
        "not are cases where the quantised global state does not pin down the history, "
        "so the GRU hidden state - and therefore Q_tot - differs."
    )

    # Stratify the residual by how good the recorded episode turned out. A
    # one-sided overestimate cannot flip sign across strata; if it does, the
    # pooled mean is a mixture of shrinkage, not an error.
    ep_return = np.asarray(
        [float(g_grid[e, np.flatnonzero(filled_grid[e])[0]]) for e in range(filled_grid.shape[0])],
        dtype=np.float64,
    )
    q1, q2 = np.quantile(ep_return, [1 / 3, 2 / 3])
    strata = {}
    for name, sel in (
        ("worst_third_episodes", ep_return <= q1),
        ("middle_third_episodes", (ep_return > q1) & (ep_return <= q2)),
        ("best_third_episodes", ep_return > q2),
    ):
        sel_t = sel[ep_np]
        strata[name] = {
            "n_transitions": int(sel_t.sum()),
            "mean_recorded_return_to_go": float(g_np[sel_t].mean()),
            "mean_q": float(q_np[sel_t].mean()),
            "q_minus_recorded_return_mean": float(residual[sel_t].mean()),
            "q_above_recorded_return_rate": float(over[sel_t].mean()),
        }

    extra = loaded.get("extra", {}) if isinstance(loaded, dict) else {}
    payload = {
        "config": args.config,
        "checkpoint": str(ckpt),
        "ckpt_t_env": extra.get("t_env"),
        "ckpt_seed": extra.get("seed"),
        "ckpt_best_eval_return": extra.get("best_eval_return"),
        "scenario": cfg["task"]["scenario"],
        "dataset": cfg["task"].get("dataset"),
        "gamma": gamma,
        "n_episodes": dataset.n_episodes,
        "n_transitions": n,
        "dataset_support": support,
        # --- what the two sides are -------------------------------------------
        "q_mean": qm,
        "q_std": float(q_np.std()),
        "recorded_return_to_go_mean": gm,
        "recorded_return_to_go_std": float(g_np.std()),
        # --- Q_tot minus realised behaviour return (NOT an estimation error) ---
        "q_minus_recorded_return_mean": float(residual.mean()),
        "q_minus_recorded_return_std": float(residual.std()),
        "q_minus_recorded_return_mean_se_episode_clustered": se_resid_ep,
        "q_minus_recorded_return_mean_clusters": k_ep,
        "q_minus_recorded_return_normalized_mean": float(np.mean(residual / denom)),
        "q_above_recorded_return_rate": float(over.mean()),
        "q_above_recorded_return_rate_se_episode_clustered": se_over_ep,
        "q_minus_recorded_return_quantiles": {
            str(p): float(np.quantile(residual, p / 100)) for p in (5, 25, 50, 75, 95)
        },
        "q_minus_recorded_return_decomposition": decomp,
        "q_minus_recorded_return_by_episode_outcome": strata,
        # --- scale relation, not an error --------------------------------------
        "recorded_return_on_q_affine_fit": {"slope": float(slope), "shift": float(shift)},
        # --- ordering: population must be stated, it changes the reading -------
        "pairwise_cross_state_random": {
            "n_pairs_sampled": int(pairs.shape[0]),
            "n_distinct_unordered_pairs": n_distinct_pairs,
            "n_episodes_represented": int(np.unique(ep_np).size),
            **counts,
            "concordance": concordance,
            "q_tie": Q_TIE,
            "g_tie": G_TIE,
            "population": "random pairs of transitions anywhere in the dataset; "
                          "almost always different states",
        },
        "pairwise_within_episode": {
            "n_pairs": int(wq_a.size),
            "n_distinct_unordered_pairs": int(wq_a.size),
            "agree": w_agree,
            "invert": w_invert,
            "tie": w_tie,
            "concordance": w_conc,
            "by_recorded_return_gap": w_by_gap,
            "population": "all pairs inside one episode; still different timesteps "
                          "and therefore different states",
        },
        "kendall_tau_cross_transition": tau,
        "kendall_subsample": m,
        "elapsed_s": round(time.time() - t0, 1),
    }
    # Deprecated aliases: previous key names, kept so older result files and
    # downstream readers keep working. Their names imply an error reading that
    # this protocol does not support; do not use them in new write-ups.
    payload.update(
        {
            "bias_mean": payload["q_minus_recorded_return_mean"],
            "bias_std": payload["q_minus_recorded_return_std"],
            "bias_normalized_mean": payload["q_minus_recorded_return_normalized_mean"],
            "overestimate_rate": payload["q_above_recorded_return_rate"],
            "residual_quantiles": payload["q_minus_recorded_return_quantiles"],
            "affine_calibration": payload["recorded_return_on_q_affine_fit"],
            "g_mean": payload["recorded_return_to_go_mean"],
            "g_std": payload["recorded_return_to_go_std"],
            "pairwise": payload["pairwise_cross_state_random"],
            "kendall_tau": tau,
        }
    )

    out = Path(resolve_path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
