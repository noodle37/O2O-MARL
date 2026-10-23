#!/usr/bin/env python3
"""Shared measurement primitives for the retention-insight validation (H1/H2).

Everything here is branch-replay machinery: a frozen teacher is rolled out,
probe states are replayed bit-for-bit, and candidate joint actions are scored
against the discounted return of "force this joint, then continue greedily".
No parameters are updated.

The one-agent deviations and behaviour-only candidate sets are deliberately
built without the frozen Q (except `hard`, which is the explicit "strongest
Q competitor" test): proposals come from legality and the recorded support,
and Q is used only afterwards as a scorer.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import torch

from scripts.verify_rank_reliability import (
    RETURN_TIE,
    SupportIndex,
    _legal_support,
    _mix_joint_q,
    bucket,
    clustered_rate,
    episode_done,
    pair_outcome,
    random_legal,
    unit_counts,
)
from scripts.verify_local_rank_inversion import (
    _agent_names,
    _forward_qs,
    _legal_tensor,
    _new_hidden,
    discounted_return,
    joint_key,
    reset_env,
)


# --------------------------------------------------------------------------- #
# Teacher rollout with rewards and (optionally) observations.
# --------------------------------------------------------------------------- #
def teacher_rollout(
    env,
    learner,
    wave: str = "greedy",
    eps: float = 0.0,
    rng: np.random.Generator | None = None,
    store_obs: bool = False,
    q_evaluator=None,
) -> dict:
    """Roll out `learner` (eps-)greedily, recording per-step Q/legal/state.

    `q_evaluator` defaults to the actor. Rewards are stored after each step so
    the episode return is available; `obs` is only kept when requested.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    q_net = learner if q_evaluator is None else q_evaluator
    names = _agent_names(learner.n_agents)
    n_actions = learner.n_actions
    rnn = _new_hidden(learner)
    rnn_eval = rnn if q_evaluator is None else _new_hidden(q_net)
    last: dict[str, int] = {}
    obs, info = reset_env(env)
    steps: list[dict] = []
    won = None
    ep_return = 0.0
    t = 0
    while True:
        state = np.asarray(info["state"], dtype=np.float32).copy()
        legal = _legal_tensor(info["legals"], names, n_actions, learner.device)
        with torch.inference_mode():
            q = _forward_qs(q_net, obs, rnn_eval, last)
            if q_evaluator is None:
                greedy = q.masked_fill(~legal, -1e9).argmax(dim=-1)
            else:
                q_act = _forward_qs(learner, obs, rnn, last)
                greedy = q_act.masked_fill(~legal, -1e9).argmax(dim=-1)
        if wave == "epsgreedy" and eps > 0 and rng.random() < eps:
            chosen = random_legal(legal, rng)
        else:
            chosen = greedy
        alive_a, alive_e = unit_counts(env)
        step = {
            "t": t,
            "state": state,
            "q": q.detach().cpu(),
            "legal": legal.detach().cpu(),
            "action": tuple(int(chosen[i]) for i in range(learner.n_agents)),
            "alive_allies": alive_a,
            "alive_enemies": alive_e,
        }
        if store_obs:
            step["obs"] = {a: np.asarray(obs[a]).copy() for a in obs}
        steps.append(step)
        act = {names[i]: int(chosen[i].item()) for i in range(learner.n_agents)}
        for agent in names:
            last[agent] = act[agent]
        obs, rew, terminal, trunc, info = env.step(act)
        reward = float(np.mean([float(np.asarray(r)) for r in rew.values()]))
        steps[-1]["reward"] = reward
        ep_return += reward
        if "battle_won" in info:
            won = bool(info["battle_won"])
        t += 1
        if episode_done(terminal, trunc) or t > 400:
            break
    return {"steps": steps, "won": won, "length": t, "return": ep_return}


# --------------------------------------------------------------------------- #
# Candidate construction (behaviour-only, no Q except the explicit `hard` test).
# --------------------------------------------------------------------------- #
def _legal_alt_actions(legal: torch.Tensor, agent: int, base_action: int) -> list[int]:
    row = legal[agent].detach().cpu().tolist()
    return [a for a in range(len(row)) if bool(row[a]) and a != base_action]


def _near_marginal_counts(near: np.ndarray, n_agents: int, n_actions: int) -> np.ndarray:
    counts = np.zeros((n_agents, n_actions), dtype=np.int64)
    if near.size == 0:
        return counts
    for i in range(n_agents):
        counts[i] = np.bincount(near[:, i].astype(np.int64), minlength=n_actions)
    return counts


def _joint_legal(joint, legal: torch.Tensor) -> bool:
    for i, a in enumerate(joint):
        if not bool(legal[i, int(a)]):
            return False
    return True


def _hamming(a, b) -> int:
    return sum(int(x) != int(y) for x, y in zip(a, b))


def build_deviations(
    base_key: tuple[int, ...],
    legal: torch.Tensor,
    near: np.ndarray,
    q: torch.Tensor,
    sidx: SupportIndex,
    kinds: tuple[str, ...],
    rng: np.random.Generator,
) -> dict[tuple[str, int], tuple[torch.Tensor, int]]:
    """One alternative action per agent per requested kind.

    primary : most frequent legal alternative in the kNN offline neighbourhood
    hard    : legal alternative with the highest frozen Q for that agent
    random  : uniform legal alternative
    """
    n_agents, n_actions = legal.shape
    near_counts = _near_marginal_counts(near, n_agents, n_actions)
    out: dict[tuple[str, int], tuple[torch.Tensor, int]] = {}
    for i in range(n_agents):
        alts = _legal_alt_actions(legal, i, base_key[i])
        if not alts:
            continue
        if "primary" in kinds:
            alt = max(alts, key=lambda a: (int(near_counts[i, a]), int(sidx.marginal[i, a]), -a))
            out[("primary", i)] = (_replace(base_key, i, alt, legal), alt)
        if "hard" in kinds:
            alt = max(
                alts,
                key=lambda a: (float(q[i, a]), int(near_counts[i, a]), -a),
            )
            out[("hard", i)] = (_replace(base_key, i, alt, legal), alt)
        if "random" in kinds:
            alt = int(rng.choice(alts))
            out[("random", i)] = (_replace(base_key, i, alt, legal), alt)
    return out


def _replace(base_key, agent: int, alt: int, legal: torch.Tensor) -> torch.Tensor:
    joint = torch.tensor(base_key, dtype=torch.long, device=legal.device)
    joint[agent] = int(alt)
    return joint


def behaviour_candidates(
    base_key: tuple[int, ...],
    legal: torch.Tensor,
    near: np.ndarray,
    sidx: SupportIndex,
    k: int,
    per_agent: int = 2,
    max_hamming: int = 2,
    n_recorded: int = 32,
) -> list[torch.Tensor]:
    """K legal joints proposed from recorded support, never from frozen Q."""
    n_agents, n_actions = legal.shape
    base = tuple(int(x) for x in base_key)
    if near.size:
        near_counts = _near_marginal_counts(near, n_agents, n_actions)
        near_rows = [tuple(int(x) for x in row) for row in near]
    else:
        near_counts = np.zeros((n_agents, n_actions), dtype=np.int64)
        near_rows = []
    legal_rows = [row for row in near_rows if _joint_legal(row, legal)]
    row_counts = Counter(legal_rows)
    cand: set[tuple[int, ...]] = {base}

    for i in range(n_agents):
        alts = sorted(
            _legal_alt_actions(legal, i, base[i]),
            key=lambda a: (-int(near_counts[i, a]), -int(sidx.marginal[i, a]), a),
        )[:per_agent]
        for alt in alts:
            row = list(base)
            row[i] = int(alt)
            row_t = tuple(row)
            if _joint_legal(row_t, legal):
                cand.add(row_t)

    ranked_rows = sorted(legal_rows, key=lambda r: (-row_counts.get(r, 0), r))
    for row in ranked_rows[:n_recorded]:
        if _hamming(row, base) <= max_hamming:
            cand.add(row)

    def score(row: tuple[int, ...]) -> float:
        if row == base:
            return float("inf")
        value = math.log1p(float(sidx.counts.get(row, 0)))
        if near_rows:
            value += 2.0 * float(row_counts.get(row, 0)) / max(len(near_rows), 1)
            value += sum(
                float(near_counts[i, row[i]]) for i in range(n_agents)
            ) / max(len(near_rows), 1)
        return value

    ordered = [base] + sorted(
        (row for row in cand if row != base), key=lambda r: (-score(r), r)
    )
    return [
        torch.tensor(row, dtype=torch.long, device=legal.device)
        for row in ordered[: max(1, int(k))]
    ]


def random_candidates(
    base_key: tuple[int, ...],
    legal: torch.Tensor,
    support_dev: torch.Tensor,
    k: int,
    rng: np.random.Generator,
) -> list[torch.Tensor]:
    """Base plus uniform draws from the recorded support (control proposal)."""
    feasible = _legal_support(support_dev, legal)
    base = tuple(int(x) for x in base_key)
    picks: list[tuple[int, ...]] = [base]
    if feasible.shape[0] > 1:
        n_pick = min(int(k) - 1, int(feasible.shape[0]))
        idx = rng.choice(int(feasible.shape[0]), size=n_pick, replace=False)
        for i in np.atleast_1d(idx):
            row = tuple(int(x) for x in feasible[int(i)].detach().cpu().tolist())
            if row not in picks:
                picks.append(row)
    n_space = int(legal.shape[0]) ** int(legal.shape[1]) if legal.numel() < 12 else 10**9
    attempts = 0
    max_attempts = max(100, 50 * int(k))
    while (
        len(picks) < int(k)
        and len(picks) < n_space
        and attempts < max_attempts
    ):
        attempts += 1
        row = tuple(int(x) for x in random_legal(legal, rng).detach().cpu().tolist())
        if _joint_legal(row, legal) and row not in picks:
            picks.append(row)
    return [
        torch.tensor(row, dtype=torch.long, device=legal.device)
        for row in picks[: max(1, int(k))]
    ]


def sample_random_pairs(
    feasible: torch.Tensor, n_pairs: int, rng: np.random.Generator
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    m = int(feasible.shape[0])
    if m < 2 or n_pairs <= 0:
        return []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    attempts = 0
    while len(pairs) < int(n_pairs) and attempts < 50 * int(n_pairs):
        attempts += 1
        i, j = rng.choice(m, size=2, replace=False)
        a = tuple(int(x) for x in feasible[int(i)].detach().cpu().tolist())
        b = tuple(int(x) for x in feasible[int(j)].detach().cpu().tolist())
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(
            (
                torch.tensor(a, dtype=torch.long, device=feasible.device),
                torch.tensor(b, dtype=torch.long, device=feasible.device),
            )
        )
    return pairs


# --------------------------------------------------------------------------- #
# Candidate-set metrics.
# --------------------------------------------------------------------------- #
def candidate_metrics(
    keys: list[tuple[int, ...]],
    q_values: list[float],
    g_values: list[float],
    tie_tol: float = RETURN_TIE,
) -> dict:
    """Top-1 / top-2 / regret / pairwise concordance for one candidate set."""
    k = len(keys)
    if k < 2:
        return {"k": k, "n_pairs": 0, "n_decided": 0, "n_tie": 0}
    q = np.asarray(q_values, dtype=np.float64)
    g = np.asarray(g_values, dtype=np.float64)
    q_top = int(np.argmax(q))
    g_top = int(np.argmax(g))
    order_g = np.sort(g)[::-1]
    top1_correct = bool(abs(g[q_top] - g[g_top]) <= tie_tol)
    top2_correct = bool(g[q_top] >= order_g[1] - tie_tol)
    n_agree = n_invert = n_tie = 0
    for i in range(k):
        for j in range(i + 1, k):
            dq = float(q[i] - q[j])
            dg = float(g[i] - g[j])
            if abs(dq) <= 1e-8 or abs(dg) <= tie_tol:
                n_tie += 1
            elif (dq > 0) == (dg > 0):
                n_agree += 1
            else:
                n_invert += 1
    n_decided = n_agree + n_invert
    return {
        "k": k,
        "n_pairs": k * (k - 1) // 2,
        "n_decided": n_decided,
        "n_tie": n_tie,
        "q_top1": keys[q_top],
        "true_top1": keys[g_top],
        "top1_correct": top1_correct,
        "top2_correct": top2_correct,
        "regret": float(g[g_top] - g[q_top]),
        "pair_concordance": (n_agree / n_decided) if n_decided else None,
        "kendall_tau": ((n_agree - n_invert) / n_decided) if n_decided else None,
        "true_best_qrank": int(np.sum(q > q[g_top] + 1e-8)),
        "q_top1_truerank": int(np.sum(g > g[q_top] + tie_tol)),
    }


# --------------------------------------------------------------------------- #
# Clustered statistics.
# --------------------------------------------------------------------------- #
def clustered_mean(values, clusters, label: str = "mean") -> dict:
    groups: dict[str, list[float]] = {}
    for value, cluster in zip(values, clusters):
        if value is None:
            continue
        v = float(value)
        if not math.isfinite(v):
            continue
        groups.setdefault(str(cluster), []).append(v)
    per = np.asarray([np.mean(v) for v in groups.values() if v], dtype=np.float64)
    out = {
        f"n_{label}_values": int(sum(len(v) for v in groups.values())),
        "n_clusters": int(per.size),
        "mean": None,
        "se": None,
        "ci95": None,
    }
    if per.size:
        out["mean"] = float(per.mean())
    if per.size >= 2:
        se = float(per.std(ddof=1) / np.sqrt(per.size))
        out["se"] = se
        out["ci95"] = [float(per.mean() - 1.96 * se), float(per.mean() + 1.96 * se)]
    return out


def rates_by(records: list[dict], key_fn) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(key_fn(record)), []).append(record)
    return {key: clustered_rate(value) for key, value in groups.items()}


def means_by(records: list[dict], value_fn, key_fn, label: str = "mean") -> dict:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(key_fn(record)), []).append(record)
    return {
        key: clustered_mean(
            [value_fn(r) for r in value],
            [r["cluster"] for r in value],
            label=label,
        )
        for key, value in groups.items()
    }


def cluster_bootstrap_diff(
    records_a: list[dict],
    records_b: list[dict],
    value_fn,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Cluster bootstrap CI for mean(a) - mean(b) over clusters present in both."""
    def per_cluster(records):
        groups: dict[str, list[float]] = {}
        for record in records:
            value = value_fn(record)
            if value is None:
                continue
            groups.setdefault(str(record["cluster"]), []).append(float(value))
        return {key: float(np.mean(vals)) for key, vals in groups.items() if vals}

    a = per_cluster(records_a)
    b = per_cluster(records_b)
    common = sorted(set(a) & set(b))
    if not common:
        return {"n_clusters": 0, "mean_diff": None, "ci95": None}
    diff = float(np.mean([a[c] for c in common]) - np.mean([b[c] for c in common]))
    rng = np.random.default_rng(seed)
    arr_a = np.asarray([a[c] for c in common], dtype=np.float64)
    arr_b = np.asarray([b[c] for c in common], dtype=np.float64)
    boots = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(common), size=len(common))
        boots.append(float(arr_a[idx].mean() - arr_b[idx].mean()))
    boots_arr = np.asarray(boots, dtype=np.float64)
    return {
        "n_clusters": len(common),
        "mean_diff": diff,
        "ci95": [
            float(np.percentile(boots_arr, 2.5)),
            float(np.percentile(boots_arr, 97.5)),
        ],
    }


def effect_bin(value: float) -> str:
    return bucket(float(value), [0.0, 0.5, 2.0, 5.0, 1e9])


def support_fields(key: tuple[int, ...], near: np.ndarray, sidx: SupportIndex) -> dict:
    sup = sidx.support(near, key) if near.size else {
        "global_count": int(sidx.counts.get(key, 0)),
        "marginal_global": [int(sidx.marginal[i, key[i]]) for i in range(len(key))],
        "state_cond_joint_count": 0,
        "cond_freq": 0.0,
        "marginal_near": [0] * len(key),
        "k_near": 0,
    }
    return {
        "global_count": int(sup["global_count"]),
        "marginal_global": list(sup["marginal_global"]),
        "min_marginal_global": int(min(sup["marginal_global"])) if sup["marginal_global"] else 0,
        "state_cond_joint_count": int(sup["state_cond_joint_count"]),
        "cond_freq": float(sup["cond_freq"]),
        "marginal_near": list(sup["marginal_near"]),
        "min_marginal_near": int(min(sup["marginal_near"])) if sup["marginal_near"] else 0,
        "k_near": int(sup["k_near"]),
    }


def discounted_from(base: dict, t: int, gamma: float) -> float:
    rewards = [float(step.get("reward", 0.0)) for step in base["steps"][t:]]
    return discounted_return(rewards, 0, gamma)
