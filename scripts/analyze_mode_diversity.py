#!/usr/bin/env python3
"""H3: coordination-mode diversity and collapse across 13 checkpoints.

Two behaviour signatures per policy:

  A. own rollout (primary) -- greedy episodes on env seeds 1..N; joint-action
     distribution, per-agent distribution, return/win/length, alive counts at
     the probe fractions.
  B. fixed-state counterfactual (secondary) -- teacher-prefix observations,
     where each policy updates its own RNN and feeds its own previous action.
     At the probe steps we record the joint action each policy would take on
     the same state, so the comparison no longer depends on where its own
     rollout happened to go.

Metrics: JS divergences between policy signatures and between episodes,
agglomerative episode clustering with silhouette-selected K, occupancy
entropy/effective modes/dominant share per policy, fixed-state distinct-mode
counts, and early/late windows over the UOT temporal checkpoints. Everything is
descriptive; no significance claims (checkpoints are highly correlated).

Nothing is trained.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("SC2PATH", "/home/jxy/StarCraftII")

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retention_insight_core import teacher_rollout  # noqa: E402
from scripts.verify_local_rank_inversion import (  # noqa: E402
    _agent_names,
    _forward_qs,
    _new_hidden,
    close_env,
)
from scripts.verify_retention_insight import select_probe_steps  # noqa: E402
from training.factory import build_learner, load_vault_dataset, make_env  # noqa: E402
from utils.checkpoint import load_qmix_checkpoint  # noqa: E402
from utils.paths import ensure_project_root, resolve_path  # noqa: E402


@dataclass
class PolicySpec:
    tag: str
    family: str
    t_env: int
    path: str


def parse_manifest(path: str) -> list[PolicySpec]:
    specs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"bad manifest line: {line!r}")
        tag, family, t_env, ckpt = parts
        specs.append(PolicySpec(tag, family, int(t_env), resolve_path(ckpt)))
    if not specs:
        raise ValueError(f"empty policy manifest: {path}")
    return specs


def load_policies(cfg, dataset, specs, device):
    policies = {}
    metas = {}
    for spec in specs:
        if not os.path.isfile(spec.path):
            raise FileNotFoundError(f"policy checkpoint missing: {spec.path}")
        learner = build_learner(cfg, dataset)
        ckpt = load_qmix_checkpoint(spec.path, learner, map_location=device)
        learner.q_network.eval()
        learner.mixer.eval()
        extra = dict(ckpt.get("extra", {}))
        t_env = int(ckpt.get("t_env", extra.get("t_env", spec.t_env)))
        policies[spec.tag] = learner
        metas[spec.tag] = {
            "tag": spec.tag,
            "family": spec.family,
            "path": spec.path,
            "manifest_t_env": spec.t_env,
            "t_env": t_env,
            "extra": extra,
        }
        print(f"[policy] {spec.tag:14s} family={spec.family:8s} t_env={t_env}", flush=True)
    return policies, metas


# --------------------------------------------------------------------------- #
# Signatures.
# --------------------------------------------------------------------------- #
def normalize(counter: Counter) -> dict:
    total = float(sum(counter.values()))
    return {key: float(value) / total for key, value in counter.items()} if total else {}


def mean_distributions(dists: list[dict]) -> dict:
    if not dists:
        return {}
    acc: Counter = Counter()
    for dist in dists:
        for key, value in dist.items():
            acc[key] += float(value)
    return {key: value / len(dists) for key, value in acc.items()}


def entropy(dist: dict) -> float:
    return -sum(p * math.log2(p) for p in dist.values() if p > 0.0)


def normalized_entropy(dist: dict) -> float:
    if len(dist) <= 1:
        return 0.0
    return entropy(dist) / math.log2(len(dist))


def js_divergence(a: dict, b: dict) -> float:
    total = 0.0
    for key in set(a) | set(b):
        p = float(a.get(key, 0.0))
        q = float(b.get(key, 0.0))
        m = 0.5 * (p + q)
        if p > 0.0:
            total += 0.5 * p * math.log2(p / m)
        if q > 0.0:
            total += 0.5 * q * math.log2(q / m)
    return max(0.0, total)


def js_matrix(dists: list[dict]) -> np.ndarray:
    n = len(dists)
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            value = js_divergence(dists[i], dists[j])
            out[i, j] = out[j, i] = value
    return out


def mean_offdiag(matrix: np.ndarray) -> float | None:
    n = matrix.shape[0]
    if n < 2:
        return None
    total = float(matrix.sum() - np.trace(matrix))
    return total / (n * (n - 1))


def bootstrap_offdiag(matrix: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    n = matrix.shape[0]
    if n < 2:
        return {"mean": None, "ci95": None}
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub = matrix[np.ix_(idx, idx)]
        values.append(float(sub.sum() / (n * (n - 1))))
    arr = np.asarray(values)
    return {
        "mean": float(mean_offdiag(matrix)),
        "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
    }


def bootstrap_mean(values, n_boot: int, rng: np.random.Generator) -> dict:
    vals = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if vals.size == 0:
        return {"mean": None, "ci95": None}
    boots = np.asarray(
        [rng.choice(vals, size=vals.size, replace=True).mean() for _ in range(n_boot)]
    )
    return {
        "mean": float(vals.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
    }


# --------------------------------------------------------------------------- #
# Own rollouts.
# --------------------------------------------------------------------------- #
def rollout_signature(env, learner, seed: int, fracs: list[float]) -> dict:
    base = teacher_rollout(
        env,
        learner,
        wave="greedy",
        rng=np.random.default_rng(3000 + seed),
        store_obs=False,
    )
    joint_counts: Counter = Counter()
    agent_counts = np.zeros((learner.n_agents, learner.n_actions), dtype=np.int64)
    for step in base["steps"]:
        key = tuple(int(x) for x in step["action"])
        joint_counts[key] += 1
        for i, action in enumerate(key):
            agent_counts[i, int(action)] += 1
    probes = {}
    for frac in fracs:
        t = int(round(float(frac) * max(base["length"] - 1, 0)))
        rec = base["steps"][t]
        probes[str(frac)] = {
            "t": t,
            "alive_allies": rec["alive_allies"],
            "alive_enemies": rec["alive_enemies"],
        }
    return {
        "joint_counts": {key: int(count) for key, count in joint_counts.items()},
        "agent_counts": agent_counts.tolist(),
        "return": float(base["return"]),
        "won": bool(base["won"]),
        "length": int(base["length"]),
        "probes": probes,
    }


# --------------------------------------------------------------------------- #
# Fixed-state counterfactual actions.
# --------------------------------------------------------------------------- #
def policy_actions_on_teacher_obs(policy, obs_seq, legal_seq, probe_steps) -> dict:
    names = _agent_names(policy.n_agents)
    rnn = _new_hidden(policy)
    last: dict[str, int] = {}
    out: dict[int, tuple[int, ...]] = {}
    for t, obs in enumerate(obs_seq):
        legal = legal_seq[t].to(policy.device)
        with torch.inference_mode():
            q = _forward_qs(policy, obs, rnn, last)
        chosen = q.masked_fill(~legal, -1e9).argmax(dim=-1)
        actions = tuple(int(chosen[i]) for i in range(policy.n_agents))
        if t in probe_steps:
            out[t] = actions
        for agent, action in zip(names, actions):
            last[agent] = action
    return out


def teacher_sequence(env, teacher, seed: int, fracs: list[float], max_probes: int) -> dict:
    base = teacher_rollout(
        env,
        teacher,
        wave="greedy",
        rng=np.random.default_rng(4000 + seed),
        store_obs=True,
    )
    steps = base["steps"]
    probe_steps = (
        select_probe_steps(base["length"], fracs, max_probes) if base["length"] >= 2 else []
    )
    return {
        "seed": int(seed),
        "obs": [step["obs"] for step in steps],
        "legal": [step["legal"] for step in steps],
        "actions": [step["action"] for step in steps],
        "probe_steps": probe_steps,
        "return": float(base["return"]),
        "won": bool(base["won"]),
        "length": int(base["length"]),
    }


# --------------------------------------------------------------------------- #
# Clustering.
# --------------------------------------------------------------------------- #
def cluster_episodes(distance: np.ndarray, k_min: int, k_max: int) -> dict:
    n = distance.shape[0]
    if n < 3:
        return {
            "linkage": [],
            "k_scores": [],
            "best_k": 1,
            "silhouette": None,
            "labels": [0] * n,
        }
    condensed = squareform(distance, checks=False)
    z = linkage(condensed, method="average")
    scores = []
    best = None
    for k in range(int(k_min), min(int(k_max), n - 1) + 1):
        labels = fcluster(z, t=k, criterion="maxclust")
        n_labels = int(len(set(int(x) for x in labels)))
        if n_labels < 2:
            continue
        sil = float(silhouette_score(distance, labels, metric="precomputed"))
        scores.append({"k": k, "n_labels": n_labels, "silhouette": sil})
        if best is None or sil > best["silhouette"]:
            best = {"k": k, "labels": labels, "silhouette": sil, "n_labels": n_labels}
    if best is None:
        labels = np.zeros(n, dtype=int)
        best = {"k": 1, "labels": labels, "silhouette": None, "n_labels": 1}
    raw = np.asarray(best["labels"], dtype=int)
    uniq = sorted(set(int(x) for x in raw))
    remap = {old: new for new, old in enumerate(uniq)}
    labels0 = [remap[int(x)] for x in raw]
    return {
        "linkage": z.tolist(),
        "k_scores": scores,
        "best_k": int(best["k"]),
        "silhouette": best["silhouette"],
        "labels": labels0,
    }


def occupancy(labels: list[int], n_clusters: int) -> dict:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_clusters)
    p = counts / max(int(counts.sum()), 1)
    return {
        "counts": counts.tolist(),
        "probs": p.tolist(),
        "entropy": entropy({i: float(v) for i, v in enumerate(p) if v > 0}),
        "effective_modes": float(2.0 ** entropy({i: float(v) for i, v in enumerate(p) if v > 0})),
        "dominant_share": float(p.max()) if p.size else None,
    }


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/5m_vs_6m.yaml")
    ap.add_argument("--policy-manifest", default="scripts/mode_policies_5m_vs_6m.txt")
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--probe-fracs", default="0.3,0.7")
    ap.add_argument("--max-probes", type=int, default=2)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--policy-tags", default=None, help="comma-separated subset (smoke)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="logs/mode_diversity.json")
    ap.add_argument("--txt", default="logs/mode_diversity.txt")
    args = ap.parse_args()

    ensure_project_root()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    cfg["system"]["device"] = args.device
    cfg["system"]["data_device"] = "cpu"
    cfg["system"]["moderate_lambda"] = False
    cfg["system"]["use_ovm"] = False
    fracs = [float(x) for x in args.probe_fracs.split(",") if x.strip()]
    seeds = list(range(args.seed0, args.seed0 + args.n_seeds))
    specs = parse_manifest(args.policy_manifest)
    if args.policy_tags:
        wanted = {x.strip() for x in args.policy_tags.split(",") if x.strip()}
        specs = [spec for spec in specs if spec.tag in wanted]
        missing = wanted - {spec.tag for spec in specs}
        if missing:
            raise ValueError(f"unknown policy tags: {sorted(missing)}")
    tags = [spec.tag for spec in specs]
    rng = np.random.default_rng(0)
    print(
        f"policies={len(specs)} seeds={seeds[0]}..{seeds[-1]} fracs={fracs} "
        f"device={args.device} updates=0",
        flush=True,
    )

    dataset = load_vault_dataset(cfg)
    print(
        f"dataset n_agents={dataset.stats.n_agents} n_actions={dataset.stats.n_actions}",
        flush=True,
    )
    policies, meta = load_policies(cfg, dataset, specs, args.device)
    teacher_tag = "teacher_best"
    if teacher_tag not in policies:
        raise ValueError("H3 requires the teacher_best policy in the manifest")
    teacher = policies[teacher_tag]
    del dataset

    episodes: list[dict] = []
    teacher_seqs: dict[int, dict] = {}
    for seed in seeds:
        env = make_env(cfg, seed=seed)
        try:
            seq = teacher_sequence(env, teacher, seed, fracs, args.max_probes)
            teacher_seqs[seed] = seq
            print(
                f"[teacher] seed={seed} win={seq['won']} L={seq['length']} "
                f"probes={seq['probe_steps']}",
                flush=True,
            )
            for spec in specs:
                sig = rollout_signature(env, policies[spec.tag], seed, fracs)
                episodes.append({"tag": spec.tag, "seed": seed, **sig})
                print(
                    f"[rollout] {spec.tag:14s} seed={seed:3d} won={sig['won']} "
                    f"L={sig['length']:3d} return={sig['return']:6.2f}",
                    flush=True,
                )
        finally:
            close_env(env)

    # Fixed-state counterfactual actions on the stored teacher sequences.
    fixed_actions: dict[str, dict[int, dict[int, tuple[int, ...]]]] = {tag: {} for tag in tags}
    for seed in seeds:
        seq = teacher_seqs[seed]
        obs_seq = seq["obs"]
        legal_seq = seq["legal"]
        for spec in specs:
            fixed_actions[spec.tag][seed] = policy_actions_on_teacher_obs(
                policies[spec.tag], obs_seq, legal_seq, seq["probe_steps"]
            )
        print(f"[fixed] seed={seed} probes={seq['probe_steps']}", flush=True)

    # Episode distributions and distances.
    episode_dists: list[dict] = []
    for ep in episodes:
        counter = Counter({tuple(key): int(count) for key, count in ep["joint_counts"].items()})
        episode_dists.append(normalize(counter))
    dist_eps = js_matrix(episode_dists)

    policy_dists = []
    for tag in tags:
        policy_dists.append(mean_distributions([d for d, ep in zip(episode_dists, episodes) if ep["tag"] == tag]))
    dist_policy = js_matrix(policy_dists)
    policy_linkage = linkage(squareform(dist_policy, checks=False), method="average")

    clustering = cluster_episodes(dist_eps, args.k_min, args.k_max)
    labels = clustering["labels"]
    n_clusters = max(max(labels) + 1, 1)

    within = {}
    for i, tag in enumerate(tags):
        idx = [j for j, ep in enumerate(episodes) if ep["tag"] == tag]
        sub = dist_eps[np.ix_(idx, idx)]
        boot = bootstrap_offdiag(sub, args.bootstrap, rng)
        joint_counts: Counter = Counter()
        for j in idx:
            joint_counts.update({tuple(k): v for k, v in episodes[j]["joint_counts"].items()})
        joint_dist = normalize(joint_counts)
        agent_counts = np.zeros_like(np.asarray(episodes[idx[0]]["agent_counts"]))
        for j in idx:
            agent_counts += np.asarray(episodes[j]["agent_counts"])
        agent_dists = []
        for agent in range(agent_counts.shape[0]):
            row = agent_counts[agent].astype(np.float64)
            agent_dists.append(normalize(Counter({a: float(v) for a, v in enumerate(row)})))
        returns = [episodes[j]["return"] for j in idx]
        wins = [1.0 if episodes[j]["won"] else 0.0 for j in idx]
        lengths = [episodes[j]["length"] for j in idx]
        occ = occupancy([labels[j] for j in idx], n_clusters)
        within[tag] = {
            "n_episodes": len(idx),
            "cross_seed_js": boot,
            "joint_entropy": entropy(joint_dist),
            "joint_entropy_norm": normalized_entropy(joint_dist),
            "dominant_joint_share": max(joint_dist.values()) if joint_dist else None,
            "agent_entropy_mean": float(np.mean([entropy(d) for d in agent_dists])),
            "return": bootstrap_mean(returns, args.bootstrap, rng),
            "win_rate": bootstrap_mean(wins, args.bootstrap, rng),
            "length_mean": float(np.mean(lengths)),
            "return_std": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
            "occupancy": occ,
            "top_joint_actions": [
                {"joint": list(key), "p": float(p)}
                for key, p in sorted(joint_dist.items(), key=lambda kv: -kv[1])[:5]
            ],
        }

    # Fixed-state signatures and probes.
    probe_records = []
    fixed_signatures: dict[str, dict] = {}
    for tag in tags:
        counts: Counter = Counter()
        for seed in seeds:
            for t in teacher_seqs[seed]["probe_steps"]:
                counts[fixed_actions[tag][seed][t]] += 1
        fixed_signatures[tag] = normalize(counts)
    for seed in seeds:
        seq = teacher_seqs[seed]
        for t in seq["probe_steps"]:
            actions = {tag: fixed_actions[tag][seed][t] for tag in tags}
            teacher_action = tuple(int(x) for x in seq["actions"][t])
            counts = Counter(actions.values())
            probs = np.asarray(list(counts.values()), dtype=np.float64) / max(len(tags), 1)
            rec = {
                "seed": int(seed),
                "t": int(t),
                "frac": t / max(seq["length"] - 1, 1),
                "teacher_action": list(teacher_action),
                "n_policies": len(tags),
                "distinct": int(len(counts)),
                "mode_share": float(probs.max()) if probs.size else None,
                "entropy": entropy({i: float(v) for i, v in enumerate(probs) if v > 0}),
                "entropy_norm": normalized_entropy({i: float(v) for i, v in enumerate(probs) if v > 0}),
                "actions": {tag: list(actions[tag]) for tag in tags},
                "hamming_to_teacher": {
                    tag: int(sum(int(a) != int(b) for a, b in zip(actions[tag], teacher_action)))
                    for tag in tags
                },
                "equal_teacher": {
                    tag: 1.0 if actions[tag] == teacher_action else 0.0 for tag in tags
                },
            }
            probe_records.append(rec)

    fixed_metrics = {}
    for tag in tags:
        sig = fixed_signatures[tag]
        ham = []
        eq = []
        for rec in probe_records:
            if "hamming_to_teacher" in rec:
                ham.append(rec["hamming_to_teacher"][tag])
                eq.append(rec["equal_teacher"][tag])
        fixed_metrics[tag] = {
            "entropy": entropy(sig),
            "entropy_norm": normalized_entropy(sig),
            "dominant_share": max(sig.values()) if sig else None,
            "hamming_to_teacher": float(np.mean(ham)) if ham else None,
            "equal_teacher": float(np.mean(eq)) if eq else None,
            "top_actions": [
                {"joint": list(key), "p": float(p)}
                for key, p in sorted(sig.items(), key=lambda kv: -kv[1])[:5]
            ],
        }
    fixed_sig_list = [fixed_signatures[tag] for tag in tags]
    dist_fixed = js_matrix(fixed_sig_list)
    fixed_linkage = linkage(squareform(dist_fixed, checks=False), method="average")

    # Early/late windows over the UOT temporal checkpoints.
    uot_specs = sorted([s for s in specs if s.family == "uot"], key=lambda s: s.t_env)
    window = {}
    if len(uot_specs) >= 4:
        early = [s.tag for s in uot_specs[: max(1, len(uot_specs) // 2)]]
        late = [s.tag for s in uot_specs[-max(1, len(uot_specs) // 2):]]
        idx_early = [tags.index(tag) for tag in early]
        idx_late = [tags.index(tag) for tag in late]

        def window_stats(idx):
            def per_probe(subset):
                vals_d, vals_h, vals_m = [], [], []
                for rec in probe_records:
                    counts = Counter(tuple(rec["actions"][tag]) for tag in subset)
                    p = np.asarray(list(counts.values()), dtype=np.float64) / len(subset)
                    vals_d.append(len(counts))
                    vals_h.append(normalized_entropy({i: float(v) for i, v in enumerate(p) if v > 0}))
                    vals_m.append(float(p.max()))
                return {
                    "distinct": float(np.mean(vals_d)),
                    "entropy_norm": float(np.mean(vals_h)),
                    "mode_share": float(np.mean(vals_m)),
                }

            return {
                "tags": [tags[i] for i in idx],
                "own_js": float(mean_offdiag(dist_policy[np.ix_(idx, idx)]) or 0.0),
                "fixed_js": float(mean_offdiag(dist_fixed[np.ix_(idx, idx)]) or 0.0),
                "own_entropy": float(np.mean([within[tags[i]]["joint_entropy"] for i in idx])),
                "own_dominant": float(np.mean([within[tags[i]]["dominant_joint_share"] for i in idx])),
                "fixed_entropy": float(np.mean([fixed_metrics[tags[i]]["entropy_norm"] for i in idx])),
                "fixed_probe": per_probe([tags[i] for i in idx]),
            }

        early_stats = window_stats(idx_early)
        late_stats = window_stats(idx_late)
        window = {
            "early": early_stats,
            "late": late_stats,
            "late_over_early": {
                "own_js": late_stats["own_js"] / early_stats["own_js"] if early_stats["own_js"] else None,
                "fixed_js": late_stats["fixed_js"] / early_stats["fixed_js"] if early_stats["fixed_js"] else None,
                "own_entropy": late_stats["own_entropy"] / early_stats["own_entropy"] if early_stats["own_entropy"] else None,
                "fixed_entropy": late_stats["fixed_entropy"] / early_stats["fixed_entropy"] if early_stats["fixed_entropy"] else None,
                "own_dominant_diff": late_stats["own_dominant"] - early_stats["own_dominant"],
                "distinct_diff": late_stats["fixed_probe"]["distinct"] - early_stats["fixed_probe"]["distinct"],
                "mode_share_diff": late_stats["fixed_probe"]["mode_share"] - early_stats["fixed_probe"]["mode_share"],
            },
        }

    # UOT trajectory trend.
    trend = []
    for spec in uot_specs:
        tag = spec.tag
        trend.append(
            {
                "tag": tag,
                "t_env": meta[tag]["t_env"],
                "own_js": within[tag]["cross_seed_js"]["mean"],
                "own_entropy": within[tag]["joint_entropy"],
                "own_dominant": within[tag]["dominant_joint_share"],
                "fixed_entropy": fixed_metrics[tag]["entropy_norm"],
                "fixed_hamming_teacher": fixed_metrics[tag]["hamming_to_teacher"],
                "fixed_equal_teacher": fixed_metrics[tag]["equal_teacher"],
                "win_rate": within[tag]["win_rate"]["mean"],
                "return_mean": within[tag]["return"]["mean"],
            }
        )
    trend_slopes = {}
    if len(trend) >= 3:
        t_env = np.asarray([row["t_env"] for row in trend], dtype=np.float64)
        for key in ("own_js", "own_entropy", "own_dominant", "fixed_entropy", "fixed_hamming_teacher"):
            vals = np.asarray([row[key] for row in trend], dtype=np.float64)
            if np.all(np.isfinite(vals)):
                trend_slopes[key] = float(np.polyfit(t_env, vals, 1)[0])

    cluster_composition = {}
    for cluster_id in range(n_clusters):
        members = [episodes[j]["tag"] for j in range(len(episodes)) if labels[j] == cluster_id]
        cluster_composition[str(cluster_id)] = {
            "n": len(members),
            "tag_counts": dict(Counter(members).most_common()),
        }

    payload = {
        "config": args.config,
        "policy_manifest": args.policy_manifest,
        "n_seeds": len(seeds),
        "seed0": seeds[0],
        "probe_fracs": fracs,
        "parameter_updates": 0,
        "policies": [meta[tag] for tag in tags],
        "teacher_sequences": [
            {
                "seed": int(seed),
                "probe_steps": teacher_seqs[seed]["probe_steps"],
                "return": teacher_seqs[seed]["return"],
                "won": teacher_seqs[seed]["won"],
                "length": teacher_seqs[seed]["length"],
                "teacher_actions": {
                    str(t): list(teacher_seqs[seed]["actions"][t])
                    for t in teacher_seqs[seed]["probe_steps"]
                },
            }
            for seed in seeds
        ],
        "episodes": [
            {
                "tag": ep["tag"],
                "seed": ep["seed"],
                "return": ep["return"],
                "won": ep["won"],
                "length": ep["length"],
                "joint_counts": {
                    ",".join(str(int(x)) for x in key): int(value)
                    for key, value in ep["joint_counts"].items()
                },
                "agent_counts": ep["agent_counts"],
                "probes": ep["probes"],
            }
            for ep in episodes
        ],
        "episode_distance": dist_eps.tolist(),
        "policy_distance": {"tags": tags, "own": dist_policy.tolist(), "fixed": dist_fixed.tolist()},
        "policy_linkage": {"own": policy_linkage.tolist(), "fixed": fixed_linkage.tolist()},
        "episode_clustering": clustering,
        "cluster_composition": cluster_composition,
        "within_policy": within,
        "fixed_state": {
            "probe_records": probe_records,
            "metrics": fixed_metrics,
            "signatures": {tag: {str(list(k)): v for k, v in fixed_signatures[tag].items()} for tag in tags},
        },
        "windows": window,
        "uot_trend": trend,
        "uot_trend_slopes": trend_slopes,
        "notes": "Descriptive only: checkpoints are correlated and seed count is small.",
    }

    out = Path(resolve_path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)

    lines = []
    lines.append("H3 mode diversity (descriptive; no training)")
    lines.append(f"policies={len(tags)} seeds={len(seeds)} probes={len(probe_records)}")
    lines.append("")

    def fmt(value, spec: str = ".3f") -> str:
        return "NA" if value is None else format(value, spec)

    lines.append("policy              t_env     ret(win)      ownJS   ownH  dom    fixedH  hamm  eq")
    for tag in tags:
        w = within[tag]
        f = fixed_metrics[tag]
        lines.append(
            f"{tag:18s} {meta[tag]['t_env']:>9d} "
            f"{fmt(w['return']['mean'], '7.2f')}({fmt(w['win_rate']['mean'], '.2f')}) "
            f"{fmt(w['cross_seed_js']['mean'], '.4f')} {fmt(w['joint_entropy'], '5.2f')} "
            f"{fmt(w['dominant_joint_share'])} {fmt(f['entropy_norm'], '6.3f')} "
            f"{fmt(f['hamming_to_teacher'])} {fmt(f['equal_teacher'])}"
        )
    lines.append("")
    lines.append(
        f"episode clustering: K={clustering['best_k']} silhouette={clustering['silhouette']}"
    )
    for tag in tags:
        occ = within[tag]["occupancy"]
        lines.append(
            f"  {tag:18s} occupancy={occ['probs']} eff_modes={occ['effective_modes']:.2f} "
            f"dominant={occ['dominant_share']:.3f}"
        )
    if window:
        lines.append("")
        lines.append("UOT windows (early vs late)")
        lines.append(f"  early: {json.dumps(window['early'])}")
        lines.append(f"  late:  {json.dumps(window['late'])}")
        lines.append(f"  late/early: {json.dumps(window['late_over_early'])}")
    if trend_slopes:
        lines.append("")
        lines.append(f"UOT trend slopes per env step: {json.dumps(trend_slopes)}")
    txt = "\n".join(lines) + "\n"
    txt_path = Path(resolve_path(args.txt))
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(txt, encoding="utf-8")
    print(txt, flush=True)
    print(f"wrote {txt_path}", flush=True)


if __name__ == "__main__":
    main()
