#!/usr/bin/env python3
"""Aggregate the 2026-10-24 cross-arm / seed-replication batch.

Reads every result JSON in a run directory (default
``checkpoints/extra_20261024``) plus ``job_status.jsonl``, and writes one
machine-readable ``index.json`` with:

* A: per-arm/per-version factorial effects g_i / I_ij, best/worst transplant,
     Q calibration, and paired episode-return bootstrap CIs vs ``all``.
* B: per-arm/per-version randomisation estimate of the teacher effect, the
     critic diagnostics, the confidence gates, and the conflict composition.
* C: per-seed next-state R^2 for additive / centralized / pairwise models and
     the interaction-to-single ratios.
* D: per-seed goal information, implementation multiplicity, variance
     reduction, persistence, and goal-support imbalance.
* E: per-arm/per-version U_v(j) for each replay group and the best group,
     with the excess over random.
* F: per-run B_v(z) vs T_v(z) for every (collector, student, budget), the two
     rankings and their Spearman correlation.

Nothing is recomputed from raw data here; this is a reader for the per-run
JSONs produced by ``scripts/launch_extra_runs.sh``.

Usage:
  python scripts/analyze_extra_runs.py --root checkpoints/extra_20261024 \
      --out checkpoints/extra_20261024/index.json
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr

N_AGENTS = 5
N_BOOT = 2000
BOOT_SEED = 20261024


def paired_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    """Mean(a-b) and percentile bootstrap CI over aligned episodes."""
    n = min(len(a), len(b))
    diff = np.asarray(a[:n], dtype=np.float64) - np.asarray(b[:n], dtype=np.float64)
    if n == 0:
        return {"n": 0, "mean": None, "ci": [None, None]}
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boots = diff[idx].mean(axis=1)
    return {
        "n": int(n),
        "mean": float(diff.mean()),
        "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
    }


def rank_spearman(b: Dict[str, float], t: Dict[str, float]):
    keys = [k for k in b if k in t]
    if len(keys) < 2:
        return None, None, None
    bv = np.array([b[k] for k in keys]) + np.arange(len(keys)) * 1e-12
    tv = np.array([t[k] for k in keys]) + np.arange(len(keys)) * 1e-12
    b_rank = sorted(keys, key=lambda k: -b[k])
    t_rank = sorted(keys, key=lambda k: -t[k])
    return float(spearmanr(bv, tv).statistic), b_rank, t_rank


def analyse_a(payload: dict, arm: str, t: str, rng: np.random.Generator) -> dict:
    configs = payload["natural"][t]
    names = configs.keys()
    base = configs["all"]
    out = {
        "seed": payload.get("seed"),
        "arm": arm,
        "t": int(t),
        "n_episodes": base.get("n_episodes"),
        "all_return": base["mean_return"],
        "all_win": base.get("win_rate"),
        "all_spearman_rho": base.get("spearman_rho"),
        "all_bias_normalized": base.get("bias_normalized_mean"),
        "all_team_deviation": base.get("mean_team_deviation"),
    }
    singles = [f"single_{i}" for i in range(N_AGENTS) if f"single_{i}" in names]
    if len(singles) == N_AGENTS:
        out["g"] = [base["mean_return"] - configs[f"single_{i}"]["mean_return"] for i in range(N_AGENTS)]
        pairs = [list(p) for p in itertools.combinations(range(N_AGENTS), 2)]
        out["pairs"] = pairs
        out["I"] = [
            configs[f"pair_{i}_{j}"]["mean_return"]
            - configs[f"single_{i}"]["mean_return"]
            - configs[f"single_{j}"]["mean_return"]
            + base["mean_return"]
            for i, j in pairs
        ]
    returns = {n: configs[n]["mean_return"] for n in names if n != "all"}
    out["returns"] = returns
    best = max(returns, key=returns.get)
    worst = min(returns, key=returns.get)
    out["best_transplant"] = {"name": best, "return": returns[best], "gap_vs_all": returns[best] - base["mean_return"]}
    out["worst_transplant"] = {"name": worst, "return": returns[worst]}
    out["spread"] = returns[best] - returns[worst]

    # Paired CIs need the per-episode returns saved by --save-episodes.
    if "episode_returns" in base:
        all_eps = base["episode_returns"]
        paired = {}
        for name in names:
            if name == "all" or "episode_returns" not in configs[name]:
                continue
            paired[name] = paired_ci(configs[name]["episode_returns"], all_eps, rng)
        out["paired_vs_all"] = paired
        if best in paired:
            out["best_paired"] = paired[best]
    return out


def analyse_b(payload: dict, arm: str) -> dict:
    a = payload.get("analysis", {})
    # The probe names this block ``overall_arm``; keep ``overall`` as a fallback
    # for runs written before the rename.
    overall = a.get("overall_arm") or a.get("overall") or {}
    gate = a.get("confidence_gate_q0.25", {}).get("kept", {})
    gate_half = a.get("confidence_gate_q0.5", {}).get("kept", {})
    sign_gate = a.get("sign_gate_tau_hat", {}).get("kept", {})
    return {
        "seed": payload.get("seed"),
        "arm": arm,
        "t": int(payload["policy_version"]),
        "trials": a.get("n_trials"),
        "conflict_episodes": payload.get("conflict_episodes"),
        "teacher_effect": overall.get("arm_diff"),
        "teacher_effect_ci": overall.get("arm_diff_ci"),
        "p_positive": overall.get("p_arm_diff_positive"),
        "tau_ipw_mean": a.get("tau_ipw_mean"),
        "tau_ipw_ci": a.get("tau_ipw_ci"),
        "tau_hat_mean": a.get("tau_hat_mean"),
        "tau_hat_positive_rate": a.get("tau_hat_positive_rate"),
        "spearman_tau_hat_Y": a.get("spearman_tau_hat_Y"),
        "spearman_neg_q_on_margin_Y": a.get("spearman_neg_q_on_margin_Y"),
        "auc_tau_hat": a.get("auc_tau_hat_for_positive_Y"),
        "auc_neg_q_on_margin": a.get("auc_neg_q_on_margin_for_positive_Y"),
        "auc_q_off_margin": a.get("auc_q_off_margin_for_positive_Y"),
        "gate_q0.25": gate.get("arm_diff"),
        "gate_q0.25_ci": gate.get("arm_diff_ci"),
        "gate_q0.5": gate_half.get("arm_diff"),
        "gate_q0.5_ci": gate_half.get("arm_diff_ci"),
        "sign_gate_kept": sign_gate.get("arm_diff"),
        "sign_gate_ci": sign_gate.get("arm_diff_ci"),
        "by_n_disagree": [
            {
                "n_disagree": row["n_disagree"],
                "n": row["arm"]["n"],
                "arm_diff": row["arm"]["arm_diff"],
                "ci": row["arm"]["arm_diff_ci"],
            }
            for row in a.get("groups_by_n_disagree", [])
        ],
        "composition": a.get("composition"),
    }


def analyse_e(payload: dict, stem: str, rng: np.random.Generator) -> dict:
    rows = {}
    for t, v in payload["versions"].items():
        groups = v["groups"]
        rets = {g: groups[g]["eval"]["return"] for g in groups}
        u = {g: groups[g]["U_return"] for g in groups}
        best = max(u, key=u.get)
        base_eps = v["theta0"].get("episode_returns")
        paired = {}
        if base_eps:
            paired = {
                g: paired_ci(groups[g]["eval"].get("episode_returns", []), base_eps, rng)
                for g in groups
                if groups[g]["eval"].get("episode_returns")
            }
        random_eps = groups.get("random", {}).get("eval", {}).get("episode_returns")
        excess_paired = {}
        if random_eps:
            excess_paired = {
                g: paired_ci(groups[g]["eval"].get("episode_returns", []), random_eps, rng)
                for g in groups
                if g != "random" and groups[g]["eval"].get("episode_returns")
            }
        rows[t] = {
            "theta0_return": v["theta0"]["return"],
            "group_return": rets,
            "U": u,
            "U_paired": paired,
            "excess_over_random_paired": excess_paired,
            "best_group": best,
            "best_excess_over_random": u[best] - u["random"],
            "td_minus_random": u["td"] - u["random"],
        }
    return {
        "file": stem,
        "seed": payload.get("seed"),
        "arm": payload.get("arm", "A"),
        "lr": payload.get("lr"),
        "updates": payload.get("updates"),
        "batch_eps": payload.get("batch_eps"),
        "eval_episodes": payload.get("eval_episodes"),
        "versions": rows,
    }


def analyse_f(payload: dict, stem: str, rng: np.random.Generator) -> dict:
    collectors = payload["collectors"]
    b = {z: c["B_own_policy"]["return"] for z, c in collectors.items()}
    students = sorted({s for c in collectors.values() for s in c.get("students", {})})
    out = {
        "file": stem,
        "collect_episodes": payload.get("collect_episodes"),
        "student_lr": payload.get("student_lr"),
        "eval_episodes": payload.get("eval_episodes"),
        "B": b,
        "behavior_return": {z: c["data"]["behavior_return"] for z, c in collectors.items()},
        "behavior_length": {z: c["data"]["mean_length"] for z, c in collectors.items()},
        "students": {},
    }
    for s in students:
        budgets = sorted(
            {bud for c in collectors.values() for bud in c.get("students", {}).get(s, {})},
            key=lambda x: int(x.lstrip("u")),
        )
        base_eps = (
            payload.get("students", {}).get(s, {}).get("base_eval", {}).get("episode_returns")
        )
        out["students"][s] = {}
        for bud in budgets:
            t, ret, t_paired = {}, {}, {}
            for z, c in collectors.items():
                entry = c.get("students", {}).get(s, {}).get(bud)
                if entry is None:
                    continue
                t[z] = entry["T_return"]
                ret[z] = entry["eval"]["return"]
                eps = entry["eval"].get("episode_returns")
                if base_eps and eps:
                    t_paired[z] = paired_ci(eps, base_eps, rng)
            rho, b_rank, t_rank = rank_spearman(b, t)
            out["students"][s][bud] = {
                "T": t,
                "T_paired": t_paired,
                "return": ret,
                "B_rank": b_rank,
                "T_rank": t_rank,
                "spearman_B_T": rho,
            }
    return out


def analyse_c(payload: dict, stem: str) -> dict:
    models = payload["models"]
    return {
        "file": stem,
        "split_seed": payload.get("split_seed"),
        "model_seed": payload.get("model_seed"),
        "epochs": payload.get("epochs"),
        "n_train": payload.get("n_train"),
        "n_test": payload.get("n_test"),
        "r2": {m: {t: v["test"][t]["r2"] for t in v["test"]} for m, v in models.items()},
        "n_params": {m: v["n_params"] for m, v in models.items()},
        "interaction_to_single": {
            m: v.get("interaction_to_single") for m, v in payload.get("cross_agent_sensitivity", {}).items()
        },
    }


def analyse_d(payload: dict, stem: str) -> dict:
    rows = {}
    for h, hv in payload["horizons"].items():
        rows[h] = {}
        for k, kv in hv["Ks"].items():
            goals = kv.get("goals", [])
            supports = [g["support"] for g in goals]
            rows[h][k] = {
                "goal_action_info": kv["goal_action_info"],
                "info_fraction": kv["goal_action_info"] / kv["weighted_action_entropy_state_only"],
                "implementations": kv["mean_implementations_ge10pct"],
                "variance_reduction_vs_state": kv["goal_variance_reduction_vs_state"],
                "kappa": kv["goal_persistence_kappa"],
                "silhouette": kv["silhouette"],
                "n_goals": len(goals),
                "support_min": min(supports) if supports else None,
                "support_max": max(supports) if supports else None,
            }
    return {"file": stem, "seed": payload.get("seed"), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="checkpoints/extra_20261024")
    ap.add_argument("--out", default="checkpoints/extra_20261024/index.json")
    args = ap.parse_args()

    rng = np.random.default_rng(BOOT_SEED)
    index = {
        "generated_from": os.path.abspath(args.root),
        "job_status": [],
        "A": [],
        "B": [],
        "C": [],
        "D": [],
        "E": [],
        "F": [],
        "missing_or_running": [],
    }

    status_path = os.path.join(args.root, "job_status.jsonl")
    if os.path.isfile(status_path):
        with open(status_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    index["job_status"].append(json.loads(line))

    files = sorted(glob.glob(os.path.join(args.root, "*.json")))
    files = [p for p in files if not p.endswith("index.json")]
    seen = set()
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.startswith("C_split") or stem.startswith("D_seed"):
            payload = json.load(open(path, encoding="utf-8"))
            if "models" in payload:
                index["C"].append(analyse_c(payload, stem))
            else:
                index["D"].append(analyse_d(payload, stem))
            seen.add(stem)
            continue
        try:
            payload = json.load(open(path, encoding="utf-8"))
        except Exception as exc:  # incomplete file still being written
            print(f"[skip] {stem}: {exc}")
            continue
        if stem.startswith("A_"):
            arm = stem.split("_")[1].replace("arm", "")
            for t in payload.get("natural", {}):
                index["A"].append(analyse_a(payload, arm, t, rng))
            seen.add(stem)
        elif stem.startswith("B_"):
            parts = stem.split("_")
            arm = parts[1].replace("arm", "")
            index["B"].append(analyse_b(payload, arm))
            seen.add(stem)
        elif stem.startswith("E_"):
            index["E"].append(analyse_e(payload, stem, rng))
            seen.add(stem)
        elif stem.startswith("F_"):
            index["F"].append(analyse_f(payload, stem, rng))
            seen.add(stem)

    # Anything announced in the status file but without a result yet.
    done = {row["tag"] for row in index["job_status"] if "end" in row}
    for row in index["job_status"]:
        if "start" in row and row["tag"] not in done and row["tag"] not in seen:
            index["missing_or_running"].append(row["tag"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, ensure_ascii=False, default=str)

    print(f"A runs: {len(index['A'])}  B: {len(index['B'])}  C: {len(index['C'])} "
          f"D: {len(index['D'])}  E: {len(index['E'])}  F: {len(index['F'])}")
    if index["missing_or_running"]:
        print("still running / missing: " + ", ".join(index["missing_or_running"]))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
