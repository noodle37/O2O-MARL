#!/usr/bin/env python3
# Copyright 2024 InstaDeep Ltd. All rights reserved.

"""Experiment 1 analysis: table 1, figure 1 and the J1/J2/J3 decisions.

Reads every arm's online.jsonl (plus the supplementary preference-retention and
coverage measurements when present), computes the design's metrics, and writes
the table, the figure and a JSON summary.

With a single seed there are no paired differences: the sign-rank test bottoms
out at p = 0.5, so the paired-difference rows are emitted as "n/a (1 seed)" and
the J-rules are evaluated on descriptive comparisons only.  This is stated in
the table itself rather than silently dropped.

Usage:
  python scripts/analyze_pv_exp1.py --map 5m_vs_6m \
      --run-root checkpoints/pv_exp1/5m_vs_6m \
      --out-dir documentation/experiment_reports/pv_exp1_5m_vs_6m
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pv.exp1_metrics import (
    METRIC_FIELD,
    ArmSummary,
    bootstrap_ci,
    drop,
    holm,
    kaplan_meier_median,
    paired_sign_rank_p,
    read_curve,
    recovery_steps,
    tail_mean,
)
from pv.plotting import ARM_LABELS, figure1

MAP_METRIC = {
    "5m_vs_6m": "win_rate",
    "2halfcheetah": "return",
}
ARMS = ("A", "B", "C", "D", "Z")
SEEDS_PER_ARM = {"A": 10, "B": 6, "C": 10, "D": 10, "Z": 6}


def arm_run_dir(run_root: str, arm: str, seed: int) -> str:
    return os.path.join(run_root, f"{arm}_seed{seed}")


def collect(
    run_root: str, seeds: Sequence[int], metric: str, t_max: int
) -> Dict[str, Dict[str, list]]:
    """curves[metric_name][arm] -> [(t, value), ...] pooled over seeds."""
    curves: Dict[str, Dict[str, list]] = {
        "win_rate": {},
        "value_bias": {},
        "preference_retention": {},
        "rho_cov": {},
    }
    for arm in ARMS:
        for key in curves:
            curves[key][arm] = []
        for seed in seeds:
            path = os.path.join(arm_run_dir(run_root, arm, seed), "online.jsonl")
            rows = read_curve(path, METRIC_FIELD[metric])
            for t, value in rows:
                curves["win_rate"][arm].append((t, value))
            bias = read_curve(path, "evaluation/q_bias")
            curves["value_bias"][arm].extend(bias)
    return curves


def supplementary(run_root: str, seeds: Sequence[int]) -> Dict[str, Dict[str, list]]:
    """Preference retention and coverage measurements, keyed by arm and t."""
    out: Dict[str, Dict[str, list]] = {"preference_retention": {}, "rho_cov": {}}
    for arm in ARMS:
        for key in out:
            out[key][arm] = []
        for seed in seeds:
            base = arm_run_dir(run_root, arm, seed)
            for name in os.listdir(base) if os.path.isdir(base) else []:
                if not name.startswith("pref_") or not name.endswith(".json"):
                    continue
                with open(os.path.join(base, name), encoding="utf-8") as handle:
                    payload = json.load(handle)
                t = int(payload.get("t_env", -1))
                out["preference_retention"][arm].append(
                    (t, float(payload.get("preference_retention", float("nan"))))
                )
            for name in os.listdir(base) if os.path.isdir(base) else []:
                if not name.startswith("cov_") or not name.endswith(".json"):
                    continue
                with open(os.path.join(base, name), encoding="utf-8") as handle:
                    payload = json.load(handle)
                t = int(payload.get("t_env", -1))
                out["rho_cov"][arm].append((t, float(payload.get("rho_cov", float("nan")))))
    for key in out:
        for arm in ARMS:
            out[key][arm].sort(key=lambda item: item[0])
    return out


def per_seed_metrics(run_root: str, seeds: Sequence[int], metric: str) -> Dict[str, dict]:
    """Per-arm, per-seed Drop / recovery / final values."""
    out: Dict[str, dict] = {}
    for arm in ARMS:
        entries = []
        for seed in seeds:
            path = os.path.join(arm_run_dir(run_root, arm, seed), "online.jsonl")
            curve = read_curve(path, METRIC_FIELD[metric])
            if not curve:
                continue
            t_off = curve[0][1]
            entries.append(
                {
                    "seed": seed,
                    "t_off": t_off,
                    "drop": drop(curve, t_off),
                    "recovery": recovery_steps(curve, t_off),
                    "final": tail_mean(curve, 5),
                }
            )
        out[arm] = {"n_seeds": len(entries), "entries": entries}
    return out


def render_table(per_arm: Dict[str, dict], supp: Dict[str, Dict[str, list]]) -> str:
    """Markdown table 1: median [IQR] per cell."""
    header = (
        "| 臂 | 种子数 | Drop | 恢复步数 (k) | WR@1M | 尺度偏差 @0 | "
        "偏好保持 @50k | ρ_cov @500k |\n|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    summaries: Dict[str, ArmSummary] = {}
    for arm in ARMS:
        info = per_arm[arm]
        summary = ArmSummary(arm=arm, n_seeds=info["n_seeds"])
        for entry in info["entries"]:
            summary.drops.append(entry["drop"])
            summary.recoveries.append(entry["recovery"])
            summary.finals.append(entry["final"])
        pref = [v for _, v in supp["preference_retention"][arm] if 0 <= _ <= 60_000]
        cov = [v for t, v in supp["rho_cov"][arm] if 400_000 <= t <= 600_000]
        summary.extra["pref@50k"] = pref
        summary.extra["cov@500k"] = cov
        summaries[arm] = summary
        row = summary.row()
        lines.append(
            f"| {ARM_LABELS[arm]} | {row['n_seeds']} | {row['drop']} | {row['recovery_k']} | "
            f"{row['final']} | n/a | {row.get('pref@50k', 'n/a')} | {row.get('cov@500k', 'n/a')} |"
        )
    # Paired-difference rows (empty with one seed, filled when seeds are added).
    for left, right in (("D", "A"), ("C", "D")):
        diffs = [
            d["drop"] - a["drop"]
            for d, a in zip(per_arm[left]["entries"], per_arm[right]["entries"])
            if len(per_arm[left]["entries"]) == len(per_arm[right]["entries"])
        ]
        if len(diffs) >= 2:
            lo, hi = bootstrap_ci(diffs)
            lines.append(
                f"| **{left} − {right} 配对差 [95% CI]** | {len(diffs)} | "
                f"{np.mean(diffs):.3f} [{lo:.3f}, {hi:.3f}] | | | | | |"
            )
        else:
            lines.append(
                f"| **{left} − {right} 配对差 [95% CI]** | {len(diffs)} | "
                f"n/a（单 seed 无配对检验） | | | | | |"
            )
    return "\n".join(lines)


def decide(per_arm: Dict[str, dict], supp: Dict[str, Dict[str, list]]) -> Dict[str, str]:
    """J1/J2/J3 verdicts on the available evidence, with both branches written out."""
    def med(arm: str, key: str) -> float:
        vals = [e[key] for e in per_arm[arm]["entries"]]
        vals = [v for v in vals if v == v]
        return float(np.median(vals)) if vals else float("nan")

    def pref_med(arm: str) -> float:
        vals = [v for _, v in supp["preference_retention"][arm]]
        vals = [v for v in vals if v == v]
        return float(np.median(vals)) if vals else float("nan")

    out: Dict[str, str] = {}
    d_drop, a_drop = med("D", "drop"), med("A", "drop")
    if d_drop == d_drop and a_drop == a_drop:
        if d_drop < a_drop and pref_med("D") > pref_med("A"):
            out["J1"] = (
                f"Drop(D)={d_drop:.3f} < Drop(A)={a_drop:.3f} 且 D 的偏好保持率更高："
                "早期崩塌主要由价值整体尺度失准引起，只保留尺度即可稳住早期。"
            )
        else:
            out["J1"] = (
                f"Drop(D)={d_drop:.3f} 与 Drop(A)={a_drop:.3f} 接近：整体尺度不是主因，"
                "解释转向依赖状态的价值失真或同伴偏移，由实验二检验。"
            )
    else:
        out["J1"] = "数据不足，无法判定。"

    b_drop, b_final, b_cov = med("B", "drop"), med("B", "final"), pref_med("B")
    a_cov = pref_med("A")
    if b_drop == b_drop:
        cov_b = [v for t, v in supp["rho_cov"]["B"] if 400_000 <= t <= 600_000]
        cov_a = [v for t, v in supp["rho_cov"]["A"] if 400_000 <= t <= 600_000]
        cov_b_m = float(np.median(cov_b)) if cov_b else float("nan")
        cov_a_m = float(np.median(cov_a)) if cov_a else float("nan")
        if cov_b_m == cov_b_m and cov_b_m > cov_a_m:
            out["J2"] = (
                f"臂 B 的 Drop={b_drop:.3f} 最小但 WR@1M={b_final:.3f} 最低、"
                f"ρ_cov={cov_b_m:.3f} 高于臂 A 的 {cov_a_m:.3f}：保留约束把样本困在覆盖区内。"
            )
        else:
            out["J2"] = (
                f"臂 B 的 ρ_cov={cov_b_m:.3f} 不高于臂 A 的 {cov_a_m:.3f}："
                "困在覆盖区这一代价不成立，B 的最终性能差需要另找原因。"
            )
    else:
        out["J2"] = "数据不足，无法判定。"

    c_final, d_final = med("C", "final"), med("D", "final")
    c_drop, d_drop2 = med("C", "drop"), med("D", "drop")
    if c_final == c_final and d_final == d_final:
        if abs(c_drop - d_drop2) < 0.05 and c_final < d_final:
            out["J3"] = (
                f"臂 C 与臂 D 的早期稳定性相当（Drop {c_drop:.3f} 对 {d_drop2:.3f}），"
                f"但 C 的 WR@1M={c_final:.3f} 低于 D 的 {d_final:.3f}："
                "整体保留在带来稳定的同时也带来了限制，二者可以拆开。"
            )
        else:
            out["J3"] = (
                f"臂 C 的 WR@1M={c_final:.3f} 不低于臂 D 的 {d_final:.3f}："
                "离线价值网络携带的信息超出整体尺度，3.2(1) 需要据此修正值得保留的成分。"
            )
    else:
        out["J3"] = "数据不足，无法判定。"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", required=True, choices=sorted(MAP_METRIC))
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--t-max", type=int, default=1_000_000)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    metric = MAP_METRIC[args.map]
    curves = collect(args.run_root, args.seeds, metric, args.t_max)
    supp = supplementary(args.run_root, args.seeds)
    per_arm = per_seed_metrics(args.run_root, args.seeds, metric)

    os.makedirs(args.out_dir, exist_ok=True)
    table = render_table(per_arm, supp)
    with open(os.path.join(args.out_dir, "table1.md"), "w", encoding="utf-8") as handle:
        handle.write(f"# 表 1 实验一结果（{args.map}，medium）\n\n{table}\n")

    fig_path = figure1(
        curves,
        os.path.join(args.out_dir, "fig1_exp1.png"),
        t_off=per_arm["A"]["entries"][0]["t_off"] if per_arm["A"]["entries"] else None,
        drop_mark=("A", per_arm["A"]["entries"][0]["drop"]) if per_arm["A"]["entries"] else None,
        recovery_mark=(
            ("A", per_arm["A"]["entries"][0]["recovery"])
            if per_arm["A"]["entries"] and per_arm["A"]["entries"][0]["recovery"]
            else None
        ),
        title=f"Experiment 1 - {args.map}",
    )

    verdicts = decide(per_arm, supp)
    with open(os.path.join(args.out_dir, "j_rules.json"), "w", encoding="utf-8") as handle:
        json.dump(verdicts, handle, indent=2, ensure_ascii=False)

    print(f"[exp1] table  -> {os.path.join(args.out_dir, 'table1.md')}")
    print(f"[exp1] figure -> {fig_path}")
    for key, value in verdicts.items():
        print(f"[exp1] {key}: {value}")
    print(f"[exp1] seeds used: {args.seeds} (single seed => no paired tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
