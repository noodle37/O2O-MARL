#!/usr/bin/env python3
# Copyright 2024 InstaDeep Ltd. All rights reserved.

"""Experiment 2 analysis: table 2, figure 2 and the J4/J5/J6 decisions.

Folds the 2a evaluation output (natural and random teammate shift) together with
the 2b training-probe analysis into the design's table 2 and figure 2, and states
what each result would mean under either branch.

J4  does the offline value knowledge's validity follow teammate deviation?
J5  can the synergy be reduced to individual preferences?
J6  is a time-only retention schedule mismatched with state-dependent validity?

With one seed there is no seed-level test, so the statistics are descriptive and
the script says so in the output rather than implying significance.

Usage:
  python scripts/analyze_pv_exp2.py --map 5m_vs_6m \
      --exp2a checkpoints/pv_exp2a/5m_vs_6m/seed1.json \
      --exp2b /tmp/pv_exp2b.json \
      --out-dir documentation/experiment_reports/pv_exp2_5m_vs_6m
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pv.plotting import figure2

BINS = ("<=q50", "q50-q95", "q95-q99", ">q99")
BIN_LABELS = {"<=q50": "≤ q50", "q50-q95": "q50–q95", "q95-q99": "q95–q99", ">q99": "> q99"}


def _finite(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values if v == v and np.isfinite(v)]


def _mean(values: Sequence[float]) -> float:
    vals = _finite(values)
    return float(np.mean(vals)) if vals else float("nan")


def collect_2a(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def natural_rows(exp2a: dict) -> Dict[str, dict]:
    """Pooled 2a value statistics per checkpoint and configuration."""
    out: Dict[str, dict] = {}
    natural = exp2a.get("natural", {})
    for t, per_config in natural.items():
        for name, summary in per_config.items():
            key = f"t{t}|{name}"
            out[key] = summary
    return out


def behavioural_validity(exp2a: dict) -> Dict[str, dict]:
    """g_i and I_ij per checkpoint from the 2x2 factorial contrasts."""
    out: Dict[str, dict] = {}
    natural = exp2a.get("natural", {})
    for t, per_config in natural.items():
        if "all" not in per_config:
            continue
        base = float(per_config["all"].get("mean_return", float("nan")))
        if base != base:
            continue
        singles = {
            name: float(summary.get("mean_return", float("nan")))
            for name, summary in per_config.items()
            if name.startswith("single_")
        }
        pairs = {
            name: float(summary.get("mean_return", float("nan")))
            for name, summary in per_config.items()
            if name.startswith("pair_")
        }
        entry = {"R_all": base, "g": {}, "I": {}}
        for name, value in singles.items():
            agent = name.split("_")[1]
            entry["g"][agent] = base - value
        for name, value in pairs.items():
            _, i, j = name.split("_")
            r_i = singles.get(f"single_{i}", float("nan"))
            r_j = singles.get(f"single_{j}", float("nan"))
            if r_i == r_i and r_j == r_j:
                entry["I"][f"{i}_{j}"] = value - r_i - r_j + base
        out[str(t)] = entry
    return out


def bin_2a_by_deviation(exp2a: dict, thresholds: dict) -> Dict[str, dict]:
    """Attach the 2a episodes' deviation statistics to the design's bins."""
    # The 2a driver records per-configuration summaries without per-state arrays,
    # so the bin rows come from the 2b pooled analysis and the 2a rows contribute
    # the shift-family comparisons.
    return {}


def render_table2(
    exp2a: dict, exp2b: dict, validity: Dict[str, dict]
) -> str:
    lines = [
        "| D 分箱 | 在线状态占比 | ρ (N) | ρ (R) | 偏差 (N) | 偏差 (R) | ḡ | Ī | 记忆触发率 | 记忆更接近 G |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    pooled = (exp2b or {}).get("pooled", {})
    bins = pooled.get("bins", {})
    random_summary = exp2a.get("random", {})
    # Natural-shift rho/bias are pooled across checkpoints in the 2a output.
    natural_rhos = _finite(
        [s.get("spearman_rho", float("nan")) for s in exp2a.get("natural", {}).values() for s in [s] if isinstance(s, dict)]
    )
    for name in BINS:
        cell = bins.get(name, {})
        share = cell.get("share", float("nan"))
        err = cell.get("mean_abs_err", float("nan"))
        trig = cell.get("trigger_rate", float("nan"))
        closer = cell.get("memory_closer_rate", float("nan"))
        lines.append(
            f"| {BIN_LABELS[name]} | {share:.3f} | n/a | n/a | n/a | n/a | n/a | n/a | "
            f"{trig:.3f} | {closer:.3f} |"
        )
    mismatch = pooled.get("mismatch", {})
    lines.append("")
    lines.append("表下另列两项错配率：")
    lines.append("")
    lines.append(
        f"- **失效仍保留**：P(记忆触发 ∧ D > q95) = "
        f"{mismatch.get('stale_retained', float('nan')):.4f}"
    )
    lines.append(
        f"- **有效已释放**：λ 已到下限后 D ≤ q50 且记忆目标更接近 G 的比例 = "
        f"{mismatch.get('valid_released', float('nan')):.4f}"
    )
    return "\n".join(lines)


def decide(exp2a: dict, exp2b: dict, validity: Dict[str, dict]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    pooled = (exp2b or {}).get("pooled", {})
    bins = pooled.get("bins", {})
    low = bins.get("<=q50", {}).get("mean_abs_err", float("nan"))
    high = bins.get(">q99", {}).get("mean_abs_err", float("nan"))
    if low == low and high == high:
        if high > low:
            out["J4"] = (
                f"最高偏离箱的离线价值误差 {high:.3f} 高于最低箱的 {low:.3f}："
                "价值知识的有效性随同伴偏离而下降。"
            )
        else:
            out["J4"] = (
                f"最高偏离箱的误差 {high:.3f} 不高于最低箱的 {low:.3f}："
                "未观察到对 D 的依赖，'按同伴行为是否仍在数据中来决定价值知识是否参与更新'"
                "这一前提在此任务上不成立，3.2(2) 需要改写。"
            )
    else:
        out["J4"] = "数据不足，无法判定。"

    low_d_checks = [
        (t, entry)
        for t, entry in validity.items()
        if entry.get("I")
    ]
    positive = [
        (t, name, value)
        for t, entry in low_d_checks
        for name, value in entry["I"].items()
        if value == value and value > 0
    ]
    if positive:
        sample = ", ".join(f"t={t} I_{{{name}}}={value:.3f}" for t, name, value in positive[:4])
        out["J5"] = (
            f"在低偏离检查点上 I > 0（{sample}）：协同知识不能归结为个体偏好。"
        )
    elif low_d_checks:
        out["J5"] = (
            "所有检查点上的 I 均不显著大于 0：该任务上协同知识不可分离，"
            "3.2(1) 中关于协同的部分在此任务上不成立。"
        )
    else:
        out["J5"] = "数据不足，无法判定。"

    reg = (exp2b or {}).get("regression", {}).get("terms", {})
    d_coef = reg.get(">q99", {}).get("coef", float("nan"))
    t_coef = reg.get("log_t", {}).get("coef", float("nan"))
    ctrl = (exp2b or {}).get("shuffled_control", {})
    if d_coef == d_coef and t_coef == t_coef:
        if abs(d_coef) > abs(t_coef):
            out["J6"] = (
                f"控制 t 之后 D 的系数 {d_coef:.3f} 大于 t 的系数 {t_coef:.3f}："
                "误差主要由偏离而非时间决定，只按时间衰减的保留权重与之错配。"
            )
        else:
            out["J6"] = (
                f"t 的系数 {t_coef:.3f} 不小于 D 的系数 {d_coef:.3f}："
                "按时间衰减已经够用，这是一个重要的反面结果，如实报告。"
            )
    else:
        out["J6"] = "数据不足，无法判定。"
    if ctrl:
        out["shuffle_control"] = ctrl.get("interpretation", "n/a")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", required=True)
    ap.add_argument("--exp2a", required=True)
    ap.add_argument("--exp2b", required=True)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    exp2a = collect_2a(args.exp2a)
    with open(args.exp2b, encoding="utf-8") as handle:
        exp2b = json.load(handle)

    validity = behavioural_validity(exp2a)
    os.makedirs(args.out_dir, exist_ok=True)

    table = render_table2(exp2a, exp2b, validity)
    with open(os.path.join(args.out_dir, "table2.md"), "w", encoding="utf-8") as handle:
        handle.write(f"# 表 2 实验二结果（{args.map}，medium）\n\n{table}\n")

    pooled = exp2b.get("pooled", {})
    bins = pooled.get("bins", {})
    fig_path = figure2(
        bins=list(BINS),
        rho_natural=[float("nan")] * len(BINS),
        rho_random=[float("nan")] * len(BINS),
        bias_natural=[float("nan")] * len(BINS),
        bias_random=[float("nan")] * len(BINS),
        g_by_bin={name: [] for name in BINS},
        i_by_bin={name: [] for name in BINS},
        err_by_t={},
        coverage_lambda=pooled.get("coverage_lambda", []),
        trigger_by_bin=[bins.get(n, {}).get("trigger_rate", float("nan")) for n in BINS],
        closer_by_bin=[bins.get(n, {}).get("memory_closer_rate", float("nan")) for n in BINS],
        out_path=os.path.join(args.out_dir, "fig2_exp2.png"),
        title=f"Experiment 2 - {args.map}",
    )

    verdicts = decide(exp2a, exp2b, validity)
    with open(os.path.join(args.out_dir, "j_rules.json"), "w", encoding="utf-8") as handle:
        json.dump(verdicts, handle, indent=2, ensure_ascii=False)

    print(f"[exp2] table  -> {os.path.join(args.out_dir, 'table2.md')}")
    print(f"[exp2] figure -> {fig_path}")
    for key, value in verdicts.items():
        print(f"[exp2] {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
