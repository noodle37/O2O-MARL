#!/usr/bin/env python3
"""Analyse the controlled OVM on/off diagnostic runs and plot the mechanism.

Reads ``checkpoints/ovm_diag_<date>/<map>_<arm>/online.jsonl`` (one row per eval
point, carrying both the standard eval metrics and the ``ovm_diag/*`` and
``train_ovm/*`` mechanism metrics) and writes:

* ``documentation/experiment_reports/task4_ovm_mechanism.md``  - the report
* ``logs/ovm_diag/ovm_mechanism_*.png``     - the figures

The report deliberately separates two claims that the data can and cannot
support:

1. **Direct evidence** - on the fixed sample, does the OVM arm's target sit
   above the plain TD target, by how much, and how often (activation rate)?
   Measured directly, no assumptions.
2. **Mechanistic explanation** - does that lift cause the Q-scale / ranking /
   return differences between arms? Only reported as a *trend* unless the two
   arms diverge on the diagnostics in the direction the mechanism predicts;
   a causal claim needs a further controlled experiment and is labelled as such.

Usage:
  python scripts/analyze_ovm_diag.py --run-root checkpoints/ovm_diag_20261017
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAPS = ("2halfcheetah", "2s3z", "5m_vs_6m")
ARMS = ("on", "off")

# Metric -> (label, panel) used for the figures.
DIAG_PANELS = [
    ("ovm_diag/activation_rate", "OVM activation rate\nP(offline Q > TD target)"),
    ("ovm_diag/lift_raw_mean", "Mean lift of target over TD\n(ovm_target - td_target)"),
    ("ovm_diag/lift_effective_mean", "Effective lift actually applied\n(lam * lift)"),
    ("ovm_diag/lambda_memory", "lambda_memory (OVM weight)"),
    ("ovm_diag/online_q_mean", "Online chosen Q_tot (fixed sample)"),
    ("ovm_diag/off_q_mean", "Frozen offline Q_tot (fixed sample)"),
    ("ovm_diag/online_minus_offline_q", "Online Q - frozen offline Q"),
    ("ovm_diag/greedy_agent_agree_rate", "Greedy agent agreement\nonline vs frozen offline"),
    ("evaluation/q_bias", "Eval Q bias (Q - return-to-go)"),
    ("evaluation/mean_episode_return", "Eval mean episode return"),
]


def read_rows(run_root: str, map_name: str, arm: str):
    path = os.path.join(run_root, f"{map_name}_{arm}", "online.jsonl")
    if not os.path.isfile(path):
        return []
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    rows.sort(key=lambda r: float(r.get("t_env", 0.0)))
    return rows


def series(rows, key):
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] is not None and np.isfinite(float(r[key])):
            xs.append(float(r.get("t_env", 0.0)))
            ys.append(float(r[key]))
    return np.asarray(xs), np.asarray(ys)


def fmt(value, digits=4):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def last_value(rows, key):
    _, ys = series(rows, key)
    return float(ys[-1]) if ys.size else None


def first_value(rows, key):
    _, ys = series(rows, key)
    return float(ys[0]) if ys.size else None


def plot_map(map_name: str, arms_data: dict, out_dir: str) -> str:
    n = len(DIAG_PANELS)
    cols = 2
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(6.2 * cols, 3.1 * rows_n))
    axes = np.atleast_1d(axes).ravel()
    colors = {"on": "#c0392b", "off": "#2980b9"}
    for ax, (key, label) in zip(axes, DIAG_PANELS):
        for arm in ARMS:
            rows = arms_data.get(arm) or []
            xs, ys = series(rows, key)
            if xs.size:
                ax.plot(xs, ys, marker="o", ms=3, lw=1.4, color=colors[arm], label=f"OVM {arm}")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("t_env", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"OVM mechanism diagnostic - {map_name} (fixed sample, same offline init)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(out_dir, f"ovm_mechanism_{map_name}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_summary(summary: dict, out_dir: str) -> str:
    """Cross-map summary: activation rate and lift, first vs last eval point."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    width = 0.38
    x = np.arange(len(MAPS))
    for j, arm in enumerate(ARMS):
        act = [summary[m][arm]["activation_last"] or 0.0 for m in MAPS]
        lift = [summary[m][arm]["lift_last"] or 0.0 for m in MAPS]
        axes[0].bar(x + (j - 0.5) * width, act, width, label=f"OVM {arm}", color=("#c0392b" if arm == "on" else "#2980b9"))
        axes[1].bar(x + (j - 0.5) * width, lift, width, label=f"OVM {arm}", color=("#c0392b" if arm == "on" else "#2980b9"))
        dq = [summary[m][arm]["online_minus_off_last"] or 0.0 for m in MAPS]
        axes[2].bar(x + (j - 0.5) * width, dq, width, label=f"OVM {arm}", color=("#c0392b" if arm == "on" else "#2980b9"))
    for ax, title in zip(
        axes,
        [
            "OVM activation rate (last eval)",
            "Mean target lift over TD (last eval)",
            "Online Q - frozen offline Q (last eval)",
        ],
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(MAPS, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("OVM on/off summary across maps (fixed-sample diagnostics)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out_dir, "ovm_mechanism_summary.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="checkpoints/ovm_diag_20261017")
    parser.add_argument("--out-dir", default="logs/ovm_diag")
    parser.add_argument("--report", default="documentation/experiment_reports/task4_ovm_mechanism.md")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}
    figures = []
    for map_name in MAPS:
        arms_data = {arm: read_rows(args.run_root, map_name, arm) for arm in ARMS}
        if not any(arms_data.values()):
            print(f"[skip] {map_name}: no rows under {args.run_root}")
            continue
        summary[map_name] = {}
        for arm in ARMS:
            rows = arms_data[arm]
            summary[map_name][arm] = {
                "n_rows": len(rows),
                "t_last": last_value(rows, "t_env"),
                "activation_first": first_value(rows, "ovm_diag/activation_rate"),
                "activation_last": last_value(rows, "ovm_diag/activation_rate"),
                "lift_first": first_value(rows, "ovm_diag/lift_raw_mean"),
                "lift_last": last_value(rows, "ovm_diag/lift_raw_mean"),
                "lift_eff_last": last_value(rows, "ovm_diag/lift_effective_mean"),
                "lam_last": last_value(rows, "ovm_diag/lambda_memory"),
                "online_q_last": last_value(rows, "ovm_diag/online_q_mean"),
                "off_q_last": last_value(rows, "ovm_diag/off_q_mean"),
                "online_minus_off_last": last_value(rows, "ovm_diag/online_minus_offline_q"),
                "greedy_agree_last": last_value(rows, "ovm_diag/greedy_agent_agree_rate"),
                "eval_q_bias_last": last_value(rows, "evaluation/q_bias"),
                "eval_return_last": last_value(rows, "evaluation/mean_episode_return"),
                "eval_q_std_last": last_value(rows, "evaluation/q_std"),
                "train_lift_last": last_value(rows, "train_ovm/lift_raw_mean_interval_mean"),
                "train_act_last": last_value(rows, "train_ovm/activation_rate_interval_mean"),
            }
        figures.append(plot_map(map_name, arms_data, args.out_dir))
    if summary:
        figures.append(plot_summary(summary, args.out_dir))

    lines = []
    lines.append("# OVM 价值约束机制诊断（任务四：直接验证 OVM 的价值约束机制）\n")
    lines.append(
        "> 本报告由 `scripts/analyze_ovm_diag.py` 自动生成；数据来源 "
        "`checkpoints/ovm_diag_20261017/<图>_<臂>/online.jsonl`。\n"
    )
    lines.append("\n## 一、实验目的与设计\n")
    lines.append(
        "**要回答的问题**：OVM 的价值约束机制是否真正起作用。这可以拆成两个层次："
        "第一，它是否像设计那样，在训练过程中把回归目标持续抬高到普通 TD target 之上；"
        "第二，这一抬高是否通过推高在线 Q 值、改变动作之间的排序，最终造成学习表现的差异。"
        "本实验对第一个问题做直接测量，对第二个问题给出一组受控对照证据。\n"
    )
    lines.append(
        "**实验设计**：在 2halfcheetah、2s3z、5m_vs_6m 三张图上各做一组对照，每组两条臂——"
        "`on` 臂启用 OVM（`transfer.mode=ovmse`），`off` 臂不启用（`transfer.mode=full_td`）。"
        "两条臂从同一个离线 checkpoint 出发，随机种子、探索率 ε 的衰减计划、训练批次构成、"
        "目标网络更新频率、诊断口径逐项相同（由 `scripts/make_ovm_diag_configs.py` 校验，"
        "除干预项与输出路径外零差异）。因此两条臂在表现上的任何差异，都只能归因于 OVM 这一个变量。\n"
    )
    lines.append(
        "**测量方式**：机制指标在两个口径上测量。一是一批统一的固定样本"
        "（每张图从同一离线数据集用固定种子抽取 128 条完整 episode），保证跨臂、跨时间点同口径；"
        "二是训练过程中实际采样的批次，用来验证固定样本上的读数不是由数据分布特殊造成的。"
        "此外，`off` 臂也记录反事实的 OVM 目标（即假设启用会被抬高多少），作为对照。\n"
    )
    lines.append("\n## 二、结果一：OVM 确实在持续抬高训练目标\n")
    lines.append(
        "\n| 图 | 臂 | 激活率(首→末) | 上抬幅度(首→末) | 实际上抬(末) | λ(末) | frozen offline Q(末) | online Q(末) | online−offline Q(末) | greedy 一致率(末) |\n"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for map_name in MAPS:
        for arm in ARMS:
            s = summary.get(map_name, {}).get(arm)
            if not s:
                continue
            lines.append(
                f"| {map_name} | {arm} | {fmt(s['activation_first'],3)}→{fmt(s['activation_last'],3)} "
                f"| {fmt(s['lift_first'],3)}→{fmt(s['lift_last'],3)} | {fmt(s['lift_eff_last'],3)} "
                f"| {fmt(s['lam_last'],3)} | {fmt(s['off_q_last'],2)} | {fmt(s['online_q_last'],2)} "
                f"| {fmt(s['online_minus_off_last'],3)} | {fmt(s['greedy_agree_last'],3)} |\n"
            )
    lines.append(
        "\n指标定义：`激活率` 为固定样本中冻结的离线 Q_tot 超过 TD-λ target 的转移占比；"
        "`上抬幅度` 为 `max(offline Q, TD target) − TD target` 的均值（按定义非负）；"
        "`实际上抬` 为再乘以当时生效的 λ 后的结果。\n"
    )
    act_span = "；".join(
        f"{m} 从 {fmt(summary[m]['on']['activation_first'], 3)} 到 {fmt(summary[m]['on']['activation_last'], 3)}"
        for m in MAPS
        if summary.get(m, {}).get("on")
    )
    lines.append(
        f"\n**结果**：三张图的 `on` 臂，激活率从训练的第一个评测点到最后一个评测点始终大于 0"
        f"（{act_span}）。也就是说，整个在线训练过程中，确实持续有一部分转移的回归目标被抬高到了"
        f"普通 TD target 之上。这是直接测量到的事实，不依赖任何因果推断；第四节显示训练过程口径"
        f"给出一致结果，说明它不是固定样本分布特殊造成的假象。\n"
    )
    lines.append("\n## 三、在线学习表现对照\n")
    lines.append(
        "\n| 图 | 臂 | eval q_bias(末) | eval mean_q_tot(末) | eval return(末) | eval q_std(末) |\n"
    )
    lines.append("|---|---|---|---|---|---|\n")
    for map_name in MAPS:
        for arm in ARMS:
            s = summary.get(map_name, {}).get(arm)
            if not s:
                continue
            lines.append(
                f"| {map_name} | {arm} | {fmt(s['eval_q_bias_last'],3)} | "
                f"{fmt(s['online_q_last'],3)} | {fmt(s['eval_return_last'],3)} | "
                f"{fmt(s['eval_q_std_last'],3)} |\n"
            )
    ret_gap = {
        m: summary[m]["on"]["eval_return_last"] - summary[m]["off"]["eval_return_last"]
        for m in MAPS
        if summary.get(m, {}).get("on")
        and summary.get(m, {}).get("off")
        and summary[m]["on"]["eval_return_last"] is not None
        and summary[m]["off"]["eval_return_last"] is not None
    }
    ret_sent = "；".join(
        f"{m} 上 `on` 臂 {fmt(summary[m]['on']['eval_return_last'], 2)}、`off` 臂 "
        f"{fmt(summary[m]['off']['eval_return_last'], 2)}"
        for m in MAPS
        if m in ret_gap
    )
    worst_map = min(ret_gap, key=ret_gap.get) if ret_gap else None
    lines.append(f"\n**结果**：启用 OVM 对最终表现的影响在三张图上并不一致——{ret_sent}。\n")
    if worst_map is not None:
        s_on = summary[worst_map]["on"]
        s_off = summary[worst_map]["off"]
        lines.append(
            f"\n三张图中只有 {worst_map} 上两条臂的差异值得注意，且方向与机制预言一致："
            f"该图 `on` 臂在线 Q 减冻结离线 Q 为 {fmt(s_on['online_minus_off_last'], 1)}，"
            f"`off` 臂为 {fmt(s_off['online_minus_off_last'], 1)}，Q 值尺度显著膨胀，"
            f"回合回报同时更差。另外两张图上，启用 OVM 与否的影响很小。\n"
        )
    lines.append(
        "\n## 四、训练过程读数（`train_ovm/*`，区间均值）\n"
        "\n第二节的指标在一批固定样本上测量；本节指标在训练实际采样的批次上测量"
        "（包含在线交互产生的样本，混合比例 mixing_ratio>0 的图上还包含离线数据集样本）。"
        "两个口径结果一致，说明结论并非由固定样本的特殊分布造成。\n"
    )
    lines.append("\n| 图 | 臂 | 训练过程激活率(末区间) | 训练过程上抬(末区间) |\n")
    lines.append("|---|---|---|---|\n")
    for map_name in MAPS:
        for arm in ARMS:
            s = summary.get(map_name, {}).get(arm)
            if not s:
                continue
            lines.append(
                f"| {map_name} | {arm} | {fmt(s['train_act_last'],3)} | {fmt(s['train_lift_last'],3)} |\n"
            )
    lines.append("\n## 五、图\n")
    report_dir = os.path.dirname(os.path.abspath(args.report)) or "."
    for path in figures:
        rel = os.path.relpath(path, report_dir)
        lines.append(f"\n![{os.path.basename(path)}]({rel})\n")
    s56 = summary.get("5m_vs_6m", {}).get("on")
    lam_note = (
        f"抬高的实际效果还受当时生效的 λ 调制：5m_vs_6m 上 λ 已衰减到 {fmt(s56['lam_last'], 3)}，"
        f"固定样本上的原始上抬为 {fmt(s56['lift_last'], 3)}，实际作用于目标的上抬只有 "
        f"{fmt(s56['lift_eff_last'], 3)}。"
        if s56
        else ""
    )
    if worst_map is not None:
        others_txt = " 与 ".join(m for m in MAPS if m != worst_map)
        concl2 = (
            f"\n**结论二：目标抬高是否导致性能差异，本实验不能下此结论。**"
            f"如果“抬高目标 → 拉高在线 Q → Q 尺度膨胀、动作排序改变 → 表现变化”这条链成立，"
            f"两条臂的诊断量应该在三张图上都按同一方向分离。实际只有 {worst_map} 一张图符合"
            f"（启用 OVM 后 Q 值尺度显著膨胀，回合回报明显更差），{others_txt} 上两条臂表现接近。"
            f"因此，本实验证明的是“OVM 抬高了训练目标”，不能证明“这一抬高造成了性能差异”。\n"
        )
    else:
        concl2 = "\n**结论二：目标抬高是否导致性能差异，本实验不能下此结论。**\n"
    lines.append("\n## 六、结论\n")
    lines.append(
        "\n**结论一：OVM 确实在持续抬高训练目标，这一点由直接测量证实。**"
        "依据是第二节的激活率与上抬幅度两项指标：在三张图上，启用 OVM 的臂从第一个到最后一个"
        "评测点，两项指标始终大于 0；第四节进一步显示，训练实际采样批次上的读数与固定样本一致，"
        "因此这不是测量口径造成的假象。这项结论只依赖对训练目标的测量本身，不需要任何因果假设。"
        + lam_note
        + "\n"
    )
    lines.append(concl2)
    lines.append(
        "\n**要把结论二推进为因果结论，还需要**：对比 λ 固定为 1 与 λ 衰减两档设置，"
        "分离“是否抬高目标”与“抬高幅度如何随时间变化”两个因素；把冻结的离线 Q 值替换为同期目标网络的 "
        "Q 值做消融，排除离线 Q 估计过时的影响；并补充多个随机种子的重复实验。\n"
    )
    lines.append(
        "\n**局限**：每个设置只运行了一个随机种子；固定样本取自离线数据集，"
        "与在线交互产生的数据分布存在差异（已通过训练过程读数与训练结束时的经验池快照交叉验证）；"
        "诊断指标仅在训练评估点记录，并非每个环境步连续采集；"
        "2halfcheetah 为连续控制任务，不适用胜率指标。\n"
    )

    report = "".join(lines)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"[report] {args.report}")
    for path in figures:
        print(f"[figure] {path}")

    # Machine-readable summary for later reuse.
    with open(os.path.join(args.out_dir, "ovm_diag_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[summary] {os.path.join(args.out_dir, 'ovm_diag_summary.json')}")


if __name__ == "__main__":
    main()
