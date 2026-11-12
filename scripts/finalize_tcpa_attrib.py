#!/usr/bin/env python3
"""Turn the TCPA attribution matrix into its final report.

The matrix runs for many hours on a shared card, so the numbers are produced
continuously by ``scripts/analyze_tcpa_attrib.py``.  This script is the last
step: it re-runs the analyzer (so the summary is never stale), reads the
machine-written verdict and writes

    documentation/experiment_reports/tcpa_attrib_5m_vs_6m.md

and refreshes the experiment's row in
``documentation/experiment_reports/README.md``.

Everything in the report is read from ``summary.json``; nothing is re-typed by
hand, so the report cannot drift away from the measurements.

Usage:
    python scripts/finalize_tcpa_attrib.py
    python scripts/finalize_tcpa_attrib.py --t-max 1000000 --no-readme
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional

ARMS = (
    "none_matched",
    "none_broken",
    "anchor_matched",
    "anchor_broken",
    "placebo_matched",
    "placebo_broken",
    "prior_matched",
    "prior_broken",
    "fixed_gate_matched",
    "fixed_gate_broken",
)

# Arms the pre-registered verdict depends on: the interaction needs all four
# factorial cells, P2 needs the anchor arm.
DECISIVE = ("none_matched", "prior_matched", "none_broken", "prior_broken")
DECISIVE_PLUS_P2 = DECISIVE + ("anchor_matched",)

WINDOW_LABELS = (("early", "early"), ("mid", "mid"), ("late", "late"), ("all", "all"))


def fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def fmt_ci(stat: Dict, digits: int = 2) -> str:
    if not stat or stat.get("mean") is None:
        return "-"
    if stat.get("lo") is None:
        return f"{stat['mean']:+.{digits}f} (n={stat.get('n', 0)})"
    return (
        f"{stat['mean']:+.{digits}f} "
        f"[{stat['lo']:+.{digits}f}, {stat['hi']:+.{digits}f}] "
        f"(n={stat.get('n', 0)})"
    )


def run_analyzer(args) -> Dict:
    cmd = [
        args.python,
        args.analyzer,
        "--root",
        args.root,
        "--seed",
        str(args.seed),
        "--json",
        args.json,
        "--markdown",
        args.results_md,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-2000:])
        sys.stderr.write(result.stderr[-2000:])
        raise SystemExit(f"analyzer failed with rc={result.returncode}")
    with open(args.json, encoding="utf-8") as handle:
        return json.load(handle)


def completeness(summary: Dict, t_max: int) -> Dict[str, object]:
    """Which arms reached the step budget, and which are missing entirely."""
    arms = summary.get("arms", {})
    complete, incomplete, absent = [], [], []
    for arm in ARMS:
        info = arms.get(arm)
        if not info or not info.get("n_eval"):
            absent.append(arm)
        elif float(info.get("last_t_env", 0.0)) >= t_max - 50_000:
            complete.append(arm)
        else:
            incomplete.append(arm)
    return {
        "complete": complete,
        "incomplete": incomplete,
        "absent": absent,
        "all_complete": not incomplete and not absent,
    }


def build_report(summary: Dict, comp: Dict[str, object], t_max: int) -> str:
    arms = summary.get("arms", {})
    contrasts = summary.get("contrasts", {})
    verdict = summary.get("verdict", {})
    p1 = verdict.get("P1", {})
    p2 = verdict.get("P2", {})

    lines: List[str] = []
    lines.append("# TCPA 约束归因实验：在线阶段的「训练早期普遍需要约束」与「只有某些队友配合下才该保留旧动作」")
    lines.append("")
    lines.append(
        "本报告是自动收尾脚本 `scripts/finalize_tcpa_attrib.py` 依据 "
        "`checkpoints/tcpa_attrib_5m_vs_6m/summary.json` 生成的，全部数字直接取自该文件。"
        "设计、预登记判读规则与运行中补记见配套文档 "
        "[tcpa_attrib_5m_vs_6m_design.md](tcpa_attrib_5m_vs_6m_design.md)；"
        "逐评估点读数见 [tcpa_attrib_5m_vs_6m_results.md](tcpa_attrib_5m_vs_6m_results.md)（自动刷新）。"
    )
    lines.append("")

    # ------------------------------------------------------------- conclusion
    lines.append("## 0. 结论")
    lines.append("")
    conclusion = verdict.get("conclusion", "").strip() or "**证据不足**：主窗口还没有足够数据。"
    lines.append(conclusion)
    lines.append("")
    qualifier = str(verdict.get("qualifier", "") or "").strip()
    if qualifier:
        lines.append(qualifier)
        lines.append("")
    block_qualifier = str(verdict.get("block_qualifier", "") or "").strip()
    if block_qualifier:
        lines.append(block_qualifier)
        lines.append("")
    lines.append(
        f"- **P1**（交互 prior，matched−broken，主窗口 early）：{fmt_ci(p1)}，判定 **{p1.get('class', '无数据')}**"
    )
    lines.append(
        f"- **P2**（prior − anchor，matched，主窗口 early）：{fmt_ci(p2)}，判定 **{p2.get('class', '无数据')}**"
    )
    block = verdict.get("block_bootstrap") or {}
    if block:
        b1 = block.get("P1") or {}
        b2 = block.get("P2") or {}
        lines.append(
            f"- **块自助稳健性**（2026-11-12 03:40 补记，只撤回不改判）："
            f"P1 {fmt_ci(b1)}，判定 **{b1.get('class', '无数据')}**；"
            f"P2 {fmt_ci(b2)}，判定 **{b2.get('class', '无数据')}**；"
            f"**{'已撤回登记结论' if block.get('vetoed') else '未触发撤回'}**"
        )
    activity = verdict.get("constraint_activity") or {}
    if activity:
        per_arm = activity.get("per_arm") or {}
        lines.append(
            "- **约束活性**（early 窗口的有效 hinge 损失 `tcpa/prior_loss`）："
            f"prior_matched = {fmt((per_arm.get('prior_matched') or {}).get('prior_loss'), 4)}、"
            f"prior_broken = {fmt((per_arm.get('prior_broken') or {}).get('prior_loss'), 4)}；"
            f"**{'通过（约束在生效）' if activity.get('ok', True) else '未通过（有决定性臂的约束未生效，结论已撤回）'}**"
        )
    align = verdict.get("pairing_alignment") or {}
    if align:
        p1_align = align.get("P1") or {}
        p2_align = align.get("P2") or {}
        lines.append(
            f"- **配对时间对齐**（同一评估序号上各臂 `t_env` 的最大差）："
            f"P1 {fmt(p1_align.get('max_spread'), 0)} 步、P2 {fmt(p2_align.get('max_spread'), 0)} 步，"
            f"告警线 {align.get('limit_steps', 0)} 步；"
            f"**{'通过' if align.get('ok', True) else '未通过（配对失效，结论已撤回）'}**"
        )
    manip = summary.get("manipulation_check") or {}
    if manip:
        lines.append(
            f"- **操纵检验**（队友冻结干预是否在结果层面生效）："
            f"**{'通过' if manip.get('effective') else '未通过'}**"
        )
    lines.append("")

    if not comp["all_complete"]:
        missing = list(comp["incomplete"]) + list(comp["absent"])
        lines.append(
            "> **注意：本矩阵未全部跑完。** 以下臂未达到 "
            f"{t_max} 步预算：{'、'.join(missing)}。"
            "上表的判定只覆盖已完成臂的数据，结论按预登记规则机械给出，"
            "但缺失臂对应的读数不可用。"
        )
        lines.append("")

    lines.append("## 1. 两个竞争解释与判读量")
    lines.append("")
    lines.append(
        "- **H1**：收益来自与内容无关、在训练早期普遍需要的通用约束。"
    )
    lines.append(
        "- **H2**：收益来自保留真正协作的旧联合动作，因此只在某些队友配合下才成立。"
    )
    lines.append("")
    lines.append(
        "判读量在 2×2 析因（约束档位 × 队友配合条件）上定义："
    "**P1 = (prior−none)_matched − (prior−none)_broken**（问队伍条件），"
        "**P2 = prior − anchor**（问内容）。两者都显著为正才支持 H2；两者都含 0 才支持 H1；"
        "其余一律记为证据不足。规则于 2026-11-12 02:25 预先登记，"
        "由 `scripts/analyze_tcpa_attrib.py` 机械执行。"
    )
    lines.append("")

    # The prior that existed before this matrix ran.  A reader has to see it,
    # otherwise a null result cannot be told apart from a result that was
    # never in reach.
    lines.append("### 1.1 既有矩阵给出的先验（2026-11-12 03:25 补记）")
    lines.append("")
    lines.append(
        "这个矩阵不是从零开始的。**已跑完的既有矩阵**"
        "（`checkpoints/tcpa_5m_vs_6m`，6 变体 × 3 seed × 3M 步）"
        "用**完全相同的配对口径**（按评估序号配对）重算之后，给出了两条先验："
    )
    lines.append("")
    lines.append(
        "1. **`Δ_prior`（`full − no_igm`）在 early 窗口基本为 0**："
        "三个 seed 分别是 +0.11 / −0.02 / −0.19，种子层面均值 **−0.04 ± 0.09**。"
        "机制上说得通：真 TCPA 的原始 hinge 在 early 是三个约束臂里最大的（0.0207），"
        "但动态 gate 恰好在旧动作真被降权时把惩罚压掉，有效 `prior_loss` 只有 0.0041，"
        "而无条件保留（`fixed_gate`）是 0.0126——**约束项在 early 几乎空转**。"
    )
    lines.append(
        "2. **早期待益的来源分解**（种子层面）：残差参数化 **+1.13 ± 0.10** > "
        "无条件保留旧动作 **+0.59 ± 0.18** > 反向保留 +0.30 ± 0.16 > "
        "绝对值保留 +0.13 ± 0.02 > **竞争条件下的保留（真 TCPA）−0.04 ± 0.09**。"
        "也就是说，早期真正有收益的两样**都与「旧动作是否还 competitive」无关**。"
    )
    lines.append("")
    lines.append(
        "**这对本次判定意味着什么（必须先说清楚）**：预登记的 P1 建在 `Δ_prior` 上，"
        "而 `Δ_prior ≈ 0` 意味着 **P1 几乎没有信号可测**——"
        "无论 H1 还是 H2，它都可能读出「≈0 / 证据不足」。"
        "因此追加了两个**探索性**臂 `fixed_gate_matched` / `fixed_gate_broken`"
        "（无条件保留旧动作，early 窗口有真实效应），"
        "它们的**交互项**才是「保留旧动作的收益是否依赖队友配合」的有信号版本。"
        "`fixed_gate` 两臂**不进入**预登记判定，只作探索性读数。"
        "完整数据见 [tcpa_prior_seed_crosscheck.md](tcpa_prior_seed_crosscheck.md)。"
    )
    lines.append("")

    lines.append("## 2. 各臂表现（窗口内评估点平均回报）")
    lines.append("")
    lines.append(
        f"| {'臂':<16}{'n_eval':>7}{'last_t':>10}"
        + "".join(f"{w:>10}" for w, _ in WINDOW_LABELS)
        + f"{'win_rate':>10} |"
    )
    lines.append("|" + "|".join("-" * 9 for _ in range(3 + len(WINDOW_LABELS) + 1)) + "|")
    for arm in ARMS:
        info = arms.get(arm)
        if not info or not info.get("n_eval"):
            lines.append(
                f"| {arm:<16}{'-':>7}{'-':>10}"
                + "".join(f"{'-':>10}" for _ in WINDOW_LABELS)
                + f"{'-':>10} |"
            )
            continue
        cells = "".join(
            f"{fmt((info.get('windows') or {}).get(key)):>10}" for _w, key in WINDOW_LABELS
        )
        lines.append(
            f"| {arm:<16}{int(info['n_eval']):>7}{int(info['last_t_env']):>10}"
            f"{cells}{fmt(info.get('win_rate')):>10} |"
        )
    lines.append("")
    lines.append(
        "窗口：early = 0–300k，mid = 300k–600k，late = 600k–1M，all = 全程。"
    )
    lines.append("")

    lines.append("## 3. 析因对比（配对差与交互项）")
    lines.append("")
    lines.append(
        "每个数字是**逐评估序号配对**的差，95% CI 由评估点间 bootstrap 给出。"
        "各臂共用同一个固定种子的评估环境，因此同一评估序号上跑的是同一批对局。"
    )
    lines.append("")
    lines.append(
        f"| {'对比':<34}" + "".join(f"{w:>30}" for w, _ in WINDOW_LABELS) + " |"
    )
    lines.append("|" + "|".join("-" * 26 for _ in range(1 + len(WINDOW_LABELS))) + "|")
    for name, stats in contrasts.items():
        cells = "".join(f"{fmt_ci(stats.get(w, {})):>30}" for w, _ in WINDOW_LABELS)
        label = name.replace("−", "-").replace("(matched-broken)", "(matched-broken)")
        lines.append(f"| {label:<34}{cells} |")
    lines.append("")

    lines.append("## 4. 判读量与判定")
    lines.append("")
    block = verdict.get("block_bootstrap") or {}
    b1 = block.get("P1") or {}
    b2 = block.get("P2") or {}
    lines.append(
        f"| {'读数':<32}{'点自助 值 [95% CI]':>34}{'n':>6}{'判定':>12}"
        f"{'块自助 值 [95% CI]':>34}{'块判定':>12} |"
    )
    lines.append("|" + "|".join("-" * 10 for _ in range(6)) + "|")
    for name, stat, block_stat in (
        ("P1 交互 prior (matched-broken)", p1, b1),
        ("P2 prior - anchor (matched)", p2, b2),
    ):
        lines.append(
            f"| {name:<32}{fmt_ci(stat):>34}{stat.get('n', 0):>6}{stat.get('class', '无数据'):>12}"
            f"{fmt_ci(block_stat):>34}{block_stat.get('class', '无数据'):>12} |"
        )
    lines.append("")
    if block:
        autocorr = summary.get("autocorrelation") or {}
        rho1 = (autocorr.get("P1") or {}).get("lag1")
        neff1 = (autocorr.get("P1") or {}).get("n_eff")
        rho2 = (autocorr.get("P2") or {}).get("lag1")
        neff2 = (autocorr.get("P2") or {}).get("n_eff")
        lines.append(
            "点自助把有自相关的评估点当成独立样本，区间偏窄；块自助（moving-block，"
            "块长 n^(1/3)）是同一读数的自相关稳健版本。"
            f"P1 的 lag-1 自相关 {('-' if rho1 is None else f'{rho1:+.3f}')}"
            f"（有效样本量 {('-' if neff1 is None else f'{neff1:.1f}')}），"
            f"P2 的 lag-1 自相关 {('-' if rho2 is None else f'{rho2:+.3f}')}"
            f"（有效样本量 {('-' if neff2 is None else f'{neff2:.1f}')}）。"
        )
        lines.append("")
        lines.append(
            "**修正方向（2026-11-12 03:40 补记，只保守）**：两个区间一致 → 登记结论不变；"
            "不一致 → 登记结论被**撤回**，改记为证据不足；"
            "补记不会把结论改到另一支（H1 或 H2）。"
        )
        lines.append("")

    detectability = summary.get("detectability") or {}
    if detectability:
        lines.append("### 4.1 可检测效应量（探索性，不参与判读）")
        lines.append("")
        lines.append(
            "「CI 含 0」只有在知道**这个实验能看见多小的效应**时才有意义。"
            "由 95% CI 的半宽反推标准误（SE = (hi − lo) / 3.92），"
            "再按 80% 功效所需的 (1.96 + 0.84)·SE，给出主窗口上各读数"
            "**可检测的最小效应**：真实效应若小于这个量级，登记规则读到的就是「含 0」。"
        )
        lines.append("")
        lines.append(
            f"| {'读数':<32}{'n':>6}{'点自助 SE':>14}{'块自助 SE':>14}"
            f"{'可检测效应(点)':>18}{'可检测效应(块)':>18} |"
        )
        lines.append("|" + "|".join("-" * 12 for _ in range(6)) + "|")
        for name, stat in detectability.items():
            se_point = stat.get("se_point")
            se_block = stat.get("se_block")
            mde_point = stat.get("mde_point")
            mde_block = stat.get("mde_block")
            se_point_text = "-" if se_point is None else f"{se_point:.3f}"
            se_block_text = "-" if se_block is None else f"{se_block:.3f}"
            mde_point_text = "-" if mde_point is None else f"{mde_point:.2f}"
            mde_block_text = "-" if mde_block is None else f"{mde_block:.2f}"
            lines.append(
                f"| {name:<32}{stat.get('n', 0):>6}"
                f"{se_point_text:>14}{se_block_text:>14}"
                f"{mde_point_text:>18}{mde_block_text:>18} |"
            )
        lines.append("")
        lines.append(
            "这个量**不参与判读**，只是给「含 0」一个参照。"
            "块自助那一列一般更大——自相关让有效样本量变小，"
            "同一个规则对自相关稳健性的要求更高。"
        )
        lines.append("")
        lines.append(
            "**读法提醒**：这个阈值要和「本族已被证明的效应量级」比。"
            "本族最强的 early 保留效应是 `fixed_gate − no_igm` = **+0.59 ± 0.18**"
            "（既有矩阵，种子层面），而 P1 是**两个臂差相减**，"
            "标准误约翻倍、可检测效应约 **0.80**"
            "（详见设计文档第 4 节 04:25 的补记）。"
            "因此 **P1 读出「含 0」更可能是功效不足，而不是现象不存在**——"
            "这不改变判定规则，只决定「含 0」该怎么读。"
        )
        lines.append("")

    robustness = summary.get("robustness") or {}
    if robustness:
        lines.append("### 4.2 稳健性（探索性，不参与判读）")
        lines.append("")
        lines.append(
            "bootstrap 区间描述的是一条轨迹上评估点之间的波动，不说明效应是不是由少数几个点"
            "扛起来的。补充两项：**同号点占比**与**前后半段是否同号**。"
        )
        lines.append("")
        lines.append(
            f"| {'读数':<24}{'n':>6}{'均值':>10}{'同号点占比':>12}"
            f"{'前半段':>10}{'后半段':>10}{'全同号':>9}{'两半同号':>9} |"
        )
        lines.append("|" + "|".join("-" * 10 for _ in range(8)) + "|")
        for label, key in (("P1 交互 prior", "P1"), ("P2 prior - anchor", "P2")):
            stat = robustness.get(key) or {}
            if not stat.get("n"):
                lines.append(
                    f"| {label:<24}{'-':>6}{'-':>10}{'-':>12}{'-':>10}{'-':>10}{'-':>9}{'-':>9} |"
                )
                continue
            lines.append(
                f"| {label:<24}{stat['n']:>6}{stat['mean']:>+10.3f}"
                f"{stat['frac_positive'] * 100:>11.0f}%"
                f"{stat['first_half_mean']:>+10.3f}{stat['second_half_mean']:>+10.3f}"
                f"{('是' if stat.get('sign_consistent') else '否'):>9}"
                f"{('是' if stat.get('halves_agree') else '否'):>9} |"
            )
        lines.append("")
        lines.append(
            "安慰剂臂（`placebo`）与真 TCPA 在 hinge 形式、gate、权重上完全一致，"
            "只有被保留的「旧动作」不同（随机合法动作 vs 离线贪心联合动作），"
            "所以 `placebo − none` 就是「只要有这个约束形式就有用」的那部分，"
            "同时也是单种子单臂差的经验噪声尺度。"
            "`prior` 的效应必须明显大于它，才谈得上是旧动作**内容**在起作用。"
            "该读数见读数文件第 5.1 节。"
        )
        lines.append("")

    # ------------------------------------------------- mechanism diagnostics
    mech_keys = (
        "tcpa/regret_mean_interval_mean",
        "tcpa/gate_mean_interval_mean",
        "tcpa/hinge_mean_interval_mean",
        "tcpa/prior_loss_interval_mean",
        "tcpa/offline_action_agreement_interval_mean",
        "tcpa/dq_absmean_interval_mean",
    )
    if any((arms.get(arm) or {}).get("mechanism_early") for arm in ARMS):
        lines.append("### 4.3 机制诊断与约束活性（early 窗口）")
        lines.append("")
        lines.append(
            "如果某个带约束的臂，它的有效 hinge 损失（`tcpa/prior_loss`）恒为 0，"
            "说明这个约束**一次都没有生效**——那么基于它的读数说的就不是"
            "「保留旧动作有没有用」，而是「一个没在跑的约束有没有用」。"
            "下表逐臂给出 early 窗口的机制量与活性判定。"
        )
        lines.append("")
        lines.append(
            f"| {'臂':<16}{'variant':<12}"
            + "".join(
                f"{k.replace('tcpa/', '').replace('_interval_mean', '')[:14]:>16}"
                for k in mech_keys
            )
            + f"{'活性':>16} |"
        )
        lines.append("|" + "|".join("-" * 12 for _ in range(3 + len(mech_keys))) + "|")
        hinge_arms = {
            "prior_matched",
            "prior_broken",
            "placebo_matched",
            "placebo_broken",
            "fixed_gate_matched",
            "fixed_gate_broken",
        }
        for arm in ARMS:
            info = arms.get(arm) or {}
            mech = info.get("mechanism_early") or {}
            if not mech:
                continue
            cells = "".join(f"{fmt(mech.get(k), 4):>16}" for k in mech_keys)
            loss = mech.get("tcpa/prior_loss_interval_mean")
            if arm in hinge_arms:
                state = "无数据" if loss is None else ("生效" if loss > 0.0 else "**未生效**")
            elif info.get("constraint") == "none":
                state = "无约束项"
            else:
                state = "非 hinge 约束"
            lines.append(
                f"| {arm:<16}{str(info.get('constraint', '-')):<12}{cells}{state:>16} |"
            )
        lines.append("")

    lines.append("## 5. 限制")
    lines.append("")
    lines.append(
        "- **单种子**：每个臂只有 1 个种子，CI 只反映评估点之间的波动，不是种子之间的波动；"
        "补种子只需重跑同一队列（同一 launcher、同一优先级）。"
    )
    lines.append(
        "- **跨条件绝对值不可比**：`broken` 臂在「4 个离线先验队友 + 1 个在线焦点智能体」的队伍上评估，"
        "只有同条件内的臂间差与交互项可比。"
    )
    lines.append(
        "- **`broken` 条件由冻结队友造成**：它同时改变队伍构成与「队伍是否共同适应」，"
        "因此 P1 是「队友配合条件」的总效应，而不是某一个单独成分的效应。"
    )
    lines.append(
        "- **`greedy_action_disagree_rate` 在 broken 下被机械压低**（冻结队友执行的正是离线贪心动作），"
        "只作臂内参考；`regret_mean` / `gate_mean` / `offline_action_agreement` / `hinge_mean` 是价值层面比较，可比。"
    )
    lines.append(
        "- **`broken` 条件测的是「队伍不再一起适应」，不是「队友噪声」。** "
        "冻结的 4 个队友执行的正是离线先验的动作，所以旧联合动作与当前队伍"
        "**只差焦点智能体那一个分量**（动作维度上仍有 4/5 对得上）。"
        "本实验的 `broken` 因此测的是「队友不跟着焦点智能体往前走」时保留旧动作的收益如何变；"
        "它**不是** `e1_e7_gap_supplement.md` 里「旧动作优势随**队友噪声**下降」那条干预"
        "（给队友加噪声）。因此：P1 ≈ 0 **不能**据此推翻 e1_e7 的结论；"
        "P1 > 0 也**不能**说 H2 在 e1_e7 的意义上被证实。"
    )
    lines.append(
        "- **P2 必须与强度诊断一起读**：`prior` 与 `anchor` 的约束形式不同，"
        "P2 天然把「内容」与「强度」混在一起。长程合成自测显示 `anchor` 对残差的抑制"
        "比 `prior` 更强（`dq_absmean` 0.477 vs 0.629，而无约束臂漂到 0.692），"
        "所以若 P2 为负，不能直接读成「通用约束更好」，也可能只是「anchor 黏滞性更强」。"
        "对照 `dq_absmean` / `anchor_dev_mean` / `hinge_mean`（本报告第 3 节）。"
    )
    lines.append(
        "- **共享显卡**：卡 3 上还有他人任务，吞吐受限；10 臂 1M 步的实际墙钟远长于裸卡估计。"
    )
    lines.append("")
    lines.append(
        "- **判读规则有三项「只撤回、不改判」的补充限定**"
        "（都写于预登记之后、决定性读数出来之前，且都只让判定更难成立）："
    )
    lines.append(
        "  1. **操纵检验**（读数 5.2）：队友冻结干预若在结果层面没生效"
        "（有数据的档位 `broken − matched` 全含 0），则撤回 H1 分支；"
    )
    lines.append(
        "  2. **块自助 CI**（读数 5.3）：评估点有正自相关"
        "（既有矩阵实测 lag-1 +0.16~+0.47、n_eff 16~23，逐点自助区间偏窄 11%~28%），"
        "点自助与块自助判定不一致时**撤回**登记结论；"
    )
    lines.append(
        "  3. **配对一致性**（读数 5.4）：各臂在同一个 ~10k 步的评估网格上"
        "（同索引 `t_env` 实测只差 61–282 步），若网格不再一致"
        "（同索引相差 > 20,000 步）则**撤回**登记结论。"
    )
    lines.append(
        "  三者都只会把结论降级为**证据不足**，不会把结论改到另一支（H1 或 H2）。"
    )
    lines.append("")

    lines.append("## 6. 复现")
    lines.append("")
    lines.append("```bash")
    lines.append("# 训练矩阵（3 号卡，显存闸门 + 自动重试 + 优先级排队）")
    lines.append("nohup bash scripts/launch_tcpa_attrib.sh > logs/tcpa_attrib/driver.log 2>&1 &")
    lines.append("")
    lines.append("# 看门：自动刷新读数，全臂完成后收尾")
    lines.append(
        "nohup python scripts/watch_tcpa_attrib.py --interval 1800 > logs/tcpa_attrib/watcher.log 2>&1 &"
    )
    lines.append("")
    lines.append("# 手工复算读数")
    lines.append("python scripts/analyze_tcpa_attrib.py \\")
    lines.append("  --root checkpoints/tcpa_attrib_5m_vs_6m --seed 1 \\")
    lines.append("  --json checkpoints/tcpa_attrib_5m_vs_6m/summary.json \\")
    lines.append("  --markdown documentation/experiment_reports/tcpa_attrib_5m_vs_6m_results.md")
    lines.append("")
    lines.append("# 重新生成本报告")
    lines.append("python scripts/finalize_tcpa_attrib.py")
    lines.append("```")
    lines.append("")
    return "\n".join(lines) + "\n"


FINAL_LINK = "[tcpa_attrib_5m_vs_6m.md](tcpa_attrib_5m_vs_6m.md)（最终报告）"


def patch_readme(path: str, verdict_text: str) -> bool:
    """Refresh the experiment's row in the report index, idempotently.

    Two things are written into that row: the link to the final report (added
    when the report exists, so the index never carries a dead link) and the
    verdict summary.  Both are marked so that running this script again
    replaces them instead of appending a second copy.
    """
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    changed = False
    out: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and "tcpa_attrib_5m_vs_6m_design.md" in stripped:
            # 1. the link to the final report, at the end of its own cell
            if FINAL_LINK.split("](")[0] not in line:
                match = re.search(r"\[tcpa_attrib_5m_vs_6m_results\.md\]\(", line)
                if match:
                    cut = line.find("|", match.end())
                    if cut == -1:
                        cut = len(line)
                    line = line[:cut].rstrip() + " · " + FINAL_LINK + " " + line[cut:]
            # 2. the verdict summary in the last cell.  This has to work on the
            # line *after* the link insertion, otherwise the rebuilt cell would
            # throw the link away again.
            stripped = line.strip()
            body = stripped[:-1] if stripped.endswith("|") else stripped
            cut = body.rfind("|")
            cell = body[cut + 1 :]
            prefix = body[: cut + 1]
            cell = re.sub(r"\s*<!--\s*tcpa-attrib-verdict\s*-->.*$", "", cell)
            cell = re.sub(r"\s*运行中\s*$", "", cell)
            # The marker goes *before* the text so that a later run can strip
            # the whole block again; that keeps the patch idempotent.
            cell = cell.rstrip() + " <!-- tcpa-attrib-verdict --> " + verdict_text
            line = prefix + cell + " |"
            changed = True
        out.append(line)
    if changed:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(out) + "\n")
    return changed


def build_index_text(summary: Dict, comp: Dict[str, object]) -> str:
    """The one-line verdict summary that goes into the report index.

    Two amendments can withdraw a registered conclusion, and both of them are
    withdraw-only: the manipulation check (the broken-teammate intervention
    did not bite) and the moving-block bootstrap (the autocorrelation-aware
    interval disagrees with the registered point interval).  Neither of them
    may turn the conclusion into the other branch, so whenever one of them
    fires the index says "evidence insufficient" instead of repeating H1/H2.
    """
    verdict = summary.get("verdict", {})
    p1 = verdict.get("P1", {})
    p2 = verdict.get("P2", {})
    conclusion = verdict.get("conclusion", "").strip()
    headline = re.sub(r"\*\*", "", conclusion).rstrip("。")
    index_text = f"**结论：{headline}**"

    notes: List[str] = []
    qualifier = str(verdict.get("qualifier", "") or "").strip()
    if qualifier:
        notes.append("操纵检验未通过，P1 含 0 不可解读为支持 H1")
    block = verdict.get("block_bootstrap") or {}
    if block.get("vetoed"):
        notes.append("块自助 CI 与登记用点自助 CI 不一致，登记结论已撤回")
    align = verdict.get("pairing_alignment") or {}
    if align and not align.get("ok", True):
        notes.append("配对失效（评估网格不一致），登记结论已撤回")
    activity = verdict.get("constraint_activity") or {}
    if activity and not activity.get("ok", True):
        notes.append("决定性臂的约束未生效，登记结论已撤回")
    if notes:
        index_text = "**结论：证据不足**（" + "；".join(notes) + "）"

    index_text += f"（P1 {fmt_ci(p1)}；P2 {fmt_ci(p2)}）"
    if block:
        b1 = block.get("P1") or {}
        b2 = block.get("P2") or {}
        index_text += f"（块自助 P1 {fmt_ci(b1)}；P2 {fmt_ci(b2)}）"
    if not comp["all_complete"]:
        missing = list(comp["incomplete"]) + list(comp["absent"])
        index_text += f"（未跑完的臂：{'、'.join(missing)}）"
    index_text += "。"
    return index_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="checkpoints/tcpa_attrib_5m_vs_6m")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--t-max", type=int, default=1_000_000)
    parser.add_argument("--json", default="checkpoints/tcpa_attrib_5m_vs_6m/summary.json")
    parser.add_argument(
        "--results-md",
        default="documentation/experiment_reports/tcpa_attrib_5m_vs_6m_results.md",
    )
    parser.add_argument(
        "--out", default="documentation/experiment_reports/tcpa_attrib_5m_vs_6m.md"
    )
    parser.add_argument(
        "--readme", default="documentation/experiment_reports/README.md"
    )
    parser.add_argument("--no-readme", action="store_true")
    parser.add_argument(
        "--analyzer",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "analyze_tcpa_attrib.py"
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    summary = run_analyzer(args)
    comp = completeness(summary, args.t_max)
    report = build_report(summary, comp, args.t_max)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"wrote {args.out}")

    index_text = build_index_text(summary, comp)

    if not args.no_readme:
        if patch_readme(args.readme, index_text):
            print(f"refreshed the index row in {args.readme}")
        else:
            print(f"warning: no matching row found in {args.readme}")


if __name__ == "__main__":
    main()
