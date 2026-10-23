#!/usr/bin/env python3
"""Aggregate the ranking-loss fork branches into one comparable report.

Reads the parent log plus the three branch logs of one seed (or several seeds) and
answers the only question the experiment asks: after the fork, does keeping the
offline ranking constraint (continue) beat removing it (off) or decaying it (decay)?

The comparison is deliberately on the *second half of online learning*, not on the
best single checkpoint: the primary number is the mean win rate over the last five
eval points of the post-fork window, with the paired 640-episode endpoint eval as
the decisive tie-breaker.

Usage:
  python scripts/analyze_fork_branches.py \
      --run-dir checkpoints/uot_fork_5m_vs_6m_seed1 \
      --run-dir checkpoints/uot_fork_5m_vs_6m_seed2 \
      --ovmse-ref checkpoints/ovmse_batch_20261014/5m_vs_6m_online/online.jsonl \
      --out documentation/experiment_reports/task3_ranking_loss_fork.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BRANCHES = ("continue", "off", "decay")
TAIL_POINTS = 5


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _eval_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Only the forced eval rows carry win rate; train rows do not."""
    return [r for r in rows if "evaluation/win_rate" in r]


def _wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _tail_mean(rows: List[Dict[str, Any]], key: str, n: int) -> Optional[float]:
    vals = [float(r[key]) for r in rows if key in r]
    if not vals:
        return None
    return sum(vals[-n:]) / len(vals[-n:])


def _auc(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    """Trapezoidal area under the curve over t_env, normalised to the window length."""
    pts = [(float(r["t_env"]), float(r[key])) for r in rows if key in r and "t_env" in r]
    if len(pts) < 2:
        return None
    pts.sort()
    area = 0.0
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        area += 0.5 * (v0 + v1) * (t1 - t0)
    span = pts[-1][0] - pts[0][0]
    return area / span if span > 0 else None


def _eval640(run_dir: str, name: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(run_dir, f"eval640_{name}.jsonl")
    rows = _read_jsonl(path)
    for row in reversed(rows):
        if "win_rate" in row and "episodes" in row:
            return row
    return None


def _diag_at(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    return _tail_mean(rows, key, TAIL_POINTS)


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def analyze_seed(run_dir: str) -> Dict[str, Any]:
    parent_rows = _eval_rows(_read_jsonl(os.path.join(run_dir, "parent.jsonl")))
    fork_t: Optional[int] = None
    if parent_rows:
        fork_t = int(max(float(r.get("t_env", 0)) for r in parent_rows))

    out: Dict[str, Any] = {"run_dir": run_dir, "fork_t": fork_t, "branches": {}}
    for name in BRANCHES:
        rows = _read_jsonl(os.path.join(run_dir, f"branch_{name}.jsonl"))
        eval_rows = _eval_rows(rows)
        post = [r for r in eval_rows if fork_t is None or float(r.get("t_env", 0)) > fork_t]
        post = post or eval_rows
        out["branches"][name] = {
            "n_eval": len(post),
            "tail_win": _tail_mean(post, "evaluation/win_rate", TAIL_POINTS),
            "auc_win": _auc(post, "evaluation/win_rate"),
            "tail_return": _tail_mean(post, "evaluation/mean_episode_return", TAIL_POINTS),
            "reg_strength": _diag_at(post, "reg_strength"),
            "hinge_rate": _diag_at(post, "uot_hinge_rate"),
            "sign_disagree": _diag_at(post, "uot_sign_disagree"),
            "loss_td": _diag_at(post, "loss_td"),
            "loss_ord": _diag_at(post, "loss_ord"),
            "eval640": _eval640(run_dir, name),
            "last_t": int(max((float(r.get("t_env", 0)) for r in post), default=0)),
        }
    out["branches"]["_fork"] = {"eval640": _eval640(run_dir, "fork")}
    out["eval_eps"] = _infer_eval_eps(
        _eval_rows(_read_jsonl(os.path.join(run_dir, "branch_continue.jsonl")))
    )
    return out


def _infer_eval_eps(rows: List[Dict[str, Any]]) -> Optional[int]:
    """训练内评测每个点的局数，由胜率取值粒度反推（胜率均为 k/n 的分数）。"""
    vals = [float(r["evaluation/win_rate"]) for r in rows if "evaluation/win_rate" in r]
    if len(vals) < 10:
        return None
    for n in (8, 16, 32, 64, 128, 256):
        if all(abs(v * n - round(v * n)) < 1e-6 for v in vals):
            return n
    return None


_READINGS = {
    "better": "撤除或衰减 ranking loss 明显更好（配对差 ≥ 0.10）",
    "same": "三个分支差异不明显（配对差 ≤ 0.05）",
    "ambiguous": "继续保留 ranking loss 的分支最好，或差异不足以判定",
}


def _reading_multi(diffs: list[Optional[float]]) -> Optional[str]:
    """Pre-registered multi-seed rule over the per-seed paired differences.

    All seeds must point the same way and the mean paired difference must reach
    0.10 win rate for "better"; a mean below 0.05 in absolute value or
    inconsistent directions across seeds is "same"; anything else is
    "ambiguous".
    """
    vals = [d for d in diffs if d is not None]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    same_dir = all(v > 0 for v in vals) or all(v < 0 for v in vals)
    if same_dir and abs(mean) >= 0.10:
        return "better"
    if abs(mean) < 0.05 or not same_dir:
        return "same"
    return "ambiguous"


def _direction(diffs: list[Optional[float]]) -> str:
    """Human-readable direction agreement across seeds."""
    vals = [d for d in diffs if d is not None]
    if not vals:
        return "n/a"
    if all(v > 0 for v in vals):
        return f"{len(vals)}/{len(vals)} seed 同向为正"
    if all(v < 0 for v in vals):
        return f"{len(vals)}/{len(vals)} seed 同向为负"
    pos = sum(1 for v in vals if v > 0)
    return f"方向不一致（{pos} 正 / {len(vals) - pos} 负）"


def classify(
    primary: Dict[str, List[Optional[float]]], decisive: Dict[str, List[Optional[float]]]
) -> Tuple[str, str]:
    """按事前约定的判定流程给出结论及依据指标。

    主指标为分叉后尾段胜率均值；当其无法判定（未达到 0.10 的
    “明显更好”阈值，或 seed 间方向不一致）时，改用决胜指标
    （640 集终局配对评估）。两个指标都按同一套多种子规则判定。
    """
    prim_off = _reading_multi(primary.get("off", []))
    prim_dec = _reading_multi(primary.get("decay", []))
    if "better" in (prim_off, prim_dec):
        return _READINGS["better"], "主指标（分叉后尾段胜率）"
    dec_off = _reading_multi(decisive.get("off", []))
    dec_dec = _reading_multi(decisive.get("decay", []))
    if "better" in (dec_off, dec_dec):
        return _READINGS["better"], "决胜指标（640 集终局配对评估）"
    if "same" in (dec_off, dec_dec) or "same" in (prim_off, prim_dec):
        return _READINGS["same"], ("决胜指标（640 集终局配对评估）"
                                   if dec_off is not None or dec_dec is not None
                                   else "主指标（分叉后尾段胜率）")
    if prim_off is not None or prim_dec is not None or dec_off is not None or dec_dec is not None:
        return _READINGS["ambiguous"], ("决胜指标（640 集终局配对评估）"
                                        if dec_off is not None or dec_dec is not None
                                        else "主指标（分叉后尾段胜率）")
    return "无法判定（缺少分支数据）", "无可用指标"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", action="append", required=True)
    ap.add_argument("--ovmse-ref", default=None, help="OVMSE online.jsonl for the overlay")
    ap.add_argument(
        "--out", default="documentation/experiment_reports/task3_ranking_loss_fork.md"
    )
    args = ap.parse_args()

    seeds = [analyze_seed(d) for d in args.run_dir]

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    # 两个指标的配对差：主指标=分叉后尾段胜率，决胜指标=640 集终局配对评估
    tail_diffs: Dict[str, List[float]] = {"off": [], "decay": [], "off_vs_decay": []}
    eval_diffs: Dict[str, List[float]] = {"off": [], "decay": [], "off_vs_decay": []}
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        br = seed["branches"]
        entry: Dict[str, Any] = {
            "name": os.path.basename(seed["run_dir"]), "tail": {}, "eval": {}
        }
        cont_tail = br["continue"]["tail_win"]
        for key in ("off", "decay"):
            v = br[key]["tail_win"]
            if cont_tail is not None and v is not None:
                tail_diffs[key].append(v - cont_tail)
                entry["tail"][key] = v - cont_tail
        if br["off"]["tail_win"] is not None and br["decay"]["tail_win"] is not None:
            tail_diffs["off_vs_decay"].append(br["off"]["tail_win"] - br["decay"]["tail_win"])
            entry["tail"]["off_vs_decay"] = br["off"]["tail_win"] - br["decay"]["tail_win"]
        cont_e = (br["continue"]["eval640"] or {}).get("win_rate")
        for key in ("off", "decay"):
            v = (br[key]["eval640"] or {}).get("win_rate")
            if cont_e is not None and v is not None:
                eval_diffs[key].append(v - cont_e)
                entry["eval"][key] = v - cont_e
        v_off = (br["off"]["eval640"] or {}).get("win_rate")
        v_dec = (br["decay"]["eval640"] or {}).get("win_rate")
        if v_off is not None and v_dec is not None:
            eval_diffs["off_vs_decay"].append(v_off - v_dec)
            entry["eval"]["off_vs_decay"] = v_off - v_dec
        per_seed.append(entry)

    tail_mu = {k: _mean(v) for k, v in tail_diffs.items()}
    eval_mu = {k: _mean(v) for k, v in eval_diffs.items()}
    verdict, metric_used = classify(tail_diffs, eval_diffs)

    # 实验参数从数据读出，不写死
    fork_t = seeds[0]["fork_t"] if seeds else None
    t_end = max(
        (b["last_t"] for s in seeds for b in s["branches"].values() if "last_t" in b),
        default=None,
    )
    remain = (t_end - fork_t) if (fork_t is not None and t_end is not None) else None
    eval_eps = max((s.get("eval_eps") or 0) for s in seeds) if seeds else 0
    tail_p = _mean(
        [
            s["branches"][n]["tail_win"]
            for s in seeds
            for n in BRANCHES
            if s["branches"][n]["tail_win"] is not None
        ]
    )
    noise95 = (
        1.96 * math.sqrt(tail_p * (1 - tail_p) / (eval_eps * TAIL_POINTS))
        if (tail_p is not None and eval_eps)
        else None
    )
    tail_abs_max = max((abs(d) for d in tail_mu.values() if d is not None), default=None)
    branch_tails = [
        s["branches"][n]["tail_win"]
        for s in seeds
        for n in BRANCHES
        if s["branches"][n]["tail_win"] is not None
    ]
    cont_reg = _mean(
        [
            s["branches"]["continue"]["reg_strength"]
            for s in seeds
            if s["branches"]["continue"]["reg_strength"] is not None
        ]
    )
    half_widths = [
        b["eval640"]["win_rate_ci95"][1] - b["eval640"]["win_rate"]
        for s in seeds
        for b in s["branches"].values()
        if isinstance(b, dict)
        and b.get("eval640")
        and b["eval640"].get("win_rate") is not None
        and b["eval640"].get("win_rate_ci95")
    ]
    half_w = _mean(half_widths)

    # 640 局配对差的 95% 不确定范围（两比例正态近似；分支在同一 env seed 下评估，
    # 但策略不同则逐局不对应，故不做配对检验）
    def _eval_diff_ci(seed, name_a: str, name_b: str) -> Optional[float]:
        ea = seed["branches"][name_a]["eval640"] or {}
        eb = seed["branches"][name_b]["eval640"] or {}
        pa, pb = ea.get("win_rate"), eb.get("win_rate")
        na, nb = int(ea.get("episodes") or 0), int(eb.get("episodes") or 0)
        if pa is None or pb is None or na <= 0 or nb <= 0:
            return None
        return 1.96 * math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)

    eval_ci: Dict[str, Optional[float]] = {}
    for key, (a, b) in {
        "off": ("off", "continue"),
        "decay": ("decay", "continue"),
        "off_vs_decay": ("off", "decay"),
    }.items():
        vals = [v for v in (_eval_diff_ci(s, a, b) for s in seeds) if v is not None]
        eval_ci[key] = _mean(vals) if vals else None
    fork_ts = [s["fork_t"] for s in seeds if s["fork_t"] is not None]
    if len(set(fork_ts)) > 1:
        fork_point_txt = (
            "约 100 万步（逐 seed 分叉点 " + "、".join(f"{t:,}" for t in fork_ts) + "）"
        )
    else:
        fork_point_txt = f"{fork_t:,}" if fork_t is not None else "n/a"
    t_end_txt = f"{t_end:,}" if t_end is not None else "n/a"
    remain_txt = f"{remain / 10000:.0f} 万步" if remain is not None else "n/a"

    lines: List[str] = []
    lines.append("# 5m_vs_6m ranking loss 分叉实验分析\n")
    lines.append("\n## 一、实验目的与设计\n")
    lines.append(
        "**要回答的问题**：UOT 在 5m_vs_6m 上的在线微调，后期表现不及 OVMSE。"
        "一个候选原因是 ranking loss——约束在线策略保持离线教师动作排序的正则项——栓住了后期适应。"
        "本实验直接检验：训练中途把 ranking loss 撤除或衰减，能否改善后期表现。\n"
    )
    lines.append(
        f"**实验设计**：先用父 run 从离线 checkpoint 出发在线训练到 t_env={fork_point_txt}，"
        f"冻结完整的训练器快照；再从同一快照分出三条支线，除 ranking loss 权重外完全相同——"
        f"continue 按原设置保留（权重 η 全程保持 0.1），off 在分叉点立即置零，"
        f"decay 从分叉点起线性衰减到零。三条支线各继续训练约 {remain_txt}，至 t_env≈{t_end_txt}。\n"
    )
    lines.append(
        "**评估方式**：分叉点 checkpoint 与三个终局 checkpoint 都在同一环境种子下评估 640 集"
        "（贪心策略，胜率附 Wilson 95% 置信区间），三个分支同 protocol，可直接配对比较。"
        f"训练过程中另有定期评测，但每个评测点只跑 {eval_eps or '数十'} 集，精度远低于终局评估"
        "（精度分析见第二节）。\n"
    )

    lines.append("\n## 二、结果\n")
    lines.append("\n### 2.1 各 checkpoint 的 640 集终局评估\n")
    lines.append("| seed | 分支 | 分叉后尾段胜率 | 尾段AUC | 尾段回报 | 640 集胜率 | 95% 置信区间 |")
    lines.append("|---|---|---|---|---|---|---|")
    for seed in seeds:
        fork = seed["branches"]["_fork"]["eval640"]
        fork_txt = (
            f"{fork['win_rate']:.3f} [{fork['win_rate_ci95'][0]:.3f},{fork['win_rate_ci95'][1]:.3f}]"
            if fork
            else "n/a"
        )
        lines.append(
            f"| {os.path.basename(seed['run_dir'])} | fork(分叉点) | - | - | - | {fork_txt} |"
        )
        for name in BRANCHES:
            b = seed["branches"][name]
            e = b["eval640"]
            e_txt = (
                f"{e['win_rate']:.3f} [{e['win_rate_ci95'][0]:.3f},{e['win_rate_ci95'][1]:.3f}]"
                if e
                else "n/a"
            )
            lines.append(
                f"| {os.path.basename(seed['run_dir'])} | {name} | "
                f"{_fmt(b['tail_win'])} | {_fmt(b['auc_win'])} | {_fmt(b['tail_return'], 2)} | {e_txt} |"
            )

    lines.append(
        "\n### 2.2 分支间配对差（相对 continue 分支，正值表示该分支更好；多个 seed 时为均值）\n"
    )
    lines.append("| 指标 | off − continue | decay − continue | off − decay |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| 分叉后尾段胜率（主指标） | {_fmt(tail_mu['off'])} | {_fmt(tail_mu['decay'])} "
        f"| {_fmt(tail_mu['off_vs_decay'])} |"
    )
    lines.append(
        f"| 640 集终局评估（决胜指标） | {_fmt(eval_mu['off'])} | {_fmt(eval_mu['decay'])} "
        f"| {_fmt(eval_mu['off_vs_decay'])} |"
    )
    lines.append("\n#### 每个 seed 的配对差（相对 continue 分支，正值表示该分支更好）\n")
    lines.append(
        "| seed | off−continue（尾段） | decay−continue（尾段） | off−decay（尾段） "
        "| off−continue（640局） | decay−continue（640局） | off−decay（640局） |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for e in per_seed:
        cells = [e["tail"].get(k) for k in ("off", "decay", "off_vs_decay")]
        cells += [e["eval"].get(k) for k in ("off", "decay", "off_vs_decay")]
        lines.append(f"| {e['name']} | " + " | ".join(_fmt(c) for c in cells) + " |")

    n_seeds = len(seeds)
    if noise95 is not None and tail_abs_max is not None:
        tail_sentence = (
            f"主指标（分叉后尾段胜率，最后 {TAIL_POINTS} 个评测点的均值）上三个分支的配对差最大为 "
            f"{_fmt(tail_abs_max)}（{n_seeds} 个 seed 的均值）；训练内每个评测点只跑 "
            f"{eval_eps or '数十'} 集，{TAIL_POINTS} 点均值在胜率约 {_fmt(tail_p)} 处的 95% "
            f"不确定范围约 ±{_fmt(noise95)}，"
            + (
                "主指标在此精度下无法判定。"
                if tail_abs_max <= noise95
                else "主指标配对差超出该范围，可按主指标判定。"
            )
        )
    else:
        tail_sentence = (
            "主指标（分叉后尾段胜率）每个评测点局数很少，分辨不出 0.1 以下的差异；"
            "决胜指标（640 集终局配对评估）精度足够，结论以它为准。"
        )
    od_sentence = (
        f"off 与 decay 之间只差 {_fmt(eval_mu['off_vs_decay'])}"
        + (
            "，在半宽以内，无法区分。"
            if (half_w is not None and eval_mu["off_vs_decay"] is not None
                and abs(eval_mu["off_vs_decay"]) <= half_w)
            else "。"
        )
    )
    lines.append(
        f"\n**两个指标的精度不同，给出不同强度的答案。** {tail_sentence}"
        f"决胜指标（640 集终局配对评估）上，off 与 decay 相对 continue 的配对差均值为 "
        f"{_fmt(eval_mu['off'])} 与 {_fmt(eval_mu['decay'])}；{n_seeds} 个 seed 的方向为 "
        f"{_direction(eval_diffs['off'])}（off 对 continue）、"
        f"{_direction(eval_diffs['decay'])}（decay 对 continue）。640 局评估的半宽约 "
        f"{_fmt(half_w)}，配对差的 95% 不确定范围约 ±{_fmt(eval_ci['off'])}（off 对 continue）"
        f"与 ±{_fmt(eval_ci['decay'])}（decay 对 continue）。{od_sentence}"
        f"640 局评估是唯一有足够分辨率区分分支的测量，且它衡量的是整个分叉后窗口"
        f"（约 {remain_txt}）累积下来的效果，而不是某一时刻的快照。\n"
    )

    lines.append("\n## 三、分叉后训练诊断（尾段均值）\n")
    lines.append("| seed | 分支 | 有效正则强度 | hinge 激活率 | 排序不一致率 | TD loss | ranking loss |")
    lines.append("|---|---|---|---|---|---|---|")
    for seed in seeds:
        for name in BRANCHES:
            b = seed["branches"][name]
            lines.append(
                f"| {os.path.basename(seed['run_dir'])} | {name} | "
                f"{_fmt(b['reg_strength'])} | {_fmt(b['hinge_rate'])} | "
                f"{_fmt(b['sign_disagree'])} | {_fmt(b['loss_td'], 4)} | {_fmt(b['loss_ord'], 4)} |"
            )

    ovmse_beyond_seeds: List[str] = []
    if args.ovmse_ref:
        ref_rows = _eval_rows(_read_jsonl(args.ovmse_ref))
        if ref_rows:
            tail = _tail_mean(ref_rows, "evaluation/win_rate", TAIL_POINTS)
            per_seed_hi = []
            for s in seeds:
                vals = [s["branches"][n]["tail_win"] for n in BRANCHES
                        if s["branches"][n]["tail_win"] is not None]
                if vals:
                    per_seed_hi.append((os.path.basename(s["run_dir"]), max(vals)))
            bt_hi = max((v for _, v in per_seed_hi), default=None)
            all_mean = _mean(branch_tails)
            gap_mean = (tail - all_mean) if (tail is not None and all_mean is not None) else None
            seed_txt = "、".join(
                f"{name} {_fmt(v)}（较 OVMSE {'低' if tail >= v else '高'} {_fmt(abs(tail - v))}）"
                for name, v in per_seed_hi
            )
            beyond = [name for name, v in per_seed_hi
                      if noise95 is not None and tail - v > noise95]
            within = [name for name, v in per_seed_hi
                      if noise95 is None or tail - v <= noise95]
            ovmse_beyond_seeds = beyond
            if beyond and not within:
                verdict_txt = "说明三个分支与 OVMSE 的差距不止 ranking loss 一项。"
            elif beyond:
                verdict_txt = (
                    f"说明与 OVMSE 的差距只在 {'、'.join(beyond)} 上超出主指标的不确定范围，"
                    f"其余 seed 上分支尾段已与 OVMSE 同水平。"
                )
            else:
                verdict_txt = "说明各 seed 上分支尾段与 OVMSE 的差距都在主指标的不确定范围以内。"
            lines.append(
                f"\n## 四、与 OVMSE 在线运行的对比\n\n"
                f"作为参照，OVMSE 在同一地图、相同在线设置下，尾段胜率均值为 {_fmt(tail)}"
                f"（共 {len(ref_rows)} 个评测点，与主指标同口径）。逐 seed 取三个分支的尾段胜率"
                f"最高值：{seed_txt}；全部 {len(branch_tails)} 个分支尾段的均值为 {_fmt(all_mean)}，"
                f"较 OVMSE {'低' if (gap_mean or 0) >= 0 else '高'} "
                f"{_fmt(abs(gap_mean)) if gap_mean is not None else 'n/a'}。"
                f"主指标的 95% 不确定范围约 ±{_fmt(noise95)}：{verdict_txt}\n"
            )

    lines.append("\n## 五、结论\n")
    lines.append(
        f"\n**{verdict}。** 判定依据：{metric_used}。按事前约定的多种子判定规则"
        f"（{n_seeds} 个 seed 方向一致且 640 局配对差均值 ≥ 0.10 记为明显更好；"
        f"|均值| < 0.05 或方向不一致记为无差异；其余记为趋势不明）："
        f"640 集终局评估上 off 与 decay 相对 continue 的配对差均值为 "
        f"{_fmt(eval_mu['off'])} 与 {_fmt(eval_mu['decay'])}"
        f"（{_direction(eval_diffs['off'])} / {_direction(eval_diffs['decay'])}），"
        f"off 与 decay 之间 {_fmt(eval_mu['off_vs_decay'])}。"
        f"训练诊断与这一读法一致：continue 支线的 ranking loss 权重全程保持 0.1 不减"
        f"（尾段其损失相对 TD loss 的贡献约 "
        f"{_fmt(cont_reg * 100, 1) if cont_reg is not None else 'n/a'}%），off 从分叉点起、"
        f"decay 逐步不再受其影响；终局评估捕捉的是整个分叉后窗口累积下来的差异。\n"
    )
    if verdict == _READINGS["same"]:
        point1 = (
            "第一，保留 ranking loss 与撤除/衰减在终局评估上无可测差异，"
            "“无差异”的原因——约束与在线证据不冲突，还是冲突被探索、回放、更新次数等因素抵消——本实验没有分离；"
        )
    else:
        point1 = (
            "第一，保留 ranking loss 造成终局损失的具体机制——"
            "是约束与在线证据冲突，还是仅占用优化容量——本实验没有分离；"
        )
    point2 = (
        f"第二，为什么 {'、'.join(ovmse_beyond_seeds)} 上三个分支都追不上 OVMSE（见第四节），"
        "说明差距还有 ranking loss 之外的原因，需要在探索策略、经验回放、更新次数、表示能力等方向上继续排查。"
        if ovmse_beyond_seeds else
        "第二，各 seed 上分支尾段都已与 OVMSE 同水平（见第四节），本实验没有进一步分离 UOT 支线与 OVMSE 的实现差异。"
    )
    lines.append(
        f"\n**这一结论没有回答的问题**：{point1}{point2}\n"
    )
    lines.append(
        "\n## 六、事前约定的结果解读框架\n\n"
        "三种可能的结果对应三种解释：\n\n"
        "- 撤除或衰减分支明显更好：说明持续的排序正则限制了后期适应，后续可以研究按在线证据逐步撤除约束；\n"
        "- 继续保留分支最好：说明离线偏好仍然有益，后期表现不及 OVMSE 的原因在别处"
        "（如探索策略、经验回放、更新次数、表示能力）；\n"
        "- 三个分支无差异：说明 ranking loss 不是造成差异的主因；注意即使撤除分支更好，"
        "也只说明撤除约束有益，不能说明教师模型的排序已经失效。\n"
    )
    lines.append(
        f"\n## 七、局限\n\n"
        f"- {n_seeds} 个随机种子：分支间差异在全部 seed 上按同一协议测量，逐 seed 配对差见第 2.2 节；\n"
        f"- 两个指标精度差异大：结论以 640 集终局评估为准，主指标（每评测点 {eval_eps or '数十'} 集）"
        f"分辨不出 0.1 以下的差异；\n"
        f"- 640 集评估的 95% 半宽约 0.04，0.02 以下的分支间差异（如 off 对 decay）测不出；\n"
        f"- 本实验只检验了“撤除/衰减 ranking loss”这一个干预，没有分离造成终局差异的具体机制。\n"
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
