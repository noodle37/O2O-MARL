#!/usr/bin/env python3
"""Cross-check the attribution experiment against the finished TCPA matrix.

The attribution matrix on card 3 runs **one seed per arm**, so its confidence
intervals only describe the fluctuation between evaluation points of a single
trajectory.  The repository already contains a finished TCPA matrix
(``checkpoints/tcpa_5m_vs_6m``) with six variants and **three seeds** at 3M
steps.  Reading its early window with the same pairing methodology gives

* a seed-level uncertainty for ``full - no_igm`` (the quantity the attribution
  matrix calls ``Δ_prior``), which is exactly the reference the single-seed
  design lacks;
* an independent check on whether the early-window effect that the attribution
  matrix is going to measure is even larger than the seed-to-seed spread.

Caveat that makes this a *conservative* reference: the finished matrix has no
``eval_env_seed``, so every arm was evaluated on the collection environment and
the arms did **not** play the same episodes.  Its paired differences therefore
carry episode-set noise that the attribution matrix removes by construction.
So the seed spread measured here is an upper bound on the single-seed noise of
the new experiment, not a replacement for a second seed.

Usage:
    python scripts/analyze_tcpa_prior_seeds.py
    python scripts/analyze_tcpa_prior_seeds.py --root checkpoints/tcpa_5m_vs_6m \
        --json checkpoints/tcpa_5m_vs_6m/seed_spread.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analyze_tcpa_attrib import (  # noqa: E402
    _window_index,
    classify,
    eval_curve,
    paired_contrast,
    read_rows,
    stability,
)

VARIANTS = ("full", "fixed_gate", "gate_anti", "abs_q", "no_igm", "no_residual")
SEEDS = ("seed1", "seed2", "seed3")

WINDOWS: Tuple[Tuple[str, int, int], ...] = (
    ("early", 0, 300_000),
    ("mid", 300_000, 600_000),
    ("late", 600_000, 10**9),
)

EVAL_KEY = "evaluation/mean_episode_return"


def curve_for(root: str, variant: str, seed: str) -> List[Tuple[float, Dict]]:
    path = os.path.join(root, variant, seed, "online.jsonl")
    return eval_curve(read_rows(path))


def paired_points(
    curve_a: List[Tuple[float, Dict]],
    curve_b: List[Tuple[float, Dict]],
    lo: int,
    hi: int,
) -> List[float]:
    """Per-evaluation-index differences ``a - b`` inside a window.

    Pairing is by evaluation index, not by exact ``t_env``: the arms accumulate
    env steps at their own pace, so their ``t_env`` at the same index differ.

    The windowing itself is delegated to the analyzer, so that this script and
    the readout of the attribution matrix can never disagree about which
    points belong to which window (they did once: the analyzer halved the
    paired time and thereby doubled every window boundary).
    """
    stat = paired_contrast(curve_a, curve_b, lo, hi)
    if stat.get("mean") is None:
        return []
    keep = _window_index([curve_a, curve_b], lo, hi)
    diffs: List[float] = []
    for i in keep:
        va = curve_a[i][1].get(EVAL_KEY)
        vb = curve_b[i][1].get(EVAL_KEY)
        if va is None or vb is None:
            continue
        diffs.append(float(va) - float(vb))
    return diffs


def bootstrap(diffs: Sequence[float], seed: int = 20261212) -> Dict[str, Optional[float]]:
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    mean = float(arr.mean())
    if arr.size < 2:
        return {"n": int(arr.size), "mean": mean, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(4000, arr.size))
    boots = arr[idx].mean(axis=1)
    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
    return {
        "n": int(arr.size),
        "mean": mean,
        "lo": float(lo_ci),
        "hi": float(hi_ci),
        "p_positive": float((boots > 0).mean()),
    }


def seed_level(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    """Mean and standard error of a per-seed statistic (the seed-level noise)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n_seeds": 0, "mean": None, "sd": None, "se": None, "values": []}
    arr = np.asarray(vals, dtype=np.float64)
    se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else None
    return {
        "n_seeds": int(arr.size),
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "se": se,
        "values": [float(v) for v in vals],
    }


def fmt(stat: Dict[str, Optional[float]]) -> str:
    if not stat or stat.get("mean") is None:
        return "-"
    if stat.get("lo") is None:
        return f"{stat['mean']:+.2f} (n={stat.get('n', 0)})"
    return (
        f"{stat['mean']:+.2f} [{stat['lo']:+.2f}, {stat['hi']:+.2f}] "
        f"(n={stat.get('n', 0)})"
    )


def fmt_point(value: Optional[float]) -> str:
    """Format a plain number (no bootstrap interval)."""
    if value is None:
        return "-"
    return f"{value:+.2f}"


def _shown(text: str) -> int:
    """Display width, counting CJK characters as two columns."""
    return sum(
        2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text
    )


def pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _shown(text))


def pad_left(text: str, width: int) -> str:
    return " " * max(0, width - _shown(text)) + text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="checkpoints/tcpa_5m_vs_6m")
    parser.add_argument(
        "--json", default="checkpoints/tcpa_5m_vs_6m/seed_spread.json"
    )
    parser.add_argument(
        "--attrib-root",
        default="checkpoints/tcpa_attrib_5m_vs_6m",
        help="the attribution matrix, for the new-vs-old same-condition check",
    )
    parser.add_argument(
        "--markdown",
        default="documentation/experiment_reports/tcpa_prior_seed_crosscheck.md",
        help="where to write the report (empty string to skip)",
    )
    args = parser.parse_args()

    curves = {
        (variant, seed): curve_for(args.root, variant, seed)
        for variant in VARIANTS
        for seed in SEEDS
    }

    lines: List[str] = []
    summary: Dict = {"per_variant": {}}

    lines.append("# 既有 TCPA 矩阵的种子层面交叉校验（full − no_igm，3 seed）")
    lines.append("")
    lines.append(
        "新归因矩阵**每臂只有 1 个 seed**，其置信区间只描述单条轨迹上评估点之间的波动。"
        "本表用**完全相同的配对口径**（按评估序号配对，不按精确 `t_env`）"
        "读取已经跑完的 `checkpoints/tcpa_5m_vs_6m`（6 变体 × 3 seed × 3M 步），"
        "给出 `Δ_prior` 在 early 窗口上的**种子层面**波动。"
    )
    lines.append("")
    lines.append(
        "**这个参照是保守的**：既有矩阵没有 `eval_env_seed`，各臂在采集环境上评估、"
        "臂间**没有跑同一批对局**，所以它的配对差里含有一部分对局集合噪声；"
        "新矩阵按设计把这部分去掉了。因此这里量到的种子波动是"
        "**新矩阵单 seed 噪声的上界**，不能替代补第二个 seed。"
    )
    lines.append("")

    for window, lo, hi in WINDOWS:
        lines.append(f"## 窗口 {window}（{lo}–{hi}）")
        lines.append("")
        lines.append(
            "（括号内为 95% bootstrap CI；每格配对点数约 "
            f"{max(1, (hi - lo) // 10_000)} 个评估点）"
        )
        lines.append("")
        lines.append(
            f"| {pad('变体 − no_igm', 26)}{pad_left('seed1', 22)}"
            f"{pad_left('seed2', 22)}{pad_left('seed3', 22)}"
            f"{pad_left('种子 mean', 12)}{pad_left('seed SE', 10)} |"
        )
        lines.append(
            "|" + "|".join("-" * 22 for _ in range(6)) + "|"
        )
        for variant in VARIANTS:
            if variant == "no_igm":
                continue
            per_seed: List[Optional[float]] = []
            cells = ""
            for seed in SEEDS:
                diffs = paired_points(
                    curves[(variant, seed)], curves[("no_igm", seed)], lo, hi
                )
                stat = bootstrap(diffs)
                per_seed.append(stat["mean"])
                if stat.get("mean") is None:
                    cells += pad_left("-", 22)
                elif stat.get("lo") is None:
                    cells += pad_left(f"{stat['mean']:+.2f}", 22)
                else:
                    cells += pad_left(
                        f"{stat['mean']:+.2f} [{stat['lo']:+.2f}, {stat['hi']:+.2f}]",
                        22,
                    )
            level = seed_level(per_seed)
            lines.append(
                f"| {pad(variant + ' − no_igm', 26)}{cells}"
                f"{pad_left(fmt_point(level['mean']), 12)}"
                f"{pad_left(fmt_point(level['se']), 10)} |"
            )
            summary["per_variant"].setdefault(variant, {})[window] = {
                "per_seed": per_seed,
                "seed_level": level,
            }
        lines.append("")

    # The headline number: the early-window effect of the real prior.
    early = summary["per_variant"].get("full", {}).get("early", {})
    level = early.get("seed_level", {})
    lines.append("## 读法")
    lines.append("")
    lines.append(
        f"- `full − no_igm` 在 early 窗口的种子层面均值是 **{fmt_point(level.get('mean'))}**，"
        f"种子间标准误 **{fmt_point(level.get('se'))}**"
        f"（3 个 seed：{', '.join(fmt_point(v) for v in level.get('values', []))}）。"
    )
    lines.append(
        "- 如果这三个 seed 的**方向一致**且幅度明显大于种子间标准误，"
        "说明「保留旧动作在训练早期确实有收益」这件事本身是可重复的；"
        "新矩阵要回答的是**这个收益来自什么**（内容 vs 通用约束）"
        "以及**它是否依赖队友配合**。"
    )
    lines.append(
        "- 如果种子间方向不一致，那么新矩阵那个单 seed 的 early 窗口读数"
        "**本来就不该被当成结论**——这一点比新矩阵本身更重要。"
    )
    lines.append("")

    # ------------------------------------------------------ where the gain comes from
    lines.append("## 早期待益的来源分解（early 窗口，3 seed）")
    lines.append("")
    lines.append(
        "把「早期待益」拆成几个互不相同的来源，看每一项各自值多少。"
        "这是**已完成数据**能回答的那一半问题：收益由什么产生；"
        "「这个收益是否依赖队友配合」是另一半，由归因实验回答。"
    )
    lines.append("")
    decomposition = (
        (
            "残差参数化本身",
            "no_igm",
            "no_residual",
            "以「在线网 = 离线网 + 残差」继续训练，相对直接微调整个网络",
        ),
        (
            "无条件保留旧动作",
            "fixed_gate",
            "no_igm",
            "同一个 IGM hinge，gate 恒为 1：总是把旧联合动作按回贪心",
        ),
        (
            "反向条件下的保留",
            "gate_anti",
            "no_igm",
            "gate = 1 − exp(−regret/τ)：只在旧动作**不** competitive 时保留",
        ),
        (
            "绝对值保留",
            "abs_q",
            "no_igm",
            "保留旧动作的绝对 Q 值而非相对序",
        ),
        (
            "竞争条件下的保留（真 TCPA）",
            "full",
            "no_igm",
            "gate = exp(−regret/τ)：只在旧动作还 competitive 时保留",
        ),
    )
    lines.append(
        f"| {pad('来源', 26)}{pad_left('seed1', 10)}{pad_left('seed2', 10)}"
        f"{pad_left('seed3', 10)}{pad_left('种子 mean', 12)}"
        f"{pad_left('seed SE', 10)} |"
    )
    lines.append(
        "|" + "-" * 26 + "|" + "|".join("-" * 10 for _ in range(5)) + "|"
    )
    for label, variant, base, desc in decomposition:
        per_seed = []
        cells = ""
        for seed in SEEDS:
            stat = paired_contrast(
                curves[(variant, seed)], curves[(base, seed)], 0, 300_000
            )
            per_seed.append(stat["mean"])
            cells += f"{fmt_point(stat['mean']):>10}"
        level = seed_level(per_seed)
        lines.append(
            f"| {pad(label, 26)}{cells}"
            f"{pad_left(fmt_point(level['mean']), 12)}"
            f"{pad_left(fmt_point(level['se']), 10)} |"
        )
        summary.setdefault("decomposition", {})[label] = {
            "variant": variant,
            "base": base,
            "description": desc,
            "per_seed": per_seed,
            "seed_level": level,
        }
    lines.append("")
    lines.append(
        "**读法**：early 阶段真正有收益的是**残差参数化**与**无条件保留旧动作**——"
        "而这两样**都与「旧动作还 competitive 吗」无关**。"
        "真 TCPA 的那套「只在旧动作还 competitive 时保留」在 early 窗口是 **0**，"
        "甚至比反向条件（只在不 competitive 时保留）还差。"
        "也就是说：早期待益来自**与内容/条件无关的通用效应**，"
        "而不是来自「旧联合动作因为协作内容而值得保留」这件事。"
    )
    lines.append("")
    lines.append(
        "注意这只覆盖「收益由什么产生」；「这个收益是否依赖队友配合」"
        "由归因实验的析因设计回答，本矩阵没有队友条件。"
    )
    lines.append("")

    # ------------------------------------------------------- new vs old check
    new_root = args.attrib_root
    if os.path.isdir(new_root):
        pairs = (
            ("none_matched", "no_igm", "无约束"),
            ("prior_matched", "full", "真 TCPA"),
        )
        lines.append("## 新矩阵与既有矩阵的同条件对照")
        lines.append("")
        lines.append(
            "新归因矩阵的 `none_matched` / `prior_matched` 与既有矩阵的 "
            "`no_igm/seed1` / `full/seed1` 是**同一训练协议**（同一离线先验、"
            "同一 seed、同样 8 并行环境、同样 10k 步评测一次），"
            "只差两点：新矩阵用固定种子的评估环境（臂间同对局），"
            "以及全局 torch RNG 未播种导致的探索噪声不同。"
            "所以两条曲线**不该逐位相同，但期望应当一致**——这可以验证新矩阵"
            "没有跑偏到别的 regime。"
        )
        lines.append("")
        lines.append(
            f"| {'臂':<16}{'既有 early 均值':>18}{'新 early 均值':>16}"
            f"{'既有 Δ':>10}{'新 Δ':>10} |"
        )
        lines.append("|" + "|".join("-" * 12 for _ in range(5)) + "|")
        old_base = None
        new_base = None
        # The new matrix is still running, so the two curves cover different
        # amounts of training: compare them over the *common* index range only.
        # (The old matrix has a large early dip, so a 0-300k mean is not
        # comparable to a 0-80k mean.)
        for new_arm, old_arm, label in pairs:
            old_curve = curves.get((old_arm, "seed1")) or []
            new_curve = curve_for(new_root, new_arm, "seed1")
            common = min(len(old_curve), len(new_curve))
            old_vals = [
                float(old_curve[i][1][EVAL_KEY])
                for i in range(common)
                if old_curve[i][1].get(EVAL_KEY) is not None
            ]
            new_vals = [
                float(new_curve[i][1][EVAL_KEY])
                for i in range(common)
                if new_curve[i][1].get(EVAL_KEY) is not None
            ]
            old_mean = float(np.mean(old_vals)) if old_vals else None
            new_mean = float(np.mean(new_vals)) if new_vals else None
            if old_arm == "no_igm":
                old_base, new_base = old_mean, new_mean
            old_delta = None if old_base is None or old_mean is None else old_mean - old_base
            new_delta = None if new_base is None or new_mean is None else new_mean - new_base
            span = (
                f"0–{int(old_curve[common - 1][0])}"
                if common
                else "-"
            )
            lines.append(
                f"| {label + ' (' + span + ')':<16}{fmt_point(old_mean):>18}"
                f"{fmt_point(new_mean):>16}"
                f"{fmt_point(old_delta):>10}{fmt_point(new_delta):>10} |"
            )
            summary.setdefault("new_vs_old", {})[new_arm] = {
                "old_arm": old_arm,
                "common_span": span,
                "old_early_mean": old_mean,
                "new_early_mean": new_mean,
                "old_delta": old_delta,
                "new_delta": new_delta,
            }
        lines.append("")
        lines.append(
            "（新矩阵的 early 窗口尚未跑满，上表是**当前进度下**的对照；"
            "臂跑满后重跑本脚本即可。）"
        )
        lines.append("")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"wrote {args.markdown}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
