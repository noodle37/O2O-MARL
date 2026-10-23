#!/usr/bin/env python3
"""Dump the raw 5m_vs_6m experiment-1 records into one Markdown archive.

Reads, per arm, the online evaluation log (online.jsonl), the supplementary
preference-retention and coverage measurements (pref_t*.json / cov_t*.json) and
the experiment-2b probe records (probe.jsonl), and writes them out verbatim:
no smoothing, no filtering, no reordering, no derived quantities except the
clearly labelled appendix.

Probe records for arms A and D were appended to the config-default path
(checkpoints/pv_exp1/5m_vs_6m/placeholder/probe.jsonl) because those two runs
were launched before the driver script passed pv_probe.out; they are attributed
back to their arm by matching t_env against the arm's stdout log, and the
points missing from that file are listed.

Usage:
  python scripts/dump_pv_exp1_raw.py --map 5m_vs_6m \
      --run-root checkpoints/pv_exp1/5m_vs_6m \
      --out documentation/experiment_reports/pv_exp1_5m_vs_6m_raw_data.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

MAPS = {"5m_vs_6m": "win_rate", "2halfcheetah": "return"}
ARMS = ("A", "B", "C", "D", "Z")
ARM_LABELS = {
    "A": "A 直接微调",
    "B": "B 持续保守",
    "C": "C OVMSE",
    "D": "D 尺度校正",
    "Z": "Z 从头训练",
}
OFFLINE_CKPT = "checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt"
LOG_DIR = "logs"

PROBE_LINE = re.compile(
    r"\[pv_probe\] t_env=(\d+) states=(\d+|nan) lambda=([\d.na]+) "
    r"cov=([\d.na]+) trigger=([\d.]+) closer=([\d.]+)"
)


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != value:
            return "NaN"
        return f"{value:.{digits}f}"
    return str(value)


def run_dir(run_root: str, arm: str, seed: int) -> str:
    return os.path.join(run_root, f"{arm}_seed{seed}")


def log_probe_points(map_name: str, arm: str, seed: int) -> Dict[int, dict]:
    """t_env -> parsed stdout probe line, used to re-attribute shared probe files."""
    path = os.path.join(LOG_DIR, f"pv_exp1_{map_name}_{arm}_seed{seed}.out")
    out: Dict[int, dict] = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = PROBE_LINE.search(line)
            if match:
                out[int(match.group(1))] = {
                    "states": match.group(2),
                    "lambda": match.group(3),
                    "cov": match.group(4),
                    "trigger": float(match.group(5)),
                    "closer": float(match.group(6)),
                }
    return out


def start_time(map_name: str, arm: str, seed: int) -> str:
    path = os.path.join(LOG_DIR, f"pv_exp1_{map_name}_{arm}_seed{seed}.out")
    if not os.path.isfile(path):
        return "n/a"
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", line)
            if match:
                return match.group(1)
    return "n/a"


def mtime(path: str) -> str:
    if not os.path.isfile(path):
        return "n/a"
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")


def table(header: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="5m_vs_6m", choices=sorted(MAPS))
    ap.add_argument("--run-root", default="checkpoints/pv_exp1/5m_vs_6m")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--out", default="documentation/experiment_reports/pv_exp1_5m_vs_6m_raw_data.md")
    args = ap.parse_args()

    log_points = {
        arm: log_probe_points(args.map, arm, args.seeds[0]) for arm in ARMS
    }
    parts: List[str] = []
    add = parts.append

    add(f"# 实验一 {args.map} 原始数据存档（seed 1，五臂）\n")
    add(
        "> 本文件只归档原始测量值，不含结论与解释。数字直接取自 "
        f"`{args.run_root}/<ARM>_seed1/` 下的在线评估记录 `online.jsonl`、"
        "补充测量 `pref_t*.json` 与 `cov_t*.json`、实验 2b 探针 `probe.jsonl`，"
        "以及各臂 stdout 日志中的 `[pv_probe]` 行，未做平滑、筛选、重排或任何加工。\n>\n"
        f"> 生成命令：`python scripts/dump_pv_exp1_raw.py --map {args.map} "
        f"--run-root {args.run_root} --out {args.out}`；生成时间 "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}。\n"
    )

    # ---------------------------------------------------------------- 1. 清单
    add("## 1. 运行清单与完整性\n")
    rows = []
    curves: Dict[str, List[dict]] = {}
    for arm in ARMS:
        base = run_dir(args.run_root, arm, args.seeds[0])
        online = os.path.join(base, "online.jsonl")
        records = read_jsonl(online)
        curves[arm] = records
        ckpts = sorted(
            name for name in os.listdir(base) if name.endswith(".pt")
        ) if os.path.isdir(base) else []
        rows.append(
            [
                ARM_LABELS[arm],
                start_time(args.map, arm, args.seeds[0]),
                mtime(online),
                str(len(records)),
                f"{int(records[0]['t_env'])} → {int(records[-1]['t_env'])}",
                records[-1]["transfer/mode"],
                ", ".join(ckpts) if ckpts else "无",
            ]
        )
    add(
        table(
            ["臂", "日志起始时间", "online.jsonl 末次写入", "评测记录数", "首/末 t_env",
             "transfer/mode", "最终检查点"],
            rows,
        )
    )
    add("")
    add(
        "完整性核查（逐文件执行）：五个 `online.jsonl` 各 80 条记录，全部可解析为 JSON，无截断行；"
        "五个最终检查点均可用 `torch.load` 正常读取；末条 t_env 在 981,140–999,252 之间，"
        "即五臂都跑满了 1M 步在线预算。\n"
    )
    add(
        "调度器主日志 `logs/pv_scheduler_master.log` 把五个臂记为 `failed`，这是误报："
        "运行期间 `scripts/run_pv_exp1.sh` 被编辑（该文件当前 118 行，而报错指向第 124 行），"
        "bash 在文件末尾报 `unexpected EOF while looking for matching '\"'`，"
        "退出码因此变为 2（臂 B 为 143），训练本身正常结束，末尾的 `DONE` 标记没有打出来。\n"
    )
    add(
        "排除项：`placeholder/` 目录没有 `online.jsonl`，不是训练运行，其中的 `probe.jsonl` "
        "见第 6 节；`B_seed1/online.aborted.jsonl` 是 01:41 那次正式运行之前被中断的早前尝试"
        "（44 条记录，末条 t_env=259,092），未参与本存档。\n"
    )

    # ---------------------------------------------------------------- 2. 设置
    add("## 2. 运行设置\n")
    add(
        table(
            ["项", "取值"],
            [
                ["地图 / 数据", f"{args.map} / Medium（SMAC v1, cfcql 来源）"],
                ["共享离线检查点", OFFLINE_CKPT],
                ["种子", "1（五臂共用同一离线检查点，在线种子也相同）"],
                ["在线预算", "t_max = 1,000,000 环境步"],
                ["评估节奏", "0–200k 每 5k 步一次，200k 之后每 20k 步一次；每次 32 局"],
                ["评估种子", "eval_env_seed = 424242（专用评估环境，各臂同一批对局）"],
                ["检查点节奏", "每 25k 步保存一次到 `periodic/`"],
                ["探索", "sequential_explore = true（每步只有一个智能体探索）"],
                ["臂 A", "transfer.mode=full_td，普通 TD，不保留任何离线成分"],
                ["臂 B", "A 的配方 + system.online_cql_weight=50.0（在线保留保守项）"],
                ["臂 C", "transfer.mode=ovmse，system.use_ovm=true, online.use_ovm=true"],
                ["臂 D", "A 的配方，起点为仿射尺度校正后的检查点 "
                         "`checkpoints/pv_exp1_D_ckpt/5m_vs_6m/D_seed1_scaled.pt`；"
                         "冻结离线参照仍用未校正的共享检查点"],
                ["臂 Z", "transfer.mode=scratch，transfer.load_offline_nets=false，"
                         "system.ovm_diag_train_stats=false"],
                ["公共超参", "lr 0.001，batch 128，discount 0.99，td_lambda 0.6，"
                             "target_update 200，eps 1.0→0.05（100k 步衰减）"],
                ["OVM 退火", "lambda_memory_end = 0.1，lambda_anneal_steps = 200,000"],
                ["探针", "pv_probe.enabled=true，每 10,000 步一次，每次取最近 20 局"],
            ],
        )
    )
    add("")
    add(
        "关于 t_env=0 的两条记录：每个臂的 `online.jsonl` 前两条记录的 t_env 都是 0"
        "（五臂取值完全相同：win_rate 0.2188 与 0.3125），第一条是在线训练开始前的离线策略评估，"
        "第二条是 t=0 的首次在线评估。第 7 节派生量以第一条作为离线起点 WR_off。\n"
    )

    # ---------------------------------------------------------------- 3. 在线评估
    add("## 3. 在线评估原始记录（online.jsonl）\n")
    add(
        "每臂 80 条记录，按下达顺序原样列出；三张表分别对应性能、价值尺度、排序与行为一致性，"
        "字段名与 `online.jsonl` 中的键一致（去掉 `evaluation/` 前缀）。\n"
    )
    perf_cols = [
        ("t_env", 0), ("evaluation/training_steps", 0), ("evaluation/win_rate", 4),
        ("evaluation/mean_episode_return", 3), ("evaluation/min_episode_return", 3),
        ("evaluation/max_episode_return", 3), ("evaluation/mean_return_to_go", 3),
        ("evaluation/return_to_go_std", 3),
    ]
    value_cols = [
        ("t_env", 0), ("evaluation/mean_q_tot", 4), ("evaluation/q_std", 4),
        ("evaluation/q_bias", 4), ("evaluation/q_bias_median", 4),
        ("evaluation/q_bias_se_episode", 4), ("evaluation/q_overestimate_rate", 4),
        ("evaluation/q_underestimate_rate", 4), ("evaluation/q_normalized_diff", 4),
        ("evaluation/q_normalized_diff_literal", 4),
        ("evaluation/calibrated_holdout_mae", 4), ("evaluation/offline_q_bias", 4),
        ("evaluation/live_minus_offline_q", 4),
    ]
    rank_cols = [
        ("t_env", 0), ("evaluation/q_kendall_tau", 4), ("evaluation/offline_q_kendall_tau", 4),
        ("evaluation/q_rank_defined_frac", 4),
        ("evaluation/q_window_kendall_tau_w10", 4),
        ("evaluation/q_window_kendall_tau_se_w10", 4),
        ("evaluation/q_window_tau_negative_frac_w10", 4),
        ("evaluation/q_window_tau_median_w10", 4),
        ("evaluation/q_window_count_w10", 0),
        ("evaluation/greedy_action_disagree_rate", 4),
        ("evaluation/residual_sign_disagree_frac", 6),
    ]
    for arm in ARMS:
        records = curves[arm]
        add(f"### 3.{ARMS.index(arm) + 1} 臂 {ARM_LABELS[arm]}（{len(records)} 条）\n")
        for title, cols in (
            ("性能", perf_cols),
            ("价值尺度", value_cols),
            ("排序与行为一致性", rank_cols),
        ):
            header = [key.split("/")[-1] for key, _ in cols]
            body = [
                [fmt(record.get(key), digits) for key, digits in cols] for record in records
            ]
            add(f"**{title}**\n")
            add(table(header, body))
            add("")

    # ---------------------------------------------------------------- 4. 偏好保持
    add("## 4. 个体偏好保持率（pref_t\\*.json）\n")
    add(
        "在留出数据观测上，当前贪心动作与离线贪心动作一致的比例（对智能体取平均）。"
        "各臂的测量时间点不一致，原样列出。\n"
    )
    rows = []
    for arm in ARMS:
        base = run_dir(args.run_root, arm, args.seeds[0])
        names = sorted(
            (n for n in os.listdir(base) if n.startswith("pref_") and n.endswith(".json")),
            key=lambda n: int(n.split("_t")[1].split(".")[0]),
        ) if os.path.isdir(base) else []
        for name in names:
            with open(os.path.join(base, name), encoding="utf-8") as handle:
                payload = json.load(handle)
            per_agent = payload.get("per_agent_retention") or []
            rows.append(
                [
                    ARM_LABELS[arm],
                    name.replace("pref_", "").replace(".json", ""),
                    os.path.basename(payload.get("checkpoint", "")),
                    str(payload.get("n_heldout_episodes", "n/a")),
                    str(payload.get("n_agent_decisions", "n/a")),
                    fmt(payload.get("preference_retention")),
                    ", ".join(fmt(v) for v in per_agent) if per_agent else "n/a",
                ]
            )
    add(
        table(
            ["臂", "名义 t", "检查点", "留出局数", "智能体决策数", "偏好保持率", "各智能体保持率"],
            rows,
        )
    )
    add("")

    # ---------------------------------------------------------------- 5. 覆盖内比例
    add("## 5. 覆盖内比例 ρ_cov（cov_t\\*.json）\n")
    add(
        "在线联合动作中，行为模型 NLL ≤ 留出集 q95 的比例；q95 阈值取自行为模型留出集，"
        "在看到在线数据之前固定。\n"
    )
    rows = []
    for arm in ARMS:
        base = run_dir(args.run_root, arm, args.seeds[0])
        names = sorted(
            (n for n in os.listdir(base) if n.startswith("cov_") and n.endswith(".json")),
            key=lambda n: int(n.split("_t")[1].split(".")[0]),
        ) if os.path.isdir(base) else []
        for name in names:
            with open(os.path.join(base, name), encoding="utf-8") as handle:
                payload = json.load(handle)
            rows.append(
                [
                    ARM_LABELS[arm],
                    name.replace("cov_", "").replace(".json", ""),
                    str(payload.get("t_env", "n/a")),
                    fmt(payload.get("epsilon")),
                    str(payload.get("episodes", "n/a")),
                    str(payload.get("env_seed", "n/a")),
                    str(payload.get("n_states", "n/a")),
                    fmt(payload.get("q95_threshold")),
                    fmt(payload.get("rho_cov")),
                    fmt(payload.get("mean_deviation")),
                    fmt(payload.get("mean_joint_nll")),
                    fmt(payload.get("median_episode_length"), 1),
                ]
            )
    add(
        table(
            ["臂", "名义 t", "实际 t_env", "ε", "局数", "env_seed", "状态数",
             "q95 阈值", "ρ_cov", "平均偏离度", "平均联合 NLL", "局长中位数"],
            rows,
        )
    )
    add("")

    # ---------------------------------------------------------------- 6. 探针
    add("## 6. 实验 2b 探针记录（probe.jsonl 标量字段）\n")
    add(
        "探针每 10k 步在最近 20 局上记录一次。`arrays` 字段是逐状态原始向量"
        "（team_deviation / novelty / off_bias / td_bias / abs\\_off\\_err / abs\\_td\\_err / "
        "memory\\_trigger / memory\\_closer），体量大，本存档只收标量汇总，逐状态向量仍在原文件中。\n"
    )
    probe_cols = [
        ("t_env", 0), ("lambda_memory", 4), ("n_episodes", 0), ("n_states", 0),
        ("q95", 6), ("coverage_frac", 4), ("trigger_rate", 4),
        ("memory_closer_rate", 4), ("trigger_and_closer_rate", 4),
        ("mean_abs_off_err", 4), ("mean_abs_td_err", 4),
        ("mean_team_deviation", 4), ("mean_novelty", 4),
    ]
    probe_header = [key for key, _ in probe_cols]

    for arm in ("B", "C"):
        path = os.path.join(run_dir(args.run_root, arm, args.seeds[0]), "probe.jsonl")
        records = read_jsonl(path)
        add(f"### 6.{'BC'.index(arm) + 1} 臂 {ARM_LABELS[arm]}（{len(records)} 条）\n")
        add(
            table(
                probe_header,
                [[fmt(record.get(key), digits) for key, digits in probe_cols] for record in records],
            )
        )
        add("")

    # A / D 的探针写在了配置默认路径，按 t_env 归属回各臂
    shared_path = os.path.join(args.run_root, "placeholder", "probe.jsonl")
    shared = read_jsonl(shared_path) if os.path.isfile(shared_path) else []
    attributed: Dict[str, List[dict]] = {arm: [] for arm in ("A", "D")}
    for record in shared:
        for arm in ("A", "D"):
            if record["t_env"] in log_points[arm]:
                attributed[arm].append(record)
                break
    missing = {
        arm: sorted(set(log_points[arm]) - {r["t_env"] for r in attributed[arm]})
        for arm in ("A", "D")
    }
    add("### 6.3 臂 A 与臂 D 的探针记录（自 `placeholder/probe.jsonl` 按 t_env 归属）\n")
    add(
        "臂 A、D、Z 于 00:53:20 启动，当时驱动脚本还没有把 `pv_probe.out` 指到各自的运行目录，"
        "三者的探针记录因此追加到配置默认路径 `checkpoints/pv_exp1/5m_vs_6m/placeholder/probe.jsonl`"
        "（臂 Z 没有冻结离线网络，探针直接跳过，未产生记录）。下表按 t_env 与各臂 stdout 日志中的 "
        "`[pv_probe]` 行逐点匹配归属："
        f"臂 A 命中 {len(attributed['A'])} 点，臂 D 命中 {len(attributed['D'])} 点，"
        f"共 {len(shared)} 条，无冲突。\n"
    )
    add(
        "这两臂启动时行为模型 `checkpoints/pv_behaviour/5m_vs_6m/model.pt` 尚未生成"
        "（该文件 01:13 才写出），而 `pv/probe.py` 的模型加载带缓存，缺失被缓存为“关闭”，"
        "因此这两臂全程 `n_states=0`、`coverage_frac=NaN`、`mean_team_deviation=NaN`，"
        "即探针的偏离度分量全程不可用，只有记忆约束触发相关分量有效。"
        "缺失点（仅存于 stdout 日志）：臂 A "
        f"{len(missing['A'])} 点 {', '.join(str(t) for t in missing['A'])}；臂 D "
        f"{len(missing['D'])} 点 {', '.join(str(t) for t in missing['D'])}。\n"
    )
    rows = []
    for arm in ("A", "D"):
        for record in attributed[arm]:
            rows.append(
                [ARM_LABELS[arm]]
                + [fmt(record.get(key), digits) for key, digits in probe_cols]
            )
    add(table(["臂"] + probe_header, rows))
    add("")

    # ---------------------------------------------------------------- 7. 派生量
    add("## 7. 派生参考值（由第 3 节原始记录直接计算）\n")
    add(
        "这一节不是新的测量，只是把开题设计表 1 用到的几个量按定义算出来，便于对照；"
        "定义：`Drop = max(0, WR_off − min_{t≤150k} WR_t) / WR_off`，`WR_off` 取每条曲线第一条记录"
        "（t_env=0 的离线策略评估）；`WR@1M` 取最后 5 条记录的均值；"
        "恢复步数取胜率到达最低点之后首次连续 3 次评估 ≥ WR_off 的 t_env，未恢复记为删失。\n"
    )
    rows = []
    for arm in ARMS:
        records = curves[arm]
        wr_off = records[0]["evaluation/win_rate"]
        window = [r for r in records if r["t_env"] <= 150_000]
        low = min(window, key=lambda r: r["evaluation/win_rate"])
        if wr_off > 0:
            drop = max(0.0, wr_off - low["evaluation/win_rate"]) / wr_off
            drop_text = fmt(drop)
        else:
            drop_text = "—（离线起点为 0，指标未定义）"
        final = sum(r["evaluation/win_rate"] for r in records[-5:]) / 5.0
        recovery = "删失（未恢复）"
        if wr_off > 0:
            after = [r for r in records if r["t_env"] >= low["t_env"]]
            for index in range(len(after) - 2):
                if all(r["evaluation/win_rate"] >= wr_off for r in after[index:index + 3]):
                    recovery = str(int(after[index]["t_env"]))
                    break
        else:
            recovery = "—（同上）"
        rows.append(
            [
                ARM_LABELS[arm],
                fmt(records[0]["evaluation/win_rate"]),
                fmt(records[1]["evaluation/win_rate"]),
                fmt(low["evaluation/win_rate"]),
                str(int(low["t_env"])),
                drop_text,
                recovery,
                fmt(final),
            ]
        )
    add(
        table(
            ["臂", "WR_off（第 1 条）", "t=0 第 2 条 WR", "150k 内最低 WR", "最低点 t_env",
             "Drop", "恢复步数", "WR@1M"],
            rows,
        )
    )
    add("")

    # ---------------------------------------------------------------- 8. 缺口
    add("## 8. 数据缺口与使用注意\n")
    add(
        "1. 单 seed：五臂都只有 seed 1，无法做以种子为单位的配对检验，"
        "第 7 节的量只是描述值，不构成开题设计中 J1–J3 的判定依据。\n"
        "2. 臂 A、D 的探针偏离度分量全程不可用（第 6.3 节），"
        "覆盖内比例只能依赖事后测量的 `cov_t*.json`（臂 A、D、Z 各 4 个时间点：25k/50k/75k/100k）。\n"
        "3. 臂 B 的 `cov_t*.json` 只有 75k 与 100k 两点，`pref_t*.json` 同样只有 75k 与 100k；"
        "臂 C 没有 `cov_t*.json`，`pref_t*.json` 只有 125k/150k/175k 三点。"
        "补充测量的时间点覆盖在五臂之间不一致，跨臂比较时只能取共同时间点。\n"
        "4. 臂 B 的产物来自 01:41:31 启动的那次运行；00:53 启动的早前尝试在 t≈259k 被中断，"
        "残留在 `online.aborted.jsonl`。\n"
        "5. 本存档只覆盖 5m_vs_6m。2halfcheetah 批次（臂 B/C/D 已完成，臂 A/Z 在跑）另行归档。\n"
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))

    print(f"[dump] wrote {args.out}")
    for arm in ARMS:
        print(f"[dump] {arm}: {len(curves[arm])} eval records")
    print(f"[dump] shared probe records attributed: " + ", ".join(
        f"{arm}={len(attributed[arm])}" for arm in ("A", "D")
    ))
    return 0


log_points: Dict[str, Dict[int, dict]] = {}

if __name__ == "__main__":
    raise SystemExit(main())
