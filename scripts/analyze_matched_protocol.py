#!/usr/bin/env python3
"""Summarise matched-protocol A-E arms: final 640-episode endpoint, factorial
contrasts, training curves, and mechanism checks.

Primary endpoint is the 640-episode evaluation of the final 3M checkpoint.
Periodic 640 evaluations and 32-episode training curves are descriptive only.
Arm pairing: all arms use the same eval seed and chunk size, so chunk-level
paired bootstrap is used when the episode checkpoints line up.

Usage:
  python scripts/analyze_matched_protocol.py \
    --root checkpoints/matched_protocol_20261020 --seeds 1 \
    --out logs/matched_protocol_summary.json \
    --md documentation/experiment_reports/task8_matched_protocol_tables.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np

ARMS = ("A", "B", "C", "D", "E")
ARM_SWITCHES = {
    "A": "UOT eta=0.1, OVM off, mixing 0",
    "B": "UOT eta=0 (vanilla), OVM off, mixing 0",
    "C": "OVM on, mixing 0.1",
    "D": "OVM on, mixing 0",
    "E": "OVM off, mixing 0.1",
}


def read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(max(p * (1.0 - p) / n + z * z / (4.0 * n * n), 0.0)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def newcombe_diff_ci(
    p_a: float, n_a: int, p_b: float, n_b: int, z: float = 1.96
) -> tuple[float, float]:
    """Newcombe hybrid-score CI for a difference of independent proportions."""
    la, ua = wilson_ci(p_a, n_a, z)
    lb, ub = wilson_ci(p_b, n_b, z)
    diff = p_a - p_b
    lower = diff - math.sqrt((p_a - la) ** 2 + (ub - p_b) ** 2)
    upper = diff + math.sqrt((ua - p_a) ** 2 + (p_b - lb) ** 2)
    return lower, upper


def bootstrap_mean_ci(
    diffs: np.ndarray, n_boot: int = 10000, seed: int = 12345
) -> tuple[float, float]:
    diffs = np.asarray(diffs, dtype=np.float64)
    if diffs.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    boot = diffs[idx].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_chunks(
    arms: Sequence[dict], field: str
) -> Optional[List[np.ndarray]]:
    """Return per-arm chunk value arrays if episode checkpoints are aligned."""
    arrays = []
    episodes = None
    for arm in arms:
        endpoint = arm.get("endpoint") or {}
        if not endpoint.get("complete"):
            return None
        chunks = endpoint.get("chunks") or []
        if not chunks:
            return None
        eps = [c.get("episodes_done") for c in chunks]
        if episodes is None:
            episodes = eps
        elif eps != episodes:
            return None
        vals = np.asarray([c.get(field, np.nan) for c in chunks], dtype=np.float64)
        if np.isnan(vals).any():
            return None
        arrays.append(vals)
    return arrays


def contrast(
    a: dict, b: dict, n_boot: int = 10000
) -> Optional[dict]:
    """Paired chunk bootstrap for A-B on win rate and mean return."""
    out: Dict[str, object] = {}
    for field, name in (("chunk_win_rate", "win_rate"), ("chunk_mean_return", "mean_return")):
        arrays = paired_chunks([a, b], field)
        if arrays is None:
            out[name] = {"paired": False}
            continue
        paired = arrays[0] - arrays[1]
        lo, hi = bootstrap_mean_ci(paired, n_boot=n_boot)
        out[name] = {
            "paired": True,
            "n_chunks": int(paired.size),
            "diff": float(paired.mean()),
            "ci95": [lo, hi],
        }
    return out


def interaction(
    c: dict, d: dict, e: dict, b: dict, field: str, n_boot: int = 10000
) -> Optional[dict]:
    arrays = paired_chunks([c, d, e, b], field)
    if arrays is None:
        return None
    c_arr, d_arr, e_arr, b_arr = arrays
    diff = c_arr - d_arr - e_arr + b_arr
    lo, hi = bootstrap_mean_ci(diff, n_boot=n_boot)
    return {
        "paired": True,
        "n_chunks": int(diff.size),
        "diff": float(diff.mean()),
        "ci95": [lo, hi],
    }


def parse_train_command(path: str) -> Dict[str, object]:
    info: Dict[str, object] = {}
    if not os.path.isfile(path):
        return info
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    m = re.search(r"--config\s+(\S+)", text)
    if m:
        info["config"] = m.group(1)
    for key, pattern in {
        "learner_type": r"learner_type=(\S+)",
        "transfer_mode": r"transfer\.mode=(\S+)",
        "mixing_ratio": r"system\.mixing_ratio=(\S+)",
        "uot_eta": r"system\.uot_eta=(\S+)",
        "checkpoint_path": r"checkpoint_path=(\S+)",
        "t_max": r"(?<!online\.)t_max=(\S+)",
    }.items():
        m = re.search(pattern, text)
        if m:
            info[key] = m.group(1)
    if "checkpoint_path" in info:
        info["offline_checkpoint"] = os.path.basename(str(info["checkpoint_path"]))
    return info


def load_online(path: str) -> dict:
    rows = read_jsonl(path)
    evals = [r for r in rows if "evaluation/win_rate" in r]
    if not evals:
        return {"path": path, "rows": 0, "evals": 0}
    first = evals[0]
    last = evals[-1]
    t_last = float(last.get("t_env", 0.0))
    curve = []
    bin_width = 250000.0
    max_bin = int(max(1, math.ceil((t_last + 1e-9) / bin_width)))
    for k in range(max_bin):
        lo, hi = k * bin_width, (k + 1) * bin_width
        sel = [
            r
            for r in evals
            if lo <= float(r.get("t_env", 0.0)) < hi
        ]
        if not sel:
            continue
        curve.append(
            {
                "bin_start": lo,
                "bin_end": hi,
                "n_evals": len(sel),
                "win_rate": float(
                    np.mean([r["evaluation/win_rate"] for r in sel])
                ),
                "mean_return": float(
                    np.mean([r["evaluation/mean_episode_return"] for r in sel])
                ),
            }
        )
    last10 = evals[-10:]
    diag_keys = [
        "evaluation/q_bias",
        "evaluation/greedy_action_disagree_rate",
        "evaluation/live_minus_offline_q",
        "uot_hinge_rate",
        "uot_sign_disagree",
        "loss_ord",
        "uot_cand_in_frac",
        "train_ovm/activation_rate_interval_mean",
        "train_ovm/lambda_memory_interval_mean",
    ]
    diag = {}
    for key in diag_keys:
        vals = [r[key] for r in last10 if key in r and r[key] is not None]
        if vals:
            diag[key] = float(np.mean(vals))
    best = max(evals, key=lambda r: float(r["evaluation/win_rate"]))
    return {
        "path": path,
        "rows": len(rows),
        "evals": len(evals),
        "t_env_first": float(first.get("t_env", 0.0)),
        "t_env_last": t_last,
        "baseline_win_rate": float(first["evaluation/win_rate"]),
        "baseline_return": float(first["evaluation/mean_episode_return"]),
        "last_win_rate": float(last["evaluation/win_rate"]),
        "last_return": float(last["evaluation/mean_episode_return"]),
        "last10_win_rate": float(
            np.mean([r["evaluation/win_rate"] for r in last10])
        ),
        "last10_return": float(
            np.mean([r["evaluation/mean_episode_return"] for r in last10])
        ),
        "best_eval_win_rate": float(best["evaluation/win_rate"]),
        "best_eval_t_env": float(best.get("t_env", 0.0)),
        "curve_250k": curve,
        "last10_diagnostics": diag,
    }


def load_640(path: str) -> Optional[dict]:
    rows = read_jsonl(path)
    if not rows:
        return None
    summary = next((r for r in rows if "checkpoint" in r), None)
    chunks = [r for r in rows if "chunk" in r]
    if summary is None:
        return {"path": path, "chunks": chunks, "complete": False}
    return {
        "path": path,
        "complete": True,
        "tag": summary.get("tag"),
        "checkpoint": summary.get("checkpoint"),
        "ckpt_t_env": summary.get("ckpt_t_env"),
        "episodes": summary.get("episodes"),
        "win_rate": summary.get("win_rate"),
        "win_rate_ci95": summary.get("win_rate_ci95"),
        "mean_return": summary.get("mean_return"),
        "chunks": chunks,
    }


def arm_paths(root: str, arm: str, seed: int) -> dict:
    d = os.path.join(root, f"{arm}_seed{seed}")
    return {
        "dir": d,
        "online": os.path.join(d, "online.jsonl"),
        "command": os.path.join(d, "train_command.txt"),
        "final_glob": glob.glob(os.path.join(d, f"*_online_*_seed{seed}.pt")),
        "eval_final": os.path.join(d, "eval640_final.jsonl"),
        "eval_periodic": sorted(
            glob.glob(os.path.join(d, "eval640_t*.jsonl")),
            key=lambda p: int(re.search(r"eval640_t(\d+)", p).group(1))
            if re.search(r"eval640_t(\d+)", p)
            else 0,
        ),
    }


def periodic_curve(paths: List[str]) -> List[dict]:
    out = []
    for path in paths:
        ev = load_640(path)
        if ev and ev.get("complete"):
            out.append(
                {
                    "file": os.path.basename(path),
                    "t_env": ev.get("ckpt_t_env"),
                    "win_rate": ev.get("win_rate"),
                    "mean_return": ev.get("mean_return"),
                }
            )
    return out


def fmt(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def ci_str(pair: Optional[Sequence[float]], digits: int = 3) -> str:
    if not pair:
        return "—"
    return f"[{fmt(pair[0], digits)}, {fmt(pair[1], digits)}]"


def build_markdown(summary: dict, n_boot: int) -> str:
    lines: List[str] = []
    lines.append("# Matched-protocol results (auto-generated)")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append(f"Root: `{summary['root']}`  Seeds: {summary['seeds']}")
    lines.append("")
    lines.append("## Arm settings and run status")
    lines.append("")
    lines.append("| arm/seed | switches | t_env last | final ckpt | 640 final | offline ckpt |")
    lines.append("|---|---|---:|---|---|---|")
    for key, arm in summary["arms"].items():
        paths = arm_paths(summary["root"], arm["arm"], arm["seed"])
        lines.append(
            "| {key} | {sw} | {t} | {ck} | {ev} | {off} |".format(
                key=key,
                sw=ARM_SWITCHES.get(arm["arm"], "—"),
                t=fmt(arm.get("online", {}).get("t_env_last"), 0),
                ck="yes" if paths["final_glob"] else "no",
                ev="yes" if (arm.get("endpoint") or {}).get("complete") else "no",
                off=arm.get("settings", {}).get("offline_checkpoint", "—"),
            )
        )
    lines.append("")
    lines.append("## Primary endpoint: 640-episode final-checkpoint evaluation")
    lines.append("")
    lines.append("| arm/seed | win rate | 95% CI | mean return | Δwin vs B | Δreturn vs B |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for key, arm in summary["arms"].items():
        ep = arm.get("endpoint") or {}
        if not ep.get("complete"):
            lines.append(f"| {key} | pending | — | — | — | — |")
            continue
        c = summary.get("contrasts_vs_B", {}).get(key, {})
        dw = c.get("win_rate", {})
        dr = c.get("mean_return", {})
        dw_s = (
            f"{dw.get('diff'):+.3f} {ci_str(dw.get('ci95'))}"
            if dw.get("paired")
            else "—"
        )
        dr_s = (
            f"{dr.get('diff'):+.2f} {ci_str(dr.get('ci95'), 2)}"
            if dr.get("paired")
            else "—"
        )
        lines.append(
            "| {key} | {w} | {ci} | {r} | {dw} | {dr} |".format(
                key=key,
                w=fmt(ep.get("win_rate")),
                ci=ci_str(ep.get("win_rate_ci95")),
                r=fmt(ep.get("mean_return"), 2),
                dw=dw_s,
                dr=dr_s,
            )
        )
    lines.append("")
    lines.append("## Factorial contrasts (paired chunk bootstrap)")
    lines.append("")
    lines.append("| contrast | Δwin (95% CI) | Δreturn (95% CI) | n chunks |")
    lines.append("|---|---:|---:|---:|")
    for name, item in summary.get("factorial", {}).items():
        if not item:
            lines.append(f"| {name} | pending | pending | — |")
            continue
        w = item.get("win_rate") or {}
        r = item.get("mean_return") or {}
        lines.append(
            "| {name} | {dw} | {dr} | {n} |".format(
                name=name,
                dw=(
                    f"{w.get('diff'):+.3f} {ci_str(w.get('ci95'))}"
                    if w.get("paired")
                    else "—"
                ),
                dr=(
                    f"{r.get('diff'):+.2f} {ci_str(r.get('ci95'), 2)}"
                    if r.get("paired")
                    else "—"
                ),
                n=w.get("n_chunks", "—") if w.get("paired") else "—",
            )
        )
    lines.append("")
    lines.append("## Mechanism checks (last 10 training evals)")
    lines.append("")
    lines.append("| arm/seed | q_bias | greedy disagree | UOT hinge | off-live Q | OVM activation | λ_memory |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key, arm in summary["arms"].items():
        d = (arm.get("online") or {}).get("last10_diagnostics", {})
        lines.append(
            "| {key} | {qb} | {gd} | {uh} | {lq} | {oa} | {lm} |".format(
                key=key,
                qb=fmt(d.get("evaluation/q_bias"), 2),
                gd=fmt(d.get("evaluation/greedy_action_disagree_rate")),
                uh=fmt(d.get("uot_hinge_rate")),
                lq=fmt(d.get("evaluation/live_minus_offline_q"), 2),
                oa=fmt(d.get("train_ovm/activation_rate_interval_mean")),
                lm=fmt(d.get("train_ovm/lambda_memory_interval_mean")),
            )
        )
    lines.append("")
    lines.append("## Descriptive: 32-episode training-eval curve, 250k bins")
    lines.append("")
    for key, arm in summary["arms"].items():
        curve = (arm.get("online") or {}).get("curve_250k", [])
        if not curve:
            continue
        lines.append(f"### {key}")
        lines.append("")
        lines.append("| t_env bin | n evals | win | return |")
        lines.append("|---|---:|---:|---:|")
        for row in curve:
            lines.append(
                f"| {int(row['bin_start'])//1000}k–{int(row['bin_end'])//1000}k "
                f"| {row['n_evals']} | {fmt(row['win_rate'])} | {fmt(row['mean_return'], 2)} |"
            )
        lines.append("")
    lines.append("## Descriptive: periodic 640-episode checkpoints")
    lines.append("")
    lines.append("| arm/seed | t_env | win | return |")
    lines.append("|---|---:|---:|---:|")
    for key, arm in summary["arms"].items():
        for row in arm.get("periodic_640", []):
            lines.append(
                f"| {key} | {fmt(row.get('t_env'), 0)} | {fmt(row.get('win_rate'))} | "
                f"{fmt(row.get('mean_return'), 2)} |"
            )
    lines.append("")
    lines.append(
        "Notes: primary endpoint is fixed to the final 3M checkpoint; periodic and "
        "32-episode curves are descriptive only. Δwin CIs are paired chunk "
        "bootstrap percentile intervals; unpaired Newcombe CIs are in the JSON."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="checkpoints/matched_protocol_20261020")
    parser.add_argument("--seeds", default="1", help="comma-separated seed list")
    parser.add_argument("--arms", default=" ".join(ARMS), help="space-separated arm list")
    parser.add_argument("--out", default="logs/matched_protocol_summary.json")
    parser.add_argument("--md", default="", help="optional markdown output path")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    arms = [a.strip() for a in str(args.arms).split() if a.strip()]
    summary: Dict[str, object] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root": args.root,
        "seeds": seeds,
        "arms": {},
        "contrasts_vs_B": {},
        "factorial": {},
    }
    arm_data: Dict[str, dict] = {}
    for seed in seeds:
        for arm in arms:
            key = f"{arm}_seed{seed}"
            paths = arm_paths(args.root, arm, seed)
            if not os.path.isdir(paths["dir"]):
                continue
            data = {
                "arm": arm,
                "seed": seed,
                "settings": parse_train_command(paths["command"]),
                "online": load_online(paths["online"]),
                "endpoint": load_640(paths["eval_final"]),
                "periodic_640": periodic_curve(paths["eval_periodic"]),
            }
            summary["arms"][key] = data
            arm_data[key] = data

    # Contrasts are computed within seed and pooled only for seed 1 display.
    for seed in seeds:
        b = arm_data.get(f"B_seed{seed}")
        for arm in arms:
            if arm == "B":
                continue
            key = f"{arm}_seed{seed}"
            a = arm_data.get(key)
            if not a or not b:
                continue
            c = contrast(a, b, n_boot=args.bootstrap)
            if c is None:
                continue
            summary["contrasts_vs_B"][key] = c
            if arm in ("A", "D", "E", "C"):
                summary["factorial"][f"{arm}−B (seed {seed})"] = c
        # Ranking vs the full OVMSE protocol.
        a_arm = arm_data.get(f"A_seed{seed}")
        c_arm = arm_data.get(f"C_seed{seed}")
        if a_arm and c_arm:
            ac = contrast(a_arm, c_arm, n_boot=args.bootstrap)
            if ac is not None:
                summary["factorial"][f"A−C (seed {seed})"] = ac
        # OVM × mixing interaction: C - D - E + B.
        d_arm = arm_data.get(f"D_seed{seed}")
        e_arm = arm_data.get(f"E_seed{seed}")
        if c_arm and d_arm and e_arm and b:
            item = {
                "win_rate": interaction(c_arm, d_arm, e_arm, b, "chunk_win_rate", args.bootstrap),
                "mean_return": interaction(c_arm, d_arm, e_arm, b, "chunk_mean_return", args.bootstrap),
            }
            summary["factorial"][f"C−D−E+B (seed {seed})"] = item

    # Add unpaired Newcombe CIs on the summary-level proportions.
    for key, c in summary["contrasts_vs_B"].items():
        seed = int(key.split("_seed")[1])
        b = arm_data.get(f"B_seed{seed}") or {}
        a = arm_data.get(key) or {}
        a_ep = a.get("endpoint") or {}
        b_ep = b.get("endpoint") or {}
        if a_ep.get("complete") and b_ep.get("complete"):
            lo, hi = newcombe_diff_ci(
                float(a_ep["win_rate"]),
                int(a_ep["episodes"]),
                float(b_ep["win_rate"]),
                int(b_ep["episodes"]),
            )
            c["win_rate"]["newcombe_ci95"] = [lo, hi]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[analyze] wrote {args.out}")
    if args.md:
        md = build_markdown(summary, args.bootstrap)
        os.makedirs(os.path.dirname(args.md) or ".", exist_ok=True)
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"[analyze] wrote {args.md}")


if __name__ == "__main__":
    main()
