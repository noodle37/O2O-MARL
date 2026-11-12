#!/usr/bin/env python3
"""Scan every experiment line under checkpoints/ and report, per run:

  * how many evaluation points were written,
  * the last ``t_env`` reached,
  * whether the run looks finished (final checkpoint present / budget reached),
  * whether the run is still live (a training process owns it).

The point is a single, mechanical answer to "did the main algorithm and the
ablation finish, and with how many seeds" that does not depend on reading
launcher logs by hand.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

ROOT = "checkpoints"


def last_t_env(path: str) -> Tuple[int, float]:
    """(number of evaluation points, largest t_env seen)."""
    log = os.path.join(path, "online.jsonl")
    n = 0
    last = 0.0
    if not os.path.isfile(log):
        return 0, 0.0
    with open(log, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "evaluation/mean_episode_return" not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            last = max(last, float(row.get("t_env", 0.0)))
    return n, last


def has_final_ckpt(path: str) -> bool:
    """True when the directory holds a final (non-step) checkpoint."""
    if not os.path.isdir(path):
        return False
    for name in os.listdir(path):
        if not name.endswith(".pt"):
            continue
        if "step" in name or "checkpoint" in name:
            continue
        return True
    return False


def live_runs() -> List[str]:
    """Command lines of all live training processes (any user)."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,args"], capture_output=True, text=True
        ).stdout
    except OSError:
        return []
    keep = []
    for line in out.splitlines():
        if "train.py" not in line:
            continue
        if "grep" in line:
            continue
        keep.append(line.strip())
    return keep


def scan_run(path: str) -> Dict[str, object]:
    n, last = last_t_env(path)
    return {
        "path": path,
        "n_eval": n,
        "last_t_env": last,
        "final_ckpt": has_final_ckpt(path),
        "exists": os.path.isdir(path),
    }


def walk_runs(base: str) -> List[str]:
    """All leaf directories that directly contain an online.jsonl."""
    runs: List[str] = []
    if not os.path.isdir(base):
        return runs
    for dirpath, dirnames, filenames in os.walk(base):
        if "online.jsonl" in filenames:
            runs.append(dirpath)
            dirnames[:] = []  # do not descend into a run
    return sorted(runs)


def summarise(line_name: str, base: str, budget: Optional[float], live_cmds: List[str]) -> None:
    runs = walk_runs(base)
    print(f"\n{'=' * 100}\n### {line_name}   (budget={budget})\n{'=' * 100}")
    if not runs:
        print("  (no runs with online.jsonl)")
        return
    done = 0
    for run in runs:
        info = scan_run(run)
        last = float(info["last_t_env"])
        rel = os.path.relpath(run, ROOT)
        live = any(rel in cmd for cmd in live_cmds)
        reached = budget is not None and last >= budget * 0.98
        finished = bool(info["final_ckpt"]) or reached
        if finished and not live:
            done += 1
        state = "LIVE" if live else ("DONE" if finished else "PART")
        pct = f"{100 * last / budget:5.1f}%" if budget else "   - "
        print(
            f"  {rel:<62} n_eval={info['n_eval']:>4} last_t={last:>10,.0f} "
            f"{pct} {state:<5} final_ckpt={'Y' if info['final_ckpt'] else 'n'}"
        )
    print(f"  --> {done}/{len(runs)} runs finished (live ones excluded)")


def main() -> None:
    live_cmds = live_runs()
    lines: List[Tuple[str, str, Optional[float]]] = [
        ("主算法 ACKF (5m_vs_6m)", "ackf_5m_vs_6m", None),
        ("消融 ACKF (5m_vs_6m)", "ackf_ablation_5m_vs_6m", None),
        ("主算法 SO2", "so2_5m_vs_6m", None),
        ("主算法 SO2 (2s3z)", "so2_2s3z", None),
        ("TCPA 主矩阵 (5m_vs_6m, 6变体×3seed)", "tcpa_5m_vs_6m", 3_000_000),
        ("TCPA (2s3z)", "tcpa_2s3z", 3_000_000),
        ("TCPA 归因矩阵 (运行中, 10臂×1seed)", "tcpa_attrib_5m_vs_6m", 1_000_000),
        ("ROAD", "road_5m_vs_6m", None),
        ("MU-APR", "muapr_5m_vs_6m", None),
        ("SAPR", "sapr_5m_vs_6m", None),
        ("PEXM", "pexm_5m_vs_6m", None),
    ]
    for name, sub, budget in lines:
        summarise(name, os.path.join(ROOT, sub), budget, live_cmds)


if __name__ == "__main__":
    main()
