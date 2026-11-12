#!/usr/bin/env python3
"""Aggregate the TCPA arm logs into a comparison table.

Reads every ``checkpoints/tcpa_<map>/<variant>/seed<seed>/online.jsonl`` and
reports, per arm:

* the evaluation curve (return / win rate) at fixed step landmarks,
* the final and best performance,
* the mechanism diagnostics (team regret, retention gate, offline action
  agreement) and how they move over training.

The baseline arms (ovmse / sapr / ackf / ...) are read with the same parser so
that every arm sits in one table.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

LANDMARKS = (0, 250_000, 500_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000)

KEYS = (
    "evaluation/mean_episode_return",
    "evaluation/win_rate",
    "tcpa/regret_mean_interval_mean",
    "tcpa/gate_mean_interval_mean",
    "tcpa/frac_gate_gt_half_interval_mean",
    "tcpa/offline_action_agreement_interval_mean",
    "tcpa/hinge_mean_interval_mean",
    "tcpa/tau_g_interval_mean",
    "tcpa/regret_violations_interval_mean",
)


def read_rows(path: str) -> List[Dict]:
    rows: List[Dict] = []
    if not os.path.isfile(path):
        return rows
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


def eval_curve(rows: List[Dict]) -> List[Dict]:
    """Evaluation rows only, deduplicated by step (keep the last of a step)."""
    by_step: Dict[float, Dict] = {}
    for row in rows:
        if "evaluation/mean_episode_return" not in row:
            continue
        step = float(row.get("t_env", 0.0))
        by_step[step] = row
    return [by_step[s] for s in sorted(by_step)]


def nearest_value(curve: List[Dict], landmark: int, key: str) -> Optional[float]:
    """Value of ``key`` at the first evaluation at or after ``landmark``."""
    for row in curve:
        if float(row.get("t_env", 0.0)) >= landmark and key in row:
            return float(row[key])
    return None


def summarize(name: str, path: str) -> Dict:
    rows = read_rows(path)
    curve = eval_curve(rows)
    summary: Dict = {"arm": name, "path": path, "n_eval": len(curve)}
    if not curve:
        return summary
    summary["last_t_env"] = float(curve[-1].get("t_env", 0.0))
    summary["final_return"] = float(
        curve[-1].get("evaluation/mean_episode_return", float("nan"))
    )
    summary["final_win_rate"] = float(curve[-1].get("evaluation/win_rate", float("nan")))
    summary["best_return"] = max(
        float(r.get("evaluation/mean_episode_return", float("-inf"))) for r in curve
    )
    for key in KEYS:
        if key.startswith("evaluation/"):
            continue
        summary[key.replace("_interval_mean", "")] = (
            nearest_value(curve, int(summary["last_t_env"]), key)
        )
    summary["rows"] = curve
    summary["curve"] = [
        (
            int(float(r.get("t_env", 0.0))),
            round(float(r.get("evaluation/mean_episode_return", float("nan"))), 2),
            r.get("evaluation/win_rate"),
        )
        for r in curve
    ]
    summary["mechanism"] = [
        (
            int(float(r.get("t_env", 0.0))),
            round(float(r.get("tcpa/regret_mean_interval_mean", float("nan"))), 4),
            round(float(r.get("tcpa/gate_mean_interval_mean", float("nan"))), 4),
            round(
                float(r.get("tcpa/offline_action_agreement_interval_mean", float("nan"))),
                4,
            ),
        )
        for r in curve
        if "tcpa/regret_mean_interval_mean" in r
    ]
    return summary


def collect(root: str) -> List[Dict]:
    """Find every ``<root>/tcpa_<map>/<variant>/<seed>/online.jsonl``."""
    arms: List[Dict] = []
    if not os.path.isdir(root):
        return arms
    for map_name in sorted(d for d in os.listdir(root) if d.startswith("tcpa_")):
        map_dir = os.path.join(root, map_name)
        if not os.path.isdir(map_dir):
            continue
        for variant in sorted(os.listdir(map_dir)):
            variant_dir = os.path.join(map_dir, variant)
            if not os.path.isdir(variant_dir):
                continue
            for seed_dir in sorted(os.listdir(variant_dir)):
                path = os.path.join(variant_dir, seed_dir, "online.jsonl")
                if os.path.isfile(path):
                    arms.append(
                        summarize(f"{map_name}/{variant}/{seed_dir}", path)
                    )
    return arms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="checkpoints")
    parser.add_argument("--json", default="", help="optional path for a JSON dump")
    args = parser.parse_args()

    arms = collect(args.root)
    if not arms:
        print("no TCPA arm logs found")
        return

    header = (
        f"{'arm':<44}{'last_t':>10}{'final_ret':>11}{'best_ret':>10}"
        f"{'win':>7}{'regret':>9}{'gate':>8}{'agree':>8}{'viol':>7}"
    )
    print(header)
    print("-" * len(header))

    def fmt(value) -> float:
        """0.0 must not become NaN through a falsy ``or``."""
        return float("nan") if value is None else float(value)

    for arm in arms:
        print(
            f"{arm['arm']:<44}{int(arm.get('last_t_env', 0)):>10}"
            f"{fmt(arm.get('final_return')):>11.2f}"
            f"{fmt(arm.get('best_return')):>10.2f}"
            f"{fmt(arm.get('final_win_rate')):>7.2f}"
            f"{fmt(arm.get('tcpa/regret_mean')):>9.4f}"
            f"{fmt(arm.get('tcpa/gate_mean')):>8.3f}"
            f"{fmt(arm.get('tcpa/offline_action_agreement')):>8.3f}"
            f"{fmt(arm.get('tcpa/regret_violations')):>7.0f}"
        )

    print("\nlandmark returns (mean_episode_return):")
    landmarks = [lm for lm in LANDMARKS]
    print(f"{'arm':<44}" + "".join(f"{lm // 1000:>8}k" for lm in landmarks))
    for arm in arms:
        curve = arm.get("rows", [])
        cells = []
        for lm in landmarks:
            value = nearest_value(curve, lm, "evaluation/mean_episode_return")
            cells.append(f"{value:>9.2f}" if value is not None else f"{'-':>9}")
        print(f"{arm['arm']:<44}" + "".join(cells))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(arms, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
