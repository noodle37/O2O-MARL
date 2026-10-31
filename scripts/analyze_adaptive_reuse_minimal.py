#!/usr/bin/env python3
"""Summarize the minimal adaptive-reuse pilot and short continuation runs."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("checkpoints/adaptive_reuse_minimal/seed2")
PILOT = Path("checkpoints/adaptive_reuse_pilot/seed2/pilot.json")


def final_row(branch):
    rows = [json.loads(line) for line in (ROOT / branch / "online.jsonl").read_text().splitlines() if line.strip()]
    return rows[-1]


def main():
    pilot = json.loads(PILOT.read_text())
    print("pilot", {
        "trials": pilot["conflict_episodes"],
        "trial_seconds": round(pilot["trial_seconds"], 2),
        "fit_seconds": round(pilot["fit_seconds"], 3),
        "arm_diff": pilot["analysis"]["overall_arm"].get("arm_diff"),
        "arm_diff_ci": pilot["analysis"]["overall_arm"].get("arm_diff_ci"),
        "response_auc": pilot["analysis"].get("auc_tau_hat_for_positive_Y"),
    })
    for branch in ("none", "joint", "adaptive"):
        row = final_row(branch)
        out = {
            "t_env": row["t_env"],
            "mean_return": row["evaluation/mean_episode_return"],
            "win_rate": row["evaluation/win_rate"],
        }
        out.update({k: v for k, v in row.items() if k.startswith("adaptive/")})
        print(branch, out)


if __name__ == "__main__":
    main()
