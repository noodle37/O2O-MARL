#!/usr/bin/env python3
"""Report whether one attribution arm has already reached its step budget.

The launcher is restartable: when it is stopped and started again it must not
start a second copy of an arm that is still running, and it must not redo an
arm that already finished.  The log is append-only, so a restarted arm leaves
the earlier run in front of the new one; like the analyzer, this script keeps
only the last run (a decrease of ``t_env`` marks a new run).

Prints one of

    done      the arm reached the budget (t_max - 50k steps)
    partial   the arm has data but has not reached the budget
    empty     the arm has no evaluation point yet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple


def last_run_points(path: str) -> List[Tuple[float, dict]]:
    """The evaluation points of the *last* run in an append-only log."""
    points: List[Tuple[float, dict]] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "evaluation/mean_episode_return" not in row:
                    continue
                t = float(row.get("t_env", 0.0))
                if points and t < points[-1][0] - 1e-9:
                    points = []  # a new run starts here
                points.append((t, row))
    except OSError:
        return []
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--t-max", type=int, default=1_000_000)
    args = parser.parse_args()

    points = last_run_points(args.log)
    if not points:
        print("empty")
        return
    last_t = max(t for t, _ in points)
    print("done" if last_t >= args.t_max - 50_000 else f"partial {int(last_t)}")


if __name__ == "__main__":
    main()
