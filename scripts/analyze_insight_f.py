#!/usr/bin/env python3
"""Summarise the Insight F collection-value vs teaching-value experiment.

For every run file and student/budget, print B_v(z) (the collector's own greedy
return) next to T_v(z) (the student's gain after training on that collector's
data), the two rankings, and their rank correlation.  A disagreement between
the B and T rankings is the evidence that collection quality and teaching
quality are distinct.

Usage:
  python scripts/analyze_insight_f.py \
      --runs checkpoints/collection_value/5m_vs_6m_gentle.json \
             checkpoints/collection_value/5m_vs_6m.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr


def rank(values):
    return [n for n, _ in sorted(values.items(), key=lambda kv: -kv[1])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    summary = {}
    for path in args.runs:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        label = os.path.splitext(os.path.basename(path))[0]
        collectors = payload["collectors"]
        B = {name: c["B_own_policy"]["return"] for name, c in collectors.items()}
        behavior = {name: c["data"]["behavior_return"] for name, c in collectors.items()}
        print(f"\n=== {label} (collect {payload['collect_episodes']} eps @ eps={payload['collect_eps']}) ===")
        print("collector        B_own    behavior_return  mean_length")
        for name, c in collectors.items():
            print(
                f"  {name:<12} {B[name]:7.3f}  {c['data']['behavior_return']:7.3f}"
                f"          {c['data']['mean_length']:.1f}"
            )
        print(f"  B ranking: {rank(B)}")

        rows = {}
        for student in payload["students"]:
            base = payload["students"][student]["base_eval"]
            budgets = sorted(
                {key for c in collectors.values() for key in c.get("students", {}).get(student, {})},
                key=lambda k: int(k.lstrip("u")),
            )
            for budget in budgets:
                T, T_win, evals = {}, {}, {}
                for name, c in collectors.items():
                    row = c.get("students", {}).get(student, {}).get(budget)
                    if row is None:
                        continue
                    T[name] = row["T_return"]
                    T_win[name] = row["T_win"]
                    evals[name] = row["eval"]["return"]
                if not T:
                    continue
                names = list(T)
                rho = (
                    float(spearmanr([B[n] for n in names], [T[n] for n in names]).statistic)
                    if len(names) > 2
                    else float("nan")
                )
                rows[f"{student}/{budget}"] = {
                    "base_return": base["return"],
                    "B": {n: B[n] for n in names},
                    "T_return": T,
                    "T_win": T_win,
                    "student_return": evals,
                    "B_rank": rank({n: B[n] for n in names}),
                    "T_rank": rank(T),
                    "spearman_B_T": rho,
                }
                print(f"\n  student={student} base={base['return']:.3f} {budget}")
                for n in names:
                    print(
                        f"    {n:<12} B={B[n]:7.3f}  T={T[n]:+7.3f}  "
                        f"student_return={evals[n]:7.3f}  T_win={T_win[n]:+.3f}"
                    )
                print(f"    B rank={rank({n: B[n] for n in names})}")
                print(f"    T rank={rank(T)}  spearman={rho:+.3f}")
        summary[label] = {"collectors": {"B": B, "behavior_return": behavior}, "students": rows}

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.runs[0])), "5m_vs_6m_analysis.json"
    )
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
