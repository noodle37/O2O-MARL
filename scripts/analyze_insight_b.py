#!/usr/bin/env python3
"""Compact cross-version summary of the Insight B verified-preference probe.

For every policy version it reports the IPW/randomization estimate of the
teacher effect, the calibrated-response-model diagnostics, the online critic's
agreement, and the confidence-gate results, plus the raw conflict composition
(disagreement count, phase, critic margins).

Usage:
  python scripts/analyze_insight_b.py \
      --runs checkpoints/verified_prefs/5m_vs_6m/t25000.json \
             checkpoints/verified_prefs/5m_vs_6m/t250000.json \
             checkpoints/verified_prefs/5m_vs_6m/t500000.json \
             checkpoints/verified_prefs/5m_vs_6m/t1000000.json \
      --out checkpoints/verified_prefs/5m_vs_6m/analysis.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def summarize(payload):
    a = payload["analysis"]
    records = payload["records"]
    q_on = np.array([r["q_on_margin"] for r in records])
    q_off = np.array([r["q_off_margin"] for r in records])
    n_dis = np.array([r["n_disagree"] for r in records])
    phase = np.array([r["phase"] for r in records])
    out = {
        "policy_version": payload["policy_version"],
        "episodes_attempted": payload["episodes_attempted"],
        "conflict_episodes": payload["conflict_episodes"],
        "no_conflict_episodes": payload["no_conflict_episodes"],
        "trials": a.get("n_trials", 0),
        "overall": a.get("overall_arm"),
        "teacher_margin_positive_rate": a.get("teacher_margin_positive_rate"),
        "tau_hat_mean": a.get("tau_hat_mean"),
        "tau_hat_positive_rate": a.get("tau_hat_positive_rate"),
        "spearman_tau_hat_Y": a.get("spearman_tau_hat_Y"),
        "spearman_neg_q_on_margin_Y": a.get("spearman_neg_q_on_margin_Y"),
        "spearman_q_off_margin_Y": a.get("spearman_q_off_margin_Y"),
        "auc_tau_hat_for_positive_Y": a.get("auc_tau_hat_for_positive_Y"),
        "auc_neg_q_on_margin_for_positive_Y": a.get("auc_neg_q_on_margin_for_positive_Y"),
        "auc_q_off_margin_for_positive_Y": a.get("auc_q_off_margin_for_positive_Y"),
        "confidence_gate_q0.25": a.get("confidence_gate_q0.25"),
        "confidence_gate_q0.5": a.get("confidence_gate_q0.5"),
        "composition": {
            "online_critic_self_margin_positive_rate": float((q_on > 0).mean()),
            "mean_q_on_margin": float(q_on.mean()),
            "mean_abs_q_on_margin": float(np.abs(q_on).mean()),
            "teacher_self_margin_positive_rate": float((q_off > 0).mean()),
            "mean_q_off_margin": float(q_off.mean()),
            "mean_abs_q_off_margin": float(np.abs(q_off).mean()),
            "mean_n_disagree": float(n_dis.mean()),
            "n_disagree_hist": {
                str(int(k)): int((n_dis == k).sum()) for k in np.unique(n_dis)
            },
            "mean_phase": float(phase.mean()),
            "median_phase": float(np.median(phase)),
        },
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = {}
    for path in args.runs:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        rows[str(payload["policy_version"])] = summarize(payload)

    times = sorted(rows, key=int)
    print(
        "version     trials  conflict  arm_diff [CI]                 tau_hat  pos    "
        "rho(tau,Y)  rho(-qon,Y)  AUC_tau  AUC_qon  gate25   gate50"
    )
    for t in times:
        r = rows[t]
        o = r["overall"]
        g25 = r["confidence_gate_q0.25"]["kept"]["arm_diff"] if r["confidence_gate_q0.25"] else float("nan")
        g50 = r["confidence_gate_q0.5"]["kept"]["arm_diff"] if r["confidence_gate_q0.5"] else float("nan")
        print(
            f"{t:>8}  {r['trials']:6d}  {r['conflict_episodes']:8d}  "
            f"{o['arm_diff']:+.3f} [{o['arm_diff_ci'][0]:+.3f},{o['arm_diff_ci'][1]:+.3f}]  "
            f"{r['tau_hat_mean']:+.3f}  {r['tau_hat_positive_rate']:.3f}  "
            f"{r['spearman_tau_hat_Y']:+.3f}      {r['spearman_neg_q_on_margin_Y']:+.3f}       "
            f"{r['auc_tau_hat_for_positive_Y']:.3f}    {r['auc_neg_q_on_margin_for_positive_Y']:.3f}    "
            f"{g25:+.3f}  {g50:+.3f}"
        )

    print("\nConflict composition")
    for t in times:
        c = rows[t]["composition"]
        hist = " ".join(f"{k}:{v}" for k, v in c["n_disagree_hist"].items())
        print(
            f"  t={t:>8}: critic_self_margin={c['online_critic_self_margin_positive_rate']:.3f} "
            f"q_on_mean={c['mean_q_on_margin']:+.3f} |q_on|={c['mean_abs_q_on_margin']:.3f} "
            f"q_off_mean={c['mean_q_off_margin']:+.3f} |q_off|={c['mean_abs_q_off_margin']:.3f} "
            f"n_dis={c['mean_n_disagree']:.2f} phase_med={c['median_phase']:.3f} [{hist}]"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
