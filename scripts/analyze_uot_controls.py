#!/usr/bin/env python3
"""Aggregate the Section-4 simple controls on 5m_vs_6m.

Reads one online.jsonl per arm plus the optional 640-episode endpoint eval and
writes a JSON summary and a markdown table. FJR is the existing fixed
joint-ranking run (same offline checkpoint, same online recipe); the five new
arms differ from it in exactly one knowledge-reuse switch.

Usage:
  python scripts/analyze_uot_controls.py \
      --run-root checkpoints/uot_controls_5m_vs_6m \
      --fjr-log checkpoints/uot_temporal_5m_vs_6m/online.jsonl \
      --fjr-eval640 checkpoints/uot_temporal_5m_vs_6m/eval640_fjr.jsonl \
      --out-json documentation/experiment_reports/task6_controls_5m_vs_6m.json \
      --out-md documentation/experiment_reports/task6_controls_5m_vs_6m_tables.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARM_LABELS = {
    "fjr": "Fixed Joint-Ranking (FJR)",
    "pad": "Per-agent Distillation (PAD)",
    "tdr": "Time-Decayed Ranking (TDR)",
    "marginal": "Per-agent marginal support",
    "state": "State-only support",
    "none": "No support",
}
TAIL_START = 2_500_000
VERY_EARLY_END = 500_000
EARLY_END = 1_000_000
T_END = 3_000_000


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
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
    return [r for r in rows if "evaluation/win_rate" in r and "t_env" in r]


def _interp(points: List[Tuple[float, float]], x: float) -> float:
    if not points:
        raise ValueError("no points")
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, v0), (x1, v1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return v1
            w = (x - x0) / (x1 - x0)
            return v0 + w * (v1 - v0)
    return points[-1][1]


def _auc_window(
    rows: List[Dict[str, Any]], key: str, t0: float, t1: float, tol: float = 20_000
) -> Optional[float]:
    """Trapezoidal mean of ``key`` over [t0, t1], with endpoint interpolation.

    The online loop exits after the last eval point (which can sit up to one
    ``test_interval`` before t_max), so a complete run is allowed to fall short
    of t1 by ``tol``; the final value is held to the boundary. Runs that stop
    earlier return None, so an incomplete run is never silently compared with a
    complete one.
    """
    pts = sorted(
        (float(r["t_env"]), float(r[key]))
        for r in rows
        if key in r and r[key] is not None and "t_env" in r
    )
    if len(pts) < 2 or pts[0][0] > t0 + tol or pts[-1][0] < t1 - tol:
        return None
    xs = [t0] + [x for x, _ in pts if t0 < x < t1] + [t1]
    vs = [_interp(pts, x) for x in xs]
    area = sum(0.5 * (a + b) * (x1 - x0) for (x0, a), (x1, b) in zip(zip(xs, vs), zip(xs[1:], vs[1:])))
    return area / (t1 - t0) if t1 > t0 else None


def _window(rows: List[Dict[str, Any]], t0: float, t1: float) -> List[Dict[str, Any]]:
    return [r for r in rows if t0 <= float(r["t_env"]) <= t1]


def _mean(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
    return sum(vals) / len(vals) if vals else None


def _best(rows: List[Dict[str, Any]], key: str) -> Tuple[Optional[float], Optional[float]]:
    vals = [
        (float(r.get("t_env", 0.0)), float(r[key]))
        for r in rows
        if key in r and r[key] is not None
    ]
    if not vals:
        return None, None
    t, v = max(vals, key=lambda pair: pair[1])
    return v, t


def _eval640(path: str) -> Optional[Dict[str, Any]]:
    for row in reversed(_read_jsonl(path)):
        if "win_rate" in row and "episodes" in row:
            return row
    return None


def _wilson(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _parse_train_meta(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    text = open(path, encoding="utf-8", errors="ignore").read()
    meta: Dict[str, Any] = {}
    match = re.search(
        r"UOT controls ranking_scope=(\w+) support_mode=(\w+) marginal_mask=(\d+)/(\d+)", text
    )
    if match:
        meta["ranking_scope"] = match.group(1)
        meta["support_mode"] = match.group(2)
        meta["marginal_mask"] = int(match.group(3))
        meta["marginal_mask_total"] = int(match.group(4))
    match = re.search(r"UOT S_off=(\d+)", text)
    if match:
        meta["s_off"] = int(match.group(1))
    match = re.search(r"uot/state_offline_weight_mean': ([\d.eE+-]+)", text)
    if match:
        meta["state_offline_weight_mean"] = float(match.group(1))
    return meta


def _arm_summary(
    label: str,
    eval_rows: List[Dict[str, Any]],
    eval640_path: str,
    train_out: str,
) -> Dict[str, Any]:
    tail = _window(eval_rows, TAIL_START, T_END)
    best_win, best_t = _best(eval_rows, "evaluation/win_rate")
    out: Dict[str, Any] = {
        "label": label,
        "n_eval": len(eval_rows),
        "last_t": int(max((float(r["t_env"]) for r in eval_rows), default=0)),
        "early_auc": _auc_window(eval_rows, "evaluation/win_rate", 0.0, EARLY_END),
        "very_early_auc": _auc_window(
            eval_rows, "evaluation/win_rate", 0.0, VERY_EARLY_END
        ),
        "full_auc": _auc_window(eval_rows, "evaluation/win_rate", 0.0, T_END),
        "tail_win": _mean(tail, "evaluation/win_rate"),
        "tail_return": _mean(tail, "evaluation/mean_episode_return"),
        "last_win": (
            float(eval_rows[-1]["evaluation/win_rate"]) if eval_rows else None
        ),
        "best_win": best_win,
        "best_t": None if best_t is None else int(best_t),
        "eval640": _eval640(eval640_path),
        "diag": {
            key: _mean(tail, key)
            for key in (
                "reg_strength",
                "uot_eta_now",
                "uot_hinge_rate",
                "uot_sign_disagree",
                "uot_n_pairs",
                "uot_cand_in_frac",
                "uot_state_support_mean",
                "pad_n_pairs",
                "pad_hinge_rate",
                "pad_sign_disagree",
            )
        },
    }
    out.update(_parse_train_meta(train_out))
    if out["eval640"] and out["eval640"].get("win_rate") is not None:
        if "win_rate_ci95" not in out["eval640"]:
            n = int(out["eval640"].get("episodes") or 0)
            out["eval640"]["win_rate_ci95"] = list(_wilson(float(out["eval640"]["win_rate"]), n))
    return out


def _diff_ci(
    a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]
) -> Optional[float]:
    """95% half-width for a difference of two independent proportion estimates."""
    if not a or not b:
        return None
    pa, pb = a.get("win_rate"), b.get("win_rate")
    na, nb = int(a.get("episodes") or 0), int(b.get("episodes") or 0)
    if pa is None or pb is None or na <= 0 or nb <= 0:
        return None
    return 1.96 * math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", default="checkpoints/uot_controls_5m_vs_6m")
    ap.add_argument("--fjr-log", default="checkpoints/uot_temporal_5m_vs_6m/online.jsonl")
    ap.add_argument("--fjr-eval640", default="checkpoints/uot_temporal_5m_vs_6m/eval640_fjr.jsonl")
    ap.add_argument("--fjr-train-out", default="logs/uot_temporal_5m_vs_6m_restart.out")
    ap.add_argument("--out-json", default="documentation/experiment_reports/task6_controls_5m_vs_6m.json")
    ap.add_argument("--out-md", default="documentation/experiment_reports/task6_controls_5m_vs_6m_tables.md")
    args = ap.parse_args()

    arms: Dict[str, Dict[str, Any]] = {}
    eval_rows = _eval_rows(_read_jsonl(args.fjr_log))
    arms["fjr"] = _arm_summary(
        ARM_LABELS["fjr"], eval_rows, args.fjr_eval640 or "", args.fjr_train_out
    )
    for arm in ("pad", "tdr", "marginal", "state", "none"):
        run_dir = os.path.join(args.run_root, arm)
        arms[arm] = _arm_summary(
            ARM_LABELS[arm],
            _eval_rows(_read_jsonl(os.path.join(run_dir, "online.jsonl"))),
            os.path.join(run_dir, "eval640.jsonl"),
            os.path.join(run_dir, "train.out"),
        )

    fjr_640 = arms["fjr"]["eval640"]
    diffs: Dict[str, Any] = {}
    for arm in ("pad", "tdr", "marginal", "state", "none"):
        a = arms[arm]["eval640"]
        d = (
            float(a["win_rate"]) - float(fjr_640["win_rate"])
            if a and fjr_640 and a.get("win_rate") is not None
            else None
        )
        diffs[arm] = {
            "eval640_minus_fjr": d,
            "eval640_diff_halfwidth95": _diff_ci(a, fjr_640),
            "tail_minus_fjr": (
                arms[arm]["tail_win"] - arms["fjr"]["tail_win"]
                if arms[arm]["tail_win"] is not None and arms["fjr"]["tail_win"] is not None
                else None
            ),
        }

    payload = {
        "protocol": {
            "map": "5m_vs_6m",
            "seed": 1,
            "online_steps": T_END,
            "eval_every": 10_000,
            "eval_episodes": 32,
            "tail_window": [TAIL_START, T_END],
            "very_early_window": [0, VERY_EARLY_END],
            "early_window": [0, EARLY_END],
            "fjr_reference": args.fjr_log,
        },
        "arms": arms,
        "diffs_vs_fjr": diffs,
    }
    # On 5m Medium every agent took every action somewhere in the vault, so the
    # marginal mask is all ones and marginal/none implement the same candidate
    # pool. Their spread is a run-to-run noise reference, not a causal contrast.
    if (
        arms["marginal"].get("marginal_mask")
        == arms["marginal"].get("marginal_mask_total")
        and arms["marginal"].get("marginal_mask")
    ):
        payload["marginal_support_vacuous"] = True
    marginal_640 = arms["marginal"]["eval640"]
    none_640 = arms["none"]["eval640"]
    payload["null_checks"] = {
        "marginal_vs_none": {
            "eval640_minus": (
                float(marginal_640["win_rate"]) - float(none_640["win_rate"])
                if marginal_640
                and none_640
                and marginal_640.get("win_rate") is not None
                and none_640.get("win_rate") is not None
                else None
            ),
            "eval640_diff_halfwidth95": _diff_ci(marginal_640, none_640),
            "tail_minus": (
                arms["marginal"]["tail_win"] - arms["none"]["tail_win"]
                if arms["marginal"]["tail_win"] is not None
                and arms["none"]["tail_win"] is not None
                else None
            ),
        }
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    def _eval640_txt(entry: Optional[Dict[str, Any]]) -> str:
        if not entry or entry.get("win_rate") is None:
            return "n/a"
        lo, hi = entry.get("win_rate_ci95", [float("nan"), float("nan")])
        return f"{entry['win_rate']:.3f} [{lo:.3f},{hi:.3f}]"

    lines = [
        "# 5m_vs_6m simple controls: tables",
        "",
        "| arm | eval points | AUC 0-0.5M | AUC 0-1M | full AUC (0-3M) | tail win (2.5-3M) | "
        "last win | 640-ep win [95% CI] | tail reg | tail eta |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in ("fjr", "pad", "tdr", "marginal", "state", "none"):
        a = arms[key]
        diag = a["diag"]
        lines.append(
            f"| {a['label']} | {a['n_eval']} | {_fmt(a['very_early_auc'])} | "
            f"{_fmt(a['early_auc'])} | {_fmt(a['full_auc'])} | {_fmt(a['tail_win'])} | "
            f"{_fmt(a['last_win'])} | {_eval640_txt(a['eval640'])} | "
            f"{_fmt(diag.get('reg_strength'), 5)} | {_fmt(diag.get('uot_eta_now'))} |"
        )
    lines += [
        "",
        "| arm | 640-ep Δ vs FJR | Δ 95% half-width | tail Δ vs FJR | ranking scope | support | S_off |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in ("pad", "tdr", "marginal", "state", "none"):
        a = arms[key]
        d = diffs[key]
        lines.append(
            f"| {a['label']} | {_fmt(d['eval640_minus_fjr'])} | "
            f"{_fmt(d['eval640_diff_halfwidth95'])} | {_fmt(d['tail_minus_fjr'])} | "
            f"{a.get('ranking_scope', 'n/a')} | {a.get('support_mode', 'n/a')} | "
            f"{a.get('s_off', 'n/a')} |"
        )
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
