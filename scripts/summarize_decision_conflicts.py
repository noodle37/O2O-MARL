#!/usr/bin/env python3
"""Post-process scripts/analyze_decision_conflicts.py output.

Answers the decision-relevance question directly:
  * among live-vs-teacher greedy conflicts, how large is |dQbar|;
  * do conflicts with large |dQbar| also carry real forced-return differences;
  * does that hold under both common continuations and across maps/phases.

Read-only: consumes the JSON written by analyze_decision_conflicts.py.

Usage:
  python scripts/summarize_decision_conflicts.py \
    --input logs/decision_conflicts_seed1.json \
    --out logs/decision_conflicts_seed1_summary.json \
    --md documentation/experiment_reports/task7_decision_conflicts_tables.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

MARGIN_BINS = (
    (0.0, 0.05, "0.00–0.05"),
    (0.05, 0.10, "0.05–0.10"),
    (0.10, 0.25, "0.10–0.25"),
    (0.25, 0.50, "0.25–0.50"),
    (0.50, math.inf, "≥0.50"),
)
RETURN_TIE = 0.1
PHASES = ("early", "mid", "late")


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def margin_bin(margin: float) -> str:
    for lo, hi, label in MARGIN_BINS:
        if lo <= margin < hi:
            return label
    return MARGIN_BINS[-1][2]


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


def aggregate(rows: List[dict]) -> Optional[dict]:
    if not rows:
        return None
    diffs = [r["diff"] for r in rows]
    regrets = [r["regret"] for r in rows]
    decided = [d for d in diffs if abs(d) > RETURN_TIE]
    return {
        "n": len(rows),
        "decided": len(decided),
        "live_better": sum(d > RETURN_TIE for d in decided),
        "live_worse": sum(d < -RETURN_TIE for d in decided),
        "tie_frac": 1.0 - len(decided) / len(rows),
        "live_better_frac_decided": (
            sum(d > RETURN_TIE for d in decided) / len(decided) if decided else None
        ),
        "live_worse_frac_decided": (
            sum(d < -RETURN_TIE for d in decided) / len(decided) if decided else None
        ),
        "mean_diff": statistics.fmean(diffs),
        "median_diff": statistics.median(diffs),
        "mean_abs_diff": statistics.fmean(abs(d) for d in diffs),
        "median_abs_diff": statistics.median(abs(d) for d in diffs),
        "p90_abs_diff": float(np.percentile(np.abs(diffs), 90)),
        "frac_abs_diff_le_0.1": sum(abs(d) <= RETURN_TIE for d in diffs) / len(diffs),
        "mean_regret": statistics.fmean(regrets),
        "median_regret": statistics.median(regrets),
        "regret_gt1_frac": sum(r > 1.0 for r in regrets) / len(regrets),
        "margin_min": min(r["margin"] for r in rows),
        "margin_median": statistics.median(r["margin"] for r in rows),
        "margin_max": max(r["margin"] for r in rows),
    }


def cluster_stat(
    rows: List[dict], stat_fn, n_boot: int = 10000, seed: int = 12345
):
    """Bootstrap over map seeds; rows from one map are correlated."""
    by_map: Dict[int, List[dict]] = {}
    for r in rows:
        by_map.setdefault(int(r["map_seed"]), []).append(r)
    values = []
    for group in by_map.values():
        v = stat_fn(group)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        values.append(float(v))
    if not values:
        return None
    map_means = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, map_means.size, size=(n_boot, map_means.size))
    boot = map_means[idx].mean(axis=1)
    return {
        "n_maps": int(map_means.size),
        "map_mean": float(map_means.mean()),
        "map_sd": float(map_means.std(ddof=1)) if map_means.size > 1 else None,
        "ci95": [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ],
    }


def loss_prob(rows: List[dict]) -> Optional[float]:
    decided = [r["diff"] for r in rows if abs(r["diff"]) > RETURN_TIE]
    if not decided:
        return None
    return sum(d < 0 for d in decided) / len(decided)


def build(records: List[dict]) -> dict:
    tags = sorted(
        {
            tag
            for r in records
            for tag in (r.get("joints") or {})
            if tag not in ("teacher", "teacher_ref")
        }
    )
    cont_tags = sorted(
        {ctag for r in records for ctag in (r.get("returns") or {})}
    )
    out: dict = {
        "n_records": len(records),
        "n_maps": len({int(r["map_seed"]) for r in records}),
        "by_base": {},
        "by_arm_cont": {},
        "cluster": {},
        "notes": [
            "Conflicts are live greedy joint != teacher greedy joint at the same state.",
            "margin = |Qbar(live joint) - Qbar(teacher joint)| from frozen offline Q.",
            "Return differences force the two joints then finish under a common continuation.",
            "Cluster CIs bootstrap over map seeds because probes within a map are correlated.",
            "Teacher joint membership in the production UOT pool is not stored; "
            "hinge/pool conclusions apply only to recorded pool statistics.",
        ],
    }
    for r in records:
        arm = r.get("base_arm", "?")
        out["by_base"][arm] = out["by_base"].get(arm, 0) + 1

    for tag in tags:
        for ctag in cont_tags:
            rows: List[dict] = []
            for r in records:
                joints = r.get("joints") or {}
                qbar = r.get("qbar") or {}
                returns = (r.get("returns") or {}).get(ctag) or {}
                if tag not in joints or "teacher" not in joints:
                    continue
                u = ",".join(str(int(x)) for x in joints[tag])
                t = ",".join(str(int(x)) for x in joints["teacher"])
                if u == t or u not in qbar or t not in qbar:
                    continue
                if u not in returns or t not in returns:
                    continue
                margin = abs(float(qbar[u]) - float(qbar[t]))
                d = float(returns[u]) - float(returns[t])
                best = max(float(v) for v in returns.values())
                rows.append(
                    {
                        "map_seed": int(r["map_seed"]),
                        "base_arm": r.get("base_arm", "?"),
                        "phase": r.get("phase", "?"),
                        "tag": tag,
                        "cont": ctag,
                        "margin": margin,
                        "diff": d,
                        "regret": best - float(returns[u]),
                    }
                )
            if not rows:
                continue
            arm_out = out["by_arm_cont"].setdefault(tag, {})
            arm_out[ctag] = {
                "all": aggregate(rows),
                "margin_bins": {},
                "phase": {},
            }
            for _, _, label in MARGIN_BINS:
                sel = [r for r in rows if margin_bin(r["margin"]) == label]
                agg = aggregate(sel)
                if agg:
                    arm_out[ctag]["margin_bins"][label] = agg
            for phase in PHASES:
                sel = [r for r in rows if r["phase"] == phase]
                agg = aggregate(sel)
                if agg:
                    arm_out[ctag]["phase"][phase] = agg
            all_agg = arm_out[ctag]["all"]
            all_agg["frac_conflicts_margin_lt_0.1"] = sum(
                r["margin"] < 0.1 for r in rows
            ) / len(rows)
            all_agg["frac_conflicts_margin_ge_0.1"] = 1.0 - all_agg[
                "frac_conflicts_margin_lt_0.1"
            ]

            # Cluster-level uncertainty for the headline loss probability.
            out["cluster"].setdefault(tag, {})[ctag] = {
                "loss_prob": cluster_stat(rows, loss_prob),
                "mean_diff": cluster_stat(
                    rows, lambda rs: statistics.fmean(r["diff"] for r in rs)
                ),
                "mean_regret": cluster_stat(
                    rows, lambda rs: statistics.fmean(r["regret"] for r in rs)
                ),
                "margin_median": cluster_stat(
                    rows, lambda rs: statistics.median(r["margin"] for r in rs)
                ),
            }
    return out


def build_markdown(summary: dict) -> str:
    lines: List[str] = []
    lines.append("# Decision-conflict breakdown (auto-generated)")
    lines.append("")
    lines.append(
        f"Records: {summary['n_records']} probes over {summary['n_maps']} maps; "
        f"base arms: {summary['by_base']}. Return tie threshold: {RETURN_TIE}."
    )
    lines.append("")
    lines.append("## Headline: live-vs-teacher conflicts by arm and continuation")
    lines.append("")
    lines.append(
        "| arm | continuation | n | abs(ΔQ̄)<0.1 | decided | live better | live worse | "
        "mean Δreturn | median Δreturn | mean abs(Δreturn) | mean regret | regret>1 |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tag, conts in summary["by_arm_cont"].items():
        for ctag, item in conts.items():
            a = item["all"]
            lines.append(
                "| {tag} | {ctag} | {n} | {near} | {dec} | {b} | {w} | {md} | {med} | "
                "{mad} | {reg} | {r1} |".format(
                    tag=tag,
                    ctag=ctag,
                    n=a["n"],
                    near=fmt(a.get("frac_conflicts_margin_lt_0.1")),
                    dec=a["decided"],
                    b=a["live_better"],
                    w=a["live_worse"],
                    md=fmt(a["mean_diff"], 2),
                    med=fmt(a["median_diff"], 2),
                    mad=fmt(a["mean_abs_diff"], 2),
                    reg=fmt(a["mean_regret"], 2),
                    r1=fmt(a["regret_gt1_frac"]),
                )
            )
    lines.append("")
    lines.append("## Margin-bin breakdown")
    lines.append("")
    lines.append(
        "| arm | continuation | abs(ΔQ̄) bin | n | better | worse | tie | "
        "mean Δreturn | mean abs(Δreturn) | frac abs(Δreturn)≤0.1 | mean regret |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tag, conts in summary["by_arm_cont"].items():
        for ctag, item in conts.items():
            for _, _, label in MARGIN_BINS:
                a = item["margin_bins"].get(label)
                if not a:
                    continue
                lines.append(
                    "| {tag} | {ctag} | {label} | {n} | {b} | {w} | {tie} | "
                    "{md} | {mad} | {near} | {reg} |".format(
                        tag=tag,
                        ctag=ctag,
                        label=label,
                        n=a["n"],
                        b=a["live_better"],
                        w=a["live_worse"],
                        tie=a["n"] - a["decided"],
                        md=fmt(a["mean_diff"], 2),
                        mad=fmt(a["mean_abs_diff"], 2),
                        near=fmt(a.get("frac_abs_diff_le_0.1")),
                        reg=fmt(a["mean_regret"], 2),
                    )
                )
    lines.append("")
    lines.append("## Phase breakdown (all conflict margins)")
    lines.append("")
    lines.append(
        "| arm | continuation | phase | n | better | worse | mean Δreturn | mean regret |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for tag, conts in summary["by_arm_cont"].items():
        for ctag, item in conts.items():
            for phase in PHASES:
                a = item["phase"].get(phase)
                if not a:
                    continue
                lines.append(
                    f"| {tag} | {ctag} | {phase} | {a['n']} | {a['live_better']} | "
                    f"{a['live_worse']} | {fmt(a['mean_diff'], 2)} | {fmt(a['mean_regret'], 2)} |"
                )
    lines.append("")
    lines.append("## Map-cluster bootstrap (headline metrics)")
    lines.append("")
    lines.append(
        "| arm | continuation | metric | maps | map mean | map SD | 95% CI |"
    )
    lines.append("|---|---|---|---:|---:|---:|---|")
    for tag, conts in summary["cluster"].items():
        for ctag, metrics in conts.items():
            for name, item in metrics.items():
                if not item:
                    continue
                lines.append(
                    f"| {tag} | {ctag} | {name} | {item['n_maps']} | "
                    f"{fmt(item['map_mean'])} | {fmt(item['map_sd'])} | "
                    f"{ci_str(item['ci95'])} |"
                )
    lines.append("")
    lines.append("Notes:")
    for note in summary["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="logs/decision_conflicts_seed1.json")
    ap.add_argument("--out", default="logs/decision_conflicts_seed1_summary.json")
    ap.add_argument("--md", default="")
    args = ap.parse_args()

    payload = read_json(args.input)
    records = payload.get("records") or []
    summary = build(records)
    summary["input"] = args.input
    summary["source_summary"] = payload.get("summary")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[summarize] wrote {args.out}")
    if args.md:
        md = build_markdown(summary)
        os.makedirs(os.path.dirname(args.md) or ".", exist_ok=True)
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"[summarize] wrote {args.md}")


if __name__ == "__main__":
    main()
