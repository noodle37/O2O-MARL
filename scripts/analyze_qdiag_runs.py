#!/usr/bin/env python3
"""Summarise the O2O Q-diagnostic runs from their online.jsonl logs.

Each run logs an offline-checkpoint evaluation at t=0 (no ``fork/branch`` key)
and then online evaluations (``fork/branch=main``). Only the latter are kept;
duplicate t_env records (t=0 appears twice) collapse to the last one.

Outputs one JSON with the full series and one Markdown table per run, plus a
headline cross-run comparison. No environment or GPU access.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

METRICS = {
    "return": "evaluation/mean_episode_return",
    "win_rate": "evaluation/win_rate",
    "q_mean": "evaluation/mean_q_tot",
    "g_mean": "evaluation/mean_return_to_go",
    "q_std": "evaluation/q_std",
    "g_std": "evaluation/return_to_go_std",
    "bias": "evaluation/q_bias",
    "bias_median": "evaluation/q_bias_median",
    "bias_se": "evaluation/q_bias_se_episode",
    "norm_abs": "evaluation/q_normalized_diff",
    "norm_literal": "evaluation/q_normalized_diff_literal",
    "over_rate": "evaluation/q_overestimate_rate",
    "under_rate": "evaluation/q_underestimate_rate",
    "tau_full": "evaluation/q_kendall_tau",
    "tau_w10": "evaluation/q_window_kendall_tau_w10",
    "tau_w10_se": "evaluation/q_window_kendall_tau_se_w10",
    "tau_w10_neg_frac": "evaluation/q_window_tau_negative_frac_w10",
    "tau_w10_median": "evaluation/q_window_tau_median_w10",
}


def load_run(path: str) -> list[dict]:
    rows: dict[float, dict] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if "fork/branch" not in raw:
                continue
            if not any(k in raw for k in METRICS.values()):
                continue
            row = {"t_env": float(raw.get("t_env", 0.0))}
            for name, key in METRICS.items():
                value = raw.get(key)
                row[name] = float(value) if isinstance(value, (int, float)) else None
            rows[row["t_env"]] = row
    return [rows[t] for t in sorted(rows)]


def mean_of(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def window(rows: list[dict], lo: float, hi: float) -> list[dict]:
    return [r for r in rows if lo <= r["t_env"] <= hi]


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {}
    t_max = rows[-1]["t_env"]
    taus = [r for r in rows if r["tau_w10"] is not None]
    worst = min(taus, key=lambda r: r["tau_w10"]) if taus else None
    best = max(taus, key=lambda r: r["tau_w10"]) if taus else None
    biases = [r for r in rows if r["bias"] is not None]
    max_bias = max(biases, key=lambda r: abs(r["bias"])) if biases else None
    overs = [r for r in rows if r["over_rate"] is not None]
    max_over = max(overs, key=lambda r: r["over_rate"]) if overs else None
    neg_evals = [r for r in rows if r["tau_w10"] is not None and r["tau_w10"] < 0]
    longest, longest_span, cur, cur_start = 0, None, 0, None
    for r in rows:
        if r["tau_w10"] is not None and r["tau_w10"] < 0:
            if cur == 0:
                cur_start = r["t_env"]
            cur += 1
            if cur > longest:
                longest, longest_span = cur, (cur_start, r["t_env"])
        else:
            cur = 0
    out = {
        "n_online_evals": len(rows),
        "initial": rows[0],
        "final": rows[-1],
        "min_tau_w10": worst,
        "max_tau_w10": best,
        "max_abs_bias": max_bias,
        "max_over_rate": max_over,
        "n_negative_tau_evals": len(neg_evals),
        "longest_negative_streak": longest,
        "longest_negative_span": longest_span,
        "recovered_t": next(
            (
                r["t_env"]
                for r in rows
                if worst is not None
                and r["t_env"] > worst["t_env"]
                and r["tau_w10"] is not None
                and r["tau_w10"] >= 0.0
            ),
            None,
        ),
    }
    for name, (lo, hi) in {
        "early": (0.0, 0.1 * t_max),
        "mid": (0.4 * t_max, 0.6 * t_max),
        "late": (0.9 * t_max, t_max + 1e-9),
    }.items():
        chunk = window(rows, lo, hi)
        out[name] = {
            "n": len(chunk),
            "return": mean_of(chunk, "return"),
            "q_mean": mean_of(chunk, "q_mean"),
            "g_mean": mean_of(chunk, "g_mean"),
            "bias": mean_of(chunk, "bias"),
            "bias_median": mean_of(chunk, "bias_median"),
            "over_rate": mean_of(chunk, "over_rate"),
            "under_rate": mean_of(chunk, "under_rate"),
            "norm_abs": mean_of(chunk, "norm_abs"),
            "tau_full": mean_of(chunk, "tau_full"),
            "tau_w10": mean_of(chunk, "tau_w10"),
            "tau_w10_neg_frac": mean_of(chunk, "tau_w10_neg_frac"),
        }
    return out


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="checkpoints/qdiag_20261020")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    run_paths = sorted(glob.glob(os.path.join(args.root, "*", "online.jsonl")))
    # Archived aborted runs (e.g. *_300k_warmup) are kept on disk for audit but
    # excluded from the trajectory tables.
    run_paths = [p for p in run_paths if not Path(p).parent.name.endswith("_warmup")]
    if not run_paths:
        raise SystemExit(f"no online.jsonl under {args.root}")
    data = {}
    for path in run_paths:
        name = Path(path).parent.name
        rows = load_run(path)
        data[name] = {"rows": rows, "summary": summarise(rows)}
        print(f"{name}: {len(rows)} online evals, t_env up to {rows[-1]['t_env'] if rows else 0:.0f}")

    out_json = args.out_json or os.path.join(args.root, "qdiag_trajectory.json")
    out_md = args.out_md or os.path.join(args.root, "qdiag_trajectory.md")
    with open(out_json, "w") as fh:
        json.dump(data, fh, indent=2)

    lines = ["# O2O Q diagnostic trajectories", ""]
    lines.append("All values are online evaluations after the t=0 offline-checkpoint "
                 "evaluation; negative-return environments make the literal normalized "
                 "difference unstable, so the sign-stable `norm_abs=(Q-G)/max(|G|,1)` "
                 "and the raw `bias` are the primary columns.")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| run | evals | τ_w10 start | τ_w10 min (t) | τ_w10 end | bias start | bias end | bias med end | max \\|bias\\| (t) | over end | under end | return end | win end |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, entry in data.items():
        s = entry["summary"]
        if not s:
            continue
        init, fin = s["initial"], s["final"]
        worst, mb = s["min_tau_w10"], s["max_abs_bias"]
        lines.append(
            "| {} | {} | {} | {} ({}) | {} | {} | {} | {} | {} ({}) | {} | {} | {} | {} |".format(
                name, s["n_online_evals"],
                fmt(init["tau_w10"]), fmt(worst["tau_w10"]) if worst else "—",
                fmt(worst["t_env"], 0) if worst else "—", fmt(fin["tau_w10"]),
                fmt(init["bias"], 2), fmt(fin["bias"], 2), fmt(fin["bias_median"], 2),
                fmt(mb["bias"], 2) if mb else "—", fmt(mb["t_env"], 0) if mb else "—",
                fmt(fin["over_rate"], 3), fmt(fin["under_rate"], 3),
                fmt(fin["return"], 2), fmt(fin["win_rate"], 3),
            )
        )
    lines.append("")
    lines.append("## Rank-collapse duration")
    lines.append("")
    lines.append("| run | evals | τ10<0 evals | longest streak | streak t span | min τ10 (t) | first τ10≥0 after min |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, entry in data.items():
        s = entry["summary"]
        if not s:
            continue
        worst = s.get("min_tau_w10")
        span = s.get("longest_negative_span")
        lines.append(
            "| {} | {} | {} | {} | {} | {} ({}) | {} |".format(
                name, s["n_online_evals"], s["n_negative_tau_evals"],
                s["longest_negative_streak"],
                "{}–{}".format(fmt(span[0], 0), fmt(span[1], 0)) if span else "—",
                fmt(worst["tau_w10"]) if worst else "—",
                fmt(worst["t_env"], 0) if worst else "—",
                fmt(s.get("recovered_t"), 0) if s.get("recovered_t") is not None else "—",
            )
        )
    lines.append("")
    lines.append("## Early / mid / late means")
    lines.append("")
    lines.append("| run | segment | n | return | Q | G | bias | bias med | over | under | norm_abs | τ_full | τ_w10 | τ_w10<0 frac |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, entry in data.items():
        for seg in ("early", "mid", "late"):
            w = entry["summary"].get(seg, {})
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    name, seg, w.get("n", 0), fmt(w.get("return"), 2),
                    fmt(w.get("q_mean"), 2), fmt(w.get("g_mean"), 2),
                    fmt(w.get("bias"), 2), fmt(w.get("bias_median"), 2),
                    fmt(w.get("over_rate"), 3),
                    fmt(w.get("under_rate"), 3), fmt(w.get("norm_abs"), 2),
                    fmt(w.get("tau_full"), 3), fmt(w.get("tau_w10"), 3),
                    fmt(w.get("tau_w10_neg_frac"), 3),
                )
            )
    lines.append("")
    for name, entry in data.items():
        rows = entry["rows"]
        if not rows:
            continue
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| t_env | return | Q | G | bias | bias med | over | under | norm_abs | τ_full | τ_w10 | τ_w10<0 frac |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        # Keep every row when there are few; otherwise sample evenly and merge
        # in the worst-τ / max-bias rows so the interesting points survive.
        if len(rows) <= 24:
            chosen = rows
        else:
            idx = sorted({round(i * (len(rows) - 1) / 11) for i in range(12)})
            chosen = [rows[i] for i in idx]
            for special in (entry["summary"].get("min_tau_w10"),
                            entry["summary"].get("max_abs_bias")):
                if special and special not in chosen:
                    chosen.append(special)
            chosen.sort(key=lambda r: r["t_env"])
        for r in chosen:
            lines.append(
                "| {:.0f} | {:.2f} | {:.2f} | {:.2f} | {:+.2f} | {:+.2f} | {:.3f} | {:.3f} | {:.2f} | {:.3f} | {:.3f} | {:.3f} |".format(
                    r["t_env"], r["return"] or 0.0, r["q_mean"] or 0.0, r["g_mean"] or 0.0,
                    r["bias"] or 0.0, r["bias_median"] or 0.0,
                    r["over_rate"] or 0.0, r["under_rate"] or 0.0,
                    r["norm_abs"] or 0.0, r["tau_full"] or 0.0, r["tau_w10"] or 0.0,
                    r["tau_w10_neg_frac"] or 0.0,
                )
            )
        lines.append("")

    with open(out_md, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
