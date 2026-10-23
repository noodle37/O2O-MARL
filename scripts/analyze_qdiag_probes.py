#!/usr/bin/env python3
"""Aggregate fixed-seed checkpoint probes produced by measure_paper_q.py.

Probe files are named ``<run>_t<step>.json``. For each run the primary table
uses window=10; W=5/20 are included as sensitivity columns. The paper averages
Kendall tau per episode first, so ``window_tau_episode_mean`` is the headline
rank number; the pooled ``window_tau_mean`` is reported alongside it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

NAME = re.compile(r"^(?P<run>.+)_t(?P<step>\d+)\.json$")

KEYS = {
    "return": ("mean_episode_return",),
    "q": ("q_mean",),
    "g": ("g_mean",),
    "bias": ("bias_mean",),
    "bias_med": ("bias_median",),
    "bias_se": ("bias_mean_se_episode",),
    "over": ("overestimate_rate",),
    "under": ("underestimate_rate",),
    "norm_abs": ("normalized_diff_abs",),
    "norm_lit": ("normalized_diff_literal",),
    "tau_full": ("kendall_tau_full_episode",),
    "tau10": ("window_tau_episode_mean",),
    "tau10_se": ("window_tau_episode_se",),
    "neg10": ("window_tau_negative_frac",),
    "slope": ("affine_fit_g_on_q", "slope"),
}


def get(summary: dict, spec):
    key, *rest = spec
    value = summary.get(key)
    if value is None:
        return None
    if rest:
        for part in rest:
            value = value.get(part) if isinstance(value, dict) else None
            if value is None:
                return None
    return float(value)


def fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def load_probes(root: str):
    probes = {}
    for path in sorted(glob.glob(os.path.join(root, "*_t*.json"))):
        m = NAME.match(Path(path).name)
        if not m:
            continue
        run, step = m.group("run"), int(m.group("step"))
        with open(path) as fh:
            payload = json.load(fh)
        results = payload.get("results", {})
        primary = str(payload.get("primary_window", 10))
        if primary not in results:
            continue
        probes.setdefault(run, {})[step] = {
            "path": path,
            "summary": results[primary],
            "w5": results.get("5", {}),
            "w20": results.get("20", {}),
            "n_episodes": payload.get("n_episodes"),
            "seed": payload.get("seed"),
        }
    return probes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="checkpoints/qdiag_20261020/probes")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    probes = load_probes(args.root)
    if not probes:
        raise SystemExit(f"no probe JSONs under {args.root}")
    out_json = args.out_json or os.path.join(args.root, "probe_trajectory.json")
    out_md = args.out_md or os.path.join(args.root, "probe_trajectory.md")

    compact = {}
    for run, rows in probes.items():
        compact[run] = {str(step): {"path": probes[run][step]["path"],
                                    **probes[run][step]["summary"],
                                    "w5": probes[run][step]["w5"],
                                    "w20": probes[run][step]["w20"]}
                        for step in rows}
    with open(out_json, "w") as fh:
        json.dump(compact, fh, indent=2)

    lines = ["# Fixed-seed checkpoint probes", ""]
    lines.append("Primary window = 10, stride 1, epsilon = 0, seed = 12345. "
                 "`tau10` averages Kendall tau within episodes before averaging "
                 "across episodes (paper order); `tau10_pooled` weights every "
                 "window equally. `neg10` is the fraction of windows with tau < 0.")
    lines.append("")
    for run in sorted(probes):
        rows = probes[run]
        lines.append(f"## {run}")
        lines.append("")
        lines.append("| t_env | return | Q | G | bias | bias med | SE | over | under | norm_abs | τ_full | τ10 | τ10 SE | τ10 pooled | neg10 | τ5 | τ20 | slope(G~Q) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for step in sorted(rows):
            entry = rows[step]
            s = entry["summary"]
            pooled = s.get("window_tau_mean")
            t5 = entry["w5"].get("window_tau_episode_mean")
            t20 = entry["w20"].get("window_tau_episode_mean")
            values = {
                "return": get(s, KEYS["return"]),
                "q": get(s, KEYS["q"]),
                "g": get(s, KEYS["g"]),
                "bias": get(s, KEYS["bias"]),
                "bias_med": get(s, KEYS["bias_med"]),
                "bias_se": get(s, KEYS["bias_se"]),
                "over": get(s, KEYS["over"]),
                "under": get(s, KEYS["under"]),
                "norm_abs": get(s, KEYS["norm_abs"]),
                "norm_lit": get(s, KEYS["norm_lit"]),
                "tau_full": get(s, KEYS["tau_full"]),
                "tau10": get(s, KEYS["tau10"]),
                "tau10_se": get(s, KEYS["tau10_se"]),
                "neg10": get(s, KEYS["neg10"]),
                "slope": get(s, KEYS["slope"]),
            }
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    step,
                    fmt(values["return"], 2), fmt(values["q"], 2), fmt(values["g"], 2),
                    fmt(values["bias"], 2), fmt(values["bias_med"], 2),
                    fmt(values["bias_se"], 2), fmt(values["over"]), fmt(values["under"]),
                    fmt(values["norm_abs"], 2), fmt(values["tau_full"]),
                    fmt(values["tau10"]), fmt(values["tau10_se"]),
                    fmt(pooled), fmt(values["neg10"]), fmt(t5), fmt(t20),
                    fmt(values["slope"]),
                )
            )
        lines.append("")

    pairs = {}
    for run in probes:
        if run.endswith("_off"):
            base = run[: -len("_off")]
            on_run = base + "_on"
            if on_run in probes:
                pairs[base] = (run, on_run)
    if pairs:
        lines.append("## Paired OVM contrast (on - off)")
        lines.append("")
        lines.append("Same checkpoint step and fixed initial states in both arms; "
                     "only the OVM target switch differs.")
        lines.append("")
        for base, (off_run, on_run) in sorted(pairs.items()):
            lines.append(f"### {base}")
            lines.append("")
            lines.append("| t_env | bias off | bias on | Δbias | τ10 off | τ10 on | Δτ10 | neg10 off | neg10 on | return off | return on |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
            shared = sorted(set(probes[off_run]) & set(probes[on_run]))
            for step in shared:
                so = probes[off_run][step]["summary"]
                sn = probes[on_run][step]["summary"]
                bo, bn = get(so, KEYS["bias"]), get(sn, KEYS["bias"])
                to, tn = get(so, KEYS["tau10"]), get(sn, KEYS["tau10"])
                lines.append(
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                        step, fmt(bo, 2), fmt(bn, 2),
                        fmt(bn - bo if None not in (bo, bn) else None, 2),
                        fmt(to), fmt(tn),
                        fmt(tn - to if None not in (to, tn) else None),
                        fmt(get(so, KEYS["neg10"])), fmt(get(sn, KEYS["neg10"])),
                        fmt(get(so, KEYS["return"]), 2), fmt(get(sn, KEYS["return"]), 2),
                    )
                )
            if not shared:
                lines.append("| — | — | — | — | — | — | — | — | — | — | — |")
            lines.append("")

    with open(out_md, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
