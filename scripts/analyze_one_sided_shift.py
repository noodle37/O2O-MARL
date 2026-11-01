#!/usr/bin/env python3
"""Read off the one-sided policy-shift experiment.

For each condition we report mean return / win rate with a Wilson CI. Every
condition is compared with the all-offline baseline O. Pairing is verified
through per-episode initial-state hashes; SC2 restarts a game in place, so only
episode 0 is guaranteed to share an initial state across processes. When the
later hashes diverge the comparison falls back to an unpaired bootstrap. The
decomposition is

  Delta A = R(all deviate)        - R(O)
  Delta B = R(teammates deviate)  - R(O)     (focal fixed offline)
  Delta C = R(focal deviates)     - R(O)     (teammates fixed offline)

and R_B - R_A / R_C - R_A say whether the focal agent's own deviation still
matters once the other side has already shifted.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def wilson_ci(p: float, n: int, z: float = 1.96) -> List[Optional[float]]:
    if n <= 0:
        return [None, None]
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _finite_mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _load(root: str, include: List[str]) -> Dict[str, Dict[str, Any]]:
    conditions: Dict[str, Dict[str, Any]] = {}
    for sub in include:
        for path in sorted(glob.glob(os.path.join(root, sub, "*.jsonl"))):
            episodes: List[Dict[str, Any]] = []
            summary: Optional[Dict[str, Any]] = None
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("summary"):
                        summary = row
                    else:
                        episodes.append(row)
            if summary is None:
                print(f"[warn] no summary row: {path}")
                continue
            episodes.sort(key=lambda r: int(r["episode"]))
            conditions[summary["tag"]] = {
                "path": path,
                "summary": summary,
                "episodes": episodes,
            }
    return conditions


def _condition_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    wins = [float(r["win"]) for r in rows]
    win_rate = float(np.mean(wins)) if wins else float("nan")
    return {
        "n": n,
        "return_mean": _finite_mean([r["return"] for r in rows]),
        "win_rate": win_rate,
        "win_ci95": wilson_ci(win_rate, n),
        "length_mean": _finite_mean([r["length"] for r in rows]),
        "timeout_rate": _finite_mean([r["timeout"] for r in rows]),
        "attack_share_mean": _finite_mean([r["attack_share"] for r in rows]),
        "focus_fire_rate_mean": _finite_mean(
            [r["focus_fire_rate"] for r in rows]
        ),
        "alive_enemies_end_mean": _finite_mean(
            [r["alive_enemies_end"] for r in rows]
        ),
        "q_off_tot_mean": _finite_mean([r["q_off_tot_mean"] for r in rows]),
        "perturb_events_total": int(
            sum(sum(r["perturb_events"]) for r in rows)
        ),
        "perturb_changes_total": int(
            sum(sum(r["perturb_changes"]) for r in rows)
        ),
        "alive_agent_steps_total": int(
            sum(r["alive_agent_steps"] for r in rows)
        ),
    }


def _paired_diff(
    rows_c: List[Dict[str, Any]],
    rows_o: List[Dict[str, Any]],
    key: str,
    chunk: int,
    n_boot: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    n = min(len(rows_c), len(rows_o))
    diffs = np.array(
        [float(rows_c[k][key]) - float(rows_o[k][key]) for k in range(n)],
        dtype=np.float64,
    )
    n_chunks = (n + chunk - 1) // chunk
    chunk_means = np.array(
        [
            diffs[i * chunk : min((i + 1) * chunk, n)].mean()
            for i in range(n_chunks)
        ],
        dtype=np.float64,
    )
    boot_chunks = rng.choice(chunk_means, size=(n_boot, n_chunks), replace=True).mean(1)
    boot_episodes = rng.choice(diffs, size=(n_boot, n), replace=True).mean(1)
    lo, hi = np.quantile(boot_chunks, [0.025, 0.975])
    elo, ehi = np.quantile(boot_episodes, [0.025, 0.975])
    return {
        "delta": float(diffs.mean()),
        "ci95_chunk": [float(lo), float(hi)],
        "ci95_episode": [float(elo), float(ehi)],
        "n": int(n),
        "n_positive_chunks": int((chunk_means > 0).sum()),
        "n_chunks": int(n_chunks),
    }


def _unpaired_diff(
    rows_c: List[Dict[str, Any]],
    rows_o: List[Dict[str, Any]],
    key: str,
    n_boot: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    c = np.array([float(r[key]) for r in rows_c], dtype=np.float64)
    o = np.array([float(r[key]) for r in rows_o], dtype=np.float64)
    boot = (
        rng.choice(c, size=(n_boot, c.size), replace=True).mean(1)
        - rng.choice(o, size=(n_boot, o.size), replace=True).mean(1)
    )
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "delta": float(c.mean() - o.mean()),
        "ci95_chunk": None,
        "ci95_episode": [float(lo), float(hi)],
        "n": int(min(c.size, o.size)),
        "n_positive_chunks": None,
        "n_chunks": None,
    }


def _compare(
    conditions: Dict[str, Dict[str, Any]],
    label: str,
    baseline: str,
    key: str,
    chunk: int,
    n_boot: int,
    rng: np.random.Generator,
    paired: bool,
) -> Optional[Dict[str, Any]]:
    if label not in conditions or baseline not in conditions:
        return None
    rows_c = conditions[label]["episodes"]
    rows_o = conditions[baseline]["episodes"]
    if paired:
        return _paired_diff(rows_c, rows_o, key, chunk, n_boot, rng)
    return _unpaired_diff(rows_c, rows_o, key, n_boot, rng)


def _check_pairing(
    conditions: Dict[str, Dict[str, Any]], baseline: str
) -> Tuple[bool, bool, List[str]]:
    notes: List[str] = []
    if baseline not in conditions:
        return False, False, [f"baseline {baseline} missing"]
    base_rows = conditions[baseline]["episodes"]
    base_eps = conditions[baseline]["summary"]["episodes"]
    paired = True
    for label, cond in conditions.items():
        if cond["summary"]["episodes"] != base_eps:
            paired = False
            notes.append(f"{label}: episodes {cond['summary']['episodes']} != {base_eps}")
            continue
        hashes = [r.get("init_state_sha1") for r in cond["episodes"]]
        base_hashes = [r.get("init_state_sha1") for r in base_rows]
        n = min(len(hashes), len(base_hashes))
        if n == 0 or any(h is None for h in hashes[:n]):
            paired = False
            notes.append(f"{label}: missing initial-state hashes")
        elif hashes[0] != base_hashes[0]:
            paired = False
            notes.append(
                f"{label}: episode-0 initial state differs from baseline "
                "(env seed mismatch)"
            )
        elif hashes[:n] != base_hashes[:n]:
            paired = False
            notes.append(
                f"{label}: initial states diverge after episode 0 because SC2 "
                "restarts in place; cross-condition pairing is unavailable, "
                "using unpaired bootstrap"
            )
    deterministic = True
    repeat = "eps_O_repeat"
    if repeat in conditions:
        a = np.array([r["return"] for r in conditions[baseline]["episodes"]])
        b = np.array([r["return"] for r in conditions[repeat]["episodes"]])
        n = min(a.size, b.size)
        if n == 0 or float(np.max(np.abs(a[:n] - b[:n]))) > 1e-6:
            deterministic = False
            notes.append("eps_O vs eps_O_repeat differ: episode pairing is not exact")
    return paired, deterministic, notes


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    return f"{value:.{digits}f}"


def _fmt_ci(diff: Optional[Dict[str, Any]], digits: int = 3) -> str:
    if not diff:
        return "-"
    ci = diff.get("ci95_chunk") or diff.get("ci95_episode")
    return f"{diff['delta']:+.{digits}f} [{ci[0]:+.{digits}f},{ci[1]:+.{digits}f}]"


def _eps_tag(eps: float) -> str:
    return f"{int(round(eps * 100)):03d}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--include", nargs="+", default=["eps", "replace"])
    ap.add_argument("--baseline", default="eps_O")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    ap.add_argument("--markdown", default="")
    args = ap.parse_args()

    conditions = _load(args.root, args.include)
    if not conditions:
        raise SystemExit(f"no conditions under {args.root}")
    paired, deterministic, notes = _check_pairing(conditions, args.baseline)
    use_paired = paired and deterministic
    if not deterministic:
        notes.append(
            "within-condition determinism check failed: eps_O vs eps_O_repeat "
            "differ; CIs treat episodes as independent draws"
        )
    rng = np.random.default_rng(args.seed)

    for label, cond in conditions.items():
        cond["stats"] = _condition_stats(cond["episodes"])

    comparisons: Dict[str, Dict[str, Any]] = {}
    for label in conditions:
        if label == args.baseline:
            continue
        entry = {}
        for key in ("return", "win"):
            diff = _compare(
                conditions, label, args.baseline, key,
                args.chunk, args.bootstrap, rng, use_paired,
            )
            if diff is not None:
                entry[key] = diff
        if entry:
            comparisons[label] = entry

    def ret(label: str) -> Optional[float]:
        cond = conditions.get(label)
        return cond["stats"]["return_mean"] if cond else None

    def win(label: str) -> Optional[float]:
        cond = conditions.get(label)
        return cond["stats"]["win_rate"] if cond else None

    # ---- eps grid at focal 0 ----
    eps_grid = []
    for eps in (0.05, 0.1, 0.2, 0.3):
        tag = _eps_tag(eps)
        labels = {
            "A": f"eps_A_e{tag}_f0",
            "B": f"eps_B_e{tag}_f0",
            "C": f"eps_C_e{tag}_f0",
        }
        if not any(label in conditions for label in labels.values()):
            continue
        row = {"eps": eps}
        for arm, label in labels.items():
            row[arm] = {
                "label": label,
                "return": ret(label),
                "win_rate": win(label),
                "delta_return": (
                    comparisons.get(label, {}).get("return")
                    if label in conditions else None
                ),
                "delta_win": (
                    comparisons.get(label, {}).get("win")
                    if label in conditions else None
                ),
            }
        a = row["A"]["return"]
        b = row["B"]["return"]
        c = row["C"]["return"]
        row["B_minus_A"] = (b - a) if a is not None and b is not None else None
        row["C_minus_A"] = (c - a) if a is not None and c is not None else None
        row["mixed_penalty"] = (
            min(ret(args.baseline), a) - b
            if a is not None and b is not None and ret(args.baseline) is not None
            else None
        )
        eps_grid.append(row)

    # ---- focal rotation at eps=0.1 ----
    rotation = []
    for focal in range(5):
        b_label = f"eps_B_e010_f{focal}"
        c_label = f"eps_C_e010_f{focal}"
        if b_label not in conditions and c_label not in conditions:
            continue
        rotation.append(
            {
                "focal": focal,
                "B": {
                    "label": b_label,
                    "return": ret(b_label),
                    "win_rate": win(b_label),
                    "delta_return": comparisons.get(b_label, {}).get("return"),
                    "delta_win": comparisons.get(b_label, {}).get("win"),
                },
                "C": {
                    "label": c_label,
                    "return": ret(c_label),
                    "win_rate": win(c_label),
                    "delta_return": comparisons.get(c_label, {}).get("return"),
                    "delta_win": comparisons.get(c_label, {}).get("win"),
                },
            }
        )

    # ---- natural replacement ----
    replace = []
    for source in ("muapr_s1", "muapr_s2", "muapr_s3", "pv_c_s1_250k"):
        labels = {
            "A": f"replace_{source}_A",
            "B": f"replace_{source}_B_f0",
            "C": f"replace_{source}_C_f0",
        }
        if not any(label in conditions for label in labels.values()):
            continue
        row = {"source": source}
        for arm, label in labels.items():
            row[arm] = {
                "label": label,
                "return": ret(label),
                "win_rate": win(label),
                "delta_return": comparisons.get(label, {}).get("return"),
                "delta_win": comparisons.get(label, {}).get("win"),
            }
        a, b = row["A"]["return"], row["B"]["return"]
        row["B_minus_A"] = (b - a) if a is not None and b is not None else None
        row["mixed_penalty"] = (
            min(ret(args.baseline), a) - b
            if a is not None and b is not None and ret(args.baseline) is not None
            else None
        )
        replace.append(row)

    # ---- verdict ----
    verdict: List[str] = []
    b01 = comparisons.get("eps_B_e010_f0", {}).get("return")
    c01 = comparisons.get("eps_C_e010_f0", {}).get("return")
    a01 = comparisons.get("eps_A_e010_f0", {}).get("return")
    if b01:
        ci = b01["ci95_chunk"] or b01["ci95_episode"]
        if ci[1] < 0:
            verdict.append(
                f"eps=0.1 focal 0: B (self fixed, teammates deviate) is "
                f"{b01['delta']:+.3f} return below O, CI [{ci[0]:+.3f},{ci[1]:+.3f}]."
            )
        elif ci[0] > 0:
            verdict.append(
                f"eps=0.1 focal 0: B is {b01['delta']:+.3f} above O, "
                f"CI [{ci[0]:+.3f},{ci[1]:+.3f}] (no drop)."
            )
        else:
            verdict.append(
                f"eps=0.1 focal 0: B delta {b01['delta']:+.3f}, "
                f"CI [{ci[0]:+.3f},{ci[1]:+.3f}] (not significant)."
            )
    if c01:
        ci = c01["ci95_chunk"] or c01["ci95_episode"]
        verdict.append(
            f"eps=0.1 focal 0: C (only self deviates) delta "
            f"{c01['delta']:+.3f}, CI [{ci[0]:+.3f},{ci[1]:+.3f}]."
        )
    if a01 and b01:
        verdict.append(
            f"eps=0.1 focal 0: R_B - R_A = "
            f"{b01['delta'] - a01['delta']:+.3f} (self deviation adds this much "
            f"on top of teammate deviation)."
        )
    neg_foci = 0
    for row in rotation:
        diff = row["B"].get("delta_return")
        if diff and (diff["ci95_chunk"] or diff["ci95_episode"])[1] < 0:
            neg_foci += 1
    if rotation:
        verdict.append(f"B is significantly below O at {neg_foci}/{len(rotation)} focal positions (eps=0.1).")

    payload = {
        "root": os.path.abspath(args.root),
        "baseline": args.baseline,
        "paired": paired,
        "deterministic": deterministic,
        "pairing_notes": notes,
        "conditions": {
            label: {
                "summary": cond["summary"],
                "stats": cond["stats"],
            }
            for label, cond in conditions.items()
        },
        "comparisons": comparisons,
        "eps_grid": eps_grid,
        "rotation": rotation,
        "replace": replace,
        "verdict": verdict,
    }

    print("\n=== conditions ===")
    for label, cond in sorted(conditions.items()):
        s = cond["stats"]
        print(
            f"{label:34s} n={s['n']:4d} ret={_fmt(s['return_mean']):>7s} "
            f"win={_fmt(s['win_rate']):>6s} "
            f"[{_fmt(s['win_ci95'][0])},{_fmt(s['win_ci95'][1])}] "
            f"focus={_fmt(s['focus_fire_rate_mean'])} "
            f"perturb={s['perturb_events_total']}"
        )

    print("\n=== eps grid focal 0 (delta return vs O) ===")
    for row in eps_grid:
        print(
            f"eps={row['eps']:.2f}  "
            f"A {_fmt(row['A']['return'])} ({_fmt_ci(row['A']['delta_return'])})  "
            f"B {_fmt(row['B']['return'])} ({_fmt_ci(row['B']['delta_return'])})  "
            f"C {_fmt(row['C']['return'])} ({_fmt_ci(row['C']['delta_return'])})"
        )

    print("\n=== focal rotation eps=0.1 ===")
    for row in rotation:
        print(
            f"focal={row['focal']}  "
            f"B {_fmt_ci(row['B']['delta_return'])}  "
            f"C {_fmt_ci(row['C']['delta_return'])}"
        )

    print("\n=== natural replacement ===")
    for row in replace:
        print(
            f"{row['source']:14s}  "
            f"A {_fmt(row['A']['return'])}  B {_fmt(row['B']['return'])} "
            f"({_fmt_ci(row['B']['delta_return'])})  C {_fmt(row['C']['return'])} "
            f"({_fmt_ci(row['C']['delta_return'])})"
        )

    print("\n=== verdict ===")
    for line in verdict:
        print("  " + line)
    if notes:
        print("\npairing notes:")
        for note in notes:
            print("  " + note)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.out}")

    if args.markdown:
        _write_markdown(args.markdown, payload)
        print(f"wrote {args.markdown}")
    return 0


def _write_markdown(path: str, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# 5m_vs_6m one-sided policy shift\n")
    mode = (
        "paired chunk bootstrap"
        if payload["paired"] and payload["deterministic"]
        else "unpaired bootstrap"
    )
    lines.append(
        f"Baseline: `{payload['baseline']}`; paired={payload['paired']}; "
        f"deterministic={payload['deterministic']}; CI mode: {mode}.\n"
    )
    if payload["pairing_notes"]:
        lines.append("Pairing notes:")
        for note in payload["pairing_notes"]:
            lines.append(f"- {note}")
        lines.append("")
    lines.append("## eps grid, focal 0\n")
    lines.append("| eps | A return | B return | C return | dA | dB | dC | B win | C win |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["eps_grid"]:
        lines.append(
            f"| {row['eps']:.2f} | {_fmt(row['A']['return'])} | "
            f"{_fmt(row['B']['return'])} | {_fmt(row['C']['return'])} | "
            f"{_fmt_ci(row['A']['delta_return'])} | {_fmt_ci(row['B']['delta_return'])} | "
            f"{_fmt_ci(row['C']['delta_return'])} | {_fmt(row['B']['win_rate'])} | "
            f"{_fmt(row['C']['win_rate'])} |"
        )
    lines.append("\n## focal rotation, eps=0.1\n")
    lines.append("| focal | B return | dB | C return | dC | B win | C win |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["rotation"]:
        lines.append(
            f"| {row['focal']} | {_fmt(row['B']['return'])} | "
            f"{_fmt_ci(row['B']['delta_return'])} | {_fmt(row['C']['return'])} | "
            f"{_fmt_ci(row['C']['delta_return'])} | {_fmt(row['B']['win_rate'])} | "
            f"{_fmt(row['C']['win_rate'])} |"
        )
    lines.append("\n## natural replacement\n")
    lines.append("| source | A return | B return | dB | C return | dC | A win | B win |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["replace"]:
        lines.append(
            f"| {row['source']} | {_fmt(row['A']['return'])} | "
            f"{_fmt(row['B']['return'])} | {_fmt_ci(row['B']['delta_return'])} | "
            f"{_fmt(row['C']['return'])} | {_fmt_ci(row['C']['delta_return'])} | "
            f"{_fmt(row['A']['win_rate'])} | {_fmt(row['B']['win_rate'])} |"
        )
    lines.append("\n## reading\n")
    for line in payload["verdict"]:
        lines.append(f"- {line}")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
