#!/usr/bin/env python3
"""Stage A analysis: does teammate agreement predict where the OVM floor helps?

For every record file written by scripts/run_tcvm_premise.py this computes

    uplift_t = [q_off_t - y_t]_+                      (what the floor adds)
    gain_t   = |y_t - G_t| - |max(q_off_t, y_t) - G_t| (floor moves target
                                                        toward the realised
                                                        return when positive)

and asks whether a per-sample signal separates gain_t > 0 from gain_t < 0
*within* a policy version, i.e. beyond what a global time schedule can express.
Candidate signals are built from the per-agent agreement between the executed
joint action and the frozen offline network, including forward windows over the
next H steps of the same episode, since the floor is a claim about the
continuation.

Usage:
  python scripts/analyze_tcvm_premise.py --dir checkpoints/tcvm_premise/5m_vs_6m
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # keep the script usable without sklearn
    roc_auc_score = None


def forward_window(values: np.ndarray, episode: np.ndarray, t: np.ndarray, horizon: int) -> np.ndarray:
    """Mean of `values` over steps [t, t+H) inside the same episode."""
    order = np.lexsort((t, episode))
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    v = values[order]
    ep = episode[order]
    out = np.empty_like(v)
    start = 0
    n = v.size
    while start < n:
        stop = start
        while stop < n and ep[stop] == ep[start]:
            stop += 1
        seg = v[start:stop]
        csum = np.concatenate([[0.0], np.cumsum(seg)])
        length = seg.size
        idx = np.arange(length)
        end = np.minimum(idx + horizon, length)
        out[start:stop] = (csum[end] - csum[idx]) / np.maximum(end - idx, 1)
        start = stop
    return out[inv]


def build_signals(rec: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    ep, t = rec["episode"], rec["t"]
    agree = rec["off_greedy"].mean(axis=1)
    dev_margin = rec["off_margin"].mean(axis=1)
    dev_logp = -rec["off_logp"].mean(axis=1)
    on_dev = rec["on_margin"].mean(axis=1)
    signals = {
        # teammate / joint agreement with the frozen offline policy (H = 1)
        "agree_h1": agree,
        "negdev_margin_h1": -dev_margin,
        "negdev_logp_h1": -dev_logp,
        # forward-window versions: the floor is a claim about the continuation
        "agree_h5": forward_window(agree, ep, t, 5),
        "agree_h10": forward_window(agree, ep, t, 10),
        "negdev_margin_h5": -forward_window(dev_margin, ep, t, 5),
        "negdev_margin_h10": -forward_window(dev_margin, ep, t, 10),
        "negdev_logp_h10": -forward_window(dev_logp, ep, t, 10),
        "agree_rest": forward_window(agree, ep, t, 10**6),
        "negdev_margin_rest": -forward_window(dev_margin, ep, t, 10**6),
        # comparators that are not teammate-conditioned
        "neg_on_dev_h1": -on_dev,
        "neg_on_dev_h10": -forward_window(on_dev, ep, t, 10),
        "neg_phase": -(t.astype(np.float64) / np.maximum(rec["length"], 1)),
        "q_off_level": rec["q_off"].astype(np.float64),
        "neg_uplift": -np.maximum(rec["q_off"] - rec["y"], 0.0),
    }
    return signals


def gated_gain(gain: np.ndarray, signal: np.ndarray, frac: float, active: np.ndarray | None = None) -> float:
    """Mean gain when the floor is applied only to the top `frac` of the signal.

    The budget is spent among `active` samples only: where the floor does not
    fire the gate is irrelevant and gain is already zero, so letting those
    samples consume the quota understates every signal equally.
    """
    if frac >= 1.0:
        return float(np.mean(gain))
    pool = np.flatnonzero(active) if active is not None else np.arange(gain.size)
    if pool.size == 0:
        return float(np.mean(gain))
    k = max(1, int(round(pool.size * frac)))
    sub = signal[pool]
    thresh = np.partition(sub, -k)[-k]
    keep = np.zeros(gain.size, dtype=bool)
    keep[pool[sub >= thresh]] = True
    return float(np.mean(np.where(keep, gain, 0.0)))


def cluster_bootstrap(gain: np.ndarray, keep: np.ndarray, episode: np.ndarray, iters: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    eps = np.unique(episode)
    index = {e: np.flatnonzero(episode == e) for e in eps}
    draws = np.empty(iters)
    for i in range(iters):
        pick = rng.choice(eps, size=eps.size, replace=True)
        rows = np.concatenate([index[e] for e in pick])
        draws[i] = np.mean(np.where(keep[rows], gain[rows], 0.0))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def analyse(path: str, bins: int, boot: int) -> Dict:
    rec = {key: value for key, value in np.load(path).items()}
    y, q_off, g = rec["y"].astype(np.float64), rec["q_off"].astype(np.float64), rec["g"].astype(np.float64)
    uplift = np.maximum(q_off - y, 0.0)
    target = np.maximum(q_off, y)
    gain = np.abs(y - g) - np.abs(target - g)
    active = uplift > 0
    signals = build_signals(rec)
    tag = os.path.splitext(os.path.basename(path))[0]

    out: Dict = {
        "tag": tag,
        "samples": int(y.size),
        "episodes": int(np.unique(rec["episode"]).size),
        "mean_y_bias": float(np.mean(y - g)),
        "mean_q_off_bias": float(np.mean(q_off - g)),
        "floor_active_rate": float(np.mean(active)),
        "mean_uplift_active": float(np.mean(uplift[active])) if active.any() else 0.0,
        "mean_gain_all": float(np.mean(gain)),
        "mean_gain_active": float(np.mean(gain[active])) if active.any() else 0.0,
        "pos_gain_rate_active": float(np.mean(gain[active] > 0)) if active.any() else 0.0,
        "signals": {},
        "raw_signal": {},
        "_gain": gain,
        "_active": active,
        "_episode": rec["episode"],
    }

    label = (gain > 0).astype(np.int32)
    for name, signal in signals.items():
        signal = signal.astype(np.float64)
        row: Dict = {}
        if roc_auc_score is not None and active.sum() > 10 and label[active].min() != label[active].max():
            row["auc_active"] = float(roc_auc_score(label[active], signal[active]))
        if roc_auc_score is not None and label.min() != label.max():
            row["auc_all"] = float(roc_auc_score(label, signal))
        row["gated_gain"] = {
            f"top{int(frac * 100)}": gated_gain(gain, signal, frac, active)
            for frac in (0.25, 0.5, 0.75, 1.0)
        }
        out["raw_signal"][name] = signal
        edges = np.quantile(signal, np.linspace(0, 1, bins + 1))
        table = []
        for b in range(bins):
            lo, hi = edges[b], edges[b + 1]
            sel = (signal >= lo) & (signal <= hi) if b == bins - 1 else (signal >= lo) & (signal < hi)
            if sel.sum() == 0:
                continue
            table.append(
                {
                    "bin": b,
                    "lo": float(lo),
                    "hi": float(hi),
                    "n": int(sel.sum()),
                    "mean_gain": float(np.mean(gain[sel])),
                    "floor_active_rate": float(np.mean(active[sel])),
                    "mean_uplift": float(np.mean(uplift[sel])),
                    "mean_y_bias": float(np.mean((y - g)[sel])),
                    "mean_q_off_bias": float(np.mean((q_off - g)[sel])),
                }
            )
        row["bins"] = table
        out["signals"][name] = row

    best = max(
        out["signals"].items(),
        key=lambda item: item[1]["gated_gain"]["top50"],
    )
    out["best_signal_top50"] = best[0]
    if boot > 0:
        signal = signals[best[0]].astype(np.float64)
        k = max(1, int(round(gain.size * 0.5)))
        thresh = np.partition(signal, -k)[-k]
        keep = signal >= thresh
        lo, hi = cluster_bootstrap(gain, keep, rec["episode"], iters=boot)
        out["best_top50_ci"] = [lo, hi]
        lo0, hi0 = cluster_bootstrap(gain, np.ones_like(keep), rec["episode"], iters=boot)
        out["ungated_ci"] = [lo0, hi0]
    return out


def threshold_sweep(results: List[Dict], name: str, grid: np.ndarray) -> List[Dict]:
    """One shared threshold across all policy versions.

    Signals built from the frozen offline network are version-invariant (the
    network is identical in every version), so a single theta is implementable
    as a real gate. The pass rate is then free to vary per version, which is
    what a top-k budget cannot express.
    """
    rows = []
    for theta in grid:
        per_version = []
        for res in results:
            signal = res["raw_signal"][name]
            gain, active = res["_gain"], res["_active"]
            keep = (signal >= theta) & active
            per_version.append(
                {
                    "tag": res["tag"],
                    "pass_rate_active": float(keep.sum() / max(active.sum(), 1)),
                    "gated_gain": float(np.mean(np.where(keep, gain, 0.0))),
                    "ungated_gain": res["mean_gain_all"],
                }
            )
        delta = [v["gated_gain"] - v["ungated_gain"] for v in per_version]
        rows.append(
            {
                "theta": float(theta),
                "mean_delta": float(np.mean(delta)),
                "min_delta": float(np.min(delta)),
                "n_worse": int(sum(d < -1e-9 for d in delta)),
                "mean_pass_rate": float(np.mean([v["pass_rate_active"] for v in per_version])),
                "per_version": per_version,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="checkpoints/tcvm_premise/5m_vs_6m")
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "*.npz")))
    if not paths:
        raise SystemExit(f"no record files in {args.dir}")
    results = [analyse(path, args.bins, args.boot) for path in paths]

    print(f"{'tag':34s} {'n':>6s} {'y-G':>7s} {'qoff-G':>8s} {'act%':>6s} {'gain':>7s} {'gainA':>7s}")
    for res in results:
        print(
            f"{res['tag']:34s} {res['samples']:6d} {res['mean_y_bias']:7.3f} "
            f"{res['mean_q_off_bias']:8.3f} {100 * res['floor_active_rate']:6.1f} "
            f"{res['mean_gain_all']:7.3f} {res['mean_gain_active']:7.3f}"
        )

    key_signals = [
        "agree_h1",
        "agree_h10",
        "agree_rest",
        "negdev_margin_h10",
        "negdev_logp_h10",
        "neg_on_dev_h10",
        "neg_phase",
    ]
    print(f"\nAUC (gain>0 | floor active) and gated gain at top-50%")
    header = f"{'tag':28s}" + "".join(f"{name[:13]:>14s}" for name in key_signals)
    print(header)
    for res in results:
        cells = ""
        for name in key_signals:
            row = res["signals"].get(name, {})
            auc = row.get("auc_active")
            cells += f"{auc:14.3f}" if auc is not None else f"{'-':>14s}"
        print(f"{res['tag']:28s}{cells}")
    print()
    for res in results:
        cells = ""
        for name in key_signals:
            row = res["signals"].get(name, {})
            cells += f"{row['gated_gain']['top50']:14.3f}" if row else f"{'-':>14s}"
        print(f"{res['tag']:28s}{cells}  ungated={res['mean_gain_all']:.3f}")

    if args.out:
        online = [r for r in results if not r["tag"].startswith("offlinedata")]
        print("\n=== shared absolute threshold (online versions only) ===")
        sweep_out = {}
        for name in ("agree_h1", "agree_h5", "agree_h10", "negdev_margin_h10", "negdev_logp_h10"):
            pooled = np.concatenate([r["raw_signal"][name][r["_active"]] for r in online])
            grid = np.unique(np.quantile(pooled, np.linspace(0.0, 0.95, 20)))
            rows = threshold_sweep(online, name, grid)
            feasible = [r for r in rows if r["n_worse"] == 0]
            best = max(feasible or rows, key=lambda r: r["mean_delta"])
            sweep_out[name] = {"rows": rows, "best": best}
            print(
                f"{name:20s} theta={best['theta']:8.4f} mean_delta={best['mean_delta']:+.3f} "
                f"min_delta={best['min_delta']:+.3f} n_worse={best['n_worse']} "
                f"pass={100 * best['mean_pass_rate']:.1f}%"
            )
            for v in best["per_version"]:
                print(
                    f"    {v['tag']:30s} pass={100 * v['pass_rate_active']:5.1f}%  "
                    f"gated={v['gated_gain']:+7.3f}  ungated={v['ungated_gain']:+7.3f}  "
                    f"delta={v['gated_gain'] - v['ungated_gain']:+7.3f}"
                )

        for res in results:
            for key in ("_gain", "_active", "_episode", "raw_signal"):
                res.pop(key, None)
        with open(args.out, "w") as handle:
            json.dump({"versions": results, "threshold_sweep": sweep_out}, handle, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
