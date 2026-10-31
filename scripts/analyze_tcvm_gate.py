#!/usr/bin/env python3
"""Stage A decision test: does a teammate-conditioned gate beat a global one?

For every per-step record written by scripts/run_tcvm_premise.py we can evaluate
the *exact* objective the mechanism optimises. With a gate g_t in [0, 1] the
gated offline value floor sets

    y_eff,t = y_t + g_t * [q_off,t - y_t]_+

so its effect on target accuracy against the realised return-to-go G_t is

    gain(g)_t = |y_t - G_t| - |y_eff,t - G_t|

which is exactly computable, positive when the floor moves the target toward the
realised return and negative when it overshoots. g == 0 recovers plain TD and
gain == 0; g == 1 recovers ungated OVM.

The question is not "is the floor good" (a global lambda already answers that)
but "does conditioning g on teammate behaviour beat any global setting of g".
So every candidate signal is scored on the frontier of

    mean gain   versus   mean gate mass  (the budget actually spent)

against a CONSTANT gate at the same mass. A constant gate is precisely what a
tuned global lambda can express, so dominating it is the premise the mechanism
needs. Teammate signals are read off the FROZEN offline network only, so they
are comparable across policy versions and implementable as a single threshold.

Usage:
  python scripts/analyze_tcvm_gate.py --dir checkpoints/tcvm_premise/5m_vs_6m
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np


def forward_window(values: np.ndarray, episode: np.ndarray, t: np.ndarray, horizon: int) -> np.ndarray:
    """Mean of `values` over steps [t, t+H) inside the same episode."""
    order = np.lexsort((t, episode))
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    v = values[order].astype(np.float64)
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


def load_version(path: str) -> Dict:
    rec = {key: value for key, value in np.load(path).items()}
    ep, t = rec["episode"], rec["t"]
    y = rec["y"].astype(np.float64)
    q_off = rec["q_off"].astype(np.float64)
    g_ret = rec["g"].astype(np.float64)
    uplift = np.maximum(q_off - y, 0.0)

    # Per-step teammate deviation from the frozen offline policy, averaged over
    # agents. Higher = the executed joint action is further from what the old
    # team would have done, so the offline value claim is less likely to hold.
    dev_disagree = 1.0 - rec["off_greedy"].mean(axis=1).astype(np.float64)
    dev_margin = rec["off_margin"].mean(axis=1).astype(np.float64)
    dev_logp = -rec["off_logp"].mean(axis=1).astype(np.float64)
    dev_on = rec["on_margin"].mean(axis=1).astype(np.float64)

    deviations = {
        "tm_disagree_h1": dev_disagree,
        "tm_disagree_h5": forward_window(dev_disagree, ep, t, 5),
        "tm_disagree_h10": forward_window(dev_disagree, ep, t, 10),
        "tm_margin_h1": dev_margin,
        "tm_margin_h5": forward_window(dev_margin, ep, t, 5),
        "tm_margin_h10": forward_window(dev_margin, ep, t, 10),
        "tm_logp_h1": dev_logp,
        "tm_logp_h10": forward_window(dev_logp, ep, t, 10),
        # non-teammate comparators: self-drift of the live online net, and the
        # within-episode phase, which a schedule could also express
        "cmp_online_h1": dev_on,
        "cmp_online_h10": forward_window(dev_on, ep, t, 10),
        "cmp_phase": t.astype(np.float64) / np.maximum(rec["length"], 1),
    }
    return {
        "tag": os.path.splitext(os.path.basename(path))[0],
        "y": y,
        "q_off": q_off,
        "g_ret": g_ret,
        "uplift": uplift,
        "episode": ep,
        "deviations": deviations,
    }


MODE = "floor"


def gain_of(ver: Dict, gate: np.ndarray) -> np.ndarray:
    """Per-step target-accuracy gain of applying the offline value with mass `gate`.

    floor: y_eff = y + g * [q_off - y]_+   (one-sided, can only raise the target)
    blend: y_eff = (1 - g) * y + g * q_off (two-sided memory anchor)
    """
    y, g_ret = ver["y"], ver["g_ret"]
    if MODE == "blend":
        eff = y + gate * (ver["q_off"] - y)
    else:
        eff = y + gate * ver["uplift"]
    return np.abs(y - g_ret) - np.abs(eff - g_ret)


def active_mask(ver: Dict) -> np.ndarray:
    """Steps where the offline value actually moves the target."""
    if MODE == "blend":
        return np.abs(ver["q_off"] - ver["y"]) > 0
    return ver["uplift"] > 0


def soft_gate(dev: np.ndarray, theta: float, temp: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((dev - theta) / max(temp, 1e-6)))


def mass_and_gain(ver: Dict, gate: np.ndarray) -> Tuple[float, float]:
    """Gate mass is averaged over steps where the offline value would actually move the target."""
    active = active_mask(ver)
    mass = float(gate[active].mean()) if active.any() else 0.0
    return mass, float(gain_of(ver, gate).mean())


def frontier(versions: List[Dict], name: str, temps: List[float], n_theta: int) -> List[Dict]:
    pooled = np.concatenate([v["deviations"][name][active_mask(v)] for v in versions])
    thetas = np.unique(np.quantile(pooled, np.linspace(0.02, 0.98, n_theta)))
    rows = []
    for temp_frac in temps:
        scale = float(np.std(pooled)) * temp_frac
        for theta in thetas:
            per = [mass_and_gain(v, soft_gate(v["deviations"][name], theta, scale)) for v in versions]
            rows.append(
                {
                    "theta": float(theta),
                    "temp": scale,
                    "mass": float(np.mean([p[0] for p in per])),
                    "gain": float(np.mean([p[1] for p in per])),
                    "min_gain": float(np.min([p[1] for p in per])),
                    "per_version": [
                        {"tag": v["tag"], "mass": p[0], "gain": p[1]}
                        for v, p in zip(versions, per)
                    ],
                }
            )
    return rows


def constant_curve(versions: List[Dict], n: int) -> List[Dict]:
    rows = []
    for c in np.linspace(0.0, 1.0, n):
        per = [mass_and_gain(v, np.full_like(v["y"], c)) for v in versions]
        rows.append(
            {
                "mass": float(c),
                "gain": float(np.mean([p[1] for p in per])),
                "min_gain": float(np.min([p[1] for p in per])),
                "per_version": [
                    {"tag": v["tag"], "mass": p[0], "gain": p[1]}
                    for v, p in zip(versions, per)
                ],
            }
        )
    return rows


def interp_constant(const_rows: List[Dict], mass: float) -> float:
    xs = np.array([r["mass"] for r in const_rows])
    ys = np.array([r["gain"] for r in const_rows])
    return float(np.interp(mass, xs, ys))


def best_at_mass(rows: List[Dict], lo: float, hi: float) -> Dict | None:
    band = [r for r in rows if lo <= r["mass"] <= hi]
    return max(band, key=lambda r: r["gain"]) if band else None


def cluster_bootstrap_delta(
    versions: List[Dict], name: str, theta: float, temp: float, const: float, iters: int, seed: int
) -> Tuple[float, float, float]:
    """CI of (teammate gate gain - constant gate gain), episodes resampled."""
    rng = np.random.default_rng(seed)
    per_ver = []
    for ver in versions:
        gate = soft_gate(ver["deviations"][name], theta, temp)
        diff = gain_of(ver, gate) - gain_of(ver, np.full_like(ver["y"], const))
        eps = np.unique(ver["episode"])
        index = [np.flatnonzero(ver["episode"] == e) for e in eps]
        per_ver.append((diff, index))
    draws = np.empty(iters)
    for i in range(iters):
        vals = []
        for diff, index in per_ver:
            pick = rng.integers(0, len(index), size=len(index))
            rows = np.concatenate([index[j] for j in pick])
            vals.append(diff[rows].mean())
        draws[i] = float(np.mean(vals))
    point = float(np.mean([d.mean() for d, _ in per_ver]))
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="checkpoints/tcvm_premise/5m_vs_6m")
    parser.add_argument("--out", default="")
    parser.add_argument("--n-theta", type=int, default=25)
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--band", type=float, nargs=2, default=(0.3, 0.7))
    parser.add_argument("--mode", default="floor", choices=["floor", "blend"])
    args = parser.parse_args()

    global MODE
    MODE = args.mode

    paths = [
        p
        for p in sorted(glob.glob(os.path.join(args.dir, "*.npz")))
        if not os.path.basename(p).startswith("offlinedata")
    ]
    if not paths:
        raise SystemExit(f"no online record files in {args.dir}")
    versions = [load_version(p) for p in paths]

    print(f"per-version accounting, mode={args.mode} (gate == 1 is the ungated mechanism)")
    print(f"{'tag':24s} {'n':>6s} {'act%':>6s} {'gain@1':>8s} {'gain@.5':>8s} {'oracle':>8s}")
    for ver in versions:
        active = active_mask(ver)
        ones = np.ones_like(ver["y"])
        oracle = np.maximum(gain_of(ver, ones), 0.0).mean()
        print(
            f"{ver['tag']:24s} {ver['y'].size:6d} {100 * active.mean():6.1f} "
            f"{gain_of(ver, ones).mean():8.3f} {gain_of(ver, 0.5 * ones).mean():8.3f} {oracle:8.3f}"
        )

    const_rows = constant_curve(versions, 41)
    best_const = max(const_rows, key=lambda r: r["gain"])
    print(
        f"\nbest CONSTANT gate (what a tuned global lambda can express): "
        f"c={best_const['mass']:.3f} gain={best_const['gain']:+.4f} "
        f"min_version={best_const['min_gain']:+.4f}"
    )

    names = [
        "tm_disagree_h1",
        "tm_disagree_h5",
        "tm_disagree_h10",
        "tm_margin_h1",
        "tm_margin_h5",
        "tm_margin_h10",
        "tm_logp_h1",
        "tm_logp_h10",
        "cmp_online_h1",
        "cmp_online_h10",
        "cmp_phase",
    ]
    temps = [0.1, 0.25, 0.5, 1.0]
    lo, hi = args.band

    results: Dict[str, Dict] = {}
    print(f"\ngate frontier, best point with mean mass in [{lo}, {hi}]")
    print(f"{'signal':18s} {'mass':>6s} {'gain':>8s} {'const@mass':>11s} {'delta':>8s} {'minver':>8s}")
    for name in names:
        rows = frontier(versions, name, temps, args.n_theta)
        best = best_at_mass(rows, lo, hi)
        if best is None:
            continue
        const_at = interp_constant(const_rows, best["mass"])
        results[name] = {
            "best": best,
            "const_at_mass": const_at,
            "delta_vs_const": best["gain"] - const_at,
        }
        print(
            f"{name:18s} {best['mass']:6.3f} {best['gain']:+8.4f} {const_at:+11.4f} "
            f"{best['gain'] - const_at:+8.4f} {best['min_gain']:+8.4f}"
        )

    teammate = {k: v for k, v in results.items() if k.startswith("tm_")}
    champion = max(teammate, key=lambda k: teammate[k]["delta_vs_const"])
    best = teammate[champion]["best"]
    print(f"\nchampion teammate signal: {champion}")
    print("  per-version gain vs constant gate at that version's own mass:")
    wins = 0
    for row in best["per_version"]:
        c_at = interp_constant(
            [
                {"mass": r["mass"], "gain": next(p["gain"] for p in r["per_version"] if p["tag"] == row["tag"])}
                for r in const_rows
            ],
            row["mass"],
        )
        wins += int(row["gain"] > c_at)
        print(
            f"    {row['tag']:24s} mass={row['mass']:5.3f} gain={row['gain']:+7.3f} "
            f"const={c_at:+7.3f} delta={row['gain'] - c_at:+7.3f}"
        )
    print(f"  versions where the teammate gate wins: {wins}/{len(best['per_version'])}")

    if args.boot > 0:
        point, blo, bhi = cluster_bootstrap_delta(
            versions, champion, best["theta"], best["temp"], best["mass"], args.boot, 0
        )
        print(
            f"\n{champion} minus constant gate at matched mass {best['mass']:.3f}: "
            f"{point:+.4f}  95% CI [{blo:+.4f}, {bhi:+.4f}]  (episode-clustered, {args.boot} draws)"
        )
        results[champion]["bootstrap"] = {"point": point, "ci": [blo, bhi], "iters": args.boot}

    if args.out:
        payload = {
            "mode": args.mode,
            "versions": [
                {
                    "tag": v["tag"],
                    "samples": int(v["y"].size),
                    "floor_active_rate": float(active_mask(v).mean()),
                    "gain_gate1": float(gain_of(v, np.ones_like(v["y"])).mean()),
                    "gain_oracle": float(np.maximum(gain_of(v, np.ones_like(v["y"])), 0.0).mean()),
                }
                for v in versions
            ],
            "constant_curve": [{k: r[k] for k in ("mass", "gain", "min_gain")} for r in const_rows],
            "best_constant": {k: best_const[k] for k in ("mass", "gain", "min_gain")},
            "signals": {
                name: {
                    "theta": res["best"]["theta"],
                    "temp": res["best"]["temp"],
                    "mass": res["best"]["mass"],
                    "gain": res["best"]["gain"],
                    "min_gain": res["best"]["min_gain"],
                    "const_at_mass": res["const_at_mass"],
                    "delta_vs_const": res["delta_vs_const"],
                    "per_version": res["best"]["per_version"],
                    "bootstrap": res.get("bootstrap"),
                }
                for name, res in results.items()
            },
            "champion": champion,
        }
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
