#!/usr/bin/env python3
"""E2/E3 read-off: held-out loss of a pre-decision benefit model, and whether a
selector has any room over the best fixed arm.

Inputs are adaptive-reuse pilot JSONs (scripts/run_adaptive_reuse_pilot.py). Each
record is one conflict episode with a randomized local intervention:
  D  Bernoulli(p) treatment indicator,
  G  return from the conflict onward,
  Y  inverse-probability outcome D*G/p - (1-D)*G/(1-p),
  X  features available before the decision.

E2 is scored as held-out loss, never as accuracy:
  * a T-learner (one regressor per arm) predicts G for held-out records; its MSE
    and MAE are compared with the constant arm-mean baseline;
  * a response model predicts Y; a logistic model predicts sign(Y>0). Both are
    compared with their constant/prior baselines by MSE/MAE and log-loss.
  Two protocols: 5-fold within checkpoint, and leave-one-checkpoint-out.

E3 compares policies with a cross-fitted AIPW estimator on the same records:
  never / always / sign(mu1-mu0) gate / plug-in oracle.
  best fixed = the better of {never, always}. The selection space is
  AIPW(gate) - best fixed; the plug-in oracle gap mean(max(mu1,mu0)) -
  max(mean mu1, mean mu0) is reported as a descriptive upper bound.

Usage:
  python scripts/analyze_e2_e3_local_benefit.py \
    --pilot C_s1_250k=checkpoints/e1_e7_supplement/pilots/C_s1_250k/pilot.json \
    --pilot C_s2_250k=... --pilot C_s3_250k=... --pilot C_s1_1M=... \
    --out checkpoints/e1_e7_supplement/e2_e3_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_verified_prefs import FEATURE_NAMES


def parse_pilot(specs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for spec in specs:
        label, _, path = spec.partition("=")
        if not label or not path:
            raise SystemExit(f"bad --pilot '{spec}' (expected label=path)")
        if not os.path.isfile(path):
            raise SystemExit(f"pilot JSON missing: {path}")
        out[label] = path
    return out


def load_records(pilots: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                 np.ndarray, np.ndarray]:
    X, D, G, Y, group = [], [], [], [], []
    for label, path in pilots.items():
        with open(path) as handle:
            payload = json.load(handle)
        for rec in payload["records"]:
            X.append([rec[k] for k in FEATURE_NAMES])
            D.append(int(rec["D"]))
            G.append(float(rec["G"]))
            Y.append(float(rec["Y"]))
            group.append(label)
    return (np.asarray(X, dtype=np.float64), np.asarray(D, dtype=int),
            np.asarray(G, dtype=np.float64), np.asarray(Y, dtype=np.float64),
            np.asarray(group))


def make_model(name: str):
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_depth=3, max_iter=120, learning_rate=0.05, min_samples_leaf=10,
            l2_regularization=1.0, early_stopping=False, random_state=0,
        )
    raise SystemExit(f"unknown model {name}")


def fit_arm(name: str, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
            fallback: float) -> np.ndarray:
    if Xtr.shape[0] < 8 or np.unique(ytr).size < 2:
        return np.full(Xte.shape[0], fallback)
    model = make_model(name)
    if name == "ridge":
        scaler = StandardScaler().fit(Xtr)
        model.fit(scaler.transform(Xtr), ytr)
        return model.predict(scaler.transform(Xte))
    model.fit(Xtr, ytr)
    return model.predict(Xte)


def folds_for(scheme: str, group: np.ndarray, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(n)
    if scheme == "loco":
        out = []
        for g in sorted(set(group.tolist())):
            test = idx[group == g]
            train = idx[group != g]
            if train.size and test.size:
                out.append((train, test))
        return out
    out = []
    for g in sorted(set(group.tolist())):
        rows = idx[group == g]
        kf = KFold(n_splits=min(5, rows.size), shuffle=True, random_state=0)
        for tr, te in kf.split(rows):
            out.append((rows[tr], rows[te]))
    return out


def cross_fit(scheme: str, model: str, X: np.ndarray, D: np.ndarray,
              G: np.ndarray, Y: np.ndarray, group: np.ndarray) -> Dict[str, np.ndarray]:
    n = X.shape[0]
    out = {k: np.full(n, np.nan) for k in
           ("mu1", "mu0", "c1", "c0", "yhat", "yc")}
    for tr, te in folds_for(scheme, group, n):
        for arm, key_mu, key_c in ((1, "mu1", "c1"), (0, "mu0", "c0")):
            sel = tr[D[tr] == arm]
            if sel.size == 0:
                sel = tr
            fallback = float(G[sel].mean()) if sel.size else float(G.mean())
            out[key_mu][te] = fit_arm(model, X[sel], G[sel], X[te], fallback)
            out[key_c][te] = fallback
        # response model on the IPW outcome, irrespective of arm
        out["yhat"][te] = fit_arm(model, X[tr], Y[tr], X[te], float(Y[tr].mean()))
        out["yc"][te] = float(Y[tr].mean())
    return out


def losses(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    err = pred - target
    return {"mse": float(np.mean(err ** 2)), "mae": float(np.mean(np.abs(err)))}


def sign_logloss(X: np.ndarray, label: np.ndarray, group: np.ndarray,
                 scheme: str) -> Tuple[float, float]:
    n = X.shape[0]
    pred = np.full(n, np.nan)
    base = np.full(n, np.nan)
    for tr, te in folds_for(scheme, group, n):
        prior = float(label[tr].mean())
        base[te] = prior
        if np.unique(label[tr]).size < 2 or tr.size < 16:
            pred[te] = prior
            continue
        scaler = StandardScaler().fit(X[tr])
        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(scaler.transform(X[tr]), label[tr])
        pred[te] = model.predict_proba(scaler.transform(X[te]))[:, 1]
    eps = 1e-6
    pred = np.clip(pred, eps, 1 - eps)
    base = np.clip(base, eps, 1 - eps)
    ll = -np.mean(label * np.log(pred) + (1 - label) * np.log(1 - pred))
    bl = -np.mean(label * np.log(base) + (1 - label) * np.log(1 - base))
    return float(ll), float(bl)


def aipw(mu1: np.ndarray, mu0: np.ndarray, D: np.ndarray, G: np.ndarray,
         policy: np.ndarray, p: float = 0.5) -> np.ndarray:
    m = np.where(policy == 1, mu1, mu0)
    w = np.where(policy == 1, D / p, (1 - D) / (1 - p))
    return m + w * (G - m)


def boot_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def arm_diff(G: np.ndarray, D: np.ndarray) -> Dict[str, float]:
    t = G[D == 1]
    c = G[D == 0]
    diff = float(t.mean() - c.mean())
    rng = np.random.default_rng(3)
    boot = np.empty(2000)
    for b in range(2000):
        boot[b] = (t[rng.integers(0, t.size, t.size)].mean()
                   - c[rng.integers(0, c.size, c.size)].mean())
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"n_treated": int(t.size), "n_control": int(c.size),
            "mean_treated": float(t.mean()), "mean_control": float(c.mean()),
            "diff": diff, "ci": [float(lo), float(hi)]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", action="append", required=True, help="label=path")
    ap.add_argument("--models", nargs="+", default=["ridge", "hgb"])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pilots = parse_pilot(args.pilot)
    X, D, G, Y, group = load_records(pilots)
    print(f"records={X.shape[0]} checkpoints={len(pilots)} features={X.shape[1]}")
    for label in pilots:
        sel = group == label
        print(f"  {label:12s} n={int(sel.sum()):4d} treated={int(D[sel].sum()):4d} "
              f"mean_G={G[sel].mean():.3f} arm_diff={arm_diff(G[sel], D[sel])['diff']:+.3f}")

    payload: Dict = {
        "n_records": int(X.shape[0]),
        "pilots": pilots,
        "feature_names": list(FEATURE_NAMES),
        "arm_diff_by_checkpoint": {label: arm_diff(G[group == label], D[group == label])
                                   for label in pilots},
        "losses": {},
        "policy": {},
        "verdict": [],
    }
    pooled_diff = arm_diff(G, D)
    payload["arm_diff_pooled"] = pooled_diff
    print(f"\npooled randomized arm diff (treated-control): {pooled_diff['diff']:+.3f} "
          f"[{pooled_diff['ci'][0]:+.3f},{pooled_diff['ci'][1]:+.3f}]")

    for scheme in ("within", "loco"):
        if scheme == "loco" and len(set(group.tolist())) < 2:
            print("\n[loco] skipped: needs at least two checkpoints")
            continue
        for model in args.models:
            cf = cross_fit(scheme, model, X, D, G, Y, group)
            mask1, mask0 = D == 1, D == 0
            l1 = losses(cf["mu1"][mask1], G[mask1])
            l0 = losses(cf["mu0"][mask0], G[mask0])
            c1 = losses(cf["c1"][mask1], G[mask1])
            c0 = losses(cf["c0"][mask0], G[mask0])
            w1 = mask1.mean()
            model_mse = w1 * l1["mse"] + (1 - w1) * l0["mse"]
            const_mse = w1 * c1["mse"] + (1 - w1) * c0["mse"]
            model_mae = w1 * l1["mae"] + (1 - w1) * l0["mae"]
            const_mae = w1 * c1["mae"] + (1 - w1) * c0["mae"]
            ly = losses(cf["yhat"], Y)
            cy = losses(cf["yc"], Y)
            label = f"{scheme}/{model}"
            ll, bl = sign_logloss(X, (Y > 0).astype(int), group, scheme)
            payload["losses"][label] = {
                "G_treated": l1, "G_control": l0,
                "G_model_mse": float(model_mse), "G_const_mse": float(const_mse),
                "G_model_mae": float(model_mae), "G_const_mae": float(const_mae),
                "G_r2_vs_const": float(1.0 - model_mse / const_mse) if const_mse > 0 else None,
                "Y_model": ly, "Y_const": cy,
                "Y_mse_reduction": float(1.0 - ly["mse"] / cy["mse"]) if cy["mse"] > 0 else None,
                "sign_logloss": ll, "sign_logloss_prior": bl,
            }
            print(f"\n[{label}] G held-out MSE {model_mse:.3f} (const {const_mse:.3f}, "
                  f"R2 {1.0 - model_mse / const_mse:+.3f}); MAE {model_mae:.3f} "
                  f"(const {const_mae:.3f})")
            print(f"           Y MSE {ly['mse']:.3f} (const {cy['mse']:.3f}); "
                  f"sign log-loss {ll:.4f} (prior {bl:.4f})")

            # Policy evaluation on the cross-fitted predictions of this cell.
            mu1, mu0 = cf["mu1"], cf["mu0"]
            gate = (mu1 > mu0).astype(float)
            pol = {
                "never": np.zeros(X.shape[0]),
                "always": np.ones(X.shape[0]),
                "gate": gate,
            }
            scores = {}
            for name, policy in pol.items():
                vals = aipw(mu1, mu0, D, G, policy)
                lo, hi = boot_ci(vals, n_boot=args.bootstrap, seed=11)
                scores[name] = {"aipw": float(vals.mean()), "ci": [lo, hi]}
            best_fixed_name = max(("never", "always"), key=lambda k: scores[k]["aipw"])
            best_fixed = scores[best_fixed_name]
            gate_score = scores["gate"]
            oracle_plugin = float(np.mean(np.maximum(mu1, mu0)))
            best_fixed_plugin = float(max(mu1.mean(), mu0.mean()))
            payload["policy"][label] = {
                "scores": scores,
                "best_fixed": best_fixed_name,
                "gate_minus_best_fixed": gate_score["aipw"] - best_fixed["aipw"],
                "plugin_oracle_gap": oracle_plugin - best_fixed_plugin,
                "plugin_mean_mu1": float(mu1.mean()),
                "plugin_mean_mu0": float(mu0.mean()),
            }
            print(f"           AIPW never {scores['never']['aipw']:+.3f} "
                  f"[{scores['never']['ci'][0]:+.3f},{scores['never']['ci'][1]:+.3f}]  "
                  f"always {scores['always']['aipw']:+.3f} "
                  f"[{scores['always']['ci'][0]:+.3f},{scores['always']['ci'][1]:+.3f}]  "
                  f"gate {gate_score['aipw']:+.3f} "
                  f"[{gate_score['ci'][0]:+.3f},{gate_score['ci'][1]:+.3f}]")
            print(f"           best fixed {best_fixed_name} -> gate - best fixed "
                  f"{gate_score['aipw'] - best_fixed['aipw']:+.3f}; "
                  f"plug-in oracle gap {oracle_plugin - best_fixed_plugin:+.3f}")

    # E2/E3 verdicts use the within-checkpoint ridge cell as the primary read-off.
    cell = payload["losses"].get("within/ridge") or next(iter(payload["losses"].values()))
    pol = payload["policy"].get("within/ridge") or next(iter(payload["policy"].values()))
    e2 = (
        f"E2: held-out G MSE reduction vs constant {cell['G_r2_vs_const']:+.3f}; "
        f"Y MSE reduction {cell['Y_mse_reduction']:+.3f}; sign log-loss "
        f"{cell['sign_logloss']:.4f} vs prior {cell['sign_logloss_prior']:.4f}"
    )
    e3 = (
        f"E3: gate - best fixed AIPW {pol['gate_minus_best_fixed']:+.3f}; "
        f"plug-in oracle gap {pol['plugin_oracle_gap']:+.3f}; "
        f"pooled randomized arm diff {pooled_diff['diff']:+.3f} "
        f"[{pooled_diff['ci'][0]:+.3f},{pooled_diff['ci'][1]:+.3f}]"
    )
    payload["verdict"] = [e2, e3]
    print("\n" + e2 + "\n" + e3)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
