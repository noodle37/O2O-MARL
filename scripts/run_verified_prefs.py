#!/usr/bin/env python3
# Copyright 2024 InstaDeep Ltd. All rights reserved.

"""Insight B: randomized verification of decision-relevant offline preferences.

For a fixed online policy version pi_v, each validation episode contributes at
most one trial.  The first state where the frozen offline teacher's joint greedy
action a_off differs from pi_v's joint greedy action a_v is treated as a
decision conflict.  The executed joint action is randomized:

    D=1  execute a_off, then continue with pi_v
    D=0  execute a_v,   then continue with pi_v

with P(D=1|x)=p (fixed at 0.5 here).  The discounted return G from the conflict
step is recorded, and the inverse-probability-weighted outcome

    Y = D G / p - (1-D) G / (1-p)

is unbiased for tau_v(x) = E[G(a_off; pi_v) - G(a_v; pi_v) | x].  The estimator
uses only information available at the decision (no episode length, no branch
returns), so it can be used as an online label.

The script also cross-fits a response model tau_hat(x) on held-out trials and
reports whether uncertainty/sign gating identifies conflicts where the teacher's
preference is harmful.  It is evaluation-only: no parameters are updated.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/run_verified_prefs.py \
    --config configs/pv_base_5m_vs_6m.yaml --t 1000000 --trials 400 \
    --policy-ckpt checkpoints/pv_exp1/5m_vs_6m/A_seed1/periodic/ovm_diag_ckpt_t1000000.pt \
    --out checkpoints/verified_prefs/5m_vs_6m/t1000000.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.factory import build_learner, load_vault_dataset, make_env
from training.rollout_helpers import legals_from_info
from utils.checkpoint import load_qmix_checkpoint
from utils.paths import ensure_project_root, resolve_path

import scripts.probe_rank_flip as prf


FEATURE_NAMES = [
    "allies",
    "enemies",
    "phase",
    "n_disagree",
    "q_off_margin",
    "q_on_margin",
    "abs_q_off_margin",
    "abs_q_on_margin",
    "q_off_live",
    "q_off_off",
    "q_on_live",
    "q_on_off",
]


def _agent_input(
    obs: np.ndarray,
    last_action: Optional[int],
    agent_index: int,
    n_actions: int,
    n_agents: int,
    add_agent_id: bool,
    device,
) -> torch.Tensor:
    pieces = [torch.from_numpy(np.asarray(obs)).float().to(device)]
    if last_action is None:
        pieces.append(torch.zeros(n_actions, device=device))
    else:
        pieces.append(
            F.one_hot(torch.tensor(int(last_action), device=device), n_actions).float()
        )
    if add_agent_id:
        pieces.append(F.one_hot(torch.tensor(agent_index, device=device), n_agents).float())
    return torch.cat(pieces, dim=-1).unsqueeze(0)


def _mix(mixer, per_agent_qs: List[float], state: np.ndarray, device) -> float:
    qs = torch.tensor(per_agent_qs, dtype=torch.float32, device=device).view(1, 1, -1)
    st = torch.as_tensor(state, dtype=torch.float32, device=device).view(1, 1, -1)
    return float(mixer(qs, st).reshape(-1)[0].item())


def _greedy(q: torch.Tensor, legal: Optional[torch.Tensor]) -> int:
    if legal is not None:
        q = q.masked_fill(~legal, -1e9)
    return int(torch.argmax(q).item())


def run_trial(
    env,
    *,
    live_q_net,
    live_mixer,
    off_q_net,
    off_mixer,
    rng: np.random.Generator,
    p: float,
    gamma: float,
    episode_limit: int,
    layout: Dict[str, Any],
    n_actions: int,
    n_agents: int,
    add_agent_id: bool,
    device,
    skip_conflicts: int = 0,
    treatment_agents: Optional[Tuple[int, ...]] = None,
) -> Optional[Dict[str, Any]]:
    """One episode; returns the selected-conflict trial record or None.

    `skip_conflicts` is drawn before the episode and ignored conflicts are
    counted without intervention.  Randomizing at the (skip+1)-th conflict
    samples later phases without branch replay and keeps the treatment
    probability independent of the state.
    """
    live_hidden = [live_q_net.initial_state(1, device) for _ in range(n_agents)]
    off_hidden = [off_q_net.initial_state(1, device) for _ in range(n_agents)]
    live_last: List[Optional[int]] = [None] * n_agents
    off_last: List[Optional[int]] = [None] * n_agents

    obs, info = env.reset()
    done = False
    t = 0
    G = 0.0
    discount = 1.0
    forced = False
    conflict_rank = 0
    record: Dict[str, Any] = {}
    won = None
    length = 0

    while not done:
        agents = sorted(obs.keys())
        legals = legals_from_info(info, agents)
        a_live: List[int] = []
        a_off: List[int] = []
        q_live_vals: List[torch.Tensor] = []
        q_off_vals: List[torch.Tensor] = []
        with torch.inference_mode():
            for i, agent in enumerate(agents):
                legal = None
                if legals is not None:
                    legal = torch.from_numpy(np.asarray(legals[agent])).bool().to(device)

                obs_in = _agent_input(
                    obs[agent], live_last[i], i, n_actions, n_agents, add_agent_id, device
                )
                q_live, live_hidden[i] = live_q_net(obs_in, live_hidden[i])
                q_live = q_live.reshape(-1)
                q_live_vals.append(q_live.detach())
                a_live.append(_greedy(q_live, legal))

                if not forced:
                    obs_in = _agent_input(
                        obs[agent], off_last[i], i, n_actions, n_agents, add_agent_id, device
                    )
                    q_off, off_hidden[i] = off_q_net(obs_in, off_hidden[i])
                    q_off = q_off.reshape(-1)
                    q_off_vals.append(q_off.detach())
                    a_off.append(_greedy(q_off, legal))

        executed = a_live
        treated = list(range(n_agents)) if treatment_agents is None else list(treatment_agents)
        if not forced and any(a_live[i] != a_off[i] for i in treated):
            conflict_rank += 1
            if conflict_rank > skip_conflicts:
                forced = True
                state = np.asarray(info["state"], dtype=np.float32)
                allies, enemies = prf.alive_counts(state, layout)

                def gather(qs: List[torch.Tensor], actions: List[int]) -> List[float]:
                    return [float(qs[i][actions[i]].item()) for i in range(n_agents)]

                on_live = _mix(live_mixer, gather(q_live_vals, a_live), state, device)
                on_off = _mix(live_mixer, gather(q_live_vals, a_off), state, device)
                off_live = _mix(off_mixer, gather(q_off_vals, a_live), state, device)
                off_off = _mix(off_mixer, gather(q_off_vals, a_off), state, device)
                q_off_margin = off_off - off_live
                q_on_margin = on_live - on_off

                D = int(rng.random() < p)
                a_treated = list(a_live)
                for i in treated:
                    a_treated[i] = a_off[i]
                executed = a_treated if D == 1 else a_live
                phase = float(t) / float(max(1, episode_limit))
                record = {
                    "trial_t": int(t),
                    "conflict_rank": int(conflict_rank),
                    "skip_conflicts": int(skip_conflicts),
                    "D": D,
                    "p": float(p),
                    "treatment_agents": treated,
                    "a_off": [int(x) for x in a_off],
                    "a_live": [int(x) for x in a_live],
                    "a_treated": [int(x) for x in a_treated],
                    "n_disagree": int(sum(a_live[i] != a_off[i] for i in range(n_agents))),
                    "n_treatment_disagree": int(sum(a_live[i] != a_off[i] for i in treated)),
                    "allies": int(allies),
                    "enemies": int(enemies),
                    "phase": phase,
                    "q_off_live": float(off_live),
                    "q_off_off": float(off_off),
                    "q_on_live": float(on_live),
                    "q_on_off": float(on_off),
                    "q_off_margin": float(q_off_margin),
                    "q_on_margin": float(q_on_margin),
                    "abs_q_off_margin": float(abs(q_off_margin)),
                    "abs_q_on_margin": float(abs(q_on_margin)),
                }

        act = {agent: int(executed[i]) for i, agent in enumerate(agents)}
        next_obs, rewards, terminal, trunc, next_info = env.step(act)
        reward = float(np.mean(list(rewards.values())))

        if forced:
            G += discount * reward
            discount *= gamma

        # Both networks condition on the history of actions actually executed:
        # that is the history of the state at which the teacher is being asked
        # for its action.
        for i in range(n_agents):
            live_last[i] = int(executed[i])
            off_last[i] = int(executed[i])

        obs, info = next_obs, next_info
        t += 1
        if "battle_won" in info:
            won = bool(info["battle_won"])
        done = all(terminal.values()) or all(trunc.values())
        length = t

    if not forced:
        return None
    record["G"] = float(G)
    record["Y"] = float(record["D"] * G / p - (1 - record["D"]) * G / (1 - p))
    record["won"] = None if won is None else int(won)
    record["episode_length"] = int(length)
    return record


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _cross_fit(X: np.ndarray, Y: np.ndarray, folds: int = 5, alpha: float = 1.0) -> np.ndarray:
    kf = KFold(n_splits=min(folds, len(Y)), shuffle=True, random_state=0)
    pred = np.zeros_like(Y, dtype=np.float64)
    for train, test in kf.split(X):
        scaler = StandardScaler().fit(X[train])
        model = Ridge(alpha=alpha).fit(scaler.transform(X[train]), Y[train])
        pred[test] = model.predict(scaler.transform(X[test]))
    return pred


def _arm_summary(
    records: List[Dict[str, Any]], mask, n_boot: int = 1000, seed: int = 1
) -> Dict[str, Any]:
    """Randomization-stratified arm difference: mean G(D=1) - mean G(D=0).

    Because D is drawn independently of the state, this difference is unbiased
    for E[tau | stratum] without inverse-probability weights, and it has much
    lower variance than the per-trial IPW mean.
    """
    sel = [r for r in records if mask(r)]
    treated = np.asarray([r["G"] for r in sel if r["D"] == 1], dtype=np.float64)
    control = np.asarray([r["G"] for r in sel if r["D"] == 0], dtype=np.float64)
    out: Dict[str, Any] = {
        "n": int(len(sel)),
        "n_treated": int(treated.size),
        "n_control": int(control.size),
        "coverage": float(len(sel)) / float(max(1, len(records))),
    }
    if treated.size < 2 or control.size < 2:
        return out
    diff = float(treated.mean() - control.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        tb = treated[rng.integers(0, treated.size, size=treated.size)]
        cb = control[rng.integers(0, control.size, size=control.size)]
        boot[b] = tb.mean() - cb.mean()
    out.update(
        {
            "mean_G_treated": float(treated.mean()),
            "mean_G_control": float(control.mean()),
            "arm_diff": diff,
            "arm_diff_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "p_arm_diff_positive": float((boot > 0).mean()),
        }
    )
    tw = np.asarray([r["won"] for r in sel if r["D"] == 1 and r.get("won") is not None])
    cw = np.asarray([r["won"] for r in sel if r["D"] == 0 and r.get("won") is not None])
    if tw.size >= 2 and cw.size >= 2:
        out["win_rate_treated"] = float(tw.mean())
        out["win_rate_control"] = float(cw.mean())
        out["win_diff"] = float(tw.mean() - cw.mean())
    return out


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) <= 1:
        return float("nan")
    return float(roc_auc_score(labels.astype(int), scores))


def analyze(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"n_trials": 0}
    X = np.asarray([[r[k] for k in FEATURE_NAMES] for r in records], dtype=np.float64)
    Y = np.asarray([r["Y"] for r in records], dtype=np.float64)
    G = np.asarray([r["G"] for r in records], dtype=np.float64)
    teacher_margin = np.asarray([r["q_off_margin"] for r in records], dtype=np.float64)
    online_margin = np.asarray([r["q_on_margin"] for r in records], dtype=np.float64)
    tau_hat = _cross_fit(X, Y)
    for rec, pred in zip(records, tau_hat):
        rec["tau_hat"] = float(pred)
    positive = (Y > 0).astype(int)
    negative_online_margin = -online_margin

    from scipy.stats import spearmanr

    out: Dict[str, Any] = {
        "n_trials": int(len(records)),
        "tau_ipw_mean": float(Y.mean()),
        "tau_ipw_ci": list(_bootstrap_ci(Y)),
        "overall_arm": _arm_summary(records, lambda r: True),
        "teacher_margin_positive_rate": float((teacher_margin > 0).mean()),
        "tau_hat_mean": float(tau_hat.mean()),
        "tau_hat_positive_rate": float((tau_hat > 0).mean()),
        "predicted_flip_rate": float((tau_hat < 0).mean()),
        "spearman_tau_hat_Y": float(spearmanr(tau_hat, Y).statistic),
        "spearman_neg_q_on_margin_Y": float(spearmanr(negative_online_margin, Y).statistic),
        "spearman_q_off_margin_Y": float(spearmanr(teacher_margin, Y).statistic),
        "auc_tau_hat_for_positive_Y": _auc(positive, tau_hat),
        "auc_neg_q_on_margin_for_positive_Y": _auc(positive, negative_online_margin),
        "auc_q_off_margin_for_positive_Y": _auc(positive, teacher_margin),
    }

    # Sign gate: follow the teacher only where the cross-fitted response model is
    # positive.  The dropped set is the model's proposed override region.
    sign_gate = tau_hat > 0
    out["sign_gate_tau_hat"] = {
        "kept": _arm_summary(records, lambda r: r["tau_hat"] > 0),
        "dropped": _arm_summary(records, lambda r: r["tau_hat"] <= 0),
    }
    # The teacher's own frozen margin, used as a baseline gate.
    out["sign_gate_q_off_margin"] = {
        "kept": _arm_summary(records, lambda r: r["q_off_margin"] > 0),
        "dropped": _arm_summary(records, lambda r: r["q_off_margin"] <= 0),
    }
    # Uncertainty gate: keep the most confident predictions.
    for q in (0.25, 0.5):
        threshold = float(np.quantile(np.abs(tau_hat), 1.0 - q))
        out[f"confidence_gate_q{q}"] = {
            "threshold": threshold,
            "kept": _arm_summary(records, lambda r: abs(r["tau_hat"]) >= threshold),
            "dropped": _arm_summary(records, lambda r: abs(r["tau_hat"]) < threshold),
        }

    # Group diagnostics with randomization-stratified arm differences.
    disagree = np.asarray([r["n_disagree"] for r in records], dtype=np.float64)
    rows = []
    for value in sorted(set(int(v) for v in disagree)):
        rows.append(
            {
                "n_disagree": value,
                "arm": _arm_summary(records, lambda r, v=value: r["n_disagree"] == v),
            }
        )
    out["groups_by_n_disagree"] = rows
    for key in ("abs_q_off_margin", "phase"):
        values = np.asarray([r[key] for r in records], dtype=np.float64)
        bins = np.quantile(values, [0.25, 0.5, 0.75])
        groups = np.digitize(values, bins)
        for rec, g in zip(records, groups):
            rec["_grp"] = int(g)
        grp_rows = []
        for g in range(4):
            members = values[groups == g]
            grp_rows.append(
                {
                    "group": g,
                    "value_range": [float(members.min()), float(members.max())]
                    if members.size
                    else None,
                    "arm": _arm_summary(records, lambda r, gg=g: r["_grp"] == gg),
                }
            )
        out[f"groups_by_{key}"] = grp_rows
        for rec in records:
            rec.pop("_grp", None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/pv_base_5m_vs_6m.yaml")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--t", type=int, required=True, help="online policy version (env steps)")
    ap.add_argument("--policy-ckpt", default=None)
    ap.add_argument("--offline-ckpt", default=None)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--max-skip", type=int, default=4, help="max conflicts ignored before the trial")
    ap.add_argument("--eval-seed", type=int, default=424242)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    ensure_project_root()
    cfg = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides)),
        resolve=True,
    )
    dataset = load_vault_dataset(cfg)
    learner = build_learner(cfg, dataset)
    device = learner.device
    n_agents = int(dataset.stats.n_agents)
    n_actions = int(dataset.stats.n_actions)
    add_agent_id = bool(cfg["system"].get("add_agent_id_to_obs", True))
    gamma = float(cfg["system"]["discount"])
    episode_limit = int(dataset.stats.max_seq_length) - 1

    offline_ckpt = args.offline_ckpt or "checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt"
    offline_ckpt = str(resolve_path(offline_ckpt))
    if not os.path.isfile(offline_ckpt):
        raise SystemExit(f"FATAL: offline checkpoint missing: {offline_ckpt}")

    policy_ckpt = args.policy_ckpt
    if policy_ckpt is None:
        policy_ckpt = os.path.join(
            "checkpoints", "pv_exp1", "5m_vs_6m", f"A_seed{args.seed}",
            "periodic", f"ovm_diag_ckpt_t{args.t}.pt",
        )
    policy_ckpt = str(resolve_path(policy_ckpt))
    if not os.path.isfile(policy_ckpt):
        raise SystemExit(f"FATAL: policy checkpoint missing: {policy_ckpt}")

    load_qmix_checkpoint(policy_ckpt, learner, load_optimizer=False)
    learner.load_offline_for_ovm(offline_ckpt)
    learner.q_network.eval()
    learner.mixer.eval()
    learner.offline_q_network.eval()
    learner.offline_mixer.eval()

    env = make_env(cfg, seed=args.eval_seed)
    layout = prf.probe_layout(env)
    rng = np.random.default_rng(args.eval_seed + args.t)

    records: List[Dict[str, Any]] = []
    n_no_conflict = 0
    t0 = time.time()
    with torch.inference_mode():
        for i in range(args.trials):
            skip = int(rng.integers(0, args.max_skip + 1))
            rec = run_trial(
                env,
                live_q_net=learner.q_network,
                live_mixer=learner.mixer,
                off_q_net=learner.offline_q_network,
                off_mixer=learner.offline_mixer,
                rng=rng,
                p=args.p,
                gamma=gamma,
                episode_limit=episode_limit,
                layout=layout,
                n_actions=n_actions,
                n_agents=n_agents,
                add_agent_id=add_agent_id,
                device=device,
                skip_conflicts=skip,
            )
            if rec is None:
                n_no_conflict += 1
            else:
                rec["episode_index"] = i
                records.append(rec)
            if (i + 1) % 50 == 0:
                print(
                    f"[verified] t={args.t} ep={i+1}/{args.trials} "
                    f"trials={len(records)} no_conflict={n_no_conflict} "
                    f"elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )

    payload = {
        "map": cfg["task"]["scenario"],
        "seed": int(args.seed),
        "policy_version": int(args.t),
        "policy_ckpt": policy_ckpt,
        "offline_ckpt": offline_ckpt,
        "episodes_attempted": int(args.trials),
        "conflict_episodes": int(len(records)),
        "no_conflict_episodes": int(n_no_conflict),
        "p": float(args.p),
        "eval_seed": int(args.eval_seed),
        "feature_names": FEATURE_NAMES,
        "records": records,
        "analysis": analyze(records),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"[verified] wrote {args.out} trials={len(records)} analysis={payload['analysis']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
