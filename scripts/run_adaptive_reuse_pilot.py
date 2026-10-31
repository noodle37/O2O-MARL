#!/usr/bin/env python3
"""Fit a local pair-intervention response model for the adaptive-reuse pilot."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joint_reuse.runner import DEFAULT_OFFLINE_REFERENCE, parse_offline_agents
from scripts import run_verified_prefs as vp
from training.factory import build_learner, load_vault_dataset, make_env
from utils.checkpoint import load_qmix_checkpoint
from utils.paths import ensure_project_root, resolve_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/pv_base_5m_vs_6m.yaml")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--policy-ckpt", required=True)
    ap.add_argument("--offline-ckpt", default=DEFAULT_OFFLINE_REFERENCE)
    ap.add_argument("--treatment-agents", default="1,4")
    ap.add_argument("--trials", type=int, default=96)
    ap.add_argument("--bootstrap", type=int, default=64)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--max-skip", type=int, default=4)
    ap.add_argument("--eval-seed", type=int, default=424242)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    ensure_project_root()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    dataset = load_vault_dataset(cfg)
    learner = build_learner(cfg, dataset)
    treatment_agents = parse_offline_agents(args.treatment_agents)
    if not treatment_agents:
        raise SystemExit("at least one treatment agent is required")

    policy_ckpt = str(resolve_path(args.policy_ckpt))
    offline_ckpt = str(resolve_path(args.offline_ckpt))
    load_qmix_checkpoint(policy_ckpt, learner, load_optimizer=False)
    learner.load_offline_for_ovm(offline_ckpt)
    learner.q_network.eval()
    learner.mixer.eval()
    learner.offline_q_network.eval()
    learner.offline_mixer.eval()

    env = make_env(cfg, seed=args.eval_seed)
    layout = vp.prf.probe_layout(env)
    rng = np.random.default_rng(args.eval_seed + args.seed)
    records = []
    no_conflict = 0
    started = time.time()
    for episode_index in range(args.trials):
        rec = vp.run_trial(
            env,
            live_q_net=learner.q_network,
            live_mixer=learner.mixer,
            off_q_net=learner.offline_q_network,
            off_mixer=learner.offline_mixer,
            rng=rng,
            p=args.p,
            gamma=float(cfg["system"]["discount"]),
            episode_limit=int(dataset.stats.max_seq_length) - 1,
            layout=layout,
            n_actions=int(dataset.stats.n_actions),
            n_agents=int(dataset.stats.n_agents),
            add_agent_id=bool(cfg["system"].get("add_agent_id_to_obs", True)),
            device=learner.device,
            skip_conflicts=int(rng.integers(0, args.max_skip + 1)),
            treatment_agents=treatment_agents,
        )
        if rec is None:
            no_conflict += 1
        else:
            rec["episode_index"] = episode_index
            records.append(rec)
        if (episode_index + 1) % 24 == 0:
            print(
                f"[adaptive-pilot] episodes={episode_index + 1}/{args.trials} "
                f"trials={len(records)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    trial_seconds = time.time() - started
    if len(records) < 10:
        raise SystemExit(f"only {len(records)} conflict trials; cannot fit response model")

    X = np.asarray([[r[k] for k in vp.FEATURE_NAMES] for r in records], dtype=np.float64)
    Y = np.asarray([r["Y"] for r in records], dtype=np.float64)
    ensemble = []
    fit_started = time.time()
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(Y), size=len(Y))
        if np.unique(Y[idx]).size < 2:
            continue
        scaler = StandardScaler().fit(X[idx])
        model = Ridge(alpha=1.0).fit(scaler.transform(X[idx]), Y[idx])
        ensemble.append((scaler, model))
    fit_seconds = time.time() - fit_started
    if not ensemble:
        raise SystemExit("failed to fit response ensemble")

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "response_model.pkl")
    artifact = {
        "feature_names": vp.FEATURE_NAMES,
        "treatment_agents": treatment_agents,
        "layout": layout,
        "ensemble": ensemble,
        "n_agents": int(dataset.stats.n_agents),
        "n_actions": int(dataset.stats.n_actions),
        "add_agent_id": bool(cfg["system"].get("add_agent_id_to_obs", True)),
        "episode_limit": int(dataset.stats.max_seq_length) - 1,
        "policy_ckpt": policy_ckpt,
        "offline_ckpt": offline_ckpt,
    }
    with open(model_path, "wb") as handle:
        pickle.dump(artifact, handle)

    analysis = vp.analyze(records)
    payload = {
        "seed": args.seed,
        "policy_ckpt": policy_ckpt,
        "offline_ckpt": offline_ckpt,
        "treatment_agents": treatment_agents,
        "episodes_attempted": args.trials,
        "conflict_episodes": len(records),
        "no_conflict_episodes": no_conflict,
        "trial_seconds": trial_seconds,
        "fit_seconds": fit_seconds,
        "ensemble_size": len(ensemble),
        "records": records,
        "analysis": analysis,
        "model_path": model_path,
    }
    json_path = os.path.join(args.out_dir, "pilot.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(
        f"[adaptive-pilot] wrote {json_path} model={model_path} "
        f"trials={len(records)} trial_seconds={trial_seconds:.1f} "
        f"fit_seconds={fit_seconds:.2f} overall={analysis['overall_arm']}",
        flush=True,
    )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
