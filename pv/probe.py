# Copyright 2024 InstaDeep Ltd. All rights reserved.

"""Experiment 2b: per-state records logged while arm C (OVMSE) trains.

At each probe point the most recent complete episodes from the online buffer are
replayed once through frozen networks to record, per state:

  * team deviation D(s) under the joint behaviour model, and the state novelty
    (disagreement across the behaviour-model ensemble);
  * the frozen offline Q_tot of the executed joint action, Q^off_tot, as a shadow
    forward with its own RNN state;
  * the one-step TD target y_TD = r + gamma * max_a' Qbar_tot(s', a') using the
    current target network (the design's y_TD; the trainer's TD-lambda target is
    a different quantity and is deliberately not used here);
  * the Monte-Carlo return-to-go G along the episode (no bootstrap at timeout);
  * the memory-constraint trigger 1[Q^off_tot > y_TD] and whether the memory
    target is closer to G than the TD target at that state;
  * the OVM weight lambda(t) in force at that step.

Records are appended as one JSON line per probe with parallel arrays, so the
downstream analysis (scripts/run_pv_exp2b.py) can bin by D, regress on D and t,
and run the shuffled-D negative control.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from pv.behaviour import JointBehaviourModel, NoveltyEnsemble, load_behaviour_model
from training.rollout import stack_episodes
from utils.paths import resolve_path


_MODEL_CACHE: Dict[str, JointBehaviourModel] = {}
_ENSEMBLE_CACHE: Dict[str, Optional[NoveltyEnsemble]] = {}


def _model(path: str, device: torch.device) -> Optional[JointBehaviourModel]:
    """Load (and memoise) the behaviour model.

    A missing file is deliberately NOT cached: the BC fit may still be running
    when arm C starts, and the probe must pick the model up as soon as it lands
    instead of being disabled for the whole run.
    """
    if not path:
        return None
    key = f"{path}|{device}"
    if key not in _MODEL_CACHE:
        if not os.path.isfile(path):
            print(f"[pv_probe] behaviour model not found at {path}; deviations disabled")
            return None
        _MODEL_CACHE[key] = load_behaviour_model(path, device=str(device))
    return _MODEL_CACHE[key]


def _ensemble(directory: str, device: torch.device) -> Optional[NoveltyEnsemble]:
    if not directory:
        return None
    key = f"{directory}|{device}"
    if key not in _ENSEMBLE_CACHE:
        members = sorted(glob.glob(os.path.join(directory, "member_*.pt")))
        _ENSEMBLE_CACHE[key] = (
            NoveltyEnsemble([load_behaviour_model(p, device=str(device)) for p in members])
            if len(members) > 1
            else None
        )
    return _ENSEMBLE_CACHE[key]


def mc_return_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Discounted return-to-go per step, no bootstrap at episode end."""
    out = np.zeros_like(rewards, dtype=np.float64)
    g = 0.0
    for t in range(rewards.shape[0] - 1, -1, -1):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def one_step_td_target(learner, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """y_TD = r + gamma * max_a' Qbar_tot(s', a') with the current target network."""
    with torch.no_grad():
        target_out = learner._forward_qs(batch, target=True)  # [b, t, n, a]
        avail = batch["avail_actions"]
        masked = torch.where(avail, target_out, torch.full_like(target_out, -9999999.0))
        max_actions = torch.argmax(masked, dim=3, keepdim=True)
        target_max_q = torch.gather(target_out, 3, max_actions)  # [b, t, n, 1]
        target_max_tot = learner.target_mixer(target_max_q, batch["state"])  # [b, t, 1]
        rewards = batch["reward"]
        terminated = batch["terminated"]
        if terminated.dim() == 3 and terminated.shape[-1] > 1:
            terminated = terminated.max(dim=-1, keepdim=True)[0]
        nxt = torch.zeros_like(target_max_tot)
        nxt[:, :-1] = target_max_tot[:, 1:]
        return rewards + float(learner.gamma) * nxt * (1.0 - terminated)


def probe_records(
    learner,
    batch: Dict[str, torch.Tensor],
    episodes: Sequence[dict],
    model: Optional[JointBehaviourModel],
    ensemble: Optional[NoveltyEnsemble],
    gamma: float,
) -> Dict[str, np.ndarray]:
    """Per-state arrays for one probe point (padded steps dropped)."""
    filled = batch["filled"]
    mask = (filled[..., 0] if filled.shape[-1] == 1 else filled.squeeze(-1)) > 0.5

    rewards = batch["reward"][..., 0].detach().cpu().numpy()  # [b, t]
    # The trainer's TD targets are defined on the first T-1 columns, so every
    # quantity is aligned to T-1 here.
    returns = np.stack(
        [mc_return_to_go(rewards[i], gamma) for i in range(rewards.shape[0])], axis=0
    )[:, :-1]

    with torch.no_grad():
        actions = batch["actions"]
        states = batch["state"]
        off_tot = learner._offline_chosen_tot(batch, actions, states)[..., 0]  # [b, t-1]
        y_td = one_step_td_target(learner, batch)[..., 0][:, :-1]  # [b, t-1]

    off_np = off_tot.detach().cpu().numpy()
    td_np = y_td.detach().cpu().numpy()

    dev = None
    novelty_np = None
    if model is not None:
        dev = model.deviations(batch)
        if ensemble is not None:
            novelty_np = ensemble.novelty(batch).detach().cpu().numpy()

    team = (
        dev["team"][:, :-1].detach().cpu().numpy()
        if dev is not None
        else np.full_like(returns, np.nan)
    )
    nov = novelty_np[:, :-1] if novelty_np is not None else np.full_like(returns, np.nan)

    off_bias = off_np - returns
    td_bias = td_np - returns
    trigger = (off_np > td_np).astype(np.float64)
    memory_closer = (np.abs(off_bias) < np.abs(td_bias)).astype(np.float64)

    sel = mask.detach().cpu().numpy()[:, :-1]
    return {
        "team_deviation": team[sel],
        "novelty": nov[sel],
        "off_bias": off_bias[sel],
        "td_bias": td_bias[sel],
        "abs_off_err": np.abs(off_bias)[sel],
        "abs_td_err": np.abs(td_bias)[sel],
        "memory_trigger": trigger[sel],
        "memory_closer": memory_closer[sel],
        "mask": sel,
    }


def run_probe(cfg: Dict, learner, episodes: List[dict], t_env: int) -> Dict:
    """Record one probe point and append it to the configured JSONL."""
    block = cfg.get("pv_probe") or {}
    out_path = str(resolve_path(block["out"]))
    n_keep = int(block.get("episodes", 20))
    picked = episodes[-n_keep:] if n_keep > 0 else list(episodes)
    if not picked:
        return {"t_env": t_env, "n_states": 0}
    max_seq = max(len(ep["steps"]) for ep in picked)
    batch = stack_episodes(picked, max_seq, learner.device)

    device = learner.device
    if getattr(learner, "offline_q_network", None) is None:
        # No frozen offline net (arm Z, from scratch): nothing to compare against.
        print("[pv_probe] no frozen offline network; skipping", flush=True)
        return {"t_env": t_env, "n_states": 0}
    model = _model(str(resolve_path(block.get("behaviour_ckpt", ""))), device)
    ensemble = _ensemble(str(resolve_path(block.get("ensemble_dir", ""))), device)
    gamma = float(cfg["system"].get("discount", 0.99))

    records = probe_records(learner, batch, picked, model, ensemble, gamma)
    thresholds_path = block.get("thresholds")
    q95 = float("nan")
    if thresholds_path and os.path.isfile(str(resolve_path(thresholds_path))):
        with open(str(resolve_path(thresholds_path)), encoding="utf-8") as handle:
            q95 = float(json.load(handle).get("q95", float("nan")))

    team = records["team_deviation"]
    finite = np.isfinite(team)
    row = {
        "t_env": int(t_env),
        "lambda_memory": float(learner.get_lambda_memory(t_env))
        if hasattr(learner, "get_lambda_memory")
        else float("nan"),
        "n_episodes": len(picked),
        "n_states": int(finite.sum()),
        "q95": q95,
        "coverage_frac": float(np.mean(team[finite] <= q95)) if finite.any() else float("nan"),
        "trigger_rate": float(records["memory_trigger"].mean()),
        "memory_closer_rate": float(records["memory_closer"].mean()),
        "trigger_and_closer_rate": float(
            np.mean(records["memory_trigger"] * records["memory_closer"])
        ),
        "mean_abs_off_err": float(np.nanmean(records["abs_off_err"])),
        "mean_abs_td_err": float(np.nanmean(records["abs_td_err"])),
        "mean_team_deviation": float(np.nanmean(team)) if finite.any() else float("nan"),
        "mean_novelty": float(np.nanmean(records["novelty"])),
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in row.items()
    }
    payload["arrays"] = {
        key: [float(x) for x in value.tolist()] if isinstance(value, np.ndarray) else value
        for key, value in records.items()
        if key != "mask"
    }
    with open(out_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(
        f"[pv_probe] t_env={t_env} states={row['n_states']} lambda={row['lambda_memory']:.3f} "
        f"cov={row['coverage_frac']:.3f} trigger={row['trigger_rate']:.3f} "
        f"closer={row['memory_closer_rate']:.3f}",
        flush=True,
    )
    return row
