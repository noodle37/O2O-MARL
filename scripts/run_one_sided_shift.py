#!/usr/bin/env python3
"""One-sided policy-shift evaluation for SMAC 5m_vs_6m.

One process evaluates exactly one (grid, source, arm, focal, eps) condition and
writes one JSONL row per episode plus a final summary row. The four arms are:

  O  all agents greedy on the offline net
  A  all agents deviate
  B  the focal agent stays offline-greedy; its teammates deviate
  C  only the focal agent deviates; its teammates stay offline-greedy

Two deviation mechanisms are supported:

  eps      every agent keeps the offline net but the designated subset takes a
           uniformly random legal action with probability eps per step.
  replace  the designated subset uses an online fine-tuned checkpoint instead.

Noise for the eps grid is hashed from (noise_seed, episode, step, agent), so the
same perturbation schedule is shared by A/B/C: they differ only in who is
eligible to receive it. This keeps the conditions paired across processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pv.exp2_shift import Net, run_episode
from training.factory import build_learner, load_vault_dataset, make_env
from utils.checkpoint import load_qmix_checkpoint
from utils.paths import ensure_project_root, resolve_path

DEFAULT_CONFIG = "configs/pv_base_5m_vs_6m.yaml"


def _layout(env) -> Dict[str, int]:
    inner = env._environment
    return {
        "n_allies": int(inner.n_agents),
        "nf_ally": int(inner.get_ally_num_attributes()),
        "n_enemies": int(inner.n_enemies),
        "nf_enemy": int(inner.get_enemy_num_attributes()),
    }


def _alive_counts(state: np.ndarray, layout: Dict[str, int]) -> tuple[int, int]:
    n_al, nf_al = layout["n_allies"], layout["nf_ally"]
    n_en, nf_en = layout["n_enemies"], layout["nf_enemy"]
    allies = sum(1 for i in range(n_al) if state[i * nf_al] > 0.0)
    base = n_al * nf_al
    enemies = sum(1 for i in range(n_en) if state[base + i * nf_en] > 0.0)
    return int(allies), int(enemies)


def _state_hash(state: Optional[np.ndarray]) -> Optional[str]:
    if state is None:
        return None
    arr = np.round(np.asarray(state, dtype=np.float64), 6)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _episode_metrics(rec, env, layout: Dict[str, int], n_agents: int) -> Dict[str, Any]:
    """Behavioural metrics from one EpisodeRecord (no environment internals)."""
    n_enemies = layout["n_enemies"]
    attack_actions = alive_actions = alive_agent_steps = 0
    attack_steps = focus_steps = 0
    attackers_per_attack: List[int] = []
    q_values: List[float] = []

    for step in rec.steps:
        actions = np.asarray(step["actions"], dtype=np.int64)
        legals = step["legals"]
        if legals is None:
            alive = np.ones(n_agents, dtype=bool)
        else:
            # Dead agents get the mask [1, 0, ...]: stop (action 1) is illegal.
            alive = np.asarray(legals)[:, 1] > 0
        alive_actions += int(alive.sum())
        alive_agent_steps += int(alive.sum())
        attacks = alive & (actions >= 6)
        n_attack = int(attacks.sum())
        if n_attack:
            attack_steps += 1
            attack_actions += n_attack
            attackers_per_attack.append(n_attack)
            targets = actions[attacks] - 6
            counts = np.bincount(targets, minlength=n_enemies)
            if int(counts.max()) >= 2:
                focus_steps += 1
        if step["q_off_tot"] is not None:
            q_values.append(float(step["q_off_tot"]))

    win = bool(rec.won) if rec.won is not None else False
    episode_limit = int(getattr(env._environment, "episode_limit", 0) or 0)
    try:
        final_state = env._environment.get_state()
        allies_end, enemies_end = _alive_counts(final_state, layout)
    except Exception:
        if rec.steps and rec.steps[-1]["state"] is not None:
            allies_end, enemies_end = _alive_counts(rec.steps[-1]["state"], layout)
        else:
            allies_end = enemies_end = None

    return {
        "return": float(rec.episode_return),
        "win": int(win),
        "length": int(rec.length),
        "timeout": int(episode_limit > 0 and rec.length >= episode_limit and not win),
        "attack_share": (attack_actions / alive_actions) if alive_actions else None,
        "focus_fire_rate": (focus_steps / attack_steps) if attack_steps else None,
        "mean_attackers_per_attack_step": (
            float(np.mean(attackers_per_attack)) if attackers_per_attack else None
        ),
        "alive_allies_end": allies_end,
        "alive_enemies_end": enemies_end,
        "q_off_tot_mean": float(np.mean(q_values)) if q_values else None,
        "alive_agent_steps": int(alive_agent_steps),
        "perturb_trials": [int(rec.perturb_trials.get(i, 0)) for i in range(n_agents)],
        "perturb_events": [int(rec.perturb_events.get(i, 0)) for i in range(n_agents)],
        "perturb_changes": [int(rec.perturb_changes.get(i, 0)) for i in range(n_agents)],
        "init_state_sha1": _state_hash(rec.steps[0]["state"] if rec.steps else None),
        "first_actions": (
            [int(a) for a in rec.steps[0]["actions"]] if rec.steps else []
        ),
    }


def _mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _configure(cfg: Dict[str, Any], mode: str, arm: str, focal: int, eps: float,
               n_agents: int) -> tuple[Dict[int, str], set, float]:
    all_idx = set(range(n_agents))
    if mode == "eps":
        sources = {i: "offline" for i in all_idx}
        if arm == "O":
            examined, used_eps = set(all_idx), 0.0
        elif arm == "A":
            examined, used_eps = set(), float(eps)
        elif arm == "B":
            examined, used_eps = {focal}, float(eps)
        else:  # C
            examined, used_eps = all_idx - {focal}, float(eps)
        if not all(v == "offline" for v in sources.values()):
            raise SystemExit("eps mode must use only the offline net")
        if arm != "O" and used_eps <= 0.0:
            raise SystemExit(f"arm {arm} needs eps > 0")
    else:  # replace
        examined, used_eps = set(all_idx), 0.0
        if arm == "O":
            sources = {i: "offline" for i in all_idx}
        elif arm == "A":
            sources = {i: "live" for i in all_idx}
        elif arm == "B":
            sources = {i: ("offline" if i == focal else "live") for i in all_idx}
        else:  # C
            sources = {i: ("live" if i == focal else "offline") for i in all_idx}
    if not 0 <= focal < n_agents:
        raise SystemExit(f"focal {focal} out of range 0..{n_agents - 1}")
    return sources, examined, used_eps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--offline-ckpt", required=True)
    ap.add_argument("--live-ckpt", default="none")
    ap.add_argument("--mode", choices=["eps", "replace"], required=True)
    ap.add_argument("--arm", choices=["O", "A", "B", "C"], required=True)
    ap.add_argument("--focal", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--episodes", type=int, default=640)
    ap.add_argument("--eval-seed", type=int, default=424242)
    ap.add_argument("--noise-seed", type=int, default=424242)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    ensure_project_root()
    episodes = min(args.episodes, 2) if args.smoke else args.episodes

    cfg = OmegaConf.to_container(
        OmegaConf.merge(
            OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides)
        ),
        resolve=True,
    )
    cfg["system"]["device"] = args.device
    dataset = load_vault_dataset(cfg)
    learner = build_learner(cfg, dataset)
    device = learner.device
    n_agents = int(dataset.stats.n_agents)
    n_actions = int(dataset.stats.n_actions)
    add_agent_id = bool(cfg["system"].get("add_agent_id_to_obs", True))

    offline_ckpt = str(resolve_path(args.offline_ckpt))
    if not os.path.isfile(offline_ckpt):
        raise SystemExit(f"offline checkpoint missing: {offline_ckpt}")
    live_ckpt: Optional[str] = None
    if args.live_ckpt and args.live_ckpt.lower() != "none":
        live_ckpt = str(resolve_path(args.live_ckpt))
        if not os.path.isfile(live_ckpt):
            raise SystemExit(f"live checkpoint missing: {live_ckpt}")

    sources, examined, used_eps = _configure(
        cfg, args.mode, args.arm, args.focal, args.eps, n_agents
    )
    if args.mode == "replace" and args.arm != "O" and live_ckpt is None:
        raise SystemExit("replace mode needs --live-ckpt for arms A/B/C")

    learner.load_offline_for_ovm(offline_ckpt)
    offline = Net(q_network=learner.offline_q_network, mixer=learner.offline_mixer)
    if live_ckpt is not None:
        load_qmix_checkpoint(live_ckpt, learner, load_optimizer=False)
        live = Net(q_network=learner.q_network, mixer=learner.mixer)
    else:
        live = offline
    for net in (offline, live):
        net.q_network.eval()
        net.mixer.eval()
        for p in net.q_network.parameters():
            p.requires_grad_(False)
        for p in net.mixer.parameters():
            p.requires_grad_(False)

    env = make_env(cfg, seed=args.eval_seed)
    layout = _layout(env)
    rng = np.random.default_rng(args.noise_seed)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tag = os.path.splitext(os.path.basename(out_path))[0]
    print(
        f"[one-sided] {tag} mode={args.mode} arm={args.arm} focal={args.focal} "
        f"eps={used_eps:g} episodes={episodes} live={live_ckpt}",
        flush=True,
    )

    episode_rows: List[Dict[str, Any]] = []
    started = time.time()
    with open(out_path, "w", encoding="utf-8") as handle:
        for k in range(episodes):
            with torch.no_grad():
                rec = run_episode(
                    env,
                    live=live,
                    offline=offline,
                    sources=sources,
                    examined=examined,
                    eps_partner=used_eps,
                    n_actions=n_actions,
                    add_agent_id=add_agent_id,
                    device=device,
                    rng=rng,
                    noise_seed=args.noise_seed,
                    episode_index=k,
                )
            row = _episode_metrics(rec, env, layout, n_agents)
            row["episode"] = k
            episode_rows.append(row)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if (k + 1) % 32 == 0 or k + 1 == episodes:
                recent = episode_rows[-32:]
                print(
                    f"[one-sided] {tag} {k + 1}/{episodes} "
                    f"ret={np.mean([r['return'] for r in recent]):.3f} "
                    f"win={np.mean([r['win'] for r in recent]):.3f}",
                    flush=True,
                )

        summary = {
            "summary": True,
            "tag": tag,
            "config": args.config,
            "grid": args.mode,
            "mode": args.mode,
            "arm": args.arm,
            "focal": int(args.focal),
            "eps": float(used_eps),
            "episodes": int(episodes),
            "eval_seed": int(args.eval_seed),
            "noise_seed": int(args.noise_seed),
            "offline_ckpt": offline_ckpt,
            "live_ckpt": live_ckpt,
            "n_agents": n_agents,
            "sources": {str(i): sources[i] for i in sorted(sources)},
            "examined": sorted(int(i) for i in examined),
            "elapsed_s": time.time() - started,
            "return_mean": _mean([r["return"] for r in episode_rows]),
            "win_rate": _mean([r["win"] for r in episode_rows]),
            "length_mean": _mean([r["length"] for r in episode_rows]),
            "timeout_rate": _mean([r["timeout"] for r in episode_rows]),
            "attack_share_mean": _mean([r["attack_share"] for r in episode_rows]),
            "focus_fire_rate_mean": _mean(
                [r["focus_fire_rate"] for r in episode_rows]
            ),
            "alive_enemies_end_mean": _mean(
                [r["alive_enemies_end"] for r in episode_rows]
            ),
            "q_off_tot_mean": _mean([r["q_off_tot_mean"] for r in episode_rows]),
            "perturb_events_sum": [
                int(sum(r["perturb_events"][i] for r in episode_rows))
                for i in range(n_agents)
            ],
            "perturb_changes_sum": [
                int(sum(r["perturb_changes"][i] for r in episode_rows))
                for i in range(n_agents)
            ],
            "perturb_events_total": int(
                sum(sum(r["perturb_events"]) for r in episode_rows)
            ),
            "perturb_changes_total": int(
                sum(sum(r["perturb_changes"]) for r in episode_rows)
            ),
            "alive_agent_steps_total": int(
                sum(r["alive_agent_steps"] for r in episode_rows)
            ),
        }
        handle.write(json.dumps(summary) + "\n")
        handle.flush()

    try:
        env.close()
    except Exception:
        pass
    print(f"[one-sided] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
