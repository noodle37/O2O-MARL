#!/usr/bin/env python3
"""Task B: decision meaning of joint-action preference flips (5m_vs_6m, seed 1).

Question: the offline teacher's joint-action preference and the preference a
continuation policy actually realises are two different orderings. This probe
measures, on the same state and the same candidate pair, whether they disagree,
whether the disagreement touches the action the online policy actually takes,
and what the disagreement costs in discounted return.

Design
------
* Population of states: the greedy trajectory of the fork-time online policy
  (the deployed policy at the fork instant), one trajectory per environment seed.
* Population of pairs: the UOT hinge's own candidate set, built once from the
  deployed policy (`uot.neighborhood.build_recorded_neighborhood`, K=8,
  Hamming <= 2) and then held fixed. Holding it fixed is what makes the
  cross-policy comparison exact: the same pair is scored under the frozen
  teacher and under every continuation policy.
    P1  (teacher argmax, deployed live argmax) when they differ - the pair that
        decides which in-support joint action is executed, and the pair the
        hinge pushes on. Scored under every policy.
    P2  a deterministic subsample of the remaining hinge pairs, scored under
        the deployed policy only - the general pair population, used for the
        single-pair flip rate and for the contrast with pairs that do not
        involve the argmax.
* Delta G: branch replay. Relaunch the environment on the same seed, replay the
  recorded prefix up to the probe step, force the candidate joint action there,
  then act greedily under that policy to the end of the episode. Delta G is the
  difference of the two discounted returns from the probe step (gamma from the
  config). Every replay must reproduce the recorded state bitwise, otherwise the
  measurement is dropped.
* Deployable features: nothing that needs Delta G. Teacher gap, live gap,
  live-vs-target gap, hinge slack and weight, candidate/support sizes, alive
  counts, battle phase, offline joint-action counts.

What the replay oracle is NOT: Delta G is measured by branching in the real
environment. It is ground truth for this measurement only; it is not a signal a
deployed gate could read. The predictability stage therefore predicts
decision-relevant mismatch from online-observable features alone, with the
oracle used only as the label.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/probe_rank_flip.py \
      --fork-snapshot checkpoints/uot_fork_5m_vs_6m_seed1_formal/fork_snapshot_main_t1009049.pt \
      --checkpoint continue=checkpoints/uot_fork_5m_vs_6m_seed1_formal/uot_online_fork_resume_continue_seed1.pt \
      --checkpoint decay=checkpoints/uot_fork_5m_vs_6m_seed1_formal/uot_online_fork_resume_decay_seed1.pt \
      --checkpoint off=checkpoints/uot_fork_5m_vs_6m_seed1_formal/uot_online_fork_resume_off_seed1.pt \
      --env-seeds 64 --workers 24 --p2-stride 8 --p2-pairs 2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("SC2PATH", "/home/jxy/StarCraftII")

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from training.factory import build_learner, load_vault_dataset, make_env  # noqa: E402
from training.rollout import stack_episodes  # noqa: E402
from uot.neighborhood import build_recorded_neighborhood  # noqa: E402
from utils.paths import ensure_project_root, resolve_path  # noqa: E402

from probe_fork_branches import _build_learner  # noqa: E402
from verify_local_rank_inversion import close_env, reset_env  # noqa: E402

RETURN_TIE = 1e-4


# --------------------------------------------------------------------------- #
# rollout helpers
# --------------------------------------------------------------------------- #
def discounted_return(rewards: List[float], start: int, gamma: float) -> float:
    acc = 0.0
    for value in reversed(rewards[start:]):
        acc = float(value) + gamma * acc
    return acc


def _mean_reward(rewards: Dict[str, Any]) -> float:
    return float(np.mean([float(np.asarray(r)) for r in rewards.values()]))


def _all_done(terminal, trunc) -> bool:
    return all(bool(np.asarray(v)) for v in terminal.values()) or all(
        bool(np.asarray(v)) for v in trunc.values()
    )


def collect_trajectory(env, learner) -> Dict[str, Any]:
    """One greedy (eps=0) episode from a freshly relaunched environment."""
    obs, info = reset_env(env)
    state = learner.new_episode_state()
    steps: List[Dict[str, Any]] = []
    rewards: List[float] = []
    won = None
    while True:
        agents = sorted(obs.keys())
        actions = learner.select_actions_eval(
            obs, {a: info["legals"][a] for a in agents}, 0.0, state,
            env_state=info.get("state"),
        )
        steps.append(
            {
                "obs": np.stack([obs[a] for a in agents]),
                "actions": np.asarray([actions[a] for a in agents]),
                "state": np.asarray(info["state"], dtype=np.float32),
                "legals": np.stack([np.asarray(info["legals"][a]) for a in agents]),
            }
        )
        obs, rew, terminal, trunc, info = env.step({a: actions[a] for a in agents})
        rewards.append(_mean_reward(rew))
        if "battle_won" in info:
            won = bool(info["battle_won"])
        if _all_done(terminal, trunc):
            break
    for i, step in enumerate(steps):
        step["reward"] = rewards[i]
        step["terminated"] = 1.0 if i == len(steps) - 1 else 0.0
    return {"steps": steps, "length": len(steps), "won": won, "rewards": rewards}


def replay_from(env, learner, prefix: List[np.ndarray], forced: Tuple[int, ...],
                probe_state: np.ndarray, gamma: float) -> Optional[Dict[str, Any]]:
    """Relaunch, replay the prefix, force `forced` at the probe step, then greedy.

    Returns None when the replayed prefix does not reproduce the recorded state
    bitwise, in which case the measurement is dropped.
    """
    obs, info = reset_env(env)
    state = learner.new_episode_state()
    for action in prefix:
        act = {f"agent_{i}": int(action[i]) for i in range(len(action))}
        obs, rew, terminal, trunc, info = env.step(act)
    if not np.array_equal(np.asarray(info["state"], dtype=np.float32), probe_state):
        return None
    act = {f"agent_{i}": int(forced[i]) for i in range(len(forced))}
    obs, rew, terminal, trunc, info = env.step(act)
    rewards = [_mean_reward(rew)]
    won = None
    if "battle_won" in info:
        won = bool(info["battle_won"])
    while not _all_done(terminal, trunc):
        agents = sorted(obs.keys())
        actions = learner.select_actions_eval(
            obs, {a: info["legals"][a] for a in agents}, 0.0, state,
            env_state=info.get("state"),
        )
        obs, rew, terminal, trunc, info = env.step({a: actions[a] for a in agents})
        rewards.append(_mean_reward(rew))
        if "battle_won" in info:
            won = bool(info["battle_won"])
    return {
        "rewards": rewards,
        "g": discounted_return(rewards, 0, gamma),
        "won": won,
        "steps": len(rewards),
    }


# --------------------------------------------------------------------------- #
# state bookkeeping
# --------------------------------------------------------------------------- #
def joint_key(row) -> Tuple[int, ...]:
    if isinstance(row, torch.Tensor):
        row = row.detach().cpu().tolist()
    return tuple(int(v) for v in row)


def probe_layout(env) -> Dict[str, Any]:
    inner = env._environment
    return {
        "n_allies": int(inner.n_agents),
        "nf_ally": int(inner.get_ally_num_attributes()),
        "n_enemies": int(inner.n_enemies),
        "nf_enemy": int(inner.get_enemy_num_attributes()),
    }


def alive_counts(state: np.ndarray, layout: Dict[str, Any]) -> Tuple[int, int]:
    n_al, nf_al = layout["n_allies"], layout["nf_ally"]
    n_en, nf_en = layout["n_enemies"], layout["nf_enemy"]
    allies = sum(1 for i in range(n_al) if state[i * nf_al] > 0.0)
    base = n_al * nf_al
    enemies = sum(1 for i in range(n_en) if state[base + i * nf_en] > 0.0)
    return int(allies), int(enemies)


def offline_joint_counts(dataset) -> Dict[Tuple[int, ...], int]:
    filled = dataset.filled.squeeze(-1) > 0.5
    actions = dataset.actions.squeeze(-1).long()
    observed = actions[filled].cpu()
    unique, counts = torch.unique(observed, dim=0, return_counts=True)
    return {joint_key(unique[i]): int(counts[i]) for i in range(unique.shape[0])}


def _lcg_pairs(pool: List[Tuple[int, int]], seed: int, t: int, limit: int):
    """Deterministic subsample of pair indices (no randomness in the runtime)."""
    if not pool:
        return []
    state = (seed * 1000003 + t * 7919 + 17) % (2**31 - 1)
    picked: List[Tuple[int, int]] = []
    seen = set()
    guard = 0
    while len(picked) < min(limit, len(pool)) and guard < 4 * len(pool) + 8:
        state = (1103515245 * state + 12345) % (2**31 - 1)
        guard += 1
        cand = pool[state % len(pool)]
        if cand in seen:
            continue
        seen.add(cand)
        picked.append(cand)
    return picked


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/uot_5m_vs_6m_fork_branch.yaml")
    ap.add_argument(
        "--fork-snapshot",
        default="checkpoints/uot_fork_5m_vs_6m_seed1_formal/fork_snapshot_main_t1009049.pt",
    )
    ap.add_argument("--checkpoint", action="append", default=[], help="name=path")
    ap.add_argument("--env-seed0", type=int, default=0)
    ap.add_argument("--env-seeds", type=int, default=64)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--stride", type=int, default=1, help="probe every stride-th decision step")
    ap.add_argument("--p2-stride", type=int, default=8,
                    help="subsample stride for the general hinge-pair family")
    ap.add_argument("--p2-pairs", type=int, default=2,
                    help="hinge pairs per subsampled state for family P2")
    ap.add_argument("--dry-run", action="store_true",
                    help="analytic pass only: report the replay count and stop")
    ap.add_argument("--out-json", default="logs/rank_flip_seed1.json")
    ap.add_argument("--out-md", default="documentation/experiment_reports/taskB_rank_flip_seed1.md")
    args = ap.parse_args()

    ensure_project_root()
    cfg = OmegaConf.to_container(OmegaConf.load(resolve_path(args.config)), resolve=True)
    gamma = float(cfg["system"]["discount"])
    sys_cfg = cfg["system"]
    k = int(sys_cfg.get("uot_k", 8))
    max_hamming = int(sys_cfg.get("uot_max_hamming", 2))
    delta = float(sys_cfg.get("uot_delta", 0.1))
    margin = float(sys_cfg.get("uot_margin", 0.1))

    branches = [tuple(item.split("=", 1)) for item in args.checkpoint]

    dataset = load_vault_dataset(cfg)
    ref, _ = _build_learner(cfg, snapshot=resolve_path(args.fork_snapshot))
    support = ref.support
    learners: List[Tuple[str, Any]] = [("ref", ref)]
    for name, path in branches:
        learner, _ = _build_learner(cfg, checkpoint=resolve_path(path), frozen_from=ref)
        learners.append((name, learner))
    policy_names = [n for n, _ in learners]
    print(
        f"[setup] policies={policy_names} k={k} hamming<={max_hamming} "
        f"margin={margin} delta={delta} gamma={gamma}",
        flush=True,
    )

    seeds = list(range(args.env_seed0, args.env_seed0 + args.env_seeds))

    env0 = make_env(cfg, seed=seeds[0])
    try:
        layout = probe_layout(env0)
    finally:
        close_env(env0)
    print(f"[layout] {layout}", flush=True)

    # ---------------- phase A: reference trajectories ---------------- #
    t_start = time.perf_counter()
    trajectories: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        env = make_env(cfg, seed=seed)
        try:
            trajectories[seed] = collect_trajectory(env, ref)
        finally:
            close_env(env)
        print(
            f"[traj] env_seed={seed} steps={trajectories[seed]['length']} "
            f"win={trajectories[seed]['won']}",
            flush=True,
        )
    print(
        f"[traj] {len(seeds)} trajectories in {time.perf_counter()-t_start:.1f}s",
        flush=True,
    )

    episodes = [{"steps": trajectories[s]["steps"]} for s in seeds]
    batch = stack_episodes(episodes, int(dataset.stats.max_seq_length), ref.device)
    # Drop the bootstrap slot on the TIME axis, as uot/learner.py does with
    # mac_out[:, :-1]. (scripts/probe_fork_branches.py slices the batch axis
    # instead and silently drops its last episode.)
    avail_t = batch["avail_actions"][:, :-1]
    actions_t = batch["actions"][:, :-1]
    states_t = batch["state"][:, :-1]

    # ---------------- phase B: analytic pass ---------------- #
    # The candidate set is built ONCE from the deployed policy and then held
    # fixed, so every policy is scored on the same pairs.
    with torch.inference_mode():
        ref_mac = ref._forward_qs(batch, target=False)
        bar_out = ref._forward_net(ref.frozen_q_network, batch)
        q_bar_agent = bar_out[:, :-1]
        cand, valid = build_recorded_neighborhood(
            ref_mac[:, :-1], q_bar_agent, avail_t, actions_t, support,
            k=k, max_hamming=max_hamming,
        )
        q_bar = ref._qtot_candidates(
            ref.frozen_mixer, q_bar_agent, cand, states_t
        )[:, :, :, 0]
        q_live: Dict[str, torch.Tensor] = {}
        q_tgt: Dict[str, torch.Tensor] = {}
        for name, learner in learners:
            mac_out = learner._forward_qs(batch, target=False)
            tgt_out = learner._forward_qs(batch, target=True)
            q_live[name] = learner._qtot_candidates(
                learner.mixer, mac_out[:, :-1], cand, states_t
            )[:, :, :, 0]
            q_tgt[name] = learner._qtot_candidates(
                learner.target_mixer, tgt_out[:, :-1], cand, states_t
            )[:, :, :, 0]
            print(f"[analytic] {name}: live/target Q on the shared candidate set ready",
                  flush=True)

    counts = offline_joint_counts(dataset)
    fork_t = int(getattr(ref, "t_env", 0))
    eta_of = {name: float(learner.eta_at(fork_t)) for name, learner in learners}

    records: List[Dict[str, Any]] = []
    # job key -> joint action to force. Deduplicated across pairs and policies.
    jobs: Dict[Tuple[str, int, int, Tuple[int, ...]], Dict[str, Any]] = {}
    free_returns: Dict[Tuple[str, int, int, Tuple[int, ...]], float] = {}
    n_states = n_cand_states = n_agree = n_exec_in_cand = 0

    def enqueue(name: str, seed: int, t: int, key: Tuple[int, ...]) -> None:
        jkey = (name, seed, t, key)
        if jkey in jobs or jkey in free_returns:
            return
        jobs[jkey] = {"policy": name, "env_seed": seed, "t": t, "joint": key}

    for bi, seed in enumerate(seeds):
        traj = trajectories[seed]
        length = traj["length"]
        for t in range(0, length, args.stride):
            n_states += 1
            step = traj["steps"][t]
            exec_key = joint_key(step["actions"])
            allies, enemies = alive_counts(step["state"], layout)
            # the deployed policy's own return from t is already known
            free_returns[("ref", seed, t, exec_key)] = discounted_return(
                traj["rewards"], t, gamma
            )
            idx = [i for i in range(valid.shape[0]) if bool(valid[i, bi, t])]
            if len(idx) < 2:
                continue
            n_cand_states += 1
            keys = [joint_key(cand[i, bi, t]) for i in idx]
            pos = {key: j for j, key in enumerate(keys)}
            bar = [float(q_bar[i, bi, t]) for i in idx]
            i_bar = idx[int(np.argmax(bar))]
            live_argmax: Dict[str, int] = {}
            for name in policy_names:
                live = [float(q_live[name][idx[j], bi, t]) for j in range(len(idx))]
                live_argmax[name] = idx[int(np.argmax(live))]
            if exec_key in pos:
                n_exec_in_cand += 1
            i_live_ref = live_argmax["ref"]
            if i_live_ref == i_bar:
                n_agree += 1
                continue

            def add(family: str, ia: int, ib: int, policies: List[str]) -> None:
                pa, pb = pos[keys[idx.index(ia)]], pos[keys[idx.index(ib)]]
                ka, kb = keys[pa], keys[pb]
                for name in policies:
                    live = [float(q_live[name][idx[j], bi, t]) for j in range(len(idx))]
                    tgt = [float(q_tgt[name][idx[j], bi, t]) for j in range(len(idx))]
                    sign = 1.0 if bar[pa] > bar[pb] else -1.0
                    slack = delta - sign * (live[pa] - live[pb])
                    records.append(
                        {
                            "family": family,
                            "env_seed": seed,
                            "t": t,
                            "policy": name,
                            "key_a": list(ka),
                            "key_b": list(kb),
                            "q_off_a": bar[pa],
                            "q_off_b": bar[pb],
                            "q_live_a": live[pa],
                            "q_live_b": live[pb],
                            "q_tgt_a": tgt[pa],
                            "q_tgt_b": tgt[pb],
                            "abs_q_gap_off": abs(bar[pa] - bar[pb]),
                            "abs_q_gap_live": abs(live[pa] - live[pb]),
                            "abs_q_gap_tgt": abs(tgt[pa] - tgt[pb]),
                            "sign_disagree": bool(
                                (live[pa] > live[pb]) != (bar[pa] > bar[pb])
                            ),
                            "hinge_active": bool(slack > 0),
                            "hinge_slack": float(slack),
                            "eta": eta_of[name],
                            "count_a": int(counts.get(ka, 0)),
                            "count_b": int(counts.get(kb, 0)),
                            "allies": allies,
                            "enemies": enemies,
                            "phase": t / max(1, length - 1),
                            "n_cand": len(idx),
                            "exec_key": list(exec_key),
                            "a_is_exec": bool(ka == exec_key),
                            "b_is_exec": bool(kb == exec_key),
                            "a_is_live_argmax": bool(ia == live_argmax[name]),
                            "b_is_live_argmax": bool(ib == live_argmax[name]),
                            "a_is_teacher_argmax": bool(ia == i_bar),
                            "b_is_teacher_argmax": bool(ib == i_bar),
                        }
                    )
                    enqueue(name, seed, t, ka)
                    enqueue(name, seed, t, kb)

            # family P1: the decision-relevant pair, scored under every policy
            add("P1", i_bar, i_live_ref, policy_names)

            # family P2: deterministic subsample of the remaining hinge pairs,
            # scored under the deployed policy only
            if t % args.p2_stride == 0:
                pool = [
                    (idx[x], idx[y])
                    for x in range(len(idx))
                    for y in range(x + 1, len(idx))
                    if {idx[x], idx[y]} != {i_bar, i_live_ref}
                ]
                for ia, ib in _lcg_pairs(pool, seed, t, args.p2_pairs):
                    add("P2", ia, ib, ["ref"])

    n_jobs = len(jobs)
    est = n_jobs * 6.3 / max(1, args.workers)
    print(
        f"[analytic] states={n_states} with_candidates={n_cand_states} "
        f"argmax_agree={n_agree} exec_in_cand={n_exec_in_cand} records={len(records)} "
        f"replay_jobs={n_jobs} free_returns={len(free_returns)} "
        f"est_wall={est/60:.1f} min at {args.workers} workers",
        flush=True,
    )
    if args.dry_run:
        return

    # ---------------- phase C: branch replays ---------------- #
    # Jobs are grouped by environment seed and each group runs sequentially in a
    # single worker thread. Sharing one env instance across threads races on the
    # SC2 process relaunch ("Already in the process of starting a game"), and one
    # env object is bound to one seed anyway.
    jobs_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    for job in jobs.values():
        jobs_by_seed.setdefault(int(job["env_seed"]), []).append(job)
    learners_by_name = dict(learners)

    def run_seed_group(seed: int) -> Dict[Tuple[str, int, int, Tuple[int, ...]], Optional[dict]]:
        traj = trajectories[seed]
        out: Dict[Tuple[str, int, int, Tuple[int, ...]], Optional[dict]] = {}
        env = make_env(cfg, seed=seed)
        done = 0
        try:
            for job in jobs_by_seed[seed]:
                key = (job["policy"], seed, job["t"], job["joint"])
                prefix = [s["actions"] for s in traj["steps"][:job["t"]]]
                try:
                    res = replay_from(
                        env, learners_by_name[job["policy"]], prefix, job["joint"],
                        traj["steps"][job["t"]]["state"], gamma,
                    )
                except Exception as exc:  # noqa: BLE001 - one bad replay must not kill the run
                    print(
                        f"[replay] retry after {type(exc).__name__} on "
                        f"{job['policy']} seed={seed} t={job['t']}",
                        flush=True,
                    )
                    close_env(env)
                    env = make_env(cfg, seed=seed)
                    try:
                        res = replay_from(
                            env, learners_by_name[job["policy"]], prefix, job["joint"],
                            traj["steps"][job["t"]]["state"], gamma,
                        )
                    except Exception:  # noqa: BLE001
                        res = None
                out[key] = res
                done += 1
                if done % 100 == 0:
                    print(f"[replay] seed={seed} {done}/{len(jobs_by_seed[seed])}", flush=True)
        finally:
            close_env(env)
        return out

    t_start = time.perf_counter()
    # Keys whose return is already known from the reference trajectory (the
    # deployed policy's own executed action) need no replay. They were never
    # enqueued, so seed the results with them here; the assembler looks every
    # record's two actions up in this dict.
    results: Dict[Tuple[str, int, int, Tuple[int, ...]], Optional[dict]] = {
        key: {"g": value, "won": None, "steps": None}
        for key, value in free_returns.items()
    }
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for chunk in pool.map(run_seed_group, sorted(jobs_by_seed)):
            results.update(chunk)
    n_ok = sum(1 for v in results.values() if v is not None)
    print(
        f"[replay] {len(results)} lookups, {n_ok} usable, "
        f"{len(results)-n_ok} dropped, {time.perf_counter()-t_start:.1f}s",
        flush=True,
    )

    # ---------------- phase D: assemble ---------------- #
    rows: List[Dict[str, Any]] = []
    for rec in records:
        name, seed, t = rec["policy"], rec["env_seed"], rec["t"]
        key_a, key_b = tuple(rec["key_a"]), tuple(rec["key_b"])
        ra = results.get((name, seed, t, key_a))
        rb = results.get((name, seed, t, key_b))
        if ra is None or rb is None:
            continue
        rec = dict(rec)
        rec["g_a"] = float(ra["g"])
        rec["g_b"] = float(rb["g"])
        rec["dg"] = float(ra["g"]) - float(rb["g"])
        rec["dq_off"] = float(rec["q_off_a"]) - float(rec["q_off_b"])
        if abs(rec["dq_off"]) <= 1e-8 or abs(rec["dg"]) <= RETURN_TIE:
            rec["outcome"] = "tie"
        elif (rec["dq_off"] > 0) == (rec["dg"] > 0):
            rec["outcome"] = "agree"
        else:
            rec["outcome"] = "flip"
        rows.append(rec)
    print(
        f"[assemble] {len(rows)} usable records "
        f"({len(records) - len(rows)} dropped by replay mismatch)",
        flush=True,
    )

    payload = {
        "config": args.config,
        "fork_snapshot": args.fork_snapshot,
        "branches": [n for n, _ in branches],
        "gamma": gamma,
        "env_seeds": seeds,
        "stride": args.stride,
        "p2_stride": args.p2_stride,
        "p2_pairs": args.p2_pairs,
        "n_states": n_states,
        "n_states_with_candidates": n_cand_states,
        "n_argmax_agree": n_agree,
        "n_exec_in_cand": n_exec_in_cand,
        "eta_at_fork": eta_of,
        "layout": layout,
        "rows": rows,
    }
    out = Path(resolve_path(args.out_json))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[write] {out}", flush=True)

    from probe_rank_flip_report import write_report  # noqa: E402

    write_report(payload, resolve_path(args.out_md))


if __name__ == "__main__":
    main()
