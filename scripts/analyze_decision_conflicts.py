#!/usr/bin/env python3
"""Decision-level audit of ranking flips at states visited by online policies.

The existing reliability measurement scores the offline teacher on recorded
offline states with live_q=q. This script measures the production question:

  at states actually visited by an online policy, does the live greedy joint
  differ from the teacher's greedy joint, and when it does, which joint has the
  higher branch-replay return?

Protocol:
  * base rollouts are greedy runs of the fork continuation arms;
  * at early/mid/late probe steps of each rollout, every evaluated policy
    replays the same action prefix and reports its greedy joint at that state;
  * each distinct joint is forced once, then the episode finishes greedily
    under a common continuation policy;
  * the production candidate pool (S_off, K=8, Hamming<=2, margin=0.1) and the
    ordinal hinge are evaluated at the same state with the evaluated policy's
    live Q and the teacher's Q.

Teacher choice: the fork arms were trained against the 10M-final Q-bar, so that
is the primary teacher here. The 9.16M best critic is included as a sensitivity
check because the task-2 reliability report measured that checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("SC2PATH", "/home/jxy/StarCraftII")

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.factory import build_learner, load_vault_dataset, make_env  # noqa: E402
from utils.checkpoint import load_qmix_checkpoint  # noqa: E402
from utils.paths import ensure_project_root, resolve_path  # noqa: E402
from uot.hinge import ordinal_hinge  # noqa: E402
from uot.neighborhood import build_recorded_neighborhood  # noqa: E402
from uot.support import RecordedJoints  # noqa: E402

from scripts.verify_local_rank_inversion import (  # noqa: E402
    RETURN_TIE,
    _agent_names,
    _forward_qs,
    _legal_tensor,
    _mix_joint_q,
    _new_hidden,
    close_env,
    discounted_return,
    joint_key,
    reset_env,
)
from scripts.verify_rank_reliability import branch_return  # noqa: E402

FRACTIONS = (0.2, 0.5, 0.8)
KEY_SEP = ","


def key_str(u) -> str:
    return KEY_SEP.join(str(int(v)) for v in u)


def key_tuple(s: str) -> tuple[int, ...]:
    return tuple(int(v) for v in s.split(KEY_SEP) if v != "")


def load_policy(cfg: dict, dataset, path: str):
    c = copy.deepcopy(cfg)
    c["learner_type"] = "ovmse_paper"
    c["system"]["use_ovm"] = False
    learner = build_learner(c, dataset)
    load_qmix_checkpoint(resolve_path(path), learner, load_optimizer=False)
    learner.q_network.eval()
    learner.mixer.eval()
    return learner


def greedy_tensor(q: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    return q.masked_fill(~legal, -1e9).argmax(dim=-1)


def rollout_base(env, actor, teacher, gamma: float, max_steps: int = 400) -> dict:
    names = _agent_names(actor.n_agents)
    rnn_a = _new_hidden(actor)
    rnn_t = _new_hidden(teacher)
    last: dict[str, int] = {}
    obs, info = reset_env(env)
    steps = []
    won = None
    t = 0
    while True:
        state = np.asarray(info["state"], dtype=np.float32).copy()
        legal = _legal_tensor(info["legals"], names, actor.n_actions, actor.device)
        with torch.inference_mode():
            q_a = _forward_qs(actor, obs, rnn_a, last)
            q_t = _forward_qs(teacher, obs, rnn_t, last)
        chosen = greedy_tensor(q_a, legal)
        act_tuple = tuple(int(chosen[i]) for i in range(actor.n_agents))
        steps.append(
            {
                "t": t,
                "state": state,
                "legal": legal.detach().cpu(),
                "q_actor": q_a.detach().cpu(),
                "q_teacher": q_t.detach().cpu(),
                "action": act_tuple,
            }
        )
        act = {names[i]: int(chosen[i].item()) for i in range(actor.n_agents)}
        for agent in names:
            last[agent] = act[agent]
        obs, rew, terminal, trunc, info = env.step(act)
        if "battle_won" in info:
            won = bool(info["battle_won"])
        t += 1
        if all(terminal.values()) or all(trunc.values()) or t > max_steps:
            break
    return {"steps": steps, "won": won, "length": t}


def greedy_at_probe(env, learner, prefix, probe_t: int, base_state):
    names = _agent_names(learner.n_agents)
    rnn = _new_hidden(learner)
    last: dict[str, int] = {}
    obs, info = reset_env(env)
    for t in range(probe_t):
        legal = _legal_tensor(info["legals"], names, learner.n_actions, learner.device)
        with torch.inference_mode():
            _ = _forward_qs(learner, obs, rnn, last)
        act = {names[i]: int(prefix[t][i]) for i in range(learner.n_agents)}
        for agent in names:
            last[agent] = act[agent]
        obs, rew, terminal, trunc, info = env.step(act)
        if all(terminal.values()) or all(trunc.values()):
            return None, "episode_end"
    state = np.asarray(info["state"], dtype=np.float32)
    if not np.array_equal(state, base_state):
        return None, "state_mismatch"
    legal = _legal_tensor(info["legals"], names, learner.n_actions, learner.device)
    with torch.inference_mode():
        q = _forward_qs(learner, obs, rnn, last)
    return {"joint": greedy_tensor(q, legal), "q": q, "legal": legal}, None


def pool_stats(learner, live_q, teacher, bar_q, legal, executed, support, state, args) -> dict:
    n_agents, n_actions = live_q.shape
    cand, valid = build_recorded_neighborhood(
        live_q.view(1, 1, n_agents, n_actions),
        bar_q.view(1, 1, n_agents, n_actions),
        legal.view(1, 1, n_agents, n_actions),
        torch.as_tensor(executed, dtype=torch.long, device=live_q.device).view(1, 1, n_agents),
        support,
        k=args.k,
        max_hamming=args.max_hamming,
        candidate_support="joint",
    )
    joints = cand[:, 0, 0]
    q_live = _mix_joint_q(learner, live_q, joints, state)
    q_bar = _mix_joint_q(teacher, bar_q, joints, state)
    _, hinge_stats = ordinal_hinge(
        q_live.view(-1, 1, 1, 1),
        q_bar.view(-1, 1, 1, 1),
        cand,
        valid,
        torch.ones(1, 1, dtype=torch.bool, device=cand.device),
        delta=args.delta,
        margin=args.margin,
    )
    keys = {key_str(joint_key(joints[i])) for i in range(joints.shape[0]) if bool(valid[i, 0, 0])}
    live_g = key_str(joint_key(greedy_tensor(live_q, legal)))
    bar_g = key_str(joint_key(greedy_tensor(bar_q, legal)))
    out = dict(hinge_stats)
    out.update(
        {
            "n_valid": int(valid.sum().item()),
            "n_pool": int(joints.shape[0]),
            "live_greedy_in_pool": float(live_g in keys),
            "qbar_greedy_in_pool": float(bar_g in keys),
        }
    )
    return out


def measure(args) -> dict:
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    cfg["system"]["device"] = args.device
    dataset = load_vault_dataset(cfg)
    stats = dataset.stats
    support = RecordedJoints.from_episode_actions(dataset.actions, dataset.filled, stats.n_actions)
    teacher = load_policy(cfg, dataset, args.teacher)
    teacher_ref = load_policy(cfg, dataset, args.teacher_ref) if args.teacher_ref else None
    arms = {}
    for spec in args.arm:
        tag, path = spec.split("=", 1)
        arms[tag] = load_policy(cfg, dataset, path)
    base_tags = [t for t in args.base_arms.split(",") if t]
    cont_tags = [t for t in args.continuations.split(",") if t]
    conts = {t: (teacher if t == "teacher" else arms[t]) for t in cont_tags}

    records = []
    map_seeds = [args.seed0 + i for i in range(args.maps)]
    for mi, map_seed in enumerate(map_seeds):
        env = make_env(cfg, seed=map_seed)
        try:
            for base_tag in base_tags:
                base = rollout_base(env, arms[base_tag], teacher, args.gamma, args.max_steps)
                L = base["length"]
                if L < 4:
                    continue
                probe_steps = sorted(
                    {int(min(max(round(f * L), 1), L - 2)) for f in FRACTIONS}
                )[: args.max_probes]
                for probe_t in probe_steps:
                    rec = base["steps"][probe_t]
                    prefix = [s["action"] for s in base["steps"][:probe_t]]
                    state = rec["state"]
                    legal_cpu = rec["legal"]
                    q_actor = rec["q_actor"].to(arms[base_tag].device)
                    q_teacher = rec["q_teacher"].to(teacher.device)
                    legal = legal_cpu.to(arms[base_tag].device)
                    executed = rec["action"]

                    joints = {}
                    q_live_at = {}
                    pool = {}
                    for tag, learner in arms.items():
                        got, err = greedy_at_probe(env, learner, prefix, probe_t, state)
                        if err is not None:
                            continue
                        joint = joint_key(got["joint"])
                        if tag == base_tag and joint != executed:
                            continue
                        joints[tag] = joint
                        q_live_at[tag] = got["q"].detach().cpu()
                    if not joints:
                        continue
                    # Teacher greedy at the same state, with the same prefix.
                    got_t, err = greedy_at_probe(env, teacher, prefix, probe_t, state)
                    if err is not None:
                        continue
                    teacher_joint = joint_key(got_t["joint"])
                    joints["teacher"] = teacher_joint
                    q_live_at["teacher"] = got_t["q"].detach().cpu()
                    if teacher_ref is not None:
                        got_r, err = greedy_at_probe(env, teacher_ref, prefix, probe_t, state)
                        if err is None:
                            joints["teacher_ref"] = joint_key(got_r["joint"])
                            q_live_at["teacher_ref"] = got_r["q"].detach().cpu()

                    # Q-bar values for every distinct joint, under the primary teacher.
                    distinct = sorted(set(joints.values()))
                    qbar = {}
                    qbar_ref = {}
                    for u in distinct:
                        u_t = torch.as_tensor(u, dtype=torch.long, device=teacher.device)
                        qbar[key_str(u)] = float(
                            _mix_joint_q(teacher, q_teacher, u_t.view(1, -1), state).item()
                        )
                        if teacher_ref is not None and "teacher_ref" in q_live_at:
                            q_ref = q_live_at["teacher_ref"].to(teacher_ref.device)
                            qbar_ref[key_str(u)] = float(
                                _mix_joint_q(teacher_ref, q_ref, u_t.view(1, -1), state).item()
                            )

                    # Forced returns under common continuations.
                    returns = {c: {} for c in cont_tags}
                    for ctag, cont in conts.items():
                        for u in distinct:
                            u_t = torch.as_tensor(u, dtype=torch.long, device=cont.device)
                            ret, err = branch_return(
                                env, cont, prefix, probe_t, u_t, state, args.gamma
                            )
                            if ret is not None:
                                returns[ctag][key_str(u)] = float(ret)

                    # Production pool for each live policy at this state.
                    pool = {}
                    for tag, learner in arms.items():
                        if tag not in q_live_at:
                            continue
                        lq = q_live_at[tag].to(learner.device)
                        llegal = legal_cpu.to(learner.device)
                        pool[tag] = pool_stats(
                            learner, lq, teacher, q_teacher, llegal, executed, support, state, args
                        )
                    pool["teacher"] = pool_stats(
                        teacher, q_teacher, teacher, q_teacher, legal_cpu.to(teacher.device),
                        executed, support, state, args
                    )

                    records.append(
                        {
                            "map_seed": map_seed,
                            "base_arm": base_tag,
                            "probe_t": probe_t,
                            "length": L,
                            "phase": "early" if probe_t / L < 0.35 else ("late" if probe_t / L > 0.65 else "mid"),
                            "won": base["won"],
                            "joints": {k: list(v) for k, v in joints.items()},
                            "qbar": qbar,
                            "qbar_ref9m": qbar_ref,
                            "returns": returns,
                            "pool": pool,
                        }
                    )
                    print(
                        f"[map {mi+1}/{len(map_seeds)} {base_tag} t={probe_t}/{L}] "
                        f"joints={ {k: v for k, v in joints.items()} } "
                        f"ret_keys={ {c: len(returns[c]) for c in cont_tags} }",
                        flush=True,
                    )
        finally:
            close_env(env)
    return {"records": records, "summary": summarize(records, cont_tags)}


def summarize(records: list[dict], cont_tags: list[str]) -> dict:
    summary: dict = {"n_probes": len(records), "by_base": {}, "conflicts": {}, "pool": {}, "disagreement": {}}
    for r in records:
        summary["by_base"][r["base_arm"]] = summary["by_base"].get(r["base_arm"], 0) + 1

    eval_tags = sorted({k for r in records for k in r["joints"]} - {"teacher", "teacher_ref"})
    # Pairwise greedy disagreement at the same state.
    for i, a in enumerate(eval_tags + ["teacher"]):
        for b in (eval_tags + ["teacher"])[i + 1:]:
            n = n_diff = 0
            for r in records:
                if a in r["joints"] and b in r["joints"]:
                    n += 1
                    n_diff += int(r["joints"][a] != r["joints"][b])
            if n:
                summary["disagreement"][f"{a}_vs_{b}"] = {"n": n, "diff": n_diff, "rate": n_diff / n}

    # Conflict and forced-return comparisons vs the teacher.
    for tag in eval_tags:
        for ctag in cont_tags:
            vals, decided, better, worse = [], 0, 0, 0
            regret = []
            for r in records:
                if tag not in r["joints"] or "teacher" not in r["joints"]:
                    continue
                u = key_str(r["joints"][tag])
                t = key_str(r["joints"]["teacher"])
                if u == t or u not in r["qbar"]:
                    continue
                if abs(r["qbar"][u] - r["qbar"][t]) < 0.1:
                    continue
                if u not in r["returns"][ctag] or t not in r["returns"][ctag]:
                    continue
                d = r["returns"][ctag][u] - r["returns"][ctag][t]
                vals.append(d)
                if abs(d) > RETURN_TIE:
                    decided += 1
                    better += int(d > 0)
                    worse += int(d < 0)
                avail = r["returns"][ctag]
                best = max(avail.values())
                regret.append(best - avail[u])
            if vals:
                summary["conflicts"].setdefault(tag, {})[ctag] = {
                    "n": len(vals),
                    "decided": decided,
                    "live_better": better,
                    "live_worse": worse,
                    "live_better_frac_decided": better / decided if decided else None,
                    "mean_diff": statistics.fmean(vals),
                    "median_diff": statistics.median(vals),
                    "mean_abs_diff": statistics.fmean(abs(v) for v in vals),
                    "median_abs_diff": statistics.median(abs(v) for v in vals),
                    "p90_abs_diff": float(np.percentile(np.abs(vals), 90)),
                    "regret_mean": statistics.fmean(regret) if regret else None,
                    "regret_median": statistics.median(regret) if regret else None,
                    "regret_gt1_frac": sum(v > 1.0 for v in regret) / len(regret) if regret else None,
                }

    for tag in sorted({k for r in records for k in r["pool"]}):
        rows = [r["pool"][tag] for r in records if tag in r["pool"]]
        if not rows:
            continue
        summary["pool"][tag] = {
            "n": len(rows),
            "live_greedy_in_pool": statistics.fmean(x["live_greedy_in_pool"] for x in rows),
            "qbar_greedy_in_pool": statistics.fmean(x["qbar_greedy_in_pool"] for x in rows),
            "n_valid_mean": statistics.fmean(x["n_valid"] for x in rows),
            "hinge_rate_mean": statistics.fmean(x["uot_hinge_rate"] for x in rows),
            "sign_disagree_mean": statistics.fmean(x["uot_sign_disagree"] for x in rows),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/matched_protocol_5m_vs_6m.yaml")
    ap.add_argument("--teacher", default="checkpoints/ovmse_paper_5m_vs_6m/ovmse_paper_seed1.pt")
    ap.add_argument("--teacher-ref", default="checkpoints/ovmse_paper_5m_vs_6m/ovmse_paper_seed1_best.pt")
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--base-arms", default="continue,off")
    ap.add_argument("--continuations", default="teacher,continue")
    ap.add_argument("--maps", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--max-probes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-hamming", type=int, default=2)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="logs/decision_conflicts_seed1.json")
    args = ap.parse_args()
    if not args.arm:
        args.arm = [
            "continue=checkpoints/uot_fork_5m_vs_6m_seed1_formal/uot_online_fork_resume_continue_seed1.pt",
            "off=checkpoints/uot_fork_5m_vs_6m_seed1_formal/uot_online_fork_resume_off_seed1.pt",
            "decay=checkpoints/uot_fork_5m_vs_6m_seed1_formal/uot_online_fork_resume_decay_seed1.pt",
        ]
    ensure_project_root()
    payload = measure(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
