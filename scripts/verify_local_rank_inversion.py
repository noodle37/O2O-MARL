#!/usr/bin/env python3
"""Check whether offline joint-action order is locally inverted.

Frozen checkpoint, no parameter updates. On each environment seed the greedy
policy is rolled out, and the legal recorded joint actions are ranked by
Q_tot at the 10th decision. Three pairs are scored against the discounted
return of "take this joint action, then act greedy":

- top: ranks 1 and 2
- bottom: the adjacent pair in the lower half with the highest offline count
- long: rank 1 and the last rank

A pair is an inversion when the Q_tot order and the return order disagree.
The same seed must reproduce the greedy prefix; otherwise that seed is dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SC2PATH", "/home/jxy/StarCraftII")

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.factory import build_learner, load_vault_dataset, make_env
from utils.checkpoint import load_qmix_checkpoint
from utils.paths import ensure_project_root, resolve_path

PROBE_STEP = 9  # 0-based index of the 10th decision
MIXER_CHUNK = 8192
RETURN_TIE = 1e-4


def _agent_names(n_agents: int) -> list[str]:
    return [f"agent_{i}" for i in range(n_agents)]


def _forward_qs(learner, observations, rnn, last) -> torch.Tensor:
    names = _agent_names(learner.n_agents)
    columns = []
    for i, agent in enumerate(names):
        obs = torch.from_numpy(np.asarray(observations[agent])).float().to(learner.device)
        last_idx = last.get(agent)
        if last_idx is None:
            last_oh = torch.zeros(learner.n_actions, device=learner.device)
        else:
            last_oh = F.one_hot(
                torch.tensor(int(last_idx), device=learner.device), learner.n_actions
            ).float()
        pieces = [obs, last_oh]
        if learner.add_agent_id_to_obs:
            pieces.append(
                F.one_hot(torch.tensor(i, device=learner.device), learner.n_agents).float()
            )
        obs_in = torch.cat(pieces, dim=-1).unsqueeze(0)
        q, rnn[agent] = learner.q_network(obs_in, rnn.get(agent))
        columns.append(q[0])
    return torch.stack(columns, dim=0)


def _legal_tensor(legals, names, n_actions: int, device) -> torch.Tensor:
    rows = []
    for agent in names:
        row = np.asarray(legals[agent]).astype(bool)
        if row.shape != (n_actions,):
            raise RuntimeError(
                f"legal mask for {agent} has shape {row.shape}, expected {(n_actions,)}"
            )
        rows.append(torch.from_numpy(row))
    return torch.stack(rows, dim=0).to(device)


def _legal_support(support: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    ok = torch.ones(support.shape[0], dtype=torch.bool, device=support.device)
    for i in range(support.shape[1]):
        ok &= legal[i].index_select(0, support[:, i])
    return support[ok]


def _mix_joint_q(learner, agent_q: torch.Tensor, chosen: torch.Tensor, state: np.ndarray) -> torch.Tensor:
    if chosen.numel() == 0:
        return agent_q.new_empty(0)
    parts = [agent_q[i].index_select(0, chosen[:, i]) for i in range(chosen.shape[1])]
    stacked = torch.stack(parts, dim=1)
    st = torch.as_tensor(state, dtype=torch.float32, device=learner.device).view(1, -1)
    out = []
    for start in range(0, stacked.shape[0], MIXER_CHUNK):
        block = stacked[start : start + MIXER_CHUNK]
        k = block.shape[0]
        mixed = learner.mixer(block.view(k, 1, -1), st.view(1, 1, -1).expand(k, 1, -1))
        out.append(mixed.view(k))
    return torch.cat(out, dim=0)


def _new_hidden(learner) -> dict:
    return {
        f"agent_{i}": learner.q_network.initial_state(1, learner.device)
        for i in range(learner.n_agents)
    }


def joint_key(row) -> tuple[int, ...]:
    if isinstance(row, torch.Tensor):
        row = row.detach().cpu().tolist()
    return tuple(int(v) for v in row)


def discounted_return(rewards: list[float], start: int, gamma: float) -> float:
    acc = 0.0
    for reward in reversed(rewards[start:]):
        acc = float(reward) + gamma * acc
    return acc


def select_pairs(counts_in_rank_order: list[int]) -> dict[str, tuple[int, int]]:
    """Index pairs into a best-to-worst list."""
    n = len(counts_in_rank_order)
    pairs: dict[str, tuple[int, int]] = {}
    if n >= 2:
        pairs["top"] = (0, 1)
    if n >= 3:
        pairs["long"] = (0, n - 1)
    start = n // 2
    best = None
    best_score = None
    for i in range(start, n - 1):
        left = counts_in_rank_order[i]
        right = counts_in_rank_order[i + 1]
        score = (min(left, right), left + right, i)
        if best_score is None or score > best_score:
            best_score = score
            best = (i, i + 1)
    if best is not None and best != pairs.get("top"):
        pairs["bottom"] = best
    return pairs


def pair_outcome(q_a: float, q_b: float, g_a: float, g_b: float) -> str:
    dq = float(q_a) - float(q_b)
    dg = float(g_a) - float(g_b)
    if abs(dq) <= 1e-8 or abs(dg) <= RETURN_TIE:
        return "tie"
    if (dq > 0) == (dg > 0):
        return "agree"
    return "invert"


def _self_check() -> None:
    counts = [10, 2, 3, 8, 7]
    pairs = select_pairs(counts)
    if pairs != {"top": (0, 1), "long": (0, 4), "bottom": (3, 4)}:
        raise RuntimeError(f"pair selection self-check failed: {pairs}")
    if select_pairs([4, 1]) != {"top": (0, 1)}:
        raise RuntimeError("pair selection self-check failed for two actions")
    if pair_outcome(1.0, 0.0, 2.0, 1.0) != "agree":
        raise RuntimeError("agree self-check failed")
    if pair_outcome(1.0, 0.0, 1.0, 2.0) != "invert":
        raise RuntimeError("invert self-check failed")
    if pair_outcome(1.0, 0.0, 1.0, 1.0) != "tie":
        raise RuntimeError("tie self-check failed")


def action_counts(dataset) -> dict[tuple[int, ...], int]:
    filled = dataset.filled.squeeze(-1) > 0.5
    actions = dataset.actions.squeeze(-1).long()
    observed = actions[filled].cpu()
    if observed.numel() == 0:
        raise RuntimeError("offline dataset has no filled joint actions")
    unique, counts = torch.unique(observed, dim=0, return_counts=True)
    table = {joint_key(unique[i]): int(counts[i]) for i in range(unique.shape[0])}
    return table


def support_tensor(counts: dict[tuple[int, ...], int]) -> torch.Tensor:
    rows = [list(key) for key in counts]
    return torch.tensor(rows, dtype=torch.long)


def close_env(env) -> None:
    inner = getattr(env, "_environment", None)
    close = getattr(inner, "close", None) if inner is not None else None
    if close is not None:
        close()


def reset_env(env):
    """Relaunch the map so each episode is the same seeded start.

    SMAC applies the seed only inside create_game. A later reset kills every
    unit and draws a new start. Relaunching has to restore max_reward as well:
    init_units adds enemy health onto max_reward whenever the episode count is
    zero, and that scale would otherwise change the returns.
    """
    inner = env._environment
    if inner._sc2_proc is not None and inner._episode_count != 0:
        inner._sc2_proc.close()
        inner._sc2_proc = None
        inner._controller = None
        inner._episode_count = 0
        inner.max_reward = inner.n_enemies * inner.reward_death_value + inner.reward_win
    return env.reset()


def run_episode(env, learner, support: torch.Tensor, forced: torch.Tensor | None) -> dict:
    names = _agent_names(learner.n_agents)
    rnn = _new_hidden(learner)
    last: dict[str, int] = {}
    obs, info = reset_env(env)
    done = False
    t = 0
    rewards: list[float] = []
    actions: list[tuple[int, ...]] = []
    probe = None
    won = None
    while not done:
        state = info.get("state")
        if state is None:
            raise RuntimeError("environment info has no global state")
        with torch.inference_mode():
            q = _forward_qs(learner, obs, rnn, last)
            legal = _legal_tensor(info["legals"], names, learner.n_actions, learner.device)
            greedy = q.masked_fill(~legal, -1e9).argmax(dim=-1)
            chosen = greedy
            if t == PROBE_STEP:
                probe = {
                    "state": np.asarray(state, dtype=np.float32).copy(),
                    "greedy": greedy.detach().cpu(),
                }
                if forced is None:
                    candidates = _legal_support(support, legal)
                    q_tot = (
                        _mix_joint_q(learner, q, candidates, state)
                        if candidates.shape[0]
                        else q.new_empty(0)
                    )
                    probe["candidates"] = candidates.detach().cpu()
                    probe["q_tot"] = q_tot.detach().cpu()
                else:
                    chosen = forced.to(learner.device)
        act = {names[i]: int(chosen[i].item()) for i in range(learner.n_agents)}
        for agent, action in act.items():
            last[agent] = action
        actions.append(tuple(act[name] for name in names))
        obs, rew, terminal, trunc, info = env.step(act)
        rewards.append(float(np.mean([float(np.asarray(r)) for r in rew.values()])))
        if "battle_won" in info:
            won = bool(info["battle_won"])
        done = all(bool(np.asarray(v)) for v in terminal.values()) or all(
            bool(np.asarray(v)) for v in trunc.values()
        )
        t += 1
    return {"rewards": rewards, "actions": actions, "probe": probe, "won": won}


def _prefix_matches(base: dict, other: dict) -> bool:
    if base["probe"] is None or other["probe"] is None:
        return False
    if other["actions"][:PROBE_STEP] != base["actions"][:PROBE_STEP]:
        return False
    return bool(np.array_equal(other["probe"]["state"], base["probe"]["state"]))


def _record_pair(
    kind: str,
    seed: int,
    ranked: torch.Tensor,
    ranked_q: torch.Tensor,
    counts: dict[tuple[int, ...], int],
    ia: int,
    ib: int,
    g_of: dict[tuple[int, ...], float],
) -> dict:
    key_a = joint_key(ranked[ia])
    key_b = joint_key(ranked[ib])
    q_a = float(ranked_q[ia])
    q_b = float(ranked_q[ib])
    g_a = float(g_of[key_a])
    g_b = float(g_of[key_b])
    count_a = int(counts[key_a])
    count_b = int(counts[key_b])
    return {
        "seed": seed,
        "kind": kind,
        "rank_a": int(ia),
        "rank_b": int(ib),
        "joint_a": list(key_a),
        "joint_b": list(key_b),
        "q_a": q_a,
        "q_b": q_b,
        "g_a": g_a,
        "g_b": g_b,
        "count_a": count_a,
        "count_b": count_b,
        "min_count": min(count_a, count_b),
        "outcome": pair_outcome(q_a, q_b, g_a, g_b),
    }


def evaluate_seed(env, learner, support, counts, gamma: float, seed: int, check_replay: bool) -> dict:
    base = run_episode(env, learner, support, forced=None)
    print(
        f"[seed {seed}] greedy steps={len(base['rewards'])} "
        f"return={sum(base['rewards']):.3f} win={base['won']}",
        flush=True,
    )
    if base["probe"] is None:
        return {"seed": seed, "status": "episode_shorter_than_probe", "pairs": []}
    if base["probe"]["candidates"].shape[0] < 2:
        return {
            "seed": seed,
            "status": "fewer_than_two_candidates",
            "n_candidates": int(base["probe"]["candidates"].shape[0]),
            "pairs": [],
        }
    if check_replay:
        replay = run_episode(env, learner, support, forced=None)
        same = replay["actions"] == base["actions"] and replay["rewards"] == base["rewards"]
        diverge_at = None
        for i, (left, right) in enumerate(zip(base["actions"], replay["actions"])):
            if left != right:
                diverge_at = i
                break
        if diverge_at is None and len(base["actions"]) != len(replay["actions"]):
            diverge_at = min(len(base["actions"]), len(replay["actions"]))
        print(
            f"[seed {seed}] replay_matches={same} steps={len(replay['rewards'])} "
            f"return={sum(replay['rewards']):.3f} diverge_at={diverge_at}",
            flush=True,
        )
        if not same or not _prefix_matches(base, replay):
            return {"seed": seed, "status": "replay_mismatch", "pairs": []}

    candidates = base["probe"]["candidates"]
    q_tot = base["probe"]["q_tot"]
    order = torch.argsort(q_tot, descending=True)
    ranked = candidates[order]
    ranked_q = q_tot[order]
    rank_counts = [counts[joint_key(ranked[i])] for i in range(ranked.shape[0])]
    pairs = select_pairs(rank_counts)
    greedy_key = joint_key(base["probe"]["greedy"])
    g_of: dict[tuple[int, ...], float] = {
        greedy_key: discounted_return(base["rewards"], PROBE_STEP, gamma)
    }

    needed: dict[tuple[int, ...], torch.Tensor] = {}
    for ia, ib in pairs.values():
        needed[joint_key(ranked[ia])] = ranked[ia]
        needed[joint_key(ranked[ib])] = ranked[ib]

    for key, joint in needed.items():
        if key in g_of:
            continue
        branch = run_episode(env, learner, support, forced=joint)
        print(
            f"[seed {seed}] branch {list(key)} steps={len(branch['rewards'])} "
            f"return={sum(branch['rewards']):.3f}",
            flush=True,
        )
        if not _prefix_matches(base, branch):
            return {"seed": seed, "status": "branch_prefix_mismatch", "pairs": []}
        g_of[key] = discounted_return(branch["rewards"], PROBE_STEP, gamma)

    records = [
        _record_pair(kind, seed, ranked, ranked_q, counts, ia, ib, g_of)
        for kind, (ia, ib) in pairs.items()
    ]
    return {
        "seed": seed,
        "status": "ok",
        "n_candidates": int(ranked.shape[0]),
        "greedy_in_support": bool(greedy_key in counts),
        "greedy_is_rank1": bool(joint_key(ranked[0]) == greedy_key),
        "episode_return": float(sum(base["rewards"])),
        "win": base["won"],
        "pairs": records,
    }


def _rate(records: list[dict]) -> dict:
    decided = [r for r in records if r["outcome"] in ("agree", "invert")]
    n_inv = sum(r["outcome"] == "invert" for r in decided)
    n = len(decided)
    gaps = [abs(r["g_a"] - r["g_b"]) for r in decided]
    return {
        "n_pairs": len(records),
        "n_tie": sum(r["outcome"] == "tie" for r in records),
        "n_decided": n,
        "n_invert": n_inv,
        "concordance": None if n == 0 else (n - n_inv) / n,
        "median_abs_return_gap": None if not gaps else float(np.median(gaps)),
    }


def summarize(seed_rows: list[dict]) -> dict:
    ok = [row["pairs"] for row in seed_rows if row["status"] == "ok"]
    flat = [pair for pairs in ok for pair in pairs]
    by_kind = {}
    for kind in ("top", "bottom", "long"):
        by_kind[kind] = _rate([pair for pair in flat if pair["kind"] == kind])
    top = [pair for pair in flat if pair["kind"] == "top" and pair["outcome"] in ("agree", "invert")]
    sparse = {"median_min_count": None, "low": None, "high": None}
    if top:
        med = float(np.median([pair["min_count"] for pair in top]))
        sparse["median_min_count"] = med
        sparse["low"] = _rate([pair for pair in top if pair["min_count"] <= med])
        sparse["high"] = _rate([pair for pair in top if pair["min_count"] > med])
    return {
        "n_seeds_ok": len(ok),
        "n_seeds": len(seed_rows),
        "pooled": _rate(flat),
        "by_kind": by_kind,
        "top_by_count": sparse,
        "seed_status": {str(row["seed"]): row["status"] for row in seed_rows},
    }


def main() -> None:
    _self_check()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--seed0", type=int, default=1)
    args = parser.parse_args()

    ensure_project_root()
    cfg = OmegaConf.to_container(OmegaConf.load("configs/5m_vs_6m.yaml"), resolve=True)
    cfg["system"]["data_device"] = "cpu"
    cfg["system"]["device"] = "cpu"
    ckpt = resolve_path(cfg["checkpoint_path"])
    gamma = float(cfg["system"]["discount"])
    print(f"checkpoint={ckpt} probe_step={PROBE_STEP + 1} gamma={gamma} updates=0", flush=True)

    dataset = load_vault_dataset(cfg)
    counts = action_counts(dataset)
    support_cpu = support_tensor(counts)
    print(
        f"unique_joint_actions={support_cpu.shape[0]} "
        f"offline_transitions={sum(counts.values())}",
        flush=True,
    )
    learner = build_learner(cfg, dataset)
    del dataset
    load_qmix_checkpoint(ckpt, learner, load_optimizer=False)
    learner.q_network.eval()
    learner.mixer.eval()
    support = support_cpu.to(learner.device)

    out = Path(resolve_path("logs")) / "verify_local_rank_inversion.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    seeds = list(range(args.seed0, args.seed0 + args.n_seeds))
    for index, seed in enumerate(seeds):
        env = make_env(cfg, seed=seed)
        try:
            row = evaluate_seed(
                env,
                learner,
                support,
                counts,
                gamma,
                seed,
                check_replay=(index == 0),
            )
        finally:
            close_env(env)
        seed_rows.append(row)
        payload = {
            "checkpoint": ckpt,
            "probe_step_1based": PROBE_STEP + 1,
            "gamma": gamma,
            "parameter_updates": 0,
            "seeds": seed_rows,
            "summary": summarize(seed_rows),
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload["summary"], indent=2), flush=True)
        if row["status"] == "replay_mismatch":
            raise RuntimeError("greedy replay did not reproduce the episode; branch returns are not comparable")

    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
