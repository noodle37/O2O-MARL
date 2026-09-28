#!/usr/bin/env python3
"""Verify what an offline QMIX checkpoint transfers, with no parameter updates.

On the frozen checkpoint, two decision rules share the candidate set
U ∩ legal, where U is the set of joint actions that appear in the offline
data. Rank takes argmax of Q_tot. Shuffle keeps that same multiset of Q_tot
values, reassigns them at random, then takes argmax. The network's own
per-agent greedy joint action is recorded on the Rank rollouts.
"""

from __future__ import annotations

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

# One verification pass, not a sweep. Eight episodes is enough to see whether
# the Rank/Shuffle return gap and the greedy-in-U rate go the same way.
N_EPISODES = 8
MIXER_CHUNK = 8192


def observed_joint_actions(dataset) -> tuple[torch.Tensor, int]:
    """Unique joint actions on filled offline steps. Shape [M, n_agents]."""
    filled = dataset.filled.squeeze(-1) > 0.5
    actions = dataset.actions.squeeze(-1).long()
    observed = actions[filled]
    if observed.numel() == 0:
        raise RuntimeError("offline dataset has no filled joint actions")
    n_actions = int(dataset.stats.n_actions)
    if int(observed.min()) < 0 or int(observed.max()) >= n_actions:
        raise RuntimeError(
            f"joint action id outside [0, {n_actions}): "
            f"min={int(observed.min())} max={int(observed.max())}"
        )
    unique = torch.unique(observed, dim=0)
    return unique.contiguous(), int(observed.shape[0])


def _agent_names(n_agents: int) -> list[str]:
    return [f"agent_{i}" for i in range(n_agents)]


def _forward_qs(learner, observations, rnn, last) -> torch.Tensor:
    """Per-agent Q at the current history. Shape [n_agents, n_actions]."""
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
            raise RuntimeError(f"legal mask for {agent} has shape {row.shape}, expected {(n_actions,)}")
        rows.append(torch.from_numpy(row))
    return torch.stack(rows, dim=0).to(device)


def _legal_support(support: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Rows of support whose every component is legal. support is [M, n_agents]."""
    ok = torch.ones(support.shape[0], dtype=torch.bool, device=support.device)
    for i in range(support.shape[1]):
        ok &= legal[i].index_select(0, support[:, i])
    return support[ok]


def _mix_joint_q(learner, agent_q: torch.Tensor, chosen: torch.Tensor, state: np.ndarray) -> torch.Tensor:
    """Q_tot for each joint action in chosen. agent_q is [n_agents, n_actions]."""
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


def _run_arm(env, learner, support: torch.Tensor, mode: str, seed: int) -> dict:
    names = _agent_names(learner.n_agents)
    generator = torch.Generator()
    generator.manual_seed(seed)
    returns = []
    wins = []
    n_steps = 0
    n_greedy_in_u = 0
    n_rank_matches_greedy = 0
    n_empty = 0
    candidate_sizes = []
    checked_multiset = mode != "shuffle"

    for episode in range(N_EPISODES):
        rnn = _new_hidden(learner)
        last: dict[str, int] = {}
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        won = None
        while not done:
            state = info.get("state")
            if state is None:
                raise RuntimeError("environment info has no global state")
            with torch.inference_mode():
                q = _forward_qs(learner, obs, rnn, last)
                legal = _legal_tensor(info["legals"], names, learner.n_actions, learner.device)
                greedy = q.masked_fill(~legal, -1e9).argmax(dim=-1)
                in_support = bool((support == greedy.view(1, -1)).all(dim=-1).any())
                candidates = _legal_support(support, legal)
                if candidates.shape[0] == 0:
                    chosen = greedy
                    n_empty += 1
                else:
                    q_tot = _mix_joint_q(learner, q, candidates, state)
                    if mode == "rank":
                        pick = int(torch.argmax(q_tot).item())
                    else:
                        perm = torch.randperm(q_tot.shape[0], generator=generator)
                        assigned = q_tot[perm.to(q_tot.device)]
                        if not checked_multiset:
                            same = torch.equal(assigned.sort().values, q_tot.sort().values)
                            if not same:
                                raise RuntimeError("shuffle did not preserve the multiset of Q_tot")
                            checked_multiset = True
                        pick = int(torch.argmax(assigned).item())
                    chosen = candidates[pick]
                    if mode == "rank":
                        n_rank_matches_greedy += int(torch.equal(chosen, greedy))
                if mode == "rank":
                    n_greedy_in_u += int(in_support)
            n_steps += 1
            candidate_sizes.append(int(candidates.shape[0]))
            actions = {names[i]: int(chosen[i].item()) for i in range(learner.n_agents)}
            for agent, action in actions.items():
                last[agent] = action
            obs, rewards, terminal, trunc, info = env.step(actions)
            ep_ret += float(np.mean([float(np.asarray(r)) for r in rewards.values()]))
            if "battle_won" in info:
                won = bool(info["battle_won"])
            done = all(terminal.values()) or all(trunc.values())
        returns.append(ep_ret)
        if won is not None:
            wins.append(float(won))
        print(
            f"[{mode}] episode {episode + 1}/{N_EPISODES} return={ep_ret:.3f} "
            f"win={won} steps_so_far={n_steps} empty_so_far={n_empty}",
            flush=True,
        )

    summary = {
        "mode": mode,
        "n_episodes": N_EPISODES,
        "returns": returns,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "win_rate": float(np.mean(wins)) if wins else None,
        "n_steps": n_steps,
        "empty_candidate_frac": float(n_empty / max(n_steps, 1)),
        "mean_candidate_size": float(np.mean(candidate_sizes)) if candidate_sizes else 0.0,
        "median_candidate_size": float(np.median(candidate_sizes)) if candidate_sizes else 0.0,
    }
    if mode == "rank":
        nonempty = max(n_steps - n_empty, 1)
        summary["greedy_in_u_frac"] = float(n_greedy_in_u / max(n_steps, 1))
        summary["rank_matches_greedy_frac"] = float(n_rank_matches_greedy / nonempty)
    return summary


def main() -> None:
    ensure_project_root()
    cfg = OmegaConf.to_container(OmegaConf.load("configs/5m_vs_6m.yaml"), resolve=True)
    ckpt = resolve_path(cfg["checkpoint_path"])
    print(f"checkpoint={ckpt}", flush=True)
    print(f"episodes_per_arm={N_EPISODES} epsilon=0 parameter_updates=0", flush=True)

    dataset = load_vault_dataset(cfg)
    # load_vault_dataset follows the training device. Unique actions only need the CPU copy.
    support_cpu, n_transitions = observed_joint_actions(dataset)
    print(
        f"offline_transitions={n_transitions} unique_joint_actions={support_cpu.shape[0]} "
        f"n_agents={dataset.stats.n_agents} n_actions={dataset.stats.n_actions}",
        flush=True,
    )
    learner = build_learner(cfg, dataset)
    del dataset
    load_qmix_checkpoint(ckpt, learner, load_optimizer=False)
    learner.q_network.eval()
    learner.mixer.eval()
    support = support_cpu.to(learner.device)
    seed = int(cfg["seed"])

    arms = {}
    for mode in ("rank", "shuffle"):
        env = make_env(cfg, seed=seed)
        try:
            arms[mode] = _run_arm(env, learner, support, mode, seed)
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()

    payload = {
        "checkpoint": ckpt,
        "n_episodes_per_arm": N_EPISODES,
        "env_seed": seed,
        "unique_joint_actions": int(support.shape[0]),
        "offline_transitions": n_transitions,
        "parameter_updates": 0,
        "arms": arms,
    }
    out = Path(resolve_path("logs")) / "verify_observed_joint_rank.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
