# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Load OG-MARL Vault trajectories as padded PyMARL-style episode tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from flashbax.vault import Vault

from utils.paths import VAULT_DIR

SMAC_EPISODE_LIMIT: Dict[str, int] = {
    "2s3z": 120,
    "5m_vs_6m": 70,
    "6h_vs_8z": 150,
    "MMM": 150,
    "3s5z": 150,
    "3s5z_vs_3s6z": 170,
    "2halfcheetah": 1000,
}

MUJOCO_DISCRETE_ACTION_SCENARIOS = frozenset({"2halfcheetah"})


@dataclass
class EpisodeDatasetStats:
    n_episodes: int
    max_seq_length: int
    total_filled_steps: int
    median_episode_length: float
    n_agents: int
    obs_dim: int
    state_dim: int
    n_actions: int


class EpisodeDataset:
    """In-memory episodes on GPU for PyMARL-style batch sampling."""

    def __init__(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        avail_actions: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        filled: torch.Tensor,
        stats: EpisodeDatasetStats,
    ):
        self.obs = obs
        self.actions = actions
        self.state = state
        self.avail_actions = avail_actions
        self.reward = reward
        self.terminated = terminated
        self.filled = filled
        self.stats = stats
        self.device = obs.device
        self.n_episodes = int(obs.shape[0])

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> Dict[str, torch.Tensor]:
        if rng is None:
            rng = np.random.default_rng()
        n = min(batch_size, self.n_episodes)
        idx = rng.choice(self.n_episodes, size=n, replace=False)
        idx_t = torch.as_tensor(idx, device=self.device, dtype=torch.long)
        return {
            "obs": self.obs.index_select(0, idx_t),
            "actions": self.actions.index_select(0, idx_t),
            "state": self.state.index_select(0, idx_t),
            "avail_actions": self.avail_actions.index_select(0, idx_t),
            "reward": self.reward.index_select(0, idx_t),
            "terminated": self.terminated.index_select(0, idx_t),
            "filled": self.filled.index_select(0, idx_t),
        }

    @property
    def total_filled_steps(self) -> int:
        return self.stats.total_filled_steps


def _to_numpy_tree(experience) -> Dict:
    import jax

    return jax.tree.map(lambda x: np.asarray(x), experience)


def _split_episode_slices(terminals: np.ndarray, truncations: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive (start, end) indices for each episode along time axis."""
    t_axis = terminals.shape[0]
    end_flags = (terminals.max(axis=-1) > 0.5) | (truncations.max(axis=-1) > 0.5)
    ends = np.flatnonzero(end_flags)
    if ends.size == 0:
        raise ValueError("Vault trajectory contains no terminal/truncation markers.")
    if ends[-1] != t_axis - 1:
        raise ValueError(
            f"Last timestep is not an episode boundary (last_end={ends[-1]}, T={t_axis})."
        )
    starts = np.concatenate([[0], ends[:-1] + 1])
    if starts.shape[0] != ends.shape[0]:
        raise ValueError("Episode boundary parsing failed (starts/ends mismatch).")
    slices = [(int(s), int(e)) for s, e in zip(starts, ends)]
    return slices


def load_episode_dataset_from_vault(
    source: str,
    env_name: str,
    scenario: str,
    dataset_name: str,
    rel_dir: str | None = None,
    device: str = "cuda",
    episode_limit: Optional[int] = None,
) -> EpisodeDataset:
    import jax

    if rel_dir is None:
        rel_dir = str(VAULT_DIR)
    jax.config.update("jax_platform_name", "cpu")
    limit = episode_limit or SMAC_EPISODE_LIMIT.get(scenario)
    if limit is None:
        raise ValueError(f"Unknown episode_limit for scenario {scenario}")

    max_seq = limit + 1
    buffer_state = Vault(
        vault_name=f"{source}/{env_name}/{scenario}.vlt",
        vault_uid=dataset_name,
        rel_dir=rel_dir,
    ).read()
    raw = _to_numpy_tree(buffer_state.experience)
    del buffer_state

    obs = raw["observations"][0]
    actions = raw["actions"][0]
    rewards = raw["rewards"][0]
    terminals = raw["terminals"][0]
    truncations = raw["truncations"][0]
    legals = raw["infos"]["legals"][0]
    states = raw["infos"]["state"][0]

    n_agents = int(obs.shape[1])
    obs_dim = int(obs.shape[2])
    state_dim = int(states.shape[1])
    discrete_mujoco = env_name == "mamujoco" and scenario in MUJOCO_DISCRETE_ACTION_SCENARIOS
    if discrete_mujoco:
        from envs.discrete_mujoco_actions import n_discrete_actions

        n_actions = n_discrete_actions(scenario)
    else:
        n_actions = int(legals.shape[2])

    slices = _split_episode_slices(terminals, truncations)
    lengths = [e - s + 1 for s, e in slices]
    if max(lengths) > limit:
        raise ValueError(
            f"Episode length {max(lengths)} exceeds episode_limit {limit} for {scenario}"
        )

    n_ep = len(slices)
    obs_out = np.zeros((n_ep, max_seq, n_agents, obs_dim), dtype=np.float32)
    act_out = np.zeros((n_ep, max_seq, n_agents, 1), dtype=np.int64)
    state_out = np.zeros((n_ep, max_seq, state_dim), dtype=np.float32)
    avail_out = np.zeros((n_ep, max_seq, n_agents, n_actions), dtype=np.float32)
    rew_out = np.zeros((n_ep, max_seq, 1), dtype=np.float32)
    term_out = np.zeros((n_ep, max_seq, 1), dtype=np.float32)
    fill_out = np.zeros((n_ep, max_seq, 1), dtype=np.float32)

    for i, (start, end) in enumerate(slices):
        seg_len = end - start + 1
        obs_out[i, :seg_len] = obs[start : end + 1]
        seg_actions = actions[start : end + 1]
        if discrete_mujoco:
            from envs.discrete_mujoco_actions import continuous_to_index

            for t in range(seg_len):
                for ag in range(n_agents):
                    act_out[i, t, ag, 0] = continuous_to_index(seg_actions[t, ag], scenario)
            avail_out[i, :seg_len] = 1.0
        else:
            act_out[i, :seg_len, :, 0] = seg_actions.astype(np.int64)
            avail_out[i, :seg_len] = legals[start : end + 1]
        state_out[i, :seg_len] = states[start : end + 1]
        team_r = rewards[start : end + 1, 0:1].mean(axis=1, keepdims=True)
        rew_out[i, :seg_len] = team_r
        term_out[i, :seg_len] = terminals[start : end + 1].max(axis=1, keepdims=True)
        fill_out[i, :seg_len] = 1.0

        boot = seg_len
        if boot < max_seq:
            obs_out[i, boot] = obs_out[i, boot - 1]
            state_out[i, boot] = state_out[i, boot - 1]
            avail_out[i, boot] = avail_out[i, boot - 1]
            term_out[i, boot] = 1.0
            fill_out[i, boot - 1] = 1.0
            fill_out[i, boot] = 0.0

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    dataset = EpisodeDataset(
        obs=torch.from_numpy(obs_out).to(dev),
        actions=torch.from_numpy(act_out).to(dev),
        state=torch.from_numpy(state_out).to(dev),
        avail_actions=torch.from_numpy(avail_out).bool().to(dev),
        reward=torch.from_numpy(rew_out).to(dev),
        terminated=torch.from_numpy(term_out).to(dev),
        filled=torch.from_numpy(fill_out).to(dev),
        stats=EpisodeDatasetStats(
            n_episodes=n_ep,
            max_seq_length=max_seq,
            total_filled_steps=int(fill_out[:, :-1].sum()),
            median_episode_length=float(np.median(lengths)),
            n_agents=n_agents,
            obs_dim=obs_dim,
            state_dim=state_dim,
            n_actions=n_actions,
        ),
    )
    return dataset


def print_dataset_stats(dataset: EpisodeDataset, label: str = "") -> None:
    s = dataset.stats
    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}episodes={s.n_episodes} max_seq={s.max_seq_length} "
        f"total_filled={s.total_filled_steps} median_len={s.median_episode_length:.1f} "
        f"agents={s.n_agents} obs={s.obs_dim} state={s.state_dim} actions={s.n_actions}"
    )
