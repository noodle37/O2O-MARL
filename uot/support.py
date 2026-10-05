"""Frozen set of recorded offline joint actions.

Membership is on the action tuple, not on (state, joint). Online data does
not add keys. Encoding matches MARLC: code = sum_i a_i * n_actions**i.
"""

from __future__ import annotations

import numpy as np
import torch


class RecordedJoints:
    def __init__(self, n_agents: int, n_actions: int, codes: np.ndarray):
        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.mults = self.n_actions ** np.arange(self.n_agents, dtype=np.int64)
        unique = np.unique(np.asarray(codes, dtype=np.int64).reshape(-1))
        self.sorted_codes = torch.from_numpy(np.sort(unique))

    def __len__(self) -> int:
        return int(self.sorted_codes.numel())

    @classmethod
    def from_episode_actions(
        cls,
        actions: torch.Tensor,
        filled: torch.Tensor,
        n_actions: int,
    ) -> "RecordedJoints":
        """Build from padded episodes. actions [E,T,N] or [E,T,N,1], filled [E,T] or [E,T,1]."""
        act = actions.detach()
        if act.dim() == 4 and act.shape[-1] == 1:
            act = act.squeeze(-1)
        mask = filled.detach()
        if mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        flat = act[mask > 0.5].reshape(-1, act.shape[-1]).cpu().numpy().astype(np.int64)
        n_agents = int(act.shape[-1])
        mults = int(n_actions) ** np.arange(n_agents, dtype=np.int64)
        codes = flat @ mults if flat.size else np.zeros((0,), dtype=np.int64)
        return cls(n_agents, n_actions, codes)

    def encode(self, joints: torch.Tensor) -> torch.Tensor:
        """joints [..., n_agents] -> int64 codes [...]."""
        mults = torch.as_tensor(self.mults, device=joints.device, dtype=torch.long)
        actions = joints.long().clamp(0, self.n_actions - 1)
        return (actions * mults).sum(dim=-1)

    def contains(self, joints: torch.Tensor) -> torch.Tensor:
        """joints [..., n_agents] -> bool [...]."""
        codes = self.encode(joints)
        sorted_codes = self.sorted_codes.to(joints.device)
        if sorted_codes.numel() == 0:
            return torch.zeros(codes.shape, dtype=torch.bool, device=joints.device)
        flat = codes.reshape(-1)
        idx = torch.searchsorted(sorted_codes, flat)
        in_range = idx < sorted_codes.numel()
        idx = idx.clamp(max=sorted_codes.numel() - 1)
        hit = in_range & (sorted_codes[idx] == flat)
        return hit.view(codes.shape)
