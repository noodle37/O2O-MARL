"""Frozen support indexes for offline-to-online ordinal transfer.

Three support definitions are used by the controls on 5m_vs_6m:

* joint:      the exact recorded joint-action tuple (the original UOT S_off);
* marginal:   every agent's action has been recorded for that agent, but the
              combination need not have been recorded jointly;
* state-only: no action filter; the ordinal loss is down-weighted on states
              whose offline visitation density is low.

The indexes are built once from the offline dataset and never updated online.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


class RecordedJoints:
    """Frozen set of recorded offline joint actions.

    Membership is on the action tuple, not on (state, joint). Online data does
    not add keys. Encoding matches MARLC: code = sum_i a_i * n_actions**i.
    """

    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        codes: np.ndarray,
        marginal_mask: Optional[np.ndarray] = None,
    ):
        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.mults = self.n_actions ** np.arange(self.n_agents, dtype=np.int64)
        unique = np.unique(np.asarray(codes, dtype=np.int64).reshape(-1))
        self.sorted_codes = torch.from_numpy(np.sort(unique))
        if marginal_mask is None:
            # No per-agent information available: every legal action counts as
            # marginally supported. Callers that need a real marginal ablation
            # must build the mask from episodes.
            marginal_mask = np.ones((self.n_agents, self.n_actions), dtype=bool)
        self.marginal_mask = torch.from_numpy(
            np.asarray(marginal_mask, dtype=bool).reshape(self.n_agents, self.n_actions)
        )

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
        marginal = np.zeros((n_agents, int(n_actions)), dtype=bool)
        if flat.size:
            for i in range(n_agents):
                marginal[i, flat[:, i]] = True
        return cls(n_agents, n_actions, codes, marginal_mask=marginal)

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

    def contains_marginal(self, joints: torch.Tensor) -> torch.Tensor:
        """joints [..., n_agents] -> whether every agent action was recorded."""
        actions = joints.long().clamp(0, self.n_actions - 1)
        mask = self.marginal_mask.to(joints.device)
        ok = torch.ones(actions.shape[:-1], dtype=torch.bool, device=joints.device)
        for i in range(self.n_agents):
            ok = ok & mask[i].gather(0, actions[..., i].reshape(-1)).view(actions.shape[:-1])
        return ok

    def support_rate(self) -> Dict[str, float]:
        total = float(self.n_actions ** self.n_agents)
        return {
            "uot/joint_support_size": float(len(self)),
            "uot/joint_support_frac": float(len(self)) / max(total, 1.0),
            "uot/marginal_actions": float(self.marginal_mask.sum().item()),
            "uot/marginal_frac": float(
                self.marginal_mask.float().sum().item() / max(self.marginal_mask.numel(), 1)
            ),
        }


class StateSupport:
    """Frozen hash-count density over offline states.

    A coarse random projection produces a cell id; the offline visitation count
    of that cell is the density score. The gate is a sigmoid on log1p(count),
    calibrated on the offline counts by a low quantile so that the least-visited
    10% of offline states sit at weight 0.5. States outside the visited cells
    (count 0) receive low weight. No action information enters this index.
    """

    def __init__(
        self,
        state_dim: int,
        *,
        n_proj: int = 8,
        n_bins: int = 8,
        table_size: int = 65536,
        seed: int = 0,
    ):
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        self.proj = torch.randn(int(state_dim), int(n_proj), generator=g)
        self.n_bins = int(n_bins)
        self.table_size = int(table_size)
        self.table = torch.zeros(self.table_size, dtype=torch.float32)
        self.threshold = 0.0
        self.temperature = 1.0
        self.offline_weight_mean = 1.0
        self.offline_weight_frac = 1.0

    def to(self, device: torch.device | str) -> "StateSupport":
        dev = torch.device(device)
        self.proj = self.proj.to(dev)
        self.table = self.table.to(dev)
        self.device = dev
        return self

    def _index(self, states: torch.Tensor) -> torch.Tensor:
        z = states.to(self.proj.dtype) @ self.proj
        bins = torch.floor(torch.sigmoid(z) * self.n_bins).long().clamp(0, self.n_bins - 1)
        powers = (self.n_bins ** torch.arange(bins.shape[-1], device=bins.device)).long()
        raw = (bins * powers).sum(dim=-1)
        return raw.abs() % self.table_size

    @torch.no_grad()
    def fit(self, states: torch.Tensor, filled: torch.Tensor) -> "StateSupport":
        """Count filled offline states. states [E,T,S], filled [E,T] or [E,T,1]."""
        mask = filled.reshape(-1) > 0.5
        idx = self._index(states.reshape(-1, states.shape[-1]))[mask]
        self.table.index_add_(
            0, idx, torch.ones(idx.shape[0], dtype=self.table.dtype, device=self.table.device)
        )
        counts = self.table[idx]
        logc = torch.log1p(counts)
        self._offline_logc = logc.detach().cpu()
        return self

    def calibrate(self, quantile: float = 0.10) -> Dict[str, float]:
        logc = self._offline_logc
        self.threshold = float(torch.quantile(logc, float(quantile)).item())
        self.temperature = float(logc.std().clamp(min=1e-3).item())
        w = torch.sigmoid((logc - self.threshold) / self.temperature)
        self.offline_weight_mean = float(w.mean().item())
        self.offline_weight_frac = float((w > 0.5).float().mean().item())
        return {
            "uot/state_threshold": self.threshold,
            "uot/state_temperature": self.temperature,
            "uot/state_offline_weight_mean": self.offline_weight_mean,
            "uot/state_offline_support_frac": self.offline_weight_frac,
        }

    @torch.no_grad()
    def weights(self, states: torch.Tensor) -> torch.Tensor:
        """states [..., S] -> weights [...] in (0, 1)."""
        idx = self._index(states.reshape(-1, states.shape[-1]))
        counts = self.table[idx]
        w = torch.sigmoid(
            (torch.log1p(counts) - self.threshold) / max(self.temperature, 1e-6)
        )
        return w.reshape(states.shape[:-1])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "proj": self.proj.detach().cpu(),
            "table": self.table.detach().cpu(),
            "threshold": torch.tensor(self.threshold),
            "temperature": torch.tensor(self.temperature),
        }

    def load_state_dict(self, payload: Dict[str, torch.Tensor]) -> None:
        self.proj = payload["proj"].to(self.proj.device)
        self.table = payload["table"].to(self.table.device).float()
        self.threshold = float(payload["threshold"])
        self.temperature = float(payload["temperature"])
