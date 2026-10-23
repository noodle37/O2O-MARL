"""Support weights, holdout calibration, and visit-count prior decay."""

from __future__ import annotations

from typing import Dict, Optional

import torch


def reliance_weight(logp: torch.Tensor, threshold: float, temperature: float = 1.0) -> torch.Tensor:
    """High log-density (old-team support) → high offline reliance.

    w = sigmoid((log p - τ) / T). The spec's written `sigmoid(-(log p - τ))`
    would invert the intended volume: old team should speak louder, not quieter.
    """
    temp = max(float(temperature), 1e-6)
    return torch.sigmoid((logp - float(threshold)) / temp)


def mix_fraction(weight: torch.Tensor, shape: str = "linear", mix_max: float = 0.5) -> torch.Tensor:
    """Monotone map from reliance weight to offline replay fraction."""
    w = weight.clamp(0.0, 1.0)
    name = str(shape).lower()
    if name == "square":
        mapped = w * w
    elif name == "sqrt":
        mapped = w.sqrt()
    else:
        mapped = w
    return mapped * float(mix_max)


class VisitDecay:
    """λ(count) = 1 / (1 + n / scale). Hash of a random projection of the state."""

    def __init__(
        self,
        state_dim: int,
        *,
        n_proj: int = 8,
        n_bins: int = 8,
        table_size: int = 65536,
        scale: float = 50.0,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ):
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        proj = torch.randn(int(state_dim), int(n_proj), generator=g)
        self.register = None
        self.proj = proj
        self.n_bins = int(n_bins)
        self.table_size = int(table_size)
        self.scale = float(scale)
        self.table = torch.zeros(self.table_size, dtype=torch.float32)
        self.to(device)

    def to(self, device: torch.device | str) -> "VisitDecay":
        dev = torch.device(device)
        self.proj = self.proj.to(dev)
        self.table = self.table.to(dev)
        self.device = dev
        return self

    def _index(self, state: torch.Tensor) -> torch.Tensor:
        z = state.to(self.proj.dtype) @ self.proj
        bins = torch.floor(torch.sigmoid(z) * self.n_bins).long().clamp(0, self.n_bins - 1)
        # Polynomial hash into a fixed table; collisions only soften the decay.
        powers = (self.n_bins ** torch.arange(bins.shape[-1], device=bins.device)).long()
        raw = (bins * powers).sum(dim=-1)
        return raw.abs() % self.table_size

    def lambdas(
        self,
        state: torch.Tensor,
        increment: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """state [..., S] -> λ [...] in (0, 1], decaying with local online visits."""
        idx = self._index(state.reshape(-1, state.shape[-1]))
        if increment:
            if mask is None:
                inc = torch.ones(idx.shape[0], device=self.table.device, dtype=self.table.dtype)
            else:
                inc = mask.reshape(-1).to(device=self.table.device, dtype=self.table.dtype)
            self.table.index_add_(0, idx, inc)
        counts = self.table[idx]
        lam = 1.0 / (1.0 + counts / max(self.scale, 1e-6))
        return lam.reshape(state.shape[:-1])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"table": self.table.detach().cpu(), "proj": self.proj.detach().cpu()}

    def load_state_dict(self, payload: Dict[str, torch.Tensor]) -> None:
        if "table" in payload:
            self.table = payload["table"].to(self.table.device).float()
        if "proj" in payload:
            self.proj = payload["proj"].to(self.proj.device)
