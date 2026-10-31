# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Meta-gradient controller for the KL weight of the offline policy prior.

The controller owns a scalar ``z`` and writes the prior weight as

    lambda_k = lambda_ref * exp(z_k).

At every meta boundary the engine differentiates the validation objective
through K inner Adam steps and returns ``meta_grad`` (the derivative of
``M_k(z) = -J_val(theta_{k+K}(z)) + (kappa / 2) z**2``). This module applies one
Adam update to ``z`` and keeps its own optimizer state, so the controller is
fully serialisable with the training checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional


def _finite(value: Any, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"muapr.{name} must be finite, got {value!r}")
    return out


@dataclass(frozen=True)
class MUAPRSettings:
    """Validated ``muapr`` config block."""
 
    enabled: bool = False
    # lambda = lambda_ref * exp(z), with lambda_ref equal to SAPR's base weight.
    lambda_ref: float = 0.1
    # Policy temperature for both the KL and the validation ratio.
    temperature: float = 1.0
    support_floor: float = 1e-8
    # Meta-boundary cadence in environment steps.
    interval_steps: int = 20_000
    start_after_steps: int = 50_000
    # Inner adaptation: K differentiable Adam steps on mixed-history batches.
    inner_steps: int = 4
    inner_batch_size: int = 32
    kl_offline_frac: float = 0.5
    # Validation split is drawn from the most recent online episodes.
    val_recent_episodes: int = 64
    val_episodes: int = 16
    # PPO-style joint ratio clipping on the validation objective.
    ratio_clip: float = 0.2
    advantage_normalize: bool = True
    gamma: float = 0.99
    # Meta objective M(z) = -J_val + (kappa / 2) z**2.
    kappa: float = 0.01
    meta_lr: float = 0.02
    meta_beta1: float = 0.9
    meta_beta2: float = 0.999
    meta_eps: float = 1e-8
    z_clip: float = 2.0
    inner_grad_clip: float = 10.0
    # Validation uses buffered episodes, so no extra environment interaction.
    count_in_budget: bool = False

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any]) -> "MUAPRSettings":
        block = dict(cfg.get("muapr") or {})
        enabled = bool(block.get("enabled", False))
        if not enabled:
            return cls(enabled=False)
        settings = cls(
            enabled=True,
            lambda_ref=_finite(block.get("lambda_ref", 0.1), "lambda_ref"),
            temperature=_finite(block.get("temperature", 1.0), "temperature"),
            support_floor=_finite(block.get("support_floor", 1e-8), "support_floor"),
            interval_steps=int(block.get("interval_steps", 20_000)),
            start_after_steps=int(block.get("start_after_steps", 50_000)),
            inner_steps=int(block.get("inner_steps", 4)),
            inner_batch_size=int(block.get("inner_batch_size", 32)),
            kl_offline_frac=_finite(block.get("kl_offline_frac", 0.5), "kl_offline_frac"),
            val_recent_episodes=int(block.get("val_recent_episodes", 64)),
            val_episodes=int(block.get("val_episodes", 16)),
            ratio_clip=_finite(block.get("ratio_clip", 0.2), "ratio_clip"),
            advantage_normalize=bool(block.get("advantage_normalize", True)),
            gamma=_finite(block.get("gamma", 0.99), "gamma"),
            kappa=_finite(block.get("kappa", 0.01), "kappa"),
            meta_lr=_finite(block.get("meta_lr", 0.02), "meta_lr"),
            meta_beta1=_finite(block.get("meta_beta1", 0.9), "meta_beta1"),
            meta_beta2=_finite(block.get("meta_beta2", 0.999), "meta_beta2"),
            meta_eps=_finite(block.get("meta_eps", 1e-8), "meta_eps"),
            z_clip=_finite(block.get("z_clip", 2.0), "z_clip"),
            inner_grad_clip=_finite(block.get("inner_grad_clip", 10.0), "inner_grad_clip"),
            count_in_budget=bool(block.get("count_in_budget", False)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.lambda_ref <= 0.0:
            raise ValueError("muapr.lambda_ref must be positive")
        if self.temperature <= 0.0:
            raise ValueError("muapr.temperature must be positive")
        if not 0.0 <= self.support_floor < 1.0:
            raise ValueError("muapr.support_floor must lie in [0, 1)")
        if self.interval_steps <= 0:
            raise ValueError("muapr.interval_steps must be positive")
        if self.start_after_steps < 0:
            raise ValueError("muapr.start_after_steps must be non-negative")
        if self.inner_steps < 1:
            raise ValueError("muapr.inner_steps must be at least 1")
        if self.inner_batch_size < 2:
            raise ValueError("muapr.inner_batch_size must be at least 2")
        if not 0.0 < self.kl_offline_frac < 1.0:
            raise ValueError("muapr.kl_offline_frac must lie in (0, 1)")
        if self.val_episodes < 2:
            raise ValueError("muapr.val_episodes must be at least 2")
        if self.val_recent_episodes <= self.val_episodes:
            raise ValueError(
                "muapr.val_recent_episodes must exceed val_episodes so the "
                "validation split leaves an inner training pool"
            )
        if not 0.0 < self.ratio_clip < 1.0:
            raise ValueError("muapr.ratio_clip must lie in (0, 1)")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("muapr.gamma must lie in (0, 1]")
        if self.kappa < 0.0:
            raise ValueError("muapr.kappa must be non-negative")
        if self.meta_lr <= 0.0:
            raise ValueError("muapr.meta_lr must be positive")
        if not 0.0 <= self.meta_beta1 < 1.0:
            raise ValueError("muapr.meta_beta1 must lie in [0, 1)")
        if not 0.0 <= self.meta_beta2 < 1.0:
            raise ValueError("muapr.meta_beta2 must lie in [0, 1)")
        if self.meta_eps <= 0.0:
            raise ValueError("muapr.meta_eps must be positive")
        if self.z_clip <= 0.0:
            raise ValueError("muapr.z_clip must be positive")
        if self.inner_grad_clip < 0.0:
            raise ValueError("muapr.inner_grad_clip must be non-negative")


def lambda_from_z(settings: MUAPRSettings, z: float) -> float:
    """lambda = lambda_ref * exp(clip(z)); matches the differentiable engine."""
    z_clipped = min(max(float(z), -settings.z_clip), settings.z_clip)
    return float(settings.lambda_ref * math.exp(z_clipped))


class MetaLambdaController:
    """Owns z and its Adam state; one update per meta boundary."""

    def __init__(self, settings: MUAPRSettings):
        if not settings.enabled:
            raise ValueError("MetaLambdaController requires muapr.enabled=true")
        settings.validate()
        self.settings = settings
        self.z = 0.0
        self.m = 0.0
        self.v = 0.0
        self.adam_step = 0
        self.version = 0
        self.last_meta_t = 0
        self.history: list[Dict[str, Any]] = []

    @property
    def lambda_value(self) -> float:
        return lambda_from_z(self.settings, self.z)

    def proposed_lambda(self, z: Optional[float] = None) -> float:
        return lambda_from_z(self.settings, self.z if z is None else z)

    def due(self, t_env: int) -> bool:
        t_env = int(t_env)
        if t_env < self.settings.start_after_steps:
            return False
        return t_env - int(self.last_meta_t) >= self.settings.interval_steps

    def _base_logs(self) -> Dict[str, Any]:
        return {
            "muapr/version": float(self.version),
            "muapr/z": float(self.z),
            "muapr/lambda": float(self.lambda_value),
            "muapr/lambda_ref": float(self.settings.lambda_ref),
            "muapr/meta_step": float(self.adam_step),
            "muapr/meta_lr": float(self.settings.meta_lr),
        }

    def update(
        self,
        meta_grad: float,
        *,
        t_env: int,
        meta_loss: float = 0.0,
        kappa_penalty: float = 0.0,
        J_val: float = 0.0,
    ) -> Dict[str, Any]:
        grad = float(meta_grad)
        if not math.isfinite(grad):
            return self.no_update("invalid_meta_grad", t_env=t_env)

        s = self.settings
        z_before = float(self.z)
        lambda_before = float(self.lambda_value)

        self.m = s.meta_beta1 * self.m + (1.0 - s.meta_beta1) * grad
        self.v = s.meta_beta2 * self.v + (1.0 - s.meta_beta2) * grad * grad
        self.adam_step += 1
        m_hat = self.m / (1.0 - s.meta_beta1**self.adam_step)
        v_hat = self.v / (1.0 - s.meta_beta2**self.adam_step)
        z_new = z_before - s.meta_lr * m_hat / (math.sqrt(v_hat) + s.meta_eps)
        z_new = min(max(z_new, -s.z_clip), s.z_clip)

        self.z = float(z_new)
        self.version += 1
        self.last_meta_t = int(t_env)

        logs = self._base_logs()
        logs.update(
            {
                "muapr/updated": 1.0,
                "muapr/z_before": z_before,
                "muapr/z_after": float(self.z),
                "muapr/lambda_before": lambda_before,
                "muapr/lambda_after": float(self.lambda_value),
                "muapr/meta_m": grad,
                "muapr/meta_m_adam": float(m_hat),
                "muapr/meta_v": float(self.v),
                "muapr/kappa_penalty": float(kappa_penalty),
                "muapr/meta_loss": float(meta_loss),
                "muapr/J_val": float(J_val),
                "muapr/meta_grad_norm": abs(grad),
            }
        )
        self.history.append(dict(logs))
        return logs

    def no_update(self, reason: str, *, t_env: int) -> Dict[str, Any]:
        self.version += 1
        self.last_meta_t = int(t_env)
        logs = self._base_logs()
        logs.update(
            {
                "muapr/updated": 0.0,
                "muapr/skip_reason": str(reason),
                "muapr/z_before": float(self.z),
                "muapr/z_after": float(self.z),
                "muapr/lambda_before": float(self.lambda_value),
                "muapr/lambda_after": float(self.lambda_value),
            }
        )
        self.history.append(dict(logs))
        return logs

    def state_dict(self) -> Dict[str, float]:
        return {
            "z": float(self.z),
            "lambda": float(self.lambda_value),
            "m": float(self.m),
            "v": float(self.v),
            "adam_step": float(self.adam_step),
            "version": float(self.version),
            "last_meta_t": float(self.last_meta_t),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        z = float(state.get("z", self.z))
        if not math.isfinite(z):
            raise ValueError(f"muapr checkpoint z must be finite, got {z}")
        self.z = min(max(z, -self.settings.z_clip), self.settings.z_clip)
        self.m = float(state.get("m", self.m))
        self.v = float(state.get("v", self.v))
        self.adam_step = int(state.get("adam_step", self.adam_step))
        self.version = int(state.get("version", self.version))
        self.last_meta_t = int(state.get("last_meta_t", self.last_meta_t))

    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.settings)


def build_muapr_controller(cfg: Mapping[str, Any]) -> Optional[MetaLambdaController]:
    settings = MUAPRSettings.from_cfg(cfg)
    if not settings.enabled:
        return None
    sapr = cfg.get("sapr") or {}
    if bool(sapr.get("enabled", False)):
        raise ValueError("muapr and sapr cannot both be enabled")
    return MetaLambdaController(settings)
