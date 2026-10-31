# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Smooth adaptive KL controller for the frozen offline policy prior.

The probe layer produces an estimate ``mu_hat`` of the prior's average team
value and its standard error ``s_hat``.  This module turns that pair into the
KL weight used by the next online interval:

    m_k = (1 - alpha) m_{k-1} + alpha * mu_hat_k
    v_k = (1 - alpha)^2 v_{k-1} + alpha^2 * s_hat_k^2
    z_k = m_k / sqrt(v_k + eps)
    lambda*_k = lambda_base + delta_lambda * tanh(beta * z_k)
    lambda_{k+1} = (1 - rho) lambda_k + rho * lambda*_k

When a probe cannot produce a finite estimate the controller keeps
``m, v, lambda`` unchanged, as required by the algorithm.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional


def _finite(value: Any, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"sapr.{name} must be finite, got {value!r}")
    return out


@dataclass(frozen=True)
class SAPRSettings:
    """Validated ``sapr`` config block shared by the controller and probe."""

    enabled: bool = False
    temperature: float = 1.0
    support_floor: float = 1e-8
    lambda_base: float = 0.1
    lambda_min: float = 0.0
    lambda_max: float = 0.5
    delta_lambda: float = 0.1
    alpha: float = 0.5
    beta: float = 0.5
    rho: float = 0.5
    eps: float = 1e-8
    interval_steps: int = 100_000
    count_probe_in_budget: bool = True
    probe_episodes: int = 32
    probe_seed_base: int = 900_001
    probe_behaviour: str = "policy"
    probe_max_attempt_factor: float = 3.0
    se_mode: str = "analytic"
    bootstrap_samples: int = 200

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any]) -> "SAPRSettings":
        block = dict(cfg.get("sapr") or {})
        enabled = bool(block.get("enabled", False))
        if not enabled:
            return cls(enabled=False)
        settings = cls(
            enabled=True,
            temperature=_finite(block.get("temperature", 1.0), "temperature"),
            support_floor=_finite(block.get("support_floor", 1e-8), "support_floor"),
            lambda_base=_finite(block.get("lambda_base", 0.1), "lambda_base"),
            lambda_min=_finite(block.get("lambda_min", 0.0), "lambda_min"),
            lambda_max=_finite(block.get("lambda_max", 0.5), "lambda_max"),
            delta_lambda=_finite(block.get("delta_lambda", 0.1), "delta_lambda"),
            alpha=_finite(block.get("alpha", 0.5), "alpha"),
            beta=_finite(block.get("beta", 0.5), "beta"),
            rho=_finite(block.get("rho", 0.5), "rho"),
            eps=_finite(block.get("eps", 1e-8), "eps"),
            interval_steps=int(block.get("interval_steps", 100_000)),
            count_probe_in_budget=bool(block.get("count_probe_in_budget", True)),
            probe_episodes=int(block.get("probe_episodes", 32)),
            probe_seed_base=int(block.get("probe_seed_base", 900_001)),
            probe_behaviour=str(block.get("probe_behaviour", "policy")).lower(),
            probe_max_attempt_factor=_finite(
                block.get("probe_max_attempt_factor", 3.0), "probe_max_attempt_factor"
            ),
            se_mode=str(block.get("se_mode", "analytic")).lower(),
            bootstrap_samples=int(block.get("bootstrap_samples", 200)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.temperature <= 0.0:
            raise ValueError("sapr.temperature must be positive")
        if not 0.0 <= self.support_floor < 1.0:
            raise ValueError("sapr.support_floor must lie in [0, 1)")
        if not (0.0 <= self.lambda_min <= self.lambda_base <= self.lambda_max):
            raise ValueError(
                "sapr requires lambda_min <= lambda_base <= lambda_max, got "
                f"[{self.lambda_min}, {self.lambda_base}, {self.lambda_max}]"
            )
        if self.delta_lambda < 0.0:
            raise ValueError("sapr.delta_lambda must be non-negative")
        max_delta = min(
            self.lambda_base - self.lambda_min, self.lambda_max - self.lambda_base
        )
        if self.delta_lambda > max_delta + 1e-12:
            raise ValueError(
                f"sapr.delta_lambda={self.delta_lambda} exceeds the range allowed by "
                f"lambda_min/lambda_max (max {max_delta})"
            )
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("sapr.alpha must lie in (0, 1]")
        if self.beta <= 0.0:
            raise ValueError("sapr.beta must be positive")
        if not 0.0 < self.rho <= 1.0:
            raise ValueError("sapr.rho must lie in (0, 1]")
        if self.eps <= 0.0:
            raise ValueError("sapr.eps must be positive")
        if self.interval_steps <= 0:
            raise ValueError("sapr.interval_steps must be positive")
        # Two probes can never put two samples in both Z groups, so the
        # two-sample SE would always be undefined. Four is the minimum that can
        # produce a usable estimate.
        if self.probe_episodes < 4:
            raise ValueError(
                "sapr.probe_episodes must be at least 4 so both treatment groups "
                "can reach two samples"
            )
        if self.probe_behaviour not in ("policy", "greedy"):
            raise ValueError(
                f"unknown sapr.probe_behaviour {self.probe_behaviour!r}; "
                "expected 'policy' or 'greedy'"
            )
        if self.probe_max_attempt_factor < 1.0:
            raise ValueError("sapr.probe_max_attempt_factor must be at least 1")
        if self.se_mode not in ("analytic", "cluster_bootstrap"):
            raise ValueError(
                f"unknown sapr.se_mode {self.se_mode!r}; "
                "expected 'analytic' or 'cluster_bootstrap'"
            )
        if self.se_mode == "cluster_bootstrap" and self.bootstrap_samples < 2:
            raise ValueError("sapr.bootstrap_samples must be at least 2 for bootstrap SE")


class SmoothAdaptiveController:
    """Bounded, smooth outer controller for the prior KL weight."""

    def __init__(self, settings: SAPRSettings):
        if not settings.enabled:
            raise ValueError("SmoothAdaptiveController requires sapr.enabled=true")
        settings.validate()
        self.settings = settings
        self.lambda_value = float(settings.lambda_base)
        self.m = 0.0
        self.v = 0.0
        self.version = 0
        self.last_probe_t = 0
        self.history: list[Dict[str, float]] = []

    def due(self, t_env: int) -> bool:
        return int(t_env) - int(self.last_probe_t) >= self.settings.interval_steps

    def _base_logs(self) -> Dict[str, float]:
        return {
            "sapr/version": float(self.version),
            "sapr/lambda_before": float(self.lambda_value),
            "sapr/m": float(self.m),
            "sapr/v": float(self.v),
        }

    def update(self, mu_hat: float, s_hat: float) -> Dict[str, float]:
        """Consume one probe estimate and advance the controller by one boundary."""
        mu = float(mu_hat)
        s = float(s_hat)
        if not math.isfinite(mu) or not math.isfinite(s) or s < 0.0:
            return self.no_update(reason="invalid_estimate")

        settings = self.settings
        m = (1.0 - settings.alpha) * self.m + settings.alpha * mu
        v = (1.0 - settings.alpha) ** 2 * self.v + settings.alpha**2 * s * s
        z = m / math.sqrt(v + settings.eps)
        lambda_star = settings.lambda_base + settings.delta_lambda * math.tanh(
            settings.beta * z
        )
        lambda_new = (1.0 - settings.rho) * self.lambda_value + settings.rho * lambda_star
        # The convex combination stays inside the bounds when lambda_0 and
        # lambda_star do; clip only to absorb floating-point drift.
        lambda_new = min(max(lambda_new, settings.lambda_min), settings.lambda_max)

        self.version += 1
        logs = self._base_logs()
        self.m, self.v, self.lambda_value = m, v, lambda_new
        logs.update(
            {
                "sapr/updated": 1.0,
                "sapr/mu_hat": mu,
                "sapr/s_hat": s,
                "sapr/m": float(m),
                "sapr/v": float(v),
                "sapr/z": float(z),
                "sapr/lambda_star": float(lambda_star),
                "sapr/lambda": float(lambda_new),
            }
        )
        self.history.append(dict(logs))
        return logs

    def no_update(self, reason: str = "insufficient_samples") -> Dict[str, float]:
        """Record a boundary that could not produce a usable estimate."""
        self.version += 1
        logs = self._base_logs()
        logs.update(
            {
                "sapr/updated": 0.0,
                "sapr/lambda_star": float(self.lambda_value),
                "sapr/lambda": float(self.lambda_value),
                "sapr/skip_reason": reason,
            }
        )
        self.history.append(dict(logs))
        return logs

    def state_dict(self) -> Dict[str, float]:
        return {
            "lambda": float(self.lambda_value),
            "m": float(self.m),
            "v": float(self.v),
            "version": float(self.version),
            "last_probe_t": float(self.last_probe_t),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        lam = float(state.get("lambda", self.lambda_value))
        if not math.isfinite(lam):
            raise ValueError(f"sapr checkpoint lambda must be finite, got {lam}")
        self.lambda_value = min(
            max(lam, self.settings.lambda_min), self.settings.lambda_max
        )
        self.m = float(state.get("m", self.m))
        self.v = float(state.get("v", self.v))
        self.version = int(state.get("version", self.version))
        self.last_probe_t = int(state.get("last_probe_t", self.last_probe_t))

    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.settings)


def build_sapr_controller(cfg: Mapping[str, Any]) -> Optional[SmoothAdaptiveController]:
    settings = SAPRSettings.from_cfg(cfg)
    if not settings.enabled:
        return None
    return SmoothAdaptiveController(settings)
