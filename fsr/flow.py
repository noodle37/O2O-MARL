"""Conditional flow matching: velocity regression, score, and ODE log-density.

Layout follows MAC-Flow / Lipman CFM: x_t = (1-t) x_0 + t x_1, target = x_1 - x_0.
Score is read from the velocity field on the interpolant. Online log p uses the
instantaneous change-of-variables formula along the probability-flow ODE.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch


VelocityFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def flow_matching_loss(
    velocity_fn: VelocityFn,
    cond: torch.Tensor,
    x1: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE between v(cond, x_t, t) and (x1 - x0). Only real (cond, x1) pairs."""
    n = x1.shape[0]
    x0 = torch.randn_like(x1)
    if t is None:
        t = torch.rand(n, device=x1.device, dtype=x1.dtype)
    xt = (1.0 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * x1
    pred = velocity_fn(cond, xt, t)
    per_row = ((pred - (x1 - x0)) ** 2).mean(dim=-1)
    if mask is None:
        return per_row.mean()
    mask_f = mask.float()
    return (per_row * mask_f).sum() / mask_f.sum().clamp(min=1.0)


def score_from_velocity(
    velocity_fn: VelocityFn,
    cond: torch.Tensor,
    x1: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    t_eval: float = 0.9,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Interpolant score ∇ log p_t(x_t) from a trained OT velocity field.

    x_0_hat = x_t - t v_θ(x_t, t); score ≈ -x_0_hat / (1-t).
    Evaluated on the CFM interpolant so (1-t) stays away from 0.
    """
    n = x1.shape[0]
    if t is None:
        t = torch.full((n,), float(t_eval), device=x1.device, dtype=x1.dtype)
    x0 = torch.randn_like(x1)
    xt = (1.0 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * x1
    vel = velocity_fn(cond, xt, t)
    x0_hat = xt - t.unsqueeze(-1) * vel
    denom = (1.0 - t).clamp(min=eps).unsqueeze(-1)
    return -x0_hat / denom


def hutchinson_divergence(
    velocity_fn: VelocityFn,
    cond: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    n_proj: int = 1,
) -> torch.Tensor:
    """Hutchinson estimator of ∇·v. Rademacher probes; one probe is the default."""
    div = x.new_zeros(x.shape[0])
    for _ in range(max(1, int(n_proj))):
        eps = torch.empty_like(x).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            vel = velocity_fn(cond, x_req, t)
            jvp = torch.autograd.grad(vel, x_req, grad_outputs=eps, create_graph=False)[0]
        div = div + (jvp * eps).sum(dim=-1)
    return div / float(max(1, int(n_proj)))


def exact_divergence(
    velocity_fn: VelocityFn,
    cond: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Exact Jacobian trace. Use only when teammate_dim is small."""
    dim = x.shape[-1]
    div = x.new_zeros(x.shape[0])
    with torch.enable_grad():
        x_req = x.detach().requires_grad_(True)
        vel = velocity_fn(cond, x_req, t)
        for d in range(dim):
            ones = torch.zeros_like(vel)
            ones[:, d] = 1.0
            grad_d = torch.autograd.grad(
                vel, x_req, grad_outputs=ones, create_graph=False, retain_graph=True
            )[0]
            div = div + grad_d[:, d]
    return div


def log_prob_ode(
    velocity_fn: VelocityFn,
    cond: torch.Tensor,
    x1: torch.Tensor,
    steps: int = 8,
    n_proj: int = 1,
    exact_div: bool = False,
) -> torch.Tensor:
    """log p_1(x_1) via instantaneous change of variables on the probability-flow ODE.

    Integrate backward from t=1 to t=0:
        log p_1(x_1) = log p_0(x_0) - ∫_0^1 ∇·v_θ(x_t, t) dt
    with p_0 = N(0, I).
    """
    steps = max(1, int(steps))
    dt = 1.0 / float(steps)
    x = x1
    integral_div = x.new_zeros(x.shape[0])
    for i in range(steps):
        t = torch.full((x.shape[0],), 1.0 - i * dt, device=x.device, dtype=x.dtype)
        vel = velocity_fn(cond, x, t)
        if exact_div:
            div = exact_divergence(velocity_fn, cond, x, t)
        else:
            div = hutchinson_divergence(velocity_fn, cond, x, t, n_proj=n_proj)
        integral_div = integral_div + div * dt
        x = x - vel * dt
    d = x.shape[-1]
    log_p0 = -0.5 * (x * x).sum(dim=-1) - 0.5 * d * math.log(2.0 * math.pi)
    return log_p0 - integral_div
