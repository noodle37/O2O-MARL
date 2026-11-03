"""Discrete Perturbed Value Update for QMIX.

SO2 (Zhang et al., arXiv:2312.07685) builds the TD target on a perturbed
next action, ``a' + ε`` with ``ε ~ clip(N(0, σ), -c, c)``. SMAC actions are
discrete and have no Euclidean metric, so MaSO2 keeps the same σ and reads
it as the probability that an agent replaces its greedy bootstrap action
with another legal action. ``c`` is the paper's clip; on a normalised
continuous box it already covers most of the local neighbourhood, and the
discrete legal set is that neighbourhood. See
``documentation/so2_o2o_marl_migration.md``.
"""

from __future__ import annotations

from typing import Optional

import torch


def perturbed_bootstrap_actions(
    q: torch.Tensor,
    avail: torch.Tensor,
    sigma: float,
    *,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the double-Q bootstrap action under discrete PVU.

    ``q`` is ``[B, T, N, A]``. Illegal actions must already be masked (a large
    negative fill) so the greedy action matches ``argmax``. ``avail`` is the
    legal-action mask, broadcastable to ``q``.

    Returns ``actions`` of shape ``[B, T, N, 1]`` and a boolean ``changed``
    mask of the same shape. ``sigma <= 0`` is exactly greedy.
    """
    if avail.dtype != torch.bool:
        avail = avail > 0.5
    greedy = torch.argmax(q, dim=-1, keepdim=True)
    if sigma <= 0.0:
        changed = torch.zeros_like(greedy, dtype=torch.bool)
        return greedy, changed

    # Uniform over legal actions: illegal entries get score 0, legal get U(0, 1).
    # Draw on the generator's device (CPU) and move, so a CUDA learner does not
    # need a CUDA generator.
    scores = _uniform(q.shape, q, generator)
    scores = torch.where(avail, scores, torch.zeros_like(scores))
    random_action = torch.argmax(scores, dim=-1, keepdim=True)

    flip = _uniform(greedy.shape, q, generator) < float(sigma)
    n_legal = avail.sum(dim=-1, keepdim=True)
    flip = flip & (n_legal > 1)
    actions = torch.where(flip, random_action, greedy)
    return actions, actions != greedy


def _uniform(
    shape: torch.Size, ref: torch.Tensor, generator: Optional[torch.Generator]
) -> torch.Tensor:
    if generator is None:
        return torch.rand(shape, device=ref.device, dtype=ref.dtype)
    drawn = torch.rand(shape, dtype=ref.dtype, generator=generator)
    if drawn.device != ref.device:
        drawn = drawn.to(ref.device)
    return drawn
