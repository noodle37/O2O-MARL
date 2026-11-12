# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Residual QMIX utility network used by TCPA.

TCPA never re-initialises the online QMIX network.  Instead the offline
CFCQL-D utilities stay in place as a frozen reference and a learnable residual
is added on top of them:

    q_on(tau, a) = q_off(tau, a) + dq_phi(tau, a).

The residual output layer is zero-initialised, so at the start of online
training the online utility is *exactly* the offline utility and the online
team value starts from the offline team value instead of overwriting it.

``ResidualQNetwork`` is a drop-in replacement for ``MarlcRNN`` wherever the
platform unrolls an agent network: it exposes ``initial_state`` and
``forward(x, hidden)`` with the same conventions, so ``qmix.core`` and
``qmix.policy`` work unchanged.  The two GRU hidden states are stacked into one
tensor of shape ``[2, 1, B, H]`` so that a single hidden-state variable can be
reset per episode exactly like the shared MARLC unroll does.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from qmix.networks import MarlcRNN


def zero_init_output(rnn: MarlcRNN) -> None:
    """Zero the action head so the network starts as the identity function."""
    nn.init.zeros_(rnn.fc2.weight)
    nn.init.zeros_(rnn.fc2.bias)


class ResidualQNetwork(nn.Module):
    """q_on = frozen_base + trainable_residual, with a stacked hidden state."""

    def __init__(self, base: MarlcRNN, residual: MarlcRNN):
        super().__init__()
        self.base = base
        self.residual = residual
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.base.eval()

    # ------------------------------------------------------------------ helpers
    def freeze_base(self, freeze: bool = True) -> None:
        for param in self.base.parameters():
            param.requires_grad_(not freeze)
        self.base.train(not freeze)

    def trainable_parameters(self):
        return list(self.residual.parameters())

    # -------------------------------------------------------------- MARLC API
    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        h_base = self.base.initial_state(batch_size, device, dtype=dtype)
        h_res = self.residual.initial_state(batch_size, device, dtype=dtype)
        return torch.stack([h_base, h_res], dim=0)

    def forward(
        self, x: torch.Tensor, hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_state is None:
            hidden_state = self.initial_state(x.shape[0], x.device, dtype=x.dtype)
        h_base, h_res = hidden_state[0], hidden_state[1]
        q_base, new_h_base = self.base(x, h_base)
        q_res, new_h_res = self.residual(x, h_res)
        q = q_base + q_res
        new_hidden = torch.stack([new_h_base, new_h_res], dim=0)
        return q, new_hidden

    # -------------------------------------------------------------- checkpoints
    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        """Accept both the wrapper format and a plain offline ``MarlcRNN`` state.

        The online phase loads the offline CFCQL-D checkpoint into the live
        network.  That checkpoint is a plain ``MarlcRNN`` state dict, so it
        belongs to the frozen base; the residual keeps its zero-initialised
        head, which is exactly the intended online starting point.
        """
        keys = list(state_dict.keys())
        if keys and not any(key.startswith("base.") for key in keys):
            return self.base.load_state_dict(state_dict, strict=strict)
        return super().load_state_dict(state_dict, strict=strict, **kwargs)


def build_residual_q_network(
    obs_in: int,
    hidden_dim: int,
    n_actions: int,
) -> ResidualQNetwork:
    """A residual network on top of an (as yet uninitialised) base network."""
    base = MarlcRNN(obs_in, hidden_dim, n_actions)
    residual = MarlcRNN(obs_in, hidden_dim, n_actions)
    zero_init_output(residual)
    return ResidualQNetwork(base, residual)
