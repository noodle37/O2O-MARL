# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Time-major batch/agent dimension helpers for RNN unrolling."""

import torch


def switch_two_leading_dims(x: torch.Tensor) -> torch.Tensor:
    """Switch first two dimensions of tensor."""
    return x.transpose(0, 1)


def merge_batch_and_agent_dim_of_time_major_sequence(x: torch.Tensor) -> torch.Tensor:
    """Merge batch and agent dimensions for time-major sequence."""
    t_steps, batch_size, n_agents = x.shape[:3]
    trailing_dims = x.shape[3:]
    return x.reshape(t_steps, batch_size * n_agents, *trailing_dims)


def expand_batch_and_agent_dim_of_time_major_sequence(x: torch.Tensor, batch_size: int, n_agents: int) -> torch.Tensor:
    """Expand merged batch and agent dimensions back."""
    t_steps, merged = x.shape[:2]
    assert merged == batch_size * n_agents
    trailing_dims = x.shape[2:]
    return x.reshape(t_steps, batch_size, n_agents, *trailing_dims)
