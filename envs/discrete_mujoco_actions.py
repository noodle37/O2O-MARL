# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""3-level torque discretization (-1, 0, 1) per joint dim for multi-agent MuJoCo."""

from __future__ import annotations

import numpy as np

TORQUE_LEVELS = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
LEVELS_PER_DIM = 3


def dims_per_agent(scenario: str) -> int:
    if scenario.lower() == "2halfcheetah":
        return 3
    raise ValueError(f"No discrete action spec for mamujoco scenario {scenario}")


def n_discrete_actions(scenario: str) -> int:
    d = dims_per_agent(scenario)
    return int(LEVELS_PER_DIM**d)


def _encode_digits(digits: np.ndarray) -> int:
    idx = 0
    base = LEVELS_PER_DIM
    for k, digit in enumerate(digits.astype(np.int64).reshape(-1)):
        idx += int(digit) * (base**k)
    return int(idx)


def _decode_index(index: int, dims: int) -> np.ndarray:
    idx = int(index)
    digits = np.zeros(dims, dtype=np.int64)
    for k in range(dims):
        digits[k] = idx % LEVELS_PER_DIM
        idx //= LEVELS_PER_DIM
    return digits


def index_to_continuous(index: int, scenario: str) -> np.ndarray:
    dims = dims_per_agent(scenario)
    digits = _decode_index(index, dims)
    return TORQUE_LEVELS[digits].astype(np.float32)


def continuous_to_index(action_vec: np.ndarray, scenario: str) -> int:
    dims = dims_per_agent(scenario)
    vec = np.asarray(action_vec, dtype=np.float32).reshape(-1)[:dims]
    digits = np.argmin(np.abs(vec[:, None] - TORQUE_LEVELS[None, :]), axis=1)
    return _encode_digits(digits)


def all_legal_mask(scenario: str) -> np.ndarray:
    n = n_discrete_actions(scenario)
    return np.ones(n, dtype=np.float32)
