# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Recent-online buffer splitting and mixed-history batching."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from training.rollout import merge_offline_online, stack_episodes


def mixed_sizes(batch_size: int, offline_frac: float) -> Tuple[int, int]:
    """Split a batch into (offline, online) episode counts, both non-zero."""
    batch_size = int(batch_size)
    if batch_size < 2:
        raise ValueError("mixed-history batch_size must be at least 2")
    off = int(round(batch_size * float(offline_frac)))
    off = min(max(off, 1), batch_size - 1)
    return off, batch_size - off


def split_recent_episodes(
    buffer: Sequence[Dict[str, Any]],
    recent_n: int,
    val_n: int,
    rng: np.random.Generator,
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """Hold out ``val_n`` of the most recent episodes as the validation split.

    Returns ``{"validation": [...], "inner": [...]}``; both lists are disjoint.
    ``None`` means the buffer cannot supply both splits.
    """
    episodes = list(buffer)
    recent_n = int(recent_n)
    val_n = int(val_n)
    if val_n < 1 or recent_n <= val_n:
        raise ValueError("recent_n must exceed val_n and val_n must be positive")
    if len(episodes) < val_n + 1:
        return None
    recent = episodes[-min(recent_n, len(episodes)) :]
    if len(recent) <= val_n:
        return None
    val_idx = sorted(int(i) for i in rng.choice(len(recent), size=val_n, replace=False))
    val_set = set(val_idx)
    validation = [recent[i] for i in val_idx]
    inner = [ep for i, ep in enumerate(recent) if i not in val_set]
    if not inner or not validation:
        return None
    return {"validation": validation, "inner": inner}


def build_mixed_batch(
    offline_data,
    inner_pool: Sequence[Dict[str, Any]],
    rng: np.random.Generator,
    *,
    batch_size: int,
    offline_frac: float,
    max_seq_length: int,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, int, int]:
    """One mixed-history batch: offline reference + recent online episodes.

    The TD row mask marks the online rows (the tail), so off-policy reference
    rows contribute to the KL term only.
    """
    off_bs, on_bs = mixed_sizes(batch_size, offline_frac)
    if len(inner_pool) < on_bs:
        raise ValueError(
            f"inner pool has {len(inner_pool)} episodes, need {on_bs} for a mixed batch"
        )
    offline_batch = offline_data.sample(off_bs, rng, device=device)
    online_idx = rng.choice(len(inner_pool), size=on_bs, replace=False)
    online_batch = stack_episodes(
        [inner_pool[int(i)] for i in online_idx], int(max_seq_length), device
    )
    batch = merge_offline_online(offline_batch, online_batch, offline_frac)
    n_off = int(offline_batch["obs"].shape[0])
    n_on = int(online_batch["obs"].shape[0])
    row_mask = torch.cat(
        [
            torch.zeros(n_off, device=device),
            torch.ones(n_on, device=device),
        ]
    )
    return batch, row_mask, n_off, n_on
