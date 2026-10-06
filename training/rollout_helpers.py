"""Helpers for rollout action / legal-action handling."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def actions_to_step_array(actions: Dict[str, Any], agents: list[str]) -> np.ndarray:
    ordered = [actions[a] for a in agents]
    first = ordered[0]
    if isinstance(first, np.ndarray) and first.ndim > 0:
        return np.stack([np.asarray(a, dtype=np.float32) for a in ordered])
    return np.array([int(a) for a in ordered], dtype=np.int64)


def legals_from_info(info: Dict[str, Any], agents: list[str]) -> Optional[Dict[str, np.ndarray]]:
    legals = info.get("legals")
    if legals is None:
        return None
    return {a: legals[a] for a in agents}
