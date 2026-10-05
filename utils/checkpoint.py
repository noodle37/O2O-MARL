# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Save and load QMIX (MarlcRNN + QMixer) checkpoints."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch


def _checkpoint_file_path(path: str, default_name: str) -> str:
    if path.endswith(".pt") or path.endswith(".pth"):
        return path
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, default_name)


def save_qmix_checkpoint(
    path: str,
    system: Any,
    include_optimizer: bool = False,
    extra: Optional[Dict[str, Any]] = None,
    default_name: str = "qmix.pt",
) -> str:
    path = _checkpoint_file_path(path, default_name)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    payload: Dict[str, Any] = {
        "q_network": system.q_network.state_dict(),
        "mixer": system.mixer.state_dict(),
        "target_q_network": system.target_q_network.state_dict(),
        "target_mixer": system.target_mixer.state_dict(),
        "training_step_ctr": int(getattr(system, "training_step_ctr", 0)),
    }
    if getattr(system, "coord", None) is not None:
        payload["coord"] = system.coord.state_dict()
        payload["target_coord"] = system.target_coord.state_dict()
    if include_optimizer and hasattr(system, "optimizer"):
        payload["optimizer"] = system.optimizer.state_dict()
    if extra:
        payload["extra"] = extra

    torch.save(payload, path)
    print(f"Saved QMIX checkpoint to {path}")
    return path


def load_qmix_checkpoint(
    path: str,
    system: Any,
    load_optimizer: bool = False,
    map_location: Optional[Any] = None,
) -> Dict[str, Any]:
    if not path:
        raise FileNotFoundError(
            "FATAL: train_mode=online requires checkpoint_path; refusing to train from scratch."
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"FATAL: checkpoint not found at '{path}'. "
            "Refusing to silently train from scratch."
        )

    if map_location is None:
        map_location = getattr(system, "device", "cpu")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    required = ("q_network", "mixer", "target_q_network", "target_mixer")
    missing = [key for key in required if key not in ckpt]
    if missing:
        raise KeyError(f"Checkpoint '{path}' is missing keys: {missing}")

    system.q_network.load_state_dict(ckpt["q_network"])
    system.mixer.load_state_dict(ckpt["mixer"])
    system.target_q_network.load_state_dict(ckpt["target_q_network"])
    system.target_mixer.load_state_dict(ckpt["target_mixer"])
    if load_optimizer and "optimizer" in ckpt and hasattr(system, "optimizer"):
        system.optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Loaded QMIX checkpoint from {path}")
    return ckpt
