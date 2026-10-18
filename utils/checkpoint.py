# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Save and load QMIX (MarlcRNN + QMixer) checkpoints and fork trainer snapshots."""

from __future__ import annotations

import copy
import os
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
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

    if hasattr(system, "export_state"):
        payload = dict(system.export_state())
        payload["training_step_ctr"] = int(getattr(system, "training_step_ctr", 0))
        payload["learner"] = "mac_flow"
    else:
        payload = {
            "q_network": system.q_network.state_dict(),
            "mixer": system.mixer.state_dict(),
            "target_q_network": system.target_q_network.state_dict(),
            "target_mixer": system.target_mixer.state_dict(),
            "training_step_ctr": int(getattr(system, "training_step_ctr", 0)),
        }
    if getattr(system, "coord", None) is not None:
        payload["coord"] = system.coord.state_dict()
        payload["target_coord"] = system.target_coord.state_dict()
    if include_optimizer and hasattr(system, "optimizer") and "optimizer" not in payload:
        payload["optimizer"] = system.optimizer.state_dict()
    if extra:
        payload["extra"] = extra

    torch.save(payload, path)
    print(f"Saved QMIX checkpoint to {path}")
    return path


def _rng_states() -> Dict[str, Any]:
    """Every random source the online loop draws from, as picklable objects."""
    states: Dict[str, Any] = {
        "np_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    states["torch_cuda"] = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    )
    return states


def _restore_rng_states(states: Dict[str, Any]) -> None:
    if states.get("np_global") is not None:
        np.random.set_state(states["np_global"])
    cpu_state = states.get("torch_cpu")
    if cpu_state is not None:
        # torch.load(map_location=cuda) moves these along with the weights, but
        # set_rng_state only accepts a CPU ByteTensor.
        if isinstance(cpu_state, torch.Tensor):
            cpu_state = cpu_state.to(device="cpu", dtype=torch.uint8)
        torch.set_rng_state(cpu_state)
    cuda_states = states.get("torch_cuda") or []
    if cuda_states and torch.cuda.is_available():
        cleaned = [
            s.to(device="cpu", dtype=torch.uint8)
            for s in cuda_states
            if isinstance(s, torch.Tensor)
        ]
        n_devices = torch.cuda.device_count()
        torch.cuda.set_rng_state_all(cleaned[:n_devices])


def save_trainer_snapshot(
    path: str,
    system: Any,
    *,
    online_buffer: Optional[Any] = None,
    buffer_maxlen: Optional[int] = None,
    t_env: int = 0,
    batch_rng: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
    default_name: str = "fork_snapshot.pt",
) -> str:
    """Freeze the whole online trainer so branches can resume from an identical state.

    Beyond weights and optimizer this carries the frozen oracle (Q-bar), the live
    replay buffer, the counters that drive the target-network sync, and every RNG
    the loop consumes. Omitting any of them makes the branches diverge for reasons
    other than the intervention.

    One thing is deliberately not shared: SMAC v1 derives its eval episode sequence
    from how many times reset() has been called, and replaying that many resets
    costs hours per branch at 3M steps. Each branch therefore starts from a fresh
    env, so the three branches are evaluated on identical episodes (which is what
    the comparison needs) while the parent's fork-point row sits on a slightly
    different episode set.
    """
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
        "t_env": int(t_env),
        "episode_counter": int(getattr(system, "episode_counter", 0)),
        "last_target_update_episode": int(getattr(system, "last_target_update_episode", 0)),
        "rng": _rng_states(),
    }
    if hasattr(system, "optimizer") and "optimizer" not in payload:
        payload["optimizer"] = system.optimizer.state_dict()
    for attr in ("frozen_q_network", "frozen_mixer", "offline_q_network", "offline_mixer"):
        net = getattr(system, attr, None)
        if net is not None:
            payload[attr] = net.state_dict()
    if online_buffer is not None:
        payload["online_buffer"] = list(online_buffer)
        if buffer_maxlen is None:
            buffer_maxlen = getattr(online_buffer, "maxlen", None)
        payload["buffer_maxlen"] = buffer_maxlen
    if batch_rng is not None:
        payload["batch_rng"] = batch_rng.bit_generator.state
    if extra:
        payload["extra"] = extra

    torch.save(payload, path)
    print(f"Saved trainer snapshot to {path}")
    return path


def load_trainer_snapshot(
    path: str,
    system: Any,
    *,
    load_optimizer: bool = True,
    map_location: Optional[Any] = None,
) -> Dict[str, Any]:
    """Restore a snapshot into `system` and return the non-weight trainer state."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"FATAL: trainer snapshot not found at '{path}'.")

    if map_location is None:
        map_location = getattr(system, "device", "cpu")
    payload = torch.load(path, map_location=map_location, weights_only=False)

    system.q_network.load_state_dict(payload["q_network"])
    system.mixer.load_state_dict(payload["mixer"])
    system.target_q_network.load_state_dict(payload["target_q_network"])
    system.target_mixer.load_state_dict(payload["target_mixer"])
    if load_optimizer and "optimizer" in payload and hasattr(system, "optimizer"):
        system.optimizer.load_state_dict(payload["optimizer"])
    for attr in ("frozen_q_network", "frozen_mixer", "offline_q_network", "offline_mixer"):
        if attr not in payload:
            continue
        net = getattr(system, attr, None)
        if net is None:
            # The frozen oracle only exists after freeze_oracle(); a resumed branch
            # skips that call, so rebuild the container from the live net. Without
            # this the ranking term silently no-ops (Q-bar is None) and every branch
            # trains as if the ranking loss were already off.
            live = getattr(system, attr.replace("frozen_", "").replace("offline_", ""), None)
            if live is None:
                continue
            net = copy.deepcopy(live)
            for param in net.parameters():
                param.requires_grad_(False)
            setattr(system, attr, net)
        net.load_state_dict(payload[attr])
    if "training_step_ctr" in payload:
        system.training_step_ctr = int(payload["training_step_ctr"])
    if "episode_counter" in payload:
        system.episode_counter = int(payload["episode_counter"])
    if "last_target_update_episode" in payload:
        system.last_target_update_episode = int(payload["last_target_update_episode"])

    out: Dict[str, Any] = {
        "t_env": int(payload.get("t_env", 0)),
        "episode_counter": int(payload.get("episode_counter", 0)),
        "extra": payload.get("extra", {}),
    }
    if "online_buffer" in payload:
        maxlen = payload.get("buffer_maxlen")
        out["online_buffer"] = deque(payload["online_buffer"], maxlen=maxlen)
    if payload.get("rng"):
        _restore_rng_states(payload["rng"])
    if payload.get("batch_rng") is not None:
        rng = np.random.default_rng()
        rng.bit_generator.state = payload["batch_rng"]
        out["batch_rng"] = rng
    print(
        f"Loaded trainer snapshot from {path} (t_env={out['t_env']}, "
        f"buffer={len(out.get('online_buffer', []))})"
    )
    return out


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

    if ckpt.get("learner") == "mac_flow" or "flow" in ckpt:
        if not hasattr(system, "load_state"):
            raise TypeError("Checkpoint is MAC-Flow but system has no load_state()")
        system.load_state(ckpt)
        if "training_step_ctr" in ckpt:
            system.training_step_ctr = int(ckpt["training_step_ctr"])
    else:
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
        if "training_step_ctr" in ckpt:
            system.training_step_ctr = int(ckpt["training_step_ctr"])
    print(f"Loaded QMIX checkpoint from {path}")
    return ckpt
