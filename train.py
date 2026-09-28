#!/usr/bin/env python3
# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Train OVMSE with Vault data and PyMARL-style episode updates."""

from __future__ import annotations

import argparse
import sys

from omegaconf import OmegaConf

from training.offline import run_offline
from training.online import run_online
from utils.paths import ensure_project_root


def _load_cfg(path: str, overrides: list[str]) -> dict:
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to yaml under MARL_code/configs/",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style overrides, e.g. phase=online seed=1",
    )
    args = parser.parse_args()
    ensure_project_root()
    cfg = _load_cfg(args.config, args.overrides)
    phase = str(cfg.get("phase", "offline")).lower()
    if phase == "online":
        online_block = cfg.get("online") or {}
        if online_block.get("t_max") is not None:
            cfg["t_max"] = online_block["t_max"]
        transfer_block = cfg.get("transfer") or {}
        mode = str(transfer_block.get("mode", "")).lower()
        if mode == "ovmse":
            cfg["system"]["use_ovm"] = True
        elif mode:
            cfg["system"]["use_ovm"] = False
        elif online_block.get("use_ovm") is not None:
            cfg["system"]["use_ovm"] = online_block["use_ovm"]
        if online_block.get("test_interval") is not None:
            cfg["test_interval"] = online_block["test_interval"]
        run_online(cfg)
    elif phase == "offline":
        run_offline(cfg)
    else:
        raise ValueError(f"Unknown phase {phase}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
