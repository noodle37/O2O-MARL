#!/usr/bin/env python3
"""Materialise the controlled OVM on/off run configs for the mechanism diagnostic.

For every map two arms are written from the same production config: ``ovmse``
(``transfer.mode=ovmse`` -> ``use_ovm=True``) and ``full_td``
(``transfer.mode=full_td`` -> ``use_ovm=False``). Everything else - offline init,
seed, epsilon schedule, batch composition, target update cadence, diagnostics -
is written from one shared template, so the two resolved YAMLs differ only in the
keys listed in ``ALLOWED_DIFF``. The script refuses to write a pair that differs
anywhere else, which is the guard that keeps the on/off contrast attributable to
OVM alone.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

ALLOWED_DIFF = {
    # The intervention: mode string and the flag it drives.
    "transfer.mode",
    "system.use_ovm",
    "online.use_ovm",
    # Output locations / labels only.
    "system_name",
    "checkpoint_dir",
    "log_path",
    "online.log_path",
    "ovm_diag",
}

# map -> (production config, shared offline best, t_max, test_interval, gpu,
#         buffer snapshot episodes, periodic ckpt cadence)
MAPS = {
    "2halfcheetah": (
        "configs/2halfcheetah.yaml",
        "checkpoints/ovmse_batch_20261014/2halfcheetah_offline/ovmse_paper_seed1_best.pt",
        300000,
        30000,
        2,
        300,
        100000,
    ),
    "2s3z": (
        "configs/2s3z.yaml",
        "checkpoints/ovmse_batch_20261014/2s3z_offline/ovmse_paper_seed1_best.pt",
        200000,
        20000,
        1,
        500,
        100000,
    ),
    "5m_vs_6m": (
        "configs/5m_vs_6m.yaml",
        "checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt",
        300000,
        30000,
        0,
        500,
        100000,
    ),
}

ARMS = (("on", "ovmse"), ("off", "full_td"))


def _arm_overrides(map_name: str, arm: str, mode: str, offline_best: str, t_max: int,
                   test_interval: int, buffer_eps: int, ckpt_every: int) -> dict:
    run_dir = f"checkpoints/ovm_diag_20261017/{map_name}_{arm}"
    offline_dir = os.path.dirname(os.path.dirname(offline_best))
    return {
        "system_name": f"ovm_diag {map_name} {arm}",
        "phase": "online",
        "seed": 1,
        "checkpoint_dir": run_dir,
        "behaviour_cache_dir": os.path.join(offline_dir, "behaviour"),
        "checkpoint_path": offline_best,
        "log_path": os.path.join(run_dir, "online.jsonl"),
        "test_interval": test_interval,
        "transfer": {"mode": mode},
        "online": {
            "t_max": t_max,
            "test_interval": test_interval,
            "use_ovm": mode == "ovmse",
            "log_path": os.path.join(run_dir, "online.jsonl"),
        },
        "ovm_diag": {
            "sample_path": f"logs/ovm_diag/fixed_sample_{map_name}.pt",
            "buffer_snapshot_episodes": buffer_eps,
            "ckpt_every": ckpt_every,
            "train_stats_stride": 4,
        },
        "system": {
            "use_ovm": mode == "ovmse",
            "ovm_diag_train_stats": True,
            "ovm_diag_stride": 4,
        },
    }


def _flatten(cfg, prefix=""):
    out = {}
    for key, value in cfg.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{path}."))
        else:
            out[path] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="configs")
    parser.add_argument("--maps", nargs="*", default=sorted(MAPS))
    args = parser.parse_args()

    manifest = []
    for map_name in args.maps:
        cfg_path, offline_best, t_max, test_interval, gpu, buffer_eps, ckpt_every = MAPS[
            map_name
        ]
        base = OmegaConf.load(cfg_path)
        if not os.path.isfile(offline_best):
            raise FileNotFoundError(f"shared offline init missing: {offline_best}")

        resolved = {}
        for arm, mode in ARMS:
            overrides = _arm_overrides(
                map_name, arm, mode, offline_best, t_max, test_interval, buffer_eps, ckpt_every
            )
            merged = OmegaConf.merge(base, OmegaConf.create(overrides))
            # train.py lifts online.* into the top level; mirror that here so the
            # written YAML is exactly what the run will see.
            container = OmegaConf.to_container(merged, resolve=True)
            online_block = container.get("online") or {}
            if online_block.get("t_max") is not None:
                container["t_max"] = online_block["t_max"]
            if online_block.get("test_interval") is not None:
                container["test_interval"] = online_block["test_interval"]
            if online_block.get("log_path"):
                container["log_path"] = online_block["log_path"]
            resolved[arm] = container

        flat_on = _flatten(resolved["on"])
        flat_off = _flatten(resolved["off"])
        keys = sorted(set(flat_on) | set(flat_off))
        diffs = [
            k for k in keys if flat_on.get(k) != flat_off.get(k) and not k.startswith("ovm_diag")
        ]
        unexpected = [k for k in diffs if k not in ALLOWED_DIFF]
        if unexpected:
            raise SystemExit(
                f"{map_name}: on/off arms differ outside the allowed keys: {unexpected}"
            )
        print(f"[{map_name}] on/off differ only in: {diffs}")

        for arm, _ in ARMS:
            out = os.path.join(args.out_dir, f"ovm_diag_{map_name}_{arm}.yaml")
            os.makedirs(args.out_dir, exist_ok=True)
            with open(out, "w", encoding="utf-8") as handle:
                handle.write(
                    f"# Generated by scripts/make_ovm_diag_configs.py from {cfg_path}.\n"
                    f"# Controlled OVM {'on' if arm == 'on' else 'off'} arm; see\n"
                    f"# documentation/experiment_reports/task4_ovm_mechanism.md for the experiment this belongs to.\n"
                )
                OmegaConf.save(OmegaConf.create(resolved[arm]), handle)
            manifest.append(
                {
                    "map": map_name,
                    "arm": arm,
                    "config": out,
                    "gpu": gpu,
                    "t_max": t_max,
                    "test_interval": test_interval,
                    "offline_best": offline_best,
                    "sample_path": f"logs/ovm_diag/fixed_sample_{map_name}.pt",
                    "run_dir": f"checkpoints/ovm_diag_20261017/{map_name}_{arm}",
                    "log": f"logs/ovm_diag/{map_name}_{arm}.out",
                }
            )
            print(f"  wrote {out}")

    manifest_path = "logs/ovm_diag/manifest.json"
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    import json

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[manifest] {manifest_path} ({len(manifest)} arms)")


if __name__ == "__main__":
    main()
