#!/usr/bin/env python3
# Copyright 2024 InstaDeep Ltd. All rights reserved.

"""Scheduler for the problem-validation experiment 1 runs.

Spreads the (map x arm x seed) jobs over the available GPUs, at most
--max-per-gpu concurrent runs per card, each detached from the session the way
nohup would (start_new_session=True) with its own log file.  Finished jobs are
skipped on restart, so an interrupted sweep resumes instead of repeating.

Dependencies handled here:
  * the 2halfcheetah map needs its offline pretraining checkpoint first; jobs for
    that map wait until the file appears (the pretraining run is started
    separately, see configs/pv_offline_2halfcheetah.yaml);
  * arm D needs its affine-rescaled start checkpoint, which is built on demand
    with scripts/make_scale_corrected_ckpt.py and occupies a GPU slot while it
    runs.

Usage:
  python scripts/pv_scheduler.py --plan
  python scripts/pv_scheduler.py --smoke
  nohup python scripts/pv_scheduler.py > logs/pv_scheduler_master.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/home/jxy/miniconda3/envs/multiAgentRL/bin/python"
DRIVER = os.path.join(REPO, "scripts", "run_pv_exp1.sh")

MAP_CONFIG = {
    "5m_vs_6m": "configs/pv_base_5m_vs_6m.yaml",
    "2halfcheetah": "configs/pv_base_2halfcheetah.yaml",
}
MAP_OFFLINE = {
    "5m_vs_6m": "checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt",
    # The offline_final checkpoint, not _best: on this map the offline CFCQL run
    # makes the greedy policy worse at every eval (-229 at t=0, then -617/-623/-626),
    # so "best-so-far" degenerates to the untrained t=0 snapshot.  Starting the arms
    # from it would make arm Z indistinguishable from the rest and leave the
    # "which offline knowledge is worth retaining" question unanswerable.
    "2halfcheetah": "checkpoints/pv_exp1_offline/2halfcheetah_medium/ovmse_paper_seed1.pt",
}
ARMS = ("A", "B", "C", "D", "Z")


def log_path(map_name: str, arm: str, seed: int, smoke: bool) -> str:
    tag = "smoke" if smoke else "exp1"
    return os.path.join(REPO, "logs", f"pv_{tag}_{map_name}_{arm}_seed{seed}.out")


def run_dir(map_name: str, arm: str, seed: int, smoke: bool) -> str:
    root = "checkpoints/pv_smoke" if smoke else "checkpoints/pv_exp1"
    return os.path.join(REPO, root, map_name, f"{arm}_seed{seed}")


def scaled_ckpt(map_name: str, seed: int) -> str:
    return os.path.join(REPO, "checkpoints", "pv_exp1_D_ckpt", map_name, f"D_seed{seed}_scaled.pt")


# Maps whose offline pretraining was launched by this session, mapped to the log
# files that may carry its completion line.  The driver's log name is derived from
# the map name, but the pretraining was started by hand with a shorter tag, so
# both spellings are accepted.
OFFLINE_DONE_MARKERS = {
    "2halfcheetah": "Offline complete",
}
OFFLINE_LOG_NAMES = {
    "2halfcheetah": (
        "pv_offline_2halfcheetah_medium.out",
        "pv_offline_2hc_medium.out",
    ),
}


def offline_finished(map_name: str) -> tuple[bool, str]:
    """Whether the shared offline checkpoint for this map is final.

    The offline checkpoint is rewritten during pretraining (best-so-far), so file
    existence alone is not enough.  For the map pretrained in this session the
    completion line in its log is the signal; every other map's checkpoint
    predates this session and is already final.
    """
    marker = OFFLINE_DONE_MARKERS.get(map_name)
    if marker is None:
        return True, ""
    for name in OFFLINE_LOG_NAMES.get(map_name, ()):
        log = os.path.join(REPO, "logs", name)
        if not os.path.isfile(log):
            continue
        with open(log, encoding="utf-8", errors="ignore") as handle:
            if marker in handle.read():
                return True, ""
    tried = ", ".join(OFFLINE_LOG_NAMES.get(map_name, ()))
    return False, f"waiting for the offline pretraining to finish (no '{marker}' in {tried})"


def last_t_env(path: str) -> int:
    """t_env of the last JSONL row, or -1 when unreadable."""
    if not os.path.isfile(path):
        return -1
    last = ""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                last = line
    if not last:
        return -1
    try:
        return int(float(json.loads(last).get("t_env", -1)))
    except Exception:
        return -1


class Job:
    def __init__(self, map_name: str, arm: str, seed: int, args):
        self.map = map_name
        self.arm = arm
        self.seed = seed
        self.args = args
        self.kind = "train"
        self.status = "pending"
        self.gpu = None
        self.proc = None
        self.note = ""

    @property
    def name(self) -> str:
        return f"{self.map}/{self.arm}/seed{self.seed}"

    @property
    def log(self) -> str:
        return log_path(self.map, self.arm, self.seed, self.args.smoke)

    def cmd(self) -> list:
        if self.kind == "scale_ckpt":
            return [
                PY,
                os.path.join(REPO, "scripts", "make_scale_corrected_ckpt.py"),
                "--config",
                MAP_CONFIG[self.map],
                "--seed",
                str(self.seed),
                "--offline-ckpt",
                MAP_OFFLINE[self.map],
                "--out",
                scaled_ckpt(self.map, self.seed),
            ]
        cmd = [
            "bash",
            DRIVER,
            self.map,
            self.arm,
            str(self.seed),
            "GPU",  # placeholder replaced at launch
        ]
        if self.args.smoke:
            cmd.append(f"t_max={self.args.t_max}")
        elif self.args.t_max:
            cmd.append(f"t_max={self.args.t_max}")
        return cmd

    def ready(self) -> tuple[bool, str]:
        if self.kind == "scale_ckpt":
            if os.path.isfile(scaled_ckpt(self.map, self.seed)):
                return False, "already built"
            # The scale-corrected start checkpoint is derived from the shared
            # offline checkpoint, so it must wait for that file to be final too.
            done, why = offline_finished(self.map)
            if not done:
                return False, why
            return True, ""
        offline = os.path.join(REPO, MAP_OFFLINE[self.map])
        if not os.path.isfile(offline):
            return False, f"waiting for offline checkpoint {MAP_OFFLINE[self.map]}"
        # The offline checkpoint is written DURING pretraining (best-so-far), so
        # its mere existence does not mean the shared start is final.  For the map
        # whose pretraining this session launched, wait for the run to actually
        # complete, otherwise arms started at different times would load different
        # versions of the same file.
        done, why = offline_finished(self.map)
        if not done:
            return False, why
        if self.arm == "D" and not os.path.isfile(scaled_ckpt(self.map, self.seed)):
            return False, "waiting for the arm-D start checkpoint"
        out = os.path.join(run_dir(self.map, self.arm, self.seed, self.args.smoke), "online.jsonl")
        t_max = self.args.t_max or 1_000_000
        if last_t_env(out) >= t_max:
            return False, "already finished"
        return True, ""

    def launch(self, gpu: int) -> None:
        cmd = self.cmd()
        cmd = [str(gpu) if part == "GPU" else part for part in cmd]
        os.makedirs(os.path.dirname(self.log) or ".", exist_ok=True)
        handle = open(self.log, "a", encoding="utf-8")
        handle.write(f"\n=== [scheduler] launch {self.name} on gpu {gpu} ===\n")
        handle.flush()
        env = dict(os.environ)
        if self.args.smoke:
            # Keep short validation runs out of the real experiment output.
            env["PV_SMOKE"] = "1"
        # The scale-checkpoint job and the training runs both print progress; without
        # unbuffered stdout those lines sit in the block buffer until the process
        # exits, which makes a long build look stalled.
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            cmd,
            cwd=REPO,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self.gpu = gpu
        self.status = "running"
        print(f"[scheduler] start {self.name} on gpu {gpu} pid {self.proc.pid}", flush=True)

    def poll(self) -> "int | None":
        """Return the GPU slot this job released by finishing, else None."""
        if self.proc is None:
            return None
        code = self.proc.poll()
        if code is None:
            return None
        self.status = "done" if code == 0 else f"failed(rc={code})"
        print(f"[scheduler] {self.name} {self.status}", flush=True)
        self.proc = None
        gpu = self.gpu
        self.gpu = None
        return gpu


def _write_status(path: str, jobs: list, slots: dict, started: float) -> None:
    summary = {
        "elapsed_s": int(time.time() - started),
        "running": [j.name for j in jobs if j.status == "running"],
        "pending": [j.name for j in jobs if j.status == "pending"],
        "done": [j.name for j in jobs if j.status == "done"],
        "skipped": [j.name for j in jobs if j.status == "skipped"],
        "failed": [j.name for j in jobs if j.status.startswith("failed")],
        "slots": slots,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def build_jobs(args) -> list:
    jobs = []
    for map_name in args.maps:
        for arm in args.arms:
            for seed in args.seeds:
                jobs.append(Job(map_name, arm, seed, args))
                if arm == "D":
                    ckpt_job = Job(map_name, "D", seed, args)
                    ckpt_job.kind = "scale_ckpt"
                    jobs.append(ckpt_job)
    return jobs


def print_plan(jobs: list, args) -> None:
    print(f"repo:        {REPO}")
    print(f"maps:        {' '.join(args.maps)}")
    print(f"arms:        {' '.join(args.arms)}")
    print(f"seeds:       {' '.join(str(s) for s in args.seeds)}")
    print(f"gpus:        {' '.join(str(g) for g in args.gpus)}")
    print(f"max per gpu: {args.max_per_gpu}")
    print(f"t_max:       {args.t_max or 1_000_000}{' (smoke)' if args.smoke else ''}")
    print(f"jobs:        {len(jobs)}")
    for job in jobs:
        ok, note = job.ready()
        state = "READY" if ok else f"SKIP ({note})"
        print(f"  {job.kind:<11} {job.name:<28} {state}")
    waiting = [
        j.name
        for j in jobs
        if not j.ready()[0] and "waiting" in j.ready()[1]
    ]
    if waiting:
        print(f"\n{len(waiting)} job(s) blocked on dependencies: {', '.join(sorted(set(waiting)))}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maps", nargs="+", default=["5m_vs_6m", "2halfcheetah"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--max-per-gpu", type=int, default=2)
    ap.add_argument("--t-max", type=int, default=0, help="override the online budget")
    ap.add_argument("--smoke", action="store_true", help="short runs under checkpoints/pv_smoke")
    ap.add_argument("--plan", action="store_true", help="print the job list and exit")
    ap.add_argument("--poll", type=int, default=20, help="status interval in seconds")
    ap.add_argument(
        "--max-wait",
        type=int,
        default=6 * 3600,
        help="give up after this long waiting for dependencies (seconds)",
    )
    args = ap.parse_args()

    for map_name in args.maps:
        if map_name not in MAP_CONFIG:
            print(f"FATAL: unknown map {map_name}", file=sys.stderr)
            return 2
    for arm in args.arms:
        if arm not in ARMS:
            print(f"FATAL: unknown arm {arm}", file=sys.stderr)
            return 2
    if not os.path.isfile(DRIVER):
        print(f"FATAL: driver missing: {DRIVER}", file=sys.stderr)
        return 2

    jobs = build_jobs(args)
    if args.plan:
        print_plan(jobs, args)
        return 0

    status_path = os.path.join(REPO, "logs", "pv_scheduler_status.json")
    slots = {gpu: 0 for gpu in args.gpus}
    started = time.time()
    waited = 0
    print(f"[scheduler] {len(jobs)} jobs, gpus={args.gpus}, max_per_gpu={args.max_per_gpu}", flush=True)

    while True:
        for job in jobs:
            released = job.poll()
            if released is not None:
                slots[released] = max(0, slots[released] - 1)

        for job in jobs:
            if job.status != "pending":
                continue
            ok, note = job.ready()
            if not ok:
                job.note = note
                # Nothing left to do for this job (already built / already run).
                if note.startswith("already"):
                    job.status = "skipped"
                continue
            free = [gpu for gpu, used in slots.items() if used < args.max_per_gpu]
            if not free:
                break
            gpu = min(free, key=lambda g: (slots[g], g))
            job.launch(gpu)
            slots[gpu] += 1

        running = [j for j in jobs if j.status == "running"]
        if running:
            _write_status(status_path, jobs, slots, started)
            time.sleep(args.poll)
            continue

        pending = [j for j in jobs if j.status == "pending"]
        if not pending:
            break
        waiting = [j for j in pending if "waiting" in (j.note or j.ready()[1])]
        if not waiting:
            print("[scheduler] no launchable jobs left; stopping", flush=True)
            break
        print(
            f"[scheduler] {len(pending)} job(s) blocked on dependencies; "
            f"sleeping {args.poll}s",
            flush=True,
        )
        for job in pending:
            print(f"    {job.name}: {job.note or job.ready()[1]}", flush=True)
        waited = waited + args.poll
        if waited >= args.max_wait:
            print(f"[scheduler] gave up after waiting {waited}s for dependencies", flush=True)
            break
        _write_status(status_path, jobs, slots, started)
        time.sleep(args.poll)

    failed = [j.name for j in jobs if j.status.startswith("failed")]
    print(f"[scheduler] finished in {int(time.time() - started)}s; failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
