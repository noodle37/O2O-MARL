#!/usr/bin/env python3
"""Watch the TCPA attribution matrix and refresh its readout automatically.

The matrix runs for many hours in the background (card 3 is shared, so the
throughput is modest).  This watcher keeps the analysis outputs in step with
the training, so the numbers are ready whenever the run is looked at instead of
having to be produced by hand.

It does four things on every tick:

1. reports each arm's progress (evaluation points, last ``t_env``, errors);
2. re-runs ``scripts/analyze_tcpa_attrib.py`` whenever any arm has advanced,
   writing the results table and the JSON summary;
3. prints the machine-written verdict (P1 / P2) so the log is self-contained;
4. exits after the final refresh once every arm has reached the step budget
   (or the launcher is gone and nothing is advancing any more).

Card 3 is shared with another user's job, so the launcher can sit in its
memory gate for a long time without any arm advancing.  That is *not* a dead
run, so the watcher distinguishes "the launcher is alive but waiting" from
"the launcher is gone": only the latter ends the watch.

Usage:
    nohup python scripts/watch_tcpa_attrib.py > logs/tcpa_attrib/watcher.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List

ARMS = (
    "none_matched",
    "prior_matched",
    "none_broken",
    "prior_broken",
    "anchor_matched",
    "anchor_broken",
    "placebo_matched",
    "placebo_broken",
)

DECISIVE = ("none_matched", "prior_matched", "none_broken", "prior_broken")


def arm_progress(path: str) -> Dict[str, object]:
    """Evaluation points, last ``t_env`` and error count of one arm."""
    info: Dict[str, object] = {"n_eval": 0, "last_t_env": 0.0, "errors": 0}
    log = os.path.join(path, "online.jsonl")
    if os.path.isfile(log):
        rows = []
        with open(log, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "evaluation/mean_episode_return" in row:
                    rows.append(float(row.get("t_env", 0.0)))
        if rows:
            info["n_eval"] = len(rows)
            info["last_t_env"] = max(rows)
    out = os.path.join(path, "train.out")
    if os.path.isfile(out):
        with open(out, encoding="utf-8", errors="ignore") as handle:
            info["errors"] = handle.read().count("Traceback")
    return info


def launcher_process_alive() -> bool:
    """True while the queue manager itself is running.

    The launcher is matched on its *exact* command line so that a shell that
    merely mentions the script (the wrapper that started it) is not counted as
    the launcher itself.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-x", "-f", "bash scripts/launch_tcpa_attrib.sh"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def arms_running() -> bool:
    """True while at least one training arm is alive."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "train.py --config configs/tcpa_attrib"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def arm_running(arm: str, root: str, seed: int) -> bool:
    """True while this arm's training process is alive.

    The pattern is the arm's checkpoint directory, which appears verbatim in
    the training command line.  The pattern is assembled at run time so that
    it never appears in this process' own command line (``pgrep -f`` would
    otherwise match the caller).
    """
    out = os.path.join(root, arm, f"seed{seed}")
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"checkpoint_dir={out}"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def missing_arms(args, progress: Dict[str, Dict]) -> List[str]:
    """Arms that are neither finished nor running.

    An arm can end up in this state when the launcher gave up on it after
    all its retries failed (a shared card can OOM at any moment).  The
    launcher never revisits such an arm by itself, so without this check a
    decisive arm could simply never run -- and the matrix would look
    "finished" with a hole in it.
    """
    missing: List[str] = []
    for arm in ARMS:
        info = progress.get(arm) or {}
        last = float(info.get("last_t_env", 0.0))
        if int(info.get("n_eval", 0)) > 0 and last >= args.t_max - 50_000:
            continue  # already reached the budget
        if arm_running(arm, args.root, args.seed):
            continue  # still training
        missing.append(arm)
    return missing


def launcher_alive() -> bool:
    """True while the launcher or one of its training arms is still running.

    Card 3 is shared with another user's job, and the launcher blocks in its
    memory gate whenever the headroom it needs is not there.  That looks like
    a stalled matrix but is not one, so the watcher must not treat it as a
    dead run and must not walk away from it.
    """
    return launcher_process_alive() or arms_running()


def restart_launcher(args, stamp: str) -> bool:
    """Bring the queue manager back after it died with arms still running.

    If the launcher dies, the arms that are already training finish normally,
    but the queue stops: no further arm is ever admitted, so the decisive
    ``prior_broken`` arm would never run and the pre-registered verdict could
    not be computed at all.  The launcher is restart-safe (it skips arms that
    are already running or already finished) and takes a lock, so restarting
    it cannot duplicate an arm.
    """
    if args.no_restart_launcher:
        return False
    script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "launch_tcpa_attrib.sh"
    )
    if not os.path.isfile(script):
        return False
    driver_log = args.driver_log
    try:
        os.makedirs(os.path.dirname(os.path.abspath(driver_log)), exist_ok=True)
        handle = open(driver_log, "a", encoding="utf-8")
        subprocess.Popen(
            ["bash", script],
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return False
    print(
        f"[{stamp}] the launcher was gone while arms were still training; "
        "restarted it (the queue continues)",
        flush=True,
    )
    return True


def read_verdict(path: str) -> str:
    """One-line verdict from the analyzer's JSON summary, if it is readable."""
    try:
        with open(path, encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return ""
    verdict = summary.get("verdict") or {}
    p1 = verdict.get("P1") or {}
    p2 = verdict.get("P2") or {}

    def one(stat: Dict) -> str:
        mean = stat.get("mean")
        lo = stat.get("lo")
        hi = stat.get("hi")
        if mean is None:
            return "无数据"
        if lo is None:
            return f"{mean:+.2f} (n={stat.get('n', 0)})"
        return f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}] (n={stat.get('n', 0)})"

    conclusion = str(verdict.get("conclusion", "")).replace("**", "").strip()
    block = verdict.get("block_bootstrap") or {}
    b1 = block.get("P1") or {}
    b2 = block.get("P2") or {}
    veto = "块自助已撤回登记结论" if block.get("vetoed") else "块自助未触发撤回"
    align = verdict.get("pairing_alignment") or {}
    if align and not align.get("ok", True):
        align_text = "配对错位（结论已撤回）"
    else:
        align_text = "配对正常"
    return (
        f"verdict: {conclusion} | P1={one(p1)} {p1.get('class', '')}"
        f" | P2={one(p2)} {p2.get('class', '')}"
        f" | 块自助P1={one(b1)} {b1.get('class', '')}"
        f" | 块自助P2={one(b2)} {b2.get('class', '')}"
        f" | {veto} | {align_text}"
    )


def run_finalizer(args, stamp: str) -> None:
    """Write the final report, whether or not every arm finished.

    The report has to exist even when the matrix ended early: it is the place
    where the missing arms are listed and where the verdict is qualified.
    """
    if args.no_finalize or not os.path.isfile(args.finalizer):
        return
    final = subprocess.run(
        [
            args.python,
            args.finalizer,
            "--root",
            args.root,
            "--seed",
            str(args.seed),
            "--t-max",
            str(args.t_max),
            "--json",
            args.json,
            "--results-md",
            args.markdown,
            "--python",
            args.python,
        ],
        capture_output=True,
        text=True,
    )
    if final.returncode == 0:
        tail = final.stdout.strip().splitlines()[-1] if final.stdout.strip() else "ok"
        print(f"[{stamp}] final report written: {tail}", flush=True)
    else:
        print(
            f"[{stamp}] finalizer FAILED rc={final.returncode}: "
            f"{final.stderr.strip()[-400:]}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="checkpoints/tcpa_attrib_5m_vs_6m")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--interval", type=int, default=1800, help="seconds between ticks")
    parser.add_argument("--t-max", type=int, default=1_000_000)
    parser.add_argument(
        "--json",
        default="checkpoints/tcpa_attrib_5m_vs_6m/summary.json",
    )
    parser.add_argument(
        "--markdown",
        default="documentation/experiment_reports/tcpa_attrib_5m_vs_6m_results.md",
    )
    parser.add_argument(
        "--analyzer",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_tcpa_attrib.py"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--once", action="store_true", help="one tick and exit")
    parser.add_argument(
        "--finalizer",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "finalize_tcpa_attrib.py"
        ),
        help="script that writes the final report once every arm is done",
    )
    parser.add_argument(
        "--no-finalize", action="store_true", help="do not run the finalizer"
    )
    parser.add_argument(
        "--no-restart-launcher",
        action="store_true",
        help="do not restart the queue manager if it dies with arms still running",
    )
    parser.add_argument(
        "--driver-log",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs",
            "tcpa_attrib",
            "driver.log",
        ),
        help="where a restarted launcher writes its own log",
    )
    args = parser.parse_args()

    previous: Dict[str, int] = {}
    stagnant_ticks = 0

    while True:
        progress = {
            arm: arm_progress(os.path.join(args.root, arm, f"seed{args.seed}"))
            for arm in ARMS
        }
        advanced = False
        for arm, info in progress.items():
            if int(info["n_eval"]) != previous.get(arm, -1):
                advanced = True
            previous[arm] = int(info["n_eval"])

        stamp = time.strftime("%F %T")
        summary_line = " ".join(
            f"{arm}={int(info['n_eval'])}@{int(info['last_t_env'])}"
            + ("!" if int(info["errors"]) else "")
            for arm, info in progress.items()
        )
        print(f"[{stamp}] {summary_line}", flush=True)

        if advanced:
            cmd = [
                args.python,
                args.analyzer,
                "--root",
                args.root,
                "--seed",
                str(args.seed),
                "--json",
                args.json,
                "--markdown",
                args.markdown,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[{stamp}] analysis refreshed -> {args.markdown}", flush=True)
                verdict = read_verdict(args.json)
                if verdict:
                    print(f"[{stamp}] {verdict}", flush=True)
            else:
                print(
                    f"[{stamp}] analysis FAILED rc={result.returncode}: "
                    f"{result.stderr.strip()[-400:]}",
                    flush=True,
                )

        done = [arm for arm in ARMS if int(progress[arm]["n_eval"]) > 0
                and float(progress[arm]["last_t_env"]) >= args.t_max - 50_000]
        if len(done) == len(ARMS):
            print(f"[{stamp}] all arms finished the budget; final refresh done", flush=True)
            run_finalizer(args, stamp)
            return

        if args.once:
            return

        stagnant_ticks = 0 if advanced else stagnant_ticks + 1
        launcher_ok = launcher_process_alive()
        arms_ok = arms_running()

        if not launcher_ok:
            missing = missing_arms(args, progress)
            if arms_ok or missing:
                # The queue manager died but the matrix is not over: arms are
                # still training, or arms were never run (or were abandoned
                # after all their retries failed).  Without a restart the queue
                # stops for good and the decisive arms would never run.
                if stagnant_ticks >= 1:
                    if restart_launcher(args, stamp):
                        stagnant_ticks = 0
                else:
                    print(
                        f"[{stamp}] the launcher is gone but the matrix is not over "
                        f"({len(missing)} arm(s) queued: {'、'.join(missing)}); "
                        "watching for progress before restarting the queue",
                        flush=True,
                    )
        elif stagnant_ticks >= 6 and launcher_alive():
            # The matrix is not dead: the launcher is waiting for card 3 to
            # free enough memory for the next arm.  Keep watching, but say so.
            print(
                f"[{stamp}] no new evaluation point for {stagnant_ticks} ticks, "
                "but the launcher is still alive (waiting for card memory) - keeping watch",
                flush=True,
            )
            if stagnant_ticks >= 24:
                print(
                    f"[{stamp}] still stuck after {stagnant_ticks} ticks; stopping the watcher",
                    flush=True,
                )
                run_finalizer(args, stamp)
                return
        elif stagnant_ticks >= 6:
            # Three hours without a single new evaluation point, no launcher
            # and no running arm: the matrix is over (crashed or finished).
            print(
                f"[{stamp}] no progress for {stagnant_ticks} ticks and nothing is running; "
                "stopping the watcher",
                flush=True,
            )
            # The matrix may have ended with an arm missing (a crashed arm
            # that was abandoned, or a launcher that died).  Write the report
            # anyway: it lists the arms that did not finish and says the
            # verdict only covers the ones that did.
            run_finalizer(args, stamp)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
