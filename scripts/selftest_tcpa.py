#!/usr/bin/env python3
"""Property self-test for the TCPA implementation.

Runs on synthetic batches, so it needs neither SMAC nor the vault data.  It
checks the properties the method relies on:

1.  the zero-initialised residual makes the online utility *identical* to the
    frozen offline utility at the start of online training;
2.  at that point the team regret is exactly zero and the retention gate is
    exactly one;
3.  the IGM ranking loss is zero while the offline action is locally greedy,
    and zero when an agent has a single legal action;
4.  the mixing network stays monotone, so the team regret is non-negative;
5.  only the residual and the mixer receive gradients;
6.  the checkpoint round-trip reproduces the same utilities.
"""

from __future__ import annotations

import os
import sys
import tempfile

from typing import Dict, List

import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcpa.learner import TCPALearner  # noqa: E402
from tcpa.losses import igm_ranking_loss  # noqa: E402
from tcpa.regret import masked_argmax, team_regret  # noqa: E402

DEVICE = "cpu"
B, T, N, A = 4, 6, 3, 5
OBS_DIM, STATE_DIM = 8, 11


def make_batch(rng: torch.Generator) -> dict:
    obs = torch.randn(B, T, N, OBS_DIM, generator=rng)
    avail = torch.ones(B, T, N, A, dtype=torch.bool)
    avail[:, :, 0, 3:] = False  # agent 0 has three legal actions
    avail[:, :, 1, 1:] = False  # agent 1 has a single legal action
    avail[0, 3, 2, :] = False  # agent 2 is dead at one step
    actions = torch.zeros(B, T, N, 1, dtype=torch.long)
    for b in range(B):
        for t in range(T):
            for i in range(N):
                legal = torch.nonzero(avail[b, t, i]).flatten()
                actions[b, t, i, 0] = legal[0] if legal.numel() else 0
    state = torch.randn(B, T, STATE_DIM, generator=rng)
    reward = torch.randn(B, T, 1, generator=rng)
    terminated = torch.zeros(B, T, 1)
    terminated[:, -1, 0] = 1.0
    filled = torch.ones(B, T, 1)
    filled[:, -1, 0] = 0.0
    return {
        "obs": obs,
        "actions": actions,
        "state": state,
        "reward": reward,
        "terminated": terminated,
        "filled": filled,
        "avail_actions": avail,
    }


def build_learner(variant: str = "full", seed: int = 0) -> TCPALearner:
    torch.manual_seed(seed)
    learner = TCPALearner(
        n_agents=N,
        n_actions=A,
        obs_dim=OBS_DIM,
        state_dim=STATE_DIM,
        variant=variant,
        seed=seed,
        device=DEVICE,
    )
    # Give the offline base a non-degenerate ranking so that the greedy
    # actions are not all zero.
    torch.manual_seed(0)
    with torch.no_grad():
        learner.q_network.base.fc2.weight.normal_(mean=0.0, std=0.5)
        learner.q_network.base.fc2.bias.normal_(mean=0.0, std=0.5)
    return learner


def perturb_residual(learner: TCPALearner, std: float = 0.5) -> None:
    """Make the online utilities differ from the frozen offline ones."""
    with torch.no_grad():
        learner.q_network.residual.fc2.weight.normal_(mean=0.0, std=std)
        learner.q_network.residual.fc2.bias.normal_(mean=0.0, std=std)


def write_offline_checkpoint(learner: TCPALearner, path: str) -> None:
    torch.save(
        {
            "q_network": learner.q_network.base.state_dict(),
            "mixer": learner.mixer.state_dict(),
        },
        path,
    )


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        raise SystemExit(f"self-test failed: {name}")


def check_attribution_arms(batch: dict) -> None:
    """The two attribution arms and the teammate-cooperation condition.

    * ``value_anchor`` is a generic trust region: it must be zero while the
      online utilities still equal the offline ones, and it must not select
      any action.
    * ``placebo`` is a matched negative control: the retention gate and the
      team regret must be *identical* to the ``full`` arm (only the retained
      action differs), while the hinge itself must differ.
    * ``coop_mode='frozen_offline'`` must make the frozen agents follow the
      frozen offline prior and leave the focal agent on the online policy.
    """
    print("\n-- attribution arms --")

    # --- value_anchor -----------------------------------------------------
    anchor = build_learner(variant="value_anchor", seed=11)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "offline.pt")
        write_offline_checkpoint(anchor, ckpt)
        anchor.load_offline_for_ovm(ckpt)
        anchor.prepare_online_start()
        logs = anchor.train_on_batch(batch, cql_weight=0.0, use_ovm=False, t_env=100)
        check(
            "value_anchor is zero while q_on == q_off",
            float(logs["tcpa/anchor_dev_mean"]) < 1e-6,
            f"dev={float(logs['tcpa/anchor_dev_mean']):.3e}",
        )
        check(
            "value_anchor uses its own weight",
            abs(float(logs["tcpa/eta_prior"]) - float(anchor.eta_anchor)) < 1e-9,
        )
        # After the residual moves away from zero the anchor must bite.
        perturb_residual(anchor)
        logs = anchor.train_on_batch(batch, cql_weight=0.0, use_ovm=False, t_env=200)
        check(
            "value_anchor grows once the residual moves",
            float(logs["tcpa/anchor_dev_mean"]) > 0.0,
            f"dev={float(logs['tcpa/anchor_dev_mean']):.3e}",
        )

    # --- placebo: matched negative control --------------------------------
    full = build_learner(variant="full", seed=12)
    placebo = build_learner(variant="placebo", seed=12)
    for learner in (full, placebo):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "offline.pt")
            write_offline_checkpoint(learner, ckpt)
            learner.load_offline_for_ovm(ckpt)
            learner.prepare_online_start()
    perturb_residual(full)
    perturb_residual(placebo)
    logs_full = full.train_on_batch(batch, cql_weight=0.0, use_ovm=False, t_env=100)
    logs_placebo = placebo.train_on_batch(batch, cql_weight=0.0, use_ovm=False, t_env=100)
    check(
        "placebo keeps the same team regret as full",
        abs(
            float(logs_full["tcpa/regret_mean"])
            - float(logs_placebo["tcpa/regret_mean"])
        )
        < 1e-9,
        f"R_full={float(logs_full['tcpa/regret_mean']):.6f} "
        f"R_placebo={float(logs_placebo['tcpa/regret_mean']):.6f}",
    )
    check(
        "placebo keeps the same retention gate as full",
        abs(
            float(logs_full["tcpa/gate_mean"])
            - float(logs_placebo["tcpa/gate_mean"])
        )
        < 1e-9,
        f"g_full={float(logs_full['tcpa/gate_mean']):.6f} "
        f"g_placebo={float(logs_placebo['tcpa/gate_mean']):.6f}",
    )
    check(
        "placebo targets a different action than full",
        abs(
            float(logs_full["tcpa/hinge_mean"])
            - float(logs_placebo["tcpa/hinge_mean"])
        )
        > 1e-9,
        f"hinge_full={float(logs_full['tcpa/hinge_mean']):.6f} "
        f"hinge_placebo={float(logs_placebo['tcpa/hinge_mean']):.6f}",
    )
    # The placebo target must be a legal action wherever an agent is alive.
    avail = batch["avail_actions"]
    target = placebo._random_legal_actions(avail)
    legal = torch.gather(avail.bool(), 3, target.unsqueeze(-1)).squeeze(-1)
    alive = avail.bool().any(dim=-1)
    check(
        "placebo target is always a legal action",
        bool(legal[alive].all()),
        f"illegal={int((~legal[alive]).sum())}",
    )

    # --- teammate-cooperation condition -----------------------------------
    coop = build_learner(variant="full", seed=13)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "offline.pt")
        write_offline_checkpoint(coop, ckpt)
        coop.load_offline_for_ovm(ckpt)
        coop.prepare_online_start()
    perturb_residual(coop, std=2.0)
    coop.coop_mode = "frozen_offline"
    coop.coop_frozen_agents = (1, 2)
    coop.coop_frozen_set = frozenset((1, 2))

    torch.manual_seed(3)
    obs = {f"agent_{i}": np.random.randn(OBS_DIM).astype(np.float32) for i in range(N)}
    legals = {
        f"agent_{i}": np.array([True] * A, dtype=np.float32) for i in range(N)
    }
    state = coop.new_episode_state()
    check(
        "the rollout state carries the frozen prior's hidden state",
        "offline_rnn" in state,
    )
    pre_offline_rnn = dict(state["offline_rnn"])
    pre_rnn = dict(state["rnn"])
    actions = coop.select_actions_eval(obs, legals, 0.0, state)

    def _obs_input(agent_index: int) -> torch.Tensor:
        agent = f"agent_{agent_index}"
        pieces = [
            torch.from_numpy(obs[agent]).float(),
            torch.zeros(A),
        ]
        if coop.add_agent_id_to_obs:
            pieces.append(
                torch.nn.functional.one_hot(
                    torch.tensor(agent_index), N
                ).float()
            )
        return torch.cat(pieces, dim=-1).unsqueeze(0)

    agree_offline = 0
    focal_online_ok = 0
    frozen_differ = 0
    differs = 0
    for i in range(N):
        agent = f"agent_{i}"
        with torch.no_grad():
            q_off, _ = coop.offline_q_network(
                _obs_input(i), pre_offline_rnn[agent]
            )
            q_on, _ = coop.q_network(_obs_input(i), pre_rnn[agent])
        off_greedy = int(torch.argmax(q_off[0]).item())
        on_greedy = int(torch.argmax(q_on[0]).item())
        if off_greedy != on_greedy:
            differs += 1
        if i in (1, 2):
            agree_offline += int(actions[agent] == off_greedy)
            frozen_differ += int(actions[agent] != on_greedy)
        else:
            focal_online_ok = int(actions[agent] == on_greedy)
    check(
        "frozen teammates follow the frozen offline prior",
        agree_offline == 2,
        f"agree={agree_offline}/2",
    )
    check(
        "the focal agent follows the online policy",
        focal_online_ok == 1,
        f"agree={focal_online_ok}/1",
    )
    check(
        "the frozen teammates are really held on the prior",
        frozen_differ >= 1,
        f"n_frozen_that_differ_from_online={frozen_differ}/2",
    )
    check(
        "the two policies really differ (the condition is not a no-op)",
        differs > 0,
        f"n_differ={differs}/{N}",
    )

    # The cooperative condition must be the default behaviour.
    coop.coop_mode = "default"
    coop.coop_frozen_set = frozenset()
    state2 = coop.new_episode_state()
    pre_rnn2 = dict(state2["rnn"])
    actions2 = coop.select_actions_eval(obs, legals, 0.0, state2)
    check(
        "the cooperative condition leaves every agent on the online policy",
        all(
            actions2[f"agent_{i}"]
            == int(
                torch.argmax(
                    coop.q_network(_obs_input(i), pre_rnn2[f"agent_{i}"])[0][0]
                ).item()
            )
            for i in range(N)
        ),
    )


def check_analysis_pairing() -> None:
    """The attribution analysis must pair arms by evaluation index.

    The arms share one fixed-seed evaluation environment, so the n-th
    evaluation of two arms plays the same episodes -- but the arms accumulate
    env steps at their own pace, so their ``t_env`` at that index differ.
    Pairing on the exact ``t_env`` would leave only the two t=0 rows in
    common.  These checks pin the index pairing, the restart handling and the
    symmetry of the interaction term.
    """
    print("\n-- attribution analysis pairing --")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from analyze_tcpa_attrib import eval_curve, interaction, paired_contrast  # noqa: E402

    def curve(points):
        return eval_curve(
            [{"t_env": t, "evaluation/mean_episode_return": v} for t, v in points]
        )

    a = curve([(0.0, 10.0), (10107.0, 9.0), (20162.0, 8.0)])
    b = curve([(0.0, 10.0), (10065.0, 7.0), (20152.0, 6.0)])

    stat = paired_contrast(a, b, 0, 10**9)
    check(
        "arms are paired by evaluation index, not by exact t_env",
        stat["n"] == 3,
        f"n={stat['n']} (exact t_env would give 1)",
    )
    check(
        "paired contrast is antisymmetric",
        abs(
            float(paired_contrast(b, a, 0, 10**9)["mean"])
            + float(stat["mean"])
        )
        < 1e-9,
    )
    check(
        "interaction of a curve with itself is exactly zero",
        abs(float(interaction(a, b, a, b, 0, 10**9)["mean"])) < 1e-9,
    )
    check(
        "a window with no points yields no statistic",
        paired_contrast(a, b, 600_000, 10**9)["mean"] is None,
    )

    # The window boundaries must be the ones that were pre-registered:
    # early = 0-300k, mid = 300k-600k, late = 600k+.  An earlier version of
    # the analyzer halved the paired time, which silently doubled every
    # boundary and let the early window swallow 0-600k.
    from analyze_tcpa_attrib import _window_index  # noqa: E402

    long_a = curve([(0.0, 10.0)] + [(i * 10_000.0, 9.0) for i in range(1, 121)])
    long_b = curve([(0.0, 10.0)] + [(i * 10_000.0, 9.0) for i in range(1, 121)])
    early_idx = _window_index([long_a, long_b], 0, 300_000)
    mid_idx = _window_index([long_a, long_b], 300_000, 600_000)
    late_idx = _window_index([long_a, long_b], 600_000, 10**9)
    check(
        "the early window really stops at 300k",
        max(long_a[i][0] for i in early_idx) < 300_000,
        f"max t in early = {max(long_a[i][0] for i in early_idx)}",
    )
    check(
        "the mid window starts at 300k and stops at 600k",
        min(long_a[i][0] for i in mid_idx) >= 300_000
        and max(long_a[i][0] for i in mid_idx) < 600_000,
        f"mid spans {min(long_a[i][0] for i in mid_idx)}"
        f"..{max(long_a[i][0] for i in mid_idx)}",
    )
    check(
        "the late window starts at 600k",
        min(long_a[i][0] for i in late_idx) >= 600_000,
        f"min t in late = {min(long_a[i][0] for i in late_idx)}",
    )
    check(
        "the three windows partition the curve",
        len(early_idx) + len(mid_idx) + len(late_idx) == len(long_a),
        f"{len(early_idx)}+{len(mid_idx)}+{len(late_idx)} vs {len(long_a)}",
    )

    # A restarted arm appends its rows; only the last run may be used.
    restarted = curve(
        [
            (0.0, 10.0),
            (0.0, 10.0),
            (10100.0, 9.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (10100.0, 2.0),
        ]
    )
    check(
        "a restarted arm keeps only its last run",
        len(restarted) == 3 and restarted[-1][1]["evaluation/mean_episode_return"] == 2.0,
        f"n={len(restarted)}",
    )


def check_pre_registered_verdict() -> None:
    """The pre-registered decision rule must map statistics to the right call.

    The rule was written down on 2026-11-12 02:25, before any of the decisive
    arms had finished.  These checks pin its three branches so the conclusion
    cannot drift with the implementation.
    """
    print("\n-- pre-registered verdict --")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from analyze_tcpa_attrib import classify, verdict_from_classes  # noqa: E402

    pos = {"mean": 1.0, "lo": 0.4, "hi": 1.6}
    null = {"mean": 0.1, "lo": -0.5, "hi": 0.7}
    neg = {"mean": -1.0, "lo": -1.6, "hi": -0.4}
    empty = {"mean": None, "lo": None, "hi": None}

    check("CI strictly above zero -> 显著为正", classify(pos) == "显著为正")
    check("CI straddling zero -> 含 0", classify(null) == "含 0")
    check("CI strictly below zero -> 显著为负", classify(neg) == "显著为负")
    check("no data -> 无数据", classify(empty) == "无数据")

    check(
        "P1 and P2 both positive -> H2",
        "支持 H2" in verdict_from_classes("显著为正", "显著为正")[0],
    )
    check(
        "P1 and P2 both null -> H1",
        "支持 H1" in verdict_from_classes("含 0", "含 0")[0],
    )
    check(
        "mixed signs -> inconclusive, not a conclusion",
        "证据不足" in verdict_from_classes("显著为正", "含 0")[0]
        and "支持 H" not in verdict_from_classes("显著为正", "含 0")[0],
    )
    check(
        "missing data -> inconclusive",
        "证据不足" in verdict_from_classes("无数据", "显著为正")[0],
    )
    check(
        "a negative P1 does not get called H1",
        "支持 H1" not in verdict_from_classes("显著为负", "含 0")[0],
    )


def check_reporting_pipeline() -> None:
    """The report pipeline must be complete, idempotent and number-faithful.

    The matrix runs unattended for a day, so the final report is written by a
    script rather than by hand.  Two failure modes would be silent and
    expensive: a README index row that grows a second copy of the verdict on
    every refresh, and a report whose numbers no longer match the summary.
    """
    print("\n-- reporting pipeline --")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import types  # noqa: E402

    from finalize_tcpa_attrib import build_report, completeness, patch_readme  # noqa: E402
    import watch_tcpa_attrib as watch_module  # noqa: E402
    from watch_tcpa_attrib import launcher_alive, read_verdict  # noqa: E402

    arms = {}
    for arm in ("none_matched", "none_broken", "prior_matched"):
        arms[arm] = {
            "constraint": arm.split("_")[0],
            "cooperation": arm.split("_")[1],
            "n_eval": 7,
            "last_t_env": 50_357,
            "windows": {"early": 12.01, "mid": None, "late": None, "all": 12.01},
            "win_rate": 0.22,
        }
    contrasts = {
        "interaction prior (matched-broken)": {
            w: {"n": 30, "mean": 1.2, "lo": 0.3, "hi": 2.1} for w in ("early", "mid", "late", "all")
        },
        "prior − anchor (matched)": {
            w: {"n": 30, "mean": 0.8, "lo": 0.1, "hi": 1.5} for w in ("early", "mid", "late", "all")
        },
    }
    summary = {
        "arms": arms,
        "contrasts": contrasts,
        "verdict": {
            "window": "early",
            "P1": {"class": "显著为正", "n": 30, "mean": 1.2, "lo": 0.3, "hi": 2.1},
            "P2": {"class": "显著为正", "n": 30, "mean": 0.8, "lo": 0.1, "hi": 1.5},
            "conclusion": "**支持 H2**（旧动作的收益来自协作内容，且依赖队友配合）。",
        },
    }

    # --- completeness: finished / unfinished / never started -------------
    comp = completeness(summary, 1_000_000)
    check(
        "completeness separates finished, unfinished and absent arms",
        comp["incomplete"] == ["none_matched", "none_broken", "prior_matched"]
        and comp["absent"] == [
            "anchor_matched", "anchor_broken", "placebo_matched", "placebo_broken",
            "prior_broken", "fixed_gate_matched", "fixed_gate_broken",
        ]
        and not comp["all_complete"],
    )
    finished = {arm: dict(info, last_t_env=1_000_000) for arm, info in arms.items()}
    comp_done = completeness({"arms": finished}, 1_000_000)
    check(
        "an arm within 50k steps of the budget counts as finished",
        comp_done["complete"] == ["none_matched", "none_broken", "prior_matched"]
        and comp_done["incomplete"] == [],
    )

    # --- the report must carry the summary's numbers --------------------
    report = build_report(summary, comp, 1_000_000)
    check("the report states the verdict", "支持 H2" in report)
    check(
        "the report quotes the P1 estimate and its interval",
        "+1.20 [+0.30, +2.10]" in report,
    )
    check(
        "the report quotes the P2 estimate and its interval",
        "+0.80 [+0.10, +1.50]" in report,
    )
    check(
        "an unfinished matrix is flagged in the report",
        "本矩阵未全部跑完" in report,
    )

    # --- the README row must be patched idempotently --------------------
    with tempfile.TemporaryDirectory() as tmp:
        readme = os.path.join(tmp, "README.md")
        row = (
            "| **实验** | [tcpa_attrib_5m_vs_6m_design.md](tcpa_attrib_5m_vs_6m_design.md)（设计）"
            "· [tcpa_attrib_5m_vs_6m_results.md](tcpa_attrib_5m_vs_6m_results.md)（自动刷新的读数） | "
            "某个说明。运行中 |\n"
            "| **别的** | [other.md](other.md) | 别的说明 |\n"
        )
        with open(readme, "w", encoding="utf-8") as handle:
            handle.write(row)

        check("the index row is found", patch_readme(readme, "**结论：支持 H2**。"))
        patch_readme(readme, "**结论：支持 H2**。")
        patch_readme(readme, "**结论：支持 H1**。")
        with open(readme, encoding="utf-8") as handle:
            patched = handle.read()
        check(
            "patching the index row is idempotent",
            patched.count("tcpa-attrib-verdict") == 1 and patched.count("支持 H2") == 0,
        )
        check(
            "a changed verdict replaces the old one",
            patched.count("支持 H1") == 1,
        )
        check(
            "the final report is linked exactly once",
            patched.count("[tcpa_attrib_5m_vs_6m.md](tcpa_attrib_5m_vs_6m.md)") == 1,
        )
        check(
            "the other rows are untouched",
            "| **别的** | [other.md](other.md) | 别的说明 |" in patched,
        )
        check(
            "the table structure is preserved",
            patched.count("|") == row.count("|"),
            f"{patched.count('|')} vs {row.count('|')}",
        )

    # --- the watcher helpers must not raise ------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        empty = os.path.join(tmp, "summary.json")
        check("a missing summary yields no verdict line", read_verdict(empty) == "")
        with open(empty, "w", encoding="utf-8") as handle:
            json.dump(summary, handle)
        verdict_line = read_verdict(empty)
        check(
            "the verdict line carries both readings",
            "P1=+1.20" in verdict_line and "P2=+0.80" in verdict_line,
            verdict_line,
        )
    check(
        "the launcher liveness probe returns a boolean",
        isinstance(launcher_alive(), bool),
    )
    check(
        "the queue-manager probe and the arm probe are separate",
        isinstance(watch_module.launcher_process_alive(), bool)
        and isinstance(watch_module.arms_running(), bool),
    )

    # --- the queue rescue must be switchable and must not fire wrongly ---
    rescue_calls = []

    def _fake_popen(cmd, **kwargs):
        rescue_calls.append(cmd)
        return types.SimpleNamespace(pid=1234)

    original_popen = watch_module.subprocess.Popen
    watch_module.subprocess.Popen = _fake_popen
    try:
        rescue_args = types.SimpleNamespace(
            no_restart_launcher=True,
            driver_log="/tmp/tcpa_driver_test.log",
        )
        check(
            "the queue rescue can be switched off",
            watch_module.restart_launcher(rescue_args, "2026-11-12 03:30:00") is False,
        )
        check("a switched-off rescue spawns nothing", len(rescue_calls) == 0)

        rescue_args.no_restart_launcher = False
        rescue_args.driver_log = "/tmp/tcpa_driver_test.log"
        started = watch_module.restart_launcher(rescue_args, "2026-11-12 03:30:00")
        check(
            "the queue rescue starts the launcher detached",
            started and len(rescue_calls) == 1 and rescue_calls[0][0] == "bash",
            f"calls={len(rescue_calls)}",
        )
    finally:
        watch_module.subprocess.Popen = original_popen

    # --- the watcher must write the report even on an early exit --------
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="wrote x\n", stderr="")

    original_run = watch_module.subprocess.run
    watch_module.subprocess.run = _fake_run
    try:
        args = types.SimpleNamespace(
            no_finalize=False,
            finalizer=os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "scripts",
                "finalize_tcpa_attrib.py",
            ),
            python=sys.executable,
            root="checkpoints/tcpa_attrib_5m_vs_6m",
            seed=1,
            t_max=1_000_000,
            json="checkpoints/tcpa_attrib_5m_vs_6m/summary.json",
            markdown="documentation/experiment_reports/tcpa_attrib_5m_vs_6m_results.md",
        )
        watch_module.run_finalizer(args, "2026-11-12 02:50:00")
        check(
            "the final report is written when the watcher stops",
            len(calls) == 1 and calls[0][1].endswith("finalize_tcpa_attrib.py"),
        )
        calls.clear()
        args.no_finalize = True
        watch_module.run_finalizer(args, "2026-11-12 02:50:00")
        check("the finalizer can be switched off", len(calls) == 0)
        calls.clear()
        args.no_finalize = False
        args.finalizer = "/nonexistent/finalizer.py"
        watch_module.run_finalizer(args, "2026-11-12 02:50:00")
        check("a missing finalizer is not an error", len(calls) == 0)
    finally:
        watch_module.subprocess.run = original_run


def check_robustness_reading() -> None:
    """The robustness reading must be computed, and must not touch the verdict.

    The bootstrap interval describes the fluctuation between evaluation points
    of one trajectory; it says nothing about whether a handful of points carry
    the whole effect.  The extra readings (sign consistency, split halves) are
    reported next to it -- but they are exploratory, so they must not be able
    to change the pre-registered verdict.
    """
    print("\n-- robustness reading --")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from analyze_tcpa_attrib import stability  # noqa: E402

    empty = stability([])
    check("no points -> no statistic", empty["n"] == 0 and empty["mean"] is None)

    steady = stability([1.0, 2.0, 3.0, 4.0])
    check(
        "a steady difference has every point on the same side",
        steady["frac_positive"] == 1.0 and steady["sign_consistent"],
    )
    check(
        "the two halves of a steady difference agree",
        steady["halves_agree"] and steady["first_half_mean"] > 0 and steady["second_half_mean"] > 0,
    )

    noisy = stability([3.0, -1.0, -3.0, 1.0])
    check(
        "a sign-flipping difference is not reported as consistent",
        not noisy["sign_consistent"] and not noisy["halves_agree"],
    )
    check(
        "the sign fraction counts the points, not the sign",
        abs(noisy["frac_positive"] - 0.5) < 1e-9,
    )

    # End to end: the analyzer must produce the robustness block and a verdict.
    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "matrix")
        # prior beats none under matched, not under broken -> P1 is positive.
        # All ten arms: the report tables render one row per arm, so the
        # "every arm present" path has to be exercised too.
        profiles = {
            "none_matched": 10.0,
            "prior_matched": 12.0,
            "none_broken": 10.0,
            "prior_broken": 10.2,
            "anchor_matched": 12.0,
            "anchor_broken": 10.0,
            "placebo_matched": 12.0,
            "placebo_broken": 10.0,
            "fixed_gate_matched": 11.5,
            "fixed_gate_broken": 10.1,
        }
        for arm, base in profiles.items():
            folder = os.path.join(root, arm, "seed1")
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "online.jsonl"), "w", encoding="utf-8") as handle:
                for i in range(12):
                    row = {
                        "t_env": float(i * 10_000),
                        "evaluation/mean_episode_return": base + 0.01 * i,
                    }
                    handle.write(json.dumps(row) + "\n")
        out_json = os.path.join(tmp, "summary.json")
        out_md = os.path.join(tmp, "results.md")
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_tcpa_attrib.py"),
                "--root",
                root,
                "--seed",
                "1",
                "--json",
                out_json,
                "--markdown",
                out_md,
            ],
            capture_output=True,
            text=True,
        )
        check("the analyzer runs on a synthetic matrix", result.returncode == 0, result.stderr[-300:])
        if result.returncode == 0:
            with open(out_json, encoding="utf-8") as handle:
                summary = json.load(handle)
            check(
                "the summary carries the robustness block",
                "robustness" in summary and summary["robustness"]["P1"]["n"] > 0,
            )
            check(
                "the verdict is computed from the pre-registered rule",
                summary["verdict"]["P1"]["class"] in ("显著为正", "含 0", "显著为负", "无数据"),
            )
            with open(out_md, encoding="utf-8") as handle:
                markdown = handle.read()
            check(
                "the readout prints the robustness section",
                "5.1 稳健性" in markdown and "噪声地板" in markdown,
            )
            check(
                "the readout prints the manipulation check",
                "5.2 操纵检验" in markdown,
            )
            check(
                "every arm has a row in the readout",
                all(
                    arm in markdown
                    for arm in (
                        "none_matched", "prior_matched", "anchor_matched",
                        "placebo_matched", "fixed_gate_matched",
                    )
                ),
            )


def _write_synthetic_matrix(root: str, profiles: Dict[str, float], ramp: float = 0.01) -> None:
    """Write one online.jsonl per arm with a constant base plus a small ramp."""
    for arm, base in profiles.items():
        folder = os.path.join(root, arm, "seed1")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "online.jsonl"), "w", encoding="utf-8") as handle:
            for i in range(12):
                handle.write(
                    json.dumps(
                        {
                            "t_env": float(i * 10_000),
                            "evaluation/mean_episode_return": base + ramp * i,
                        }
                    )
                    + "\n"
                )


def check_manipulation_qualifier() -> None:
    """The H1 branch must be withdrawn when the manipulation did not bite.

    The registered rule reads "P1 contains 0 and P2 contains 0" as H1.  That
    reading presupposes the broken-teammate manipulation produced a measurable
    change; otherwise a P1 of zero means the experiment never produced the
    contrast.  The amendment may only *withdraw* the H1 reading, so the
    mechanical verdict string must stay exactly as registered while the
    qualifier is added next to it.
    """
    print("\n-- manipulation check --")
    import subprocess

    analyzer = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "analyze_tcpa_attrib.py"
    )

    def _run(root: str, tmp: str) -> Dict:
        out_json = os.path.join(tmp, "summary.json")
        result = subprocess.run(
            [
                sys.executable, analyzer, "--root", root, "--seed", "1",
                "--json", out_json, "--markdown", os.path.join(tmp, "results.md"),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return {"_error": result.stderr[-300:]}
        with open(out_json, encoding="utf-8") as handle:
            return json.load(handle)

    with tempfile.TemporaryDirectory() as tmp:
        # (a) nothing differs anywhere: the manipulation did not bite.
        root = os.path.join(tmp, "flat")
        _write_synthetic_matrix(
            root,
            {
                "none_matched": 10.0, "none_broken": 10.0,
                "prior_matched": 10.0, "prior_broken": 10.0,
                "anchor_matched": 10.0, "anchor_broken": 10.0,
                "placebo_matched": 10.0, "placebo_broken": 10.0,
            },
        )
        summary = _run(root, tmp)
        check("the analyzer runs on the flat matrix", "_error" not in summary)
        if "_error" not in summary:
            verdict = summary["verdict"]
            check(
                "the mechanical verdict string is unchanged",
                "支持 H1" in verdict["conclusion"],
                verdict["conclusion"][:60],
            )
            check(
                "a failed manipulation check withdraws the H1 reading",
                bool(verdict["qualifier"]) and not verdict["manipulation_effective"],
            )
            check(
                "the qualifier says the conclusion cannot be read as H1",
                "不能读作" in verdict["qualifier"],
            )

    with tempfile.TemporaryDirectory() as tmp:
        # (b) the manipulation bites but the constraint effects do not differ
        #     between the two conditions -> P1 = 0, P2 = 0, and H1 *is* readable.
        root = os.path.join(tmp, "bit")
        _write_synthetic_matrix(
            root,
            {
                "none_matched": 10.0, "none_broken": 8.0,
                "prior_matched": 12.0, "prior_broken": 10.0,
                "anchor_matched": 12.0, "anchor_broken": 10.0,
                "placebo_matched": 12.0, "placebo_broken": 10.0,
            },
        )
        summary = _run(root, tmp)
        check("the analyzer runs on the biting matrix", "_error" not in summary)
        if "_error" not in summary:
            verdict = summary["verdict"]
            check(
                "an effective manipulation passes the check",
                verdict["manipulation_effective"],
            )
            check(
                "no qualifier is added when the manipulation worked",
                not verdict["qualifier"],
            )
            check(
                "the registered H1 verdict stands",
                "支持 H1" in verdict["conclusion"],
                verdict["conclusion"][:60],
            )


def _write_series_matrix(root: str, series: List[float]) -> None:
    """Write a matrix whose only difference is ``prior_matched − none``.

    Every other arm is flat, so ``P1`` and ``P2`` both see exactly ``series``:
    the paired difference of the decisive readouts is the series under test.
    """
    n = len(series)
    flat = {
        "none_matched": 10.0,
        "none_broken": 10.0,
        "prior_broken": 10.0,
        "anchor_matched": 10.0,
        "anchor_broken": 10.0,
        "placebo_matched": 10.0,
        "placebo_broken": 10.0,
        "fixed_gate_matched": 10.0,
        "fixed_gate_broken": 10.0,
    }
    for arm, base in flat.items():
        folder = os.path.join(root, arm, "seed1")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "online.jsonl"), "w", encoding="utf-8") as handle:
            for i in range(n):
                handle.write(
                    json.dumps(
                        {
                            "t_env": float(i * 9_000),
                            "evaluation/mean_episode_return": base,
                        }
                    )
                    + "\n"
                )
    folder = os.path.join(root, "prior_matched", "seed1")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "online.jsonl"), "w", encoding="utf-8") as handle:
        for i in range(n):
            handle.write(
                json.dumps(
                    {
                        "t_env": float(i * 9_000),
                        "evaluation/mean_episode_return": 10.0 + series[i],
                    }
                )
                + "\n"
            )


def check_block_bootstrap() -> None:
    """The autocorrelation-aware interval must widen the CI, and the amendment
    that uses it may only withdraw a conclusion.

    The registered rule was written with a point bootstrap over the evaluation
    points.  Those points are autocorrelated (measured lag-1 +0.16..+0.47 on
    the finished matrix), so the point interval is too narrow.  The
    moving-block bootstrap is the autocorrelation-aware version of the same
    reading, and the amendment that compares the two may only *withdraw* the
    registered conclusion -- it must never turn it into the other branch.
    """
    print("\n-- moving-block bootstrap (withdraw-only amendment) --")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from analyze_tcpa_attrib import (  # noqa: E402
        _bootstrap,
        _bootstrap_block,
        block_veto,
        classify,
        lag1_autocorr,
    )
    from finalize_tcpa_attrib import build_index_text, completeness  # noqa: E402

    # (a) i.i.d. noise: the two intervals have about the same width.
    rng = np.random.default_rng(20261212)
    iid = rng.normal(0.0, 1.0, size=31)
    point = _bootstrap(iid)
    block = _bootstrap_block(iid)
    point_width = point["hi"] - point["lo"]
    block_width = block["hi"] - block["lo"]
    check(
        "on i.i.d. noise the block interval is about as wide as the point one",
        0.7 <= block_width / point_width <= 1.4,
        f"ratio={block_width / point_width:.2f}",
    )

    # (b) a positively autocorrelated series: the block interval is wider.
    trend = 3.0 * (0.9 ** np.arange(31))
    point = _bootstrap(trend)
    block = _bootstrap_block(trend)
    point_width = point["hi"] - point["lo"]
    block_width = block["hi"] - block["lo"]
    check(
        "on an autocorrelated series the block interval is wider",
        block_width > point_width,
        f"point={point_width:.3f} block={block_width:.3f}",
    )
    check(
        "the autocorrelation of the series is detected",
        (lag1_autocorr(trend) or 0.0) > 0.5,
        f"lag1={lag1_autocorr(trend)}",
    )

    # (c) shift the series so the two readings disagree: the point interval
    #     excludes zero, the block interval contains it.
    shift = -0.5 * (point["lo"] + block["lo"])
    shifted = trend + shift
    point = _bootstrap(shifted)
    block = _bootstrap_block(shifted)
    check(
        "the point interval excludes zero",
        classify(point) == "显著为正",
        f"point=[{point['lo']:+.3f}, {point['hi']:+.3f}]",
    )
    check(
        "the block interval contains zero",
        classify(block) == "含 0",
        f"block=[{block['lo']:+.3f}, {block['hi']:+.3f}]",
    )

    # (d) the veto rule itself.
    vetoed, reasons = block_veto({"P1": "显著为正"}, {"P1": "含 0"})
    check("a disagreement vetoes the reading", vetoed and bool(reasons))
    vetoed, reasons = block_veto({"P1": "含 0"}, {"P1": "含 0"})
    check("an agreement does not veto", not vetoed and not reasons)

    # (e) the veto may withdraw a conclusion, never grant one.
    granted = {
        "verdict": {
            "conclusion": "**支持 H1**（收益由通用早期约束解释）。",
            "P1": {"mean": 0.1, "lo": -0.5, "hi": 0.7, "n": 31, "class": "含 0"},
            "P2": {"mean": 0.1, "lo": -0.5, "hi": 0.7, "n": 31, "class": "含 0"},
            "block_bootstrap": {
                "P1": {"mean": -0.4, "lo": -0.9, "hi": 0.1, "n": 31, "class": "显著为负"},
                "P2": {"mean": 0.1, "lo": -0.5, "hi": 0.7, "n": 31, "class": "含 0"},
                "vetoed": True,
                "reasons": ["P1：登记用点自助判「含 0」，块自助判「显著为负」"],
            },
        },
        "arms": {},
    }
    comp = completeness(granted, 1_000_000)
    index_text = build_index_text(granted, comp)
    check(
        "a veto withdraws a registered H1 instead of keeping it",
        index_text.startswith("**结论：证据不足**") and "支持 H1" not in index_text,
        index_text[:80],
    )

    # (f) end to end: the analyzer must veto a registered H2 whose block
    #     interval disagrees, and the index must say so.
    import subprocess

    analyzer = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "analyze_tcpa_attrib.py"
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "matrix")
        _write_series_matrix(root, list(shifted))
        out_json = os.path.join(tmp, "summary.json")
        out_md = os.path.join(tmp, "results.md")
        result = subprocess.run(
            [
                sys.executable, analyzer, "--root", root, "--seed", "1",
                "--json", out_json, "--markdown", out_md,
            ],
            capture_output=True,
            text=True,
        )
        check("the analyzer runs on the autocorrelated matrix", result.returncode == 0, result.stderr[-300:])
        if result.returncode == 0:
            with open(out_json, encoding="utf-8") as handle:
                summary = json.load(handle)
            verdict = summary["verdict"]
            block_info = verdict["block_bootstrap"]
            check(
                "the registered reading is still positive",
                verdict["P1"]["class"] == "显著为正" and verdict["P2"]["class"] == "显著为正",
                f"P1={verdict['P1']['class']} P2={verdict['P2']['class']}",
            )
            check(
                "the block reading disagrees",
                block_info["P1"]["class"] == "含 0" and block_info["P2"]["class"] == "含 0",
                f"P1={block_info['P1']['class']} P2={block_info['P2']['class']}",
            )
            check("the disagreement vetoes the conclusion", block_info["vetoed"])
            check(
                "the veto is stated as a withdrawal",
                "撤回" in str(verdict.get("block_qualifier", "")),
            )
            check(
                "the registered conclusion string is preserved",
                "支持 H2" in verdict["conclusion"],
                verdict["conclusion"][:60],
            )
            comp = completeness(summary, 1_000_000)
            index_text = build_index_text(summary, comp)
            check(
                "the index reports the withdrawal instead of H2",
                index_text.startswith("**结论：证据不足**") and "支持 H2" not in index_text,
                index_text[:80],
            )
            with open(out_md, encoding="utf-8") as handle:
                markdown = handle.read()
            check(
                "the readout prints the autocorrelation section",
                "5.3 自相关稳健性" in markdown,
            )
            check(
                "the readout reports the effective sample size",
                "n_eff" in markdown,
            )


def _write_matrix_off_grid(root: str, restarted_arm: str = "prior_broken") -> None:
    """Write a matrix in which one arm does not follow the common 10k grid.

    The arm's log holds two runs; the second one evaluates on a 3k-step grid
    instead of the common 10k one.  At the same evaluation index that arm is
    then tens of thousands of steps away from the others, which is exactly
    the situation the pairing check has to catch.
    """
    n = 12
    for arm in (
        "none_matched",
        "prior_matched",
        "none_broken",
        "anchor_matched",
        "anchor_broken",
        "placebo_matched",
        "placebo_broken",
        "fixed_gate_matched",
        "fixed_gate_broken",
    ):
        folder = os.path.join(root, arm, "seed1")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "online.jsonl"), "w", encoding="utf-8") as handle:
            for i in range(n):
                handle.write(
                    json.dumps(
                        {
                            "t_env": float(i * 9_000),
                            "evaluation/mean_episode_return": 10.0 + (0.2 if arm.startswith("prior") else 0.0),
                        }
                    )
                    + "\n"
                )
    folder = os.path.join(root, restarted_arm, "seed1")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "online.jsonl"), "w", encoding="utf-8") as handle:
        # first run: reached 99k steps and then stopped
        for i in range(n):
            handle.write(
                json.dumps(
                    {"t_env": float(i * 9_000), "evaluation/mean_episode_return": 10.1}
                )
                + "\n"
            )
        # second run: same arm, but evaluated on a different grid
        for i in range(n):
            handle.write(
                json.dumps(
                    {"t_env": float(i * 3_000), "evaluation/mean_episode_return": 10.2}
                )
                + "\n"
            )


def _write_matrix_restarted_on_grid(root: str, restarted_arm: str = "prior_broken") -> None:
    """Write a matrix in which one arm crashed and restarted on the same grid.

    This is the realistic case: the arm is re-run from the same offline
    initialisation and evaluates every 10k steps as before, so its n-th
    evaluation still means "n*10k steps of its own run".  The index pairing
    stays valid and the check must not fire.
    """
    n = 12
    for arm in (
        "none_matched",
        "prior_matched",
        "none_broken",
        "anchor_matched",
        "anchor_broken",
        "placebo_matched",
        "placebo_broken",
        "fixed_gate_matched",
        "fixed_gate_broken",
    ):
        folder = os.path.join(root, arm, "seed1")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "online.jsonl"), "w", encoding="utf-8") as handle:
            for i in range(n):
                handle.write(
                    json.dumps(
                        {
                            "t_env": float(i * 9_000),
                            "evaluation/mean_episode_return": 10.0 + (0.2 if arm.startswith("prior") else 0.0),
                        }
                    )
                    + "\n"
                )
    folder = os.path.join(root, restarted_arm, "seed1")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "online.jsonl"), "w", encoding="utf-8") as handle:
        for i in range(n):
            handle.write(
                json.dumps(
                    {"t_env": float(i * 9_000), "evaluation/mean_episode_return": 10.1}
                )
                + "\n"
            )
        # the same grid after the restart
        for i in range(n):
            handle.write(
                json.dumps(
                    {"t_env": float(i * 9_000), "evaluation/mean_episode_return": 10.2}
                )
                + "\n"
            )


def check_pairing_alignment() -> None:
    """The index pairing has to mean "the same training step" -- and that has
    to be checked, not assumed.

    Every arm evaluates on the same ~10k-step grid, so the n-th evaluation
    of an arm is at roughly ``10k * n`` steps of its own run, whichever
    wall-clock time it started at.  The check measures the spread of
    ``t_env`` at the same index; a large spread means the grids diverged
    (an arm restarted with a different evaluation cadence, two runs
    interleaved in one log) and the reading has to be withdrawn.
    """
    print("\n-- pairing alignment --")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from analyze_tcpa_attrib import (  # noqa: E402
        MISALIGN_LIMIT_STEPS,
        eval_cadence,
        eval_curve,
        pairing_spread,
    )
    from finalize_tcpa_attrib import build_index_text, completeness  # noqa: E402

    def curve(points):
        return eval_curve(
            [{"t_env": t, "evaluation/mean_episode_return": v} for t, v in points]
        )

    aligned_a = curve([(0.0, 10.0), (100_000.0, 11.0), (200_000.0, 12.0)])
    aligned_b = curve([(0.0, 10.0), (100_100.0, 11.0), (200_050.0, 12.0)])
    stat = pairing_spread([aligned_a, aligned_b], 0, 300_000)
    check(
        "a small drift between the arms is not a misalignment",
        stat["max_spread"] is not None
        and float(stat["max_spread"]) < MISALIGN_LIMIT_STEPS,
        f"max_spread={stat['max_spread']}",
    )

    off_grid = curve([(0.0, 10.0), (1_000.0, 11.0), (2_000.0, 12.0)])
    stat = pairing_spread([aligned_a, off_grid], 0, 300_000)
    check(
        "an arm on a different evaluation grid is flagged",
        stat["max_spread"] is not None
        and float(stat["max_spread"]) > MISALIGN_LIMIT_STEPS,
        f"max_spread={stat['max_spread']}",
    )

    # The cadence is what makes the index pairing meaningful.
    grid = curve([(0.0, 10.0)] + [(i * 10_000.0, 10.0 + i) for i in range(1, 12)])
    cad = eval_cadence(grid)
    check(
        "the evaluation cadence is measured from the curve",
        cad["mean_step"] is not None and 9_000 < float(cad["mean_step"]) < 11_000,
        f"mean_step={cad['mean_step']}",
    )

    import subprocess

    analyzer = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "analyze_tcpa_attrib.py"
    )

    def _run(root: str, tmp: str) -> Dict:
        out_json = os.path.join(tmp, "summary.json")
        result = subprocess.run(
            [
                sys.executable, analyzer, "--root", root, "--seed", "1",
                "--json", out_json, "--markdown", os.path.join(tmp, "results.md"),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return {"_error": result.stderr[-300:]}
        with open(out_json, encoding="utf-8") as handle:
            return json.load(handle)

    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "off_grid")
        _write_matrix_off_grid(root)
        summary = _run(root, tmp)
        check("the analyzer runs on the off-grid matrix", "_error" not in summary)
        if "_error" not in summary:
            verdict = summary["verdict"]
            align = verdict["pairing_alignment"]
            check(
                "the pairing check fails when the grids diverge",
                not align["ok"] and "P1" in align["misaligned"],
                f"misaligned={align['misaligned']}",
            )
            check(
                "the failure is stated as a withdrawal",
                "撤回" in str(verdict.get("align_qualifier", "")),
            )
            comp = completeness(summary, 1_000_000)
            index_text = build_index_text(summary, comp)
            check(
                "the index reports the broken pairing",
                index_text.startswith("**结论：证据不足**") and "配对失效" in index_text,
                index_text[:100],
            )
            with open(os.path.join(tmp, "results.md"), encoding="utf-8") as handle:
                markdown = handle.read()
            check(
                "the readout prints the alignment section",
                "5.4 评估节奏与配对一致性检查" in markdown,
            )

    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "restarted")
        _write_matrix_restarted_on_grid(root)
        summary = _run(root, tmp)
        check("the analyzer runs on the restarted matrix", "_error" not in summary)
        if "_error" not in summary:
            align = summary["verdict"]["pairing_alignment"]
            check(
                "a restart on the same grid does not break the pairing",
                align["ok"],
                f"misaligned={align['misaligned']}",
            )


def check_detectability() -> None:
    """"The CI contains zero" only means something next to the detectable size.

    A null reading has to be paired with the effect the experiment could have
    seen, otherwise it cannot be told apart from a result that was never in
    reach.  The number is derived from the CI the analyzer already computes,
    so it cannot drift from the readout.
    """
    print("\n-- detectable effect --")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from analyze_tcpa_attrib import minimum_detectable_effect  # noqa: E402

    stat = {"mean": 0.5, "lo": -0.5, "hi": 1.5, "n": 31}
    mde = minimum_detectable_effect(stat)
    se = (1.5 - (-0.5)) / 3.92
    check(
        "the detectable effect follows from the interval width",
        mde is not None and abs(float(mde) - 2.80 * se) < 1e-9,
        f"mde={mde} expected={2.80 * se}",
    )
    check(
        "no interval -> no detectable effect",
        minimum_detectable_effect({"mean": 1.0, "lo": None, "hi": None}) is None,
    )

    import subprocess

    analyzer = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "analyze_tcpa_attrib.py"
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "matrix")
        _write_synthetic_matrix(
            root,
            {
                "none_matched": 10.0,
                "prior_matched": 12.0,
                "none_broken": 10.0,
                "prior_broken": 10.2,
                "anchor_matched": 12.0,
                "anchor_broken": 10.0,
                "placebo_matched": 12.0,
                "placebo_broken": 10.0,
                "fixed_gate_matched": 11.5,
                "fixed_gate_broken": 10.1,
            },
        )
        out_json = os.path.join(tmp, "summary.json")
        out_md = os.path.join(tmp, "results.md")
        result = subprocess.run(
            [
                sys.executable, analyzer, "--root", root, "--seed", "1",
                "--json", out_json, "--markdown", out_md,
            ],
            capture_output=True,
            text=True,
        )
        check("the analyzer runs on the synthetic matrix", result.returncode == 0, result.stderr[-300:])
        if result.returncode == 0:
            with open(out_json, encoding="utf-8") as handle:
                summary = json.load(handle)
            detect = summary.get("detectability") or {}
            check(
                "the summary carries the detectable effect",
                bool(detect) and all(
                    entry.get("mde_point") is not None for entry in detect.values()
                ),
                f"entries={len(detect)}",
            )
            with open(out_md, encoding="utf-8") as handle:
                markdown = handle.read()
            check(
                "the readout prints the detectability section",
                "5.5 可检测效应量" in markdown,
            )


def _write_matrix_with_inert_constraint(
    root: str, inert_arm: str = "prior_matched"
) -> None:
    """Write a matrix in which one hinge arm's constraint never fires.

    The rows carry the mechanism diagnostics, so the analyzer can see that
    the decisive arm's ``tcpa/prior_loss`` stays at exactly zero: the hinge
    never triggered, and any read-out built on it says nothing about the
    constraint.
    """
    profiles = {
        "none_matched": 10.0,
        "prior_matched": 12.0,
        "none_broken": 10.0,
        "prior_broken": 10.2,
        "anchor_matched": 12.0,
        "anchor_broken": 10.0,
        "placebo_matched": 12.0,
        "placebo_broken": 10.0,
        "fixed_gate_matched": 11.5,
        "fixed_gate_broken": 10.1,
    }
    for arm, base in profiles.items():
        folder = os.path.join(root, arm, "seed1")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "online.jsonl"), "w", encoding="utf-8") as handle:
            for i in range(12):
                row = {
                    "t_env": float(i * 9_000),
                    "evaluation/mean_episode_return": base + 0.01 * i,
                    "tcpa/prior_loss_interval_mean": (
                        0.0 if arm == inert_arm else 0.004
                    ),
                    "tcpa/hinge_mean_interval_mean": 0.02,
                }
                handle.write(json.dumps(row) + "\n")


def check_constraint_activity() -> None:
    """A decisive arm whose constraint never fired cannot be read.

    If the hinge loss stays at exactly zero, the constraint never engaged
    and the read-out says nothing about it.  The check has to catch that and
    withdraw the conclusion -- again, only ever withdrawing.
    """
    print("\n-- constraint activity --")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from finalize_tcpa_attrib import build_index_text, completeness  # noqa: E402

    import subprocess

    analyzer = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "analyze_tcpa_attrib.py"
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "matrix")
        _write_matrix_with_inert_constraint(root)
        out_json = os.path.join(tmp, "summary.json")
        out_md = os.path.join(tmp, "results.md")
        result = subprocess.run(
            [
                sys.executable, analyzer, "--root", root, "--seed", "1",
                "--json", out_json, "--markdown", out_md,
            ],
            capture_output=True,
            text=True,
        )
        check("the analyzer runs on the inert-constraint matrix", result.returncode == 0, result.stderr[-300:])
        if result.returncode == 0:
            with open(out_json, encoding="utf-8") as handle:
                summary = json.load(handle)
            verdict = summary["verdict"]
            activity = verdict["constraint_activity"]
            check(
                "an inert decisive arm is detected",
                not activity["ok"] and "prior_matched" in activity["inert_decisive_arms"],
                f"inert={activity['inert_decisive_arms']}",
            )
            check(
                "the inactivity is stated as a withdrawal",
                "撤回" in str(verdict.get("activity_qualifier", "")),
            )
            comp = completeness(summary, 1_000_000)
            index_text = build_index_text(summary, comp)
            check(
                "the index reports the inactive constraint",
                index_text.startswith("**结论：证据不足**") and "约束未生效" in index_text,
                index_text[:100],
            )
            with open(out_md, encoding="utf-8") as handle:
                markdown = handle.read()
            check(
                "the readout prints the activity section",
                "4.2 约束活性检查" in markdown,
            )


def check_missing_arms() -> None:
    """An arm that never made it must be visible as missing.

    The launcher gives up on an arm after all its retries fail (a shared card
    can OOM at any moment) and never revisits it.  Without the watcher's
    missing-arm check the queue would stop with a hole in it and the decisive
    arm would simply never run.
    """
    print("\n-- missing arms --")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from watch_tcpa_attrib import missing_arms  # noqa: E402

    import subprocess

    class _Args:
        root = ""
        seed = 1
        t_max = 1_000_000

    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "matrix")
        args = _Args()
        args.root = root

        # a finished arm: its log reached the budget
        finished = os.path.join(root, "none_matched", "seed1")
        os.makedirs(finished, exist_ok=True)
        with open(os.path.join(finished, "online.jsonl"), "w", encoding="utf-8") as handle:
            for t in (0.0, 500_000.0, 990_000.0):
                handle.write(
                    json.dumps(
                        {"t_env": t, "evaluation/mean_episode_return": 10.0}
                    )
                    + "\n"
                )
        progress = {
            "none_matched": {"n_eval": 3, "last_t_env": 990_000.0},
            "prior_broken": {"n_eval": 0, "last_t_env": 0.0},
        }
        missing = missing_arms(args, progress)
        check(
            "a finished arm is not missing",
            "none_matched" not in missing,
            f"missing={missing}",
        )
        check(
            "an arm that never ran is missing",
            "prior_broken" in missing,
            f"missing={missing}",
        )

        # a running arm: the process carries the arm's checkpoint directory
        running = os.path.join(root, "prior_broken", "seed1")
        os.makedirs(running, exist_ok=True)
        fake = subprocess.Popen(
            [
                "bash",
                "-c",
                f'exec -a "train.py --config configs/tcpa_attrib --checkpoint_dir={running}" sleep 30',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            import time

            time.sleep(1.0)
            missing = missing_arms(args, progress)
            check(
                "a running arm is not missing",
                "prior_broken" not in missing,
                f"missing={missing}",
            )
        finally:
            fake.terminate()
            fake.wait(timeout=10)


def check_full_pipeline_dry_run() -> None:
    """Dry-run the path that will fire automatically when the matrix ends.

    When the last arm reaches its budget the watcher calls the finalizer,
    which re-runs the analyzer, writes the final report and patches the
    experiment's row in the report index.  That whole chain has to work on a
    *complete* matrix -- with every window populated -- because it is the
    deliverable the user will open.  It is dry-run here with synthetic data
    so that a problem shows up now rather than at the end of a 31-hour run.
    """
    print("\n-- full pipeline dry run --")
    import subprocess

    analyzer = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "analyze_tcpa_attrib.py"
    )
    finalizer = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "finalize_tcpa_attrib.py"
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "matrix")
        bases = {
            "none_matched": 10.0,
            "none_broken": 10.0,
            "prior_matched": 10.4,
            "prior_broken": 10.1,
            "anchor_matched": 10.3,
            "anchor_broken": 10.0,
            "placebo_matched": 10.1,
            "placebo_broken": 10.0,
            "fixed_gate_matched": 10.6,
            "fixed_gate_broken": 10.2,
        }
        for arm, base in bases.items():
            folder = os.path.join(root, arm, "seed1")
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "online.jsonl"), "w", encoding="utf-8") as handle:
                for i in range(101):  # 0 .. 1,000,000 steps, every 10k
                    value = base + 0.01 * i + (0.3 if i > 70 else 0.0)
                    row = {
                        "t_env": float(i * 10_000),
                        "evaluation/mean_episode_return": value,
                        "tcpa/prior_loss_interval_mean": (
                            0.0 if arm.startswith("none_") else 0.004
                        ),
                        "tcpa/hinge_mean_interval_mean": (
                            0.0 if arm.startswith("none_") else 0.02
                        ),
                        "tcpa/gate_mean_interval_mean": 0.72,
                        "tcpa/regret_mean_interval_mean": 0.08,
                    }
                    handle.write(json.dumps(row) + "\n")

        out_json = os.path.join(tmp, "summary.json")
        out_md = os.path.join(tmp, "results.md")
        result = subprocess.run(
            [
                sys.executable, analyzer, "--root", root, "--seed", "1",
                "--json", out_json, "--markdown", out_md,
            ],
            capture_output=True,
            text=True,
        )
        check("the analyzer runs on a complete matrix", result.returncode == 0, result.stderr[-300:])
        if result.returncode != 0:
            return
        with open(out_json, encoding="utf-8") as handle:
            summary = json.load(handle)
        check(
            "every arm of the complete matrix is in the summary",
            len(summary.get("arms") or {}) == 10,
            f"arms={len(summary.get('arms') or {})}",
        )
        windows = (summary["arms"]["none_matched"]["windows"] or {})
        check(
            "all three windows are populated",
            all(windows.get(w) is not None for w in ("early", "mid", "late")),
            f"windows={windows}",
        )

        final_md = os.path.join(tmp, "final.md")
        result = subprocess.run(
            [
                sys.executable, finalizer,
                "--root", root, "--seed", "1", "--t-max", "1000000",
                "--json", out_json, "--results-md", out_md, "--out", final_md,
                "--no-readme",
            ],
            capture_output=True,
            text=True,
        )
        check("the finalizer runs on a complete matrix", result.returncode == 0, result.stderr[-300:])
        if result.returncode != 0:
            return
        with open(final_md, encoding="utf-8") as handle:
            report = handle.read()
        check(
            "a complete matrix is not flagged as unfinished",
            "本矩阵未全部跑完" not in report,
        )
        check("the report states the verdict", "## 0. 结论" in report)
        check(
            "the report carries the mechanism table",
            "机制诊断与约束活性" in report,
        )
        check(
            "the report carries the detectable-effect table",
            "可检测效应量" in report,
        )
        check(
            "the report carries the prior from the finished matrix",
            "既有矩阵给出的先验" in report,
        )

        # The index row has to be patched with the verdict, idempotently.
        from finalize_tcpa_attrib import patch_readme  # noqa: E402

        readme = os.path.join(tmp, "README.md")
        with open(readme, "w", encoding="utf-8") as handle:
            handle.write(
                "| **实验** | [tcpa_attrib_5m_vs_6m_design.md](tcpa_attrib_5m_vs_6m_design.md)（设计） | 说明。运行中 |\n"
                "| **别的** | [other.md](other.md) | 别的说明 |\n"
            )
        from finalize_tcpa_attrib import build_index_text, completeness  # noqa: E402

        comp = completeness(summary, 1_000_000)
        index_text = build_index_text(summary, comp)
        check("the index text carries the verdict", "**结论：" in index_text, index_text[:80])
        patch_readme(readme, index_text)
        patch_readme(readme, index_text)
        with open(readme, encoding="utf-8") as handle:
            patched = handle.read()
        check(
            "patching the index row twice is idempotent",
            patched.count("tcpa-attrib-verdict") == 1,
            f"markers={patched.count('tcpa-attrib-verdict')}",
        )
        check(
            "the patched row no longer says 运行中",
            "运行中" not in patched,
        )


def check_arm_state() -> None:
    """The launcher must be able to restart without starting a duplicate arm.

    A duplicated arm would write into the same ``online.jsonl`` and corrupt the
    pairing, so the "is this arm already running / already finished" test is
    part of the experiment's correctness, not just of the launcher.
    """
    print("\n-- arm state (restart safety) --")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from tcpa_arm_state import last_run_points  # noqa: E402

    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "online.jsonl")

        def write(rows):
            with open(log, "w", encoding="utf-8") as handle:
                for t in rows:
                    handle.write(
                        json.dumps(
                            {"t_env": t, "evaluation/mean_episode_return": 10.0}
                        )
                        + "\n"
                    )

        check("a missing log has no points", last_run_points(os.path.join(tmp, "nope.jsonl")) == [])

        write([0.0, 10_000.0, 20_000.0])
        points = last_run_points(log)
        check("a normal log keeps every evaluation point", len(points) == 3)

        # A crashed arm appends a fresh run; only the last run may count, or
        # the launcher would call a restarted arm "finished".
        write([0.0, 10_000.0, 950_000.0, 0.0, 10_000.0])
        points = last_run_points(log)
        check(
            "a crashed arm keeps only its last run",
            len(points) == 2 and points[-1][0] == 10_000.0,
            f"n={len(points)}",
        )

        write([i * 10_000.0 for i in range(100)])
        points = last_run_points(log)
        check(
            "a finished arm is recognised",
            points[-1][0] == 990_000.0,
            f"last_t={points[-1][0]}",
        )


def check_anchor_long_horizon() -> None:
    """The anchor arm must really constrain, not merely exist.

    P2 = ``prior - anchor`` is the H1 control, and the H1 reading says "the
    benefit comes from a content-agnostic early constraint".  If the anchor
    were inert, P2 would read "the specific old action beats a generic
    constraint" for the wrong reason -- the generic constraint would simply
    not have been applied.  So the anchor has to hold the online utilities at
    the offline ones while the unconstrained arm drifts away.

    The comparison uses identical synthetic data and identical seeds, so the
    only difference between the two arms is the constraint term.
    """
    print("\n-- anchor over many steps --")
    batch = make_batch(torch.Generator().manual_seed(7))

    def run(variant: str, steps: int = 40):
        learner = build_learner(variant=variant, seed=5)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "offline.pt")
            write_offline_checkpoint(learner, ckpt)
            learner.load_offline_for_ovm(ckpt)
            learner.prepare_online_start()
        perturb_residual(learner)
        traj = []
        for step in range(steps):
            logs = learner.train_on_batch(
                batch, cql_weight=0.0, use_ovm=False, t_env=step * 100
            )
            traj.append(
                (
                    float(logs["tcpa/dq_absmean"]),
                    float(logs.get("tcpa/anchor_dev_mean", 0.0)),
                    float(logs["loss"]),
                )
            )
        return traj

    free = run("no_igm")
    anchor = run("value_anchor")

    check(
        "the unconstrained arm drifts away from the offline utilities",
        free[-1][0] > free[0][0],
        f"dq {free[0][0]:.4f} -> {free[-1][0]:.4f}",
    )
    check(
        "the anchor holds the online utilities at the offline ones",
        anchor[-1][0] < free[-1][0],
        f"anchor dq={anchor[-1][0]:.4f} vs free dq={free[-1][0]:.4f}",
    )
    check(
        "the anchor stays numerically stable over many steps",
        all(np.isfinite([dq, dev, loss][0]) for dq, dev, loss in anchor)
        and all(np.isfinite(loss) for _dq, _dev, loss in anchor),
    )
    check(
        "the anchor deviation stays bounded",
        0.0 < anchor[-1][1] < 100.0,
        f"dev={anchor[-1][1]:.4f}",
    )


def main() -> None:
    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(7)
    batch = make_batch(rng)

    learner = build_learner()
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "offline.pt")
        write_offline_checkpoint(learner, ckpt)
        learner.load_offline_for_ovm(ckpt)
        learner.prepare_online_start()

        obs_aug = learner._augment_obs(batch["obs"], batch["actions"])
        resets = learner._resets_from_batch(batch)
        q_off = learner._offline_utilities(obs_aug, resets)
        dq = learner._online_utilities(obs_aug, resets, target=False)
        q_on = q_off + dq

        # 1. zero-initialised residual -> identical utilities
        check(
            "residual zero-init keeps q_on == q_off",
            torch.allclose(q_on, q_off, atol=1e-6),
            f"max|dq|={float(dq.abs().max()):.3e}",
        )
        check(
            "frozen base receives no gradient",
            all(not p.requires_grad for p in learner.q_network.base.parameters()),
        )

        # 2. team regret is zero and the gate is one at initialisation
        info = team_regret(q_on, q_off, batch["avail_actions"], batch["state"], learner.target_mixer)
        regret = info["regret"]
        check(
            "team regret is zero at initialisation",
            float(regret.abs().max()) < 1e-6,
            f"max|R|={float(regret.abs().max()):.3e}",
        )
        gate = torch.exp(-regret.clamp(min=0.0) / 1.0)
        check("gate is one at initialisation", torch.allclose(gate, torch.ones_like(gate)))

        # 3. the hinge is zero while the offline action is locally greedy
        loss, stats = igm_ranking_loss(
            q_on[:, :-1],
            info["a_off"][:, :-1],
            batch["avail_actions"][:, :-1],
            gate[:, :-1],
            batch["filled"][:, :-1],
        )
        check(
            "IGM loss is zero when the offline action is greedy",
            float(loss) < 1e-6,
            f"loss={float(loss):.3e}",
        )
        single = batch["avail_actions"][:, :-1, 1].sum(dim=-1) == 1
        check(
            "single-legal-action agents exist in the batch",
            bool(single.any()),
            f"n_single={int(single.sum())}",
        )
        q_max = q_on[:, :-1].masked_fill(
            ~batch["avail_actions"][:, :-1].bool(), -1e9
        ).max(dim=3).values
        hinge = q_max - torch.gather(
            q_on[:, :-1], 3, info["a_off"][:, :-1].unsqueeze(-1)
        ).squeeze(-1)
        check(
            "single-legal-action agents contribute no hinge",
            float(hinge[..., 1].abs().max()) < 1e-6,
            f"max hinge={float(hinge[..., 1].abs().max()):.3e}",
        )

        # 4. monotonicity of the mixer -> regret is non-negative
        agent_qs = torch.randn(B, T - 1, N)
        states = batch["state"][:, :-1]
        base_tot = learner.target_mixer(agent_qs, states)
        bumped = agent_qs.clone()
        bumped[..., 0] += 0.5
        bumped_tot = learner.target_mixer(bumped, states)
        check(
            "mixer is monotone in the agent utilities",
            bool((bumped_tot >= base_tot - 1e-5).all()),
        )

        # 5. a training step runs and only the residual/mixer get gradients
        learner.optimizer.zero_grad(set_to_none=True)
        logs = learner.train_on_batch(batch, cql_weight=0.0, use_ovm=False, t_env=100)
        residual_grad = sum(
            float(p.grad.abs().sum()) for p in learner.q_network.residual.parameters() if p.grad is not None
        )
        mixer_grad = sum(
            float(p.grad.abs().sum()) for p in learner.mixer.parameters() if p.grad is not None
        )
        check("training step produces finite loss", torch.isfinite(torch.tensor(logs["loss"])))
        check("residual receives gradients", residual_grad > 0.0, f"sum|dgrad|={residual_grad:.3e}")
        check("mixer receives gradients", mixer_grad > 0.0, f"sum|dgrad|={mixer_grad:.3e}")
        for key in (
            "tcpa/regret_mean",
            "tcpa/gate_mean",
            "tcpa/tau_g",
            "tcpa/offline_action_agreement",
            "tcpa/igm_loss" if "tcpa/igm_loss" in logs else "tcpa/hinge_mean",
        ):
            check(f"log field {key} present", key in logs)

        # 6. checkpoint round-trip (compare against the post-update utilities)
        q_after = learner._online_utilities(obs_aug, resets, target=False)
        state = {
            "q_network": learner.q_network.state_dict(),
            "mixer": learner.mixer.state_dict(),
            "target_q_network": learner.target_q_network.state_dict(),
            "target_mixer": learner.target_mixer.state_dict(),
        }
        path = os.path.join(tmp, "tcpa.pt")
        torch.save(state, path)
        other = build_learner()
        other.load_offline_for_ovm(ckpt)
        other.prepare_online_start()
        other.q_network.load_state_dict(state["q_network"])
        other.mixer.load_state_dict(state["mixer"])
        other.target_q_network.load_state_dict(state["target_q_network"])
        other.target_mixer.load_state_dict(state["target_mixer"])
        q_other = other._online_utilities(obs_aug, resets, target=False)
        check(
            "checkpoint round-trip reproduces the utilities",
            torch.allclose(q_after, q_other, atol=1e-6),
            f"max diff={float((q_after - q_other).abs().max()):.3e}",
        )

    # 7. the no_residual arm trains the base instead of a residual
    plain = build_learner(variant="no_residual")
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "offline.pt")
        write_offline_checkpoint(plain, ckpt)
        plain.load_offline_for_ovm(ckpt)
        plain.prepare_online_start()
        check(
            "no_residual keeps a frozen reference copy of the offline net",
            plain.offline_q_network is not plain.q_network.base,
        )
        logs = plain.train_on_batch(batch, cql_weight=0.0, use_ovm=False, t_env=100)
        check("no_residual trains", torch.isfinite(torch.tensor(logs["loss"])))

    # 8. the attribution arms
    check_attribution_arms(batch)

    # 9. the analysis the attribution table depends on
    check_analysis_pairing()

    # 10. the pre-registered decision rule
    check_pre_registered_verdict()

    # 11. the reporting pipeline (final report + index row)
    check_reporting_pipeline()

    # 12. the exploratory robustness reading
    check_robustness_reading()

    # 13. the manipulation check on the H1 branch
    check_manipulation_qualifier()

    # 14. the launcher restart safety
    check_arm_state()

    # 15. the autocorrelation-aware interval and its withdraw-only amendment
    check_block_bootstrap()

    # 16. the pairing-alignment check (a restarted arm must be visible)
    check_pairing_alignment()

    # 17. the detectable effect behind a "contains zero" reading
    check_detectability()

    # 18. the constraint-activity check (a decisive arm whose hinge never fired)
    check_constraint_activity()

    # 19. the missing-arm check (an abandoned arm must not be lost silently)
    check_missing_arms()

    # 20. the end-to-end path that fires automatically when the matrix ends
    check_full_pipeline_dry_run()

    # 21. the anchor arm really constrains (the H1 control must not be inert)
    check_anchor_long_horizon()

    print("\nall TCPA self-tests passed")

if __name__ == "__main__":
    main()
