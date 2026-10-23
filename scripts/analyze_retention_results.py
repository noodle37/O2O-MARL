#!/usr/bin/env python3
"""Consolidate the H1/H2 and H4 artifacts into one report-ready summary.

H3 has its own TXT writer (scripts/analyze_mode_diversity.py), so it is only
pointed at here. This script reads:

  logs/retention_insight.json              merged H1/H2 shards
  checkpoints/pexm_5m_vs_6m/online.jsonl   PEX-M training/eval history
  checkpoints/pexm_5m_vs_6m/pexm_selector_seed1.json
  logs/eval640_*.jsonl                     endpoint evaluations

and prints plain-text tables. Nothing is recomputed from raw records; use
scripts/verify_retention_insight.py --merge-dir for that.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

GLOBAL_PAIR_REFERENCE = 0.614


def _num(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _rate(entry: dict | None, digits: int = 3) -> str:
    if not entry or entry.get("concordance") is None:
        n = 0 if not entry else entry.get("n_decided", 0)
        return f"n/a (n={n})"
    text = f"{_num(entry['concordance'], digits)}"
    ci = entry.get("ci95")
    if ci:
        text += f" [{_num(ci[0], digits)}, {_num(ci[1], digits)}]"
    text += f" (n={entry.get('n_decided', entry.get('n_pairs', 0))}, clusters={entry.get('n_clusters', 0)})"
    return text


def _mean(entry: dict | None, digits: int = 3) -> str:
    if not entry or entry.get("mean") is None:
        return "n/a"
    text = _num(entry["mean"], digits)
    ci = entry.get("ci95")
    if ci:
        text += f" [{_num(ci[0], digits)}, {_num(ci[1], digits)}]"
    return text


def _diff(entry: dict | None, digits: int = 3) -> str:
    if not entry or entry.get("mean_diff") is None:
        return "n/a"
    ci = entry.get("ci95")
    text = _num(entry["mean_diff"], digits)
    if ci:
        text += f" [{_num(ci[0], digits)}, {_num(ci[1], digits)}]"
    return text + f" (clusters={entry.get('n_clusters', 0)})"


def section_retention(path: Path) -> list[str]:
    if not path.is_file():
        return [f"[H1/H2] missing {path}"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    s = payload.get("summary", {})
    lines = [
        "[H1/H2] counterfactual deviation + candidate selectability",
        f"  episodes={s.get('n_episodes')} wins={s.get('n_wins')} "
        f"states={s.get('n_states')} deviations={s.get('n_deviations')} "
        f"random_pairs={s.get('n_random_pairs')} global_pairs={s.get('n_global_pairs')} "
        f"candidate_sets={s.get('n_selectability_sets')} candidate_pairs={s.get('n_candidate_pairs')}",
        f"  frozen={payload.get('frozen_checkpoint')} base_replay_gap_max={_num(s.get('base_replay_gap_max'), 4)}",
        "",
        "  H1: deviation ranking vs random joint pairs",
        f"    random pairs        {_rate(s.get('random_pairs'))}",
        f"    global diagnostic   {_rate(s.get('global_diag'))}",
    ]
    for kind, entry in (s.get("deviation_by_kind") or {}).items():
        counts = (s.get("deviation_effect_counts") or {}).get(kind, {})
        lines.append(
            f"    deviation {kind:<8} {_rate(entry)}  "
            f"dev_better/worse/same={counts.get('dev_better')}/{counts.get('dev_worse')}/{counts.get('same')} "
            f"mean_signed={_num(counts.get('mean_signed'))}"
        )
    lines += [
        f"    deviation - random  {_diff(s.get('deviation_minus_random'))}",
        f"    deviation - global  {_diff(s.get('deviation_minus_global_diag'))}",
        f"    negative knowledge  {_rate(s.get('negative_knowledge'))}  (deviation strictly worse)",
        f"    positive knowledge  {_rate(s.get('positive_knowledge'))}  (deviation strictly better)",
        f"    signed effect       {_mean(s.get('deviation_signed_effect'))}",
        f"    abs effect          {_mean(s.get('deviation_abs_effect'))}",
        "",
        "  H1: effect sizes by deviation kind",
    ]
    for kind, entry in (s.get("deviation_abs_effect_by_kind") or {}).items():
        lines.append(f"    {kind:<8} abs_gap={_mean(entry)}")
    for kind, entry in (s.get("deviation_signed_effect_by_kind") or {}).items():
        lines.append(f"    {kind:<8} signed_gap={_mean(entry)}")

    lines += ["", f"  H2: offline-Q top-1 inside a candidate set (global-pair reference {GLOBAL_PAIR_REFERENCE:.3f})"]
    sel = s.get("selectability_by_set") or {}
    for set_name, entry in sel.items():
        lines.append(
            f"    {set_name:<13} sets={entry.get('n_sets')} k={_num(entry.get('mean_k'), 1)} "
            f"top1={_rate(entry.get('top1_accuracy'))} chance={_num(entry.get('top1_chance_mean'))}"
        )
        lines.append(
            f"                  top1-chance={_mean(entry.get('top1_minus_chance'))} "
            f"regret={_mean(entry.get('regret'))} p50={_num(entry.get('regret_p50'))} "
            f"p90={_num(entry.get('regret_p90'))}"
        )
        lines.append(
            f"                  pair_concordance={_rate(entry.get('pair_concordance'))} "
            f"true_best_qrank={_mean(entry.get('true_best_qrank'))} "
            f"q_top1_truerank={_mean(entry.get('q_top1_truerank'))}"
        )
        lines.append(
            f"                  - global={_diff((s.get('selectability_minus_global') or {}).get(set_name))} "
            f"- random={_diff((s.get('selectability_minus_random') or {}).get(set_name))}"
        )
    return lines


def _auc(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2:
        return None
    trapz = getattr(np, "trapezoid", None) or np.trapz
    span = float(x[-1] - x[0])
    if span <= 0:
        return None
    return float(trapz(y, x) / span)


def section_pexm(log_path: Path, selector_path: Path, early_n: int) -> list[str]:
    if not log_path.is_file():
        return [f"[H4] missing {log_path}"]
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda r: float(r.get("t_env", 0.0)))
    if not rows:
        return [f"[H4] empty {log_path}"]

    t = np.asarray([float(r.get("t_env", 0.0)) for r in rows])
    def series(name: str) -> np.ndarray:
        return np.asarray([float(r.get(name, np.nan)) for r in rows])

    on_ret = series("eval_online/mean_episode_return")
    on_win = series("eval_online/win_rate")
    off_ret = series("eval_offline/mean_episode_return")
    off_win = series("eval_offline/win_rate")
    early = slice(0, min(int(early_n), len(rows)))
    off_ref_ret = float(np.nanmean(off_ret))
    off_ref_win = float(np.nanmean(off_win))

    total_choices = series("pexm/interval_offline") + series("pexm/interval_online")
    forced = series("pexm/interval_forced_online")
    forced_share = float(np.nansum(forced) / max(np.nansum(total_choices), 1.0))
    online_share = float(rows[-1].get("pexm/online_share_cum", np.nan))

    def modality(prefix: str) -> str:
        return (
            f"mu={_num(rows[-1].get(f'pexm/{prefix}_mu'))} "
            f"sd={_num(rows[-1].get(f'pexm/{prefix}_sd'))} "
            f"n={_num(rows[-1].get(f'pexm/{prefix}_n'), 0)}"
        )

    lines = [
        "[H4] PEX-M online pilot",
        f"  intervals={len(rows)} t_env={_num(t[-1], 0)} early_window={early_n} eval(s)",
        f"  frozen offline reference (mean over evals): return={_num(off_ref_ret)} win={_num(off_ref_win)}",
        f"  online early: min_return={_num(np.nanmin(on_ret[early]))} "
        f"min_win={_num(np.nanmin(on_win[early]))} early_mean_return={_num(np.nanmean(on_ret[early]))}",
        f"  online full : min_return={_num(np.nanmin(on_ret))} "
        f"final_return={_num(on_ret[-1])} final_win={_num(on_win[-1])}",
        f"  early unlearning gap (offline ref - online early min return)={_num(off_ref_ret - np.nanmin(on_ret[early]))}",
        f"  final gap (offline ref - online final return)={_num(off_ref_ret - on_ret[-1])}",
        f"  time-mean win: online={_num(_auc(t, on_win))} offline={_num(_auc(t, off_win))}",
        f"  time-mean ret: online={_num(_auc(t, on_ret))} offline={_num(_auc(t, off_ret))}",
        f"  selector online_share_cum={_num(online_share)} forced_online_share={_num(forced_share)} "
        f"last_interval_offline/online/forced={_num(rows[-1].get('pexm/interval_offline'), 0)}/"
        f"{_num(rows[-1].get('pexm/interval_online'), 0)}/{_num(rows[-1].get('pexm/interval_forced_online'), 0)}",
        f"  posterior offline {modality('offline')} | online {modality('online')}",
    ]
    if selector_path.is_file():
        payload = json.loads(selector_path.read_text(encoding="utf-8"))
        lines.append(
            f"  selector JSON t_env={_num(payload.get('t_env'), 0)} "
            f"history_rows={len(payload.get('history', []))}"
        )
    return lines


def section_eval640(paths: list[Path]) -> list[str]:
    lines = ["[H4] 640-episode endpoint evaluations"]
    if not paths:
        lines.append("  no eval640_*.jsonl found")
        return lines
    for path in sorted(paths):
        summaries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "chunk" in row:
                continue
            if "episodes" in row and ("win_rate" in row):
                summaries.append(row)
        if not summaries:
            lines.append(f"  {path.name}: no summary row")
            continue
        for row in summaries:
            label = row.get("mode") or row.get("tag") or "?"
            ci = row.get("win_rate_ci95") or [None, None]
            extra = ""
            if row.get("online_share") is not None:
                extra = f" online_share={_num(row['online_share'])}"
            lines.append(
                f"  {path.name:<28} {label:<9} win={_num(row.get('win_rate'))} "
                f"[{_num(ci[0])}, {_num(ci[1])}] ret={_num(row.get('mean_return'))} "
                f"eps={row.get('episodes')}{extra}"
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention", default="logs/retention_insight.json")
    parser.add_argument("--pexm-log", default="checkpoints/pexm_5m_vs_6m/online.jsonl")
    parser.add_argument("--selector", default="checkpoints/pexm_5m_vs_6m/pexm_selector_seed1.json")
    parser.add_argument("--eval-glob", default="logs/eval640_*.jsonl")
    parser.add_argument("--early-evals", type=int, default=10)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    lines = section_retention(Path(args.retention))
    lines += [""] + section_pexm(Path(args.pexm_log), Path(args.selector), args.early_evals)
    lines += [""] + section_eval640([Path(p) for p in sorted(glob.glob(args.eval_glob))])
    lines += ["", f"[H3] see logs/mode_diversity.txt (13 policies, 20 seeds)"]
    text = "\n".join(lines)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"text": text}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
