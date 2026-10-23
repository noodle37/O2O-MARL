#!/usr/bin/env python3
"""Aggregate the Task-B rank-flip probe into a mechanism report.

Reads the JSON written by scripts/probe_rank_flip.py and answers, with
trajectory-clustered uncertainty:

  1. how often the teacher's joint-action preference is inverted by the return
     a continuation policy actually realises, per policy and per pair family;
  2. whether the inverted pairs are the ones that decide the executed action;
  3. what the inversions cost in discounted return, and what the constraint
     would gain where the teacher is right;
  4. where the inversions concentrate (battle phase, alive counts, teacher gap,
     hinge activity);
  5. whether online-observable features can predict decision-relevant mismatch.

Nothing here uses the replay oracle as a feature: Delta G is the label only.

Usage:
  python scripts/probe_rank_flip_report.py --json logs/rank_flip_seed1.json \
      --out documentation/experiment_reports/taskB_rank_flip_seed1.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURES = [
    "abs_q_gap_off",
    "abs_q_gap_live",
    "abs_q_gap_tgt",
    "q_off_a",
    "q_off_b",
    "q_live_a",
    "q_live_b",
    "sign_disagree",
    "hinge_active",
    "hinge_slack",
    "eta",
    "count_a",
    "count_b",
    "allies",
    "enemies",
    "phase",
    "t",
    "n_cand",
]


def _cluster_rate(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """Rate over rows, with the trajectory (env seed) as the cluster unit."""
    if not rows:
        return {"n": 0, "rate": None, "se_binom": None, "se_cluster": None}
    hits = [1.0 if r[label] else 0.0 for r in rows]
    n = len(hits)
    rate = float(np.mean(hits))
    se_binom = math.sqrt(rate * (1 - rate) / n)
    per_seed: Dict[int, List[float]] = {}
    for r, h in zip(rows, hits):
        per_seed.setdefault(int(r["env_seed"]), []).append(h)
    if len(per_seed) < 2:
        se_cluster = None
    else:
        means = np.array([float(np.mean(v)) for v in per_seed.values()])
        se_cluster = float(means.std(ddof=1) / math.sqrt(len(means)))
    return {
        "n": n,
        "n_seeds": len(per_seed),
        "rate": rate,
        "se_binom": se_binom,
        "se_cluster": se_cluster,
    }


def _mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return float(np.mean(vals))


def _cluster_mean(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    if not rows:
        return {"n": 0, "mean": None, "se_cluster": None}
    vals = [float(r[key]) for r in rows]
    per_seed: Dict[int, List[float]] = {}
    for r, v in zip(rows, vals):
        per_seed.setdefault(int(r["env_seed"]), []).append(v)
    if len(per_seed) < 2:
        se = None
    else:
        means = np.array([float(np.mean(v)) for v in per_seed.values()])
        se = float(means.std(ddof=1) / math.sqrt(len(means)))
    return {"n": len(vals), "mean": float(np.mean(vals)), "se_cluster": se}


def _fmt(x: Optional[float], digits: int = 3) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def _rate_row(name: str, rows: List[Dict[str, Any]], label: str) -> str:
    st = _cluster_rate(rows, label)
    if st["n"] == 0:
        return f"| {name} | 0 | - | - | - |"
    ci = ""
    if st["se_cluster"] is not None:
        lo, hi = st["rate"] - 1.96 * st["se_cluster"], st["rate"] + 1.96 * st["se_cluster"]
        ci = f"[{lo:.3f}, {hi:.3f}]"
    else:
        lo, hi = st["rate"] - 1.96 * st["se_binom"], st["rate"] + 1.96 * st["se_binom"]
        ci = f"[{lo:.3f}, {hi:.3f}]"
    return (
        f"| {name} | {st['n']} | {st['rate']:.3f} | "
        f"{_fmt(st['se_cluster'] if st['se_cluster'] is not None else st['se_binom'])} | {ci} |"
    )


def _bucket(rows: List[Dict[str, Any]], key: str, edges: List[float],
            labels: List[str]) -> List[str]:
    out = []
    for lo, hi, name in zip([-1e9] + edges, edges + [1e9], labels):
        sub = [r for r in rows if lo < float(r[key]) <= hi]
        out.append(_rate_row(name, sub, "flip"))
    return out


def _predict(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Online-observable features only; the oracle is the label, never a feature."""
    p1 = [r for r in rows if r["family"] == "P1"]
    if len(p1) < 40:
        return {"n": len(p1), "skipped": "fewer than 40 P1 records"}
    labels = np.array([1.0 if r["outcome"] == "flip" else 0.0 for r in p1])
    if labels.sum() < 10 or labels.sum() > len(labels) - 10:
        return {"n": len(p1), "skipped": "degenerate label balance"}
    X = np.array([[float(r.get(f, 0.0)) for f in FEATURES] for r in p1], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    groups = np.array([int(r["env_seed"]) for r in p1])

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_splits = min(4, len(set(groups)))
    if n_splits < 2:
        return {"n": len(p1), "skipped": "fewer than 2 trajectory clusters"}
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, labels, groups))
    out: Dict[str, Any] = {
        "n": len(p1),
        "n_positive": int(labels.sum()),
        "pos_rate": float(labels.mean()),
        "n_trajectories": len(set(groups)),
        "n_splits": n_splits,
        "features": FEATURES,
        "models": {},
    }
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "gbdt": GradientBoostingClassifier(random_state=0),
    }
    for name, model in models.items():
        oof = np.full(len(labels), np.nan)
        for train, test in splits:
            if len(set(labels[train])) < 2:
                continue
            model.fit(X[train], labels[train])
            oof[test] = model.predict_proba(X[test])[:, 1]
        ok = ~np.isnan(oof)
        if ok.sum() < 10 or len(set(labels[ok])) < 2:
            out["models"][name] = {"skipped": "fold coverage too small"}
            continue
        auc = float(roc_auc_score(labels[ok], oof[ok]))
        out["models"][name] = {"auc": auc, "n_scored": int(ok.sum())}
        if name == "logistic":
            weights = model.named_steps["logisticregression"].coef_[0]
            order = np.argsort(-np.abs(weights))
            out["models"][name]["top_features"] = [
                (FEATURES[i], float(weights[i])) for i in order[:8]
            ]
    return out


def write_report(payload: Dict[str, Any], out_path: str) -> None:
    rows: List[Dict[str, Any]] = payload["rows"]
    for r in rows:
        r["flip"] = r["outcome"] == "flip"
    policies = ["ref"] + list(payload["branches"])
    policies = [p for p in policies if any(r["policy"] == p for r in rows)]

    L: List[str] = []
    L.append("# Task B: 联合动作偏好翻转的决策意义（5m_vs_6m, seed 1）")
    L.append("")
    L.append("## 零、设置")
    L.append("")
    L.append(
        f"- fork 快照（冻结 teacher + fork 时刻在线策略）：`{payload['fork_snapshot']}`"
    )
    L.append(f"- continuation policies：`{', '.join(policies)}`（fork 时刻 eta："
             f"{payload['eta_at_fork']}）")
    L.append(f"- 状态总体：fork 时刻在线策略在 {len(payload['env_seeds'])} 个环境种子上的贪心轨迹，"
             f"共 {payload['n_states']} 个决策状态；argmax 一致（teacher 与 live 同选）状态 "
             f"{payload['n_argmax_agree']} 个")
    L.append(f"- 候选对总体：UOT hinge 自己的候选集（K=8, Hamming<=2, margin 过滤），"
             f"步长 stride={payload['stride']}；P2 抽样 stride={payload['p2_stride']}, "
             f"每状态 {payload['p2_pairs']} 对")
    L.append(f"- Delta G：分支回放（同种子重启、前缀逐位复现、强制该联合动作后按该策略贪心），"
             f"gamma={payload['gamma']}；可用记录 {len(rows)} 条")
    L.append("- 聚类与不确定度：以轨迹（环境种子）为聚类单元")
    L.append("- 红线：分支回放得到的真实 Delta G 只作标签，不作为可部署 gate 的特征")
    L.append("")

    L.append("## 一、偏好翻转率")
    L.append("")
    L.append("| 策略 | 族 | n | 翻转率 | 聚类 SE | 95% CI |")
    L.append("|---|---|---|---|---|---|")
    for p in policies:
        for fam in ("P1", "P2"):
            sub = [r for r in rows if r["policy"] == p and r["family"] == fam]
            if not sub:
                continue
            L.append(_rate_row(f"{p} / {fam}", sub, "flip"))
    L.append("")
    L.append("P1 = (teacher argmax, live argmax)，即决定执行哪个联合动作、且 hinge 正在推的那一对；"
             "P2 = 候选集内其余 hinge 对的抽样，作为一般对总体的对照。")
    L.append("")

    L.append("## 二、决策相关性")
    L.append("")
    L.append("| 判据 | 族 | n | 比例 | 聚类 SE | 95% CI |")
    L.append("|---|---|---|---|---|---|")
    for p in policies:
        for fam in ("P1", "P2"):
            sub = [r for r in rows if r["policy"] == p and r["family"] == fam]
            if not sub:
                continue
            L.append(_rate_row(
                f"{p} / {fam} 涉及执行动作", sub, "b_is_exec",
            ))
    L.append("")
    L.append("| 判据 | 族 | n | 比例 | 聚类 SE | 95% CI |")
    L.append("|---|---|---|---|---|---|")
    for p in policies:
        for fam in ("P1", "P2"):
            sub = [r for r in rows if r["policy"] == p and r["family"] == fam]
            if not sub:
                continue
            L.append(_rate_row(
                f"{p} / {fam} 涉及 teacher argmax", sub, "a_is_teacher_argmax"
            ))
    L.append("")

    L.append("## 三、回报代价")
    L.append("")
    L.append("对 P1 对，hinge 约束把在线选择（live argmax，记为 b）推向 teacher 的偏好（a），"
             "因此该对实现的回报变化就是 dg = g_a - g_b。")
    L.append("")
    L.append("| 策略 | n | E[dg] | 聚类 SE | 翻转时 E[dg] | 一致时 E[dg] | 期望损失/对 |")
    L.append("|---|---|---|---|---|---|---|")
    for p in policies:
        sub = [r for r in rows if r["policy"] == p and r["family"] == "P1"]
        if not sub:
            continue
        net = _cluster_mean(sub, "dg")
        flips = [r for r in sub if r["flip"]]
        agrees = [r for r in sub if r["outcome"] == "agree"]
        e_flip = _mean([r["dg"] for r in flips])
        e_agree = _mean([r["dg"] for r in agrees])
        p_flip = (len(flips) / len(sub)) if sub else 0.0
        loss = (-e_flip * p_flip) if (e_flip is not None) else None
        L.append(
            f"| {p} | {len(sub)} | {_fmt(net['mean'])} | {_fmt(net['se_cluster'])} | "
            f"{_fmt(e_flip)} | {_fmt(e_agree)} | {_fmt(loss)} |"
        )
    L.append("")
    L.append("| 策略 | 翻转对 | 翻转 \\|dg\\| 中位数 | 翻转 \\|dg\\| 均值 |")
    L.append("|---|---|---|---|")
    for p in policies:
        flips = [
            r for r in rows
            if r["policy"] == p and r["family"] == "P1" and r["flip"]
        ]
        if not flips:
            continue
        mags = [abs(r["dg"]) for r in flips]
        L.append(
            f"| {p} | {len(flips)} | {float(np.median(mags)):.3f} | {_fmt(_mean(mags))} |"
        )
    L.append("")

    L.append("## 四、条件化：翻转集中在哪")
    L.append("")
    p1 = [r for r in rows if r["family"] == "P1"]
    if p1:
        L.append("### 战斗阶段（phase = t / 回合长度）")
        L.append("")
        L.append("| 阶段 | n | 翻转率 | 聚类 SE | 95% CI |")
        L.append("|---|---|---|---|---|")
        L.extend(_bucket(p1, "phase", [0.25, 0.5, 0.75], ["前段", "中前", "中后", "尾段"]))
        L.append("")
        L.append("### 存活人数")
        L.append("")
        L.append("| 己方存活 | n | 翻转率 | 聚类 SE | 95% CI |")
        L.append("|---|---|---|---|---|")
        L.extend(_bucket(p1, "allies", [2, 3, 4], ["<=2", "3", "4", "5"]))
        L.append("")
        L.append("### teacher 差距 |dq_off|")
        L.append("")
        L.append("| \\|dq_off\\| | n | 翻转率 | 聚类 SE | 95% CI |")
        L.append("|---|---|---|---|---|")
        q = [float(r["abs_q_gap_off"]) for r in p1]
        edges = [float(np.quantile(q, 1 / 3)), float(np.quantile(q, 2 / 3))]
        L.extend(_bucket(p1, "abs_q_gap_off", edges, ["小", "中", "大"]))
        L.append("")
        L.append("### hinge 活跃（该对当前被排序损失推着）")
        L.append("")
        L.append("| hinge | n | 翻转率 | 聚类 SE | 95% CI |")
        L.append("|---|---|---|---|---|")
        for flag, name in ((True, "active"), (False, "inactive")):
            L.append(_rate_row(name, [r for r in p1 if r["hinge_active"] is flag], "flip"))
        L.append("")

    L.append("## 五、可预测性（只用在线可观测特征）")
    L.append("")
    pred = _predict(rows)
    if "skipped" in pred:
        L.append(f"样本不足，未做预测：{pred['skipped']}（n={pred.get('n')}）。")
    else:
        L.append(
            f"标签：P1 决策相关翻转（n={pred['n']}，正例 {pred['n_positive']}，"
            f"正例率 {pred['pos_rate']:.3f}），特征 {len(pred['features'])} 维，"
            f"按轨迹分组 {pred['n_splits']} 折，组内不泄漏。"
        )
        L.append("")
        L.append("| 模型 | out-of-fold AUC | 打分条数 |")
        L.append("|---|---|---|")
        for name, res in pred["models"].items():
            if "skipped" in res:
                L.append(f"| {name} | - ({res['skipped']}) | - |")
            else:
                L.append(f"| {name} | {res['auc']:.3f} | {res['n_scored']} |")
        L.append("")
        for name, res in pred["models"].items():
            if "top_features" in res:
                L.append(f"logistic 权重（前 8）："
                         + ", ".join(f"{f}={w:+.3f}" for f, w in res["top_features"]))
                L.append("")

    L.append("## 六、两种读法")
    L.append("")
    flips_all = [r for r in p1 if r["flip"]]
    p_flip_all = _cluster_rate(p1, "flip")
    net_all = _cluster_mean(p1, "dg")
    aucs = [
        res.get("auc")
        for res in (pred.get("models", {}) if isinstance(pred, dict) else {}).values()
        if isinstance(res, dict) and res.get("auc") is not None
    ]
    best_auc = max(aucs) if aucs else None
    L.append("### 支持继续研究 ranking 选择性迁移")
    L.append("")
    if p_flip_all["n"]:
        L.append(
            f"- P1 决策相关对上翻转率 {p_flip_all['rate']:.3f}"
            + (f"（聚类 SE {p_flip_all['se_cluster']:.3f}）" if p_flip_all["se_cluster"] else "")
            + f"，n={p_flip_all['n']}：teacher 的偏好在与在线的真实偏好直接冲突的那一对上"
            "有可观比例是反的，说明排序知识不是免费资产。"
        )
    if flips_all:
        mags = [abs(r["dg"]) for r in flips_all]
        L.append(
            f"- 翻转对的 \\|dg\\| 中位数 {float(np.median(mags)):.3f}、均值 "
            f"{_fmt(_mean(mags))}：反的偏好并非都无关紧要，跟随它要付出真实回报。"
        )
    if net_all["mean"] is not None:
        L.append(
            f"- 决策相关对的 E[dg] = {net_all['mean']:.3f}（聚类 SE {_fmt(net_all['se_cluster'])}）"
            "是排序约束在每个决策上的期望效果，可正可负，本身就是选择性迁移的收益上界。"
        )
    if best_auc is not None:
        L.append(
            f"- 在线可观测特征的 out-of-fold AUC 最高 {best_auc:.3f}"
            + ("，显著高于 0.5，说明决策相关失配在部署时可被提前识别，gate 有可行性。"
               if best_auc > 0.58 else "，接近随机，gate 可行性存疑。")
        )
    L.append("")
    L.append("### 反驳继续研究 ranking 选择性迁移")
    L.append("")
    L.append(
        "- 若翻转集中在 |dg| 很小的近并列对上，或集中在 hinge 不活跃、不涉及执行动作的 P2 对上，"
        "则翻转率虽高但不可实现回报代价趋近于零，选择性迁移的收益空间被高估。"
    )
    L.append(
        "- 若 E[dg] 在决策相关对上为正或接近零，说明 teacher 偏好在被策略读取的那一对上大体是对的，"
        "无脑保留排序约束已经够用，额外的 gate 只是引入复杂度和方差。"
    )
    if best_auc is not None and best_auc <= 0.58:
        L.append(
            f"- 在线可观测特征 AUC 仅 {best_auc:.3f}：决策相关失配在部署时不可预测，"
            "任何 gate 都只能在事后知道该不该信 teacher，选择性迁移在原理上不可行。"
        )
    L.append(
        "- 本测量只用 seed 1 的一个 fork 家族；fork 实验本身已显示撤除排序约束无显著改善"
        "（3 个 seed 的 off-continue 差在不确定度内），若本报告同样给出接近零的净效果，"
        "ranking 选择性迁移缺少可追逐的机制目标。"
    )
    L.append("")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[report] wrote {out}", flush=True)

    # console summary
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="logs/rank_flip_seed1.json")
    ap.add_argument("--out", default="documentation/experiment_reports/taskB_rank_flip_seed1.md")
    args = ap.parse_args()
    payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
    write_report(payload, args.out)
