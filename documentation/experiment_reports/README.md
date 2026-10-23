# O2O MARL 阶段实验报告（SMAC 5m_vs_6m）

本文件夹收录《O2O MARL 下一阶段任务计划书》五项任务的实验报告，以及整合全部任务的最终报告。原始日志、检查点与统计输出存放在 `logs/` 与 `checkpoints/`（按 .gitignore 不入库）；报告中除图片链接外的路径均相对仓库根目录。

| 任务 | 报告 | 结论摘要 |
|---|---|---|
| **最终报告（整合五项任务）** | [final_report_5m_vs_6m_ranking_mechanism.md](final_report_5m_vs_6m_ranking_mechanism.md) | 《5m_vs_6m 联合动作排序可靠性与在线适应失效机制分析》：按计划书各项要求整合全部结果。实验支撑的主要结论：离线排序具有弱但可重复的预测能力（整体一致率 0.614，95% 置信区间 [0.582, 0.646]），仅 Q 值差距较大的动作对明显可靠（0.65–0.74），贪心决策直接比较的第一名与第二名（0.576）与随机水平无法区分；排序可靠性是离线 critic 自身的属性（13 个在线检查点上一致率 0.605–0.681）；撤除或衰减排序约束未产生可测量的终局改善（3 个种子 640 集配对差 +0.052 / +0.024）；OVM 在在线训练全程持续抬高回归目标。第七节以表格逐项列出计划书内各问题的结论与支撑证据 |
| 任务一：核查 Q 值与排序测量的正确性 | [task1_measurement_protocol_audit.md](task1_measurement_protocol_audit.md) | 审计两个测量脚本的指标口径，明确每个指标能说明什么、不能说明什么；Q 与数据集记录回报的差不等于真实价值估计误差，跨状态直接比较排序的问题已定性 |
| 任务二：验证离线联合动作排序可靠性 | [task2_rank_reliability.md](task2_rank_reliability.md) | 96 个状态、3,488 组联合动作对的分支回放实测：离线排序整体一致率 0.614（CI [0.582, 0.646]），greedy 决策附近≈抛硬币；Q gap 与数据支持程度都不能稳定预测排序正确性 |
| 任务二（续）：离线教师排序可靠性的时间演化 | [task2_rank_temporal.md](task2_rank_temporal.md) | 13 个后续策略（离线参照、在线初始化、标准 UOT 主线每 50 万步检查点、任务三分叉快照与三个 3M 终点）在同一批状态/动作对上实测：一致率 0.605–0.681 窄带，不随在线训练变化；主线六点相对 teacher 的配对差 +0.008→+0.063 为微弱上漂而非“过期”，与静态版 Q4 结论一致 |
| 任务三：5m_vs_6m Ranking Loss 分叉实验 | [task3_ranking_loss_fork.md](task3_ranking_loss_fork.md) | 同一 trainer 快照分出 continue / off / decay 三支各续训约 199 万步（3 个 seed）：640 集终局配对评估上撤除/衰减相对保留的配对差均值为 0.052/0.024 且 seed 间方向不一致，三个分支差异不明显；尾段胜率上撤除/衰减同向略优（+0.054/+0.060）但被每评测点 32 集的精度吞掉 |
| 任务四：直接验证 OVM 的价值约束机制 | [task4_ovm_mechanism.md](task4_ovm_mechanism.md) | 三张图同 init 的 on/off 受控对照：OVM 确实在整个训练过程中持续把目标抬高到普通 TD target 之上（直接测量）；"抬高导致性能差异"未建立因果 |
| 任务五：3s5z_vs_3s6z 上 OVM 与无 OVM 在线训练全程对照 | [task5_ovm_vs_no_ovm_3s5z.md](task5_ovm_vs_no_ovm_3s5z.md) | 同一离线模型出发的两条对照臂约 285 万步：前期（15 万步起）OVM 臂明显更好，无 OVM 臂回报跌至 2.3、排序被打乱；无 OVM 臂逐步恢复并于约 180–210 万步反超；终点无 OVM 臂 15.57 对 OVM 臂 13.93，且 Q 偏差降到接近零而 OVM 臂回升到 11.6 |
| 匹配协议后的 TDR / PAD+TDR 与 init × schedule 交互 | [task9_tdr_padtdr_matched_init.md](task9_tdr_padtdr_matched_init.md) | 在 matched protocol 的 5.08M-best 离线初始化上补跑 TDR 与 PAD+TDR（各 2 seed，配对 chunk bootstrap）：同一 η 调度在 10M-final 初始化下比固定 η 高 +0.217，在 5.08M-best 初始化下反而低 −0.058；固定调度下仅换初始化即使 TDR 终点变化 +0.166。结论：TDR 的优势来自初始化 × 调度的交互，不是调度的主效应；联合 vs 个体排序在两种子上符号翻转，不能宣称联合排序有稳定额外收益。同时给出 OVM × 混合交互的机制核查（TD target / 激活率 / Q 漂移）与尚缺的梯度贡献分解 |

## 复现方式

- 最终报告：不新增测量，全部数字取自下列任务级复现；数据/脚本/checkpoint 索引见最终报告第九节
- 任务二：`scripts/analyze_rank_reliability.py`（输入 `logs/rank_reliability_full.json`，统计输出存档于 `logs/rank_reliability_analysis.txt`）
- 任务二（时间演化）：测量 `scripts/run_rank_temporal_auto.sh <manifest> fixed <tag>`（输入 `scripts/verify_rank_temporal.py` 同款分支回放，manifest 格式见 `logs/rank_temporal_batch1_manifest.txt`），合并数据 `logs/rank_temporal_<tag>.json`；分析 `scripts/analyze_rank_temporal.py --in logs/rank_temporal_batch1.json --in logs/rank_temporal_batch2.json`（多 `--in` 合并两批共 13 臂，输出存档于 `logs/rank_temporal_batch2_analysis.txt`）
- 任务三：`python scripts/analyze_fork_branches.py --run-dir checkpoints/uot_fork_5m_vs_6m_seed1_formal --run-dir checkpoints/uot_fork_5m_vs_6m_seed2_formal --run-dir checkpoints/uot_fork_5m_vs_6m_seed3 --ovmse-ref checkpoints/ovmse_batch_20261014/5m_vs_6m_online/online.jsonl`（默认输出到本目录）
- 任务四：`python scripts/analyze_ovm_diag.py --run-root checkpoints/ovm_diag_20261017`（默认输出到本目录，机制图在 `logs/ovm_diag/`）

任务三的实验运行脚本为 `scripts/run_uot_fork_5m_vs_6m.sh`，任务四为 `scripts/run_ovm_diag.sh`；两者的原始日志分别在 `checkpoints/uot_fork_5m_vs_6m_seed1_formal/` 与 `checkpoints/ovm_diag_20261017/`。
