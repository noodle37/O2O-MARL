# O2O MARL 阶段实验报告（SMAC 5m_vs_6m）

本文件夹收录《O2O MARL 下一阶段任务计划书》四项任务的实验报告。原始日志、检查点与统计输出存放在 `logs/` 与 `checkpoints/`（按 .gitignore 不入库）；报告中除图片链接外的路径均相对仓库根目录。

| 任务 | 报告 | 结论摘要 |
|---|---|---|
| 任务一：核查 Q 值与排序测量的正确性 | [task1_measurement_protocol_audit.md](task1_measurement_protocol_audit.md) | 审计两个测量脚本的指标口径，明确每个指标能说明什么、不能说明什么；Q 与数据集记录回报的差不等于真实价值估计误差，跨状态直接比较排序的问题已定性 |
| 任务二：验证离线联合动作排序可靠性 | [task2_rank_reliability.md](task2_rank_reliability.md) | 96 个状态、3,488 组联合动作对的分支回放实测：离线排序整体一致率 0.614（CI [0.582, 0.646]），greedy 决策附近≈抛硬币；Q gap 与数据支持程度都不能稳定预测排序正确性 |
| 任务三：5m_vs_6m Ranking Loss 分叉实验 | [task3_ranking_loss_fork.md](task3_ranking_loss_fork.md) | 同一 trainer 快照分出 continue / off / decay 三支各续训约 199 万步：640 集终局配对评估上撤除与衰减优于保留约 0.16–0.17，撤除与衰减之间无差别 |
| 任务四：直接验证 OVM 的价值约束机制 | [task4_ovm_mechanism.md](task4_ovm_mechanism.md) | 三张图同 init 的 on/off 受控对照：OVM 确实在整个训练过程中持续把目标抬高到普通 TD target 之上（直接测量）；"抬高导致性能差异"未建立因果 |
| 任务五：3s5z_vs_3s6z 上 OVM 与无 OVM 在线训练全程对照 | [task5_ovm_vs_no_ovm_3s5z.md](task5_ovm_vs_no_ovm_3s5z.md) | 同一离线模型出发的两条对照臂约 285 万步：前期（15 万步起）OVM 臂明显更好，无 OVM 臂回报跌至 2.3、排序被打乱；无 OVM 臂逐步恢复并于约 180–210 万步反超；终点无 OVM 臂 15.57 对 OVM 臂 13.93，且 Q 偏差降到接近零而 OVM 臂回升到 11.6 |

## 复现方式

- 任务二：`scripts/analyze_rank_reliability.py`（输入 `logs/rank_reliability_full.json`，统计输出存档于 `logs/rank_reliability_analysis.txt`）
- 任务三：`python scripts/analyze_fork_branches.py --run-dir checkpoints/uot_fork_5m_vs_6m_seed1_formal --ovmse-ref checkpoints/ovmse_batch_20261014/5m_vs_6m_online/online.jsonl`（默认输出到本目录）
- 任务四：`python scripts/analyze_ovm_diag.py --run-root checkpoints/ovm_diag_20261017`（默认输出到本目录，机制图在 `logs/ovm_diag/`）

任务三的实验运行脚本为 `scripts/run_uot_fork_5m_vs_6m.sh`，任务四为 `scripts/run_ovm_diag.sh`；两者的原始日志分别在 `checkpoints/uot_fork_5m_vs_6m_seed1_formal/` 与 `checkpoints/ovm_diag_20261017/`。
