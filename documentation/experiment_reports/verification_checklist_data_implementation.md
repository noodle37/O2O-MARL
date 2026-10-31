# 实验数据与实现单独核对表（问题 2–8）

> 本文件为改正文前的核对记录。全部检查均为只读；另在 `/tmp` 下做了四次只读重放实证：ε 独立性、joint_reuse 选优复算、Sim2O 候选结构、MaMuJoCo 转移复现。证据均以 `文件:行号` 标注。
>
> 判定口径：**符合** / **部分符合** / **不符合** / **数据缺失**。

---

## A. 已有原始数据核对

### A1 随机扰动实验（问题 2）

| 核对项 | 判定 | 证据与缺口 |
|---|---|---|
| 全部智能体独立加 ε | **符合** | `pv/exp2_shift.py:150-173` 逐 agent 循环内独立抽样；驱动侧 `scripts/run_pv_exp2a.py:286-289` 令 `offline_agents=全部`、`examined=set()`；stub 重放 5 agent 偏离率各自围绕 ε 独立波动，非广播、非只扰焦点 |
| 逐步记录（焦点/同伴动作、状态、Q、G） | **不符合** | `obs/actions/state/q_off_tot` 仅在内存即被丢弃（`scripts/run_pv_exp2a.py:169,257,282`）；落盘仅 7 个聚合键，无 per-agent Q/G、无焦点-同伴拆分 |
| Spearman 口径 | **不符合** | `scripts/run_pv_exp2a.py:163` 把跨 episode 的逐步 (Q^off_tot, G) 拼成长向量求**跨状态 pooled ρ**，每状态仅一个观测，结构上不是同状态候选排序；同状态分支回放只在另一实验存在（`task2_rank_reliability.md:5`，96 状态 3,488 对，一致率 0.614 [0.582, 0.646]） |
| 相关系数/偏差区间 | **部分符合** | 正文为单点估计（原 FHJP 整合报告，已随算法删除），主运行未开 `--save-episodes`；但 `checkpoints/missing_ae_20261024` 等 21 个含逐局回报的同族运行从未被 bootstrap，可补救。注意它们只改 rng 流位置，不是独立策略重复，只能估抽样误差 |

**小结**：该实验能支撑的定性主张是"队友扰动越大，离线 Q_tot 与实际回报的跨状态关联越弱、归一化偏差越负"（剂量反应单调，且在 21 个独立噪声实现上方向一致）。不能支撑"同状态内离线动作排序在扰动下失效"，也不能支撑带区间的定量声称。

### A2 15 种移植配置（问题 7）

- 磁盘实为 **36 个 (arm, version, seed) 单元**（每单元 16 配置 × 200 局）；**"26 个版本"在仓库中无法复现**（报告口径 30、选优分析口径 23、有逐局数据 29），改正文前须先统一计数口径。
- 逐局回报：29/36 有；**7 个仅点估计**——恰是 arm A seed1 的整条版本梯 `A@0/25k/50k/100k/250k/500k/1M@s1`（`checkpoints/pv_exp2a/5m_vs_6m/seed1_natural_shard{1,2}.json`），即**跨步数证据完全没有区间**；joint_reuse 在线实验 9/9 分支只有 chunk 粒度（`training/evaluation.py:87` 的 `return_episodes` 分支从未启用）。
- 选优与增益估计**未拆分**：专用脚本 `scripts/analyze_joint_vs_independent_selection.py:10-14,108-127` 做了 split-half + 簇 bootstrap，但真正驱动训练的保留组合来自"前 50/100 局"规则（`scripts/run_joint_reuse_seed.sh:33-46`，已复算复现），与报告 ΔJ 共用同一 200 局，`scripts/analyze_insight_ae_full.py` 对 argmax 选优无去偏；且点估计按单元等权（+0.1872）与区间按簇等权（+0.1281）不同权。
- 种子分布不平衡：arm A 7 个版本全单 seed；1M 处"臂 C vs 臂 A"实为 seed1 vs seed2/3，臂与种子混淆（`insight_ae_full_20261025.md:254` 自认）；6 个已落盘的 `checkpoints/extra_seeds_20261024` A 单元未进任何分析，不平衡本可修复。

### A3 640 局评估（问题 7）

- 每个文件确为单一 checkpoint，但跨 3 个训练种子；更严重的是 `scripts/eval_640.py:62` 以**训练种子**为环境种子（`scripts/run_joint_reuse_seed.sh:26,81`），未用项目标准 `eval_env_seed=424242`（周期评估反而用了，`joint_reuse/runner.py:107-108`）→ **跨 seed 终点对比不在同一评估情景集上**；终点 `t_env` 500056–500306 亦未对齐。
- 区间口径表述**符合**：`insight_ae_full_20261025.md:256` 明确"只反映评估 episode 抽样误差，不含训练 seed/checkpoint 不确定性"，`matched_protocol_5m_vs_6m_640_findings.md:55` 同；唯一过度承诺是 `joint_reuse_online_validation_plan.md:90` 的"逐局区间"（实际只有 chunk）。

---

## B. 需新增的对照/实验（现状均为缺失或部分）

| 项 | 判定 | 要点 |
|---|---|---|
| B1 同伴变化因果分离（问题 2） | **缺失** | 只有"全队扰动"一个剂量家族，无"只扰同伴/只扰焦点/全体"三组；同状态分支回放未与扰动联动 |
| B2 混合回报与学生学习（问题 3） | **部分** | ρ 混合在循环层（`training/online.py:446,539-574`），OVM 为 `max(离线Q, TD)`（`ovmse/ovm.py:68`），λ 退火为预设调度（`ovmse/ovm.py:29-36`）。**没有任何一次运行同时记录混合策略回报（仅 V1）与纯在线臂回报（仅 V0，`offline/online_arm_return`）**；学生只评在线策略；无"后续增量"字段。对照：纯在线已配对，固定比例只有 ρ=0.1 单点、**无 ρ 扫描**，λ 退火 vs 固定无对照，ROAD 单 seed 无报告 |
| B3 离线 RNN 历史消融（问题 4） | **缺失** | joint_reuse 离线网用自提议闭环历史（`joint_reuse/policy.py:157-161`），基础 QMIX/FSR/Sim2O 用已执行动作；无只切换历史来源的开关或实验，原 FHJP-V1 规范明确列为 future work（正文未过度声称）。episode 复位正确无泄漏；但存在训练-推理口径错配且从未量化，且 `insight_af_validation_5m_vs_6m.md:124` 曾把同类"反事实历史"当 bug 修复，仓库内部口径不一致 |
| B4 Sim2O 功能性检查（问题 5） | **缺失（诊断为零）** | trace 仅 4 字段（`sim2o/learner.py:484-493`），`n_disagree` 因 `training/evaluation.py:50-59` 的 if/elif 被丢弃，`online.jsonl` 中出现 0 次；无选择频率/评分差/平票率。结构上先验冻结、参数已分叉（mean\|Δ\|=0.055）不会退化为始终全在线；只读重放 1716 次决策显示 **92.3% 的步候选集塌成 1 个（先验=在线）**，有选择时全在线 52.3%/全离线 44.7%，平票率 97.7%——即真实风险是"无选择 + 评分无区分"，而非退化；但这是离线状态分布上的单 seed 存在性证据，**无按 t_env 的日志，不能作为结论** |
| B5 MaMuJoCo 数据协议（问题 6） | **部分，且有一处 env bug** | vault 原始为连续 float32，训练集为 27 档最近邻量化 int64（`data/vault_dataset.py:163-179`、`envs/discrete_mujoco_actions.py:9-51`）；仅 8.6% 动作向量在栅格上，91% 转移被量化改变；无一致性验证脚本。实测：按 env 现有语义仅 1.7% 复现下一状态，而把每 agent 三个分量按 partition 反序后 **80/80 精确复现**（中位误差 8.6e-7）——数据自洽，是 `envs/custom/multiagent_mujoco/mujoco_multi.py:126-130` 未按 `obsk.py:193-194` 的 partitions 重映射。量化标签解码后重放 0/80 复现，故"离散标签↔下一状态"严格一致**原理上做不到**：先修 env 映射，再择（写入局限 / 改连续协议 / 重新采集）。另 `data/vault_dataset.py:177-179` 对非 2halfcheetah 有潜在 bug |
| B6 独立选优验证与多训练种子（问题 7） | **部分** | split-half 验证脚本存在但未用于实际选优；joint_reuse 在线仅 3 seed（`checkpoints/pv_exp1` 下有 6 个 250k 检查点未用）；无独立测试回合 |

---

## D. 改正文前必须修订的声称

1. **问题 2**：Spearman 须从"同状态动作排序"改为"跨状态价值关联"；ρ/偏差要么用已有 21 个运行补 bootstrap 区间，要么降级为方向性结论。
2. **问题 7**："26 个版本"数字不可复现，须统一计数口径；选优与 ΔJ 共用数据须明示"条件于选优"；640 局评估环境种子不一致须修正或披露。
3. **问题 3**：不能声称混合/纯在线同步统计与 ρ 扫描。
4. **问题 4**：只能陈述为设计选择并列 future work，不能声称已完成消融。
5. **问题 5**：只能声称"结构上不退化"；候选选择频率、评分差、平票率须补记后再下结论。
6. **问题 6**：离散标签与动力学不一致是实质性缺陷，优先修 `mujoco_multi.py:126-130` 的动作映射。
