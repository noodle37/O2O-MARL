# O2O MARL Q 值偏差与排序实验设计（论文 arXiv:2312.07685 协议）

## 一、目的

复刻论文《A Perspective of Q-value Estimation on Offline-to-Online Reinforcement Learning》中
测量 Q 值偏差与排序的实验，将其用于 O2O MARL，回答三个问题：

1. O2O MARL 的 Q_tot 相对策略实际回报是否存在系统偏差，方向与幅度如何；
2. Q_tot 对同一策略轨迹内状态的排序是否准确，Kendall τ 是否在在线阶段下降；
3. OVM（offline value memory）开关是否改变偏差与排序。

不复刻 SO2 算法。测量对象是 QMIX + CQL 离线预训练、在线混合回放的 O2O 流程。

## 二、环境与受控设置

| 环境 | 数据 | 离线检查点 | 动作空间 | 回合长度 |
|---|---|---|---|---|
| 5m_vs_6m | Vault Medium，5000 回合 / 135,256 填充步 | `checkpoints/ovmse_paper_5m_vs_6m/ovmse_paper_seed1_best.pt` | 每智能体 12 个离散动作 | 约 27 步，胜/负终止 |
| 2halfcheetah | OG-MARL Good，1000 回合 / 1,000,000 填充步 | `checkpoints/ovmse_2halfcheetah_offline/ovmse_paper_seed1_best.pt` | 每智能体 27 个力矩分箱 | 1000 步 |

四组在线运行，保持离线检查点、随机种子、在线预算、评估回合数一致，只切换 OVM：

| 运行 | 环境 | transfer.mode | use_ovm | 在线步数 | 评估间隔 | 评估回合 |
|---|---|---|---|---|---|---|
| `qdiag_2halfcheetah_off` | 2halfcheetah | full_td | false | 3M | 100k | 32 |
| `qdiag_2halfcheetah_on` | 2halfcheetah | ovmse | true | 3M | 100k | 32 |
| `qdiag_5m_vs_6m_off` | 5m_vs_6m | full_td | false | 300k | 5k | 32 |
| `qdiag_5m_vs_6m_on` | 5m_vs_6m | ovmse | true | 300k | 5k | 32 |

5m 使用 5k 评估间隔，因为既有日志显示排序崩塌发生在 1 万至 3 万步的暂态窗口；稀疏评估会
错过该窗口。2halfcheetah 使用 3M 预算：每个回合固定 1000 步，在线 batch 为 128 个回合，
replay buffer 要到 128k 步才填满，300k 只够约 86 次梯度更新；3M 才与 5m 的更新次数同量级
（约 1400 次）。2halfcheetah 每 100k 评估、每 250k 保存周期检查点；5m 每 25k 保存。
四组全部以 `nohup` 运行在 GPU1。

## 三、Q 真值的定义

对当前贪心策略（ε=0）的一条完整回合轨迹：

- `Q_est(t) = Q_tot(s_t, u_t)`：QMIX mixer 对实际执行的联合动作给出的值；
- `Q_true(t) = G_t = Σ_{k≥0} γ^k r_{t+k}`：沿该轨迹从 t 起的折扣回报；
- 团队奖励取各智能体奖励均值，与训练时一致；
- γ=0.99；回合终止即停止累加，不跨回合自举。

这与论文的“沿当前策略延长轨迹并计算真实回报”一致。评估在冻结网络上只做前向推理，
不更新参数。离线侧另用 `scripts/measure_paper_q.py` 在固定初始种子的回合上复测。

## 四、偏差指标

1. **原始偏差** `bias = Q - G`，及其均值、中位数、标准差、分位数；
2. **窗口/回合聚类标准误**：以回合为独立样本单位，避免把回合内相关的时间步当作独立样本；
3. **符号稳定归一化偏差** `(Q - G) / max(|G|, 1)`：2halfcheetah 的真实回报为负，
   直接除 `G` 会翻转符号，故作为主归一化口径；
4. **论文逐字归一化** `(Q - G) / G`：保留以对照原文，但在 `G≤0` 或接近 0 时只作参考；
5. **高估率 / 低估率 / 并列率**：`Q>G`、`Q<G`、`|Q-G|≤1e-8` 的步占比。

判定“存在偏差”的口径：高估率与低估率显著偏离 50%，且原始偏差与归一化偏差的回合聚类
标准误不跨 0；同时报告 Q 与 G 的均值、标准差和仿射拟合 `G ≈ aQ+b`，以区分幅度偏差与
排序偏差。

## 五、排序指标

1. **整回合 Kendall τ**：对一条回合内全部 `(Q_t, G_t)` 计算 τ，再对回合取平均；
2. **论文滑窗 τ（主指标）**：窗口 W=10 个连续状态-动作对、步长 1，每个窗口计算
   `K_i = Kendallτ({Q_t}, {G_t})`，回合内平均后跨回合平均；
3. **敏感性**：W=5 与 W=20；
4. **负窗口比例**：`K_i < 0` 的窗口占比；
5. **窗口 τ 的回合聚类标准误**。

整回合 τ 对大尺度漂移不敏感：Q 与 G 可以一起变、只差一个倍数，τ 仍可能很高。滑窗 τ
限制在局部 10 步内，更接近“同一状态附近动作值排序”的问题，因此作为主指标。两者同时报告。

## 六、评估协议与噪声控制

- 在线评估回合数为 32；评估时 ε=0，隐藏状态在每个回合开始时重置；
- 在线评估的环境重置种子未在 `evaluate()` 内逐回合固定，因此 2halfcheetah 的回合间
  初始状态方差较大；最终以周期检查点 + 固定种子探针（`--seed 12345`，8 回合）作为
  跨时间可比的排序读数，在线 jsonl 作为连续监控；
- 原 `MujocoMulti` 未把 seed 传入 Gym 环境（`seed()` 为空实现），已改为直接重置底层
  `np_random`；两次独立探针的回合奖励与全部指标现已逐位一致；
- 5m 的胜负是二值结果，32 回合的胜率标准误约 0.09；因此胜率只作辅助，偏差与 τ 为主；
- 单训练种子，结果作描述性量化，不作显著性推断；回合聚类标准误只刻画评估回合的抽样误差。

## 七、已知边界

2halfcheetah 的离线检查点 `best_eval_return=-450.8`，而数据集中行为策略的回合回报为
`+6924±1271`。离线策略在该离散动作/环境组合上明显欠拟合，且离线训练日志中 Q 均值从
0.4 升到 239、真实回报始终在 -50 附近，偏差从 +54 升到 +287、高估率 99%–100%，整回合
τ 在 -0.20 至 0.21 之间。因此 2halfcheetah 用作“弱离线策略 + 已失准离线 Q”的压力测试，
不能当作“从强离线策略出发”的干净 O2O 证据；其结论需与 5m 相互印证。5m 的离线策略
评估回报约 12.2（胜率 0.25），与数据集水平一致，是主要证据。

## 八、产物

| 内容 | 路径 |
|---|---|
| 论文口径指标实现 | `training/q_metrics.py`（`summarize_paper_diagnostics`） |
| 冻结检查点探针 | `scripts/measure_paper_q.py` |
| 四组在线配置 | `configs/qdiag_{2halfcheetah,5m_vs_6m}_{off,on}.yaml` |
| 在线日志 | `checkpoints/qdiag_20261020/<run>/online.jsonl` |
| 周期检查点 | `checkpoints/qdiag_20261020/<run>/periodic/` |
| 轨迹汇总 | `scripts/analyze_qdiag_runs.py` |
| 固定种子探针结果 | `checkpoints/qdiag_20261020/probes/` |
| 最终报告 | `documentation/paper_q_bias_marl_report.md` |
