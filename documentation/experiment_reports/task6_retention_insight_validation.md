# 保留什么、如何保留：四条最小验证实验

## 一、目的

前五份报告给出的证据是：离线教师的绝对 Q 值有偏、联合动作排序整体只有约 0.61 的一致率、局部 top-1/top-2 接近随机；完全不用离线知识会在早期严重遗忘，完全保留又会限制后期适应。据此提出的核心 insight 是：

> **保留可行的协调集合与行为模式，释放偏好顺序；把离线知识从“目标/正则项”降级为“提议 + 打分”，在决策层校准，而不是校准 Q。**

本轮不复现 FSR，也不引入 flow matching。目标是用四个便宜、可证伪的测量，判断这条 insight 是否值得进入算法设计。四个实验分别对应四个假设：

| 编号 | 假设 | 如果成立说明什么 |
|---|---|---|
| H1 | 成功轨迹上的单 agent 反事实偏离，其 Q̄ 排序可靠性显著高于随机联合动作对；大效应偏离尤其可靠 | “负知识/可行性”是可保留的，且比全局排序更可靠 |
| H2 | 在一个由行为数据构造的 K 候选集合内，Q̄ 选 top-1 的命中率显著高于全局 pair ranking，且接近候选内真实最优 | “提议 + 打分”比“全局排序”更可靠，proposal + scorer 结构有实证基础 |
| H3 | 在线训练会把多个协调模式压缩成一个（late checkpoint 的跨 seed/跨策略多样性下降） | 需要 mode library / selector，而不是只保留一个策略 |
| H4 | 回合级 PEX-M（冻结离线策略 + 在线新策略 + Thompson 选择器）同时改善早期最低回报与后期上限 | “保留行为、释放 critic”可以作为算法骨架 |

四个实验都不更新任何参数；只有 H4 的 pilot 会在线训练，且不引入 FSR/flow。

## 二、已有证据与本轮新增测量

已有证据（详见 `final_report_5m_vs_6m_ranking_mechanism.md` 与 `task3_ranking_loss_fork.md`）：

- 96 个探针状态、3,488 个联合动作对：全局 pair 一致率 0.614 [0.582, 0.646]；top-1 vs top-2 为 0.576；小 |ΔQ| 对 0.519；|ΔQ|>5 的对约 0.65–0.74。
- 状态条件联合支撑极稀疏：最近 64 个离线状态里，联合动作命中只有 18/3,072。
- ranking loss 撤除/衰减与 continue 的 640 局终局配对差只有 +0.052/+0.024，方向跨 seed 不一致。
- OVMSE 的在线尾段胜率约 0.725；UOT 在 3M 附近约 0.62。

本轮新增的四类测量，正好填补上面证据链中缺的四个环节：单 agent 偏离、候选内选择、模式多样性、回合级行为保留。

---

## 三、实验一：单 agent 反事实偏离的排序可靠性

### 3.1 问题

在成功轨迹的状态上，把某一个 agent 的动作换成一个可行的替代动作，联合动作的回报通常会明显变差。Q̄ 是否能可靠地判断“原联合动作优于这个单 agent 偏离”？这类**大效应、单坐标**的关系，是否比随机联合动作对更可靠？

### 3.2 数据与探针

- 教师：`checkpoints/uot_temporal_5m_vs_6m/uot_online_none_seed1.pt`（UOT 3M 终局）。原计划用 offline best，但实测其贪心胜率约 0.2（80 seed 只能得到约 16 条成功轨迹），UOT 终局在同样的 12 个 seed 上 6 胜，能提供约 40 条成功轨迹。这里检验的是“冻结 Q 的大效应排序是否可靠”，不依赖 Q 来自 offline 还是 online；与 task 2 的 0.614 只作为参考基线，报告时注明教师不同。
- 环境：SMAC v1 `5m_vs_6m`，Medium 数据集；环境 seed 1..80（可扩到 200），贪心 rollout；另跑一组 eps=0.3 的 eps-greedy rollout 作为第二 population。
- **成功轨迹**：episode 的 `battle_won=True`。教师贪心胜率约 0.5，预计 80 个 seed 能拿到约 40 条成功轨迹；若不足 8 条，自动扩 seed 或把“高回报但未胜”的轨迹作为 secondary population 并显式标注。
- 每条成功轨迹取 2 个探针步：`frac=0.3, 0.7`（四舍五入到合法步），单条轨迹最多 2 个状态。
- 探针状态上的 **base joint** = 该轨迹在该步实际执行的联合动作（成功 joint）。

### 3.3 偏离构造

对每个 agent i，构造三类单 agent 偏离（只改 i，其余 agent 保持 base joint）：

1. `primary`：在离线 kNN 邻域（最近 64 个离线状态）里，i 的**状态条件边际频率最高**的合法替代动作。这是“行为数据认为可行、但成功轨迹没选”的偏离。
2. `hard`：在 i 的合法动作里，**Q̄ 给分最高**的替代动作。这是最难的竞争者，直接检验 Q̄ 在近邻动作上的分辨力。
3. `random`：i 的合法替代动作中随机一个（固定 RNG）。

三类偏离分别记录，不混在一起统计。每个状态最多 5×3=15 个偏离 joint，去重后 branch-replay。

### 3.4 Branch replay

对每个 joint：

1. 从同一环境 seed 重新启动 episode；
2. 逐位重放 base 轨迹的前缀动作（bit-for-bit）；
3. 在探针步强制该 joint；
4. 之后按教师贪心继续，直到 episode 结束；
5. 折扣回报从探针步起算（γ=0.99）。

探针步的 state 必须与 base rollout bit-exact 相等，否则丢弃该分支。**base joint 本身也 branch-replay 一次**，用贪心续跑的回报作为 G_base；这样 eps-greedy 轨迹的未来探索噪声不会污染对照。

### 3.5 随机对对照

在同一批探针状态上，从“合法 × 离线联合支撑”里随机抽 6 对联合动作，branch-replay 两端，计算同样的 Q̄ 排序 vs 回报排序一致率。这是**同状态、同 continuation、同统计口径**的全局 pair 基线；task 2 的 0.614 只作为外部参照，不直接拿来比较。

### 3.6 指标

- 一致率（concordance）：`sign(Q_base - Q_dev) == sign(G_base - G_dev)`，tie 单列。
- 按偏离类型、按 agent、按 `|ΔG|` 分箱（[0,0.5)、[0.5,2)、[2,5)、≥5）、按 `|ΔQ|` 分箱。
- **负知识可靠率**：在 `G_dev < G_base` 的偏离里，Q̄ 正确把 base 排在 dev 之上的比例。
- 单 agent 偏离的效应分布：`G_base - G_dev` 的均值/中位数、变差/变好/不变的比例。
- 统计：以 `(seed, wave)` 为 cluster 的 cluster-robust mean、SE、95% CI；偏离 vs 随机的差用同一 cluster 的配对 bootstrap。

### 3.7 判定规则

- **支持 H1**：偏离一致率 − 随机一致率 ≥ 0.10，且配对 95% CI 不含 0；同时 `|ΔG|≥2` 分箱的一致率 ≥ 0.70。
- **不支持**：差值 ≤ 0，或 CI 跨 0，或大效应分箱 < 0.60。
- 其余为“证据不足/方向不明”。

---

## 四、实验二：候选可选择性（proposal + scorer）

### 4.1 问题

不训练 flow，直接由行为数据构造 K 个联合候选；用 Q̄ 在候选内选 top-1，再用 branch replay 得到这 K 个候选的真实回报排名。Q̄ 的**候选内选择**是否比全局 pair ranking 更可靠？

### 4.2 候选集合

所有候选都只由行为数据/合法性构造，**不先用 Q̄ 筛选**；Q̄ 只在最后打分。每个探针状态构造三套集合：

1. `support_dev`（K≤6）：base joint + 每个 agent 的 `primary` 单 agent 偏离。对应“保留可行单 agent 替代”的最小提议集。
2. `support_neigh`（K≤8）：
   - base joint；
   - 最近 64 个离线状态里，每个 agent 的状态条件边际 top-2 动作替换 base 中的该 agent；
   - 最近邻离线联合动作中合法、且与 base 的 Hamming 距离 ≤2 的 top 行；
   - 去重后按行为分数（邻域联合计数 + 各 agent 边际计数之和）排序取前 K。
3. `random`（K≤8，对照）：base + 从“合法 × 离线联合支撑”随机抽的 joint。

`support_dev` 是 `support_neigh` 的子集，但单独统计，因为它直接对应实验一的偏离结构。

### 4.3 测量

对每个集合的每个候选 branch-replay（与实验一共用 `g_of` 缓存），得到：

- Q̄ 分数 `q_tot`（用探针步记录的 per-agent Q 与 mixer 计算）；
- 真实折扣回报 `G`；
- Q̄ top-1 是否等于真实 top-1（tie 容差 1e-4）；
- 真实最优是否在 Q̄ top-2；
- regret = `G_true_best - G_selected`；
- 候选内所有 C(K,2) 对的 Q̄/回报一致率；
- Kendall τ（Q̄ 排名 vs 回报排名）；
- 真实最优的 Q̄ 排名、Q̄ top-1 的真实排名。

### 4.4 与全局基线的比较

- **全局 pair 基线**：实验一的同状态随机对一致率（不是直接引用 0.614）。
- **候选内 pair 一致率**：K 个候选内部所有对的一致率。
- **top-1 命中率**：与随机选 top-1 的期望 1/K 比较；同时报告“top-1 命中率 − 1/K”的 cluster-bootstrap CI。

### 4.5 判定规则

- **支持 H2**：候选内 top-1 命中率的 95% CI 下界 > 1/K + 0.10；或候选内 pair 一致率 ≥ 全局基线 + 0.10 且 top-1 命中率显著高于 1/K。辅助条件：selected-in-top2 ≥ 0.85，median regret ≤ 1.0。
- **不支持**：top-1 命中率 ≤ 1/K + 0.05，或候选内 pair 一致率 ≤ 全局基线。
- 其余为证据不足。

### 4.6 解释边界

候选集只有 K≤8，top-1 的随机基线是 1/K≈0.125–0.167，和全局 pair 的 0.5 基线不可直接比。文档和报告里会同时给出“命中率 vs 1/K”和“pair 一致率 vs 全局 pair”，避免把两种基线混为一谈。

---

## 五、实验三：协调模式多样性与 collapse

### 5.1 问题

在线训练是否把多个协调模式压成一个？如果会，单一策略保留不足，需要 mode library/selector。

### 5.2 13 个策略 checkpoint

主集合（13 个）：

| 家族 | tag | 路径 |
|---|---|---|
| 离线教师 | `teacher_best` | `checkpoints/ovmse_paper_5m_vs_6m/ovmse_paper_seed1_best.pt` |
| 离线教师 | `teacher_final` | `checkpoints/ovmse_paper_5m_vs_6m/ovmse_paper_seed1.pt` |
| UOT 轨迹 seed1 | `uot_t500k` … `uot_t3m` | `checkpoints/uot_temporal_5m_vs_6m/periodic/ovm_diag_ckpt_t{500000,1000000,1500000,2000000,2500000,3000000}.pt` |
| UOT fork seed1 | `fork_t1m` | `checkpoints/uot_fork_5m_vs_6m_seed1_formal/fork_snapshot_main_t1009049.pt` |
| UOT fork seed1 | `continue/off/decay` | `checkpoints/uot_fork_5m_vs_6m_seed1_formal/uot_online_fork_resume_{continue,off,decay}_seed1.pt` |
| OVMSE 在线 | `ovmse_online` | `checkpoints/ovmse_batch_20261014/5m_vs_6m_online/ovmse_paper_online_ovmse_seed1.pt` |

`fork_t1m` 属于 fork 快照格式，需要从 `q_network/mixer` 字段读取（脚本会识别 snapshot）。其余 fork seed2/seed3 终局作为 secondary 集合，不进入主 13。

### 5.3 两种行为签名

**A. 自身 rollout 签名（主）**

- 每个策略在 seed 1..20 上贪心 rollout，记录：
  - episode return、win、length；
  - 每步联合动作序列；
  - 全局联合动作频次向量（A^N 维稀疏）、每 agent 动作频次；
  - 探针 frac 处的存活单位数。
- 每个策略的签名 = 跨 seed 平均的联合动作分布 + 每 agent 动作分布 + return/length 统计。

**B. 固定状态反事实签名（辅）**

- 用教师贪心/eps-greedy 在 20 个 seed 上收集前缀 observation/legal/action 序列；
- 对每个策略，在教师前缀 observation 上按**该策略自己的贪心动作**更新 RNN，记录探针步的联合动作；
- 这样得到每个策略在同一批状态上的条件动作，避免“各自 rollout 到不同状态”导致的不可比。

### 5.4 指标

- **策略内跨 seed 多样性**：每对 seed 的联合动作分布 JS 散度均值；动作熵；dominant action share；return 标准差。
- **策略间距离**：20 策略两两 JS 散度（rollout 签名）；层次聚类。
- **episode 聚类**：把所有策略×seed 的 episode 按其联合动作分布聚类（K=2..8，按 silhouette 选 K）；报告每个策略的 cluster occupancy 熵、有效模式数 `exp(H)`、dominant share。
- **固定状态模式数**：每个探针状态上，13 个策略给出的不同联合动作数、动作分布熵；早窗口（UOT 时间序列的前一半，3 个 checkpoint）vs 晚窗口（后一半，3 个）的对比。
- **时间趋势**：UOT 轨迹 6 个 checkpoint 的上述指标 vs t_env；报告斜率和早/晚窗口比值。

### 5.5 判定规则（描述性）

- **支持 H3**：晚窗口的策略内多样性或 occupancy 熵相对早窗口下降 ≥30%，或晚窗口 dominant share ≥0.60 而早窗口 <0.4；固定状态模式数同步下降。
- 若各项基本持平或晚窗口更高：不支持“在线训练必然 collapse”；mode library 的必要性需要别的证据。
- 由于 checkpoint 之间高度相关、每个训练 run 只有 1 个训练 seed，本实验只做描述性统计和 bootstrap 区间，不做显著性宣称。

---

## 六、实验四：回合级 PEX-M 在线 pilot

### 6.1 结构

- **离线策略 π_off**：`ovmse_paper_seed1.pt`（与 UOT 的起点相同）的冻结副本，贪心 + 与在线策略相同的 ε 探索。
- **在线策略 π_on**：同一起点初始化的独立 QMIX；只做在线 TD(λ) 更新，无 OVM、无 CQL、无 UOT ranking、无离线数据混批（`mixing_ratio=0`）。这与 UOT 的唯一差别是：没有 ranking hinge，多了冻结行为策略和选择器。
- **回合级 Thompson 选择器**：每个 episode 开始前，从两个 arm 的 return 后验各采一个均值，选较大的 arm；episode 结束后用该 episode 的 return 更新后验（Normal-Inverse-Gamma）。为避免在线 arm 被饿死，保留 `min_online=0.25` 的强制在线比例；该比例和选择器的选择率都会记录。
- 选择器只在训练/评估时集中式选择；执行时选中的仍是一个联合策略，每 agent 跑自己的网络，保持 decentralized execution。

### 6.2 训练设置

- 环境/评估协议与 UOT 完全一致：8 个并行 env、buffer 5000、batch 128、ε 1.0→0.05 在 100k 步、每 10k 步评测 32 集、3M 总步数、seed 1。
- 训练数据：两个 arm 产生的 episode 都进 buffer；在线策略从所有 episode 做 off-policy TD。
- 日志：每 10k 步同时评测 π_on（贪心）和 π_off（贪心），记录 win/return/min-return、arm 在线比例、强制在线比例、两个 arm 的后验均值/方差、选择器命中率。注意 SMAC v1 在同一进程内跨 episode 的 reset 也不是 bit-exact 的，因此 online/offline 两次评测是独立样本；离线 arm 的曲线同时充当“初始策略”的重复测量，用于估计评测噪声和早期遗忘。
- 终局：保存 `pexm_online_seed1.pt` 和 `pexm_selector_seed1.json`；每 100k 步额外保存 `periodic/pexm_ckpt_t*.pt`。

### 6.3 对比与指标

- 对照：UOT seed1（`uot_temporal_5m_vs_6m/online.jsonl`，3M）与 OVMSE seed1（`ovmse_batch_20261014/5m_vs_6m_online/online.jsonl`，3M）。不比较 FSR。
- 早期遗忘：前 200k 步内 win rate / mean return 的最小值与均值；以冻结离线 arm 的评测均值作为“初始策略”参考（它每 10k 重复测量同一策略，可直接给出评测噪声带），报告在线 arm 相对该参考的下降幅度。
- 学习速度：0–3M 的 win-rate AUC（对 t_env 做梯形积分，按相同评测网格对齐）。
- 后期上限：最后 5 个评测点的 win rate 均值；终局 640 集贪心评估（Wilson 95% CI）。
- 选择器行为：在线 arm 的实际比例、后验均值轨迹、离线 arm 的评测曲线（检验“冻结策略是否真的不遗忘”）。
- 终局 640 集评估三种部署方式：π_on 单独、π_off 单独、Thompson 选择器（固定后验采样）、后验均值选择器（确定性）。

### 6.4 判定规则（pilot，单 seed）

- **支持 H4**：早期最低 win rate 不低于 UOT（差值 ≥0 或至少不被 32 集评测噪声淹没），且 0–3M AUC 高于 UOT，且终局 640 局 win rate 不低于 UOT 的 95% CI 下界。
- **部分支持**：早期遗忘明显改善但后期上限没有；或后期上限改善但早期仍遗忘。
- **不支持**：早期和后期都不优于 UOT，或在线 arm 比例 <10%（说明是选择器饿死，不是 insight 失效，需要改选择器）。
- 单 seed 只作为 pilot；若方向成立，再扩 3 seed。

---

## 七、产物与路径

| 实验 | 脚本 | 输出 |
|---|---|---|
| H1+H2 | `scripts/verify_retention_insight.py` + `scripts/run_retention_insight.sh` | `logs/retention_insight.json`（分片在 `logs/retention_shards/`） |
| H3 | `scripts/analyze_mode_diversity.py` | `logs/mode_diversity.json` + `logs/mode_diversity.txt` |
| H4 | `training/pexm.py`、`scripts/train_pexm.py`、`scripts/eval_pexm_640.py`、`scripts/run_pexm_5m_vs_6m.sh` | `checkpoints/pexm_5m_vs_6m/online.jsonl`、`pexm_online_seed1.pt`、`pexm_selector_seed1.json`、`eval640_pexm_seed1.jsonl`、`eval640_uot_seed1.jsonl`、`eval640_ovmse_seed1.jsonl` |

所有 branch-replay 测量都是 `parameter_updates=0`。测量脚本默认 CPU；PEX-M 用 GPU 2（空卡），与现有实验隔离。

## 八、运行方式

```bash
# H1 + H2：按 seed 分片，nohup 后台
nohup bash scripts/run_retention_insight.sh > logs/retention_insight.out 2>&1 &
echo $! > logs/retention_insight.pid

# H3：模式多样性（依赖 H1 的教师 rollout，可独立跑）
nohup /home/jxy/miniconda3/envs/multiAgentRL/bin/python \
  scripts/analyze_mode_diversity.py \
  --config configs/5m_vs_6m.yaml \
  --policy-manifest scripts/mode_policies_5m_vs_6m.txt \
  --n-seeds 20 --device cpu \
  --out logs/mode_diversity.json > logs/mode_diversity.out 2>&1 &
echo $! > logs/mode_diversity.pid

# H4：PEX-M pilot（训练 + 终局 640 评估）
nohup bash scripts/run_pexm_5m_vs_6m.sh > logs/pexm_driver.out 2>&1 &
echo $! > logs/pexm_driver.pid
```

实验完成后用同一脚本收数（H1/H2 直接读 merge 结果，H4 读训练曲线与 640 终局）：

```bash
/home/jxy/miniconda3/envs/multiAgentRL/bin/python scripts/analyze_retention_results.py
```

## 九、结果解释框架

- **H1 成立、H2 成立**：说明“可行集 + 决策层打分”确实比全局排序可靠；下一步做 proposal generator（可以是 flow，也可以先用行为候选）与 calibrated selector。
- **H1 成立、H2 不成立**：说明单 agent 偏离的大效应可靠，但 K 候选内的精细选择仍不可靠；下一步重点应放在“候选生成的质量/多样性”和“选择器的校准”，而不是 Q̄ 排序本身。
- **H1 不成立**：说明连大效应单 agent 偏离都不可靠，Q̄ 只能当弱先验；应转向纯行为保留/策略蒸馏，不再把 Q 排序当知识。
- **H3 成立**：mode library/selector 是必要的；如果 H3 不成立，单一策略保留可能足够，算法可以更轻。
- **H4 成立**：行为保留 + 释放 critic 能同时解决早期遗忘和后期上限，可以进入完整算法设计；若不成立但早期改善，说明方向对但选择器/训练配比需要改。

## 十、局限

- 所有测量都在 5m_vs_6m/Medium 上；结论是否迁移到 3s5z、2s3z、连续动作任务未知。
- 成功轨迹数量受教师胜率限制；H1 的有效样本是轨迹数，不是 pair 数，CI 会较宽。
- branch replay 依赖 SMAC v1 的确定性；探针步 state 不 bit-exact 的分支会被丢弃并记录。
- H3 的 checkpoint 高度相关，且只有 3 个 seed；只做描述性判断。
- H4 是单 seed pilot，不能替代多种子显著性检验；`min_online=0.25` 是防止 arm 饿死的工程选择，会作为超参数一并报告。
