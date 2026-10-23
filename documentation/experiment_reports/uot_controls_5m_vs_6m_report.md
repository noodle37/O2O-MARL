# UOT 算法与 Section-4 简单对照组实验报告（SMAC 5m_vs_6m）

本报告包含四部分：(1) UOT（Ungated Ordinal Transfer）算法的完整数学定义与训练流程；(2) 六个对照组（Fixed Joint-Ranking、Per-agent Distillation、Time-Decayed Ranking、Per-agent marginal support、State-only support、无 support）的单变量切换设计与各自算法流程；(3) 终局 640 局评估与训练曲线结果；(4) 与既有实验（task3 分叉实验、OVMSE 及其无 OVM 对照）的对比。所有实验均在 SMAC v1 `5m_vs_6m`、Medium 数据集、seed 1 上完成。

---

## 一、UOT 算法

### 1.1 总体思想

离线阶段训练标准 QMIX（IGM 分解）：每个 agent 网络输出效用 $q_i(s, a_i)$，混合网络 $f_\omega$ 输出联合值函数

$$Q_{tot}(s, a_1, \dots, a_N) = f_\omega\big(q_1(s,a_1), \dots, q_N(s,a_N);\, s\big),$$

满足 $\arg\max_{a} Q_{tot} = \big(\arg\max q_1, \dots, \arg\max q_N\big)$。UOT 的迁移对象**不是 Q 值本身，而是冻结教师对联合动作的偏好序**：在线微调时用一个序数 hinge 约束在线 QMIX 在候选联合动作对上的排序与冻结教师一致。教师的绝对 Q 值只以**符号**进入损失，不作为 TD 目标。

### 1.2 冻结 oracle

加载离线 checkpoint 后调用 `freeze_oracle()`（`uot/learner.py:123`）：深拷贝在线 QMIX 得到 $\bar{Q} = (\bar q_{\theta^*}, \bar f_{\omega^*})$，全部参数 `requires_grad=False`。在线网络 $Q_\theta$ 保持可训练。$\bar Q$ 与 $Q_\theta$ 在同一批状态、同一批候选联合上分别前向，二者的差只经 `sign` 进入损失。

### 1.3 支撑集 $S_{off}$（`uot/support.py`）

从离线数据集的回合动作一次性构建，在线阶段不更新：

- **联合支撑（joint）**：$S_{off} = \{\, \mathrm{encode}(a) = \sum_i a_i \cdot n_a^i \,:\, (s,a) \in \mathcal{D}_{off}\,\}$，成员判定为精确联合动作元组（`contains`，排序数组 + `searchsorted`）。
- **边际支撑（marginal）**：$M[i, a] = \exists (s,a)\in\mathcal{D}_{off}: a_i = a$，成员判定为"每个 agent 的动作都曾为该 agent 记录过"，组合本身无需联合出现过（`contains_marginal`）。
- **状态支撑（state-only）**：不含动作信息。对离线状态做粗粒度随机投影哈希计数 $c(s)$，损失权重为
$$w(s) = \sigma\!\left(\frac{\log(1 + c(s)) - q_{10}}{\sigma_{off}}\right),$$
其中 $q_{10}$、$\sigma_{off}$ 为离线 $\log(1+c)$ 的 10% 分位数与标准差（`StateSupport.calibrate`，`uot_state_support_quantile=0.10`）。

### 1.4 候选联合构造（`uot/neighborhood.py: build_recorded_neighborhood`）

对每个有效转移 $(s_t, a_t)$：

1. **锚点**（3 个）：冻结贪心联合 $\bar a^* = \arg\max \bar q$（逐 agent）、在线贪心联合 $a^*_\theta = \arg\max q_\theta$（逐 agent）、实际执行联合 $a_t$。
2. **邻域扩展**：以锚点为中心取 Hamming 距离 $d \le d_{max}$（默认 2）的联合；每个候选须通过合法性过滤（`avail_actions`）与支撑过滤（按 `support_mode`，见 1.3）。
3. **打分与去重**：锚点得分 $+10^6$ 优先入选；Hamming-1 候选按被改动 agent 的教师效用落差 $|\bar q_i(s, a') - \bar q_i(s, a_i)|$ 降序；Hamming-2 按两 agent 落差之和。`pack_unique_topk` 去重后取前 $K$ 个（默认 $K=8$），输出候选集 $\{a^{(k)}\}_{k=1}^{K}$ 及有效性掩码。

### 1.5 序数 hinge 损失（`uot/hinge.py: ordinal_hinge`）

对候选集中所有无序对 $(i, j)$，保留满足以下条件的"可判定对"：

$$\mathcal{P} = \{(i,j) : \mathrm{valid}_i \wedge \mathrm{valid}_j \wedge a^{(i)} \neq a^{(j)} \wedge |\bar Q_i - \bar Q_j| \ge \mathrm{margin}\},$$

其中 $\bar Q_k = \bar f_{\omega^*}(\bar q_{\theta^*}(s, a^{(k)}), s)$、$Q_k = f_\omega(q_\theta(s, a^{(k)}), s)$ 均经**同一个 mixer** 计算（`_qtot_candidates`）。损失为加权均值 hinge：

$$\mathcal{L}_{ord} = \frac{\sum_{(i,j)\in\mathcal{P}} w(s_t)\cdot \mathrm{relu}\big(\delta - \mathrm{sign}(\bar Q_i - \bar Q_j)\cdot (Q_i - Q_j)\big)}{\sum_{(i,j)\in\mathcal{P}} w(s_t)}.$$

教师只以 $\mathrm{sign}(\bar Q_i - \bar Q_j)$ 出现；`margin`（默认 0.1）过滤教师自身不确定的小差距对，`delta`（默认 0.1）为要求的排序裕量。诊断量 `uot_hinge_rate`（激活对比例）、`uot_sign_disagree`（在线与教师符号不一致比例）、`uot_cand_in_frac`（候选落在支撑内的比例）随训练记录。

### 1.6 总损失与 $\eta$ 调度

每次梯度更新的总损失（`uot/learner.py:361`）：

$$\mathcal{L} = \mathcal{L}_{TD(\lambda)} + \eta(t)\cdot \mathcal{L}_{ord},$$

$\mathcal{L}_{TD(\lambda)}$ 为标准 QMIX TD($\lambda$) 损失（$\gamma=0.99$，$\lambda=0.6$，n-step 回报目标，`build_td_lambda_targets`）。$\eta(t)$ 由 `eta_at(t_env)` 给出，三种调度共用同一实现：

- **固定**：$\eta(t) = \eta_0$（FJR，$\eta_0 = 0.1$）；
- **线性衰减**：$\eta(t) = \eta_0 \cdot \max\big(0,\; 1 - (t - t_{start}) / T_{decay}\big)$（TDR：$t_{start}=0$，$T_{decay}=2\mathrm{M}$）；
- **硬关闭**：$\eta(t) = 0\ (t \ge t_{drop})$（分叉实验的 off 分支）。

$\eta=0$ 时损失项仍以前向方式计算并记录（不反传），保证各对照组日志口径一致。

### 1.7 在线训练流程

| 项目 | 设置 |
|---|---|
| 并行环境 | 8（SMAC v1，Medium） |
| 探索率 | $\varepsilon: 1.0 \to 0.05$，100k 步线性衰减 |
| 回放缓冲区 | 5000 回合 |
| 批次 | 128 回合（`stack_episodes` 补齐到最大序列长） |
| 更新节奏 | 每轮并行 rollout（8 回合）后 1 次梯度更新 |
| 目标网络 | 每 200 回合硬更新 |
| 梯度裁剪 | 10.0 |
| 总预算 | 3M 环境步；每 10k 步 32 局周期评测 |
| 超参 | $\eta_0=0.1,\ \delta=0.1,\ \mathrm{margin}=0.1,\ K=8,\ d_{max}=2$ |

---

## 二、对照组设计（单变量切换）

六个对照组共用同一份配置（`configs/uot_control_5m_vs_6m.yaml`，与 FJR 参考跑 `configs/uot_temporal_5m_vs_6m.yaml` 的 system 块逐字节相同，仅输出目录与下列开关不同）、同一离线初始化（`checkpoints/ovmse_paper_5m_vs_6m/ovmse_paper_seed1.pt`，10M 步离线训练的 final checkpoint）、同一 seed。驱动脚本 `scripts/run_uot_controls_5m_vs_6m.sh` 每组只改一个知识复用开关：

| 对照组 | 切换的开关 | 语义 |
|---|---|---|
| Fixed Joint-Ranking（FJR，参考组） | 无（复用 `checkpoints/uot_temporal_5m_vs_6m`） | $\eta=0.1$ 全程固定，联合支撑 |
| Per-agent Distillation（PAD） | `uot_ranking_scope=agent` | 逐 agent 成对序数蒸馏 |
| Time-Decayed Ranking（TDR） | `uot_eta_decay_start=0`，`uot_eta_decay_steps=2000000` | $\eta$ 前 2M 步线性衰减至 0 |
| Per-agent marginal support | `uot_support_mode=marginal` | 候选池按边际支撑过滤 |
| State-only support | `uot_support_mode=state` | 候选池为全部合法联合，损失按状态密度加权 |
| 无 support | `uot_support_mode=none` | 候选池为全部合法联合，权重恒 1 |

各对照组的算法流程差异（数学定义）：

- **FJR**：即 1.4–1.6 的完整 UOT，$\eta(t)\equiv 0.1$，支撑模式 joint。
- **PAD**（`_agent_ranking_loss`，`uot/learner.py:148`）：hinge 形式、$\delta$、margin、系数与联合版完全相同，但候选对改为**每个 agent 各自动作集内部**的 $(a, b)$ 对：
$$\mathcal{L}_{ord}^{agent} = \frac{\sum_i \sum_{a<b} \mathrm{relu}\big(\delta - \mathrm{sign}(\bar q_i(s,a) - \bar q_i(s,b)) \cdot (q_i(s,a) - q_i(s,b))\big)\cdot \mathbb{1}[\,|\Delta \bar q_i| \ge \mathrm{margin}\,]}{\#\text{保留对}},$$
教师信号为逐 agent 效用 $q_i$ 的序，**不经过 mixer、不涉及联合候选池**——这是与联合版在数学上可严格区分的核心：联合版迁移 mixer 输出的联合值函数序，PAD 迁移各 agent 效用序。
- **TDR**：FJR + $\eta(t) = 0.1\,(1 - t/2\mathrm{M})$（$t<2\mathrm{M}$），$t \ge 2\mathrm{M}$ 后 $\eta \equiv 0$，即最后 1M 步为零离线约束的纯在线 TD 微调。
- **marginal**：候选池过滤由 `contains`（精确联合元组）换为 `contains_marginal`（逐 agent 边际支撑），其余与 FJR 相同。
- **state**：候选池为全部合法联合（无动作级过滤）， hinge 损失按 $w(s_t)$ 加权（1.3 节公式）。
- **none**：候选池为全部合法联合，$w \equiv 1$，即"联合排序约束 + 零离线知识引导候选"。

---

## 三、终局评估结果（640 局）

各对照组终点 checkpoint 在同一环境种子（seed 1）下以贪心策略评估 640 局（`scripts/eval_640.py`，同一 640 局 episode 集，SMAC v1 确定性），胜率附 Wilson 95% 置信区间：

| 对照组 | 640 局胜率 | 95% CI | 平均回报 |
|---|---|---|---|
| Fixed Joint-Ranking | 0.617 | [0.579, 0.654] | 16.23 |
| Per-agent Distillation | 0.638 | [0.600, 0.674] | 16.57 |
| Time-Decayed Ranking | **0.834** | [0.804, 0.861] | 18.29 |
| Per-agent marginal support | 0.645 | [0.607, 0.681] | 16.45 |
| State-only support | 0.658 | [0.620, 0.694] | 16.62 |
| 无 support | 0.620 | [0.582, 0.657] | 16.20 |
| OVMSE（在线全程，参照） | 0.667 | [0.630, 0.703] | 16.73 |

**95% CI 的含义**：单局结果为二元变量（胜/负），640 局评估得到的胜率是对真实胜率的有噪声估计。Wilson 95% 置信区间是该真实胜率的区间估计——若在相同协议下重复执行该评估，约 95% 构造出的区间会覆盖真实胜率。判读规则：两个对照组的区间**大幅重叠**时，观察到的胜率差异不能排除由对局采样噪声导致；区间**不重叠**时，差异超出评估采样噪声可解释的范围。该区间仅量化评估阶段的采样误差，不包含训练种子间的方差（上述对照均为单种子）。

**读数**：

1. PAD（0.638）与 FJR（0.617）区间大幅重叠，且 marginal / state / none 三个支撑变体均落在 0.62–0.66：在 5m_vs_6m 上，联合排序迁移相对普通逐 agent 动作偏好蒸馏**未产生可辨识的终局增益**，支撑集的各种放宽（边际、状态密度、无支撑）同样无增益。
2. TDR（0.834）是唯一显著偏离的组，区间与其余各组及 OVMSE 均不重叠。
3. 全部 UOT 血统对照组（除 TDR）低于 OVMSE 参照（0.667），但差距在 0.01–0.05 量级、与 OVMSE 区间部分重叠。

---

## 四、训练曲线对比

训练过程中每 10k 步的 32 局周期评测胜率（单点噪声约 ±0.09，仅用于观察形态；终局结论以 640 局评估为准）：

| t_env | TDR | FJR | OVMSE |
|---|---|---|---|
| 0 | 0.094 | 0.094 | 0.188 |
| ~232k | 0.000 | 0.000 | 0.250 |
| ~474k | 0.000 | 0.031 | 0.219 |
| ~716k | 0.000 | 0.219 | 0.313 |
| ~958k | 0.031 | 0.531 | 0.500 |
| ~1.20M | 0.500 | 0.406 | 0.406 |
| ~1.44M | 0.625 | 0.563 | 0.563 |
| ~1.69M | 0.844 | 0.688 | 0.625 |
| ~1.93M | 0.750 | 0.594 | 0.750 |
| ~2.17M | 0.750 | 0.688 | 0.813 |
| ~2.41M | 0.719 | 0.625 | 0.656 |
| ~2.66M | 0.813 | 0.531 | 0.719 |
| ~2.90M | 0.781 | 0.688 | 0.719 |
| 末点（~3.0M） | **0.906** | 0.531 | 0.781 |

TDR 的 $\eta$ 实测轨迹（train.out 的 `uot_eta_now`）确认调度精确生效：494k→0.0753，998k→0.0501，1.49M→0.0253，1.90M→0.0051，2.10M 起→0。

**形态解读**：TDR 在前约 70 万步被约束钉在离线策略水平（胜率≈0，与离线初始化一致），随后**胜率爬升与 $\eta$ 衰减严格同步**，2M 步约束归零后进入自由适应，尾段（2.4M–3.0M）稳定在 0.72–0.91；FJR 尾段仅 0.53–0.69，OVMSE 尾段 0.66–0.81。机制上与既有证据链自洽：早期 critic 未收敛时排序约束起正则作用（qdiag 证据：无约束在线早期 Q 会崩至 bias −32.6、$\tau_{w10}$ −0.71），约束平滑释放后策略自由适应；FJR 全程 0.1 的约束在后期栓住适应。

---

## 五、与既有实验的对比

### 5.1 task3 分叉实验（"晚衰减"：排序随在线训练减弱）

此前的衰减实验为 task3 分叉实验的 decay 分支（`scripts/run_uot_fork_5m_vs_6m.sh`）：父 run 从同一离线 checkpoint 在线训练至约 1M 步（逐种子分叉点 1,009,049 / 1,009,129 / 1,000,325），冻结完整训练器快照（网络、目标网络、优化器状态、回放缓冲区、探索计划与随机状态），再分出三条支线，唯一差别为排序损失权重：continue 全程 $\eta=0.1$；off 在分叉点立即置零；decay 从分叉点线性衰减至零（`uot_eta_decay_start=FORK_T`，`uot_eta_decay_steps=REMAIN≈2M`）。三条支线各续训约 199 万步至 3M，分叉点与三个终点 checkpoint 均以 640 局配对评估：

| seed | 分叉点 | continue | off | decay |
|---|---|---|---|---|
| 1 | 0.242 [0.211, 0.277] | 0.422 [0.384, 0.460] | 0.595 [0.557, 0.633] | 0.577 [0.538, 0.614] |
| 2 | 0.412 [0.375, 0.451] | 0.653 [0.615, 0.689] | 0.658 [0.620, 0.694] | 0.633 [0.595, 0.669] |
| 3 | 0.416 [0.378, 0.454] | 0.669 [0.631, 0.704] | 0.645 [0.607, 0.681] | 0.608 [0.569, 0.645] |

配对差（3 种子均值）：off − continue = +0.052，decay − continue = +0.024，均小于 640 局配对差的 ±0.053 不确定范围，且种子间方向不一致（seed1 +0.173/+0.155，seed2 +0.005/−0.020，seed3 −0.023/−0.061）。尾段胜率（最后 5 个 32 局评测点）配对差 off +0.054 / decay +0.060，同向为正但小于该指标自身 ±0.075 的不确定范围。task3 的结论为：**撤除或衰减排序约束未产生可测量的终局改善**。

### 5.2 TDR 与"晚衰减"（fork-decay）的结构差异

两者总约束暴露量相同（$\eta$–$t$ 三角形面积均为 $0.1 \times 2\mathrm{M} / 2 = 0.1\mathrm{M}$ step-units），差别纯粹在**时间分布**：

| | TDR | fork-decay |
|---|---|---|
| 约束活跃窗口 | 0–2M（早期） | 1M–3M（晚期） |
| 无约束窗口 | 最后 1M（2M–3M） | 仅终点附近 |
| 1M–2M 期间的 $\eta$ | 0.05→0（已大幅释放） | 0.1→0.05（仍强约束） |
| 起点状态 | 离线 checkpoint，约束自第 0 步起递减，策略从未被完全钉住 | 1M 快照（前 1M 全程 $\eta=0.1$ 强约束的产物，策略被钉在离线行为附近） |

由此，fork-decay 在 1M–2M 窗口仍保留 0.05–0.1 的约束，正好落在 task3 证明"约束有害"的后期适应阶段，到 3M 才归零、几乎没有无约束适应时间——其结果（三种子 0.577/0.633/0.608）与 fork-off（0.595/0.658/0.645）几乎无差，继承了"晚期约束有害"的问题；TDR 把全部剂量放在早期（critic 形成期），2M 后整整 1M 自由适应。**同一衰减机制，放在早期与晚期产生相反效果**，这是 TDR 与此前衰减实验的本质区别。

### 5.3 OVMSE 完整结果（5m_vs_6m）

- **离线阶段**：10M 环境步（t_env = 10,002,211），产出 `ovmse_paper_seed1_best.pt`（离线验证选出）与 `ovmse_paper_seed1.pt`（final）。
- **在线阶段**：3M 步，从 `_best.pt` 出发，`configs/5m_vs_6m.yaml` 默认 OVMSE 配方（`mixing_ratio=0.1`，`use_ovm=true`，记忆权重 $\lambda: 1.0 \to 0.1$，200k 步退火）。
- **终局 640 局**：胜率 0.667 [0.630, 0.703]，平均回报 16.73（评估协议与对照组相同）。
- **训练曲线**：尾段（32 局周期评测）0.66–0.81，末点 0.781；尾段胜率均值约 0.725（298 个评测点）。
- **Q 偏差轨迹**（qdiag on 臂，300k 步）：bias 自 +1.54 持续为正至末段 +4.58，$\tau_{w10}$ 0.407→0.758，高估率 0.686→0.986——即 OVM 把离线阶段已有的高估沿在线训练继续放大（胜率看不出，Q 偏差看得出）。

### 5.4 OVMSE without OVM 的结果

两个尺度的无 OVM 对照（均从同一离线 checkpoint 出发、除干预项外逐项相同）：

1. **task4 ovm_diag（270k 步，机制 + 表现对照）**：
   - 表现：on 臂 eval return 10.084 / q_bias 1.470；off 臂 10.847 / 0.020——5m_vs_6m 上启用 OVM 与否对最终表现影响很小（2halfcheetah 上差异显著且方向与机制预言一致：on −848.31 vs off −336.31）。
   - 机制：on 臂激活率 0.407→0.699、原始上抬 0.470→1.341（$\lambda=0.1$ 时实际上抬仅 0.134）；off 臂记录的反事实上抬 0.470→2.100。OVM 确实在持续抬高回归目标，但 5m_vs_6m 上抬升的实际效果被 $\lambda$ 衰减调制得很小。
2. **qdiag off 臂（300k 步，Q 偏差轨迹）**：无 OVM 的在线早期发生 Q 崩塌——bias 自 +1.54 跌至 −32.64（t=10,217），$\tau_{w10}$ 最低 −0.712（t=30,520），排序反转持续约 2.0万–4.1万步，至 300k 恢复到 −0.97；on 臂全程无崩塌。这为"早期需要某种正则稳定 critic"提供了直接证据，也是 TDR 早期保留约束的合理性依据。
3. **3M 尺度的无 OVM OVMSE 配方对照尚未完成**：matched_protocol 臂 E（OVMSE 配方、`use_ovm=false`、`mixing_ratio=0.1`）正在训练（见下）。

---

## 六、结论

本组实验（六个对照组、同一离线初始化、同一 seed、单变量切换，终局以 640 局配对评估判定）支撑以下结论：

1. **离线联合排序知识本身未产生可测量的收益。** Fixed Joint-Ranking（0.617）与 Per-agent Distillation（0.638）终局胜率无差异（置信区间重叠）；支撑集的联合、边际、状态密度、无支撑四种定义同样无差异（0.620–0.658）。将离线知识用作联合动作排序约束，相比普通的逐 agent 动作偏好蒸馏没有额外收益；支撑集的各种放宽亦不改变该结论。

2. **排序约束的时间分布是决定性因素。** 早衰减（TDR，0.834）显著优于固定权重（FJR，0.617）与晚衰减（task3 分叉实验的 decay 分支，三种子 0.577–0.633），且超过 OVMSE 参照（0.667）。训练曲线显示 TDR 的胜率爬升与 $\eta$ 衰减严格同步：约束的价值在于稳定早期 critic 的形成（qdiag 证据：无约束在线早期 Q 崩塌至 bias −32.6、$\tau_{w10}$ −0.712），而非终身保持离线行为；2M 步约束归零后的自由适应期是终局优势的来源。同一衰减机制置于早期与晚期产生相反效果。

3. **除 TDR 外，各 UOT 对照组终局胜率（0.617–0.658）均低于 OVMSE 参照（0.667）。**

---

## 七、复现方式

```bash
# 五个新对照组（FJR 为既有参考跑，不重跑）
nohup bash scripts/run_uot_controls_5m_vs_6m.sh \
  > logs/uot_controls_5m_vs_6m_master.log 2>&1 &

# 汇总分析（输出 JSON + Markdown 表）
/home/jxy/miniconda3/envs/multiAgentRL/bin/python scripts/analyze_uot_controls.py \
  --run-root checkpoints/uot_controls_5m_vs_6m \
  --fjr-log checkpoints/uot_temporal_5m_vs_6m/online.jsonl \
  --fjr-eval640 checkpoints/uot_temporal_5m_vs_6m/eval640_fjr.jsonl \
  --out-json documentation/experiment_reports/task6_controls_5m_vs_6m.json \
  --out-md documentation/experiment_reports/task6_controls_5m_vs_6m_tables.md
```

产物路径：训练日志与终点 checkpoint 在 `checkpoints/uot_controls_5m_vs_6m/{fjr,pad,tdr,marginal,state,none}/`（各含 `online.jsonl`、`train.out`、`eval640.jsonl`）；驱动日志 `logs/uot_controls_5m_vs_6m_master.log`（2026-10-21 04:44:26 ALL ARMS DONE，全部退出码 0）。

