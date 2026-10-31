> 注：FHJP 算法及其专属报告已于 2026-10-31 删除。本文保留原正文作为历史记录，其中 `fhjp/` 代码路径、`fhjp_*.md` 报告与 FHJP 训练 checkpoint 均已不存在。

# FHJP / UOT / OVMSE：算法细节、实验臂设置与对比结果（详版）

> 本文是 `fhjp_vs_ovmse_uot.md`（自动生成的汇总表）的配套详版文档：先讲清三个算法各自做了什么、公式是什么、代码落在哪里，再逐个说明本次实验里每个臂的具体配置与数据流，最后给结果。
> 所有公式都与仓库里的实现一一对应，代码位置以 `文件::符号` 标注。

## 0. 阅读地图

| 节 | 内容 |
|---|---|
| 1 | 三个算法共同的基础：QMIX、TD(λ) 目标、在线训练循环 |
| 2 | FHJP：联合行为提议 + 版本末双在轨评估臂 + 不可逆退出 |
| 3 | UOT：冻结 oracle 的序数 hinge 蒸馏 |
| 4 | OVMSE：离线价值记忆（OVM）目标 + 离线回放混合 |
| 5 | 实验臂：每个臂具体打开了哪些开关、跑了什么配置 |
| 6 | 结果：640 集终点、统计对比、提议器诊断、训练曲线 |
| 7 | 复现方式 |

---

## 1. 共同基础

三个臂都是 QMIX（值分解 + 单调混合），区别只在于**离线知识以什么形式进入在线更新**。

### 1.1 网络与混合

每个智能体 $i$ 用 GRU 产生 $Q_i(h_t,a)$，输入为（观测，上一步动作 one-hot，agent id）。团队价值由单调混合网络给出：

$$
Q_{\mathrm{tot}}(h_t,\mathbf a_t)=\mathrm{Mixer}\big(Q_1(h_t,a_t^1),\dots,Q_N(h_t,a_t^N);s_t\big)
$$

Mixer 对每个 $Q_i$ 单调，因此**逐 agent 取 argmax 就是联合意义下的贪心**：

$$
\mathbf a_t^{\mathrm{greedy}}=\Big(\arg\max_{a\in\mathcal A_i}Q_i(h_t,a)\Big)_{i=1}^{N}
$$

这条性质是 FHJP「不需要在联合动作空间搜索」和 UOT「用逐 agent 贪心当锚点」的共同依据。

### 1.2 TD(λ) 目标

三个算法用的都是同一个 λ-回报目标（`ovmse/targets.py::build_td_lambda_targets`）：

$$
G_t=\lambda\gamma G_{t+1}+(1-\lambda)\gamma \max_{\mathbf a}Q_{\mathrm{tot}}(h_{t+1},\mathbf a;\theta^-)
$$

其中折扣 $\gamma=0.99$、$\lambda=0.6$，目标动作由目标网络 $\theta^-$ 逐 agent 贪心给出。损失是掩码均方误差：

$$
\mathcal L_{\mathrm{TD}}=\frac{\sum_t m_t\,\big(Q_{\mathrm{tot}}(h_t,\mathbf a_t;\theta)-G_t\big)^2}{\sum_t m_t}
$$

$m_t$ 是 episode 填充掩码（`training/online.py` 中每个并行 rollout 完成后做一次梯度步）。

### 1.3 在线训练循环（三臂完全一致）

| 项目 | 5m_vs_6m | 2s3z |
|---|---|---|
| 在线预算 `t_max` | 3,000,000 环境步 | 1,000,000 环境步 |
| 在线 replay `buffer_size` | 5000 条 episode | 5000 条 episode |
| `batch_size` / `batch_size_run` | 128 / 8 | 128 / 2 |
| 学习率 / `td_lambda` / `discount` | 1e-3 / 0.6 / 0.99 | 1e-3 / 0.6 / 0.99 |
| `target_update_interval` | 200 episode | 50 episode |
| `grad_norm_clip` | 10.0 | 1.0 |
| 探索 ε | 1.0 → 0.05，100k 步线性衰减 | 同 |
| 训练中评估 | 每 10k 步 32 集 | 同 |
| 起点 | `checkpoints/ovmse_batch_20261014/<map>_offline/ovmse_paper_seed1_best.pt`（连同 Adam 优化器状态一起恢复） | 同 |

离线数据集是 cfcql **Medium**（`task.dataset=Medium`）。除 OVMSE 的 mixing 臂外，离线 episode **不进入**在线 replay。

---

## 2. FHJP（Finite Handover of Joint Proposals）

规范见 `documentation/joint_proposal_handover.md`，实现见 `fhjp/proposer.py`、`fhjp/eval_arms.py`、`fhjp/controller.py`。

### 2.1 决策单位与两个联合候选

最小决策单位是**完整联合动作**，不是逐 agent 贡献。每个决策点取两个候选：

$$
\mathbf a_t^{\mathrm{off}}=\Big(\arg\max_{a}Q_i^{\mathrm{off}}(h_t,a)\Big)_{i=1}^{N},
\qquad
\mathbf a_t^{\mathrm{on}}=\Big(\arg\max_{a}Q_i^{\mathrm{on}}(h_t,a;\theta)\Big)_{i=1}^{N}
$$

离线 Q 网络只用来**排序**（决定动作），其数值不进入任何训练目标。两条历史故意不对称（`fhjp/proposer.py` 模块 docstring）：

- 在线网络以**已执行动作**为历史——这是可部署策略真实面对的历史；
- 离线网络以**它自己上一步的贪心动作**为历史，即 $\pi^{\mathrm{off}}$ 的闭环，与离线评估臂一致。

### 2.2 冲突处的 0.5/0.5 随机执行

定义冲突指示

$$
\delta_t=\mathbb 1\big[\mathbf a_t^{\mathrm{off}}\neq \mathbf a_t^{\mathrm{on}}\big]
$$

$\delta_t=0$ 时两个候选相同，直接执行；$\delta_t=1$ 时以固定概率 $p=0.5$ 二选一：

$$
\mathbf a_t^{c}=\begin{cases}\mathbf a_t^{\mathrm{off}},&z_t=1\\ \mathbf a_t^{\mathrm{on}},&z_t=0\end{cases},
\qquad z_t\sim\mathrm{Bernoulli}(p),\ p=0.5
$$

随后照常施加原有 ε-greedy 探索得到最终执行动作。$p$ **不是可学习权重**，而是随机化比较协议：两个联合候选的被选概率已知且相等，在线联合行为始终作为对照分支存在。

### 2.3 普通在线 TD 更新

新采集轨迹只进在线 replay，执行**原版 QMIX TD 损失**（1.2 节）。整个算法中不存在：

$$
Q^{\mathrm{off}}\ \text{进入 TD 目标},\qquad
\lambda\mathcal L_{\mathrm{offline}},\qquad
\text{离线数据优先回放},\qquad
\text{pairwise ranking loss}
$$

也**不存在重要性修正**：QMIX 的 TD 更新本来就是离策略更新，$0.5/0.5$ 提议随机化产生的转移可以直接进普通 TD 批次。每条步只记录两个诊断量 `source∈{none,offline,online}` 与 `conflict=δ_t`，只进日志，不进 replay、不进损失。

### 2.4 版本末的双在轨评估臂

每个版本 $v$ 结束时**冻结**在线参数 $\theta$，用独立评估 RNG（`default_rng(eval_seed_base+v)`，`eval_seed_base=900001`）生成 $M$ 个互相独立的种子 $s_m$。对每个种子分别新建环境、从初始 RNN 隐状态闭环跑完整一局（$\varepsilon=0$，不施加提议随机化，不写 replay，不做梯度更新）：

$$
G_m^{\mathrm{off}}=\sum_{t}\gamma^t r_{m,t}^{\mathrm{off}}\ (\pi^{\mathrm{off}}\ \text{闭环}),
\qquad
G_m^{\mathrm{on}}=\sum_{t}\gamma^t r_{m,t}^{\mathrm{on}}\ (\pi_k^{\mathrm{greedy}}\ \text{闭环})
$$

配对差与版本统计量：

$$
\Delta_m=G_m^{\mathrm{off}}-G_m^{\mathrm{on}},\qquad
\widehat\tau_v=\frac1M\sum_m\Delta_m,\qquad
s_v^2=\frac{1}{M-1}\sum_m(\Delta_m-\widehat\tau_v)^2
$$

两条臂各自在**自己的目标策略轨迹分布**上采样，两个单臂估计分别无偏，作差仍无偏；共享种子只降低配对差的方差。每条评估轨迹计入总环境交互预算（`count_eval_in_budget: true`）。

### 2.5 退出判据（不可逆）

$$
\text{退出}\iff \Big(M\ge M_{\min}\Big)\ \wedge\
\Big(\underbrace{\widehat\tau_v+z_{1-\alpha}\,s_v/\sqrt M}_{\text{单侧 95\% 上界}}\le 0\Big),
\qquad \alpha=0.05
$$

即离线联合行为的剩余价值上界已不大于 0 且样本量足够时，**永久**关闭提议器；之后离线网络不再被读取，训练方程与纯在线 MARL 逐式相同，且不再重新开启。每个版本的 $\widehat\tau_v$ 只由该版本边界新采集的 $2M$ 条轨迹决定，不累计历史版本。实现在 `fhjp/eval_arms.py::should_exit`。

### 2.6 本次运行的 FHJP 参数

| 参数 | 5m_vs_6m | 2s3z |
|---|---|---|
| `proposer.p` | 0.5 | 0.5 |
| `proposer.version_len` | 500,000 步 | 200,000 步 |
| `proposer.eval_episodes_per_arm`（$M$） | 32 | 32 |
| `proposer.min_eval_episodes`（$M_{\min}$） | 16 | 16 |
| `proposer.alpha` | 0.05 | 0.05 |
| `proposer.eval_seed_base` | 900001 | 900001 |
| `proposer.count_eval_in_budget` | true | true |
| 离线项 | `cql_weight=0`、`online_cql_weight=0`、`mixing_ratio=0`、`use_ovm=false` | 同 |

配置文件：`configs/fhjp_5m_vs_6m.yaml`、`configs/fhjp_2s3z.yaml`。

---

## 3. UOT（Ungated Ordinal Transfer）

实现：`uot/learner.py`、`uot/hinge.py`、`uot/neighborhood.py`、`uot/support.py`。

### 3.1 冻结 oracle

在线开始时把**加载进来的那份 QMIX**（即离线初始化权重）深拷贝并冻结，得到 $\bar Q$（`uot/learner.py::freeze_oracle`，在 `training/online.py` 载入 checkpoint 后调用）。$\bar Q$ 只被查询，**从不进入 Bellman 目标、从不选动作**：

$$
\bar Q\text{ 只以 }\mathrm{sign}\big(\bar Q_{\mathrm{tot}}(\mathbf c_i)-\bar Q_{\mathrm{tot}}(\mathbf c_j)\big)\text{ 的形式进入损失}
$$

同时从离线数据集动作里构建冻结的支持集 $S_{\mathrm{off}}$（`uot/support.py::RecordedJoints.from_episode_actions`，`training/factory.py`）：记录出现过的联合动作元组编码，成员判定只在动作元组上，在线数据不新增键。

### 3.2 候选池：记录邻域

每个填充步取三个锚点（`uot/neighborhood.py::build_recorded_neighborhood`）：

$$
\text{anchors}=\{\mathbf a^{\bar g}_{\text{冻结贪心}},\ \mathbf a^{g}_{\text{在线贪心}},\ \mathbf a^{\text{exec}}_{\text{已执行}}\}
$$

候选是 $S_{\mathrm{off}}$ 中与任一锚点 Hamming 距离 $\le d_{\max}$（本次 $d_{\max}=2$）的联合动作，且必须当前合法。超过 $K$（本次 $K=8$）个时按「距离小者优先，再按被翻转 agent 上的冻结逐 agent Q 差距大者优先」取前 $K$ 个唯一联合动作。该差距只决定哪些联合动作进入额外求和，损失本身仍用 $\bar Q_{\mathrm{tot}}$ 与 $Q_{\mathrm{tot}}$。

### 3.3 序数 hinge 损失

对候选集合中所有无序对 $(i,j)$（`uot/hinge.py::ordinal_hinge`）：

$$
\text{keep}(i,j)=\Big[\mathbf c_i\neq \mathbf c_j\Big]\wedge\big[\bar Q_{\mathrm{tot}}(\mathbf c_i)-\bar Q_{\mathrm{tot}}(\mathbf c_j)\big|\ge \mu\big]\wedge\text{合法}\wedge\text{有效}
$$

$$
\mathcal L_{\mathrm{ord}}=
\frac{\sum_{(i,j)\in\mathrm{keep}}\mathrm{relu}\Big(\delta-\mathrm{sign}\big(\bar Q_{\mathrm{tot}}(\mathbf c_i)-\bar Q_{\mathrm{tot}}(\mathbf c_j)\big)\cdot\big(Q_{\mathrm{tot}}(\mathbf c_i)-Q_{\mathrm{tot}}(\mathbf c_j)\big)\Big)}
{\sum_{(i,j)\in\mathrm{keep}}1}
$$

即：把冻结教师认为更优的候选，要求在线 $Q_{\mathrm{tot}}$ 至少高出 $\delta$。本次 $\mu=0.1$（margin）、$\delta=0.1$。只有教师差距 $\ge\mu$ 的对才参与，因此传递的是**排序**而非数值。

### 3.4 总损失与权重

$$
\mathcal L=\mathcal L_{\mathrm{TD}}+\eta(t)\,\mathcal L_{\mathrm{ord}}
$$

$\eta$ 支持三种调度（`uot/learner.py::eta_at`）：固定值、`eta_drop_at` 硬关闭、`eta_decay_start + eta_decay_steps` 线性衰减到 0。本次匹配臂 A 用固定 $\eta=0.1$；臂 B 令 $\eta=0$，即该项完全关闭（此时 UOT 退化为普通在线 QMIX TD）。诊断量 `uot_hinge_rate`、`uot_sign_disagree`、`uot_cand_in_frac` 只进日志。

---

## 4. OVMSE

实现：`ovmse/ovm.py`（OVM 混合）、`ovmse/learner.py`（训练步）、`ovmse/cql.py`（CQL）、`training/online.py` + `training/rollout.py::merge_offline_online`（离线回放混合）。

### 4.1 冻结离线 Q

载入离线 checkpoint 得到冻结的 `offline_q_network` + `offline_mixer`（`ovmse/ovm.py::load_offline_for_ovm`）。对批次中**已执行动作**计算离线团队价值：

$$
Q^{\mathrm{off}}_{\mathrm{tot}}(h_t,\mathbf a_t)=\mathrm{Mixer}^{\mathrm{off}}\Big(\big[Q_i^{\mathrm{off}}(h_t,a_t^i)\big]_i\Big)
$$

注意这里离线 RNN 是以已执行动作展开的反事实展开（诊断口径），离线数值只用于构造 OVM 目标。

### 4.2 OVM 目标与混合 TD 损失

开启 OVM 时（`ovmse/ovm.py::_ovm_td_loss`）：

$$
\tilde y_t=\max\Big(Q^{\mathrm{off}}_{\mathrm{tot}}(h_t,\mathbf a_t),\ G_t\Big),
\qquad
\mathcal L_{\mathrm{OVM}}=
\frac{\sum_t m_t\Big[(1-\lambda^{\mathrm{mem}}_t)\big(Q_{\mathrm{tot}}(h_t,\mathbf a_t)-G_t\big)^2+\lambda^{\mathrm{mem}}_t\big(Q_{\mathrm{tot}}(h_t,\mathbf a_t)-\tilde y_t\big)^2\Big]}{\sum_t m_t}
$$

$\lambda^{\mathrm{mem}}_t$ 由 1.0 线性退火到 `lambda_memory_end`（`ovmse/ovm.py::get_lambda_memory`）：

$$
\lambda^{\mathrm{mem}}_t=\max\Big(\lambda_{\mathrm{end}},\ 1-\frac{1-\lambda_{\mathrm{end}}}{T_{\mathrm{anneal}}}\,t\Big)
$$

本次匹配臂 $\lambda_{\mathrm{end}}=0.1$、$T_{\mathrm{anneal}}=200{,}000$。也就是说：训练初期回归目标被离线价值**抬高**（$\tilde y_t\ge G_t$），随后权重逐退火到 0.1 并保持。关闭 OVM 时损失就是 1.2 节的普通 TD 损失。

### 4.3 离线回放混合（mixing_ratio）

$\rho$（`mixing_ratio`）非零时（`training/online.py` 第 537–572 行），每次梯度步按 episode 数切分批量：

$$
n_{\mathrm{off}}=\mathrm{round}(128\cdot\rho),\qquad n_{\mathrm{on}}=128-n_{\mathrm{off}}
$$

在线 batch 从在线 replay 采 $n_{\mathrm{on}}$ 条 episode，离线 batch 从离线数据集采 $n_{\mathrm{off}}$ 条 episode，沿 batch 维拼接（`merge_offline_online`）后做**一次** TD 更新。本次 $\rho=0.1$，即每 128 条 episode 中约 13 条来自离线数据集（日志里的 `eff_offline_frac` 按填充步数复核）。$\rho=0$ 时完全不回放离线数据。

### 4.4 CQL 项

`cql_weight`（离线阶段，本次配置 5m 的 paper 臂为 50.0）与 `online_cql_weight`（在线阶段）分别控制保守项。本次所有在线臂的 `online_cql_weight=0.0`，因此在线更新里没有 CQL 项；paper 臂的 `cql_weight=50.0` 只在离线阶段生效。

---

## 5. 实验臂：每个臂具体做了什么

匹配协议（`scripts/run_matched_protocol_5m_vs_6m.sh`、`scripts/eval_matched_protocol.sh`）在同一张 5m_vs_6m 上从同一离线初始化出发，只切换下面的开关；FHJP 臂（`scripts/launch_fhjp.sh`）用同一初始化、同一在线循环，只是把提议器接进采集侧。

### 5.0 开关总表

| 臂 | learner_type | transfer.mode | uot_eta | mixing_ratio | use_ovm | λ_mem_end / anneal | 在线 CQL | 配置文件 |
|---|---|---|---|---|---|---|---|---|
| FHJP | ovmse_paper | full_td | —（无此项） | 0.0 | false | 不适用（无 OVM） | 0.0 | `configs/fhjp_5m_vs_6m.yaml` |
| UOT（臂 A） | uot | none | 0.1 | 0.0 | false | 0.1 / 200k（未启用） | 0.0 | `configs/matched_protocol_5m_vs_6m.yaml` + 臂覆盖 |
| vanilla（臂 B） | uot | none | **0.0** | 0.0 | false | 0.1 / 200k（未启用） | 0.0 | 同上 |
| OVMSE（臂 C） | ovmse_paper | ovmse | —（无此项） | **0.1** | **true** | 0.1 / 200k | 0.0 | 同上 |
| OVMSE（臂 D） | ovmse_paper | ovmse | — | **0.0** | **true** | 0.1 / 200k | 0.0 | 同上 |
| 仅混合（臂 E） | ovmse_paper | none | — | **0.1** | false | 0.1 / 200k（未启用） | 0.0 | 同上 |
| OVMSE paper 臂 | ovmse_paper | ovmse | — | 0.1 | true | 0.1 / 200k | 0.0 | `configs/5m_vs_6m.yaml`、`configs/2s3z.yaml` |

五个匹配臂的公共超参（`configs/matched_protocol_5m_vs_6m.yaml`）：`batch_size=128`、`batch_size_run=8`、`lr=1e-3`、`td_lambda=0.6`、`discount=0.99`、`target_update_interval=200`、`grad_norm_clip=10`、ε 1.0→0.05/100k、UOT 侧 `uot_delta=0.1`、`uot_margin=0.1`、`uot_k=8`、`uot_max_hamming=2`、`uot_ranking_scope=joint`、`uot_support_mode=joint`。

### 5.1 FHJP（seed 1 / 2 / 3）

- 采集侧接 `fhjp/controller.py::ProposerController`：每个决策点算离线/在线两个联合贪心，冲突时以 0.5 抽样执行其一；`source`、`conflict` 只进日志。
- 每 `version_len`（5m 500k、2s3z 200k）环境步到一次版本边界：冻结在线参数，跑 $2M=64$ 条在轨评估 episode（每臂 32 条，共享种子），算 $\widehat\tau_v$ 与单侧上界，满足条件即永久退出。
- 更新侧零改动：普通 TD(λ)，无 CQL、无 OVM、无离线回放、无排序损失、无倾向概率。
- 评估 episode 计入预算（`count_eval_in_budget: true`），因此同样 3M 预算下纯训练步数略少于其它臂。

### 5.2 UOT（匹配臂 A，`eta=0.1`）

- `learner_type=uot`：在线 QMIX TD(λ) + 序数 hinge；oracle $\bar Q$ 是加载后的那份权重的冻结拷贝。
- 每个梯度步：在线 batch（128 条 episode，$\rho=0$ 所以全部来自在线 replay）→ 前向得到 $\bar Q$ 与在线 $Q$ → 以三个锚点构造 Hamming≤2 的记录邻域候选（≤8 个）→ 对教师差距 ≥0.1 的候选对施加 $\delta=0.1$ 的 hinge → $\mathcal L=\mathcal L_{\mathrm{TD}}+0.1\,\mathcal L_{\mathrm{ord}}$。
- 无 OVM、无离线回放；`transfer.mode=none` 表示不做任何网络侧迁移改写，直接用离线初始化权重开训。

### 5.3 OVMSE（匹配臂 C，mixing 0.1）——本次终点胜率最高的臂之一

- `learner_type=ovmse_paper`、`transfer.mode=ovmse`、`online.use_ovm=true`、`mixing_ratio=0.1`。
- 每次梯度步的 128 条 episode 中约 13 条从 cfcql Medium 离线数据集采样，与 115 条在线 episode 拼接后做一次 TD(λ) 更新。
- 损失用 4.2 节的 OVM 形式：目标 $\tilde y_t=\max(Q^{\mathrm{off}}_{\mathrm{tot}},G_t)$，权重 $\lambda^{\mathrm{mem}}_t$ 由 1.0 在 200k 步内退火到 0.1。
- 无 UOT 排序项、无在线 CQL。

### 5.4 OVMSE（匹配臂 D，mixing 0）

与臂 C 唯一差别是 `mixing_ratio=0.0`：**只开 OVM、完全不再放离线数据**。因此它度量的是「OVM 目标单独」的贡献，与臂 C 的差即「离线回放混合」的贡献。

### 5.5 仅混合（匹配臂 E）

`transfer.mode=none`、`use_ovm=false`、`mixing_ratio=0.1`：既无 OVM 目标、也无 UOT 排序项，只是每步把约 13 条离线 episode 混进批量。它把「离线数据本身」与「离线数据在 OVM 目标下的用法」分开。

### 5.6 vanilla 在线 QMIX（匹配臂 B，`eta=0`）

`learner_type=uot` 但 `uot_eta=0.0`，因此序数项权重为 0（代码仍会计算并记录该项，只是不带梯度）；`mixing_ratio=0`、`use_ovm=false`。**这就是 FHJP 去掉提议器之后的配方**，是 FHJP 最直接的对照：两者的差别只有「冲突处 0.5/0.5 的联合提议 + 版本末评估臂」。

### 5.7 OVMSE paper 臂（`configs/5m_vs_6m.yaml` / `configs/2s3z.yaml`）

`scripts/run_ovmse_3maps.sh` 跑的参考臂：同一离线初始化、`phase=online`、`mixing_ratio=0.1`、`online.use_ovm=true`、`lambda_memory_end=0.1`、`lambda_anneal_steps=200000`、`online_cql_weight=0.0`。与臂 C 的配方相同，差别只在配置来源与运行批次（臂 C 由匹配协议驱动脚本运行，paper 臂由 3-map 链式脚本运行），因此两者可互为重复实验。2s3z 上的 OVMSE 只有这一条臂（1M 步）。

---

## 6. 结果

终点口径：对最终在线 checkpoint 跑 **640 集**评估（32 集一组，共 20 组），报告胜率的 Wilson 95% CI；两条臂评估种子相同时用 20 组的**配对 chunk bootstrap**（10000 次）给差值的 CI，评估种子不同时胜率用 Newcombe 混合分数 CI、回报用非配对 chunk bootstrap。训练曲线是每 10k 步 32 集的描述性口径，按 250k 步分箱。

### 6.1 终点：5m_vs_6m（3M 在线步）

| 臂 | seed | win rate | 95% CI | mean return | ckpt t_env |
|---|---:|---:|---|---:|---:|
| FHJP | 1 | **0.728** | [0.692, 0.761] | **17.32** | 3,000,069 |
| FHJP | 2 | **0.650** | [0.612, 0.686] | **16.50** | 3,000,200 |
| UOT（臂 A, η=0.1） | 1 | 0.727 | [0.691, 0.760] | 17.30 | 3,000,136 |
| OVMSE（臂 C, mixing 0.1） | 1 | **0.736** | [0.700, 0.769] | **17.44** | 3,000,126 |
| OVMSE（臂 D, mixing 0） | 1 | 0.439 | [0.401, 0.478] | 14.50 | 3,000,022 |
| OVMSE（paper 臂） | 1 | 0.667 | [0.630, 0.703] | 16.73 | 3,000,050 |
| vanilla 在线 QMIX（臂 B） | 1 | 0.548 | [0.510, 0.587] | 15.36 | 3,000,216 |
| 仅混合（臂 E） | 1 | 0.542 | [0.503, 0.580] | 15.41 | 3,000,118 |

FHJP 跨 seed：win 0.689 ± 0.055（0.650–0.728），return 16.91 ± 0.57。

### 6.2 终点：2s3z（1M 在线步）

| 臂 | seed | win rate | 95% CI | mean return | ckpt t_env |
|---|---:|---:|---|---:|---:|
| FHJP | 1 | **0.991** | [0.980, 0.996] | **19.96** | 1,000,051 |
| FHJP | 2 | **0.992** | [0.982, 0.997] | **19.95** | 1,000,073 |
| FHJP | 3 | **0.986** | [0.973, 0.993] | **19.92** | 1,000,024 |
| OVMSE（paper 臂） | 1 | 0.972 | [0.956, 0.982] | 19.82 | 1,000,014 |

FHJP 跨 seed：win 0.990 ± 0.003，return 19.94 ± 0.02。

### 6.3 统计对比（Δ 为正表示 FHJP 更好）

| 图 | 对比 | Δwin（95% CI） | Δreturn（95% CI） | 方法 |
|---|---|---:|---:|---|
| 5m_vs_6m | FHJP s1 − UOT-A s1 | +0.002 [−0.048, 0.050] | +0.01 [−0.44, 0.45] | 配对 |
| 5m_vs_6m | FHJP s1 − OVMSE-C s1 | −0.008 [−0.055, 0.039] | −0.12 [−0.55, 0.31] | 配对 |
| 5m_vs_6m | FHJP s1 − OVMSE-paper s1 | +0.061 [0.016, 0.106] | +0.59 [0.16, 1.02] | 配对 |
| 5m_vs_6m | FHJP s1 − OVMSE-D s1 | +0.289 [0.247, 0.336] | +2.82 [2.38, 3.29] | 配对 |
| 5m_vs_6m | FHJP s1 − vanilla-B s1 | **+0.180 [0.127, 0.236]** | **+1.95 [1.42, 2.51]** | 配对 |
| 5m_vs_6m | FHJP s1 − mix-E s1 | +0.186 [0.133, 0.236] | +1.91 [1.39, 2.38] | 配对 |
| 5m_vs_6m | FHJP s2 − UOT-A s1 | −0.077 [−0.127, −0.026] | −0.80 [−1.20, −0.38] | 非配对 |
| 5m_vs_6m | FHJP s2 − OVMSE-C s1 | −0.086 [−0.136, −0.035] | −0.93 [−1.36, −0.51] | 非配对 |
| 5m_vs_6m | FHJP s2 − OVMSE-D s1 | +0.211 [0.157, 0.263] | +2.00 [1.53, 2.49] | 非配对 |
| 5m_vs_6m | FHJP s2 − OVMSE-paper s1 | −0.017 [−0.069, 0.035] | −0.22 [−0.67, 0.24] | 非配对 |
| 5m_vs_6m | FHJP s2 − vanilla-B s1 | +0.102 [0.048, 0.154] | +1.14 [0.57, 1.71] | 非配对 |
| 5m_vs_6m | FHJP s2 − mix-E s1 | +0.108 [0.054, 0.161] | +1.09 [0.64, 1.52] | 非配对 |
| 2s3z | FHJP s1 − OVMSE-paper s1 | +0.019 [0.005, 0.031] | +0.14 [0.05, 0.23] | 配对 |
| 2s3z | FHJP s2 − OVMSE-paper s1 | +0.020 [0.006, 0.037] | +0.13 [0.03, 0.23] | 非配对 |
| 2s3z | FHJP s3 − OVMSE-paper s1 | +0.014 [−0.002, 0.031] | +0.10 [0.01, 0.20] | 非配对 |

### 6.4 FHJP 提议器诊断（版本边界）

$\widehat\tau_v$ = 32 对共享种子评估 episode 的（离线臂回报 − 在线臂回报）配对均值；退出条件为单侧 95% 上界 ≤ 0 且 $M\ge16$。

| 图/seed | 版本 | t_env | τ_v | τ_v 上界 | 离线臂 | 在线臂 | 冲突率 | 离线选中率 | 评估环境步 | 退出 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5m/seed1 | 1 | 500,178 | +3.174 | +3.970 | 9.36 | 6.19 | 0.530 | 0.501 | 1,537 | 否 |
| 5m/seed1 | 2 | 1,000,278 | −2.122 | −0.988 | 9.78 | 11.90 | 0.451 | 0.499 | 1,692 | **是** |
| 5m/seed2 | 1 | 500,060 | +0.037 | +1.213 | 9.36 | 9.33 | 0.526 | 0.501 | 1,652 | 否 |
| 5m/seed2 | 2 | 1,000,177 | −2.456 | −0.994 | 9.78 | 12.23 | 0.433 | 0.500 | 1,704 | **是** |
| 5m/seed3 | 1 | 500,175 | +1.436 | +2.300 | 9.36 | 7.93 | 0.561 | 0.500 | 1,633 | 否 |
| 5m/seed3 | 2 | 1,000,311 | +1.395 | +2.541 | 9.78 | 8.38 | 0.424 | 0.500 | 1,619 | 否 |
| 5m/seed3 | 3 | 1,500,509 | −0.174 | +1.203 | 9.59 | 9.76 | 0.440 | 0.500 | 1,670 | 否 |
| 5m/seed3 | 4 | 2,000,714 | −2.387 | −1.125 | 8.93 | 11.32 | 0.469 | 0.500 | 1,738 | **是** |
| 2s3z/seed1 | 1 | 200,089 | −1.349 | −0.621 | 13.10 | 14.45 | 0.682 | 0.497 | 3,450 | **是** |
| 2s3z/seed2 | 1 | 200,046 | −1.622 | −1.039 | 13.10 | 14.72 | 0.707 | 0.501 | 3,600 | **是** |
| 2s3z/seed3 | 1 | 200,024 | +11.667 | +12.352 | 13.10 | 1.43 | 0.699 | 0.501 | 4,677 | 否 |
| 2s3z/seed3 | 2 | 400,053 | −1.875 | −1.266 | 13.14 | 15.02 | 0.727 | 0.501 | 3,595 | **是** |

读数要点：

- 5m_vs_6m 上 seed1/seed2 在第 2 个版本（约 100 万步）退出，seed3 在第 4 个版本（约 200 万步）退出；2s3z 上 seed1/seed2 第 1 个版本（20 万步）退出，seed3 第 2 个版本（40 万步）退出。退出之后这些 run 就是普通在线 QMIX TD。
- 离线选中率全程 ≈0.50（正是固定 $p=0.5$ 在冲突集合内的表现），冲突率随版本下降（5m 上 0.53→0.43），说明在线策略在向离线联合动作收敛。
- 2s3z/seed3 第 1 版本的在线臂回报只有 1.43（在线策略还没学会），因此 $\widehat\tau_v$ 大幅为正；到第 2 版本在线臂回到 15.02，$\widehat\tau_v$ 立刻转负并退出。这正对应 2s3z 早期 FHJP 训练曲线低于 OVMSE 的现象（见 6.6）。

### 6.5 早期 vs 后期（32 集训练评估，250k 分箱）

| 图 | 臂 | seed | 前 250k win / ret | 末 250k win / ret |
|---|---|---:|---:|---:|
| 5m_vs_6m | FHJP | 1 | 0.023 / 6.6 | 0.674 / 16.8 |
| 5m_vs_6m | FHJP | 2 | 0.023 / 7.0 | 0.604 / 16.1 |
| 5m_vs_6m | FHJP | 3 | 0.029 / 7.4 | 0.594 / 16.0 |
| 5m_vs_6m | UOT-A | 1 | 0.019 / 6.6 | 0.673 / 16.8 |
| 5m_vs_6m | OVMSE-C | 1 | 0.040 / 9.2 | 0.682 / 16.9 |
| 5m_vs_6m | OVMSE-D | 1 | 0.028 / 8.5 | 0.448 / 14.5 |
| 5m_vs_6m | OVMSE-paper | 1 | 0.069 / 9.5 | 0.724 / 17.4 |
| 5m_vs_6m | vanilla-B | 1 | 0.017 / 5.7 | 0.522 / 15.1 |
| 5m_vs_6m | mix-E | 1 | 0.031 / 6.9 | 0.570 / 15.8 |
| 2s3z | FHJP | 1 | 0.668 / 17.2 | 0.979 / 19.9 |
| 2s3z | FHJP | 2 | 0.738 / 17.7 | 0.984 / 19.9 |
| 2s3z | FHJP | 3 | 0.454 / 14.0 | 0.973 / 19.8 |
| 2s3z | OVMSE-paper | 1 | 0.806 / 18.7 | 0.951 / 19.7 |

### 6.6 完整训练曲线（32 集评估，250k 分箱，win / return）

**5m_vs_6m**

| t_env 分箱 | FHJP s1 | FHJP s2 | FHJP s3 | UOT-A s1 | OVMSE-C s1 | OVMSE-D s1 | OVMSE-paper s1 | vanilla-B s1 | mix-E s1 |
|---|---|---|---|---|---|---|---|---|---|
| 0k–250k | 0.023 / 6.6 | 0.023 / 7.0 | 0.029 / 7.4 | 0.019 / 6.6 | 0.040 / 9.2 | 0.028 / 8.5 | 0.069 / 9.5 | 0.017 / 5.7 | 0.031 / 6.9 |
| 250k–500k | 0.060 / 9.4 | 0.072 / 9.4 | 0.074 / 9.4 | 0.068 / 9.5 | 0.217 / 11.8 | 0.087 / 10.0 | 0.228 / 12.0 | 0.007 / 7.8 | 0.040 / 9.1 |
| 500k–750k | 0.233 / 11.8 | 0.244 / 11.9 | 0.147 / 10.7 | 0.122 / 10.4 | 0.357 / 13.5 | 0.174 / 11.4 | 0.314 / 13.0 | 0.024 / 8.6 | 0.172 / 11.2 |
| 750k–1000k | 0.352 / 13.3 | 0.346 / 13.3 | 0.263 / 12.4 | 0.246 / 12.0 | 0.448 / 14.4 | 0.246 / 12.1 | 0.406 / 13.8 | 0.090 / 10.0 | 0.296 / 12.7 |
| 1000k–1250k | 0.421 / 14.0 | 0.401 / 13.9 | 0.325 / 13.1 | 0.380 / 13.6 | 0.475 / 14.7 | 0.305 / 12.9 | 0.443 / 14.3 | 0.246 / 12.0 | 0.350 / 13.3 |
| 1250k–1500k | 0.480 / 14.7 | 0.425 / 14.2 | 0.411 / 14.1 | 0.385 / 13.7 | 0.568 / 15.7 | 0.306 / 13.0 | 0.514 / 15.2 | 0.357 / 13.3 | 0.390 / 13.8 |
| 1500k–1750k | 0.539 / 15.3 | 0.499 / 15.0 | 0.432 / 14.4 | 0.494 / 14.8 | 0.629 / 16.3 | 0.390 / 13.9 | 0.594 / 16.0 | 0.372 / 13.5 | 0.395 / 13.9 |
| 1750k–2000k | 0.572 / 15.6 | 0.514 / 15.1 | 0.450 / 14.6 | 0.568 / 15.5 | 0.621 / 16.2 | 0.421 / 14.2 | 0.648 / 16.6 | 0.454 / 14.3 | 0.445 / 14.4 |
| 2000k–2250k | 0.604 / 15.9 | 0.580 / 15.8 | 0.502 / 15.0 | 0.616 / 16.2 | 0.664 / 16.8 | 0.403 / 14.0 | 0.667 / 16.8 | 0.489 / 14.7 | 0.491 / 14.8 |
| 2250k–2500k | 0.651 / 16.5 | 0.604 / 16.0 | 0.604 / 16.2 | 0.606 / 16.1 | 0.642 / 16.5 | 0.400 / 14.1 | 0.675 / 16.9 | 0.524 / 15.1 | 0.515 / 15.1 |
| 2500k–2750k | 0.672 / 16.8 | 0.585 / 15.9 | 0.697 / 17.1 | 0.642 / 16.5 | 0.688 / 17.0 | 0.430 / 14.4 | 0.691 / 17.0 | 0.522 / 15.1 | 0.516 / 15.2 |
| 2750k–3000k | 0.674 / 16.8 | 0.604 / 16.1 | 0.594 / 16.0 | 0.673 / 16.8 | 0.682 / 16.9 | 0.448 / 14.5 | 0.724 / 17.4 | 0.522 / 15.1 | 0.570 / 15.8 |

**2s3z**

| t_env 分箱 | FHJP s1 | FHJP s2 | FHJP s3 | OVMSE-paper s1 |
|---|---|---|---|---|
| 0k–250k | 0.668 / 17.2 | 0.738 / 17.7 | 0.454 / 14.0 | 0.806 / 18.7 |
| 250k–500k | 0.948 / 19.7 | 0.969 / 19.8 | 0.853 / 18.9 | 0.920 / 19.5 |
| 500k–750k | 0.986 / 19.9 | 0.981 / 19.9 | 0.975 / 19.8 | 0.953 / 19.7 |
| 750k–1000k | 0.979 / 19.9 | 0.984 / 19.9 | 0.973 / 19.8 | 0.951 / 19.7 |

### 6.7 从结果能直接读到的几件事

1. **5m_vs_6m 上 FHJP 与 UOT、OVMSE-C 打平**：同评估种子的配对 bootstrap 下 Δwin 分别是 +0.002 与 −0.008，CI 都跨 0；三者终点评级（0.65–0.74）也互相重叠。
2. **FHJP 明显高于同配方去掉提议器的 vanilla 对照**：Δwin +0.180 [0.127, 0.236]、Δreturn +1.95 [1.42, 2.51]；第二个 seed 上仍是 +0.102 [0.048, 0.154]。也就是说差别来自「冲突处 0.5/0.5 联合提议 + 版本末评估臂」这一机制，而不是别的配置差异。
3. **只开 OVM 不开混合（臂 D）是最差的 5m 臂**（0.439），低于 vanilla（0.548）；加上 0.1 离线回放（臂 C）后回到 0.736。OVM 的收益在这套初始化下依赖离线数据同时回放。
4. **2s3z 上 FHJP 三个 seed 都略高于 OVMSE paper 臂**（Δwin +0.014～+0.020），但该图胜率已接近天花板（0.97–0.99），回报差只有 0.10–0.14，量级很小；更有信息量的是早期动态：OVMSE 前 250k 就有 0.806，FHJP 是 0.454–0.738，之后 FHJP 反超。
5. **提议器的退出时点与曲线一致**：5m 上离线臂回报始终停在 9.4–9.8，而在线臂从 6.2 升到 11.9–12.2，第 2 个版本 $\widehat\tau_v$ 转负；2s3z 上在线臂一旦追上（seed3 第 2 版本 15.02 vs 13.14），$\widehat\tau_v$ 立刻转负。退出判据确实在「离线联合行为不再有可测正收益」的时点触发。

---

## 7. 复现

```bash
# 1) 640 集终点评估（FHJP 各 seed + OVMSE 2s3z 臂；已评估过的自动跳过）
bash scripts/eval_fhjp.sh

# 2) 汇总 + 生成自动版报告（本文所有表格的数值来源）
python scripts/analyze_fhjp_vs_arms.py \
    --out logs/fhjp_vs_arms_summary.json \
    --md documentation/experiment_reports/fhjp_vs_ovmse_uot.md
```

数据位置：FHJP 训练曲线 `checkpoints/fhjp_<map>/seed<n>/online.jsonl`，终点评估 `checkpoints/fhjp_<map>/seed<n>/eval640_final.jsonl`，匹配臂 `checkpoints/matched_protocol_20261020/<arm>_seed1/`，OVMSE paper 臂 `checkpoints/ovmse_batch_20261014/<map>_online/`。算法规范 `documentation/joint_proposal_handover.md`，实现 `fhjp/`、`uot/`、`ovmse/`。
