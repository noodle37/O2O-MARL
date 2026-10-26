# 文章总结：Online Pre-Training for Offline-to-Online Reinforcement Learning (OPT)

**发表信息**：ICML 2025（PMLR 267），LG AI Research 与 KAIST 联合发表。作者包括 Yongjae Shin、Jeonghye Kim、Whiyoung Jung 等，通讯作者 Woohyung Lim。

**一句话概括**：本文提出 OPT，在传统的"离线预训练 → 在线微调"两阶段之间插入一个新的"在线预训练（Online Pre-Training）"阶段，专门预训练一个**新的价值函数** $Q^{\mathrm{on\text{-}pt}}$，使其在正式在线微调前就能适应在线数据分布，从而解决离线预训练价值函数估计不准导致的微调性能差的问题。

---

## 1. 解决 O2O RL 中的什么问题？为什么现有算法没解决？

**问题**：离线预训练智能体在在线微调阶段性能不佳——甚至不如从头训练（from scratch，即在离线数据初始化的 replay buffer 上直接在线训练）。如图 1 所示，walker2d-random 中 from scratch 从一开始就超过 off-to-on；antmaze-umaze-diverse 中 off-to-on 前期尚可但最终被 from scratch 超越。

**根源**：分布偏移（distribution shift）导致的**价值估计不准确**（inaccurate value estimation）。离线训练中为防止 OOD 动作被高估，价值函数被训练得保守/有偏；在线微调时数据分布不断变化，$Q^{\mathrm{off\text{-}pt}}$ 无法准确评估新策略产生的动作，从而引导策略向错误方向更新（被 Nakamoto et al. 2024 称为 counter-intuitive trends）。

**现有方法为何未解决**：

- **Cal-QL**（NeurIPS 2023）：给价值更新加下界校准，但**从离线阶段到在线阶段全程使用同一个价值函数**，保守性在微调阶段仍持续阻碍策略改进；
- **SO2**（AAAI 2024）：对价值更新加扰动并提高更新频率，但计算成本显著更高，且实验显示其在 100k 步表现好、延长到 300k 步后性能反而下降；
- **OEMA**（IEEE TKDE 2023）的元适应目标是针对策略学习的，不针对价值函数；
- Off2On 虽在 MuJoCo 上强，但在 Antmaze-large 和 Adroit 上表现差。

核心共同缺陷：**所有方法都依赖一个贯穿离线与在线阶段的单一价值函数**，而这个价值函数恰好是问题的来源。

## 2. 核心创新点

三个核心创新：

1. **引入一个全新的价值函数 $Q^{\mathrm{on\text{-}pt}}$**，与 $Q^{\mathrm{off\text{-}pt}}$ 并存、分工互补（时间域上的双价值函数），打破了"单一价值函数贯穿两阶段"的范式；
2. **提出新的学习阶段——Online Pre-Training**：在离线预训练与在线微调之间，用固定离线策略 $\pi^{\mathrm{off}}$ 收集 $N_\tau=25k$ 条在线样本 $B_{\mathrm{on}}$，结合离线数据 $B_{\mathrm{off}}$，通过**元适应（meta-adaptation）目标**预训练 $Q^{\mathrm{on\text{-}pt}}$，使其在微调开始前就已适应在线分布。注意：这一步只训练 $Q^{\mathrm{on\text{-}pt}}$，$\pi^{\mathrm{off}}$ 与 $Q^{\mathrm{off\text{-}pt}}$ 均冻结，不会扰乱策略；
3. **两价值函数加权组合的微调策略**：在线微调阶段策略损失为

$$\mathcal{L}_{\pi}^{\mathrm{finetune}}(\phi)=\mathbb{E}_{s\sim B}\big[-\{(1-\kappa)Q^{\mathrm{off\text{-}pt}}(s,\pi_\phi(s))+\kappa Q^{\mathrm{on\text{-}pt}}(s,\pi_\phi(s))\}\big]$$

其中 $\kappa$ 采用**线性调度**（附录 H：多数环境从 0.1 升到 0.9，MuJoCo random 数据集上 $\kappa\equiv1$ 即完全弃用 $Q^{\mathrm{off\text{-}pt}}$），配合 balanced replay（均衡回放）加速在线样本利用。

附加创新：OPT 是一个**通用增强模块**，可插拔接入任何基于价值函数的 RL 算法（TD3、SPOT、IQL、RLPD 均被验证）。

## 3. 核心模块、作用机制与完整算法流程

### 三个核心模块

**模块 1：Online Pre-Training 的数据设计（$B_{\mathrm{off}} + B_{\mathrm{on}}$）**
- 仅用 $B_{\mathrm{off}}$ 训练会使 $Q^{\mathrm{on\text{-}pt}}$ 复制 $Q^{\mathrm{off\text{-}pt}}$，失去引入意义；仅用 $B_{\mathrm{on}}$（由固定 $\pi^{\mathrm{off}}$ 生成）会过拟合到该策略、无法泛化到后续改进的策略。因此必须混合使用，这在图 5 与表 9 的消融中得到验证（仅用 $B_{\mathrm{on}}$ 预训练在 random 数据集上甚至不如随机初始化）。

**模块 2：元适应目标函数**（受 OEMA 启发，改造为面向价值函数）：

$$\mathcal{L}_{Q^{\mathrm{on\text{-}pt}}}^{\mathrm{pretrain}}(\psi)=\mathcal{L}_{Q^{\mathrm{on\text{-}pt}}}^{\mathrm{off}}(\psi)+\mathcal{L}_{Q^{\mathrm{on\text{-}pt}}}^{\mathrm{on}}\big(\psi-\alpha\nabla\mathcal{L}_{Q^{\mathrm{on\text{-}pt}}}^{\mathrm{off}}(\psi)\big)$$

其中
$$\mathcal{L}_{Q^{\mathrm{on\text{-}pt}}}^{\mathrm{off}}(\psi)=\mathbb{E}_{(s,a,r,s')\in B_{\mathrm{off}}}\big[\big(Q_\psi^{\mathrm{on\text{-}pt}}(s,a)-(r+\gamma Q_{\bar\psi}^{\mathrm{on\text{-}pt}}(s',\pi_\phi(s')))\big)^2\big]$$
$$\mathcal{L}_{Q^{\mathrm{on\text{-}pt}}}^{\mathrm{on}}(\psi)=\mathbb{E}_{(s,a,r,s')\in B_{\mathrm{on}}}\big[\big(Q_\psi^{\mathrm{on\text{-}pt}}(s,a)-(r+\gamma Q_{\bar\psi}^{\mathrm{on\text{-}pt}}(s',\pi_\phi(s')))\big)^2\big]$$

内层梯度先让 $\psi$ 朝 $B_{\mathrm{off}}$ 的 TD 方向走一步，再在 $B_{\mathrm{on}}$ 上评估损失——即"**先在离线数据上快速适应一步，再要求它能适应在线样本**"，从而让 $Q^{\mathrm{on\text{-}pt}}$ 具备对未来在线分布的可迁移性。表 9 显示：直接相加（无元适应）的两个损失存在学习动力学冲突，在 Antmaze-large 上完全失效，验证了元适应的必要性。

**模块 3：$\kappa$ 调度 + Balanced Replay**
- $\kappa$ 初始较小（重 $Q^{\mathrm{off\text{-}pt}}$，离线数据上可靠），随微调推进线性增大（重 $Q^{\mathrm{on\text{-}pt}}$，快速适应在线数据）；表 4/表 7 显示调度本身关键（固定 $\kappa=0,0.5,1$ 均明显更差，尤其 Antmaze-large），但对终点取值不敏感；
- Balanced replay：按 Lee et al. 2022 的方法，提高在线样本在 minibatch 中的占比，加速 $Q^{\mathrm{on\text{-}pt}}$ 适应；附录 C.1 表明即使换成普通的 Online Replay，OPT 仍显著优于基线（说明增益不主要来自 balanced replay）。

### 完整算法流程（Algorithm 1）

1. **离线预训练（Offline Pre-Training）**：常规离线 RL（如 TD3+BC 或 SPOT），1M 梯度步，得到 $\{Q^{\mathrm{off\text{-}pt}}, \pi^{\mathrm{off}}\}$；
2. **在线预训练（Online Pre-Training，占 25k 环境步 + 50k 梯度步 $N_{\mathrm{pretrain}}$）**：
   - 用冻结的 $\pi^{\mathrm{off}}$ 收集 $N_\tau=25k$ 条转移存入 $B_{\mathrm{on}}$；
   - 重复 50k 步：从 $B_{\mathrm{off}}$ 和 $B_{\mathrm{on}}$ 各采样 minibatch，按式(3)的元适应目标更新 $Q^{\mathrm{on\text{-}pt}}$（其他组件全部冻结）；
3. **在线微调（Online Fine-Tuning，剩余 275k 步）**：
   - 初始化均衡回放 buffer $B_{BR}\leftarrow B_{\mathrm{off}}\cup B_{\mathrm{on}}$；
   - 每步从 $B_{BR}$ 采样，分别用标准 TD 学习更新 $Q^{\mathrm{off\text{-}pt}}$ 和 $Q^{\mathrm{on\text{-}pt}}$；
   - 用加权组合损失 $\mathcal{L}_{\pi}^{\mathrm{finetune}}$（式 4）更新 $\pi$（$\pi_\phi$ 由 $\pi^{\mathrm{off}}$ 初始化），$\kappa$ 按时间表线性增大。

## 4. 是否有理论证明？

**没有。** 本文是纯实证（empirical）工作，全文没有任何收敛性、最优性或误差界的理论分析。附录 D 用"训练至近最优的 TD3 价值函数作为参考"的估计偏差（estimation bias）实验做了**定性的机制验证**（图 9、图 10）：TD3+OPT 的偏差显著低于 TD3；$Q^{\mathrm{on\text{-}pt}}$ 比 $Q^{\mathrm{off\text{-}pt}}$ 更快降低偏差，且组合 $Q$ 的偏差最低——这是实验性的机制证据而非理论证明。

## 5. 是否讨论了"Offline 知识保留/运用/阻碍"问题？

**部分涉及，但没有作为独立研究问题系统展开。**

- **保留**：$Q^{\mathrm{off\text{-}pt}}$（以及 $\pi^{\mathrm{off}}$ 作为初始化）就是保留下来的离线知识；
- **如何在 Online 使用**：通过 $(1-\kappa)Q^{\mathrm{off\text{-}pt}}+\kappa Q^{\mathrm{on\text{-}pt}}$ 的加权组合参与策略改进，且权重随训练从离线知识平滑过渡到在线知识；
- **离线知识是否构成阻碍**：**明确讨论了**。图 6 的 t-SNE 可视化显示 random 数据集上离线分布与在线 rollout 分布从微调开始就差异巨大，导致 $Q^{\mathrm{off\text{-}pt}}$ 严重有偏、不可信；附录 H 因此规定 MuJoCo 三个 random 环境 $\kappa\equiv1$，即**完全弃用** $Q^{\mathrm{off\text{-}pt}}$。图 7 与表 11 的消融也表明：仅靠（经 Online Pre-Training 更新的）$Q^{\mathrm{off\text{-}pt}}$ 而不引入新价值函数，random 数据集上性能大幅崩塌。这说明作者承认并实证了：**低质量离线数据训练出的价值函数对在线阶段是阻碍**。

但本文并未深入讨论"哪些离线知识值得保留"这一更一般的问题（如策略知识、数据知识、模型知识等），只聚焦于价值函数这一层。

## 6. 是否适合迁移到 MARL？分析

**总体判断：思路可迁移、有潜力，但存在显著挑战，不能简单照搬。**

**适合迁移的理由：**
1. "分布偏移导致离线预训练价值函数估计不准"在 MARL 中同样存在且更严重——CTDE 框架下集中式 critic 在离线数据上训练，在线微调时联合动作分布变化剧烈，O2O 的问题在 MARL 中被多篇工作确认；
2. "训练一个新 critic 使其预适应在线联合数据，再双 critic 加权调度"的范式不依赖单智能体结构，集中式 critic 天然适合作为 $Q^{\mathrm{on\text{-}pt}}$ 的载体；
3. 元适应目标（先在内层适应离线批次、再要求适应在线批次）与智能体数量无关，可直接套用；
4. MARL 中离线预训练 critic 同样有偏（对未见过的联合动作组合），引入专门面向在线数据的 critic 逻辑上更有必要。

**迁移带来的挑战：**
1. **$B_{\mathrm{on}}$ 的收集成本**：单智能体 25k 步 ≈ 25k 环境交互；MARL 中若以联合 transition 计尚可，但多智能体在线探索初期极不稳定，用固定离线联合策略采集的样本质量和覆盖度会更差，$N_\tau$ 的有效规模需要放大；
2. **联合动作空间维数爆炸**：critic 对 OOD 联合动作的估计偏差更大，$Q^{\mathrm{off\text{-}pt}}$ 的偏差问题更严重，元适应预训练是否还能在 25k 量级样本内收敛成疑；
3. **信用分配**：$Q^{\mathrm{on\text{-}pt}}$ 需要同时服务于策略改进与（可能存在的）个体价值分解，双 critic 加权可能干扰 value decomposition；
4. **非平稳性**（见第 7 问）。

## 7. 上述问题在 MARL 中是否不同、是否更具挑战性？

**是的，更具挑战性，原因有二：**

**（a）非平稳性（non-stationarity）**：单智能体 RL 中环境是平稳的，分布偏移仅来自策略自身的改进；MARL 中每个智能体策略都在变，"环境"对其他智能体天然非平稳。这意味着：
- 本文的 $\kappa$ 调度逻辑（从 $Q^{\mathrm{off\text{-}pt}}$ 平滑过渡到 $Q^{\mathrm{on\text{-}pt}}$，假设偏差随时间单调改善）在 MARL 中可能失效——$Q^{\mathrm{on\text{-}pt}}$ 自身的训练目标也在移动（其他智能体策略变化），预训练学到的"可适应能力"的保质期更短；
- $B_{\mathrm{on}}$ 由固定 $\pi^{\mathrm{off}}$ 采集这一设计，在 MARL 中意味着联合策略固定、各智能体策略也固定，采集到的数据多样性比单智能体更低，过拟合到 $\pi^{\mathrm{off}}$ 的风险加剧。

**（b）联合动作（joint action）**：OOD 问题从"单动作"变为"联合动作组合"，离线数据几乎不可能覆盖所有联合动作，$Q^{\mathrm{off\text{-}pt}}$ 的偏差更严重。有趣的是这反过来可能让本文方法更有价值——正因为在 MARL 中"离线知识是否构成阻碍"的答案更偏向"是"（类似本文 random 数据集情形），一个提前适应在线联合数据的 $Q^{\mathrm{on\text{-}pt}}$ 和按数据质量决定弃用程度（$\kappa$）的机制，可能比单智能体场景收益更大；但代价是预训练难度和样本需求都上升。

另外，$Q^{\mathrm{off\text{-}pt}}$ 在 MARL 中的可靠性对数据集质量的依赖更陡：本文在 medium/medium-replay 上用 $\kappa$ 从 0.1–0.3 起步，在 MARL 中这一初始值如何设定将更敏感。

## 8. 本文的不足（结合前述问题，特别是迁移到 MARL 的不足）

1. **无理论保证**：$\kappa$ 调度、元适应目标为什么有效只有直觉解释和实验证据，没有偏差收敛或性能下界分析；
2. **超参数与数据质量强耦合**：$N_\tau=25k$、$N_{\mathrm{pretrain}}=50k$、$\kappa$ 的初值/终值/衰减时长都需按域和数据质量人工设定（附录 H 共 19 组配置），泛化成本高；图 8 显示 $N_\tau<25k$ 时性能明显下降，说明方法对预训练样本量有一定门槛；
3. **计算开销增加约 50%**（图 12：TD3 约 4000s vs TD3+OPT 约 6000s），在 MARL 中 critic 本身更大更复杂，这一开销会被进一步放大；
4. **$B_{\mathrm{on}}$ 由单一固定策略生成**，预训练样本多样性受限，全靠元适应目标弥补；
5. **对"哪些离线知识值得保留"没有一般性回答**，只处理了价值函数；策略知识（如 $\pi^{\mathrm{off}}$ 的约束、VAE 模型等）如何保留未被讨论；
6. **迁移到 MARL 的具体不足**（衔接第 6、7 问）：联合动作导致 $Q^{\mathrm{on\text{-}pt}}$ 预训练收敛更难；非平稳性使元适应学到的适应能力过期更快；信用分配与价值分解如何与双 critic 加权结合完全未讨论；探索阶段联合策略的不稳定性使 25k 步固定策略采集的数据质量不可控；且本文所有实验均为单智能体，无任何多智能体验证。

## 9. 对照算法、发表信息、实验设置与消融/验证实验

### 对照算法及出处

| 基线 | 出处 |
|---|---|
| TD3（backbone） | Fujimoto et al., **ICML 2018** |
| TD3+BC | Fujimoto & Gu, **NeurIPS 2021** |
| Off2On | Lee et al., **CoRL 2022**（PMLR） |
| OEMA | Guo et al., **IEEE TKDE 2023** |
| PEX | Zhang et al., arXiv **2023**（未正式发表） |
| ACA (Actor-Critic Alignment) | Yu & Zhang, **ICML 2023** |
| SO2 | Zhang et al., **AAAI 2024** |
| FamO2O | Wang et al., **NeurIPS 2024** |
| Cal-QL | Nakamoto et al., **NeurIPS 2023** |
| IQL | Kostrikov et al., arXiv **2021b** |
| SPOT | Wu et al., **NeurIPS 2022** |
| RLPD | Ball et al., **ICML 2023** |
| BOORL | Hu et al., arXiv **2024** |

### 主实验

- **环境**：D4RL 三域——**MuJoCo**（halfcheetah/hopper/walker2d × random/medium/medium-replay）、**Antmaze**（umaze/umaze-diverse/medium-play/medium-diverse/large-play/large-diverse）、**Adroit**（pen/hammer/door/relocate-cloned）；
- **协议**：离线 1M 梯度步；在线共 300k 环境步（OPT 用前 25k 做 Online Pre-Training + 275k 微调）；10 个随机种子，报告均值与 95% 置信区间；
- **结果**：TD3+OPT 在 MuJoCo 总分 939.1 vs Off2On 860.9（最佳基线）；SPOT+OPT 在 Antmaze 565.3 vs Cal-QL 503.4；Adroit 309.1 vs SPOT 200.2；IQM 指标（图 3）置信区间不重叠，统计显著；IQL 接入 OPT 也在 Antmaze 显著提升；RLPD 接入 OPT 后 918.7 > Off-to-on RLPD 887.6 > Vanilla 866.0。

### 问题验证/机制分析实验（附录 D）

- 训练一个近最优 TD3 作为"真值"critic，对 10 个固定状态-动作对，在微调全程测量估计偏差：TD3+OPT 偏差始终低于 TD3；单独对比 $Q^{\mathrm{off\text{-}pt}}$、$Q^{\mathrm{on\text{-}pt}}$、组合 $Q$（图 10），证明偏差降低来自 $Q^{\mathrm{on\text{-}pt}}$ 的有效在线适应——这是针对核心假设" inaccurate value estimation "的直接验证。

### 消融实验

1. **$Q^{\mathrm{on\text{-}pt}}$ 初始化方式**（§5.1，图 5、表 8/9）：随机初始化 vs 仅用 $B_{\mathrm{on}}$ 预训练 vs OPT 元适应预训练 vs $B_{\mathrm{off}}+B_{\mathrm{on}}$ 直接相加——OPT 全胜，且 Antmaze-large 上其他方法接近 0 分；
2. **$\kappa$ 调度**（§5.4，表 4/7）：固定 $\kappa\in\{0,0.5,1\}$ vs 线性调度 0.1→{0.7,0.8,0.9}——调度必要，终点不敏感；
3. **是否引入新价值函数**（§5.3，图 7、表 11）：去掉 $Q^{\mathrm{on\text{-}pt}}$ 后 random 数据集上崩溃（总分 933.9→820.6）；
4. **$N_\tau$**（§5.5，图 8/12、表 10）：6.25k/12.5k/25k/50k/100k，25k 后饱和，取 25k；
5. **Balanced replay**（附录 C.1，表 6）：换成 Online Replay 仍有 20.2% 提升（vs 24.5%），说明增益不主要来自回放策略；
6. **数据集特性 → $\kappa$ 调度依据**（§5.2，图 6）：t-SNE 可视化 $B_{\mathrm{off}}$、初始 rollout $B_{\mathrm{init}}$、结束 rollout $B_{\mathrm{final}}$ 的分布差异，解释不同数据集上 $\kappa$ 初值的设定逻辑；
7. **扩展性**（§4.2、§5.6、附录 G）：接入 IQL（不同策略/价值结构）、RLPD（ensemble + 高 UTD）、与 BOORL/SO2 延长到 300k 步的公平对比（表 14，OPT 933.9 > BOORL 852.1 > SO2 818.5，且二者用 ensemble 计算成本更高）。

## 10. 核心 Claims 及其证明情况

| 核心 Claim | 证明形式 |
|---|---|
| C1：离线预训练智能体微调差源于价值估计不准，引入**新的价值函数**可解决（Q1） | 无理论；有**问题验证实验**（附录 D 偏差分析，图 9/10）+ 消融（图 7/表 11：去掉新价值函数则崩溃）+ 主实验 |
| C2：**Online Pre-Training（元适应目标 + 混合数据）** 是训练 $Q^{\mathrm{on\text{-}pt}}$ 的最佳方式（Q2） | 无理论；消融实验（图 5、表 8/9 四种初始化对比；表 9 证明元适应优于直接损失相加） |
| C3：双价值函数**加权 + $\kappa$ 调度**优于单一价值函数 | 无理论；表 4/7 消融 + 表 15 跨 19 个环境的调度配置 + 主实验 |
| C4：OPT 在 300k 在线步限制下达到/超过 SOTA，**平均提升约 30%** | 无理论；主实验（表 1–3、图 3 IQM、10 种子统计）+ 附录 E（TD3 +24.8%、SPOT MuJoCo +21.3%/Antmaze +54.4% 等） |
| C5：OPT 是**通用模块**，可接入不同 backbone（TD3/SPOT/IQL/RLPD） | 无理论；扩展实验（§4.2 图 4、§5.6 表 5、附录 E/F） |

**总结**：本文的所有 claims 均由实验支撑（问题验证实验、主实验、消融实验、统计分析），**没有任何理论证明**。最强的 claim（C1+C2）恰恰对应于您第 5 问所关心的"离线知识在在线阶段的阻碍与保留"问题——作者用 random 数据集上 $\kappa\equiv1$（弃用离线价值函数）与去掉新价值函数后崩溃的实验，实证了离线价值知识在分布差异大时确实构成阻碍，而用加权调度在两者之间平滑过渡是缓解手段。这一发现若迁移至 MARL，预期会因联合动作空间与非平稳性而更加突出，但也正因此使 OPT 式的新 critic 预训练在 MARL 中可能比单智能体场景更具价值——当然，这需要后续工作验证。