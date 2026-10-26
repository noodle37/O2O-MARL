# 文章总结：Offline-to-Online Reinforcement Learning with Classifier-Free Diffusion Generation (CFDG)

**发表信息**：ICML 2025（PMLR 267），上海交通大学（Xiao Huang、Xu Liu 为共同一作，通讯作者 Shuai Li）与 Adobe Research 合作完成。

**一句话概括**：本文从**数据增强（data-centric）视角解决 O2O RL 问题：用 classifier-free guidance 的条件扩散模型把离线数据和在线数据当作**两个类别标签**同时学习、同时生成，再通过重加权让生成数据更贴近当前在线策略分布，作为"即插即用"的数据增强模块接入 IQL、PEX、APL 等现有 O2O 算法，平均提升约 15%。

---

## 1. 解决 O2O RL 中的什么问题？为什么现有算法没解决？

**问题**：在线微调阶段**离线数据与在线数据分布不一致（distribution shift）**，且两者特性截然不同——离线数据分布均匀、多样性好，可防止策略过早收敛到次优；在线数据分散、更贴近当前策略，能提供稳定性、加速收敛。直接用离线 replay（如 Off2On）忽视分布偏移；仅 replay 两类数据则受限于在线数据量少、覆盖窄。

**现有生成式数据增强方法为何未解决**：现有方法（如 EDIS, ICML 2024）仅用**离线数据**做增强，用能量函数引导扩散采样向在线策略对齐。本文通过 t-SNE 分析（图 1）发现：EDIS 的生成数据大部分保留了离线数据的特征、只与在线数据"部分相似"，仍存在明显 gap；且 EDIS 需要额外训练三个能量函数/分类器式引导模型，训练开销大、引导过程复杂、跨任务适应性差。SynthER（NeurIPS 2023）则仅用扩散模型增强在线数据，放弃了离线数据的多样性价值。

**根源**：①没有分别刻画离线与在线数据两种分布的不同特性与不同作用；②对齐引导机制（分类器/能量函数）复杂且昂贵；③只增强单类数据。这引出本文的核心问题："能否用更简单的方式引导模型生成更高质量的数据？"

## 2. 核心创新点

1. **分布分析发现**：现有方法（EDIS）生成的数据仍与在线数据存在显著差距；且"直接增强在线数据"比"增强离线数据"更有效（因为在线数据天然贴合当前策略），但在线数据量太少。
2. **把离线数据与在线数据视为两个类别标签，用 classifier-free guidance（CFG）扩散模型对两类数据同时生成**：单网络（输入 null token $\varnothing$ 即无条件）一次训练同时支持两类条件采样，无需额外预训练分类器，训练成本更低、更易适配不同 RL 任务的分布变化。采样时：

$$\tilde{\epsilon}_{\theta}(\mathbf{z}_\lambda, \mathbf{c}) = (1+w)\,\epsilon_{\theta}(\mathbf{z}_\lambda, \mathbf{c}) - w\,\epsilon_{\theta}(\mathbf{z}_\lambda)$$

3. **数据重加权（reweighting）**：合成数据整体占比 $r=1/3$，且合成数据内部**在线合成 : 离线合成 = 8:2**，让生成数据更多地向在线策略分布对齐，在提升性能的同时保持稳定性。
4. **通用性**：与两种数据使用范式（IQL/PEX 的 1:1 混合范式、APL/SUNG 的 OORB 伯努利采样范式）都能兼容，接入后 IQL、PEX 提升 15%，APL 提升 11%。

## 3. 核心模块、作用机制与完整流程

### 核心模块

**模块 1：条件扩散模型 $\mathcal{M}$（基于 EDM, Karras et al. 2022）**
- 单一去噪网络 $\epsilon_\theta(\mathbf{z}_\lambda, \mathbf{c})$，以 $\mathbf{c}\in\{\text{offline}, \text{online}\}$ 为条件；训练时以概率 $p_{\mathrm{uncond}}$ 将 $\mathbf{c}$ 替换为 null token $\varnothing$ 实现无条件分支，从而 $\epsilon_\theta(\mathbf{z}_\lambda)=\epsilon_\theta(\mathbf{z}_\lambda, \varnothing)$，分类器梯度可估计为：

$$\nabla_{\mathbf{z}_t}\log p(c|\mathbf{z}_t) = -\frac{1}{\sigma_t}\big(\epsilon_\theta(\mathbf{z}_t, c, t) - \epsilon_\theta(\mathbf{z}_t, t)\big)$$

- 相比 classifier guidance（Dhariwal & Nichol 2021）无需在高噪声输入上训练鲁棒分类器，**零额外分类器开销**。

**模块 2：按间隔更新的数据生成与四 buffer 体系**
- 设置模型更新频率（IQL/PEX：100K 步；APL：10K 步），每隔一段时间用 $D_{\mathrm{off}}\cup D_{\mathrm{on}}$ 更新 $\mathcal{M}$，分别采样写入 $D_{\mathrm{off\_syn}}$ 与 $D_{\mathrm{on\_syn}}$（合成 buffer 大小 1M），摊平扩散模型训练与采样的高额时间成本。

**模块 3：数据重加权与两类基底的适配机制**
- 对 IQL/PEX（每 batch 50% 在线 + 50% 离线）：batch = 在线数据 ∥ 离线数据 ∥ 合成数据，合成比例 $r=1/3$；
- 对 APL（OORB 范式，以概率 $p$ 取在线、$1-p$ 取离线，且 $\lambda=0/1$ 决定是否启用保守正则，$\lambda\leftarrow 0$ if online else $1$）：合成数据被视作在线数据或离线数据的一部分参与采样与训练；
- 合成数据内部按 **8:2** 偏向在线合成数据，使整体生成分布更贴合当前在线策略。

### 完整算法流程（Algorithm 1）

1. **离线预训练**：仅用 $D_{\mathrm{off}}$ 按所选基底算法（IQL/PEX/APL）的离线损失 $\mathcal{L}_{\mathrm{offline}}^{Q_\theta}, \mathcal{L}_{\mathrm{offline}}^{\pi_\phi}$ 训练 critic 与策略；
2. **在线微调循环**：
   - 每环境步：用当前策略交互，$(s,a,s',r)\to D_{\mathrm{on}}$；
   - 到达更新频率时：用 $D_{\mathrm{off}}$ 与 $D_{\mathrm{on}}$ 更新条件扩散模型 $\mathcal{M}$；分别以条件 $\mathbf{c}=\text{offline}$、$\mathbf{c}=\text{online}$ 采样，写入 $D_{\mathrm{off\_syn}}$、$D_{\mathrm{on\_syn}}$；
   - 每梯度步：从 $D_{\mathrm{on}}\cup D_{\mathrm{off}}\cup D_{\mathrm{off\_syn}}\cup D_{\mathrm{on\_syn}}$ 按重加权规则采样 batch，按基底算法的在线损失 $\mathcal{L}_{\mathrm{online}}^{Q_\theta}, \mathcal{L}_{\mathrm{online}}^{\pi_\phi}$ 更新 critic 与策略。

整个 CFDG 模块对基底算法零侵入——只改变"数据从哪来"，不改变任何损失函数形式。

## 4. 是否有理论证明？

**没有。** 本文是纯实证工作，无收敛性、误差界或最优性分析。与之最接近的"机制验证"是：①t-SNE 可视化（图 1）；②Jensen-Shannon 散度定量分析（表 2）：离线数据与在线数据的 JS 散度 0.69，EDIS 生成数据 0.56，CFDG 生成数据 **0.50**（state-action-transition 各维度均更低），实证"生成质量更高、与在线数据更对齐"这一 claim。

## 5. 是否讨论了"Offline 知识保留/运用/阻碍"？

**以数据视角隐式、部分地讨论了，且给出了一个相当明确的答案。**

- **保留什么**：完整保留离线数据集 $D_{\mathrm{off}}$，并利用其"分布均匀、多样性好、可防止过早收敛到次优"的特性（引言中明确引用 Nair et al. 2020；Thrun & Littman 的观点：offline 数据靠多样性防止 premature convergence，online 数据提供稳定性）；
- **如何在 Online 使用**：不是直接使用，而是①与在线数据 1:1 或伯努利混合 replay；②作为条件扩散的一个类别被"再生成"，生成数据既保留离线数据的多样性结构又向在线策略对齐；③通过重加权（在线合成占 8/10）控制离线知识的注入强度；
- **离线知识是否构成阻碍**：**明确讨论并实证了**。图 1 与 §4.2 表明 EDIS 式地"让离线数据强行对齐在线分布"效果反而不如直接增强在线数据——即不经恰当处理的离线知识对齐是有损的、构成拖累；表 2 的 JS 散度也说明裸离线数据（0.69）与在线数据差距大，直接用会引入分布偏移。本文的解答是"两类别条件生成 + 重加权"而非简单 replay 或强行对齐。

局限：本文只从**数据层**讨论知识保留，未涉及离线策略/价值函数等参数化知识。

## 6. 是否适合迁移到 MARL？分析

**思路可迁移且对 MARL 有天然吸引力，但挑战显著。**

**适合迁移的理由：**
1. MARL 的 O2O 问题同样（甚至更重）表现为离线联合数据与在线联合数据分布偏移，且 CTDE 下集中式 critic 严重依赖数据覆盖——数据增强正是 MARL 离线 RL 的主流补救方向（如 MADT 式扩散增强），CFDG 的"双类别条件生成"直接对应 MARL 中"离线联合经验 vs 在线联合经验"；
2. CFG 机制本身与智能体数量无关，只需把数据标签扩展到 joint transition；重加权机制（偏向当前策略数据）对 MARL 的探索-利用平衡同样有意义；
3. 模块对基底算法零侵入，可叠加在 MA-IQL、MA-PEX 类方法上。

**迁移挑战：**
1. **条件类别不再只有两类**：MARL 中可能需要"离线数据 × 各智能体策略版本"更细的条件标签，否则非平稳的联合分布会使"online"标签内部差异极大，扩散模型难以用一个条件刻画；
2. **合成数据的可靠性**：MARL 的转移依赖联合动作，扩散模型建模 $(s, \mathbf{a}, s', r)$ 的联合分布维度更高、数据更稀疏，生成样本更容易包含无效联合动作，被 critic 高估后会污染训练；
3. **价值分解兼容性**：生成数据若破坏个体-global 一致性（如生成了 global 奖励与个体动作不匹配），CTDE 的 value decomposition 会被干扰；
4. **计算成本**：扩散模型在 MARL 中需建模更大维度的数据，附录表 6 已显示单智能体就增加 3 小时/20% 训练时间，MARL 中会被进一步放大。

## 7. 上述问题在 MARL 中是否不同、更具挑战性？

**是的，且 CFDG 的核心机制会被非平稳性与联合动作直接冲击：**

- **非平稳性**：CFDG 隐含的假设是"online 数据整体贴合当前策略、可以用一个条件标签稳定刻画"。但 MARL 中其他智能体策略持续变化，$D_{\mathrm{on}}$ 的分布随时间漂移，固定更新频率（10K/100K 步）下扩散模型学到的"online 类别"可能是多个历史策略的混合，8:2 的重加权反而可能把过时联合分布的权重放大；标签设计与更新频率的调参难度显著上升。
- **联合动作**：OOD 问题从动作级变为联合动作级，离线数据对联合动作的覆盖近乎空白，扩散模型生成"看似合理但从未真实出现"的联合转移的风险大增；且 CFG 的引导方向（离线/在线）在 MARL 中还需考虑"与哪些对手的联合动作"这一额外维度。
- **不过有一点对 MARL 更有利**：本文的实证结论"重加权偏向当前策略数据 + 离线数据提供多样性"恰好对应 MARL 中最稀缺的资源（高质量在线联合样本），因此如果标签与可靠性问题能解决，CFDG 范式在 MARL 中的潜在收益可能比单智能体更大——但这是待验证的假设，本文无任何 MARL 实验。

## 8. 本文的不足（含迁移到 MARL 的不足）

1. **无理论保证**：为何 CFG 双类别生成 + 8:2 重加权有效，只有直觉与 JS 散度、消融实验，无误差分析（生成数据偏差如何影响 critic 的界）；
2. **超参较多且跨任务固定**：$r=1/3$、在线合成：离线合成 $=8:2$、$w$（引导强度）、更新频率（10K vs 100K 不统一）、$p_{\mathrm{uncond}}$、128 步去噪等均需在每任务上适用同一组值，鲁棒性论证有限（仅 5 种子、部分环境方差很大，如 hopper-r 上 IQL 16±13 → 10±1 反而下降、APL 51±30 → 30±40 下降）——表 1 中 hopper-random 上 CFDG 反而**劣化**基底，说明重加权偏向在线数据在低质量数据集上可能有害，作者未深入分析；
3. **生成数据无价值/不确定性筛选**：与 SUNG 式不确定性过滤不同，本文所有合成数据直接进入训练，MARL 中这会放大虚假联合动作的危害；
4. **计算与时间开销**：扩散模型引入额外 20% 训练时间与 1G 显存（表 6），按间隔更新只是折中；
5. **迁移到 MARL 的特有不足**：条件标签无法刻画策略非平稳性；联合转移分布建模困难、生成样本可靠性无法保证；与 value decomposition/信用分配不兼容的问题未讨论；8:2 固定重加权无法适应 MARL 中质量差异极大的离线联合数据；全部实验为单智能体。

## 9. 对照算法、发表信息、实验与消融

### 基线及出处

| 基线 | 出处 |
|---|---|
| IQL（基底+对比） | Kostrikov et al., arXiv 2021（未正式发表） |
| PEX | Zhang et al., **ICLR 2023** |
| APL | Zheng et al., **AAAI 2023**（Vol. 37） |
| SynthER（模型式数据增强对比） | Lu et al., **NeurIPS 2023** |
| EDIS（能量引导扩散，SOTA 对比） | Liu et al., **ICML 2024** |
| 其他引用方法：CQL、AWAC、Off2On、Cal-QL、SUNG、Diffuser、Decision Transformer 等 | NeurIPS 2020 / arXiv 2020 / CoRL 2022 / NeurIPS 2023 / arXiv 2023 / ICML 2022 / NeurIPS 2021 |

### 主实验

- **环境**：D4RL **Locomotion**（halfcheetah/hopper/walker2d × random/medium-replay/medium/medium-expert，共 12 任务）、**AntMaze**（medium-play/medium-diverse/large-play/large-diverse，不做 umaze）；附录 A.1 补 **Adroit**（relocate/pen/door-human-v1）；
- **协议**：IQL/PEX 离线 1M 步 + 在线 1M 步；APL 离线 1M 步 + 在线 0.1M 步（与原论文一致）；5 个随机种子；
- **结果**（表 1）：Locomotion 总分 IQL 810→933、PEX 890→1024、APL 972→1081；AntMaze 总分 IQL 250→266、PEX 264→284；Adroit（表 4）CFDG 在三个 human 数据集上显著优于 IQL 基座与 EDIS，平均提升 50%+。

### 问题验证/分析实验

- **分布分析**：t-SNE（图 1）对比 EDIS 与 CFDG 生成数据的分布形态；
- **JS 散度**（表 2）：离线 0.69 / EDIS 生成 0.56 / CFDG 生成 0.50（越低越好），验证"生成数据更贴近在线数据"的核心假设；
- **与模型式方法对比**（图 2、图 3）：IQL 基底上 12 个 Locomotion 任务的学习曲线，CFDG 全程优于 SynthER 与 EDIS，halfcheetah 上聚合提升约 15%。

### 消融实验（§4.3，表 3）

在 Locomotion 三环境 × 四数据集上对比三个变体：
- CFDG w/o guidance（退化为标准扩散生成）；
- CFDG w/o offline DA（只做在线数据增强，相当于 SynthER 式）；
- 完整 CFDG。

结果：完整版平均 82.10 vs 74.13（无引导）vs 76.77（无离线增强），证明 **classifier-free guidance** 与**双类别数据增强**两个组件都是必要的。

## 10. 核心 Claims 及证明情况

| 核心 Claim | 证明形式 |
|---|---|
| C1：现有方法生成的数据与在线数据仍有 gap（问题分析） | 无理论；**问题验证实验**：t-SNE（图 1）+ JS 散度（表 2） |
| C2：双类别 CFG 同时增强离线与在线数据优于单类增强与标准扩散 | 无理论；**主实验**（表 1 接入 IQL/PEX/APL）+ **消融**（表 3：去掉 guidance 或去掉离线增强均退化）+ 对比 SynthER/EDIS（图 2/3） |
| C3：重加权（偏向在线合成 8:2）使生成数据更贴合在线策略、提升性能且保持稳定 | 无理论；表 2 的 JS 散度 + 主实验；**反例存在**：hopper-random 上性能反而下降，说明该 claim 对数据集质量敏感 |
| C4：CFDG 是通用模块，可接入不同数据使用范式的 O2O 算法 | 无理论；跨 IQL/PEX（1:1 范式）与 APL（OORB 范式）的实验（表 1） |
| C5：CFDG 在多个 D4RL 域上超越 SOTA 数据增强方法，平均提升 15% | 无理论；Locomotion/AntMaze 主实验 + Adroit 附录实验 + 学习曲线 |

**总结**：本文全部 claims 均由实验支撑（分布分析、JS 散度、主实验、消融），无任何理论证明。就您第 5 问关注的"离线知识保留与阻碍"而言，本文从数据角度给出了明确立场：**离线数据值得保留（多样性价值），但必须经条件生成"再语境化"并以重加权控制注入比例，否则直接对齐或直接使用会产生分布偏移这一阻碍**——这一逻辑在 MARL 中会因非平稳性与联合动作而更具挑战性，但也正因为 MARL 中在线联合样本稀缺，"生成式数据增强 + 偏向当前策略的重加权"可能成为比单智能体场景更有价值的方向。