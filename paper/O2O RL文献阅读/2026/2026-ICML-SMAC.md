# 文章整体内容概述

本文研究 **Offline-to-Online (O2O) 强化学习**：先用离线 RL 预训练 actor-critic，再用在线 value-based RL（如 SAC）微调。作者观察到：**现有离线 RL 方法得到的强 checkpoint 一经在线微调，性能往往立刻崩溃（initial performance drop）**。作者给出了一个**几何解释**：离线算法收敛到的高回报参数点与在线算法收敛的最优点之间，被**低回报的“山谷”（low-reward valley）**隔开，梯度更新必须穿越这片低回报区域才能到达在线最优点，导致性能先崩后回升。

为验证该假设，作者引入**线性连通性（linear connectivity）**概念：若两个 checkpoint 在参数空间直线连线上的回报单调（不下降），则称它们线性连通。实验（插值实验、参数平面 reward landscape、t-SNE 轨迹投影）表明：**凡是在线微调出现性能 drop 的任务，离线 checkpoint 与 SAC 微调 checkpoint 之间都存在低回报山谷**；hopper 任务中不出现 drop 的方法其插值曲线也是单调的——连通性与 O2O 稳定性高度吻合。

在此诊断基础上，作者提出 **SMAC（Score Matched Actor-Critic）**。核心思想来自最大熵 RL 的恒等式：在 SAC 最优解处，

$$  
\nabla_a \log \pi^{\ast}(a|s) = \frac{1}{\alpha}\nabla_a Q^{\ast}(s,a)  
$$

即在线 SAC 最优点天然满足"策略 score 与 critic 动作梯度对齐"这一**局部结构性质**。SMAC 在离线阶段就用一个 RvS（Reinforcement via Supervision）条件化的扩散模型估计数据集动作分布的 score $\nabla_a \log \pi^{\mathcal{D}}(a|s)$，并在 SAC critic 损失上加入 score-matching 正则项：

$$  
\mathcal{L}^{SM}(\theta,\psi)=\mathbb{E}_{s\sim\mathcal{D},\,a\sim\mathcal{B}(\mathcal{A})}\left[\left\|\nabla_a Q_\theta(s,a)-\alpha_\psi(s)\,\epsilon_\omega(s,a,w,1)\right\|_2^2\right]  
$$

$$  
\mathcal{L}^{SMAC}(\theta,\psi)=\kappa\,\mathcal{L}^{SM}(\theta,\psi)+\mathcal{L}^{AC}(\theta)  
$$

使离线 critic 的局部几何结构提前对齐在线 SAC 会趋近的结构，从而让离线最优点与在线最优点处于同一座"山丘"上，而非隔着山谷。此外离线阶段使用 **Muon 优化器**作为经验稳定器（倾向于收敛到更平坦的极小值）。

**主要实验结果**：在 6 个 D4RL 任务上，SMAC 微调到 SAC/TD3 在 6/6 任务上无性能 drop；在 4/6 任务上相对最强 baseline 降低在线 regret 34–58%；在 Table 1 的归一化 regret 上，SMAC 对全部四种在线算法（SAC/TD3/TD3+BC/AWR）均为最低。机制验证（连通性实验）、消融（RvS、Muon、$\kappa$、batch size、计算量匹配）均支持其 claim。

---

## 问题 1：解决 O2O RL 中的什么问题？为什么现有算法没解决？

**解决的问题**：离线预训练的 actor-critic checkpoint 在用标准在线 value-based RL（SAC、TD3）微调时发生的**初始性能崩溃（initial transfer drop）**。这在在线交互昂贵/有时间限制/物理部署的场景中是致命的——即使策略最终能恢复，早期的崩溃也已抹杀了离线预训练的价值。

**为什么现有算法没解决**（本文视角）：

1. **问题的几何本质被忽视**：现有方法要么在离线和在线两个阶段都持续正则化 critic（CalQL/CQL、Nakamoto et al. 2023、Wen et al. 2024），要么在在线阶段加策略正则使 actor 不偏离数据（TD3+BC、PEX、Zhang et al. 2023a）。前者把问题**错误设定（mis-specify）**——在线阶段仍保持悲观约束，限制了适应速度；后者**欠设定（under-specify）**——BC 约束虽然稳定但收敛到次优策略，适应慢、最终性能低（文中实验证明：TD3+BC 能避免 drop 但收敛到次优；同目标微调 IQL→IQL、CalQL→CalQL 也稳定但远慢于 SMAC→SAC，见附录 I）。
2. 这些方法只关注"离线 checkpoint 有多强"或"在线更新有多稳"，**没有关注离线最优点与在线最优点在参数空间几何上是否连通**。本文证据表明：离线性能高（如 Door 大数据集上所有方法离线性能都很强）并不能保证平滑迁移——baselines 仍然崩溃（附录 J）。所以这是一个**连通性/几何问题**，而非单纯的性能或正则化强度问题，这是以往文献没有诊断和解决的。

---

## 问题 2：核心创新点

1. **诊断性创新**：首次（在 O2O RL 中）用**线性连通性**与 reward landscape 可视化将 O2O 迁移 drop 与"离线/在线最优点之间的低回报山谷"联系起来，并提供了插值实验、参数平面等高线图、t-SNE 轨迹投影三类证据（Section 5）。
2. **方法性创新——Score-matching critic 正则化**：利用最大熵恒等式 $\nabla_a \log \pi^{\ast}(a|s)=\frac{1}{\alpha}\nabla_a Q^{\ast}(s,a)$，在离线阶段将 critic 的动作梯度正则到**数据集动作 score** 上，使离线 checkpoint 具备在线 SAC 最优点的局部结构性质，从而在几何上"预对齐"在线优化路径，而非靠在线阶段持续约束。
3. **RvS 条件化 score 估计**：扩散 score 模型以轨迹回报 $w$（min-max 归一化，$w{=}1$ 表示最优轨迹中的动作）为条件，使 score 估计偏向**高回报动作**而非数据集平均动作，更贴近 SMAC 背后的最优策略结构动机（附录 E 证明去掉 RvS 会削弱迁移）。
4. **经验组件**：离线阶段使用 Muon 优化器（谱范数最速下降，倾向更平坦极小值）作为稳定器（附录 N）。
5. **目标定位**：SMAC 是纯离线算法，目标是产出能**与任意标准在线 actor-critic（SAC/TD3/TD3+BC/AWR）兼容**的 checkpoint，而不是绑定某个特殊在线算法（与 Yu & Zhang 2023 等需要特殊参数化的方法不同）。

---

## 问题 3：核心模块及其在算法流程中的作用

**模块一：RvS 条件化扩散 score 模型 $\epsilon_\omega$**

- 作用：离线静态数据上训练一次（CleanDiffuser 库 + IDQL-MLP 架构），用扩散模型在最小小噪声步 $k{=}1$ 处的预测近似 $\nabla_a\log p(a|s,w)$。以轨迹回报 $w$ 为条件，使估计偏向高回报动作。
- 该模型在离线训练后**不再更新**，在线阶段直接使用其 score 估计（这也是不评估"在线迁移到 SMAC"的原因：在线更新扩散模型代价高且会灾难性遗忘）。

**模块二：Score-matching critic 正则项 $\mathcal{L}^{SM}(\theta,\psi)$**

- 对每个状态 $s\sim\mathcal{D}$，以 50/50 混合从当前策略 $\pi$ 和动作空间均匀分布采样动作 $a\sim\mathcal{B}(\mathcal{A})$，最小化

$$  
\mathcal{L}^{SM}(\theta,\psi)=\mathbb{E}\left[\left\|\nabla_a Q_\theta(s,a)-\alpha_\psi(s)\,\epsilon_\omega(s,a,w,1)\right\|_2^2\right]  
$$

- 其中 $\alpha_\psi(s)$ 是可学习的状态条件系数（用网络学习比例系数，使正则项表达力更强）。
- 机制：在数据支持的动作上，使 critic 动作梯度对齐 score 的梯度结构（SAC 最优点满足的结构）；在数据外动作上，score 场作为**结构化悲观**——值函数朝数据支持动作倾斜回去，而非 CQL 式的均匀惩罚。

**模块三：标准 SAC critic 损失 + 策略损失**

$$  
\mathcal{L}^{AC}(\theta)=\mathbb{E}_{s,a,r,s'\sim\mathcal{D}}\left[(Q_\theta(s,a)-r-\gamma Q_{\bar\theta}(s',a'))^2\right]  
$$

$$  
\mathcal{L}^{SMAC}(\theta,\psi)=\kappa\mathcal{L}^{SM}(\theta,\psi)+\mathcal{L}^{AC}(\theta),\qquad \mathcal{L}^{\pi}(\phi)=\mathbb{E}\left[-Q_\theta(s,a)+\log\pi_\phi(a|s)\right]  
$$

并沿用 target Q-network 与 Q 集成（REDQ 式）抑制 Q 过估计。

**模块四：Muon 优化器（仅离线阶段）**

- 用谱范数最速下降替代 Adam 的 max-of-max 范数步，收敛到更平坦、条件数更小的极小值，与更强下游迁移相关联；附录 N 显示换 Adam 后 SMAC 在 3/6 任务出现迁移 drop。

**完整算法流程（Algorithm 1，仅离线阶段）**：

1. 用 RvS + 扩散损失预训练 $\epsilon_\omega$（静态数据，约 3–4 GPU 小时）。
2. 初始化 $Q_\theta$、$\pi_\phi$、$\alpha_\psi$、target network $\bar\theta=\theta$。
3. 每个离线 step：
    - 从 $\mathcal{D}$ 采样 batch $\{(s_i,a_i,r_i,s_i')\}$；
    - 为 score-matching 采样动作 $\bar a_i$（一半来自 $\pi$、一半均匀）；
    - $\theta \leftarrow \theta - \delta_Q\,\text{Muon}(\nabla_\theta\mathcal{L}^{SMAC})$；$\psi \leftarrow \psi - \delta_\omega\,\text{Muon}(\nabla_\psi\mathcal{L}^{SMAC})$；$\phi \leftarrow \phi - \delta_\pi\,\text{Muon}(\nabla_\phi\mathcal{L}^{\pi})$；
    - Polyak 更新 $\bar\theta \leftarrow \lambda_Q\theta+(1-\lambda_Q)\bar\theta$。
4. 输出 $(Q_\theta,\pi_\phi)$ 作为在线 SAC/TD3 等的初始化，在线阶段**不加任何特殊正则**，完全交给标准在线算法。

---

## 问题 4：是否有理论证明？

本文**没有**提供收敛性或 regret 界的正式定理证明，但包含两类"理论性质"推导：

1. **最大熵恒等式（Section 2）**：在最大熵 RL 最优解处策略呈 Boltzmann 形式 $\log\pi^{\ast}(a|s)=\frac{1}{\alpha}Q^{\ast}(s,a)-\log\int\exp(\frac{1}{\alpha}Q^{\ast}(s,\bar a))d\bar a$，对 $a$ 求梯度消去归一化项即得 $\nabla_a\log\pi^{\ast}(a|s)=\frac{1}{\alpha}\nabla_aQ^{\ast}(s,a)$。这是 SMAC 正则化的动机来源。
2. **附录 U：SAC 与 PPO 在最优点满足该恒等式**。
    - 对 SAC：假设 $\nabla_a\log\pi(a|s)=\frac{1}{\alpha}\nabla_aQ(s,a)$，积分并归一化得 $\pi(a|s)=\frac{\exp(\frac{1}{\alpha}Q(s,a))}{Z(s)}$，SAC 策略损失恰为 $D_{KL}\left(\pi\|\frac{\exp(Q/\alpha)}{Z}\right)$，故 $\pi$ 最小化损失——即恒等式与 SAC 最优一致（且附录 F 用 Reacher-v2 的 2D 动作空间可视化验证了训练中恒等式大致成立）。
    - 对 PPO：类似地将 PPO 收敛时的目标写成 $-D_{KL}\left(\pi\|\frac{\exp(Q/c_2)}{Z}\right)+V(s)-Z(s)$ 的形式，说明满足 $\nabla_a\log\pi=\frac{1}{c_2}\nabla_aQ$ 的策略最大化该目标。

需要强调：恒等式被用作**正则化目标的动机**，而非假设数据集策略最优；连通性假设也未被形式化证明，主要靠实验证据支撑。

---

## 问题 5：是否讨论了 Offline 阶段哪些知识值得保留、Online 阶段如何用、以及 Offline 知识是否会阻碍 Online？

**有讨论，但用的是"几何结构"而非"知识保留"的语言**：

1. **值得保留的知识 = critic 的局部动作梯度结构**。SMAC 主张离线阶段值得保留的不是 BC 式的行为克隆约束（那会在在线阶段限制探索、导致次优），也不是 CQL 式的 uniform 悲观惩罚，而是 $\nabla_a Q(s,a)$ 与数据集 score 对齐的**局部几何**：它既在数据支持区给出与在线 SAC 一致的改进方向，又在 OOD 区提供"值坡流回数据支持动作"的结构化悲观。
2. **Online 阶段的使用方式**：不加任何特殊机制——直接以 SMAC checkpoint 初始化标准在线 SAC/TD3，在线算法从第一天就能沿着"已经对齐"的梯度结构继续改进。这正是问题设定（Section 3）中的两个标准：稳定迁移 $\mathbb{E}[\mathcal{J}(\pi_1)]\ge\mathbb{E}[\mathcal{J}(\pi_0)]$ 与低在线 regret $\mathrm{Regret}_N=\sum_{t=1}^N(\mathcal{J}(\pi^{\ast})-\overline{\mathcal{J}(\pi_t)})$。
3. **Offline 知识是否阻碍 Online**——这正是本文的核心论点：**现有离线方法的"知识"（CQL 的悲观形状、IQL/TD3+BC 的隐式 BC 约束）在在线阶段构成阻碍**。证据包括：
    - 附录 I：同目标微调（IQL→IQL、CalQL→CalQL）稳定但远慢于 SMAC→SAC——说明阻碍来自"离线目标与在线目标不匹配"，继续沿用离线目标虽稳但限制适应；
    - Section 7.3：离线阶段带策略约束的 IQL/TD3+BC 与在线 BC 正则（TD3+BC 微调）更匹配；而 CalQL/SMAC 这类"只训 critic 自由最大化"的离线算法与 TD3（无 BC）微调更匹配——离线优化目标与在线更新规则不配对会造成性能恶化（如 SMAC→TD3+BC 在 kitchen/walker2d 因 BC 项拉回次优动作而长期退化）；
    - 附录 N：Muon 收敛到的平坦极小值反而更难从中逃出（Ibayashi & Imaizumi 2023; Kleinberg et al. 2018）——这说明某些"好的离线性质"（平坦性）对迁移有双刃剑效应。

---

## 问题 6：算法是否适合迁移到 MARL？分析

**结论：思想可迁移，但直接照搬面临实质性挑战，需要重新设计。**

**适合迁移的理由**：

1. O2O 问题在 MARL 中同样存在甚至更严重——多智能体离线数据昂贵（真实机器人/多传感器系统），预训练 + 在线微调是现实需求；初始性能崩溃在多智能体系统中代价更高（协调崩溃可能是灾难性的）。
2. "离线最优点与在线最优点是否连通"的几何诊断与 MARL 无关，完全可移植；score-matching 正则化框架（critic 动作梯度对齐行为 score）在多智能体 critic（CTDE 下的集中式 critic）上数学形式不变。
3. 多智能体领域已有扩散策略与行为模型的积累（如基于扩散的多智能体行为克隆、MADiffusion 类工作），score 估计有现成载体。

**挑战**：

1. **联合动作空间维数爆炸**：score 估计 $\nabla_{\boldsymbol a}\log\pi^{\mathcal{D}}(\boldsymbol a|s)$ 需在联合动作空间上训练扩散模型，维度随智能体数线性增长，扩散 score 估计质量与计算代价急剧恶化；而 score 质量直接决定正则项有效性（本文 κ 消融显示 under-regularization 是主要失效模式）。
2. **非平稳性**：离线数据中其他智能体的策略固定，而在线阶段所有智能体同时在学习——行为分布 $\pi^{\mathcal{D}}$ 本身在在线阶段失去意义（数据是被旧联合策略收集的），联合 score 对齐可能把 critic 锚定在过时的交互模式上，结构化悲观在 MARL 中可能变成"协调惰性"。
3. **信用分配**：集中式 critic $Q(s,\boldsymbol a)$ 的动作梯度 $\nabla_{a_i}Q$ 对单个智能体动作的 score 匹配，在存在强交互（非线性协调）时，$\nabla_{a_i}Q$ 与 $\nabla_{a_i}\log\pi^{\mathcal{D}}(\boldsymbol a|s)$ 的关系不再有清晰的单智能体 Boltzmann 解释——附录 U 的恒等式推导依赖单智能体 MDP 结构，多智能体下该恒等式仅在独立策略、独立学习的理想化假设下逐智能体成立。
4. **部分可观测与异构策略**：MARL 中策略以观测历史为条件，score 建模对象变成 $\pi(a_i|\tau_i)$，且异构智能体需要不同的 score 模型。
5. **连通性定义本身变复杂**：参数空间中联合策略的 landscape 是多智能体策略空间的乘积结构，mode 数量随智能体数组合爆炸，"同一座山丘"的几何假设更难满足。

---

## 问题 7：（结合问题 5）在 MARL 中，知识保留/运用/阻碍的问题有何不同？

**保留什么**：单智能体 SMAC 保留的是 critic 动作梯度的局部结构。MARL 中，除了 critic 几何，更有价值的保留对象可能包括：

- **协调结构/联合动作模式的 score 几何**：即"哪些联合动作组合在数据中是高回报的"，这是单智能体没有的知识类型；
- 但联合 score 恰好是**最难保留且最容易过时**的：它编码的是"旧队友行为下的协调模式"。

**如何运用**：SMAC 在线阶段"零机制直接用"的设计在 MARL 中风险更大。因为在线阶段所有智能体策略同步漂移，离线学到的联合 score 锚点会快速失效；而单智能体中数据集策略虽也非最优，但其 score 作为局部先验在短时间在线训练中仍近似有效（本文在线实验仅 200k 步）。MARL 可能需要**在线持续校正** score 模型或退化为逐智能体 score（但这又破坏协调信息）。

**哪些 Offline 知识会构成阻碍——MARL 中更严重**：

- 单智能体中，阻碍主要来自"离线目标与在线目标的几何错位"（CQL 悲观形状、BC 约束）。MARL 新增两类阻碍：
    1. **交互过拟合**：联合 score 匹配把 critic 锚定在旧联合策略的交互模式上，在线阶段队友策略变化会主动惩罚正确的协调探索——即离线知识从"被动阻碍"变成"主动误导"。
    2. **非平稳性放大的连通性问题**：在线 SAC 在单智能体中是固定优化过程，MARL 中每个智能体的"在线优化目标"随队友策略漂移而持续变化，离线 checkpoint 对齐的"在线最优点"本身在移动；连通性需要在**策略空间的轨道**上定义，而非两点之间的直线。
- **联合动作带来的额外挑战**：结构化悲观（值坡流回数据支持的联合动作）在单智能体中防止 OOD 崩溃，在 MARL 中可能阻止发现数据从未覆盖但回报更高的**新协调模式**（如从各自为战转向分工），这是单智能体 OOD 探索中不存在的损失。

---

## 问题 8：本文仍存在的不足（含迁移到 MARL 的不足）

1. **成本问题**：扩散 score 模型训练额外需要 3–4 GPU 小时（L40S），完整离线阶段耗时约为 IQL 的 3 倍、CalQL 的 2 倍（附录 K 做了计算量匹配对照，额外计算不能解释性能差距，但成本仍是实际部署障碍）。
2. **方法动机绑定熵正则 actor-critic**：SMAC 最直接适用于 SAC 类结构；对 PPO 只给了最优点的恒等式论证（附录 U.2），实际只验证了 SAC/TD3/TD3+BC/AWR 四种在线算法，不保证对任意在线更新规则的迁移（Section 9 自认）。
3. **连通性证据的适用范围**：连通性/landscape 证据主要来自 Kitchen 等有限任务族，t-SNE 只能做投影不能刻画全局几何；优化轨迹不必为直线（作者自己承认 Figure 4 只能作为佐证）。
4. **对在线超参敏感**：附录 D 显示在线 batch size 显著影响平滑迁移（大 batch 适应更快），说明稳定性部分依赖在线优化配置，而非完全由离线 checkpoint 决定。
5. **κ 需任务级调参**（附录 M）：κ 过小 under-regularization 失效，虽对粗粒度取值鲁棒，但仍需逐任务族选择。
6. **MARL 迁移的额外不足**（结合问题 6、7）：联合动作 score 估计成本高且质量难保证；离线联合 score 在在线非平稳性下会过时甚至主动误导；最大熵恒等式的理论动机在独立学习假设之外不成立；连通性诊断在多智能体参数空间中更难验证；critic 集成 + 扩散 score 模型 + 多智能体训练的组合在计算上难以扩展。

---

## 问题 9：对照算法、发表信息、实验设置、消融与验证实验

**对照算法**（均为离线 RL，可产出可被标准在线算法微调的 actor-critic）：

|算法|发表信息|
|---|---|
|CQL (Kumar et al., 2020)|arXiv 2020（NeurIPS 2020）|
|CalQL (Nakamoto et al., 2023)|NeurIPS 2023|
|IQL (Kostrikov et al., 2022)|ICLR 2022|
|TD3+BC (Fujimoto & Gu, 2021)|NeurIPS 2021|

（未纳入 Yu & Zhang 2023，因其参数化不直接返回标准 actor-critic；未纳入 PEX/PROTO/O3F 等专用 O2O 在线算法，因其与本文"一个离线 checkpoint 适配多种在线算法"的问题设定相反，见附录 S.2。）

**主实验**：6 个 D4RL 任务——hopper-medium-replay-v2、walker2d-medium-replay-v2（无限时域，用 CQL 变体，无 MC return）、kitchen-partial-v0、door-binary-v0、pen-binary-v0、relocate-binary-v0（binary 版为移除失败轨迹、达成目标即终止的修改版，附录 S.1）。协议：离线预训练 → 用离线策略采 5,000 条 on-policy 数据 warm-start replay buffer（沿用 Nakamoto et al. 2023; Zhou et al. 2024）→ 在线 200k 步，batch 中 50% 离线 + 50% 在线数据。每个离线-在线组合 5 seeds（Figure 3 用 4 seeds）。

**问题验证实验（连通性）**：

- Figure 2：Kitchen 上由 $\theta_{pre}$、$\theta_{SAC}$、$\theta_{TD3+BC}$ 三点张成的参数平面 reward 等高线图（附录 R 的方法）；
- Figure 3：离线 checkpoint 与 SAC 微调 checkpoint 之间的线性插值 $\theta(t)=\theta_{offline}t+(1-t)\theta_{online}$ 的回报曲线（kitchen/door/hopper）；
- Figure 4：预训练 + 两条微调轨迹 checkpoint 的 t-SNE 投影；
- Figure 13/14：Kitchen 上同目标微调（IQL→IQL、CalQL→CalQL）的曲线与插值，验证"目标匹配则连通且稳但慢"。

**消融实验**：

- 附录 D / Figure 8-9：离线 batch size（32–256，在线固定 512）与在线 batch size（64–1024，离线固定 32）；
- 附录 E / Figure 10：去掉 RvS 条件化训练扩散模型；
- 附录 N / Figure 19-20：baselines 换 Muon；SMAC 换 Adam；
- 附录 M / Figure 18：κ 扫描（0–160）；
- 附录 K / Figure 16：计算量匹配（IQL 训 3 倍步数、CalQL 训 2 倍步数）；
- 附录 J / Figure 15：Door 超 1000 万样本大数据集；
- 附录 L / Figure 17：额外任务 hopper-medium（无极端失败轨迹）与 OGBench Cube（预设动作生成数据，弱化行为策略假设）；
- 附录 H / Figure 12 + Table 4：AWR 微调；
- Tables 2–5：四种在线算法下的逐环境 regret（含 R* 的 best/second/third 着色）。

**主结果**：Figure 5（→SAC）、6（→TD3）、7（→TD3+BC）、Table 1（归一化平均 regret：SMAC→SAC 0.031、→TD3 0.090、→TD3+BC 0.226、→AWR 0.380，全部为最低）。

---

## 问题 10：核心 claims 及其证明情况

**Claim 1（诊断）**：O2O 迁移中的初始性能 drop 与离线 checkpoint 和在线 SAC 最优点之间缺乏线性连通性（存在低回报山谷）相吻合。

- _证明形式_：**问题验证实验**。Figure 3 插值（4/5 非 SMAC 方法在 kitchen/door 有山谷 ↔ 对应任务 Figure 5 中出现 drop；hopper 中 3/4 方法插值单调 ↔ 无 drop）；Figure 2 参数平面可视化；Figure 4 t-SNE 投影佐证；Figure 14 同目标微调连通且稳的对照实验。**无线理论证明**（连通性未被形式化为定理）。

**Claim 2（方法）**：SMAC 的 score-matching 正则化产生与在线 SAC 最优点线性连通、且可被标准在线算法直接平滑改进的离线 checkpoint。

- _证明形式_：机制动机来自恒等式 $\nabla_a\log\pi^{\ast}(a|s)=\frac{1}{\alpha}\nabla_aQ^{\ast}(s,a)$ 的推导（Section 2）与 SAC/PPO 在最优点满足该恒等式的推导（附录 U，属**解析推导而非收敛定理**）；实验上 Figure 3 中 SMAC 插值单调、Figure 2 中 SMAC 与 SAC 最优点同处一座山丘。

**Claim 3（性能）**：SMAC 在 6 个 D4RL 任务上实现平滑迁移。

- _证明形式_：**主实验**。Figure 5：SMAC→SAC 在 6/6 任务无初始 drop（baselines：CalQL 3/4、IQL 4/6、TD3+BC 5/6 任务有 drop）且达到最高性能；Figure 6：→TD3 6/6 平滑；Figure 7：→TD3+BC 4/6 平滑（kitchen/walker2d 因 BC 项退化，作者有分析）；Figure 12：→AWR 4/6 平滑且 5/6 任务优于 baselines。

**Claim 4（效率）**：SMAC 显著降低在线 regret。

- _证明形式_：Table 1（归一化 regret 对四种在线算法全最低）与 Tables 2–5（4/6 任务 regret 比最强 baseline 低 34–58%，另两环境为第二、第三低），5 seeds 均值 ± 标准误。

**Claim 5（归因）**：score-matching 是主机制，RvS 与 Muon 是重要但次要组件。

- _证明形式_：**消融实验**。附录 E（去 RvS 迁移变弱）、附录 N（SMAC 换 Adam 后 3/6 任务出现 drop；baselines 加 Muon 无改善甚至崩溃——Figure 19）、附录 M（κ 敏感性，under-regularization 是主要失效模式）。

**Claim 6（排除替代解释）**：离线性能强或离线计算多并不能解释迁移差距。

- _证明形式_：附录 J（Door 千万级数据，baselines 离线都强但仍崩溃、SMAC 立即改善）；附录 K（计算量匹配的 IQL/CalQL 仍显著差于 SMAC，Figure 16）。

总体评价：本文属于"**诊断假设 + 机制方法 + 系统性实验验证**"型论文，理论部分为动机性推导（恒等式及其在最优点的成立性），核心因果 claim 主要依靠精心设计的对照与消融实验链支撑，证据链相当完整。