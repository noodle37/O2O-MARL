# 文章总结：Penalizing Infeasible Actions and Reward Scaling in Reinforcement Learning with Offline Data (PARS)

**发表信息**：ICML 2025（PMLR 267），LG AI Research 与 KAIST 联合发表（作者团队与您的第一篇 OPT 论文是同一团队，通讯作者同为 Woohyung Lim）。

**一句话概括**：本文从**函数逼近器（critic 网络）本身的行为特性**出发解决 O2O RL 的 Q 值外推误差问题：证明了 ReLU MLP 在数据范围外呈线性外推是外推误差的根源，提出两个极简改动——带 LayerNorm 的奖励缩放（RS-LN，让网络"分辨"OOD 动作并使其 Q 值自然下降）与不可行动作惩罚（PA，在可行域外硬约束 Q 值趋向 $Q_{\min}$），组合成基于 TD3+BC 的 PARS 算法，在离线训练和在线微调两阶段均达到 SOTA，尤其在极难的 AntMaze Ultra 上独家成功。

---

## 1. 解决 O2O RL 中的什么问题？为什么现有算法没解决？

**问题**：离线数据的 Q 值外推误差（extrapolation error）——OOD 动作的 Q 值被高估，导致策略选择错误动作；这一问题贯穿离线训练和在线微调两个阶段。本文进一步指出**关键根源**：ReLU 类 MLP 在训练数据范围之外收敛为线性函数（Xu et al. 2021），把数据凸包边界上观测到的趋势沿边界向外持续延伸。本文首次将 OOD 动作细分为：

$$  
a \in \begin{cases} \mathcal{A}_{\mathrm{OOD\text{-}in}}(s), & \text{if } a \in \mathrm{Conv}(\mathcal{A}_s) \\ \mathcal{A}_{\mathrm{OOD\text{-}out}}(s), & \text{if } a \notin \mathrm{Conv}(\mathcal{A}_s) \end{cases}  
$$

其中 $\mathcal{A}_{\mathrm{OOD\text{-}out}}$ 完全无数据约束，误差不可控增长。表 1 的 max-Q 动作范数实验证明：即使加了 LayerNorm（RLPD/Ball et al. 2023 的做法），max-Q 动作仍几乎总是落在 $\mathcal{A}_{\mathrm{OOD\text{-}out}}$ 与可行域边界交汇处——**LN 只能把 Q 值约束在权重范数的界内，却不能让数据范围外的 Q 值下降**。

**现有方法为何未解决**：

- **CQL/MSG/SVR/SAC-RND 等基于密度差或辅助行为模型的惩罚**：不区分 $\mathcal{A}_{\mathrm{OOD\text{-}in}}$ 与 $\mathcal{A}_{\mathrm{OOD\text{-}out}}$，一并惩罚；动作维度升高或任务复杂时近似误差增大；且会殃及附近的 ID 动作；辅助模型从离线迁到在线时还需额外适配；
- **大规模 critic ensemble**（SAC-N/EDAC）：最多需要 500 个 critic，成本极高，且 AntMaze 上仍然无效；
- **LN 类方法**（RLPD）：只 bound、不降低。

本文提出"好的外推"标准（图 3）：$\mathcal{A}_{\mathrm{OOD\text{-}out}}$ 内的 Q 值必须低于数据范围内的最大 Q 值、从凸包边界起走平或下降——这一目标此前没有任何方法直接实现。

## 2. 核心创新点

1. **问题诊断创新**：首次系统刻画 $\mathcal{A}_{\mathrm{OOD\text{-}out}}$ 上的线性外推问题，并用 max-Q 动作范数实验证明 LN alone 无法解决（表 1）；把 OOD 从动作域细化为"凸包内 OOD"与"凸包外 OOD"两层。
2. **RS-LN（Reward Scaling + LayerNorm）**：提出"奖励缩放必须与 LN 联用"的新视角（与以往 Kumar et al. 2020、Kostrikov et al. 2022 等在 AntMaze 上的奖励操纵不同）。机制：增大奖励尺度 $c_{\mathrm{reward}}$ 迫使网络学习更高分辨率的特征（类比 $y=x$ vs $y=5x$ 的分段常数逼近误差），LN 保证输入体积不变（将输入限制在单位球内），二者结合使 ID 动作与 $\mathcal{A}_{\mathrm{OOD\text{-}out}}$ 动作在函数逼近器"感知"中相似度降低——由 NTK $K_\phi(x,x')=\langle\nabla_\theta f_\theta(x),\nabla_\theta f_\theta(x')\rangle$ 度量——于是 ID 区域的正向梯度更新几乎不传导到 OOD 区域，OOD 的 Q 值保持/回归到零附近，相对 ID 区域自然偏低。附带的首次发现：奖励尺度增大激活更多休眠神经元（dormant neurons，Sokar et al. 2023），将**网络表达力与 OOD 缓解首次联系起来**（图 7、表 12）。
3. **PA（Penalizing Infeasible Actions）**：在可行动作域之外（$\mathcal{A}_I$，如 $\pm100$~$\pm1000$ 倍于 $\mathcal{A}_F$ 边界处）设置守卫区间并施加硬约束，把 $\underline{\mathcal{A}}_I = \bigcup_{i=1}^n \{(-\infty, L_{I,i}] \cup [U_{I,i}, \infty)\}$ 内的 Q 值压向 $Q_{\min}=c_{\mathrm{reward}}\cdot r_{\min}/(1-\gamma)$，从外部"压弯"线性外推趋势使其向下。
4. **极简与通用性**：基于 TD3+BC，只加几行代码（附录 E），无辅助模型、无过度保守；RS-LN 还可扩展到 in-sample 方法（IQL，表 5）。

## 3. 核心模块、作用机制与完整流程

### 核心模块

**模块 1：RS-LN**

- 对 TD 目标做奖励缩放：$\mathcal{T}(s,a,s') = c_{\mathrm{reward}}\cdot r(s,a) + \gamma\,\mathbb{E}_{a'\sim\pi_\theta} Q_\phi(s',a')$；
- critic 网络加 LayerNorm（对每层特征 $\hat{\mathbf{h}}_i = \frac{\mathbf{h}_i-\mu_i}{\sqrt{\sigma_i^2+\epsilon}}\odot\boldsymbol{\eta}_i+\boldsymbol{\beta}_i$）；
- 作用：提高特征分辨率 → 降低 ID 与 OOD-out 的 NTK 相似度（图 6 实验证明 $c_{\mathrm{reward}}:1\to100$ 时边界点与 OOD 区域的梯度相似度显著下降）→ OOD 区域 Q 值不再随 ID 更新被抬高 → 相对 ID 区域自然变低。

**模块 2：PA 损失**

$$  
\mathcal{L}_{\mathrm{PA}} = \min_\phi \mathbb{E}_{s\sim\mathcal{D},\, a\in\underline{\mathcal{A}}_I}\big[(Q_\phi(s,a)-Q_{\min})^2\big]  
$$

总损失：

$$  
\mathcal{L}_{\mathrm{Total}} = \min_\phi\left\{\mathbb{E}_{s,a,s'\sim\mathcal{D}}\big[(Q_\phi(s,a)-\mathcal{T}(s,a,s'))^2\big] + \alpha\,\mathbb{E}_{\substack{s,s'\sim\mathcal{D} \\ a\in\underline{\mathcal{A}}_I}}\big[(Q_\phi(s,a)-Q_{\min})^2\big]\right\}  
$$

- 不可行动作按各维均匀采样：$a_i \sim \mathrm{Uniform}(L_{I,i}-\Delta_L, L_{I,i}]$ 或 $\mathrm{Uniform}[U_{I,i}, U_{I,i}+\Delta_U)$（取 $\Delta_L=|L_I|, \Delta_U=|U_I|$）；
- 守卫区间（guard interval）保证惩罚不干扰 $\mathcal{A}_F$ 内的策略评估（图 11：$|L_I|=|U_I|$ 取 100~1000 倍边界最稳）。

**模块 3：critic ensemble（可选）**  
离线策略评估取 ensemble 最小值；在线微调时取子集平均，避免 ensemble 最小值限制在线探索。critic 数：AntMaze 4 个、MuJoCo/Adroit 10 个。

### 完整算法流程

1. **离线阶段**（基于 TD3+BC）：
    - 采样 batch，计算 PA 所需的不可行动作；
    - 用 $\mathcal{L}_{\mathrm{Total}}$ 更新 critic（含 LN 的 Q 网络、奖励缩放后的 TD 目标、PA 项）；
    - 策略改进沿用 TD3+BC：$\max_\pi \mathbb{E}[Q(s,\pi(s))] - \beta\cdot\mathbb{E}[(\pi(s)-a)^2]$；
2. **在线微调阶段**（300k 步）：
    - 在线数据与离线数据按比例混合（AntMaze/Adroit 50/50；MuJoCo 视数据集 5% 或 50% 离线数据）；
    - **保留** RS-LN 与 PA（图 10 证明去掉 PA 会导致不稳定）；
    - 策略改进从"取 ensemble 最小"切换为"子集平均"；UTD=20，其余同 TD3+BC 在线微调。

## 4. 是否有理论证明？

**有，但较初步**（附录 C）：

- **C.1 PARS 损失的收敛性**：定义分段算子

$$  
T_{\mathrm{pars}}^\pi Q(s,a)=\begin{cases} T^\pi Q(s,a), & (s,a)\in\mathrm{ID} \\ \mathbb{E}_{(s',a')\in k\mathrm{NN}(s,a;\mathrm{ID})}[T^\pi Q(s',a')], & (s,a)\in\mathrm{OOD}_{\mathrm{in}} \\ Q_{\min}, & (s,a)\in\mathrm{OOD}_{\mathrm{out}} \end{cases}  
$$

分三种情况证明 $T_{\mathrm{pars}}^\pi$ 在 $\|\cdot\|_\infty$ 下是 $\gamma$-收缩（前两种情况 $\le\gamma\|Q_1-Q_2\|_\infty$，第三种 $=0$），故 Q 函数收敛。这是标准的 contraction 论证。同时给出 guard interval 的设计理由：$Q_{\min}$ 赋值会无意中影响邻近值，故需守卫区间。

- **C.2 RS-LN 在 IQL 上的适用性分析**：统一 TD3+BC 与 IQL 的 critic 损失为 $\mathcal{L}_{\mathrm{Critic}}=\mathbb{E}[(Q_\phi(s,a)-Y)^2]$，$Y=c_{\mathrm{reward}}\cdot r(s,a)+\gamma G(s')$，区别仅在 $G(s')$ 取 $\mathbb{E}_{a'\sim\pi}Q_\phi(s',a')$ 还是 $V_\psi(s')$；论证目标值随 $c_{\mathrm{reward}}$ 增大、LN 防止灾难性高估，因此 RS-LN 可推广。

注意：**RS-LN 的核心机制（NTK 相似度下降、表达力提升）没有定理，只有玩具实验 + NTK 可视化 + 休眠神经元分析**；收缩性证明只覆盖了 PA 损失的形式化算子，不覆盖深度网络实际训练动态。

## 5. 是否讨论了"Offline 知识保留/运用/阻碍"？

**涉及，且立场鲜明，但主要从 critic 正则化角度。**

- **保留与运用**：离线训练得到的策略与 critic 被完整保留到在线微调阶段，**不做任何专门的过渡处理**（"we do not propose new specialized techniques for online fine-tuning"）——本文的核心主张恰恰是：只要 critic 的正则化（RS-LN + PA）做对了，离线知识可以**直接、平滑地**延续到在线阶段。
- **离线知识是否构成阻碍**：明确承认并直接回应了该问题——引言与实验都引用了"离线训练的策略和 critic 有时会阻碍在线微调"（Zhang et al. 2023b; Kong et al. 2024 等）的既有发现，表 3 中 RLPD（不用离线预训练）强于多数 O2O 方法正是这一现象的体现；但 PARS 超过了 RLPD，作者以此论证"**阻碍的根源是 critic 外推误差而非离线预训练本身**"，即离线知识本身值得保留，构成阻碍的是其外推误差形式。这一回答与您的第一篇论文（OPT）的立场互补：OPT 认为该阻碍需要用新价值函数解决，PARS 认为只需修正好单一 critic 的外推行为。
- §6.5 还从**策略提取方式**角度讨论了知识运用：in-sample 学习（IQL 式加权 BC）把策略限制在数据支撑内，虽避免 OOD 动作，但也无法发现 $\mathcal{A}_{\mathrm{OOD\text{-}in}}$ 内的全局最优动作、且严重限制在线探索；TD3+BC 式行为约束策略梯度允许偏离数据支撑（配合修好的 Q 函数），才是离线+在线两阶段都强的关键。

## 6. 是否适合迁移到 MARL？分析

**部分适合，且其"诊断"比"处方"更有迁移价值，但整体挑战不小。**

**适合迁移的理由：**

1. "ReLU critic 在数据凸包外线性外推"是网络性质，与单/多智能体无关，诊断框架（$\mathcal{A}_{\mathrm{OOD\text{-}in/out}}$ 划分、max-Q 动作范数审计、NTK 相似度）可直接搬到 MARL 的集中式 critic 上；
2. MARL 离线 critic 的 OOD 问题更严重（联合动作覆盖更稀缺），RS-LN"降低 ID↔OOD 梯度耦合"的思路对集中式 critic 理论上同样有效；
3. PA 的实现极简（采几个超界动作、加一项 MSE），不依赖行为模型或密度估计，天然避开了 MARL 中行为模型难以建模联合策略分布的痛点；
4. 若未来能将 critic ensemble 减到双 critic（附录 B 的期望），计算量在 MARL 下也可接受。

**迁移挑战：**

1. **$Q_{\min}$ 与奖励缩放在 MARL 中更难设定**：多智能体任务奖励结构复杂（团队奖励/个体奖励混合、负奖励常见），附录 B 已指出 PARS 要求目标 Q 值为正（负 Q 时梯度传导逻辑反转），MARL 中这一前提更常失效；
2. **联合动作维数**：$\mathcal{A}_{\mathrm{OOD\text{-}out}}$ 随联合维度指数增大，RS-LN 能否在合理尺度下维持足够特征分辨率存疑；PA 需要按联合维采样不可行动作，维度灾难；
3. **价值分解**：VDN/QMIX 等分解结构中，PA 施加在集中式 $Q_{tot}$ 上如何传导到个体 Q，会不会破坏 additivity/monotonicity 约束，完全未讨论；
4. **ensemble 成本**：MARL 中 10 critic × N 智能体的开销可能不可接受。

## 7. 上述问题在 MARL 中是否不同、更具挑战性？

**是的，两个 MARL 特性都会放大问题：**

- **非平稳性**：RS-LN 依赖"ID 与 OOD 的相似度结构在训练中相对稳定"这一隐式前提；MARL 中其他智能体策略持续变化使 ID/ID 的边界本身移动，critic 需要持续重新划分凸包内外，NTK 相似度分析不再静态。PA 则是静态几何约束，不随对手变化，这一点反而是其优势——但也意味着它无法区分"因对手策略变化而新出现的可行联合动作"与真正的不可行动作。
- **联合动作**：OOD-out 区域在联合动作空间中占比更大，线性外推的"单维上升即可误导策略"问题（§3.1 末段）在联合维度下概率急剧放大；且守卫区间在关节动作各维上如何取值（有的智能体动作界不同）更复杂。
- 与问题 5 的关联结论：本文"离线知识值得保留、阻碍源于 critic 外推"的论断在 MARL 中会**更脆弱**——MARL 的既有文献同样发现离线预训练策略/价值阻碍在线微调，但由于联合动作外推误差更严重，"修正单一 critic"能否替代"引入新 critic/新策略"（如您第一篇 OPT 的方案）在 MARL 中更不确定，很可能需要两者结合。

## 8. 本文的不足（含 MARL 迁移不足）

1. **理论证明薄弱**：收缩性证明只是对理想化分段算子的标准论证，未覆盖深度网络、函数逼近误差、RS-LN 的 NTK 机制；核心 claim 的支撑仍以实验为主；
2. **正 Q 值假设**（附录 B 自承）：奖励以惩罚为主时机制失效，需要人为平移奖励，任务适配成本高；
3. **超参数多且敏感**：$c_{\mathrm{reward}}\in\{1,...,10000\}$、$\alpha\in[10^{-4},1]$、$\beta$、策略噪声、ensemble 数、离线/在线数据比例、各域 $\gamma$ 均不同（附录 F.3 共约 10 组超参、逐数据集调优）；表 13 显示 AntMaze 单组超参虽可用但性能低于调参版；
4. **仍依赖 critic ensemble**（MuJoCo/Adroit 需 10 个），与"极简"叙事存在张力；
5. **仅覆盖确定性策略+连续动作**：对 SAC 类随机策略、离散动作未验证；
6. **MARL 迁移的特有不足**：$Q_{\min}$/奖励缩放在团队奖励与负奖励下如何设定无答案；PA 作用于联合动作维时守卫区间设计复杂；与 value decomposition 的兼容性完全未讨论；非平稳环境下 RS-LN 的相似度机制可能失效；全部实验单智能体。

## 9. 对照算法、发表信息、实验与消融

### 离线基线（10 个）及出处

|基线|出处|
|---|---|
|TD3+BC（backbone）|Fujimoto & Gu, **NeurIPS 2021**|
|IQL|Kostrikov et al., **ICLR 2022**|
|CQL|Kumar et al., **NeurIPS 2020**|
|SAC-N / EDAC|An et al., **NeurIPS 2021**|
|MSG|Ghasemipour et al., **NeurIPS 2022**|
|SPOT|Wu et al., **NeurIPS 2022**|
|SVR|Mao et al., **NeurIPS 2023**|
|SAC-RND|Nikulin et al., **ICML 2023**|
|ReBRAC|Tarasov et al., **NeurIPS 2023**|
|MCQ|Tarasov et al., **CORL workshop 2022**（实现）|

另有 goal-conditioned 基线：GC-IQL（ICLR 2022）、WGCSL（ICLR 2022）、DWSL（ICML 2023）、HIQL（NeurIPS 2023）、GCPC（NeurIPS 2023）。

### 在线微调基线（8 个）

CQL（NeurIPS 2020）、Off2On（CoRL 2022）、SPOT（NeurIPS 2022）、RLPD（ICML 2023）、PEX（ICLR 2023）、Cal-QL（NeurIPS 2023）、Uni-O4（ICLR 2024）、ReBRAC（NeurIPS 2023）。

### 实验设置

- **环境**：D4RL 三域 28 个数据集——**AntMaze**（umaze/umaze-diverse/medium-play/medium-diverse/large-play/large-diverse/ultra-play/ultra-diverse，ultra 为更大新迷宫）、**Adroit**（pen/door/hammer/relocate × cloned/expert）、**MuJoCo**（halfcheetah/hopper/walker2d × random/medium-replay/medium/medium-expert）；离线 5 种子；在线微调 300k 步，全部用官方实现复现。
- **离线结果**（表 2）：三域平均 PARS 均列第一——AntMaze 81.6（SVR/ReBRAC 60.6）、Adroit 79.9（ReBRAC 75.0）、MuJoCo 77.3（SAC-N 76.3）；AntMaze Ultra 上 PARS 66.4/51.4 vs 最佳基线 22.4/14.2（约 +280%）。
- **在线微调结果**（表 3）：AntMaze 平均 95.2（RLPD 78.4 居基线之首）、Adroit-cloned 113.2（RLPD 102.5）、MuJoCo 111.6（Off2On 96.4）；**AntMaze ultra-play/ultra-diverse 与 Adroit relocate-cloned 上 PARS 是唯一成功的算法**；PARS 超过 RLPD，直接挑战"离线预训练阻碍微调"的流行结论。
- **goal-conditioned 对比**（表 4）：AntMaze 平均 81.6 vs HIQL 75.3、GCPC 69.4。

### 问题验证/分析实验

- **max-Q 动作范数审计**（表 1）：证明 LN 无法阻止 max-Q 动作落在 $\mathcal{A}_{\mathrm{OOD\text{-}out}}$ 边界；
- **toy 回归实验**（图 5、14）：$y=c_{\mathrm{reward}}\sqrt{x_1^2+x_2^2}$，对比 No LN / LN / LN+大 $c_{\mathrm{reward}}$ / LN+PA 的拟合面；
- **NTK 相似度图**（图 6）：验证大奖励尺度降低 ID-OOD 梯度耦合；
- **休眠神经元实验**（图 7、表 12）：$c_{\mathrm{reward}}$ 增大 → dormant 比例 91%→54-58% → 表达力提升与性能关联；
- **激活函数分析**（图 15、18）：ReLU 最稳健，无激活时 RS-LN 失效；
- **RS-LN 扩展到 IQL**（表 5）及 TD3+BC vs IQL 策略提取对比分析（§6.5、图 12）。

### 消融实验

1. **组件消融**（图 9、16）：None / PA only / LN only / LN+PA × 不同 $c_{\mathrm{reward}}$，两组件互补、大奖励尺度+LN+PA 最优；
2. **PA 在线阶段去留**（图 10）：离线用 PA、在线去掉会不稳定；
3. **不可行域距离**（图 11）：$|L_I|=|U_I|$ 取 $\mathcal{A}_F$ 边界的 1~10000 倍，100~1000 稳健；
4. **$\alpha$ 敏感性**（表 14）：0.001~0.1 良好，过大干扰 TD、过小不起效；
5. **critic ensemble 数量**（图 17）：AntMaze 影响小，MuJoCo/Adroit 有益；
6. **单组超参验证**（表 13）：AntMaze 全域一组超参仍超 SOTA；
7. **计算成本**（图 19）：训练 8.4s/5000 步、显存 582MB，优于 SAC-N/MSG/SAC-RND/ReBRAC。

## 10. 核心 Claims 及证明情况

|核心 Claim|证明形式|
|---|---|
|C1：外推误差的根源是 $\mathcal{A}_{\mathrm{OOD\text{-}out}}$ 上的线性外推，且 LN alone 不能解决|问题验证实验：表 1（max-Q 范数）+ toy 实验（图 5）+ 文献引用（Xu et al. ICLR 2021；Yue et al. NeurIPS 2024）|
|C2：RS-LN 通过降低 ID↔OOD 的 NTK 相似度，使 OOD 的 Q 值相对下降；且与表达力（休眠神经元）相关|无定理；toy NTK 图（图 6）+ dormant 实验（图 7、表 12）+ 玩具回归（图 5、14）|
|C3：PA 使 $\mathcal{A}_{\mathrm{OOD\text{-}out}}$ 的 Q 值进一步下降且不影响可行域内评估|**有收缩性理论**（附录 C.1，理想化算子）+ guard interval 分析（图 11）+ 消融（图 9、16）|
|C4：PARS 在离线与在线微调两阶段均达 SOTA，AntMaze Ultra 独家成功|主实验（表 2、3、4）+ 学习曲线（图 13）+ 计算成本（图 19）|
|C5：离线预训练的阻碍源于 critic 外推误差，修正后可超过 from-scratch 的 RLPD|实验论证：表 3 PARS > RLPD + §6.2 讨论（无理论）|
|C6：RS-LN 可推广到 in-sample 方法（IQL）|表 5 实验 + 附录 C.2 的统一损失形式分析（非严格证明）|
|C7：极简实现 + 无需专门在线过渡技术即可平滑微调|附录 E 代码（几行）+ 图 10（PA 全程保留更稳定）+ 主实验|

**总结**：本文有一小块理论（PA 损失的算子收缩性，附录 C.1），核心创新点 RS-LN 的机制完全是**实证性论证**（toy + NTK 可视化 + 休眠神经元 + 消融 + 主实验）。就您第 5 问关注的"离线知识保留与阻碍"而言，本文的立场在三篇论文中最鲜明：**离线预训练的策略和 critic 值得完整保留，阻碍在线阶段的不是离线知识本身，而是 critic 在 $\mathcal{A}_{\mathrm{OOD\text{-}out}}$ 上的外推行为；只要通过 RS-LN + PA 把外推方向修好，离线知识可以无摩擦地延续到在线阶段**——这与 OPT（引入新 critic）和 CFDG（用生成数据再语境化离线数据）形成三种互补的回答。该论断迁移到 MARL 时，因联合动作外推误差更严重、价值分解与非平稳性未处理，其适用性会比单智能体场景更不确定。