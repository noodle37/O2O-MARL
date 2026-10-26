下面按你上传的 NeurIPS 2025 论文 **《Robust Policy Expansion for Offline-to-Online RL under Diverse Data Corruption》** 来总结并回答 10 个问题。论文提出方法 **RPEX: Robust Policy EXpansion**，本质是把 **Inverse Probability Weighting, IPW** 注入 **Policy Expansion, PEX**，用来提升 Offline-to-Online RL 在 offline/online 数据被污染时的鲁棒性与在线探索效率。

---

## 一、文章主要内容总结

文章研究 **Offline-to-Online Reinforcement Learning, O2O RL**：先在离线数据上预训练策略与价值函数，再用少量在线交互微调。现实问题中，离线数据集和在线交互都可能存在随机噪声或恶意污染，包括状态、动作、奖励、动力学四类 corruption。已有鲁棒离线 RL 如 RIQL 主要证明/改进 **Q target 的 heavy-tailedness**，而 O2O 中更关键的是污染还会让 **policy 本身变得 heavy-tailed**，使在线探索采样到大量低质量轨迹，降低 fine-tuning 效率。

核心观察是：污染 induced heavy-tailed policy；离线评估时常用确定性 argmax 动作，所以重尾影响有限，但在线探索依赖从策略采样，重尾会把概率质量不可控地分给坏动作。因此 RPEX 在 PEX 的动作集合选择概率中加入 IPW 修正项：对低优势且低概率的动作更强惩罚，对高优势动作主要由原 PEX 的 Boltzmann/Q 值项决定。实验在 D4RL MuJoCo medium-replay、AntMaze、随机/对抗/混合污染下验证，RPEX 在多数 RIQL/IQL 基线上取得更好 O2O 性能，并仅带来约 8.1% 训练时间增加。

---

## 1. 文章解决 O2O RL 中的什么问题？为什么现有算法没解决？

### 解决的问题
文章解决的是：**在 offline 预训练数据和 online 交互数据均可能受到状态、动作、奖励、动力学污染时，如何进行鲁棒且高效的 O2O fine-tuning**。

形式化地，离线阶段得到被污染离线缓冲  
$$
\hat{\mathcal D}_{\mathrm{offline}}
=\{(\hat s_t,\hat a_t,\hat r_t,\hat s'_{t+1})\}_{t=1}^N ,
$$
在线交互虽然环境交互本身是干净的，但存入 replay buffer 时可能被污染：
$$
\hat{\mathcal D}\leftarrow \hat{\mathcal D}\cup
\{(\hat s_t,\hat a_t,\hat r_t,\hat s'_{t+1})\}.
$$
攻击可随机注入噪声，如 dynamics 攻击：
$$
\hat s' = s' + \lambda \odot \operatorname{std}(s'),
\quad \lambda\sim \mathcal U[-\epsilon,\epsilon]^{d_s},
$$
也可用 PGD 优化
$$
\hat s'=\arg\min_{\hat s'\in \mathbb B_d(s',\epsilon)}
Q_p(\hat s',\pi_p(\hat s'))
$$
来 adversarially 压低 Q 值。

### 为什么现有算法没有解决好
现有 O2O 方法如 Cal-QL、WSRL、PEX 主要解决 offline-to-online 的“保守性/遗忘/unlearning”问题，即让离线策略不要过保守、在线能探索到好数据。它们默认离线数据和在线数据基本可靠，没有系统处理 corruption。

鲁棒离线 RL 如 RIQL 虽处理 corrupted offline data，但重点是 critic 端 Q target 的重尾：
$$
r(s,a)+\gamma V(s')
$$
在 dynamics 攻击下变得 heavy-tailed，因此用 Huber loss 与 quantile/ensemble critic 稳定 Q 学习。论文 Figure 1(b) 进一步指出：污染不仅放大 Q target 的 heavy-tailedness，也显著放大 **policy** 的 heavy-tailedness；甚至 RIQL 缓解了 Q 重尾，却可能让 policy 更重尾。由于 O2O 在线阶段必须采样探索，policy heavy-tail 会直接损害 exploration efficiency。这一问题在纯 offline robust RL 中不明显，因为 offline evaluation 往往取最大概率动作；但在 O2O online exploration 中会被放大。因此现有算法缺少一个专门在 O2O action selection/exploration 层面对抗 policy heavy-tail 的机制。

---

## 2. 本文核心创新点是什么？

核心创新可概括为三点：

1. **问题新定位**：首次明确把 O2O RL 的鲁棒性瓶颈从“Q target heavy-tailedness”扩展到“policy heavy-tailedness”。  
   论文用 kurtosis
   $$
   \operatorname{Kurt}[X]
   =\mathbb E\left[\left(\frac{X-\mu}{\sigma}\right)^4\right]
   $$
   度量策略重尾，发现攻击下策略 kurtosis 显著升高，且比 Q target 的重尾更普遍。

2. **方法简单有效：IPW + PEX**。PEX 原选择概率为
   $$
   P_{\mathbf w}[i]
   =
   \frac{\exp(Q_\phi(s,a_i)/\alpha)}
   {\sum_j \exp(Q_\phi(s,a_j)/\alpha)} ,
   \quad i\in[1,\dots,K].
   $$
   RPEX 将其改为
   $$
   P_{\mathbf w}[i]
   =
   \frac{
   \exp(Q_\phi(s,a_i)/\alpha)
   +\kappa\, w^{\pi_i}_{\mathrm{ipw}}
   }{
   \sum_j
   \left[
   \exp(Q_\phi(s,a_j)/\alpha)
   +\kappa\, w^{\pi_j}_{\mathrm{ipw}}
   \right]
   },
   $$
   其中
   $$
   w^{\pi_i}_{\mathrm{ipw}}
   =
   \operatorname{CLIP}
   \left(
   \frac{Q_\phi(s,a_i)-V_\psi(s)}
   {\pi_i(a_i|s)},
   \mathrm{MIN},\mathrm{MAX}
   \right).
   $$
   文中取 $\mathrm{MIN}=-10000,\mathrm{MAX}=100$，让负优势且被原策略低概率选择的动作受到更强惩罚。

3. **实用发现**：RPEX 不是孤立 trick，还系统考察了 O2O corruption 下几个关键因素：state normalization 有帮助；高 UTD 主要对 action corruption 有利，其他 corruption 下可能有害；policy extraction 在 state attack 下用 AlignIQL 更好；离线 buffer 是否保留影响较小。

一句话概括创新：**把分类/因果推断中处理 heavy-tail 的 IPW 思想迁移到 O2O 的策略扩展动作选择里，直接修正污染造成的“坏动作被赋予不可控非零探索概率”的问题。**

---

## 3. 本文方法有哪些核心模块？如何在算法流程中发挥作用？完整流程是什么？

### 核心模块

1. **离线鲁棒预训练模块：IQL/RIQL**
   - IQL 学习 value：
     $$
     \mathcal L_V(\psi)
     =
     \mathbb E_{(s,a)\sim\hat{\mathcal D}_{\mathrm{offline}}}
     \left[
     L^\tau_2(Q_{\hat\phi}(s,a)-V_\psi(s))
     \right],
     $$
     $$
     L^\tau_2(u)=|\tau-\mathbf 1(u<0)|u^2 .
     $$
   - critic MSE：
     $$
     \mathcal L_Q(\phi)
     =
     \mathbb E_{(s,a,s')\sim\hat{\mathcal D}_{\mathrm{offline}}}
     \left[
     \left(
     r(s,a)+\gamma V_\psi(s')-Q_\phi(s,a)
     \right)^2
     \right].
     $$
   - RIQL 将 critic loss 换成 Huber loss：
     $$
     \mathcal L_Q
     =
     \mathbb E
     \left[
     l_H^\delta(r+\gamma V(s')-Q(s,a))
     \right].
     $$
   作用：在被污染离线数据上得到可用的 $\pi_\beta,Q_\phi,V_\psi$。

2. **状态归一化模块**  
   用被污染离线数据估计
   $$
   \mu_o=\frac1{2N}\sum_{i=1}^N(s_i+s_i'),
   \quad
   \sigma_o^2=\frac1{2N}\sum_{i=1}^N[(s_i-\mu_o)^2+(s_i'-\mu_o)^2],
   $$
   再做
   $$
   s_i\leftarrow \frac{s_i-\mu_o}{\sigma_o},
   \quad
   s_i'\leftarrow \frac{s_i'-\mu_o}{\sigma_o}.
   $$
   作用：降低 observation/dynamics 攻击造成的尺度漂移。

3. **Policy Expansion 模块**  
   构建策略集合
   $$
   \Pi=[\pi_\beta,\pi_\theta],
   $$
   其中 $\pi_\beta$ 离线预训练并冻结，$\pi_\theta$ 在线新策略。每个状态采样
   $$
   a_1\sim\pi_\beta(\cdot|s),\quad a_2\sim\pi_\theta(\cdot|s),
   $$
   由 $P_{\mathbf w}[i]$ 决定最终动作。作用：既保留离线先验，又允许在线策略扩张。

4. **IPW 重尾修正模块**  
   核心权重
   $$
   w^{\pi_i}_{\mathrm{ipw}}
   =
   \operatorname{CLIP}
   \left(
   \frac{Q_\phi(s,a_i)-V_\psi(s)}
   {\pi_i(a_i|s)},
   \mathrm{MIN},\mathrm{MAX}
   \right).
   $$
   作用机制：
   - 若 $Q-V<0$ 且 $\pi_i(a_i|s)$ 很小，则 $w^{\pi_i}_{\mathrm{ipw}}$ 很负，惩罚该动作；
   - 若 $Q-V>0$，经 clip 后主要由 $\exp(Q/\alpha)$ 主导，不盲目放大；
   - 因此它近似一种“针对坏动作的自适应置信度惩罚”，缓解 corruption 造成的 uncontrolled exploration。

5. **在线学习模块**  
   用混合 buffer
   $$
   \hat{\mathcal D}_{\mathrm{offline}}\cup \hat{\mathcal D}
   $$
   继续训练在线 critic/actor，offline loss 与 online loss 可为 IQL/RIQL loss。动作 corruption 时提高 UTD；其他 corruption 用 UTD=1。state attack 下 policy extraction 用 AlignIQL：
   $$
   \pi^\star(a|s)\propto
   \mu(a|s)
   \exp\{-\eta(Q(s,a)-V(s))^2\},
   $$
   其他情况用 AWR。

### 完整算法流程

Algorithm 1 RPEX 可整理为：

1. 输入离线算法 IQL/RIQL loss $\{L^{Q}_{\mathrm{offline}},L^{\pi_\beta}_{\mathrm{offline}}\}$，在线算法 loss $\{L^{Q}_{\mathrm{online}},L^{\pi_\theta}_{\mathrm{online}}\}$。
2. 初始化 critic/value/actor 参数 $\phi,\psi,\beta,\theta$，载入被污染离线 buffer $\hat{\mathcal D}_{\mathrm{offline}}$。
3. 对环境和 $\hat{\mathcal D}_{\mathrm{offline}}$ 做 state normalization。
4. 离线阶段：
   $$
   \phi\leftarrow\phi-\lambda_Q\nabla_\phi L^{Q}_{\mathrm{offline}}(\phi),
   \quad
   \beta\leftarrow\beta-\lambda_\pi\nabla_\beta L^{\pi_\beta}_{\mathrm{offline}}(\beta),
   $$
   迭代 $2\times10^6$ 步，得到 $\pi_\beta,Q_\phi,V_\psi$。
5. 构造扩展策略
   $$
   \tilde\pi=[\pi_\beta,\pi_\theta],
   $$
   冻结 $\pi_\beta$，转移 $Q_\phi$ 到在线阶段。
6. 在线阶段每个环境步：
   - 采样
     $$
     a_1\sim\pi_\beta(\cdot|s_t),\quad a_2\sim\pi_\theta(\cdot|s_t);
     $$
   - 用 Eq. (6) 的 $P_{\mathbf w}$ 选择最终动作 $a_t$；
   - 环境返回 $s_{t+1}\sim\mathcal P(\cdot|s_t,a_t)$；
   - 攻击/污染 transition，得到
     $$
     (\hat s_t,\hat a_t,\hat r_t,\hat s'_{t+1});
     $$
   - 存入在线 buffer：
     $$
     \hat{\mathcal D}\leftarrow
     \hat{\mathcal D}\cup
     \{(\hat s_t,\hat a_t,\hat r_t,\hat s_{t+1}')\}.
     $$
7. 每个 gradient step 从 offline 与 online buffer 采样，更新 critic 与在线 actor；动作攻击时 UTD 取 4，否则一般取 1。
8. 训练 $1\times10^6$ 在线步；评估时对 PEX/RPEX 选择权重最高动作。

---

## 4. 本文是否进行理论证明？如有，简要概述

有，但属于**目标函数推导与正则项直觉证明**，不是完整收敛定理或样本复杂度保证。

主要结果是 **Proposition 6.1**：给定候选动作 $a_1\sim\pi_\beta,a_2\sim\pi_\theta$，论文推导 PEX/RPEX 的离散选择策略 $P_{\mathbf w}$ 最大化一个带熵正则与 IPW 正则的目标：
$$
\mathbb E_{a_1\sim\pi_\beta,a_2\sim\pi_\theta}
\left[
\sum_i
P_{\mathbf w}(i|s,a_1,a_2)
\left(
Q_\phi(s,a_i)
-\alpha\log P_{\mathbf w}(i|s,a_1,a_2)
\right)
+
\kappa_1
\frac{Q_\phi(s,a_i)-V_\psi(s)}
{\pi_i(a_i|s)}
P_{\mathbf w}(i|s,a_1,a_2)
\right].
$$

第一项是 maximum entropy RL 风格目标；第二项解释为重尾修正/纠偏正则。对 $P_{\mathbf w}$ 求梯度并置零，得到
$$
P_{\mathbf w}(i|s,a_1,a_2)
\propto
\exp\left(
\frac{Q_\phi(s,a_i)}{\alpha}
+
\frac{\kappa_1}{\alpha}
\frac{Q_\phi(s,a_i)-V_\psi(s)}
{\pi_i(a_i|s)}
\right).
$$
令 $\kappa=\kappa_1/\alpha$，即得到论文 Eq. (6)。

论文还给出了 regularization 的解释：当 $Q(s,a)-V(s)<0$ 时，低 $\pi_i(a_i|s)$ 的坏动作会被更强地压低选择概率；当 $Q-V>0$ 时通过 clip 避免过度奖励。注意：论文没有给出该机制在污染 MDP 下的收敛率、后悔界或鲁棒最优性证明；理论主要是“为什么这个选择概率形式是合理的”。

---

## 5. 本文是否讨论了 O2O RL 中 Offline 阶段什么知识值得保留、Online 阶段如何使用、是否阻碍 Online？

论文没有把它作为独立章节系统讨论“知识保留/迁移/负迁移”，但有明确隐含设计：

### Offline 知识值得保留的部分
RPEX 保留并使用三类离线知识：

1. **冻结离线策略 $\pi_\beta$**  
   放入策略集合 $\Pi=[\pi_\beta,\pi_\theta]$，作为在线探索的先验动作来源。

2. **离线 critic/value $Q_\phi,V_\psi$**  
   既用于 PEX 动作选择：
   $$
   \exp(Q_\phi(s,a_i)/\alpha),
   $$
   也用于 IPW：
   $$
   \frac{Q_\phi(s,a_i)-V_\psi(s)}{\pi_i(a_i|s)}.
   $$
   这相当于用离线价值函数判断候选动作是 above-average 还是 below-average。

3. **离线数据/归一化统计量**  
   在线训练可混合离线 buffer；归一化统计量 $\mu_o,\sigma_o$ 来自被污染离线数据，用于稳定状态尺度。

### Online 如何使用
在线阶段不直接继续大改 $\pi_\beta$，而是冻结它作为 expansion policy 的一部分；在线学习主要更新新策略 $\pi_\theta$ 与 critic/value。选择动作时用 $Q,V$ 与 $\pi_i(a_i|s)$ 计算权重；训练时用 offline+online 数据更新。这体现“保留行为先验，但让在线策略扩张”的范式。

### Offline 知识是否会阻碍 Online？
论文明确指出会，尤其当 offline 知识被污染时：

- corruption 使 AWR/IQL 学出的 policy heavy-tailed；
- heavy-tailed $\pi_\beta$ 在在线探索中会不可控地探索坏动作；
- RIQL 减轻 Q 重尾，但 Figure 1(b) 显示它可能反而让 policy 更重尾；
- 观察攻击和动力学攻击下，离线价值函数不准会扭曲 Bellman backup，使在线提升尤其困难；
- Table 6 也讨论：即便在非污染在线设置 $(\epsilon_2,c_2)=(0,0)$，若 value function 不准，某些潜在好动作可能被误判为负优势而被 IPW 惩罚，不过总体仍优于 RIQL-PEX。

所以本文实际是：**保留离线策略/价值作为先验，但承认被污染或不准的离线知识会通过 heavy-tailed policy 与错误 value signal 阻碍 online exploration；RPEX 正是为缓解这种阻碍。**

不足是：它没有进一步讨论更细粒度知识选择，例如何时应当丢弃 $\pi_\beta$、如何检测离线知识已失效、如何做 policy/value 的 confidence-gated retention。

---

## 6. 本文算法是否适合迁移到 MARL？理由与挑战

**部分适合，但不能直接照搬。** RPEX 的思想——“污染导致 policy heavy-tail，进而在 online exploration 中放大负迁移；用 IPW 在动作选择层修正”——对 MARL 很有价值，因为 MARL 本身探索成本高、样本效率低、对队友策略变化敏感，且更容易受到数据污染和对抗攻击影响。但迁移会遇到显著挑战。

### 适合迁移的理由

1. **O2O 在 MARL 更现实**：多智能体系统常先从日志/演示/历史交互离线学习，再上线微调；真实机器人集群、自动驾驶、推荐与游戏 AI 都可能有 noisy/corrupted logs。
2. **heavy-tail exploration 问题更严重**：MARL 联合动作空间大，若每个 agent 的策略都因污染产生重尾，联合探索会组合爆炸式采样坏联合动作。
3. **IPW 的思想可扩展**：可把单体
   $$
   \frac{Q(s,a)-V(s)}{\pi(a|s)}
   $$
   扩展为中心化 critic 下的 joint-action advantage：
   $$
   A(s,\mathbf a)=Q^{\mathrm{tot}}(s,\mathbf a)-V(s),
   $$
   再用联合策略概率 $\pi(\mathbf a|s)$ 或分解概率做逆概率权重。
4. **policy expansion 天然契合 MARL**：可冻结若干离线 teammate policies，在线新增/微调 policies，用 centralized Q 选择组合，类似“先验队友策略集合 + 在线策略扩张”。

### 迁移挑战

1. **联合动作概率分母问题**  
   单体 RPEX 需要 $\pi_i(a_i|s)$。MARL 中若用去中心化 actor，通常只有
   $$
   \pi_i(a_i|s_i)
   $$
   或近似
   $$
   \prod_i \pi_i(a_i|s_i),
   $$
   不一定能拿到真实 joint policy $\pi(\mathbf a|s)$。IPW 对分母很敏感，概率估计稍差就会导致权重爆炸或错误惩罚。

2. **信用分配与值分解耦合**  
   常用 VDN/QMIX 等有
   $$
   Q^{\mathrm{tot}}=f_\psi(Q_1,\dots,Q_n; s).
   $$
   corruption 可能同时污染个体 Q、mixer、奖励 shaping 与 joint transition；把单体 advantage $Q-V$ 直接换成 $Q^{\mathrm{tot}}-V$ 可能把 mixer 误差误归因到某个 agent 的 IPW 权重上。

3. **非平稳性更强**  
   单体 O2O 的非平稳主要来自环境探索与 policy 更新；MARL 还来自其他 agent 策略更新。离线预训练学到的 teammate behavior/value 在 online 时更快过时，冻结 $\pi_\beta$ 可能从“先验”变成“负资产”。

4. **通信/观测受限**  
   若 partial observability，$Q(s,a)$ 要变成 $Q(\tau,\mathbf a)$，IPW 需要历史轨迹 $\tau$ 与个体/联合动作概率。POMAR L 下历史重尾、信念不准会让权重更不稳定。

5. **攻击面扩大**  
   单体 corruption 影响一个 transition；MARL 中污染一个 agent 的 observation/action/reward 会通过 joint action 与 shared reward 影响所有 agent，credit assignment 更难。

6. **计算与优化稳定性**  
   需要为每个 agent 采样候选动作，再组合联合动作候选；若有 $K$ 个策略与 $n$ 个 agent，联合候选规模可能从 $K$ 升到 $K^n$ 或需要因式分解近似。高 UTD 在 corruption 下本就有方差问题，MARL 更严重。

7. **理论假设更难成立**  
   论文推导依赖固定 MDP、可从 $\pi_i$ 采样并计算概率、$Q,V$ 可用。MARL 的非平稳性使 MDP 假设弱化，CTDE 训练与去中心化执行之间存在 mismatch，Proposition 6.1 的熵正则解释未必直接成立。

结论：**思想可迁移为 “Robust Multi-Agent Policy Expansion with Centralized IPW”，但必须解决 joint-action probability、credit assignment、value factorization、nonstationarity 与 partial observability 五个核心问题。**

---

## 7. 结合问题 5：MARL 中 offline 知识保留/online 使用/阻碍是否更不同、更具挑战性？

是，且挑战显著增加。

### 单体 O2O 中
Offline 保留的是：$\pi_\beta,Q,V,\mu_o/\sigma_o$，可能还有 offline buffer。Online 使用方式是：冻结 $\pi_\beta$ 进入 policy set，用 $Q,V$ 选择动作，用 IPW 惩罚坏动作，用 offline+online 数据训练 $\pi_\theta$。阻碍来自：corrupted value 使 advantage 不准；heavy-tailed $\pi_\beta$ 让在线探索采样坏轨迹；RIQL 只稳 Q 不稳 policy。

### MARL 中应保留的 offline 知识更多
除单体策略价值外，还可能值得保留：
- teammate policies $\pi_{\beta}^{1},\dots,\pi_{\beta}^{n}$；
- centralized critic / value decomposition 权重；
- 通信协议或 message encoder；
- 任务结构/角色分配/子团队策略；
- 队友行为模型或世界模型先验；
- 安全约束与事故规避先验。

Online 使用方式可设计为：冻结一组离线 teammate policies 作为 expansion set，在线新增 learnable policies；用 centralized $Q^{\mathrm{tot}}$ 选择 joint action 或每个 agent 的动作；用 IPW 在 joint action 层或 per-agent conditional action 层惩罚低优势高风险的联合动作；用 offline+online 数据继续训练 CTDE critic 与 actors。

### 为什么 MARL 更不同、更难
1. **阻碍来源更多**  
   单体主要是自身 policy/value corruption；MARL 还包括队友策略过时、mixer 失配、通信通道污染、奖励分配不均、个体动作不可区分。

2. **非平稳性使 offline 知识更快失效**  
   单体 offline policy stale 通常仍包含 MDP 中有效模式；MARL 中其他 agent 策略变化会改变最佳响应，旧 teammate policy 可能主动诱导错误协同。

3. **联合动作导致 IPW 作用对象改变**  
   单体惩罚的是坏动作 $a$；MARL 需要判断坏的是联合动作 $\mathbf a$、某个 agent 的 action，还是 communication/message。错误归因会把好 agent 的合理动作压掉，或放过真正有害的 agent。

4. **credit assignment 受 corruption 放大**  
   shared reward 下，一个被污染 agent 的错误动作会降低团队回报，所有 agent 的 advantage 都被污染；IPW 若基于错误 $Q^{\mathrm{tot}}-V$ 会系统性偏差。

5. **探索效率问题更尖锐**  
   联合动作空间大，heavy-tail 不再只是“偶尔采坏动作”，而可能是“频繁采到破坏性协同”。因此 RPEX 的 motivation 在 MARL 更强，但简单加 IPW 也更危险。

因此，MARL 版本不能只做逐 agent RPEX；更合理的是 **centralized training 下做 joint-action IPW + per-agent responsibility attribution + stale-prior gating**，即在保留 offline 协同先验的同时，动态判断哪些离线知识仍可信。

---

## 8. 本文仍有哪些不足？结合迁移到 MARL 会如何放大？

论文已承认的局限：IPW 需要精确动作概率 $\pi_i(a_i|s)$，不利于 diffusion policy 等概率不透明或难以精确求 likelihood 的策略。此外：

1. **理论保证有限**  
   Proposition 6.1 是目标函数推导和正则解释，不是鲁棒收敛/最优性证明；对 corruption rate、coverage、clip 范围如何影响性能没有定量界。

2. **对 observation/dynamics corruption 仍难**  
   论文解释 observation 攻击会破坏探索动作选择并扭曲 Bellman backup；dynamics 攻击导致 Bellman backup error，RIQL 只能缓解不能根除。RPEX 在 Hopper/Walker observation/dynamics 上提升有限甚至仍低。

3. **依赖离线 Q,V 质量**  
   IPW 的本质仍用 $Q-V$ 判断动作好坏。若 value function 被严重污染，RPEX 会惩罚一些本不应惩罚的动作；论文用 clip、$\kappa$、normalization 缓解，但没有自动失效检测机制。

4. **超参鲁棒性靠经验**  
   $\kappa,\alpha^{-1}$、clip range、UTD、policy extraction 都影响结果。论文显示多数情况稳定，但在极端 clip 如 $(0,100)$ 会显著变差；说明机制对“只惩罚不奖励”的设计较敏感。

5. **未处理概率不可求策略**  
   diffusion policy、large stochastic policy、implicit policy 难以获得精确 $\pi(a|s)$，限制了与离线 RL 中强生成式策略结合。

6. **负迁移/知识丢弃机制不足**  
   没有明确回答：什么时候 $\pi_\beta$ 应被降权或移除？什么时候 $Q,V$ 不可信应进入保守模式？这在非平稳或高 corruption 下很关键。

迁移到 MARL 后，这些不足会放大：
- 需要 joint action likelihood，diffusion/multi-agent policy 更难；
- centralized critic 与 value decomposition 误差会被 IPW 放大；
- shared reward 下 value signal 更 noisy，$Q^{\mathrm{tot}}-V$ 的 advantage 更不可靠；
- stale teammate policy 比 stale single policy 更危险；
- communication、partial observability、nonstationarity 使“精确概率 + 稳定 advantage”的前提更难满足；
- 高 UTD 在 MARL 更容易引发训练不稳定；
- evaluation 更难，因为不仅要平均 return，还要衡量协同失败、个体责任、安全约束违反。

---

## 9. 本文对照算法有哪些？发表时间/venue？做了什么实验、环境是什么？消融和问题验证如何做？

### 主要对照算法
文中主要比较的是 IQL/RIQL 及其 PEX/RPEX 变体：

| 算法 | 角色 | 文中引用信息 |
|---|---|---|
| IQL | 离线/在线基线；offline policy 直接继续在线训练 | Kostrikov et al., 2021 ICML 与 2021/2022 ICLR 版本；论文引用 [25], [26] |
| RIQL | 鲁棒 IQL baseline；用 Huber loss、observation normalization、quantile/ensemble Q | Yang et al., 2023 arXiv:2310.12955 [50]；论文按 RIQL 实现 |
| IQL-PEX | IQL + Policy Expansion | PEX: Zhang et al., 2023 arXiv:2302.00935 [53] |
| RIQL-PEX | RIQL + PEX | 同上，PEX 机制 |
| IQL-RPEX / RIQL-RPEX，后者即 RPEX | 本文方法 | NeurIPS 2025 投稿论文 |

文中 related work 还提到 Cal-QL、WSRL、EDAC、AlignIQL、RDT、RORL 等，但主表核心对照是 IQL、RIQL、IQL-PEX、RIQL-PEX、RPEX。对于发表 venue，为避免误报：PEX 与 RIQL 在参考文献中主要标为 arXiv preprint；IQL 有明确 ICML 2021 与 ICLR 2022 两个相关版本；RPEX 是本文 NeurIPS 2025。

### 主实验环境
主要在 **D4RL MuJoCo medium-replay** 任务：
- Halfcheetah-MR
- Walker2d-MR
- Hopper-MR

以及附录的更难稀疏奖励任务：
- AntMaze-large-play / large-diverse，重点测 dynamics 与 observation corruption。

### 攻击与 corruption 设置
离线预训练污染率 $c_1=0.3$，在线每轨迹以概率 $c_2=0.5$ 被污染；主实验尺度 $\epsilon=1$。攻击元素包括：
- observation/state corruption：
  $$
  \hat s=s+\lambda\odot\operatorname{std}(s)
  $$
- action corruption：
  $$
  \hat a=a+\lambda\odot\operatorname{std}(a)
  $$
- reward corruption：
  $$
  \hat r\sim\mathcal U[-30\epsilon,30\epsilon]
  $$
- dynamics/next-state corruption：
  $$
  \hat s'=s'+\lambda\odot\operatorname{std}(s')
  $$
- mixed attack：offline dynamics attack，online random dynamics+reward attack；
- adversarial dynamics attack：用 PGD 解
  $$
  \min_{\hat s'} Q_p(\hat s',\pi_p(\hat s'))
  $$
  within $\ell_\infty$-style ball。

### 实验结果形式
- Table 1：Medium-Replay 随机 corruption，offline score $\to$ O2O score，5 seeds，报告最后三次评估均值与标准差。
- Figure 8：mixed 与 adversarial attack 下 Hopper-MR 对比。
- Table 5：AntMaze large/diverse 的 dynamics/observation corruption。
- Figure 9：完整训练曲线。
- Table 6：不同在线 corruption level，包括干净在线 $(\epsilon_2,c_2)=(0,0)$。

主结果：RPEX 在 RIQL-based 12 个设置中的 11 个最高；平均 O2O score：IQL 50.3，IQL-PEX 53.5，IQL-RPEX 59.1；RIQL 54.9，RIQL-PEX 66.6，RPEX 75.16。相对 vanilla PEX 约有 13.6% 提升的表述来自摘要/实验综合。

### 消融实验与问题验证实验
1. **问题验证实验：policy heavy-tail**
   - Figure 1(b)：比较 CLEAN IQL、RIQL、IQL 在 observation/action/dynamics/reward 攻击下的 policy kurtosis。
   - Figure 2：grid-world/toy maze 展示 corrupted value function 下 IQL 选择更长、更差轨迹，IQL+IPW 改善。
   - Figure 3：Halfcheetah-MR reward attack 下动作分布与训练曲线，验证 IPW 降低重尾并提升回报。

2. **UTD 消融**
   - Figure 4：RPEX/RIQL/RIQL-PEX 在 UTD=1 与 4 下比较；结论是高 UTD 主要帮助 action corruption，其他 corruption 不一定。
   - Table 2：只对 critic 用高 UTD、actor 保持 UTD=1，发现可缓解问题但方差增大。

3. **policy type / normalization / $\kappa$ 消融**
   - Figure 5：Hopper-MR；stochastic policy 优于 deterministic；state normalization 显著提升；$\kappa\in\{0.1,0.3,1.0\}$ 不敏感但可调。

4. **policy extraction 消融**
   - Figure 6：AlignIQL 对 observation attack 明显更好，AWR 在其他 corruption 更好。

5. **offline buffer retention 消融**
   - Figure 7：保留/不保留 offline buffer 对 RPEX 影响很小，说明不过度依赖 offline data。

6. **clip range 消融**
   - Table 3：Walker2d-MR dynamics/action corruption；常用 $(-10000,100)$ 稳定，极端 $(0,100)$ 大幅变差，$(-10000,10000)$ 也有差异，说明需要保留对负 advantage 的强惩罚但避免正项爆炸。

7. **训练成本**
   - Table 4：IQL 17.4s/epoch，IQL-PEX 30.9，IQL-RPEX 31.7；RIQL 22.7，RIQL-PEX 33.1，RIQL-RPEX 37.6；RPEX 比 PEX 约多 8.1% 时间但提升约 13.6%。

---

## 10. 本文有哪些核心 claim？是否有理论或实验证明？

| 核心 claim | 证据类型 | 对应结果 | 强度评价 |
|---|---|---|---|
| C1：O2O 中 offline/online 多类型 corruption 会严重损害性能 | 实验 | Table 1, Fig. 8, Table 5, Fig. 9 | 强 |
| C2：数据 corruption 不仅使 Q target heavy-tail，也使 policy heavy-tail，并在 online exploration 中造成低效探索 | 问题验证/分析实验 | Fig. 1(b), Fig. 2, Fig. 3, Appendix D | 强，但主要是经验与可视化 |
| C3：IPW 能缓解 policy heavy-tail，从而提高 O2O 性能 | 方法 + 主实验 + 理论直觉 | Eq. (6)(7), Prop. 6.1, Table 1, Fig. 3, Fig. 8 | 中强；理论不是收敛保证 |
| C4：RPEX 简单修改即可显著优于 IQL/RIQL/PEX | 主实验 | Table 1 平均 59.1/75.16 vs baselines；摘要称相对 PEX 约 +13.6% | 强，限于 D4RL/MuJoCo/AntMaze 设定 |
| C5：state normalization 对 O2O corruption 有帮助；高 UTD 主要利于 action corruption，其他情况可能有害 | 消融 | Fig. 4, Fig. 5, Table 2 | 中强，环境与任务有限 |
| C6：policy extraction 重要，state attack 下 AlignIQL 更好 | 消融 | Fig. 5, Fig. 6, Eq. (8) | 中等，场景特定 |
| C7：RPEX 不依赖持续保留 offline buffer 也稳定 | 消融 | Fig. 7 | 中等 |
| C8：性能提升带来的计算开销较小 | 训练成本实验 | Table 4；约 +8.1% epoch time vs PEX | 中等 |
| C9：局限是 IPW 需要精确动作概率，难用于 diffusion/implicit policy | 作者明确 limitation | Conclusion and Limitations | 强，属于 acknowledged limitation |

特别说明：论文最有力的 claim 是 **C2+C3+C4 的闭环**：先证明/展示 corruption-induced policy heavy-tail，再提出 IPW 修正动作选择，再在 random/adversarial/mixed corruption 与 D4RL/AntMaze 上取得主实验提升。相对薄弱的是 **理论只解释形式来源，没有给出鲁棒学习保证**；并且对 observation/dynamics corruption 的提升仍有限，迁移到 MARL 时 joint-action probability、信用分配和非平稳性会成为主要瓶颈。