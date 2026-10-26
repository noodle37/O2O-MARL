## 总体概括

FINO 面向 **offline-to-online RL, O2O RL**：先在离线数据集上预训练策略，再用很短的在线交互预算 fine-tune。作者指出，现有生成式策略方法，尤其 flow matching policy，在 offline 阶段为了稳定通常把条件概率路径收缩到数据点上；例如 FQL 中取 $\sigma_{\min}=0$，训练目标退化为点对点的 behavior cloning：

$$  
\mathcal{L}_{\pi}(\theta)=\mathbb{E}_{x_0\sim\mathcal{N}(0,I),\,s,a=x_1\sim\mathcal{D},\,t\sim\mathrm{Unif}[0,1]}\left[\lVert v_{\theta}(t,s,x_t)-(x_1-x_0)\rVert_2^2\right],\quad x_t=(1-t)x_0+tx_1.  
$$

这会让策略支撑集几乎不超出离线数据。到了 online 阶段，这反而成为探索障碍。FINO 的核心改动是在 flow matching 的输入路径上注入时变高斯噪声：

$$  
\epsilon_t\sim\mathcal{N}(0,\alpha_t^2 I),\qquad \alpha_t^2=(\eta^2-2\eta)t^2+2\eta t,  
$$

并训练

$$  
\mathcal{L}_{\mathrm{FINO}}(\theta)=\mathbb{E}\left[\left\lVert v_{\theta}(t,s,x_t+\epsilon_t)-\big(x_1-(1-\eta)x_0\big)\right\rVert_2^2\right],  
$$

使条件路径均值仍为 $t x_i$，方差变为

$$  
p_t^{\mathrm{FINO}}(x\mid x_i)=\mathcal{N}\left(x\mid t x_i,\ \big(1-(1-\eta)t\big)^2 I\right),  
$$

从而在不显著偏移离线行为均值的前提下扩大动作覆盖。online 阶段再用多候选动作采样和熵引导温度调节平衡探索/利用。整体 backbone 是 FQL，包含 flow policy $v_{\theta}$、one-step policy $\pi_{\omega}$、Q 函数 $Q_{\phi}$。

---

## 1. 文章解决 O2O RL 中的什么问题？为什么现有算法还没解决好？

**要解决的核心问题**：O2O RL 中，离线预训练得到的生成式/流策略过度贴合离线数据，导致 online fine-tuning 时探索不足、样本效率低；同时在线预算有限，必须动态平衡 exploration 与 exploitation。

文中 Figure 1 的动机实验很关键：在 `antmaze-giant-navigate` 中，离线数据主要包含上方成功路径，FQL online 后智能体大多仍沿起点附近和上方路径活动，性能受限；FINO 则能覆盖更多可行通道。作者结论是：offline 时“贴合数据分布”是稳定性的来源，但 O2O 中若从 offline 一开始就只学数据点本身，会把“稳定”变成 online 的“束缚”。

**为什么现有算法没解决好**，可分三类：

1. **保守型 offline/O2O 方法**  
    Cal-QL、ReBRAC 等继承 offline RL 的保守性，防止 OOD 动作，但也限制 online 探索。ReBRAC 这类 Gaussian policy 在复杂高维/多模态任务上表达能力也有限。
2. **强在线方法但不能很好利用离线先验**  
    RLPD 很强，但本质偏 online RL，只用离线数据初始化，没有在 offline 阶段就为 online 探索设计策略结构。
3. **已有生成式策略方法只是“offline 方法直接继续 online”**  
    FQL/IFQL/IDQL 类方法有多候选采样，但 offline 的 flow/扩散策略仍偏向点匹配数据分布。FINO 在 Appendix D 还证明：若把零均值噪声直接加到 velocity target 上，期望梯度与原 flow matching 相同，不能真正改变训练；因此“噪声加到哪个位置”不是细节，而是方法有效性的关键。直接往最终动作加 Gaussian noise 也不能像 FINO 那样让 one-step policy 在更大动作域内做 Q 最大化，容易随机扩散甚至退化。

---

## 2. 本文核心创新点是什么？

主要有两个相互耦合的创新。

### 创新一：面向 flow matching 的噪声注入预训练，而不是普通动作噪声

FINO 把噪声注入 flow matching 的条件路径：

$$  
\tilde{x}_t=x_t+\epsilon_t,\qquad \epsilon_t\sim\mathcal{N}(0,\alpha_t^2 I),  
$$

并采用时变方差

$$  
\alpha_t^2=(\eta^2-2\eta)t^2+2\eta t,\qquad \eta\in[0,1].  
$$

它满足边界：$\alpha_0^2=0$，$\alpha_1^2=\eta^2>0$。理论含义是：终端时刻 FINO 的条件分布方差为 $\eta^2$，大于标准 flow matching 的近似零方差；同时均值保持 $t x_i$。因此它“扩张动作空间但不胡乱漂移”，更像在数据支撑周围做可控扩张。

实现上作者说也用平滑 shifted exponential schedule 近似同一边界，即 $\alpha_t=\eta e^{5(t-1)}$，并经验发现目标用 $x_1-x_0$ 与 $x_1-(1-\eta)x_0$ 性能无差；但理论命题按 $x_1-(1-\eta)x_0$ 陈述。

### 创新二：online 阶段熵引导的多候选动作采样

online 时对同一状态采 $N_{\mathrm{sample}}$ 个候选动作：

$$  
\{a_1,\dots,a_{N_{\mathrm{sample}}}\}\sim\pi_{\omega}(s),  
$$

不按贪婪直接选 $\arg\max Q$，而是按

$$  
p_{\mathrm{sampling}}(i)=\frac{\exp(\xi Q_{\phi}(s,a_i))}{\sum_{j=1}^{N_{\mathrm{sample}}}\exp(\xi Q_{\phi}(s,a_j))}  
$$

采样执行。小 $\xi$ 更平，偏探索；大 $\xi$ 更尖，偏利用。温度随策略熵更新：

$$  
\xi_{\mathrm{new}}=\xi-\alpha_{\xi}\big(\mathcal{H}-\bar{\mathcal{H}}\big),  
$$

其中 $\mathcal{H}$ 是当前估计策略熵，$\bar{\mathcal{H}}$ 是目标熵。推理时则确定性取 Q 最大动作，保证评估稳定。

这两个创新分别对应 O2O 的两个阶段：offline 负责“预先获得可用多样性”，online 负责“把这种多样性转成样本高效的探索，并随训练进程调探索强度”。

---

## 3. 方法有哪些核心模块？如何在算法流程中发挥作用？完整流程是什么？

核心模块有五个：

1. **噪声注入 flow policy $v_{\theta}$**  
    offline 用 FINO 目标训练，使 flow policy 的条件路径方差大于标准 flow matching，获得更宽但围绕数据的行为先验。
    
2. **one-step policy $\pi_{\omega}$**  
    由于逐步解 ODE 在线交互太贵，FQL 用一步策略蒸馏 flow policy，并同时最大化 Q：
    
    $$  
    \mathcal{L}_{\pi}(\omega)=\mathbb{E}_{s\sim\mathcal{D},\,z\sim\mathcal{N}(0,I)}\left[-Q_{\phi}(s,a_{\omega}(s,z))+\alpha\lVert a_{\omega}(s,z)-a_{\theta}(s,z)\rVert_2^2\right].  
    $$
    
    FINO 的扩张 flow policy 使该 Q 最大化发生在一个更宽、更合理的动作区域上，而不是只在原数据支撑上贪心。
    
3. **价值函数 $Q_{\phi}$ 与 TD 更新**  
    负责给候选动作排序，并把在线新数据纳入策略改进。
    
4. **熵估计模块**  
    one-step policy 是蒸馏加 Q 最大化，分布不可解析。作者用从同一状态采样的多个动作拟合 GMM：
    
    $$  
    p(a)=\sum_{k=1}^{K}\pi_k\mathcal{N}(a\mid\mu_k,\Sigma_k),  
    $$
    
    再用 Huber 等人的 GMM 熵近似：
    
    $$  
    \mathcal{H}\approx\sum_{k=1}^{K}\pi_k\left(-\log\pi_k+\frac{1}{2}\log\big((2\pi e)^d\lvert\Sigma_k\rvert\big)\right).  
    $$
    
    默认 $K=3$，每状态采 200 个动作估计。
    
5. **熵引导候选采样器**  
    online 交互时采样候选并按 softmax(Q) 选择；每 $N_{\xi}$ 步估计熵并更新 $\xi$；评估/推理时切换为 $\arg\max_a Q(s,a)$。
    

**完整流程**可写为：

- **输入**：flow policy $v_{\theta}$、one-step policy $\pi_{\omega}$、Q 函数 $Q_{\phi}$、候选数 $N_{\mathrm{sample}}$、熵更新间隔 $N_{\xi}$。
    
- **Offline pre-training 循环**：
    
    1. 从 $\mathcal{D}$ 采 $(s,a,r,s')$；
    2. 用噪声注入 flow matching 更新 $v_{\theta}$：目标输入为 $x_t+\epsilon_t$，回归方向为 $x_1-(1-\eta)x_0$；
    3. 用蒸馏 + Q 最大化更新 one-step policy $\pi_{\omega}$；
    4. 用 TD loss 更新 $Q_{\phi}$。
- **Online fine-tuning 循环**：
    
    1. 对当前状态 $s$，从 $\pi_{\omega}(s)$ 采 $N_{\mathrm{sample}}$ 个候选动作；
    2. 用 $p_{\mathrm{sampling}}(i)\propto\exp(\xi Q_{\phi}(s,a_i))$ 选一个动作执行；
    3. 将新转移放入 replay；
    4. 继续用同一套目标更新 $\omega,\theta,\phi$；
    5. 每 $N_{\xi}$ 步用 GMM 估计策略熵 $\mathcal{H}$，按 $\xi\leftarrow\xi-\alpha_{\xi}(\mathcal{H}-\bar{\mathcal{H}})$ 更新温度；
    6. 推理/评估时确定性输出
    
    $$  
    a^*=\arg\max_{a\in\{a_i\}}Q_{\phi}(s,a).  
    $$
    

关键超参：动作域均为 $[-1,1]$ 时固定 $\eta=0.1$；$N_{\mathrm{sample}}=\tfrac{1}{2}\dim(\mathcal{A})$；目标熵 $\bar{\mathcal{H}}=-\dim(\mathcal{A})$；$N_{\xi}=50000$。

---

## 4. 是否有理论证明？如有，简要概述

有，主要是三条。

### Proposition 1：条件概率路径的均值不变、方差增大

给定数据点 $x_i$，注入 $\epsilon_t\sim\mathcal{N}(0,\alpha_t^2 I)$ 后，扰动流为

$$  
\tilde{\phi}_t(x)=t x_i+(1-t)x+\epsilon_t.  
$$

因为独立高斯之和仍为高斯，可得

$$  
p_t^{\mathrm{FINO}}(x\mid x_i)=\mathcal{N}\left(x\mid t x_i,\ \big(1-(1-\eta)t\big)^2 I\right).  
$$

与标准 flow matching 的 $\mathcal{N}\big(x\mid t x_i,(1-(1-\sigma_{\min})t)^2 I\big)$ 相比，均值同为 $t x_i$，而 FINO 方差满足 $\big(1-(1-\eta)t\big)^2\ge\big(1-(1-\sigma_{\min})t\big)^2$，在相应边界意义下更大。

### Theorem 1：诱导合法连续归一化流

条件变换可写成

$$  
x_t=\phi_t(x_0)=t x_i+\big(1-(1-\eta)t\big)x_0.  
$$

其导数为

$$  
\frac{d}{dt}\phi_t(x_0)=x_i-(1-\eta)x_0,  
$$

即条件向量场为

$$  
v_t(x\mid x_i)=x_i-(1-\eta)x_0.  
$$

作者进一步由 flow matching 中 marginal path 与 marginal vector field 的定义说明，$v_t(x)$ 与 $p_t(x)$ 满足连续性方程，因此 FINO 仍是一个有效的 continuous normalizing flow，不是任意加噪破坏生成模型。

### Theorem 2：边际概率路径方差不小于标准 flow matching

在有限数据集 $\mathcal{D}=\{x_1,\dots,x_N\}$、i.i.d. 采样假设下，对任意 $t\in[0,1]$，有

$$  
\operatorname{Var}(X_t^{\mathrm{FINO}})\ge\operatorname{Var}(X_t^{\mathrm{FM}}),\quad X_t^{\mathrm{FINO}}\sim p_t^{\mathrm{FINO}}(x),\ X_t^{\mathrm{FM}}\sim p_t^{\mathrm{FM}}(x).  
$$

证明思路是把 marginal variance 分解为条件方差项与条件均值间离散项；Proposition 1 已给出均值相等且条件方差更大，所以差值非负。

另外，Appendix D 有一个重要否定性理论结果：若只对 velocity target 加零均值噪声 $\epsilon$，目标中交叉项期望为零、噪声范数项与 $\theta$ 无关，因此期望梯度与原始 flow matching 相同。这从反面说明 FINO 选择“注入到输入路径/概率路径”而不是“污染回归目标”是必要的。

---

## 5. 是否讨论了 O2O 中 offline 阶段什么知识值得保留、online 如何使用？是否讨论 offline 知识会阻碍 online？

**有讨论，但主要是围绕“行为分布的支撑/多样性”这一层知识，不是系统的知识保留框架。**

FINO 隐含地回答了：offline 阶段值得保留的不是“严格等于数据点”的策略，而是：

- 离线行为分布的**均值结构/可行行为模式**，即 $p_t^{\mathrm{FINO}}$ 保持均值 $t x_i$；
- 在数据支撑周围的**可控扩张区域**，即终端方差 $\eta^2$ 带来的更大动作覆盖；
- 经 Q 蒸馏后仍与扩张 flow policy 一致的 one-step policy，使 online 能在更宽区域内做价值改进。

online 的使用方式也很明确：不是继续保守模仿，而是把扩张 flow policy 蒸馏成可交互的 one-step policy，再用多候选采样和熵调温把多样性转成探索；当熵低于/高于目标时，通过

$$  
\xi_{\mathrm{new}}=\xi-\alpha_{\xi}(\mathcal{H}-\bar{\mathcal{H}})  
$$

自动偏探索或偏利用。

关于阻碍，论文明确承认：offline 的过度点匹配会限制 online 探索，这就是 motivation。换言之，offline 的“稳定保守知识”在 online 阶段可能变成阻碍。FINO 的处理不是抛弃 offline 先验，而是**保留均值、放松方差**。但它没有系统讨论哪些更抽象的知识应保留，例如 Q 的不确定性、安全约束、技能原语、对手模型、状态可达性边界等；也没有真正解决 distribution shift，Appendix A 明确把 distribution shift 留作未来工作。

---

## 6. 方法是否适合迁移到 MARL？理由与挑战？

**部分适合，但不能直接照搬；更准确地说是“思想可迁移，机制需重构”。**

可迁移的部分：

- 单智能体层面，每个 agent 的 policy 可替换为 flow matching policy，并用 FINO 式噪声注入扩大个体动作多样性；
- “offline 保留均值、受控扩大方差”的思想可用于多智能体 offline 数据不足时的探索；
- 若使用 CTDE，集中式 critic 可对联合动作或个体动作候选打分，再经通信/策略解码执行；
- 熵引导探索的思想可改为“团队熵/联合熵/个体熵加权”的调度。

核心挑战：

1. **联合动作维度灾难**  
    单智能体用 $N_{\mathrm{sample}}=\tfrac{1}{2}\dim(\mathcal{A})$ 已偏高；MARL 中联合动作维度约为 $\sum_{i=1}^{n}\dim(\mathcal{A}_i)$，候选数和 Q 评估成本会爆炸。必须做因子化采样、逐 agent 候选、低维潜动作或通信受限采样。
    
2. **非平稳性破坏熵调节**  
    FINO 的 $\xi$ 更新假设策略熵能反映探索-利用平衡；但 MARL 中其他 agent 策略也在变，当前 agent 熵升高/降低未必来自自身探索，可能来自 teammate/opponent 策略漂移。GMM 熵估计在非平稳、部分可观测下更不可靠。
    
3. **探索的外部性与协调**  
    单智能体探索差主要害自己；MARL 中一个 agent 的随机探索可能破坏团队协调，导致联合 Q 估计偏差。softmax(Q) 选候选需要联合 Q 足够可信；若用值分解如
    
    $$  
    Q_{\mathrm{tot}}(\boldsymbol{\tau},\mathbf{a})\approx\sum_i Q_i(\tau_i,a_i)  
    $$
    
    或单调混合，探索还要满足可分解性和 IGM 原则，否则会选到“个体 Q 高但联合差”的动作。
    
4. **offline 数据覆盖更危险**  
    多智能体联合状态-动作空间巨大，offline 数据几乎不可能覆盖协调策略。FINO 增大边际方差在单智能体是“围绕数据扩张”，在 MARL 可能更容易踩到破坏协同的 OOD 联合动作。
    
5. **信用分配与对手适应**  
    online fine-tuning 中需要判断收益来自自身探索、队友改变还是对手策略变化。FINO 没有信用分配模块；直接迁移会把探索奖励归因弄混。
    

所以：**适合迁移的是“噪声注入生成式策略 + 熵/价值引导候选采样”的思想；不适合直接迁移的是固定 $\eta$、$N_{\mathrm{sample}}=\tfrac{1}{2}\dim(\mathcal{A})$、单 agent GMM 熵和单个温度 $\xi$ 的具体实现。**

---

## 7. 结合问题 5：offline 知识保留/online 使用/阻碍在 MARL 中是否不同？是否因非平稳性和联合动作更具挑战？

是，明显不同，且 MARL 更难。

在单智能体 FINO 中，offline→online 的知识主线较清楚：

保留行为分布均值 $t x_i$，放宽终端方差到 $\eta^2$，online 用扩张后的 flow policy 蒸馏 one-step policy，并用

$$  
p(i)\propto e^{\xi Q(s,a_i)},\qquad \xi\leftarrow\xi-\alpha_{\xi}(\mathcal{H}-\bar{\mathcal{H}})  
$$

决定探索强度。这里的“阻碍”主要是 offline 点匹配太窄。

在 MARL 中，同样的问题变成三层：

- **保留什么 offline 知识**：不仅是单个 agent 的行为均值，还包括联合行为模式、通信协议、角色分工、对手/队友响应的先验。若只扩张每个 agent 的边际动作分布，可能保留了个体合理动作却破坏联合协调。
    
- **online 运用哪些知识**：不能只用个体 Q；需要集中式联合 Q 或值分解 Q 来判断联合动作价值。候选采样也必须在联合动作空间进行，否则 softmax(Q) 没有全局意义。
    
- **offline 知识何时构成阻碍**：在 MARL 中阻碍更强。离线数据往往只覆盖少数协调均衡；online 时队友/对手策略变化，旧协调模式可能迅速失效。若像 FINO 一样提高各 agent 策略方差，非平稳环境会把“温和扩张”放大成“协同崩坏”。联合动作还意味着单个 agent 的熵降低不代表系统探索降低，可能是其响应了他人策略；因此单 agent 目标熵
    
    $$  
    \bar{\mathcal{H}}=-\dim(\mathcal{A})  
    $$
    
    不能直接推广为
    
    $$  
    \bar{\mathcal{H}}_{\mathrm{joint}}=-\dim(\mathcal{A}_{\mathrm{joint}}),  
    $$
    
    后者维度大且语义错误。
    

因此，MARL 版本应把 FINO 的“个体方差扩张”改成“协调感知扩张”：例如在 monotonic value decomposition 约束下扩张个体动作、用集中 critic 过滤灾难性联合候选、用团队熵/互信息/角色熵替代单 agent 熵，并引入对手/队友策略变化的非平稳校正。否则非平稳性和联合动作会让 FINO 的两个核心机制同时失稳。

---

## 8. 本文是否仍有不足？尤其迁移到 MARL 会有哪些不足？

有。论文自己在 Appendix A 提到两点：熵用 GMM 近似，带来计算开销且不精确；distribution shift 仍未解决。结合你的问题，还可补充：

1. **offline 知识保留讨论不够系统**  
    它只明确了“保留均值、扩大方差”，没有讨论 Q 不确定性、安全约束、技能、可达集、数据质量差异等哪些该强保留、哪些该放松。
2. **噪声强度 $\eta$ 依赖动作域假设**  
    实验环境动作均归一化到 $[-1,1]$，故固定 $\eta=0.1$。若动作量纲差异大、未归一化或高维连续控制，固定 $\eta$ 可能过保守或过激进。
3. **$N_{\mathrm{sample}}=\tfrac{1}{2}\dim(\mathcal{A})$ 是经验折中**  
    论文在 Figure 7 说明性能随候选数上升但边际收益递减；这隐含成本-收益权衡，但缺少对高维任务候选复杂度更理论化的刻画。
4. **熵引导是启发式闭环，不是稳定性证明**  
    $\xi$ 更新依赖 GMM 熵估计，估计误差会反馈到探索强度；论文没有给出该闭环收敛或不会震荡的保证。
5. **迁移到 MARL 的额外不足**
    - 联合动作采样复杂度爆炸；
    - 单 agent 熵不能代表团队探索；
    - 非平稳性使目标熵和 $\xi$ 更新失准；
    - 值分解下联合候选选择可能违反 IGM；
    - 扩张方差更易产生破坏协调的 OOD 联合动作；
    - 没有信用分配与对手适应机制；
    - 部分可观测导致 GMM 熵估计更不准。

---

## 9. 对照算法、发表信息、实验环境与消融/问题验证实验

**主对照算法**：

|算法|论文/出处信息|类型|
|---|---|---|
|ReBRAC|Tarasov et al., NeurIPS 2023, “Revisiting the minimalist approach to offline RL”|Gaussian policy，offline/O2O 强基线|
|Cal-QL|Nakamoto et al., NeurIPS 2023, “Cal-QL: Calibrated offline RL pre-training for efficient online finetuning”|保守 O2O|
|RLPD|Ball et al., ICML 2023, “Efficient online RL with offline data”|在线为主，用离线数据初始化|
|IFQL|文中作为 FQL/IDQL 思想到 flow matching 的改编；IDQL 为 Hansen-Estruch et al. 2023 arXiv，FQL 为 Park et al. ICML 2025|flow 版多候选方法|
|FQL|Park et al., ICML 2025, “Flow Q-Learning”|backbone|

**实验设置**：所有方法统一为 **1M offline pre-training + 500K online fine-tuning**，10 个随机种子，报告均值与 95% CI。评估每 50k 环境步跑 50 episodes。

**环境**：

- **OGBench**：35 个 single-task 任务，7 个环境 × 5 个任务：`humanoidmaze-medium-navigate`, `humanoidmaze-large-navigate`, `antmaze-large-navigate`, `antmaze-giant-navigate`, `antsoccer-arena-navigate`, `cube-double-play`, `puzzle-4x4-play`。humanoid 动作维度 21；antmaze/antsoccer 为 8 维；cube/puzzle 机械臂为 5 维。
- **D4RL**：10 个任务：6 个 antmaze `umaze`, `umaze-diverse`, `medium-play`, `medium-diverse`, `large-play`, `large-diverse`；4 个 adroit cloned：`pen`, `door`, `hammer`, `relocate`，动作维度超过 24。OGBench 与 D4RL antmaze 用 success rate，adroit 用 normalized score。

**主实验**：Table 1、Figure 3 的聚合结果；完整结果在 Table 4、Figures 9-10。FINO 在 humanoidmaze、antmaze-giant、cube、puzzle、D4RL adroit 等困难任务提升明显。例如 Table 1 中 `humanoidmaze-medium-navigate`：FINO $50\pm7\to97\pm1$，FQL 为 $53\pm6\to61\pm2$；`cube-double-play` FINO $34\pm3\to79\pm2$，FQL $31\pm3\to73\pm2$。

**问题验证/动机实验**：

- Figure 1：`antmaze-giant-navigate` 中比较 FQL 与 FINO 的性能和访问频率，验证 offline 点匹配导致 online 探索受限。
- Figure 2：二维动作 toy，四个圆形数据集区域；FLOW 样本几乎只在数据圆内，FINO 围绕数据扩张覆盖更大但合理区域。
- Figure 8 与 Appendix D：解释直接 Action Noise 与 FINO 的差别；FINO 的扩张进入 Q 更高的区域，而动作噪声向随机方向扩散。

**消融与分析实验**：

- Figure 4 / Table 5：`Direct Noise` baseline，证明噪声注入 flow matching 优于直接动作加噪。
- Figure 5 / Table 5：`ER-Noise`，用熵缩放高斯噪声直接加到动作上，证明熵应引导候选采样，而不是简单调噪声尺度。
- Figure 6 左：`w/o Noise`、`w/o Guidance` 控制实验；噪声注入尤其关键，熵引导在后期防止过度利用/退化。
- Figure 7：$N_{\mathrm{sample}}$ 从 $\tfrac{1}{8}\dim(\mathcal{A})$ 到 $\dim(\mathcal{A})$，性能上升但边际递减，故选 $\tfrac{1}{2}\dim(\mathcal{A})$。
- Figure 6 中/右：训练与推理时间；相对 FQL 略增，但远小于 Cal-QL 的训练开销，推理少于 IFQL。
- Appendix D：分析不同注入位置；注入 velocity target 期望梯度不变。

---

## 10. 核心 claim 是什么？是否有理论或实验证明？

可归纳为五个核心 claim。

**Claim 1：标准 flow matching/FQL 的点匹配会在 offline 预训练中过度收缩支撑，限制 online 探索。**

证据：主要是实验与问题验证，而非定理。Figure 1 显示 FQL 在 `antmaze-giant-navigate` online 访问受限；Figure 6 左 `w/o Noise` 性能显著下降；Table 4 中 FINO 相对 FQL 在 navigate/manipulation 多任务提升。

**Claim 2：在 flow matching 的输入路径注入时变噪声，可扩大动作覆盖而保持生成模型有效。**

证据：理论有 Proposition 1、Theorem 1、Theorem 2；实验有 Figure 2 toy、Figure 8、主表。特别是 Theorem 2 给出

$$  
\operatorname{Var}(X_t^{\mathrm{FINO}})\ge\operatorname{Var}(X_t^{\mathrm{FM}}),  
$$

支撑“覆盖更宽”。

**Claim 3：噪声必须注入概率路径/输入，而不是 velocity target 或最终动作。**

证据：Appendix D 对 velocity target 有期望梯度不变的否定性理论；Figure 4、Figure 8 及 door-cloned 等结果说明 Action Noise 不稳定甚至退化，而 FINO 能让 Q 最大化利用扩张区域。

**Claim 4：熵引导候选采样能在有限 online 预算下平衡探索/利用。**

证据：主实验 Table 1/Figure 3；Figure 5 对 ER-Noise 明显优势；Figure 6 左 `w/o Guidance` 后期下降。这里是强实验支持，但没有闭环收敛理论。

**Claim 5：FINO 比保守 O2O、强 online、flow 多候选方法更样本高效，且计算开销可接受。**

证据：Table 1/Table 4 的 45 个任务结果；Figure 3 聚合学习曲线；Figure 6 训练/推理时间；Figure 7 的候选数-性能权衡。该 claim 由主实验与效率分析共同支持。

总结一句：FINO 的最大价值不是单纯“给 flow policy 加噪声”，而是把 O2O 看成一个两阶段耦合问题——offline 用受控噪声注入预置可探索性，online 用熵引导把预置多样性转成样本高效的探索-利用平衡；其方差扩张有定理支撑，但样本高效性、熵闭环稳定性和 distribution shift 仍主要依赖实验与启发式设计。