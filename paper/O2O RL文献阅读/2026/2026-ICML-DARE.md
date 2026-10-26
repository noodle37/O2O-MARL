## 文章主要内容总览

论文提出 **DARE: Dynamic Alignment for RElaxation**，用于 offline-to-online RL（O2O RL）。核心观点是：在线微调时，真正重要的不是样本来自 offline 还是 online，而是样本在“行为空间”中更像 offline 行为还是 online 行为。作者先用诊断实验说明 offline 与 online 数据在行为空间中并非可分，而是有明显重叠，因此按数据来源分配保守/放松约束会造成 objective–data mismatch。DARE 用一个行为模型 $\pi_b$ 产生参考动作 $a_b(s)$，对每个样本计算行为偏差

$$  
d(s,a):=\mathrm{dist}(a,a_b(s)),  
$$

并默认用 $\ell_2$ 距离。随后把样本来源判别建模为关于标量统计量 $D=d(s,a)$ 的 Bayes 后验问题，在高斯类条件分布近似下得到 logistic-quadratic posterior；再根据二次项驻点与对称阈值构造样本级“交换规则”，把一部分 offline-like/online-like 样本在 minibatch 中重新分配约束角色。理论上证明该交换能提高 offline-like 与 online-like 分布的均值间隔和阈值假设类下的 $H\Delta H$ 分离度。实验上在 Cal-QL 与 IQL 上实例化为 DARE-C / DARE-I，在 D4RL locomotion、AntMaze、Adroit、Kitchen 等任务上稳定优于强基线。

---

## 1. 解决 O2O RL 中的什么问题？为什么现有算法没解决好？

解决的核心问题是 **O2O 微调中的“保守约束”与“在线适应”冲突**，更具体说是 **objective–data mismatch**：

- Offline RL 为了避免分布外动作导致 Q 过估计，通常加保守约束，例如 CQL 的显式正则项

$$  
\mathcal{L}_{\mathrm{CQL}}(Q)  
=\mathbb{E}_{(s,a,s')\sim\mathcal{D}}  
\left[\left(Q(s,a)-\hat{\mathcal{B}}^{\pi}\hat{Q}_{\mathrm{target}}(s,a)\right)^2\right]  
+\alpha\,\mathcal{R},  
$$

或 IQL 的 expectile 隐式保守价值回归。

- 进入 online fine-tuning 后，策略不断变化，新收集数据相对 offline 支持集产生 distribution shift。若继续统一施加强保守约束，会限制探索与适配；若直接去掉约束，又容易价值发散。
- 现有方法大体分两类：
    1. 对所有样本用统一目标或软过渡，如 Cal-QL、FamO2O、Uni-RL；
    2. 按数据 origin 区分 offline/online 目标，如 OPT/PEX 类思路。

但作者的关键诊断表明：offline 与 online 样本在行为空间中存在显著重叠，Fig. 1 的 t-SNE 与行为距离分布显示两类样本不是两个分离簇，而是沿连续行为轴交叠。因此“来源标签 $C\in\{0,1\}$”并不能可靠代表样本应扮演的约束角色。统一目标掩盖了样本异质性，而 origin-based 分配又会把本该保守校准的样本错误放松、或把本该支持适应的样本错误保守化。

所以问题尚未被解决的本质原因是：**影响约束需求的是相对 offline 行为支持的行为一致性，而不是数据采集来源；而既有方法大多只在“算法层/相层/来源层”调节约束，缺少样本级、行为驱动、可动态重分配的约束机制。**

---

## 2. 核心创新点

核心创新可概括为三点：

1. **从 origin-based 到 behavior-based 的样本级约束放松。** DARE 不用“offline/online 来源”决定约束，而用每个样本相对行为模型 $\pi_b$ 的偏差 $d(s,a)$ 决定它更像 offline-like 还是 online-like。
2. **Bayes-optimal posterior induced exchange mechanism。** 在似然比可由 $d$ 表示的假设下，证明标量统计量 $D=d(s,a)$ 对来源后验推断充分；在高斯类条件近似

$$  
D\mid C=c\sim \mathcal{N}(\mu_c,\sigma_c^2),\quad c\in\{0,1\},  
$$

下得到

$$
\mathbb{P}(C=0\mid D=d)=\frac{1}{1+\dfrac{\rho_1\sigma_0}{\rho_0\sigma_1}\exp\left(-\dfrac{(d-\mu_1)^2}{2\sigma_1^2}+\dfrac{(d-\mu_0)^2}{2\sigma_0^2}\right)}.
$$

该 posterior 的定性结构由二次函数

$$
\begin{aligned}
\ell(d) &= -\frac{(d-\mu_1)^2}{2\sigma_1^2}+\frac{(d-\mu_0)^2}{2\sigma_0^2} \\
&= \ell_1 d^2+\ell_2 d+\ell_3
\end{aligned}
$$

决定。由此构造阈值 $\tau,\tau_{\mathrm{side}}$ 与方向符号 $\delta_\mu,\delta_\sigma$，形成统一的样本交换规则，而不是靠人工枚举四种情形。  
3. **算法无关、可插拔的样本交换操作。** DARE 只需要 per-sample behavioral alignment，因此可接在 Cal-QL/IQL/TD3+BC/BOORL/LAROO 等上；行为模型既可用 offline 上训练的 energy-guided diffusion，也可用 online policy actor；online-like 样本的目标也可替换为 SAC/TD-only/PEX 等不同目标。

---

## 3. 核心模块、在流程中的作用与完整算法流程

核心模块包括：  
**A. 行为参考模块 $\pi_b$**  
产生参考动作 $a_b(s)$，定义样本行为偏差

$$  
d(s,a)=\lVert a-a_b(s)\rVert_2.  
$$

它把高维状态-动作对压缩成一维行为统计量，是后续后验与交换规则的输入。  
**B. 类条件统计量估计模块**  
周期性从 offline dataset $\mathcal{D}_{\mathrm{off}}$ 与 online buffer $\mathcal{D}_{\mathrm{on}}$ 各采样，估计

$$  
(\mu_0,\sigma_0),\quad (\mu_1,\sigma_1),  
$$

分别对应 offline/online 两类样本的 $D$ 分布。论文实现中每 1000 步用各 10,000 个样本更新。  
**C. 后验与阈值模块**  
定义

$$  
\delta_\mu=\mathrm{sign}(\mu_0-\mu_1),\qquad  
\delta_\sigma=\mathrm{sign}(\sigma_0-\sigma_1),  
$$

$$
d^{\star}=\frac{\mu_1/\sigma_1^2-\mu_0/\sigma_0^2}{1/\sigma_1^2-1/\sigma_0^2},\qquad
\tau=\frac{\mu_0+\mu_1}{2},\qquad
\tau_{\mathrm{side}}=2d^{\star}-\tau.
$$

其中 $d^{\star}$ 是 posterior logistic 二次项的驻点，$\tau$ 是均值中点，$\tau_{\mathrm{side}}$ 是 $\tau$ 关于 $d^{\star}$ 的反射阈值，用来刻画由方差差异导致的“侧边区域”。  
**D. 样本交换规则**  
对来自 $C=0$ 的样本，交换到 $C=1$ 当且仅当

$$  
\delta_\mu\delta_\sigma(d-\tau_{\mathrm{side}})\gt 0  
\quad\mathrm{and}\quad  
\delta_\mu(d-\tau)\lt 0.  
$$

对来自 $C=1$ 的样本，交换到 $C=0$ 当且仅当

$$  
\delta_\mu\delta_\sigma(d-\tau_{\mathrm{side}})\gt 0  
\quad\mathrm{and}\quad  
\delta_\mu(d-\tau)\gt 0.  
$$

含义是：$\delta_\mu(d-\tau)$ 判断样本是否越过均值中点、更接近另一类；$\delta_\mu\delta_\sigma(d-\tau_{\mathrm{side}})\gt 0$ 判断样本是否落在由驻点诱导的方差主导侧区。只有两个条件同时满足才交换，因此只重分配“行为上模棱两可且位于后验侧区”的高置信样本，避免对不确定样本激进放松。  
**E. 平衡交换操作**  
在每个 minibatch $b=b_{\mathrm{off}}\cup b_{\mathrm{on}}$ 上，先形成候选池

$$  
\mathcal{P}_{\mathrm{off}\to\mathrm{on}},\quad  
\mathcal{P}_{\mathrm{on}\to\mathrm{off}},  
$$

再平衡交换数量

$$  
K=\min\left(  
\lvert\mathcal{P}_{\mathrm{off}\to\mathrm{on}}\rvert,  
\lvert\mathcal{P}_{\mathrm{on}\to\mathrm{off}}\rvert  
\right),  
$$

可选上界 $K_{\max}$。得到交换后批次 $b'_{\mathrm{off}}, b'_{\mathrm{on}}$。  
**F. 算法特定目标模块**

- **DARE-C**：offline-like 的 $b'_{\mathrm{off}}$ 保留 Cal-QL 正则项 $\alpha\mathcal{R}$，online-like 的 $b'_{\mathrm{on}}$ 去掉正则项：

$$
\begin{aligned}
\mathcal{L}_{\mathrm{DARE}}^{\mathrm{CalQL}}(Q)
=\mathbb{E}\Big[&\mathbf{1}_{b'_{\mathrm{off}}}\big((Q-\hat{\mathcal{B}}^{\pi}\hat{Q})^2+\alpha\mathcal{R}\big) \\
+&\mathbf{1}_{b'_{\mathrm{on}}}(Q-\hat{\mathcal{B}}^{\pi}\hat{Q})^2\Big].
\end{aligned}
$$

- **DARE-I**：offline-like 样本保留 IQL value target 与 PEX 策略目标；online-like 样本把 value target 换成 TD-based $\max Q$

$$  
r+\gamma\max_{a'}\hat{Q}_{\mathrm{target}}(s',a'),  
$$

策略更新改用 entropy-regularized SAC。  
**完整流程**：初始化 offline 组件 $\mathcal{C}$ 与行为模型 $\pi_b$；在线每步收集样本入 $\mathcal{D}_{\mathrm{on}}$；采样混合 minibatch；计算 $d(s,a)$；用最近 buffer-level 统计量执行 exchange rule；形成候选池并平衡交换得到 $b'_{\mathrm{off}}, b'_{\mathrm{on}}$；对两类样本施加不同 conservative/adaptive 目标更新 critic/actor/value；周期性更新 $(\mu_c,\sigma_c)$。

---

## 4. 是否进行理论证明？简要概述

有，且较完整。主线是“为什么一维行为统计量足够”和“为什么交换会增强可分性”。

- **Thm. 1 后验充分性**：若类条件密度满足似然比表示

$$  
\frac{p(X=x\mid C=0)}{p(X=x\mid C=1)}=r(d(s,a)),  
$$

则存在可测函数 $g$ 使

$$  
\mathbb{P}(C=0\mid X=x)=g(d(s,a)).  
$$

即高维状态-动作判别可约化到标量行为偏差 $D$。

- **Thm. 2 驻点位置**：logistic-quadratic posterior 的驻点

$$
d^{\star}=\frac{\mu_1/\sigma_1^2-\mu_0/\sigma_0^2}{1/\sigma_1^2-1/\sigma_0^2}
$$

满足 $d^{\star}\notin(\mu_0,\mu_1)$。因此在两均值之间 posterior 严格单调，均值中点 $\tau$ 可作为一致分离参考；outside 区间才由驻点与反射阈值 $\tau_{\mathrm{side}}$ 处理方差主导侧区。

- **Prop. 1**：符号形式交换规则等价于 Fig. 2 中四种 $(\mu_0,\mu_1),(\sigma_0,\sigma_1)$ 相对顺序下的区间条件。
- **Thm. 3 均值间隔增强**：若交换 $x\in\mathcal{B}_0,\, y\in\mathcal{B}_1$ 满足

$$  
\delta_\mu(x-\tau)\lt 0,\qquad \delta_\mu(y-\tau)\gt 0,  
$$

交换后

$$  
\Delta'=\Delta+\kappa(y-x),\quad  
\kappa=\frac{1}{n_0}+\frac{1}{n_1}\gt 0,  
$$

且 $\delta_\mu(y-x)\gt 0$，故

$$
\lVert\Delta'\rVert=\lVert\Delta\rVert+\kappa\lVert y-x\rVert\gt\lVert\Delta\rVert.
$$

- **Lem. 1 + Cor. 1 + Thm. 4 分布分离不下降**：对经验 CDF $F_c$，单次交换使

$$
\Delta F'(t)=\Delta F(t)-\delta_\mu\kappa\,\mathbf{1}\{t\in I\},
$$

其中 $I=\{t:\delta_\mu x\le \delta_\mu t\lt \delta_\mu y\}$。对阈值假设类

$$
H_{\mathrm{thr}}=\{h_t(z)=\mathbf{1}\{z\le t\}\},
$$

有

$$
d_{H_{\mathrm{thr}}\Delta H_{\mathrm{thr}}}(P,Q)=2(\Delta F_{\max}-\Delta F_{\min}).
$$

交换只在一个区间上平移 $\Delta F$，不会缩小其值域，因此

$$  
d'_{H_{\mathrm{thr}}\Delta H_{\mathrm{thr}}}(P',Q')  
\ge  
d_{H_{\mathrm{thr}}\Delta H_{\mathrm{thr}}}(P,Q).  
$$

- **Appendix B 高斯假设稳健性**：用 Gamma 族建模非负距离 $d\gt 0$，其似然比驻点为

$$
d^{\star}_{\mathrm{Gamma}}=\frac{\mu_1^2\sigma_2^2-\mu_2^2\sigma_1^2}{\mu_1\sigma_2^2-\mu_2\sigma_1^2},
$$

与高斯驻点差异

$$
\delta_p=d^{\star}_{\mathrm{Gamma}}-d^{\star}_{\mathrm{Gaussian}}
=\frac{(\mu_1-\mu_2)^2\sigma_1^2\sigma_2^2}{(\mu_1\sigma_2^2-\mu_2\sigma_1^2)(\sigma_2^2-\sigma_1^2)}
$$

对均值间隔是二阶小量；并且仍保持 $d^{\star}_{\mathrm{Gamma}}\notin(\min\{\mu_1,\mu_2\},\max\{\mu_1,\mu_2\})$。说明 Thm. 2 的关键结论不依赖高斯假设。

局限是：理论证明集中在“交换规则增强一维投影后的统计可分性”，没有证明端到端 O2O 的价值估计收敛或策略改进界。

---

## 5. 是否讨论 Offline 阶段什么知识值得保留、Online 如何运用、Offline 知识是否会阻碍 Online？

有隐含讨论，但不是以“知识保留/迁移”的术语系统展开。  
论文实际区分了两类 offline 知识：

1. **保守校准知识**：CQL 正则项 $\alpha\mathcal{R}$、IQL expectile value target、PEX 的 offline-consistent policy regularization。这些被保留给 $b'_{\mathrm{off}}$ 中的 offline-like 样本，用来稳定价值估计、防止 OOD 动作过估计。
2. **行为先验/行为支持知识**：由 $\pi_b$ 表示的 offline 行为模型，以及由它诱导的 $a_b(s)$。它在 online 阶段被用作“行为参考”，通过 $d(s,a)$ 判断样本是否偏离 offline 行为支持，从而决定放松或保留约束。  
    Online 阶段运用方式不是简单丢弃 offline 知识，而是“选择性继承”：offline-like 样本继续用保守目标，online-like 样本改用更适应性的目标，如 TD 的 $\max Q$、entropy-regularized SAC。行为模型在 fine-tuning 中还会继续更新，因此参考不是完全静态的。  
    关于阻碍，论文明确承认：**offline 保守知识在 online 阶段可能构成阻碍**。当策略进入在线探索后，行为分布漂移，继续统一施加保守约束会与探索冲突；作者在 Limitations 进一步指出，若 offline dataset coverage 不足或与 online trajectory 明显 mismatch，行为模型 $\pi_b$ 的估计会不可靠，进而影响从 offline regularization 到 online adaptation 的过渡。也就是说，offline 知识并非越多越好：无差别保留会抑制适应；质量差的行为先验还会误导样本交换。  
    不过，本文没有深入讨论更广泛的“哪些表征/技能/价值知识值得保留”“如何防止灾难性遗忘”“offline 学到的 Q/策略在 online 的 reusability”等知识迁移问题；它聚焦的是“保守约束知识是否应按样本动态放松”。

---

## 6. 是否适合迁移到 MARL？理由与挑战

**部分适合，但不能直接照搬；更适合作为 O2O-MARL 中“样本级保守/探索再分配”的候选机制。**  
适合迁移的理由：

- O2O-MARL 同样有 offline 保守预训练与 online 探索冲突；多智能体 online 交互更昂贵，更需要安全、稳定地从 offline 到 online。
- 队友策略会漂移，数据来源标签 $C$ 在 MARL 中同样不可靠；行为一致性思想比 origin 标签更合理。
- DARE 只依赖 per-sample alignment，理论上可把单智能体动作 $a$ 替换为联合动作 $\mathbf{a}=(a^1,\dots,a^N)$，定义

$$
d(\mathbf{s},\mathbf{a})=\lVert \mathbf{a}-\mathbf{a}_b(\mathbf{s})\rVert_2,
$$

仍可用一维统计量做后验交换。  
主要挑战：

1. **联合动作空间与行为建模更难。** 连续 MARL 的联合动作维度高且多模态；$\pi_b(\mathbf{a}\mid\mathbf{s})$ 要建模队友协同策略，diffusion/energy-guided 成本显著上升。
2. **非平稳性更强。** 即使环境平稳，其他智能体策略变化也会使 online 数据分布持续漂移；buffer-level 的 $(\mu_c,\sigma_c)$ 周期估计更容易滞后，Gaussian/Gamma 类条件假设更弱。
3. **似然比充分性假设更难成立。** Thm. 1 要求类条件似然比可由标量 $d$ 表示；在 MARL 中，同一联合动作偏差可能对应不同队友意图/信用分配结果，单靠 $d$ 可能不足。
4. **部分观测与通信限制。** 若智能体只观测局部 $o^i$，则 $\mathbf{s}$ 不充分，$a_b$ 与 $d$ 的参考意义下降；需要 belief/通信/中心化 critic 近似。
5. **个体约束放松可能破坏协同。** 单智能体中放松某样本约束主要影响自身探索；MARL 中放松一个智能体的约束可能改变联合动作分布，损害其他智能体或团队回报，需要 team-level credit 与稳定性约束。
6. **理论分离度不等于策略改进。** Thm. 3/4 只保证一维行为投影更可分；MARL 还需要处理均衡、对手适应、协调博弈和信用分配，样本交换不直接推出团队策略改进。

因此，迁移时应把 DARE 的“交换规则”保留，但必须重构行为模型、状态/动作定义、统计更新频率与安全约束。

---

## 7. 结合问题 5：Offline 知识保留/Online 运用/Offline 知识阻碍，在 MARL 中是否不同、是否更具挑战性？

不同，且更具挑战性。  
在单智能体 O2O 中，本文的知识运用较清晰：offline 提供保守约束与行为先验，online 通过 $d(s,a)$ 选择哪些样本继续被保守约束锚定，哪些样本切换到探索目标。主要风险是行为模型不准导致错误放松。  
在 MARL 中，同样问题会被放大：

- **Offline 阶段值得保留的知识更复杂。** 不只是单智能体保守正则，还包括 teammate model、协同模式、联合价值分解、通信协议、角色分配等。哪些知识在 online teammates 改变后仍有效，本身更难判断。
- **Online 运用知识更难。** 单智能体可用 $a_b(s)$ 判断动作是否离线一致；MARL 需要判断联合动作 $\mathbf{a}$ 是否仍符合历史协同。即使个体动作都像 offline，组合起来也可能产生未见过且糟糕的联合效果；反之个体偏离 offline，也可能是必要协同探索。
- **Offline 知识更可能成为阻碍。** 队友策略漂移在 MARL 中是常态，offline 学到的联合行为先验可能快速过时；保守保留会抑制协同探索，过度放松又可能破坏 coordination。本文已指出 offline coverage/mismatch 会影响行为模型，MARL 中“coverage”还包括联合策略覆盖与队友组合覆盖，条件更苛刻。
- **环境非平稳与联合动作叠加。** 问题 5 在单智能体中主要是“行为支持是否变化”；在 MARL 中变成“环境回报结构 + 队友策略 + 联合动作支持 + 信用分配”同时变化。Gaussian/Gamma 局部分布近似与周期更新更容易失效。
- **交换样本角色不等于交换团队知识。** DARE 交换的是 minibatch 中样本的 conservative/adaptive 角色；MARL 还需要决定团队目标、个体探索率、通信是否开启、信用如何分配，这些超出 DARE 当前机制。

所以，DARE 的思想可迁移，但“保留什么 offline 知识、online 用哪些知识、哪些 offline 知识会阻碍 online”在 MARL 中没有简单答案，需要把行为一致性扩展到联合策略与队友意图层面。

---

## 8. 本文仍存在哪些不足？结合前述问题，尤其迁移到 MARL 的不足

主要不足：

1. **行为模型质量依赖强。** Limitations 已承认：offline dataset coverage 有限或与 online trajectory mismatch 时，$\pi_b$ 的行为估计不可靠。MARL 中这一点更严重，因为联合动作覆盖和队友组合覆盖更难充分。
2. **理论只到交换规则层面。** Thm. 3/4 证明均值间隔与 $H\Delta H$ 分离增强，但没有连接端到端价值误差、策略改进率或收敛保证。MARL 中还需要 team return、credit assignment、均衡稳定性的理论。
3. **高斯/Gamma 是局部近似。** Appendix B 缓解了 Gaussian 敏感，但真实 behavior-distance 分布可能多峰、重尾、随训练强非平稳；非平稳 MARL 中更脆弱。
4. **交换只改目标角色，不解决表征/技能保留。** 问题 5 中更深层的“offline 表征是否应冻结、价值先验是否可重用、如何避免遗忘”未系统讨论。
5. **计算成本。** Diffusion-based behavior model 带来额外开销；Tab. 15 显示 DARE-C(Diffusion) 在 hopper-medium-replay 上约 2.1h，高于 Cal-QL 0.5h 和 EDIS-C 1.6h；policy-based 更快但略低性能。MARL 中联合扩散模型成本会进一步放大。
6. **迁移 MARL 的结构性缺口。** 需要中心化训练/分散执行下的行为参考、联合动作统计、队友策略漂移建模、通信约束、信用分配和安全探索；这些在本文中没有处理。特别是把个体样本错误标记为 online-like 可能不仅影响自己，还通过联合动作影响团队稳定性。

---

## 9. 对照算法、发表年份/会议、实验设置、环境、消融与问题验证实验

**主对照算法**及出处按论文参考文献：

- CQL：Kumar et al., **NeurIPS 2020**。
- IQL：Kostrikov et al., **ICLR 2022**。
- Cal-QL：Nakamoto et al., **NeurIPS 2023**。
- PEX：Zhang, Xu & Yu, **ICLR 2023**。
- EDIS：Liu et al., **ICML 2024**；文中分 EDIS-C 与 EDIS-I。
- TD3+BC：Fujimoto & Gu, **NeurIPS 2021**，附录中用于 BOORL/LAROO 扩展；BOORL 对应 Hu et al. **ICML 2024**，LAROO 对应 Guo et al. **ICLR 2026**（参考文献年份）。  
    **实验设置**：D4RL。主文为 MuJoCo Locomotion 与 AntMaze；另报告 Adroit、Kitchen 附录。所有方法先 offline 训练 1M 步，再 online fine-tune 0.2M 步；batch size 256；混合 minibatch $b=b_{\mathrm{off}}\cup b_{\mathrm{on}}$；结果为最后 4 次评估、5 个随机种子均值 $\pm$ 标准差。DARE 仅新增一个主要超参：$(\mu_c,\sigma_c)$ 更新频率，论文设为每 1000 步，用 offline/online 各 10,000 样本估计。  
    **主实验结果**：Tab. 1 中 CQL 组总分 DARE-C 为 1415.4，高于 EDIS-C 1338.2、Cal-QL 1272.7、CQL 1219.5；IQL 组 DARE-I 为 1253.0，高于 PEX 1165.6、EDIS-I 1118.5、IQL 1085.3。DARE 在 CQL 组 15 个任务中 13 个最高，在 IQL 组 12 个最高；总提升约 11%。Locomotion 上，例如 HC-MR：DARE-C 77.6 vs Cal-QL 51.1；H-MR：DARE-I 104.2 vs PEX 93.5；AntMaze-umaze-diverse：DARE-I 69.2 vs IQL 31.6、PEX 30.6、EDIS-I 34.9。  
    **问题验证/分析/消融**：
- **Q1 主性能**：Tab. 1、App. E 学习曲线；附录 F 扩展 Kitchen、TD3+BC/BOORL/LAROO。
- **Q2 交换是否活跃自适应**：Fig. 3、App. G Fig. 9 显示交换数多在 20–50；Fig. 4/13 显示估计的 $\Delta\mu,\Delta\sigma$ 并不保持固定单调顺序，说明规则必须自适应。
- **Q3 Bayes posterior 是否必要**：Fig. 5 比较 posterior-induced exchange、monotonic Gaussian、direct alignment-distance；posterior 更稳，直接/单调方法在分布顺序反转时不可靠。
- **Q4 不同行为模型/online 目标**：Tab. 2、Tab. 11、Tab. 13、Fig. 10–13；包括 DARE-I.w/o S、w/o P、w/o S&P，以及 policy-based vs diffusion-based behavior model。
- **Q5 反行为分配验证**：Tab. 3、Fig. 14–15；把 online-like 分到 offline、offline-like 分到 online 的 curiosity-style 反向操作显著变差，支持“行为一致性”作为角色分配依据。
- **Q6 去掉交换机制**：Fig. 6、Fig. 16；DARE-C/DARE-I 均优于 w/o Exchange，平均差异为正。
- **额外消融**：交换上限 $K_{\max}=32$ 总分 877.1，$K_{\max}=128$ 实际不封顶为 878.2；距离度量 $L_1$ / Mahalanobis / $L_2$ 在 Tab. 14 中差异不大；runtime 在 Tab. 15。

---

## 10. 核心 claim 及证明/实验支撑

**Claim 1：数据来源不是可靠的约束分配依据，offline/online 行为重叠会导致 origin-based 目标错配。**  
支撑：诊断实验 Fig. 1；反行为分配实验 Tab. 3/Fig. 14–15；直接距离与单调 Gaussian 失败 Fig. 5；w/o exchange 下降 Fig. 6/16。  
**Claim 2：行为偏差统计量 $d(s,a)$ 可作为 Bayes 后验推断的充分标量，posterior-induced exchange 有原则来源。**  
支撑：Thm. 1；高斯 posterior 闭式解；Fig. 2 四情形结构；Prop. 1 等价性；Appendix B 的 Gamma 稳健性。  
**Claim 3：交换规则能 provably 增强 offline-like 与 online-like 的可分性。**  
支撑：Thm. 3 严格增大 $\lVert\mu_0-\mu_1\rVert$；Lem. 1/Cor. 1/Thm. 4 保证阈值 $H\Delta H$ divergence 不下降；实验侧 Fig. 3/4/9/13 显示交换持续发生且统计量关系非平凡变化。  
**Claim 4：样本级约束放松能提升 O2O fine-tuning 稳定性与最终性能，并且机制算法无关。**  
支撑：主实验 Tab. 1；学习曲线 App. E；Adroit/Kitchen/TD3+BC/BOORL/LAROO 附录结果；不同 online objective 与 behavior model 下仍有效 Tab. 2/11/13；距离度量不敏感 Tab. 14；runtime/开销分析 Tab. 15。

需要注意：最强的 claim 是“交换规则增强统计可分性”和“实验中稳定提升”；较弱或未完成的是“端到端 O2O 收敛/策略改进保证”，以及“offline 知识哪些应保留”的系统性答案。