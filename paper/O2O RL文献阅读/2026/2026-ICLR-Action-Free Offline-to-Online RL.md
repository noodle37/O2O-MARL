# 一、论文主要内容总结

本文研究的是一种比普通 Offline-to-Online RL 更困难的问题：

> **当离线数据完全没有动作标签，只包含 \((s,r,s')\) 元组时，如何从离线数据中提取可迁移知识，并用它加速在线 RL。**

作者将这一设置称为 **Action-Free Offline-to-Online RL**。它适用于医疗、金融、机器人等场景中，由于隐私、存储或传感器限制，只记录了状态变化，而没有记录具体动作。

论文提出的框架包含两个阶段：

1. **Offline 阶段：OSO-DecQN**
    - 将连续状态差分离散为“增加、不变、减少”三类；
    - 学习 \(Q(s,\Delta s)\)，即在当前状态下选择某种离散状态变化的价值；
    - 使用 value decomposition、ensemble、double Q-learning 和保守正则化保证可扩展性与稳定性。
2. **Online 阶段：状态策略引导学习**
    - 用离线学到的 \(Q(s,\Delta s)\) 选择期望的状态变化；
    - 在线训练一个逆动力学模型 IDM，将 \(\Delta s\) 翻译为可执行动作；
    - 通过概率 \(\beta\) 在“离线状态策略动作”和“在线 agent 自身动作”之间切换；
    - 在线训练过程中逐渐降低 \(\beta\)，实现从离线引导探索向在线自主策略过渡。

实验表明，该方法能够在 D4RL 和 DeepMind Control 类任务上提高在线学习速度和渐近性能，并扩展到最高 78 维状态空间。

---

# 二、十个问题逐一回答

## 1. 本文解决 O2O RL 中的什么问题？为什么现有算法尚未解决好？

### 1.1 本文解决的问题

本文解决的是 **Action-Free Offline-to-Online RL**：

$$  
\mathcal D_{\mathrm{off}}=\{(s,r,s')\}  
$$

而不是标准离线 RL 中常见的：

$$  
\mathcal D_{\mathrm{off}}=\{(s,a,r,s')\}  
$$

因此，离线阶段不能直接学习：

$$  
Q(s,a)  
$$

也不能直接训练可执行动作策略：

$$  
\pi(a|s)  
$$

本文的关键问题是：

> **在没有动作标签的离线数据上，离线阶段应该学习什么知识，才能在 online 阶段被转化为可执行的引导信号？**

### 1.2 为什么这个问题难？

主要有四个原因。

#### 1. 标准 offline RL 几乎都依赖动作标签

标准离线 RL 的 critic、actor、policy constraint、conservative regularization 都建立在 \(Q(s,a)\) 上。没有动作 \(a\)，这些方法无法直接应用。

#### 2. 学习连续下一状态回归很不稳定

一种自然替代方法是直接学习：

$$  
s'\quad\text{或}\quad s'-s  
$$

但论文实验表明，直接模仿连续下一状态或连续状态差分会导致很差的预测性能。连续高维回归容易：

- 不稳定；
- 过拟合；
- 产生不可达状态；
- 无法有效捕捉哪些状态变化真正具有高长期价值。

#### 3. 现有 state-only 方法缺乏稳定、可扩展的在线引导机制

AF-Guide 使用 Decision Transformer 学习 state policy，但：

- Transformer 预训练成本高；
- 使用 guiding reward 的在线引导机制缺乏一致性；
- 在多个环境和不同数据质量下没有稳定优于在线基线。

#### 4. 现有动作离散化方法不能直接用于 action-free 数据

DecQN 等方法离散化的是动作：

$$  
a_i\in\mathcal A_i  
$$

但 action-free 数据没有动作，因此不能直接拿来用。本文将离散化对象从“动作”改成了“归一化后的状态差分”。

### 1.3 现有算法为什么还没有完整解决？

现有方法通常只满足以下三项中的一部分：

1. 能从 action-free 离线数据中学习；
2. 能扩展到高维状态空间；
3. 能在 online 阶段持续、稳定地加速 online agent。

本文的动机就是同时满足这三点。

---

## 2. 本文的核心创新点是什么？

## 2.1 核心创新一：状态差分离散化

对每个状态维度，将归一化后的状态变化离散为：

$$  
\Delta s_i\in\{-1,0,1\}  
$$

具体地：

$$  
\Delta s_i^\epsilon(s,s')=  
\begin{cases}  
-1, & z(s_i')-z(s_i)<-\epsilon\\  
1, & z(s_i')-z(s_i)>\epsilon\\  
0, & \text{otherwise}  
\end{cases}  
$$

其中 \(z(\cdot)\) 表示 z-score 归一化。

这使模型学习：

- \(+1\)：该状态维度应增加；
- \(0\)：该状态维度基本不变；
- \(-1\)：该状态维度应减少。

这一变换具有 scale-invariant 特性，不直接依赖原始状态维度的数值范围。

## 2.2 核心创新二：OSO-DecQN

本文将 DecQN 的动作维度分解扩展为 **状态差分维度分解**：

$$
Q_\theta(s,\Delta s)
=
\frac{1}{M}
\sum_{j=1}^{M}
U_{\theta_j}^{j}(s,\Delta s_j)
$$

其中 \(M\) 是状态维度。

由于每个状态维度只有三个离散选择，因此：

$$  
\arg\max_{\Delta s}Q_\theta(s,\Delta s)  
$$

可以逐维求解，而不必枚举 \(3^M\) 个联合状态变化。

## 2.3 核心创新三：面向 action-free offline RL 的保守正则化

正则项为：

$$
\mathcal R_\theta
=
\sum_{(s,\Delta s)\sim\mathcal D}
\left[
\log
\sum_{\Delta s'}
\exp Q_\theta(s,\Delta s')
-
Q_\theta(s,\Delta s)
\right]
$$

它有两个作用：

1. 抑制离线数据中未覆盖的状态变化的过高价值；
2. 让预测状态变化保持在离线数据支持的可达范围内。

这对应 action-free 设置下特有的 **state reachability constraint**。

## 2.4 核心创新四：在线 IDM 与 policy switching

离线状态策略不能直接执行，因为它输出的是期望状态变化，而不是动作。因此本文在线学习一个逆动力学模型：

$$  
I_\phi(s,\Delta s)\approx a  
$$

动作选择规则为：

$$  
a=  
\begin{cases}  
I_\phi\left(s,\arg\max_{\Delta s}Q(s,\Delta s)\right), & \zeta<\beta\\  
\pi_{\mathrm{on}}(s), & \text{otherwise}  
\end{cases}  
$$

其中：

$$  
\zeta\sim U(0,1)  
$$

\(\beta\) 控制 offline guidance 和 online policy 的混合比例。

---

## 3. 本文有哪些核心模块？它们如何在算法流程中发挥作用？完整流程是什么？

## 3.1 模块一：状态差分离散化模块

输入离线转移：

$$  
(s,r,s')  
$$

先对每个状态维度做归一化，再计算离散状态差分：

$$  
\Delta s\in\{-1,0,1\}^{M}  
$$

该模块把不稳定的连续状态预测问题转化为离散分类 / Q-learning 问题。

## 3.2 模块二：OSO-DecQN 离线状态价值模块

学习：

$$  
Q_\theta(s,\Delta s)  
$$

即“在当前状态 \(s\) 下，促成离散状态变化 \(\Delta s\) 的价值”。

算法中，下一状态的 bootstrap 状态变化根据 critic 的 softmax 采样：

$$  
\Delta s'  
\sim  
\mathrm{softmax}(Q_\theta(s',\cdot))  
$$

TD target 为：

$$
y
=
r+\gamma Q_{\bar\theta}(s',\Delta s')
$$

训练目标为：

$$
\mathcal L(\theta)
=
\left[y-Q_\theta(s,\Delta s)\right]^2
+
\alpha\mathcal R_\theta
$$

其中：

- 第一项是 Q-learning 回归；
- 第二项是保守正则；
- \(\bar\theta\) 是 target network 参数。

## 3.3 模块三：隐式状态策略

离线训练完成后，隐式状态策略定义为：

$$
\Delta s^\star(s)
=
\arg\max_{\Delta s}
Q_\theta(s,\Delta s)
$$

由于 value decomposition，可逐维求解：

$$
\Delta s^\star(s)
=
\left(
\arg\max_{\Delta s_1}U^1(s,\Delta s_1),
\ldots,
\arg\max_{\Delta s_M}U^M(s,\Delta s_M)
\right)
$$

## 3.4 模块四：在线逆动力学模型 IDM

IDM 学习：

$$  
I_\phi(s,\Delta s)\rightarrow a  
$$

其在线监督损失为：

$$
\mathcal L_{\mathrm{IDM}}(\phi)
=
\|a-I_\phi(s,\Delta s)\|_1
$$

其中：

- \(a\) 是 online agent 实际执行的动作；
- \(\Delta s\) 是实际观察到的离散状态变化。

论文还引入 offline relabelling/bootstrapping 项。对离线状态转移，用当前 online policy 生成伪动作：

$$  
a_{\mathrm{on,off}}=\pi_{\mathrm{on}}(s_{\mathrm{off}})  
$$

然后训练：

$$  
I_\phi(s_{\mathrm{off}},\Delta s_{\mathrm{off}})  
\approx  
a_{\mathrm{on,off}}  
$$

完整 IDM 损失为：

$$
\mathcal L(\phi)
=
\|a_{\mathrm{on,off}}-I_\phi(s,\Delta s_{\mathrm{off}})\|_1
+
\|a-I_\phi(s,\Delta s)\|_1
$$

注意，这不是恢复真实离线动作，而是利用离线状态转移扩充 IDM 训练数据。

## 3.5 模块五：policy switching

在线动作选择：

$$  
a=  
\begin{cases}  
I_\phi(s,\Delta s^\star(s)), & \zeta<\beta\\  
\pi_{\mathrm{on}}(s), & \text{otherwise}  
\end{cases}  
$$

论文采用退火策略：

$$  
\beta_{\max}=0.5  
$$

每 100k timestep 减少 0.1，直到：

$$  
\beta_{\min}=0  
$$

因此，训练早期更多利用离线状态策略探索；后期完全依赖在线策略。

---

## 3.6 完整算法流程

### Offline 阶段

1. 读取 action-free 数据集：

$$  
\mathcal D=\{(s,r,s')\}  
$$

1. 对状态维度做 z-score 归一化。
2. 将 \((s,s')\) 转换为离散状态差分：

$$  
\Delta s\in\{-1,0,1\}^{M}  
$$

1. 初始化 ensemble critic 和 target critic。
2. 重复以下步骤：
    - 采样 \((s,r,s')\)；
    - 计算数据中的 \(\Delta s\)；
    - 在下一状态采样 bootstrap \(\Delta s'\)；
    - 计算：

$$  
y=r+\gamma Q_{\bar\theta}(s',\Delta s')  
$$

- 最小化：

$$  
(Q_\theta(s,\Delta s)-y)^2+\alpha\mathcal R_\theta  
$$

- 更新 target network。

1. 得到隐式状态策略：

$$  
\Delta s^\star(s)=\arg\max_{\Delta s}Q_\theta(s,\Delta s)  
$$

### Online 阶段

1. 初始化 online agent，例如 TD3、SAC 或 DecQN\(_N\)。
2. 从零初始化 IDM。
3. 在每个时间步：
    - 用离线 critic 计算：

$$  
\Delta s^\star(s)  
$$

- 以概率 \(\beta\) 执行：

$$  
a=I_\phi(s,\Delta s^\star(s))  
$$

- 否则执行：

$$  
a=\pi_{\mathrm{on}}(s)  
$$

1. 将在线转移存入 replay buffer。
2. 用 replay buffer 更新 online agent。
3. 用在线动作和真实状态变化更新 IDM。
4. 用离线 \((s,\Delta s)\) 和当前 online policy 生成的伪动作，对 IDM 做额外 bootstrapping 训练。
5. 随训练逐步降低 \(\beta\)，最终完全使用在线策略。

---

# 4. 本文是否进行了理论证明？如有，请简要概述

## 4.1 有理论结果，但不是完整算法收敛证明

本文给出了一个关于 **状态差分离散化造成的价值函数近似误差** 的定理。

### 定理含义

如果将每个状态维度上的连续状态增量划分为 \(k\) 个均匀 bin，则原始连续 MDP 与离散化 MDP 的最优价值函数差异满足：

$$
\left\|V^\star-V_D^\star\right\|_{\infty}
=
O\left(\frac{H\sqrt{M}}{k}\right)
$$

其中：

- \(M\) 是状态维度；
- \(H=\delta_s^{\max}-\delta_s^{\min}\) 是每维平均状态增量的范围；
- \(k\) 是每个状态维度上的 bin 数。

该结果说明，离散化越细，即 \(k\) 越大，价值函数近似误差越小。

### 附录中的完整界

在附录的更严格假设下，有：

$$  
\left\|V^\star-V_D^\star\right\|_{\infty}  
\leq  
\frac{\Delta_r}{1-\gamma}  
+  
\frac{\gamma}{4(1-\gamma)^2}  
\Delta_r  
\sqrt{\Lambda_\Delta}  
\frac{H\sqrt M}{k}  
$$

其中：

$$  
\Delta_r=r_{\max}-r_{\min}  
$$

$$
\Lambda_\Delta
=
\sup_s
\lambda_{\max}
\left(
\Sigma_\Delta(s)^{-1}
\right)
$$

主要假设包括：

1. 奖励有界；
2. \(\gamma\in[0,1)\)；
3. 状态增量满足高斯分布；
4. 增量协方差与动作无关；
5. 各维平均增量有界；
6. 每维使用均匀 bin 离散化。

### 需要注意

这个定理只说明：

> **状态差分离散化本身带来的 MDP 近似误差可以随 bin 数增加而减小。**

它没有证明：

- OSO-DecQN 收敛；
- 在线 guided learning 收敛；
- IDM 学习收敛；
- policy switching 收敛；
- 算法的样本复杂度；
- 离线知识一定能提升在线性能。

另外，完整上界中存在一个：

$$  
\frac{\Delta_r}{1-\gamma}  
$$

的常数项，它不随 \(k\) 减小。主文将其简写成 \(O(H\sqrt M/k)\) 是在把奖励范围、折扣和协方差相关量视为固定常数时得到的缩放形式。因此，这是一个有条件下界形式的近似结果，而不是严格的渐近一致性证明。

---

# 5. 本文是否讨论了 Offline 阶段什么知识值得保留、Online 阶段如何使用，以及 Offline 知识是否会阻碍 Online 学习？

## 5.1 本文明确保留的离线知识

本文保留的核心离线知识不是动作策略，而是：

$$  
Q(s,\Delta s)  
$$

以及由它定义的隐式状态策略：

$$  
\Delta s^\star(s)=\arg\max_{\Delta s}Q(s,\Delta s)  
$$

可以将其理解为离线阶段学到的：

- 哪些状态变化有长期价值；
- 哪些状态维度应该上升、下降或保持不变；
- 与离线数据兼容的高价值状态转移方向。

此外，离线状态转移 \((s,\Delta s)\) 也被保留下来，用于在线训练 IDM 的 bootstrapping 项。

## 5.2 本文没有保留什么

由于离线数据没有动作，本文不能直接保留：

- 离线 actor；
- 可执行动作策略；
- \(Q(s,a)\)；
- 离线动作价值函数。

IDM 也没有在离线阶段预训练，而是在 online 阶段从零学习。

## 5.3 Online 阶段如何使用离线知识

离线知识通过三个环节转化为在线动作：

### 第一步：离线 critic 选择目标状态变化

$$  
\Delta s^\star=\arg\max_{\Delta s}Q(s,\Delta s)  
$$

### 第二步：IDM 将目标状态变化翻译为动作

$$  
a_{\mathrm{guide}}=I_\phi(s,\Delta s^\star)  
$$

### 第三步：policy switching 决定是否执行引导动作

$$  
a=  
\begin{cases}  
a_{\mathrm{guide}}, & \zeta<\beta\\  
\pi_{\mathrm{on}}(s), & \text{otherwise}  
\end{cases}  
$$

因此，离线知识不是直接初始化 online actor，而是作为 **探索方向和动作候选生成器** 使用。

## 5.4 本文是否讨论离线知识会阻碍 online 学习？

有讨论，但主要是经验性和机制性的。

### 1. 固定较大的 \(\beta\) 可能伤害 online 学习

如果长期使用较高的 offline guidance 概率：

- IDM 预测误差会被频繁执行；
- online agent 难以充分探索和利用自己学到的策略；
- offline state policy 可能将 agent 限制在旧行为模式中。

附录 D.3 表明，固定 \(\beta=0.8\) 在 HalfCheetah 和 Walker2D 上性能明显较差。

### 2. 无正则的离线 state policy 会产生不可达状态预测

去掉保守正则化后，DecQN\(_N\) 的预测误差接近随机策略，说明离线 critic 可能选择数据支持之外、实际不可达的状态变化。

### 3. \(\beta\) 退火机制本质上承认离线知识只在早期最有用

作者将 \(\beta\) 从 0.5 逐步降到 0，说明：

> 离线知识主要用于 online agent 尚弱时的引导探索；当 online agent 变强后，应减少对离线知识的依赖。

## 5.5 不足之处

本文没有系统讨论：

- 如何识别哪些离线知识是有害的；
- 如何自动选择需要保留和遗忘的状态维度；
- 如何处理 offline prior 与 online optimal policy 冲突；
- 是否应根据 state policy 的置信度动态调整 \(\beta\)；
- offline 数据质量较差时如何主动降权或屏蔽 guidance。

因此，本文对问题 5 的处理是 **有设计和实验体现，但没有形成完整的知识选择与遗忘理论**。

---

# 6. 本文算法是否适合迁移到 MARL？为什么？

## 6.1 总体判断

**思想适合迁移到 MARL，但不能直接照搬，尤其在标准 CTDE 和去中心化执行设置下需要重大修改。**

如果假设存在一个集中控制器，可以直接控制所有智能体的联合动作，那么迁移相对自然；如果要求每个智能体去中心化执行，则困难会明显增加。

## 6.2 为什么适合迁移到 MARL？

### 1. Action-free 数据在 MARL 中也可能很常见

例如：

- 多机器人只记录轨迹，不记录控制指令；
- 金融市场中只能观察市场状态，不能观察每个机构的具体交易；
- 交通系统中只记录全局流量变化，而没有记录每个参与者的动作。

因此，从 \((s,r,s')\) 学习团队状态转移价值具有现实意义。

### 2. 学习“期望联合状态变化”比直接恢复联合动作更现实

MARL 中联合动作是：

$$  
\mathbf a=(a^1,\ldots,a^n)  
$$

如果直接学联合动作策略，动作空间很大；而学习联合状态变化：

$$  
\Delta \mathbf s  
$$

可能更抽象、更稳定。

### 3. Value decomposition 与 MARL 天然相关

DecQN 本身就受到 MARL value decomposition 思想启发。OSO-DecQN 的状态维度分解也可以看成一种简化的分解式 critic：

$$
Q_{\mathrm{tot}}(\mathbf s,\Delta\mathbf s)
=
\frac1M\sum_{j=1}^M U^j(s,\Delta s_j)
$$

这与 MARL 中的 additive value factorization 有相似之处。

### 4. \(\beta\) 退火适合非平稳在线学习

MARL 中离线先验往往只在早期有用。随队友策略更新，离线知识可能逐渐过时，因此从 offline guidance 向 online autonomy 过渡是合理的。

---

## 6.3 迁移到 MARL 的主要挑战

### 挑战一：联合动作 IDM 更加困难

单智能体 IDM 学习：

$$  
I_\phi(s,\Delta s)\to a  
$$

MARL 中则必须学习：

$$  
I_\phi(\mathbf s,\Delta\mathbf s)  
\to  
(a^1,\ldots,a^n)  
$$

同一全局状态变化可能由许多不同联合动作实现，逆映射更不稳定、更多模态。

### 挑战二：状态维度分解可能无法表达复杂协同

如果某个状态维度变化需要多个智能体协调完成，简单地将全局 \(Q\) 分解为各状态维度 utility 的平均值可能无法表达高阶交互：

$$  
Q_{\mathrm{tot}}  
\neq  
\frac1M\sum_j U^j(s,\Delta s_j)  
$$

可能需要：

- monotonic mixing；
- attention mixing；
- graph factorization；
- 联合状态子空间分解。

### 挑战三：集中式 \(\arg\max_{\Delta\mathbf s}\) 与去中心化执行冲突

离线 critic 选择全局状态变化：

$$
\Delta\mathbf s^\star
=
\arg\max_{\Delta\mathbf s}
Q_{\mathrm{tot}}(\mathbf s,\Delta\mathbf s)
$$

这需要全局状态、全局 critic 和全局通信。标准 MARL 执行阶段往往只有局部观测 \(\tau^i\)，不能直接执行这一全局选择规则。

### 挑战四：非平稳性更严重

单智能体中，offline state policy 的状态转移规律相对固定；MARL 中其他智能体的策略不断变化，因此离线学到的：

$$  
Q_{\mathrm{off}}(\mathbf s,\Delta\mathbf s)  
$$

可能很快过时。

### 挑战五：联合动作耦合会导致独立 IDM 失效

如果每个智能体独立学习：

$$  
I_{\phi_i}(s^i,\Delta s^i)\to a^i  
$$

则无法处理：

- 动作互补；
- 动作冲突；
- 时序协同；
- 通信依赖。

因此，MARL 中很可能需要联合 IDM 或带通信机制的 IDM。

---

# 7. 与问题 5 关联：这些问题在 MARL 中是否不同？是否更有挑战性？

## 7.1 结论

**明显不同，而且在 MARL 中更具挑战性。**

在单智能体中，本文主要面对的问题是：

- 离线状态策略是否可执行；
- IDM 是否准确；
- \(\beta\) 是否合适；
- 离线状态变化是否可达。

在 MARL 中，还要额外面对：

- 离线知识基于旧队友策略；
- 全局状态变化与个体动作之间缺少一一对应；
- 联合动作空间耦合；
- 去中心化执行约束；
- 团队级 credit assignment。

## 7.2 Offline 阶段在 MARL 中可能保留的知识

|离线知识|可能用途|风险|
|---|---|---|
|全局状态变化价值 \(Q(\mathbf s,\Delta\mathbf s)\)|指导团队朝高价值状态转移|队友策略变化后失效|
|个体状态变化价值 \(Q^i(\tau^i,\Delta s^i)\)|支持去中心化执行|难以表达协同|
|联合轨迹模式|保留团队协调行为|可能锁死旧均衡|
|团队逆动力学先验|将状态变化转成联合动作|action-free 数据中难以学习|
|通信或角色分配模式|加速协调|旧通信机制可能不适应在线阶段|

## 7.3 Online 阶段如何使用

可能的使用方式包括：

1. 集中式状态变化规划：

$$
\Delta\mathbf s^\star
=
\arg\max_{\Delta\mathbf s}
Q_{\mathrm{tot}}(\mathbf s,\Delta\mathbf s)
$$

1. 联合 IDM 翻译：

$$
\mathbf a
=
I_\phi(\mathbf s,\Delta\mathbf s^\star)
$$

1. 个体状态变化引导：

$$
\Delta s^{i,\star}
=
\arg\max_{\Delta s^i}
Q^i(\tau^i,\Delta s^i)
$$

1. 团队级 \(\beta\) 或个体级 \(\beta_i\) 退火。

## 7.4 MARL 中 offline 知识阻碍 online 的方式更严重

### 1. 旧协同均衡阻碍新均衡

离线数据中的联合行为可能是旧队友策略下的局部最优。在线阶段如果继续强烈遵循该先验，团队可能无法切换到更优协调模式。

### 2. 非平稳性导致离线 critic 快速过时

离线阶段学到的价值函数基于：

$$  
\pi_{\mathrm{off}}^{-i}  
$$

但 online 阶段实际队友策略是：

$$  
\pi_{\mathrm{on}}^{-i}  
$$

当：

$$  
\pi_{\mathrm{off}}^{-i}  
\neq  
\pi_{\mathrm{on}}^{-i}  
$$

时，离线状态转移价值也会失真。

### 3. 联合动作使“状态变化正确”不等于“动作正确”

即使 \(\Delta\mathbf s^\star\) 是合理的，联合 IDM 也可能输出：

- 某个智能体动作正确、其他智能体动作错误；
- 个体动作都合理、组合后却不协调；
- 短期能实现状态变化、长期却破坏团队结构。

因此，MARL 中需要的不只是 \(\beta\) 退火，还可能需要：

- 队友变化检测；
- 离线知识置信度估计；
- 联合动作不确定性惩罚；
- 动态知识降权；
- 通信一致性约束；
- 团队安全约束。

---

# 8. 本文是否仍然存在不足？结合前文，特别是迁移到 MARL 时有哪些不足？

## 8.1 作者自己承认或明确指出的不足

### 1. 依赖局部平滑且可逆的动力学关系

IDM 有效的前提是：

$$  
(s,\Delta s)\rightarrow a  
$$

存在较稳定的局部映射。

如果环境具有：

- 强不连续动力学；
- 同一状态变化对应多个完全不同的动作；
- 高度多模态逆映射；

则简单 IDM 可能失效。

### 2. 三值状态离散化会损失信息

\(\{-1,0,1\}\) 离散化虽然稳定，但无法表达：

- 变化幅度；
- 变化速度；
- 多步时序关系；
- 状态变化之间的复杂相关。

论文也在未来工作中提到可使用 adaptive discretisation。

### 3. 依赖奖励信号

本文所谓 action-free 仍然要求离线数据包含：

$$  
(s,r,s')  
$$

如果数据连 reward 也没有，例如普通视频，那么方法仍不能直接使用。作者也指出从视觉数据提取可靠 reward 仍是未来挑战。

### 4. IDM 需要在线从零训练

在线初期 IDM 不准确，而 \(\beta\) 较大时这些错误会被频繁执行，可能产生较差动作。

### 5. 对 \(\beta\) 和正则系数较敏感

固定较大的 \(\beta\) 会伤害性能，而不同数据集使用不同正则权重 \(\alpha\)，说明方法仍需要一定任务调参。

## 8.2 从理论角度看到的不足

### 1. 理论只覆盖离散化误差

没有证明：

- OSO-DecQN 收敛；
- 正则化后的 critic 一定优于无正则；
- IDM 学习一定稳定；
- policy switching 一定提升在线回报。

### 2. 定理假设较强

例如高斯状态增量、协方差与动作无关等假设在很多真实机器人或金融环境中未必成立。

### 3. 完整上界存在不随 \(k\) 减小的常数项

因此，主文中的：

$$  
O\left(\frac{H\sqrt M}{k}\right)  
$$

更适合作为固定其他量时的缩放表述，而不是严格的渐近一致性结论。

## 8.3 迁移到 MARL 时的额外不足

1. **全局状态需求**
    - 难以适用于 partial observability。
2. **联合 IDM 难度高**
    - 输出维度为 \(\sum_i d_i\)，且逆映射更复杂。
3. **状态维度分解无法保证表达协同**
    - 简单 additive decomposition 可能低估联合状态变化的交互价值。
4. **离线知识更容易过时**
    - 非平稳性比单智能体更严重。
5. **联合动作耦合**
    - 不能将“每个智能体独立完成自己的状态变化”简单等同于好的团队动作。
6. **集中式规划与去中心化执行冲突**
    - 若执行阶段不能使用全局 critic，则必须重新设计可执行的状态 policy。

---

# 9. 本文的对照算法、发表信息、实验环境和消融实验

## 9.1 主实验中的对照方法

|方法|发表年份与场所|在本文中的角色|
|---|---|---|
|TD3|ICML 2018|连续动作在线基线；测试 OSO-DecQN 是否能加速 TD3|
|DecQN\(_N\)|DecQN 原论文 arXiv 2022 / ICLR 2023；ensemble 版本与 Revalued ICLR 2024 密切相关|动作因子化在线基线|
|AF-Guide|arXiv 2023|最直接的 action-free state policy 对照方法|
|SAC|NeurIPS 2019|附录中验证方法可迁移到另一个 online agent|
|BC\(^a\)|经典行为克隆方法|带动作标签的 imitation 参考|
|BC\(^{s'}\)、BC\(^{s'-s}\)、BC\(^{\Delta s}\)|本文构造的分析基线|验证连续状态预测与离散状态预测差异|
|DecQN\(_N\) w/o regularization|本文消融版本|验证保守正则化的重要性|
|O3F、IQL、Cal-QL、AWAC|见下文|附录中带动作标签的强 O2O 对照|

其中 AF-Guide 是 2023 年 arXiv preprint，未见正式会议版本记录 ；DecQN 原始论文正式发表于 ICLR 2023 ；Revalued 发表于 ICLR 2024 。

## 9.2 附录中带动作标签的 O2O 对照

|方法|发表年份与场所|目的|
|---|---|---|
|O3F|NeurIPS 2022 Deep RL Workshop|用动作标签进行 optimistic offline-to-online fine-tuning 的强基线|
|IQL|ICLR 2022|经典离线 value regularization 方法，在线继续微调|
|Cal-QL|NeurIPS 2023|校准离线 Q 初始化以支持在线微调|
|AWAC|arXiv 2020|使用离线数据加速在线 actor-critic 微调|

公开记录显示 O3F 为 NeurIPS 2022 Deep RL Workshop ；IQL 为 ICLR 2022 ；Cal-QL 为 NeurIPS 2023 ；AWAC 最初为 arXiv 2020 。

需要注意：上传论文附录 G 中写作 “AWAC (Luo et al., 2023)”，但 AWAC 本身通常是 Nair et al. 于 2020 年提出的方法，论文这里可能存在引用错位。

## 9.3 实验环境

### 1. D4RL MuJoCo 连续控制任务

- HalfCheetah
- Walker2D
- Hopper

数据集质量包括：

- medium-replay
- medium
- medium-expert
- expert

### 2. Action-factorised DeepMind Suite

- Quadruped-Walk
- Cheetah-Run

其中 Quadruped-Walk 的状态维度达到 78，是论文用于验证高维可扩展性的核心环境。

### 3. 数据集规模

论文使用的离线数据集包括约：

- Hopper medium-replay：402k
- HalfCheetah medium-replay：202k
- Walker2D medium-replay：302k
- medium：1M
- medium-expert：2M
- expert：1M

Action-factorised 数据集：

- random-medium-expert：200k
- medium：1M
- medium-expert：2M
- expert：1M

所有动作信息在预训练前被移除，只保留：

$$  
(s,r,s')  
$$

## 9.4 主实验内容

### 实验一：D4RL + TD3

用不同质量的离线 action-free 数据预训练 OSO-DecQN，再引导 TD3 在线学习。

结果：

- 多数任务收敛更快；
- 渐近性能更高；
- Hopper 改进最小，因为该任务较简单，TD3 本身学习已经很快。

### 实验二：DeepMind Suite + DecQN\(_N\)

用 OSO-DecQN 引导动作因子化在线 agent。

结果：

- 在 Cheetah-Run 和 Quadruped-Walk 上均有提升；
- Quadruped-Walk 的 78 维状态空间证明了可扩展性。

### 实验三：与 AF-Guide 比较

在 HalfCheetah、Walker2D、Hopper 上，AF-Guide 多数情况下低于 TD3 基线，而本文方法能够稳定提升在线 agent。

### 实验四：与带动作标签 O2O 方法比较

在 Walker2D 上与 O3F、IQL、Cal-QL、AWAC 比较。

由于这些基线可以使用动作标签训练离线 actor，因此在线开始时性能更高；但 action-free 的 OSO-DecQN 在 250k online steps 内可以追赶或接近这些方法，说明其虽然信息更少，但知识利用效率较高。

### 实验五：SAC 迁移实验

将同样 guidance 机制应用到 SAC，无需重新精细调参，仍能提升收敛速度和最终性能。

---

## 9.5 分析实验与问题验证实验

### 分析一：为什么需要离散化？

比较：

- BC\(^{s'}\)：模仿连续下一状态；
- BC\(^{s'-s}\)：模仿连续状态差分；
- BC\(^{\Delta s}\)：模仿离散状态差分；
- BC\(^a\)：带动作标签的行为克隆。

结果显示：

- 连续 \(s'\) 和 \(s'-s\) 预测很差；
- \(\Delta s\) 离散目标可达到接近带动作 BC 的效果；
- 离散状态差分预测误差显著低于连续目标。

### 分析二：为什么需要正则化？

比较：

- OSO-DecQN；
- 去掉正则化的 DecQN\(_N\)。

结果显示：

- 无正则版本的预测误差接近随机策略；
- 说明其会选择不可达状态变化；
- OSO-DecQN 能同时改善可达性和长期回报。

### 分析三：state policy 是否真的能超过离线数据本身？

Table 1 显示 OSO-DecQN 在许多数据集上的返回高于 BC\(^a\) 或行为数据参考，说明它不是简单模仿数据，而是通过 Q-learning 学到了超越数据集平均表现的状态转移建议。

### 分析四：IDM 是否只是简单翻译器？

论文通过：

- IDM loss 曲线；
- 在线 IDM 与 expert IDM 比较；
- 不同网络层数和 batch size 消融；
- IDM loss 与最终回报的弱相关性；

证明在测试环境中，轻量 IDM 已足够，性能主要由 state policy 的质量决定。

---

## 9.6 消融实验

### 1. 离散化阈值 \(\epsilon\)

测试：

$$  
\epsilon\in\{10^{-3},10^{-4},10^{-5},10^{-6}\}  
$$

结果说明方法对较大范围 \(\epsilon\) 较鲁棒。

### 2. 二值离散与三值离散

比较：

- 2 bins：增加 / 减少；
- 3 bins：增加 / 不变 / 减少。

结果表明：

- 2 bins 也能学习；
- 3 bins 通常更稳定；
- 2 bins 会把数值噪声错误解释成状态变化。

### 3. \(\beta\) 固定值与退火

比较：

- \(\beta=0.2\)
- \(\beta=0.5\)
- \(\beta=0.8\)
- 线性退火

结果显示：

- 固定过大 \(\beta\) 会伤害 HalfCheetah 和 Walker2D；
- \(\beta\) 退火通常兼顾早期探索与后期利用；
- 说明 offline guidance 应逐渐减少。

### 4. IDM 架构和 batch size

测试：

- 2、3、4 层网络；
- batch size 384、512、768。

结果变化不大，说明简单 IDM 在这些环境中不是性能瓶颈。

### 5. IDM offline bootstrapping

比较完整损失和去掉离线伪动作 bootstrapping 的版本。

结果：

- 完整方法平均 return：115.8
- 去掉 bootstrapping：112.7

说明离线状态转移对 IDM 训练有额外帮助。

---

# 10. 本文的核心 claim 及证明情况

|核心 claim|证明方式|证明强度|
|---|---|---|
|即使离线数据没有动作，也可以提取对 online RL 有用的知识|主实验 Fig. 1；AF-Guide 对比；Walker2D O2O 对比|强实验支持|
|离散状态差分比连续状态预测更稳定、更准确|BC\(^{s'}\)、BC\(^{s'-s}\)、BC\(^{\Delta s}\) 分析；离散预测误差表；\(\epsilon\) 和 bin 消融|强实验支持|
|保守正则化对抑制 OOD 状态变化和 overestimation 很重要|有/无正则 DecQN\(_N\) 对比；预测误差分析|强消融支持|
|OSO-DecQN 学到的不只是行为克隆，而能超过数据集经验性能|Table 1 中 OSO-DecQN 高于 BC\(^a\) 的多个结果|较强实验支持|
|状态策略可以引导不同 online agent|TD3 主实验、DecQN\(_N\) 实验、SAC 附录实验|较强实验支持|
|policy switching 与 \(\beta\) 退火比固定 guidance 更稳定|附录 D.3 的 \(\beta\) 敏感性实验|有专门消融支持|
|轻量 IDM 已足以翻译状态变化|IDM 架构、batch size、loss 动态分析|有实验分析支持|
|方法可扩展到高维状态空间|value decomposition；Quadruped-Walk 78 维实验|有实验支持|
|离散化带来的价值误差随 bin 数增加而减小|Theorem 1 / Appendix B|有理论界，但假设较强，且不证明完整算法收敛|
|在线 guidance 一定优于普通 online RL|多数实验支持|并非所有任务都显著提升，例如 Hopper 提升很小|
|方法适用于任意复杂环境|未证明|论文明确承认在不连续或多模态逆动力学下可能失败|

## 最终评价

本文最有力的核心贡献是：

1. **把 action-free offline RL 从“无法直接使用标准 RL”转化为可学习的离散状态变化 Q-learning 问题。**
2. **用保守正则化解决 state reachability 与 overestimation。**
3. **用 IDM + policy switching 将不可执行的 state policy 转化为 online guidance。**
4. **通过大量主实验、分析实验和消融证明该方法比 AF-Guide 更稳定、更可扩展。**

但它的理论结果仅覆盖离散化误差，并不能证明整个 OSO-DecQN + IDM + policy switching 框架的收敛性。此外，在 MARL 中，联合动作、非平稳性、去中心化执行和团队逆动力学问题都会使其直接迁移变得困难。