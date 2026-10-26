# 文章主要内容总结

这篇文章提出了 **Pre-Reduced Q-Learning（PRQL）**，用于解决 Offline-to-Online Reinforcement Learning（O2O RL）中的**在线微调初期策略崩溃（policy collapse）**问题。

其核心思想可以概括为：

> **在 Online 微调之前，先主动“修剪”离线 Critic 对 OOD 动作的错误高估；在 Online 微调期间，通过记忆正则保留离线阶段学到的有效价值知识，同时不再使用行为克隆约束。**

整体流程为：

1. 用 TD3-BC 在离线数据集上预训练策略和 Critic；
2. 训练一个中间策略，在离线 Critic 上寻找 Q 值被高估的动作；
3. 训练中间 Critic，降低这些动作的 Q 值，同时通过 TD 损失维持数据分布内动作的 Q 值；
4. 用中间 Critic 和离线策略初始化 Online Critic 与 Online Actor；
5. Online 阶段只使用新采集的数据，并通过记忆正则防止遗忘离线阶段学到的价值知识。

---

## 1. 文章解决 O2O RL 中的什么问题？为什么现有算法没有很好解决？

### 1.1 解决的问题

文章解决的是 O2O RL 中的两个相互矛盾的问题：

1. **保守微调导致学习过慢**
2. **放开约束进行自由探索又导致 Online 初期策略崩溃**

具体来说：

- 如果继续使用 TD3-BC、BCQ、CQL 等离线算法进行微调，其行为克隆或悲观约束会被 Online 阶段继承，策略仍然局限于离线数据分布，难以快速利用在线数据提升。
- 如果直接换成 TD3、SAC 等普通 Online RL 算法，虽然探索能力更强，但离线 Critic 往往会**高估 OOD 动作**。在线微调一开始，Actor 就会优先选择这些“错误高价值”的动作，导致性能突然大幅下降，即文章所称的 policy collapse。

### 1.2 产生 policy collapse 的机制

TD3 的目标 Q 值可以写成：

$$
y=r+\gamma \min_{i=1,2}Q_{\bar{\phi}^{i}}
\left(s',\bar{\pi}_{\theta}(s')+\epsilon\right).
$$

如果离线 Critic 对某个 OOD 动作 $\bar{\pi}_{\theta}(s')$ 给出了过高估计，那么：

$$
Q_{\bar{\phi}^{i}}(s',a'_{\mathrm{OOD}})
\gg
Q_{\mathrm{true}}(s',a'_{\mathrm{OOD}}).
$$

这会使目标值 $y$ 被错误放大，进一步导致当前状态动作 $(s,a)$ 的 Q 值被错误提高。这个误差会沿 Bellman backup 传播和累积，最终使 Actor 错误地偏向 OOD 动作。

文章将 Online 初期的崩溃解释为：

> Agent 在在线交互中不断“试错”，逐渐把被高估的 OOD Q 值修正回来；但修正期间的性能损失非常大。

### 1.3 为什么现有算法没有彻底解决？

现有方法主要有两类。

#### 第一类：继续保留离线约束

例如继续使用 TD3-BC、自适应 BC、CQL 等。

问题是：

- 离线约束和 Online 探索所需的最优约束强度不同；
- 最优约束强度随任务、数据集质量和训练阶段变化；
- 约束过强会继承保守性，导致策略提升缓慢。

#### 第二类：混合离线数据与在线数据

例如 balanced replay、离线—在线混合 replay buffer。

问题是：

- 大量旧离线数据会降低在线新数据的利用率；
- 旧数据会稀释 Online 学习信号；
- 仍可能无法完全解决 Critic 对 OOD 动作的高估。

因此，作者认为现有方法的核心不足是：  
**要么在 Online 阶段持续施加约束，牺牲探索；要么完全放开约束，继承离线 Critic 的错误价值估计。**

PRQL 的思路则是提前修正价值地形，使 Online 阶段可以“无约束但不崩溃”。

---

## 2. 本文的核心创新点是什么？

### 创新点一：微调前预降低 OOD 动作的 Q 值

作者不是直接在 Online 阶段限制 Actor，而是在微调前训练：

- 一个中间策略 $\pi_m$；
- 一个中间 Q 网络 $Q_m$。

其中：

- $\pi_m$ 专门寻找当前 Critic 认为价值最高的动作；
- 这些动作通常就是离线 Critic 最容易高估的 OOD 动作；
- $Q_m$ 通过特殊损失降低这些动作的 Q 值。

这相当于在 Online 开始前，对离线 Critic 做一次“价值地形修剪”。

### 创新点二：选择性降低，而不是全局悲观化

传统悲观方法通常统一压低 OOD 区域 Q 值，可能把有用的数据分布内动作也压低。

PRQL 的损失同时包含：

$$
\eta_1 Q_m^i(s,\pi_m(s))
$$

和离线 TD 误差。前者降低中间策略动作的价值，后者保证离线数据动作的 Q 值仍然准确。

因此，它更接近：

> **只降低最可能被 Actor 错误选择的高估动作，而不是对所有未知区域施加统一悲观惩罚。**

### 创新点三：Online 阶段仅使用在线数据，并加入记忆正则

为了提升在线样本效率，PRQL 在 Online 阶段不使用旧离线 transition，只使用新采集的在线数据。

但这样可能造成 Critic 遗忘离线数据上的有效价值，因此作者加入：

$$
L_Q^{\mathrm{REGU}}(\phi_{\mathrm{on}}^i)
=
\mathbb{E}_{s\sim D_{\mathrm{on}}}
\left[
\left(
Q_{\mathrm{on}}^i(s,\pi_{\mathrm{off}}(s))
-
Q_{\mathrm{off}}^i(s,\pi_{\mathrm{off}}(s))
\right)^2
\right].
$$

它要求 Online Critic 在离线策略动作附近，仍然记住 Offline Critic 学好的价值。

### 创新点四：Online Actor 可以完全去除 BC 约束

由于 OOD 高估已经提前降低，Online 阶段可以直接使用：

$$
L_{\pi}(\theta_{\mathrm{on}})
=
\mathbb{E}_{s\sim D_{\mathrm{on}}}
\left[
-\lambda Q_{\mathrm{on}}^1(s,\pi_{\mathrm{on}}(s))
\right],
$$

而不需要行为克隆项：

$$
(\pi_{\mathrm{on}}(s)-a)^2.
$$

因此，Online 策略可以更自由地探索和提升。

---

## 3. 方法有哪些核心模块？完整算法流程是什么？

## 3.1 模块一：Offline 预训练

首先使用 TD3-BC 在离线数据集 $D_{\mathrm{off}}$ 上训练：

- 离线 Actor：$\pi_{\mathrm{off}}$；
- 双 Critic：$Q_{\mathrm{off}}^1,Q_{\mathrm{off}}^2$。

TD3-BC 的 Actor 损失为：

$$
L_{\pi}^{\mathrm{TD3-BC}}(\theta_{\mathrm{off}})
=
\mathbb{E}_{(s,a)\sim D_{\mathrm{off}}}
\left[
-\lambda Q_{\mathrm{off}}^1(s,\pi_{\mathrm{off}}(s))
+
(\pi_{\mathrm{off}}(s)-a)^2
\right].
$$

Critic 损失为：

$$
L_Q^{\mathrm{TD3-BC}}(\phi_{\mathrm{off}}^i)
=
\mathbb{E}_{(s,a,r,s')\sim D_{\mathrm{off}}}
\left[
\left(
r+
\gamma\min_{j=1,2}
\bar Q_{\mathrm{off}}^j
(s',\bar\pi_{\mathrm{off}}(s')+\epsilon)
-
Q_{\mathrm{off}}^i(s,a)
\right)^2
\right].
$$

其中 $\bar Q$、$\bar\pi$ 分别表示目标 Critic 和目标 Actor。

---

## 3.2 模块二：中间策略搜索高估动作

随机初始化一个中间策略 $\pi_m$，并令：

$$
\phi_m^1\leftarrow\phi_{\mathrm{off}}^1,\qquad
\phi_m^2\leftarrow\phi_{\mathrm{off}}^2.
$$

中间策略通过最大化 $Q_m^1$ 来寻找当前价值估计最高的动作：

$$
L_{\pi}(\theta_m)
=
\mathbb{E}_{s\sim D_{\mathrm{off}}}
\left[
-Q_m^1(s,\pi_m(s))
\right].
$$

直观地说，$\pi_m$ 并不是为了最终部署，而是为了“攻击”当前 Critic，找出它最可能高估的动作。

---

## 3.3 模块三：中间 Critic 预降低 Q 值

中间 Critic 的损失为：

$$
\begin{aligned}
L_Q(\phi_m^i)
=
\mathbb{E}_{(s,a,r,s')\sim D_{\mathrm{off}}}
\Bigg[
&
\eta_1 Q_m^i(s,\pi_m(s))\\
&+
\left(
r+
\gamma\min_{j=1,2}
\bar Q_{\mathrm{off}}^j
(s',\bar\pi_{\mathrm{off}}(s')+\epsilon)
-
Q_m^i(s,a)
\right)^2
\Bigg].
\end{aligned}
$$

该损失有两个作用：

1. 第一项  
   $$
   \eta_1 Q_m^i(s,\pi_m(s))
   $$
   使 $\pi_m(s)$ 对应动作的 Q 值下降；

2. 第二项是离线 TD 误差，使 $Q_m$ 仍能准确拟合离线数据中的 $(s,a,r,s')$。

因此，$Q_m$ 大致实现了：

- 对最可能被错误选择的高价值动作进行降值；
- 对离线数据分布内的动作维持价值精度。

需要注意，这并不是把 OOD Q 值校准到真实值，而只是让 OOD 峰值不再高于有效动作。

---

## 3.4 模块四：Online Critic 与记忆正则

微调开始时：

$$
\phi_{\mathrm{on}}^i\leftarrow\phi_m^i,\qquad
\theta_{\mathrm{on}}\leftarrow\theta_{\mathrm{off}}.
$$

Online Critic 的总损失为：

$$
\begin{aligned}
L_Q(\phi_{\mathrm{on}}^i)
=
&
\eta_2 L_Q^{\mathrm{REGU}}(\phi_{\mathrm{on}}^i)\\
&+
\mathbb{E}_{(s,a,r,s')\sim D_{\mathrm{on}}}
\left[
\left(
r+
\gamma\min_{j=1,2}
\bar Q_{\mathrm{on}}^j
(s',\bar\pi_{\mathrm{on}}(s')+\epsilon)
-
Q_{\mathrm{on}}^i(s,a)
\right)^2
\right].
\end{aligned}
$$

其中第一项保留离线知识，第二项学习在线新数据。

---

## 3.5 模块五：无 BC 的 Online Actor

Online Actor 的损失为：

$$
L_{\pi}(\theta_{\mathrm{on}})
=
\mathbb{E}_{s\sim D_{\mathrm{on}}}
\left[
-\lambda Q_{\mathrm{on}}^1(s,\pi_{\mathrm{on}}(s))
\right].
$$

与普通 TD3-BC 相比，它去掉了行为克隆项，因此探索不再被显式限制在离线行为分布附近。

---

## 3.6 完整算法流程

### 阶段一：Offline 预训练

1. 从 $D_{\mathrm{off}}$ 采样 batch；
2. 用 TD3-BC 更新 $Q_{\mathrm{off}}^1,Q_{\mathrm{off}}^2,\pi_{\mathrm{off}}$；
3. 训练 $0.5\mathrm{M}$ 步。

### 阶段二：微调前 Q 值预降低

1. 初始化：
   $$
   \phi_m\leftarrow\phi_{\mathrm{off}};
   $$
2. 随机初始化 $\pi_m$；
3. 重复 $0.2\mathrm{M}$ 步：
   - 从 $D_{\mathrm{off}}$ 采样；
   - 用
     $$
     L_{\pi}(\theta_m)=
     -\mathbb E[Q_m^1(s,\pi_m(s))]
     $$
     更新 $\pi_m$；
   - 用包含“降值损失 + TD 损失”的目标更新 $Q_m$。

### 阶段三：Online Fine-tuning

1. 初始化：
   $$
   Q_{\mathrm{on}}\leftarrow Q_m,\qquad
   \pi_{\mathrm{on}}\leftarrow\pi_{\mathrm{off}};
   $$
2. 在线环境交互，得到 $(s,a,r,s')$，存入 $D_{\mathrm{on}}$；
3. 从 $D_{\mathrm{on}}$ 采样 batch；
4. 用“TD 损失 + 记忆正则”更新 $Q_{\mathrm{on}}$；
5. 用无 BC 的 Actor 损失更新 $\pi_{\mathrm{on}}$；
6. 更新 TD3 目标网络；
7. 训练 $0.5\mathrm{M}$ 步。

---

## 4. 本文是否进行了理论证明？

没有。

文章没有给出：

- 收敛性证明；
- Bellman backup 误差上界；
- policy collapse 不出现或概率降低的证明；
- Q 预降低误差界；
- 记忆正则条件下的性能保证；
- 样本复杂度界。

本文的分析主要是第 4.1 节中的**定性机制分析**：作者从 TD 目标和 bootstrap error 的传播角度解释为什么 OOD 高估会导致 policy collapse，但该分析不构成严格理论证明。

因此，本文结论主要依赖实验验证，而非理论保证。

---

## 5. 本文是否讨论了 Offline 阶段哪些知识值得保留、Online 阶段如何使用，以及 Offline 知识是否会阻碍 Online？

## 5.1 有讨论，但不是以系统的“知识选择”框架讨论

本文对 Offline 知识的处理可以总结如下。

| Offline 知识 | PRQL 的处理方式 | Online 阶段如何使用 | 可能造成的阻碍 |
|---|---|---|---|
| 离线策略参数 $\pi_{\mathrm{off}}$ | 保留并初始化 Online Actor | 作为微调起点，并作为记忆正则中的 anchor action | 可能继承保守行为 |
| 数据分布内动作的 Q 值 | 通过 $Q_m$ 的 TD 项和记忆正则保留 | 用于 Online bootstrap、策略提升和稳定微调 | 若离线数据质量差，保留错误价值 |
| OOD 动作的 Q 值 | 主动降低 | 避免 Actor 在微调初期选择这些动作 | 原值本身会阻碍 Online 学习 |
| 行为克隆约束 | Online 阶段移除 | 不再限制 Actor 靠近离线动作 | 离线阶段需要它，但 Online 阶段会限制探索 |
| 离线 transition 数据 | 不 replay，不保存在 Online buffer | 其价值被压缩进 Critic 参数中 | 丢弃了部分可直接利用的数据 |

### 明确保留的知识

本文明确认为值得保留的是：

$$
Q_{\mathrm{off}}^i(s,\pi_{\mathrm{off}}(s)),
$$

即离线策略对应动作上的价值。

这通过记忆正则实现：

$$
Q_{\mathrm{on}}^i(s,\pi_{\mathrm{off}}(s))
\approx
Q_{\mathrm{off}}^i(s,\pi_{\mathrm{off}}(s)).
$$

### 明确不保留或主动削弱的知识

本文认为 Offline Critic 对 OOD 动作的高估是危险知识，因此不保留，而是通过 $\pi_m$ 和 $Q_m$ 提前削弱。

### Offline 知识阻碍 Online 的情况

文章主要指出了两类阻碍：

1. **OOD Q 值高估**  
   会导致 Online Actor 一开始就选择错误动作。

2. **Offline 策略和 BC 约束过强**  
   会限制 Online 探索，使策略提升缓慢。

此外，从方法设计还能看出一个隐含观点：

3. **低质量离线价值不应该过度保留**  
   例如 random 数据集几乎没有可靠价值，因此作者设置：
   $$
   \eta_2=0.
   $$

但文章没有系统讨论更一般的知识选择问题，例如：

- 哪些离线状态值得保留；
- 哪些行为模式应该保留；
- 在线状态分布偏移后，旧的 $Q(s,\pi_{\mathrm{off}}(s))$ 是否仍有效；
- 当任务目标或动力学发生变化时，应保留多少旧知识。

---

## 6. 本文算法是否适合迁移到 MARL？

## 6.1 结论

**PRQL 的核心思想适合迁移到 MARL，但原算法不能直接迁移。**

适合迁移的是：

- 微调前识别并降低危险高估动作价值；
- Online 阶段保留有价值的离线价值知识；
- 用参数级记忆代替离线数据 replay。

不能直接迁移的是：

- 基于单智能体确定性 Actor 的 TD3-BC 框架；
- 单动作 OOD 的定义；
- 单一 Critic 和单一状态假设；
- 固定环境动力学假设；
- 无联合动作和 teammate policy 变化问题。

---

## 6.2 为什么原算法不能直接迁移到 MARL？

### 原因一：OOD 从单动作变为联合动作

单智能体中，OOD 主要是：

$$
a\notin \operatorname{supp}\beta(\cdot\mid s).
$$

MARL 中真正关键的是：

$$
\mathbf a=(a_1,\ldots,a_N)
\notin
\operatorname{supp}\beta(\cdot\mid\mathbf o),
$$

其中 $\mathbf a$ 是联合动作。

即使每个智能体的动作 $a_i$ 单独看都在其边缘行为分布内，它们的联合组合也可能是严重 OOD。联合动作空间通常随智能体数量指数增长，因此 OOD 问题更严重。

### 原因二：非平稳性

单智能体 Online 微调时，环境转移动力学通常固定：

$$
P(s'|s,a).
$$

MARL 中，从单个智能体视角看，其他智能体策略不断变化，因此有效转移为：

$$
P_{\boldsymbol{\pi}_{-i}}(s'|s,a_i).
$$

当 teammate policy 更新时，同一个 $a_i$ 的后果也会变化。Offline Critic 学到的价值：

$$
Q(s,\mathbf a)
$$

可能依赖于旧的联合策略，Online 阶段未必继续成立。

### 原因三：联合 Q 的高估不能直接归因到单个智能体

单智能体 PRQL 可以降低：

$$
Q(s,\pi_m(s)).
$$

MARL 中需要降低的是：

$$
Q_{\mathrm{tot}}(\mathbf o,\mathbf a_m).
$$

如果联合 Q 被高估，很难知道是：

- 哪个智能体动作导致的；
- 是单智能体错误；
- 还是多个智能体组合不协调；
- 还是 centralized Critic 对联合动作整体外推错误。

若简单降低联合 Q，可能破坏 QMIX、QPLEX 等方法需要的值分解结构或 IGM 条件。

### 原因四：记忆正则会锚定旧联合策略

单智能体中，记忆项为：

$$
Q(s,\pi_{\mathrm{off}}(s)).
$$

MARL 中自然扩展为：

$$
Q_{\mathrm{tot}}
\left(
\mathbf o,
\boldsymbol{\pi}_{\mathrm{off}}(\mathbf o)
\right).
$$

但 Online 微调过程中，如果其他智能体策略改变，那么固定：

$$
a_i=\pi_{\mathrm{off}}^i(o_i)
$$

可能不再是一个合理动作。记忆项会保留“旧联合策略下的协调价值”，反而阻碍形成新的协同关系。

### 原因五：部分可观测性

PRQL 假设输入完整状态 $s$。MARL 通常只有：

$$
o_i\neq s,
$$

需要使用 RNN、Transformer 或通信机制。本文没有讨论历史状态、隐藏信息、通信带宽对知识迁移的影响。

### 原因六：算法基础不同

PRQL 基于：

- TD3；
- 确定性 Actor；
- clipped double Q；
- 连续动作。

MARL 中更常见的是：

- 离散动作；
- 随机策略；
- value decomposition；
- CTDE；
- 通信受限 Actor；
- opponent modeling。

因此，不能简单把 TD3-BC 换成现有离线 MARL 算法就保证 PRQL 的机制有效。

---

## 6.3 可以如何改造？

一种可能的 MARL-PRQL 形式是：

1. Offline 阶段使用 centralized Critic 或 value factorization Critic；
2. 训练一个只用于 Offline 分析的中间联合策略：
   $$
   \boldsymbol\mu_m(\mathbf o)
   =
   \arg\max_{\mathbf a}
   Q_{\mathrm{tot}}(\mathbf o,\mathbf a);
   $$
3. 降低联合 Q 在 $\boldsymbol\mu_m(\mathbf o)$ 上的值；
4. 使用 credit-aware 或 factorization-aware 的降值方法，避免破坏值分解；
5. Online 阶段使用 decentralized Actor，并通过旧联合策略动作构造记忆正则；
6. 对 teammate policy 变化引入衰减、条件化或 opponent-conditioned value anchor。

但这些都是非平凡修改，不是直接套用。

---

## 7. 问题 5 所讨论的 Offline 知识保留与阻碍问题，在 MARL 中是否不同、是否更有挑战性？

是的，而且在 MARL 中明显更有挑战性。

| 问题 | 单智能体 O2O RL | MARL O2O RL |
|---|---|---|
| 环境平稳性 | 转移动力学通常固定 | 其他智能体策略变化导致非平稳性 |
| OOD 对象 | 单个动作 $a$ | 联合动作 $\mathbf a=(a_1,\ldots,a_N)$ |
| OOD 空间 | 通常是连续动作空间 | 联合动作空间随智能体数指数增长 |
| Q 值错误归因 | 可归因到当前 Actor 输出 | 需要 credit assignment |
| 记忆锚点 | $Q(s,\pi_{\mathrm{off}}(s))$ | $Q(\mathbf o,\boldsymbol\pi_{\mathrm{off}}(\mathbf o))$ |
| 策略兼容性 | 一个 Actor 输出即可 | 各智能体动作必须协调兼容 |
| 知识有效性 | 同任务下通常较稳定 | 其他智能体变化会使旧知识快速失效 |
| 值函数结构 | 单个 Q | centralized Q、monotonic mixing、IGM、Transformer critic 等 |
| 探索 | 单动作探索 | 联合探索可能破坏团队协调 |
| 崩溃后果 | 单策略性能下降 | 团队整体崩溃，且责任难以定位 |

### 7.1 Offline 阶段应保留什么？

在 MARL 中，仅保留：

$$
Q(\mathbf o,\boldsymbol\pi_{\mathrm{off}}(\mathbf o))
$$

可能不够。更有价值的离线知识可能包括：

- 稳定的个体行为先验；
- 有效的子团队协调模式；
- teammate-conditioned value；
- 通信协议或角色分工；
- 部分观测下的历史表示；
- 对常见队友策略的响应方式；
- 高价值联合动作区域，而不是单一全局动作。

### 7.2 Online 阶段应使用什么？

Online 阶段可能不仅需要最大化联合 Q，还应：

- 保持部分离线行为先验；
- 鼓励协同探索；
- 对 teammate policy 变化进行条件化；
- 逐步降低对旧 teammate policy 的依赖；
- 保持有效通信和角色一致性。

### 7.3 Offline 知识何时会成为阻碍？

在 MARL 中阻碍更明显：

1. **旧协同模式阻碍新协同模式形成**
2. **旧 teammate policy 条件价值已经失效**
3. **联合动作高估比单动作更严重**
4. **记忆正则可能锁定过时联合行为**
5. **一个智能体的探索可能被其他智能体解释为环境变化**
6. **对抗场景中，对手会主动利用离线策略的固定模式**

因此，MARL 中必须回答的不只是“保留多少旧知识”，而是：

> **保留哪些个体知识、哪些协调知识、在何种 teammate 条件下保留，以及如何随 teammate policy 更新而遗忘。**

这是单智能体 PRQL 没有解决的问题。

---

## 8. 本文还存在哪些不足？尤其是迁移到 MARL 时的不足？

## 8.1 论文自己承认的不足

### 1. 没有研究跨任务 Offline-to-Online

文章考虑的是：

- Offline 学一个任务；
- Online 继续微调同一个任务。

没有研究：

- Offline 在任务 A；
- Online 在任务 B；
- 动力学变化；
- 奖励函数变化；
- teammate 变化。

而在任务变化场景中，记忆正则保留的 $Q_{\mathrm{off}}$ 可能会阻碍 Online 学习。

### 2. 强依赖 TD3-BC

PRQL 建立在确定性 TD3-BC 上，不能直接推广到：

- BEAR；
- CQL；
- BCQ；
- SAC 类随机 Actor；
- diffusion policy；
- discrete-action RL。

### 3. 实验环境较简单

实验仅限于：

- 单智能体；
- 全观测；
- MuJoCo 连续控制；
- D4RL。

没有验证：

- AntMaze；
- Adroit；
- 机器人真实交互；
- 部分可观测；
- 多任务；
- 多智能体；
- 高维视觉输入。

### 4. Q 预降低不是完整校准

PRQL 只是把最可能被选择的高估动作相对压低，并不能保证：

$$
Q_m(s,a_{\mathrm{OOD}})
\approx
Q_{\mathrm{true}}(s,a_{\mathrm{OOD}}).
$$

### 5. 缺少理论保证

没有证明：

- 降低 OOD Q 后 collapse 必然消失；
- 记忆正则一定改善收敛；
- 最终策略优于某个下界；
- 误差不会超过给定阈值。

### 6. 超参数与数据质量相关

例如：

- $\eta_2=0$ 用于 random；
- $\eta_2=1$ 用于 medium 和 medium-replay；
- $\eta_2=5$ 用于 medium-expert。

这意味着算法需要根据不同数据质量调节记忆强度，缺乏完全统一的自动选择机制。

---

## 8.2 迁移到 MARL 时的额外不足

如果迁移到 MARL，原方法还存在以下问题：

1. **单一中间策略无法表达联合策略空间**
2. **中心化 argmax 联合动作可能不可执行**
3. **联合动作 OOD 比单动作严重得多**
4. **降值损失不知道应归因到哪个智能体**
5. **记忆项会锚定旧联合策略**
6. **teammate 非平稳性会削弱离线价值知识**
7. **部分可观测性要求显式历史建模**
8. **可能破坏值分解结构和 IGM 条件**
9. **通信受限时无法直接查询中心化价值函数**
10. **对抗 MARL 中，Offline 策略模式可能被对手利用**
11. **联合探索可能引发灾难性团队崩溃**
12. **单团队奖励下的 credit assignment 会使记忆正则更难设计**

因此，即便思想可迁移，也需要显著重构，而不是简单扩展。

---

## 9. 本文的对照算法、实验环境、主实验和验证实验

## 9.1 对照算法及发表信息

| 算法 | 核心思想 | 发表信息 |
|---|---|---|
| AWAC | 优势加权 Actor-Critic，通过隐式策略约束桥接 Offline 与 Online | CoRL 2020  |
| IQL | 使用 expectile regression 避免直接查询 OOD 动作，再通过 advantage-weighted regression 提取策略 | ICLR 2022  |
| SPOT | 用 VAE 建模行为策略支持集，对动作密度进行支持约束 | NeurIPS 2022  |
| Cal-QL | 校准 Offline Q 值下界，使 Online 微调时 Q 尺度更合适 | NeurIPS 2023  |
| PEX | 冻结 Offline 策略并将其加入策略集合，与新策略自适应组合 | ICLR 2023  |
| TD3-FT | 先用 TD3-BC 离线预训练，再直接用 TD3 在线微调 | 由 TD3（ICML 2018）与 TD3-BC（NeurIPS 2021）组合而成的基线 |

其中，本文对 AWAC、IQL、SPOT、Cal-QL、PEX 均按原论文推荐超参数进行了重实现。

---

## 9.2 实验环境

本文使用：

- 物理引擎：MuJoCo；
- 离线强化学习基准：D4RL；
- 任务：
  - Hopper；
  - Walker2D；
  - HalfCheetah；
- 数据质量：
  - random；
  - medium；
  - medium-replay；
  - medium-expert。

共计 12 个任务数据集。

训练设置：

- Offline 预训练：$0.5\mathrm{M}$ 步；
- Q 预降低阶段：$0.2\mathrm{M}$ 步；
- Online 微调：$0.5\mathrm{M}$ 步；
- 5 个随机种子；
- 每 2000 步评估一次；
- 每次评估 10 个 episode；
- 分数按 D4RL 归一化到约 $0\sim100$。

PRQL 和 TD3-FT 在 Online 阶段只使用新在线数据；其余方法按原论文设置混合使用离线和在线数据。

---

## 9.3 主实验

主实验比较各算法在：

- Offline 结束时的分数；
- Online 微调开始时的分数；
- Online 微调结束时的分数。

论文定义：

$$
\mathrm{Dec}=a-b,
$$

表示从 Offline 结束到 Online 初始的性能下降；

$$
\mathrm{Inc}=c-b,
$$

表示从 Online 初始到 Online 结束的性能提升。

平均结果为：

| 算法 | Offline $\rightarrow$ Online初始 $\rightarrow$ Online最终 | 平均下降 Dec | 平均提升 Inc |
|---|---:|---:|---:|
| AWAC | 29.2 $\rightarrow$ 30.4 $\rightarrow$ 34.6 | -1.2 | 5.4 |
| IQL | 43.1 $\rightarrow$ 43.9 $\rightarrow$ 58.3 | -0.8 | 15.1 |
| SPOT | 58.3 $\rightarrow$ 57.9 $\rightarrow$ 70.9 | 0.4 | 12.6 |
| Cal-QL | 41.0 $\rightarrow$ 39.0 $\rightarrow$ 40.5 | 2.0 | -0.5 |
| PEX | 47.5 $\rightarrow$ 35.6 $\rightarrow$ 73.3 | 11.8 | 25.8 |
| TD3-FT | 55.0 $\rightarrow$ 26.6 $\rightarrow$ 80.3 | 28.4 | 25.4 |
| **PRQL** | **55.0 $\rightarrow$ 50.3 $\rightarrow$ 87.3** | **4.7** | **32.3** |

主要结论是：

- PRQL 的平均最终得分最高；
- PRQL 的平均提升最大；
- PRQL 的初始下降远小于 TD3-FT；
- PRQL 并非在每个数据集上都绝对最优，但在聚合结果和大多数任务上表现最好。

---

## 9.4 问题验证实验

### 1. 证明 policy collapse 存在

论文 Fig. 2 用 walker2d-medium-v2 显示：

- 继续用 TD3-BC 微调，提升慢；
- 改用 TD3 微调，最终提升较大，但 Online 初期性能突然大幅下降。

这直接验证了文章要解决的核心问题。

### 2. 验证 Q 预降低确实有效

论文 Fig. 7 在 walker2d-medium-v2 上：

- 训练一个 $\pi_x$ 最大化离线 Q；
- 取 $a^*=\pi_x(s)$ 作为高价值动作；
- 比较 $Q_m(s,a)$ 与 $Q_m(s,a^*)$；
- 用 t-SNE 将状态降到一维展示。

结果显示：

- Offline 结束时，很多 $Q(s,a^*)$ 明显高于数据内动作；
- 经过 $Q_m$ 训练后，$Q(s,a^*)$ 明显下降；
- 数据内动作的 Q 值基本保持不变；
- 约 $0.2\mathrm{M}$ 步后降值效果趋于稳定。

### 3. 研究 $\eta_1$ 对降值的影响

作者改变：

$$
\eta_1\in\{0,0.5,1,5,10,20\}.
$$

结果表明：

- $\eta_1$ 太小，高估动作降不下来；
- $\eta_1$ 太大，离线数据内动作的 Q 值也会被显著扰动；
- $\eta_1=1$ 是折中默认值。

### 4. 比较 Online-only 数据与 Offline-Online 混合数据

Fig. 8 比较：

- 仅使用 Online buffer；
- 使用 Offline-Online 混合 buffer。

结果显示，仅使用 Online 数据时微调效果更好，说明混合 replay 中的旧数据会降低在线样本效率；而记忆正则能够补偿不 replay 旧数据造成的遗忘。

### 5. 记忆正则消融

作者改变：

$$
\eta_2\in\{0,0.5,1,5,10,20\}.
$$

结果显示：

- $\eta_2=0$ 时会出现明显性能崩溃；
- $\eta_2$ 太大时，Online Critic 难以学习新数据；
- Medium 数据集适合 $\eta_2=1$；
- Medium-Expert 数据集适合 $\eta_2=5$；
- Random 数据集设置为 $\eta_2=0$，避免记住低质量离线价值。

### 6. 训练时间与资源实验

作者还比较了训练时间和资源消耗。

论文报告：

- PRQL 的主体 Online 微调时间与 TD3-FT、SPOT 接近；
- 额外的 $0.2\mathrm{M}$ 降值阶段只增加约 $0.3$ 小时；
- AWAC、IQL、Cal-QL、PEX 更耗时；
- PRQL 内存占用约 $2.15\mathrm{GB}$；
- GPU 显存占用约 $519\mathrm{MB}$。

---

## 10. 本文的核心 claim 是什么？是否被理论或实验证明？

## 10.1 Claim 1：PRQL 能缓解 Online 初期 policy collapse

### 支持证据

- Fig. 2 验证了普通 TD3 微调会出现 collapse；
- 主学习曲线显示 PRQL 初始下降较小；
- Table 1 中 PRQL 平均初始下降为 $4.7$，明显小于 TD3-FT 的 $28.4$；
- $\eta_1=0$ 的实验说明，如果不进行 Q 降值，初始微调性能会明显变差。

### 证据类型

主要是实验证据，没有理论证明。

---

## 10.2 Claim 2：PRQL 的 Q 预降低能够压低 OOD 高估，同时保留数据分布内动作价值

### 支持证据

- Fig. 7 显示 over-estimated Q 随训练下降；
- in-data Q 基本不变；
- Table 2 和 Fig. 10 显示 $\eta_1$ 控制降值强度；
- $\eta_1=1$ 能较好兼顾降值与保留 in-distribution Q。

### 证据类型

分析实验和超参数实验。

但需要注意，本文只证明了“相对降值”，没有证明“校准到真实 Q 值”。

---

## 10.3 Claim 3：仅使用 Online 数据并结合记忆正则，比混合 replay 更高效

### 支持证据

- Fig. 8 显示 Online-only buffer 的最终性能优于混合 buffer；
- Fig. 9 和 Fig. 11 显示 $\eta_2=0$ 会导致明显崩溃；
- 合适的 $\eta_2$ 能提高稳定性和样本效率。

### 证据类型

设计比较实验和消融实验。

---

## 10.4 Claim 4：PRQL 的最终微调性能优于大多数基线

### 支持证据

Table 1 中：

- PRQL 平均最终得分为 $87.3$，高于所有基线；
- PRQL 平均提升为 $32.3$，也是最高；
- 在多数 Random、Medium、Medium-Replay 和部分 Medium-Expert 数据集上表现最好。

### 限制

- 不是在每个单独数据集上都严格最优；
- 例如某些 Medium-Expert 任务上 TD3-FT 或 PEX 的最终结果接近甚至略高于 PRQL；
- 因此“最优”应理解为**总体平均意义和大多数任务上的优势**。

---

## 10.5 Claim 5：PRQL 计算开销较低

### 支持证据

- Fig. 12 显示 PRQL 主体微调时间与 TD3-FT、SPOT 接近；
- 额外降值阶段时间较小；
- Table 3 显示其内存和显存占用与其他方法接近。

### 证据类型

运行时间和资源统计实验。

---

# 总体评价

这篇文章的核心价值在于，它没有继续在 Online 阶段设计更复杂的策略约束，而是将问题前移：

> **既然 Offline Critic 对 OOD 动作存在错误高估，那么就在 Online 开始前先改变 Actor 所面对的价值地形。**

同时，它用记忆正则把“保留离线知识”从“持续 replay 旧数据”改成“保留参数化价值知识”，从而实现了：

- 初期较稳定；
- Online 阶段无 BC 约束；
- 只使用新数据；
- 保留有效离线价值；
- 避免明显 policy collapse。

其主要弱点则是：

- 没有理论证明；
- 只验证单智能体、全观测、同任务 O2O；
- 对任务变化和分布偏移支持不足；
- 强依赖 TD3-BC；
- 对 Offline 知识的选择仍比较简单；
- 迁移到 MARL 时，会遇到联合动作 OOD、非平稳性、credit assignment、旧协同关系锚定和部分可观测性等显著更困难的问题。