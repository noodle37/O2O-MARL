下面总结你上传的 NeurIPS 2025 论文 **《Uni-RL: Unifying Online and Offline RL via Implicit Value Regularization》**，并按你的 10 个问题回答。与上一篇 RPEX 不同，这篇不是专门做“污染鲁棒 O2O”，而是提出一个统一 online / offline / offline-to-online 的 model-free RL 框架；它对 O2O RL 的贡献主要在于：**把 offline 阶段对行为策略 $\pi_{\mathcal D}$ 的约束，在线阶段切换为对迭代改进参考策略 $\bar\pi$ 的约束，从而缓解 O2O 中约束过强导致 conservative fine-tuning，以及直接放开约束导致 early performance drop 的问题。**

---

# 文章主要内容总结

论文核心目标是：不为 online、offline、offline-to-online 分别设计算法，而是用一个统一目标处理三类设定。统一问题是带参考策略 $\mu$ 的约束优化：

$$
\pi^*
=
\arg\max_\pi
\mathbb E_{(s,a)\sim d^\pi}[r(s,a)]
\quad
\text{s.t.}
\quad
\mathbb E_{s\sim d^\pi}
\left[
D_f(\pi(\cdot|s)\|\mu(\cdot|s))
\right]
\le \epsilon ,
$$

其中

$$
D_f(p\|q)=\mathbb E_{x\sim q}\left[f\left(\frac{p(x)}{q(x)}\right)\right]
$$

是 $f$-divergence。通过 Lagrangian 对偶得到 behavior-regularized MDP 与 in-sample value learning；参考策略 $\mu$ 的选择统一三类场景：

- offline：$\mu=\pi_{\mathcal D}$，即数据集行为策略；
- online：$\mu=\bar\pi$，其中 $\bar\pi$ 是向当前策略软更新的目标策略；
- offline-to-online：offline 阶段用 $\mu=\pi_{\mathcal D}$，online 阶段切换为 $\mu=\bar\pi$，且 $\bar\pi$ 从 offline 初始化策略出发迭代改进。

policy extraction 上，论文不满意 IVR 的 weighted BC 过于 mode-covering，也不满意 MaxQ+BC 容易 OOD 与 overestimation，于是提出 **In-sample Policy Gradient, InPG**：仍用 weighted BC 的稳定形式，但把权重 $w(s,a)$ 当成可学习变量，使其诱导的 BC 梯度逼近 in-sample policy gradient。

---

## 1. 本文解决 O2O RL 中的什么问题？为什么现有算法还没解决好？

### 对 O2O RL 的具体问题
本文针对 O2O 的核心矛盾是：

1. **offline 学到的策略/价值需要保留**：直接用无约束 online RL 微调，容易因为 distribution shift 在 fine-tuning 早期掉性能，即 early performance drop。
2. **但 offline 数据约束也不能太强**：IQL、AWAC、Cal-QL 一类方法对 offline dataset 或 offline policy 保持较强约束，在线阶段仍被差数据锚定，导致 fine-tuning 只 marginal improvement。论文 Figure 5 明确显示 constraint-based 方法如 IQL、AWAC 在 Adroit、AntMaze 上提升有限。

其机制是：offline IVR 以 $\mu=\pi_{\mathcal D}$ 学习；若 online 继续用 $\mu=\pi_{\mathcal D}$，则 online replay buffer 中包含大量早期随机/次优数据，正则会把策略锚到 buffer behavior policy。若完全去掉约束，又会丢失 offline 初始化带来的稳定性。

### 为什么现有算法没解决
- online 方法 SAC/TD3/PPO 不处理 offline distribution shift；
- offline 方法 CQL/IQL/IVR 偏保守，不擅长在线持续改进；
- O2O 专门方法 Cal-QL/AWAC/Off2On/PROTO 常依赖特定 offline 算法或额外校正项，不够 general；
- PEX 类方法冻结 offline policy 并训练 expansion policy，但本文指出其思路不同，Uni-RL 是通过 reference policy 的 implicit trust region 统一 value learning 与 policy extraction；
- TRPO/PPO/MPO 有 trust region，但通常是 on-policy 或需显式策略概率约束，样本效率不如 Uni-RL 的 off-policy implicit value regularization。

所以本文解决的是：**O2O fine-tuning 中“稳定继承 offline 初始化”与“逐步摆脱次优 offline 约束”的平衡问题。**

---

## 2. 核心创新点是什么？

核心创新有三个：

1. **统一 reference-policy 约束框架**  
   把 IVR 的 dataset behavior constraint 推广为 reference-policy constraint：
   $$
   D_f(\pi\|\mu)\le \epsilon .
   $$
   offline 时 $\mu=\pi_{\mathcal D}$；online/O2O 时 $\mu=\bar\pi$，且
   $$
   \bar\pi\leftarrow \lambda\pi+(1-\lambda)\bar\pi .
   $$
   这相当于 implicit trust region：约束不是固定在某个旧数据集，而是随当前策略缓慢更新。

2. **in-sample unified value learning**  
   价值学习只用参考策略动作样本，不显式依赖 $\pi(a|s)$ 与 $\mu(a|s)$，从而可扩展到 diffusion/flow matching 等概率难求的策略。对 KL divergence 有 closed-form IVR objective；对 $\chi^2$ 也有对应形式。

3. **InPG 统一 policy extraction**  
   不显式做 MaxQ，也不完全 mode-covering weighted BC，而是学习权重
   $$
   w_{\mathrm{InPG}}^*(s,a)
   =
   \arg\min_{w\ge 0}
   \mathbb E_{s\sim\mathcal D}
   \left[
   \left(
   G_{\mathrm{PG}}-G_{\mathrm{BC}}(w)
   \right)^2
   \right],
   $$
   其中
   $$
   G_{\mathrm{PG}}
   =
   \nabla_\theta
   \mathbb E_{a\sim\pi_\theta}[Q(s,a)],
   \quad
   G_{\mathrm{BC}}(w)
   =
   \mathbb E_{a\sim\mu}
   [
   w(s,a)\nabla_\theta\log\pi_\theta(a|s)
   ].
   $$
   用解出的 $w^*_{\mathrm{InPG}}$ 做 weighted BC，兼具 mode-seeking 与 in-sample 稳定。

一句话：**Uni-RL 把“对行为数据正则”改成“对参考策略正则”，并用 InPG 让 BC 权重隐式执行策略梯度，从而统一三类 RL setting。**

---

## 3. 核心模块、如何发挥作用、完整算法流程

### 模块 A：统一约束优化与对偶 value learning
原问题：

$$
\max_\pi \mathbb E_{(s,a)\sim d^\pi}[r(s,a)]
\quad
\text{s.t. }
\mathbb E_{s\sim d^\pi}
[D_f(\pi(\cdot|s)\|\mu(\cdot|s))]
\le \epsilon .
$$

Lagrangian 变成行为正则 MDP：

$$
\pi^*
=
\arg\max_\pi
\mathbb E_\pi
\left[
\sum_{t=0}^\infty
\gamma^t
\left(
r(s_t,a_t)
-\alpha
g\left(
\frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}
\right)
\right)
\right],
$$

其中 $g(x)=f(x)/x$，且 $g(1)=0$。对应 Bellman backup：

$$
(\mathcal T_f^\pi Q)(s,a)
=
r(s,a)+\gamma\mathbb E_{s'|s,a}[V(s')],
$$

$$
V(s)
=
\mathbb E_{a\sim\pi}
\left[
Q(s,a)
-\alpha
g\left(
\frac{\pi(a|s)}{\mu(a|s)}
\right)
\right].
$$

作用：把 policy-level constraint 转成 value regularization，避免显式策略散度估计。

### 模块 B：in-sample V/Q learning
$$
\min_V
\mathbb E_{s\sim\mathcal D,a\sim\mu}
\left[
V(s)+\alpha f_{\mathrm{IVR}}
\left(
\frac{Q(s,a)-V(s)}{\alpha}
\right)
\right],
$$

$$
\min_Q
\mathbb E_{(s,a,s')\sim\mathcal D}
\left[
\left(
r(s,a)+\gamma V(s')-Q(s,a)
\right)^2
\right].
$$

对 KL divergence：

$$
f_{\mathrm{IVR}}(x)=\exp(x);
$$

对 $\chi^2$：

$$
f_{\mathrm{IVR}}(x)
=
\mathbf 1(x>-2)(x^2/4+x+1).
$$

作用：只需从 $\mu$ 采动作即可学 $V,Q$，离线时从 dataset 采动作天然满足。

### 模块 C：iteratively refined reference policy
offline：

$$
\mu=\pi_{\mathcal D}.
$$

online/O2O：

$$
\mu=\bar\pi,
\quad
\bar\pi\leftarrow \lambda\pi+(1-\lambda)\bar\pi .
$$

若 $\lambda=1$，则第 $k$ 轮用上一轮策略 $\pi_{k-1}$ 作为 reference。作用：形成 implicit trust region，既防止 online 初期大偏差，又允许逐步放松 dataset constraint。

### 模块 D：InPG policy extraction
先算目标策略梯度方向 $G_{\mathrm{PG}}$，再用可学习权重 $w(s,a)$ 调整 weighted BC 梯度：

$$
G_{\mathrm{BC}}(w)
=
\mathbb E_{a\sim\mu}
[
w(s,a)\nabla_\theta\log\pi_\theta(a|s)
].
$$

通过最小化

$$
\mathbb E_s
\|G_{\mathrm{PG}}-G_{\mathrm{BC}}(w)\|^2
$$

学习 $w^*_{\mathrm{InPG}}$，然后 policy update 为

$$
\pi^*_{\mathrm{InPG}}
=
\arg\max_\pi
\mathbb E_{s\sim\mathcal D,a\sim\mu}
[
w^*_{\mathrm{InPG}}(s,a)\log\pi(a|s)
].
$$

实现上 $w$ 用神经网络参数化，输出 clip 到非负；论文实现中还做 batch 0-1 normalization 并 clip 到 $[w_{\min},w_{\max}]=[0.1,0.9]$。

### 完整流程
Algorithm 1：

1. 初始化 $Q,V,\pi,\bar\pi$。
2. 对 offline 训练：重复采样 $(s,a,r,s')\sim\mathcal D$，令 $\mu=\pi_{\mathcal D}$，更新 $V,Q,\pi$。
3. 对 online/O2O 训练：
   - 用当前 $\pi$ 探索，把新 transition 加入 replay $\mathcal D$；
   - 令 $\mu=\bar\pi$，用 in-sample objective 更新 $V,Q$；
   - 用 InPG 更新 $\pi$；
   - 软更新 $\bar\pi\leftarrow\lambda\pi+(1-\lambda)\bar\pi$。
4. 若使用 diffusion/flow policy，则把 weighted log-likelihood 替换成对应 matching loss：
   $$
   \mathcal L_{\mathrm{Flow}}(\theta)
   =
   \mathbb E_{s\sim\mathcal D,a=x^1\sim\mu}
   \left[
   \|v_\theta(t,s,x^t)-(x^1-x^0)\|^2
   \right],
   $$
   diffusion 类似地用 score matching loss。

---

## 4. 是否有理论证明？简要概述

有。主要证明 InPG 相对 IVR/直接 PG 的性质。

### Theorem 1：InPG 策略与参考策略的 $f$-divergence 有界
若

$$
w_{\min}=\min_x w(x),
\quad
w_{\max}=\max_x w(x),
\quad
Z=\mathbb E_{x\sim\mu}[w(x)],
$$

则

$$
f\left(\frac{w_{\min}}{Z}\right)
\le
D_f(\pi^*_{\mathrm{InPG}}\|\mu)
\le
f\left(\frac{w_{\max}}{Z}\right).
$$

含义：通过控制权重范围，可控制新策略偏离 reference policy 的程度。

### Theorem 2：存在 $w_{\min},w_{\max}$ 使 InPG 性能不低于 IVR
论文给出条件：

$$
\frac{w_{\max}}{Z}
\le
f^{-1}
\left(
\mathbb E_{s\sim\mathcal D}
[
D_f(\pi^*_{\mathrm{IVR}}(\cdot|s)\|\mu(\cdot|s))
]
\right),
$$

可保证

$$
J(\pi^*_{\mathrm{InPG}})
\ge
J(\pi^*_{\mathrm{IVR}}).
$$

含义：InPG 是在固定约束范围内找更优 policy，而不是把约束当 penalty，因此可能比 IVR 解更好。

### Theorem 3：直接 PG 无安全改进保证
定义

$$
\pi^*_{\mathrm{PG}}
=
\arg\max_\pi
\mathbb E_{s\sim\mathcal D,a\sim\pi}[Q(s,a)],
$$

则一般不保证

$$
J(\pi^*_{\mathrm{PG}})\ge J(\mu).
$$

证明核心是直接最大化 Q 会引入正则惩罚项

$$
-\alpha D_f(\pi^*_{\mathrm{PG}}\|\mu),
$$

所以可能差于 reference policy。该定理支持为什么需要 in-sample projection，而不是直接用 policy gradient。

局限：理论是目标层面与 improvement 条件，不给 finite-sample convergence 或 O2O regret bound；Theorem 2 的存在性条件也较理想化。

---

## 5. 是否讨论 Offline 阶段什么知识值得保留、Online 如何使用、是否阻碍 Online？

有，且是本文 O2O 设计的核心。

### Offline 阶段值得保留的知识
1. **offline policy 初始化**：作为 online fine-tuning 起点；
2. **offline 学到的 $Q,V$**：用于后续 policy extraction；
3. **offline dataset 的行为信息**：offline 阶段以 $\mu=\pi_{\mathcal D}$ 约束；
4. 在 O2O 中，offline pretrain 的价值函数直接为 online $\bar\pi$ 初始化提供基础。

### Online 阶段如何使用
online 阶段不继续强制约束 $\pi_{\mathcal D}$，而是把 reference 切换为

$$
\bar\pi\leftarrow \lambda\pi+(1-\lambda)\bar\pi .
$$

因此 offline 初始化策略相当于初始 reference policy；online 新数据通过 $\pi$ 收集，$\bar\pi$ 作为 trust-region center 缓慢跟随当前策略。这样既保留 offline policy/value 的初始化收益，又避免永远被差 offline data 锚定。

### Offline 知识是否阻碍 Online？
论文明确认为是。Figure 2 比较了 IVR $(\mu=\pi_{\mathcal D})$ 与 Uni-RL $(\mu=\bar\pi)$：直接 online IVR 会被 replay buffer 中早期次优数据过约束，Q value 与 score 上升慢；改用 $\bar\pi$ 后更好。Figure 5 也显示 IQL/AWAC 这类 dataset-constrained 方法在 online 提升有限。

不过本文没有深入讨论更细粒度问题：例如 offline value 错误时哪些知识应丢弃、offline policy 何时应完全停止作为 reference、reference policy 污染/失效检测等。

---

## 6. 是否适合迁移到 MARL？理由与挑战

**思想上适合，直接迁移不简单。**

### 适合的理由
- CTDE 下的 MARL 很自然有 centralized critic 与 decentralized actors，类似 Uni-RL 的 value/policy 解耦；
- 多智能体离线数据同样有 behavior policy 约束问题，offline-to-online 在 robot team、autonomous driving、multi-agent games 中很现实；
- reference policy 可以扩展为 joint reference policy 或 teammates’ policies；online 阶段用迭代 reference 缓解旧队友策略过时；
- in-sample learning 对 offline MARL 很重要，因为 joint action OOD 更严重。

### 主要挑战
1. **联合动作 reference policy**  
   单体 Uni-RL 的 $\mu(a|s)$ 到 MARL 要变成
   $$
   \mu(\mathbf a|s)
   =
   \mu(a^1,\dots,a^n|s),
   $$
   或近似成
   $$
   \prod_i \mu^i(a^i|s^i),
   $$
   后者在协同任务里通常不准确。

2. **值函数与信用分配耦合**  
   centralized
   $$
   Q^{\mathrm{tot}}(s,\mathbf a)
   $$
   与 VDN/QMIX 的 mixer 会引入额外误差；InPG 中
   $$
   G_{\mathrm{PG}}=\nabla_\theta\mathbb E_{\mathbf a\sim\pi_\theta}[Q^{\mathrm{tot}}(s,\mathbf a)]
   $$
   对哪个 agent 的梯度更接近、哪个应被加权，需要额外 credit assignment。

3. **非平稳性更强**  
   online 中每个 agent 同时变化，$\bar\pi$ 从“稳定旧策略”变成“持续移动的目标”。reference policy 若更新太快，trust region 失效；太慢，又跟不上 teammates。

4. **联合动作 OOD 更严重**  
   in-sample 只保证参考 joint actions 不 OOD；一旦参考策略本身覆盖差，online 仍可能学不到关键协同动作。

5. **生成模型策略概率难求**  
   MARL 若用 diffusion/flow 生成联合动作，InPG 的 weighted likelihood/matching loss 可迁移，但 joint action likelihood 与梯度估计更难。

6. **理论假设弱化**  
   定理依赖固定 MDP、reference distribution 可采样、value learning 足够好；MARL 的 nonstationarity、partial observability、credit assignment 都会破坏这些假设。

---

## 7. 结合问题 5：MARL 中 Offline 知识保留、Online 使用与阻碍是否更不同、更难？

是。单体 O2O 中，offline 主要保留一个 policy/value 初始化；MARL 中可保留的是**协同知识**：joint behavior prior、teammate policies、value decomposition/mixer、communication prior、role structure。

Online 使用也应不同：不能只用单个 $\bar\pi$，更合理是维护 teammate-wise references
$$
\bar\mu=\{\bar\pi^1,\dots,\bar\pi^n\}
$$
或 joint reference $\bar\pi(\mathbf a|s)$， centralized critic 用
$$
Q^{\mathrm{tot}}(s,\mathbf a)-V(s)
$$
做 advantage；InPG 权重既可在 joint-action 层学习，也可分解为 per-agent responsibility weights。

阻碍更严重：
- offline teammate policy 在 online 更快过时；
- 旧 value decomposition 可能与新策略组合不匹配；
- 一个 agent 的变化会改变其他 agent 的最优 reference；
- shared reward 下错误 credit assignment 会污染所有 agent 的 InPG 权重；
- joint action 空间大使“固定约束范围 $[w_{\min},w_{\max}]$”更难校准；
- nonstationarity 使 implicit trust region 的中心 $\bar\pi$ 本身成为移动目标。

因此，MARL 版本需要额外机制：**stale reference detection、teammate policy gating、per-agent credit-aware InPG、joint support coverage estimation**。否则 Uni-RL 的 trust-region 优点会变成“锚定在旧协同关系上”。

---

## 8. 本文仍存在哪些不足？尤其迁移到 MARL 时

1. **不处理数据污染/对抗攻击**  
   与 RPEX 不同，本文没有考虑 offline/online corruption。reference policy 若来自污染数据，$\bar\pi$ 会把坏行为当成 trust center。

2. **理论较理想化**  
   Theorem 2 是存在条件；实际 $w,Q,V$ 都要估计，没有 finite-sample guarantee。

3. **依赖 reference action sampling**  
   offline 还好，online 若 $\mu=\bar\pi$ 策略表达差或采样不足，in-sample objective 会偏差。

4. **超参仍敏感**  
   $\alpha,\lambda,w_{\min},w_{\max}$ 都需调。论文给了统一范围，但并非无参。

5. **O2O 中没有显式讨论 offline 知识失效时的剪枝**  
   只会通过 $\bar\pi$ 渐进放松约束，不会主动丢弃明显有害的 offline prior。

6. **policy extraction 额外学习权重网络**  
   InPG 需要训练 $w(s,a)$，增加实现复杂度；QP 只被梯度法近似求解，稳定性依赖优化。

7. **仅 model-free 单智能体 MDP**  
   作者 limitation 也承认未扩展到 model-based RL 与 LLM+RL。

8. **迁移 MARL 的额外不足**  
   joint action likelihood、credit assignment、partial observability、communication、nonstationarity 都会削弱 InPG 与 reference constraint 的有效性；而且一旦 teammate reference 错了，影响是全队性的。

---

## 9. 对照算法、发表信息、实验环境与消融/问题验证实验

### Online RL baselines
- TD3：Fujimoto et al., ICML 2018；
- SAC：Haarnoja et al., ICML 2018；
- PPO：Schulman et al., arXiv 2017；
- MPO：Abdolmaleki et al., ICLR 2019/arXiv 2018。

实验环境：OpenAI Gym、DeepMind Control、PyBullet 共 8 个：
AntBulletEnv-v0、BipedalWalker-v3、finger-turn-hard、HalfCheetah-v3、Hopper-v3、Humanoid-v3、quadruped-walk、Walker2d-v3。每个 5 seeds、500k steps，Gaussian policy。

### Offline RL baselines
- 10%BC：Chen et al., NeurIPS 2021；
- TD3+BC：Fujimoto & Gu, arXiv 2021；
- CQL：Kumar et al., NeurIPS 2020；
- IQL：Kostrikov et al., arXiv 2021 / ICLR 2022；
- IVR：Xu et al., arXiv 2023；
- Diffusion-QL：Wang et al., 2023；
- IDQL：Hansen-Estruch et al., arXiv 2023。

环境：D4RL MuJoCo medium/medium-replay/medium-expert，以及 AntMaze umaze/medium/large 的 play/diverse。Uni-RL 同时报 Gaussian 与 Diffusion policy 版本。

### Offline-to-online RL baselines
- AWAC：Nair et al., arXiv 2020 / IJRR 2021 前后；
- IQL：同上；
- Cal-QL：Nakamoto et al., NeurIPS 2023；
- RLPD：Ball et al., ICML 2023，论文为公平去掉 high-UTD trick。

环境：D4RL Adroit cloned：pen、door、hammer、relocate；AntMaze medium/large play/diverse。O2O 先 250k offline pretraining，再 online fine-tuning 500k，5 seeds。

### 问题验证/分析实验
1. **Figure 2**：验证 online IVR 用 $\mu=\pi_{\mathcal D}$ 会过保守；改用 $\mu=\bar\pi$ 后 Q value 与 score 更高，并且在 Weighted BC 与 MaxQ 两类 extraction 下都有效。
2. **Figure 3**：Maze2D 动作可视化，说明 IVR weighted BC 更 mode-covering，动作集中在 offline action 分布附近；Uni-RL/InPG 能选到靠近动作空间边界但更优的动作。
3. **Figure 5**：O2O 主验证，说明 constraint-based IQL/AWAC 提升有限，Cal-QL 受限于 CQL 本身质量，RLPD 在好 offline data 下样本效率不如 Uni-RL。

### 消融/组件性实验
本文没有像 RPEX 那样大量单独消融表，但有组件性证据：
- Gaussian vs Diffusion policy：Table 1 显示 diffusion 进一步提升，验证 in-sample/InPG 可扩展到生成式 policy；
- IVR vs Uni-RL：Table 1 中 Uni-RL 普遍优于 IVR，说明 policy extraction 改进有效；
- weighted BC vs InPG：Figure 3 可视化权重作用；
- $\alpha,\lambda$ 附录 Tables 3–5 报告每环境取值，但没有做大规模 sensitivity heatmap；
- 权重 clip：附录说明用 0-1 normalization 后 clip 到 $[0.1,0.9]$。

---

## 10. 核心 claims 及是否有理论/实验证明

| 核心 claim | 证明形式 | 证据 | 评价 |
|---|---|---|---|
| Uni-RL 可用统一目标处理 online/offline/O2O | 方法 + 实验 | Figure 1、Figure 4、Table 1、Figure 5 | 强，覆盖 6 benchmark、23 env |
| 直接 online 使用 IVR $(\mu=\pi_{\mathcal D})$ 会过保守 | 问题验证 | Figure 2 | 强 |
| $\mu=\bar\pi$ 的 implicit trust region 提升 online sample efficiency | 主实验 + 分析 | Figure 2、Figure 4 | 强 |
| InPG 比 IVR weighted BC 更好利用 Q/value | 理论 + 可视化 + offline 主实验 | Theorem 1–3、Figure 3、Table 1 | 中强；理论非 finite-sample |
| InPG 比直接 MaxQ/PG 更稳、更不 OOD | 理论 + 实验 | Theorem 3、Figure 3、Table 1 | 中强 |
| 方法可扩展到 diffusion/flow policy | 实验 | Table 1 Uni-RL Diffusion、Eq. (9) | 强，但只在 D4RL 验证 |
| O2O 中优于 constraint-based 与 unconstrained 方法 | 主实验 | Figure 5 | 强，限于 D4RL Adroit/AntMaze |
| 方法仍有限制 | 作者 limitation | model-free only；未做 model-based/LLM+RL；代码 submission 时未公开 | 强，属于明确承认 |

总结：这篇论文最强的 claim 是 **“reference-policy constraint + in-sample value learning + InPG extraction 能统一 online/offline/O2O”**，并有较完整 benchmark 支撑；相对较弱的是它**不解决 corruption、多智能体非平稳性、offline 知识失效检测**，而这些恰是迁移到真实世界或 MARL 时最关键的问题。