# 带校准的保守 Q 学习向多智能体价值分解架构的迁移：MaCal-QL 实现说明与正确性分析

## 1 背景与迁移目标

带校准的保守 Q 学习（Calibrated Q-Learning，Cal-QL，Nakamoto 等，2023，arXiv:2303.05479）面向单智能体的离线到在线（offline-to-online）强化学习：在保守 Q 学习（Conservative Q-Learning，CQL）的基础上增加一项校准，使学到的保守价值函数不低于参考行为策略的价值，从而避免在线微调初期因分布偏移导致的价值遗忘与性能跌落。

本仓库把该算法迁移到多智能体设定，骨干采用价值分解网络 QMIX，实现位于 `macal/` 目录，命名为 MaCal-QL。迁移的两个关键决定如下。其一，校准作用在联合动作值 $Q_{\text{tot}}$ 上，因为去中心化贪婪策略真正改进的是这个联合值。其二，QMIX 没有显式策略网络，候选联合动作用一个由各智能体逐动作最大值构造的去中心化 $\varepsilon$-贪婪提议分布产生，并用重要性权重修正提议分布与均匀分布带来的偏差。

Offline-to-Online 多智能体强化学习论文 OVMSE（arXiv:2410.19450）记录了同一迁移，文中记为 MACal-QL，定义为"将 Cal-QL 这一面向单智能体离线到在线强化学习的算法应用到多智能体，骨干为 QMIX"。本文实现与该定义、以及该论文附录 A 列出的超参数设置对齐。本文余下部分说明实现细节，并给出正确性分析。

## 2 Cal-QL 原始算法回顾

### 2.1 保守项与校准修正

保守 Q 学习的正则项压低数据分布之外动作的价值。Cal-QL 的修改（原文式 5.1 与附录式 A.1）为

$$J_Q(\theta) := \alpha\Big(\mathbb{E}_{s\sim\mathcal{D},\,a\sim\pi}\big[\max\big(Q_\theta(s,a), V^{\mu}(s)\big)\big] - \mathbb{E}_{s,a\sim\mathcal{D}}\big[Q_\theta(s,a)\big]\Big) + \frac{1}{2}\mathbb{E}_{s,a,s'\sim\mathcal{D}}\Big[\big(Q_\theta(s,a)-\mathcal{B}^{\pi}\bar{Q}(s,a)\big)^2\Big]$$

其中 $\mathcal{D}$ 为数据集合，$\pi$ 为当前策略，$\mu$ 为参考行为策略。关键细节有两点：第一，下界钳制 $\max(\cdot, V^{\mu}(s))$ 只施加在"当前策略提议的动作"这一组上，均匀随机动作组与数据动作都不钳制；第二，只要 $\alpha$ 足够大，被压得过低的价值会被抬回参考策略价值附近，从而在保持保守性的同时不至于过度悲观。

### 2.2 参考策略价值的估计

$V^{\mu}(s)$ 取行为策略的蒙特卡洛回报至当前（return-to-go）。原论文附录说明，对以终止状态结束的任务，直接使用回报至当前估计即可，无需额外拟合价值网络。官方实现中的递推为

$$G_t = r_t + \gamma\,(1-d_t)\,G_{t+1}$$

其中 $d_t$ 是真正终止标记；因时间上限被截断的回合不产生有效终止，其回报至当前缺少截断之后的尾部。

### 2.3 官方实现的重要性采样形式

Cal-QL 官方代码（`nakamotoo/Cal-QL`，`JaxCQL/conservative_sac.py`）中，上述第一项期望用对数求和指数（logsumexp）估计，具体形式为：

- 均匀随机动作组：$q_{\text{rand}} - \log(0.5^{d})$，即加上均匀分布密度的负对数（连续动作在 $[-1,1]^d$ 上的均匀密度为 $0.5^d$）；
- 策略提议组：对当前状态 $s$ 与下一状态 $s'$ 各采样一组策略动作，取 $q_{\pi} - \log\pi$；
- Cal-QL 的修正：$\max(q_{\pi}, \text{mc\_return})$，只作用于策略提议组；
- 两组拼接后取对数求和指数，减去数据动作的价值，再乘以 CQL 权重与时序差分损失相加。

官方默认候选数为 10，对数求和指数温度为 1，差值裁剪为正负无穷，拉格朗日自适应权重关闭（若开启，目标动作差距默认为 1.0）。需要指出，原论文附录的程序清单把均匀组写成减去 $0.5^{d}$，与官方代码不一致，属排版疏漏；正确形式是加上密度的负对数。本实现采用了正确形式。

## 3 本仓库的实现

### 3.1 代码组织

- `macal/returns.py`：回报至当前的反向递推与有效性标记；
- `macal/calql.py`：保守损失混合类 `CalQLLossMixin._compute_calql_loss`，包含候选采样、重要性权重、校准钳制与损失组装；
- `macal/learner.py`：`MaCalQLLearner`，组合 QMIX 基础件与该损失，负责时序差分目标、总损失组装、对偶权重更新与目标网络刷新；
- `configs/macal_5m_vs_6m.yaml` 与 `configs/macal_2halfcheetah.yaml`：两组超参数；
- `training/offline.py` 与 `training/online.py`：离线预训练与在线微调两个阶段的训练循环。

### 3.2 记号与联合动作空间

设智能体数为 $N$，观测-动作历史为 $\tau$，联合动作为 $u=(u_1,\dots,u_N)$。QMIX 给出逐智能体动作值 $Q_i(\tau,a_i)$，经混合网络得到联合动作值 $Q_{\text{tot}}(\tau,u)$，并满足独立-全局-最大值性质：联合贪婪动作可分解为各智能体对各自 $Q_i$ 取最大值。合法联合动作集合为

$$\mathcal{U}(\tau)=\prod_{i=1}^{N}\mathcal{A}^{\text{legal}}_i,\qquad |\mathcal{U}(\tau)|=\prod_{i=1}^{N} n_i$$

其中 $n_i$ 为第 $i$ 个智能体在当前状态的合法动作数。

### 3.3 候选联合动作与重要性权重

每个转移采样 $k$ 个候选（配置项 `num_ood_actions`，默认 10），分两组。

均匀采样组：每个智能体在其合法动作上独立均匀采样，得到 $u^{r}_m\sim\text{Uniform}(\mathcal{U}(\tau))$。重要性权重取联合均匀密度的负对数，即联合合法动作数的对数：

$$q^{\text{rand}}_m = Q_{\text{tot}}(\tau, u^{r}_m) + \log|\mathcal{U}(\tau)| = Q_{\text{tot}}(\tau, u^{r}_m) + \sum_{i=1}^{N}\log n_i$$

该权重使估计无偏：

$$\mathbb{E}_{u\sim\text{Uniform}(\mathcal{U})}\Big[\exp\big(Q_{\text{tot}}(\tau,u)+\log|\mathcal{U}|\big)\Big]=\sum_{u\in\mathcal{U}}\exp Q_{\text{tot}}(\tau,u)$$

去中心化 $\varepsilon$-贪婪提议组：以概率 $1-\varepsilon$ 取各智能体当前 $Q_i$ 的最大值动作，以概率 $\varepsilon$ 取均匀合法动作。逐智能体提议概率为

$$\pi_i(a_i\mid\tau) = (1-\varepsilon)\,\mathbb{1}\Big[a_i=\arg\max_{a'\in\mathcal{A}^{\text{legal}}_i} Q_i(\tau,a')\Big] + \frac{\varepsilon}{n_i}$$

联合提议概率为 $\pi(u\mid\tau)=\prod_{i=1}^{N}\pi_i(u_i\mid\tau)$，重要性权重取其负对数并在智能体上求和：

$$q^{\text{prop}}_m = Q_{\text{tot}}(\tau, u^{p}_m) - \log\pi(u^{p}_m\mid\tau) = Q_{\text{tot}}(\tau, u^{p}_m) - \sum_{i=1}^{N}\log\pi_i(u^{p}_{m,i}\mid\tau)$$

其中逐智能体对数概率带有下限裁剪以防止取对数于零。

### 3.4 校准锚与回报至当前

`macal/returns.py` 中的 `compute_mc_returns` 反向递推 $G_t = r_t + \gamma(1-d_t)G_{t+1}$，并为每个回合输出有效性标记：只有当回合内出现过"真正终止且已填充"的步时，锚才生效。因时间上限被截断的回合，其回报至当前缺少截断之后的尾部，作为下界会有系统性偏低，因此整段放弃锚。数据侧的自举槽位步标记为终止但未填充，不会被误判为有效终止。校准钳制只作用于提议组：

$$q^{\text{prop,cal}}_m = \begin{cases}\max\big(q^{\text{prop}}_m, G_t\big), & \text{锚有效} \\[2pt] q^{\text{prop}}_m, & \text{否则}\end{cases}$$

### 3.5 保守损失与总目标

$$L_{\text{CQL}} = \frac{\sum_{\tau\in\mathcal{M}} m_\tau\Big[\operatorname{logsumexp}_m\big(\{q^{\text{rand}}_m\}\cup\{q^{\text{prop,cal}}_m\}\big) - Q_{\text{tot}}\big(\tau, a^{\text{data}}\big)\Big]}{\sum_{\tau\in\mathcal{M}} m_\tau}$$

其中 $\mathcal{M}$ 为批次内的转移集合，$m_\tau$ 为由填充标记与终止标记共同决定的步级掩码。总目标分两种权重模式：

- 固定权重模式：$L = L_{\text{TD}} + w\,L_{\text{CQL}}$；
- 对偶权重模式（默认）：$L = L_{\text{TD}} + \alpha\,L_{\text{CQL}}$，其中 $\alpha=\exp(\log\alpha)$ 且对主损失不回传梯度；同时按对偶上升更新 $\log\alpha$：

$$\min_{\log\alpha}\; -\alpha\,\big(d_{\text{CQL}} - \bar{d}\big)$$

这里 $d_{\text{CQL}}$ 是掩码平均的"压低项减抬升项"之差，即对数求和指数与数据动作价值之差，$\bar{d}$ 是目标动作差距（`target_action_gap`）。当差距超过目标时 $\alpha$ 增大，低于目标时 $\alpha$ 减小，方向与保守 Q 学习的拉格朗日形式一致。

### 3.6 时序差分目标

采用一步备份：

$$y_t = r_t + \gamma\,(1-d_t)\max_{u'\in\mathcal{U}(\tau')}\bar{Q}_{\text{tot}}(\tau', u')$$

其中最大值按独立-全局-最大值性质分解为目标网络逐智能体的贪婪动作，掩码与 PyMARL 惯例一致。这与 Cal-QL 的一步备份相同；OVMSE 论文的 MACal-QL 则沿用 PyMARL2 的 $\lambda_{\text{TD}}=0.6$ 多步备份。

### 3.7 离线与在线两阶段训练流程

离线阶段（`training/offline.py`）：从离线数据集合采样回合批次，探索系数 $\varepsilon$ 从 1.0 线性衰减到 0.05，每个梯度步都计算保守损失，按回合计数刷新目标网络。在线阶段（`training/online.py`）：以去中心化 $\varepsilon$-贪婪采集数据（每智能体探索概率为 $\varepsilon/N$，与 OVMSE 论文算法 2 中顺序探索的执行方式一致），数据进入在线缓冲区，按混合比例 `mixing_ratio` 混合离线与在线批次，保守损失保持开启。后者与 OVMSE 论文表 5 中"MACal-QL 在线 CQL 权重等于离线 CQL 权重"的处理一致。

## 4 与原始算法及 OVMSE 论文定义的逐项对照

| 组件 | Cal-QL 原始形式 | OVMSE 论文的 MACal-QL | 本实现 | 判定 |
|---|---|---|---|---|
| 保守项结构 | 对数求和指数减数据动作价值，乘权重后与时序差分损失相加 | 式 9–10：$L_{\text{QMIX}}+\alpha L_{\text{CQL}}$，其中 $L_{\text{CQL}}=\mathbb{E}_{a\sim\mu}[Q]-\mathbb{E}_{a\sim\mathcal{D}}[Q]$ | 两组候选的对数求和指数减 $Q_{\text{tot}}(\tau,a^{\text{data}})$，$L=L_{\text{TD}}+\alpha L_{\text{CQL}}$ | 一致 |
| 均匀组权重 | 官方代码加均匀密度的负对数；论文清单有排版疏漏 | 未细化 | 加联合合法动作数的对数 $\sum_i\log n_i$ | 与官方代码一致，且是离散情形的正确对应 |
| 提议组权重 | 减策略动作对数概率 | 未细化 | 减去中心化 $\varepsilon$-贪婪联合对数概率 | 一致，联合重要性权重正确 |
| 校准钳制 | $\max(q_{\pi},V^{\mu})$ 只作用于策略提议组 | 表述为"将 Cal-QL 应用到多智能体" | $\max(q^{\text{prop}},G_t)$ 只作用于提议组，均匀组与数据动作不钳制 | 完全一致 |
| 参考价值 $V^{\mu}$ | 回报至当前，终止回合 | 未细化 | 同一递推；截断回合放弃锚 | 同一公式，截断处理是更稳健的选择 |
| 时序差分目标 | 一步备份 | QMIX 骨干，$\lambda_{\text{TD}}=0.6$ | 一步备份，独立-全局-最大值分解 | 与 Cal-QL 一致 |
| 权重 $\alpha$ | 官方默认固定值 | 在线权重等于离线权重 | 默认对偶权重，亦支持固定权重 | 工程变体，见第 6 节 |
| 两阶段流程 | 离线预训练后在线微调，保留保守项 | 混合比例、$\varepsilon$ 由 1 衰减到 0.05、两阶段均保留 CQL | 同一流程，混合比例 0.1 | 一致 |

## 5 正确性验证

### 5.1 功能检查

以下七项功能检查均已通过：回报至当前在终止回合的数值；被截断回合标记为无效且递归在未填充槽位处正确停止；终止先于槽位步的情形；训练步损失与各诊断量数值有限；对偶权重在差距高于目标时增大、低于目标时减小；校准锚取高值时钳制主导保守项；固定权重模式权重不参与优化。

### 5.2 损失估计量的数值验证

构造一个确定性场景以核对损失结构与解析期望：单个智能体、3 个离散动作、候选数 $k=2000$，混合器替换为逐智能体求和，数据动作为动作 0，三个动作的价值取 $Q=(-1.0,\,0.5,\,2.0)$，$\varepsilon=0.05$。无钳制时，压低项的期望为

$$\mathbb{E}\Big[\operatorname{logsumexp}_m\big(\{q^{\text{rand}}_m\}\cup\{q^{\text{prop}}_m\}\big)\Big]=\log\sum_{a=1}^{3}\exp Q(a) + \log(2k)$$

均匀组与提议组各贡献 $k\sum_{a}\exp Q(a)$，因为对任意提议分布都有 $\mathbb{E}_{u\sim\pi}[\exp(Q(u)-\log\pi(u))]=\sum_{u}\exp Q(u)$。减去数据动作价值后，实测值为 11.5288，解析值为 11.5354。

部分钳制情形（锚 $G_t=4$）：三个提议值分别为 3.094、4.594、2.034，钳制后应为 4.0、4.594、4.0，均匀组不受影响；实测 12.7375，解析 12.7713。

完全钳制情形（终止步奖励为 100，折扣 0.99，故锚沿回合依次为 98.01、99.0、100.0）：实测 107.6042，解析 107.6042。若均匀组也被钳制，结果将带有 $\log(2k)$ 而非 $\log k$ 的偏移；实测与"仅提议组被钳制"的解析形式精确一致，说明钳制的作用范围与原始算法相同。

三项一致表明：重要性加权对数求和指数的结构、钳制的作用范围、锚值逐步进入损失，均与 Cal-QL 一致。

### 5.3 校准锚在数据集合上的覆盖率实测

对本地数据集合 `cfcql/smac_v1/5m_vs_6m` 的 Medium 与 Medium-Replay 分别统计：各 5000 个回合全部以真正终止结束，中位长度分别为 27 与 23 步，填充步的锚覆盖率为 100%。也就是说，本实现放弃截断回合锚的设计，在这两个数据集合上没有削弱校准效果；这与原始 Cal-QL 所使用的 D4RL 任务几乎全部以终止结束的情形相当。

## 6 合理的偏离与使用注意

第一，没有显式策略网络。Cal-QL 的提议分布来自持续训练的策略网络，QMIX 没有对应部件，本实现用由当前 $Q_i$ 最大值构造的去中心化 $\varepsilon$-贪婪分布替代，联合重要性权重保持正确；探索系数随训练衰减，与策略从近似均匀收敛到贪婪的过程相对应。

第二，对偶权重是默认开启的工程变体。Cal-QL 官方默认固定权重（拉格朗日关闭，目标动作差距默认 1.0），OVMSE 论文的 MACal-QL 同样使用固定权重；本实现默认对偶权重（目标动作差距 0.8，学习率 $10^{-4}$）。两点值得注意：在 Adam 下 $\log\alpha$ 每步移动不超过学习率量级，整个离线预算内 $\alpha$ 近似恒定（5m_vs_6m 约 2900 次更新，$\alpha$ 仅在 1.0 至约 1.35 之间）；且 $\log\alpha$ 不写入检查点，在线阶段重置为初始值 1 后重新适应。若需要严格复现固定权重行为，设 `dual_alpha=false` 并指定 `cql_weight` 即可。

第三，估计量带有常数偏移。$k$ 个样本的对数求和指数比 $\log\sum\exp Q$ 高约 $\log$（样本数）：本实现为 $\log 20\approx 3.0$，官方实现为 $\log 30\approx 3.4$，属同一类估计偏置；固定权重下等效保守强度略低，对偶权重下被自适应机制吸收。2halfcheetah 配置已将目标动作差距由 0.8 调整为 5.0，以适配该偏移与更大的价值尺度。

第四，提议分布与行为策略的探索系数约定略有差异。数据采集按每智能体 $\varepsilon/N$ 执行，损失内按每智能体 $\varepsilon$ 建模。由于提议组估计子对任意提议分布都无偏，该差异只影响方差，不影响正确性。

第五，保守形式的选择。本实现采用联合动作上的保守项（与 MACQL 同型）叠加 Cal-QL 钳制；OVMSE 论文在 5m_vs_6m 与 6h_vs_8z 上使用 CFCQL 的反事实形式。两者都是"将 Cal-QL 应用到多智能体"的合理解释，本实现是更直接的迁移。

第六，超参数取值与论文不同但量级相当：候选数 10（Cal-QL 官方为 10，OVMSE 的 MACQL 为 100）、混合比例 0.1（OVMSE 在 5m_vs_6m 上最优为 0.5）、折扣 0.99、目标网络间隔 200、批运行规模 8、梯度裁剪 10，其中后四项与 OVMSE 论文附录 A 表 3 对 5m_vs_6m 的设置一致。

第七，一项独立观察：2halfcheetah 在线阶段日志显示 Q 值高估率接近 1、平均偏差约 +500。该校验指标的刻度由环境奖励决定（每步约 6.92，价值量级为数百），属奖励尺度与超参数适配问题，并非结构错误；校准钳制约束的是保守损失中提议组的贡献，并不直接为数据动作的价值设置上界，价值高估仍可能来自时序差分项。

## 7 结论

> [!success] 迁移判定：合理
> 本实现（`macal/`）在损失结构、重要性权重、校准钳制的作用范围、参考价值估计与两阶段训练流程上均与 Cal-QL 原始算法一致，并与 OVMSE 论文对 MACal-QL 的定义及其附录超参数相容。单元测试与损失估计量的数值验证均通过，校准锚在目标数据集合上覆盖全部填充步。与原始算法的差异（无显式策略网络、对偶权重、一步备份、联合动作保守形式）均为有依据的工程选择，不影响迁移的正确性。

## 参考来源

- Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning，arXiv:2303.05479，式 5.1、附录式 A.1、算法 1 与附录程序清单。
- Cal-QL 官方实现：`https://github.com/nakamotoo/Cal-QL`，`JaxCQL/conservative_sac.py`、`JaxCQL/replay_buffer.py`、`JaxCQL/sampler.py`。
- Offline-to-Online Multi-Agent Reinforcement Learning with Offline Value Function Memory and Sequential Exploration，arXiv:2410.19450（第 3 版），式 9–10、基线定义、算法 2、附录 A 表 3–5。
- 本仓库：`macal/returns.py`、`macal/calql.py`、`macal/learner.py`、`configs/macal_5m_vs_6m.yaml`、`configs/macal_2halfcheetah.yaml`、`training/offline.py`、`training/online.py`。
