# 联合协调支撑上的序数回放（v1）

这是显式序数回放的记录。后续算法不再沿用这份损失。

离线到在线的多智能体强化学习里，值得复用的离线知识是联合协调支撑上的排序，不是联合价值的绝对数值，也不是单智能体的边缘动作分布。排序只在离线数据真正访问过的状态上使用。在线转移只做时序差分。

## 支撑集

离线数据中至少出现过一次的联合动作记为 \(\mathcal{U}\)。对离线状态 \(s\)，邻域 \(\mathcal{N}(s)\subset\mathcal{U}\) 以该状态下冻结网络的贪心联合动作和数据中实际执行的联合动作为锚，保留汉明距离不超过 \(d\)、且属于 \(\mathcal{U}\) 的联合动作，至多 \(K\) 个。

比较对 \(\mathcal{P}(s)\) 由 \(\mathcal{N}(s)\) 中两个不同的联合动作组成，并且

\[
\big|\bar{Q}(s,\mathbf{u})-\bar{Q}(s,\mathbf{v})\big|\ge\mu.
\]

\(\bar{Q}\) 是离线检查点的冻结副本。它只在这些离线状态上提供符号

\[
\sigma(s,\mathbf{u},\mathbf{v})=\operatorname{sign}\big(\bar{Q}(s,\mathbf{u})-\bar{Q}(s,\mathbf{v})\big).
\]

\(\sigma\) 是联合价值的高低，不拆成各智能体效用的边缘排序。\(\bar{Q}\) 的数值不作为回归目标。

## 在线时序差分

在线数据 \(\mathcal{D}_{\mathrm{on}}\) 上，目标动作由目标网络自己的贪心给出，不读取 \(\bar{Q}\)：

\[
y_t=r_t+\gamma(1-d_t)\max_{\mathbf{a}'}Q_{\theta^-}(s_{t+1},\mathbf{a}').
\]

沿轨迹的 \(\lambda\)-回报把上面的一步自举写成

\[
v_t=\max_{\mathbf{a}}Q_{\theta^-}(s_t,\mathbf{a}),
\]

\[
G_t^{\lambda}=r_t+\gamma(1-d_t)\Big[(1-\lambda)\,v_{t+1}+\lambda\,G_{t+1}^{\lambda}\Big].
\]

\[
\mathcal{L}_{\mathrm{TD}}(\theta)=\mathbb{E}_{\mathcal{D}_{\mathrm{on}}}\Big[\big(Q_\theta(s_t,\mathbf{a}_t)-G_t^{\lambda}\big)^2\Big].
\]

\(Q_\theta\) 是可训练的单调值分解。\(Q_{\theta^-}\) 是它的目标网络，按固定的回合间隔从 \(\theta\) 复制。

## 离线序数回放

离线数据 \(\mathcal{D}_{\mathrm{off}}\) 上只比较联合顺序：

\[
\mathcal{L}_{\mathrm{ord}}(\theta)=\mathbb{E}_{s\sim\mathcal{D}_{\mathrm{off}}}\left[\frac{1}{|\mathcal{P}(s)|}\sum_{(\mathbf{u},\mathbf{v})\in\mathcal{P}(s)}\Big[\delta-\sigma(s,\mathbf{u},\mathbf{v})\big(Q_\theta(s,\mathbf{u})-Q_\theta(s,\mathbf{v})\big)\Big]_+\right].
\]

\(\mathcal{P}(s)\) 为空时，该状态的序数项为 0。

## 总损失

\[
\mathcal{L}(\theta)=\mathcal{L}_{\mathrm{TD}}(\theta)+\beta\,\mathcal{L}_{\mathrm{ord}}(\theta),\qquad \beta>0.
\]

\(\beta\) 不随时间变化。一次更新从 \(\mathcal{D}_{\mathrm{on}}\) 和 \(\mathcal{D}_{\mathrm{off}}\) 各取一个批次。在线批次不计算 \(\mathcal{L}_{\mathrm{ord}}\)，离线批次不计算 \(\mathcal{L}_{\mathrm{TD}}\)。

行为策略是 \(Q_\theta\) 的 \(\varepsilon\)-贪心。评估时 \(\varepsilon=0\)，动作由 \(Q_\theta\) 的贪心给出。

## 实验设置

两个任务都从对应地图离线阶段按回报保存的检查点起步，数据集为 Medium，\(\beta=0.1\)，\(\delta=0.1\)，\(\mu=0.1\)，\(K=8\)，\(d=2\)。

| | 5m_vs_6m | 2s3z |
|---|---|---|
| 在线环境步 | \(3\times 10^6\) | \(1\times 10^6\) |
| 并行环境数 | 8 | 2 |
| 梯度裁剪 | 10 | 1 |
| 检查点 | `checkpoints/ovmse_paper_5m_vs_6m/ovmse_paper_seed1_best.pt` | `checkpoints/ovmse_paper_2s3z/ovmse_paper_seed1_best.pt` |

实现见 `jor/learner.py`。UOT 仍是在线状态上的序数约束，代码在 `uot/`，本算法不修改它。
