# 联合行为提议的有限交接（Finite Handover of Joint Proposals, FHJP）

> 名称暂定。这是一份算法规范：公式 + 可直接接入现有 QMIX 在线流程的实现步骤。
> 本文档取代 `joint_order_rehearsal_v1.md` 的序数回放路线；不包含离线数据筛选、价值锚定、排序损失。
>
> **定稿口径：训练阶段用随机化联合提议产生数据，版本末用两条独立的在轨评估臂判断离线提议是否还有价值。** 前一版曾用行为倾向概率做逆概率加权估计，因确定性目标策略下的支撑退化而废弃，理由见 5.4；本版不计算、不存储、不使用任何倾向概率。

## 0. 一句话定义

**训练阶段。** 离线策略与当前在线策略在**决策冲突处**提供两个完整联合动作，以已知概率 $0.5$ 随机执行其中一个，用产生的真实环境轨迹执行**普通在线更新**；不计算、不存储、不使用行为倾向概率。

**版本判定。** 每个在线训练版本末，冻结当前在线参数，用**共享环境种子的两条独立在轨评估臂**（一条跑离线贪心策略，一条跑在线贪心策略，均 $\varepsilon=0$）估计离线联合行为的剩余价值；一旦不再具有可测正收益，离线提议器**永久退出**，算法严格退化为普通在线 MARL。

> 训练阶段用随机化联合提议产生数据，版本末用两条独立的在轨评估臂判断离线提议是否还有价值。

---

## 1. 符号

| 符号 | 含义 |
|---|---|
| $N$ | 智能体数；$a_t^i\in\mathcal A_i$，$\mathbf a_t=(a_t^1,\dots,a_t^N)$ |
| $h_t$ | 联合历史（各 agent 观测、上一步动作、时间步） |
| $r_t$ | 团队奖励；$\gamma$ 折扣；$T_m^{\mathrm{off}},T_m^{\mathrm{on}}$ 为第 $m$ 对两条评估臂的长度（可不同） |
| $\pi^{\mathrm{off}}$ | 离线贪心策略（确定性），由离线 Q 的联合贪心给出；既做提议候选，也做离线评估臂 |
| $\pi_k^{\mathrm{greedy}}$ | 在线贪心策略（$\varepsilon=0$，确定性，即可部署策略）；既做提议候选，也做在线评估臂 |
| $p$ | 提议随机化概率，$p=0.5$，固定不学习 |
| $z_t,\delta_t$ | 候选选择指示、冲突指示 |
| $s_m$ | 第 $m$ 对评估轨迹共享的环境种子 |
| $G_m^{\mathrm{off}},G_m^{\mathrm{on}}$ | 第 $m$ 对中两条评估臂的折扣回报 |
| $\Delta_m,\widehat\tau_v$ | 配对差及其版本均值 |
| $Q_{\mathrm{tot}},\theta,\theta^-$ | 在线混合价值网络、在线参数、目标网络参数 |
| $\mathcal D_{\mathrm{on}}$ | 在线 replay，只含新采集轨迹 |
| $v$ | 在线训练版本编号；$M$ 每版本评估轨迹数 |

---

## 2. 决策单位：完整联合动作，不是逐 agent 贡献

已有结果表明单体贡献不可加、pair 排名不可跨版本复用。因此最小决策单位是两个**完整**联合动作之间的选择：

$$
\mathbf a_t^{\mathrm{off}}=(a_t^{1,\mathrm{off}},\dots,a_t^{N,\mathrm{off}})
\quad\text{与}\quad
\mathbf a_t^{\mathrm{on}}=(a_t^{1,\mathrm{on}},\dots,a_t^{N,\mathrm{on}}).
$$

对 QMIX，两者分别由各 agent 的离线 Q 与在线 Q 取贪心动作得到：

$$
\mathbf a_t^{\mathrm{off}}
=\Big(\arg\max_{a\in\mathcal A_i}Q_i^{\mathrm{off}}(h_t,a)\Big)_{i=1}^{N},
\qquad
\mathbf a_t^{\mathrm{on}}
=\Big(\arg\max_{a\in\mathcal A_i}Q_i^{\mathrm{on}}(h_t,a;\theta)\Big)_{i=1}^{N}.
$$

由于 QMIX 的 mixer 对每个 $Q_i$ 单调，逐 agent 贪心即 $Q_{\mathrm{tot}}$ 意义下的联合贪心，因此上式不需要在联合动作空间上搜索。

离线 Q 在这里**只确定动作排序**，其数值不进入任何训练目标。

---

## 3. 第一部分：联合行为提议

### 3.1 冲突判定与随机执行

$$
\delta_t=\mathbb 1\!\left[\mathbf a_t^{\mathrm{off}}\neq \mathbf a_t^{\mathrm{on}}\right].
$$

$\delta_t=0$ 时两个候选相同，无需选择。$\delta_t=1$ 时以固定概率 $p$ 执行其中一个：

$$
\mathbf a_t^{c}=
\begin{cases}
\mathbf a_t^{\mathrm{off}}, & z_t=1,\\[2pt]
\mathbf a_t^{\mathrm{on}}, & z_t=0,
\end{cases}
\qquad
z_t\sim\operatorname{Bernoulli}(p),\qquad p=0.5.
$$

这里的 $p$ **不是可学习权重**，而是沿用随机化比较协议，使两个联合行为候选具有已知、相等的被选择概率，从而在线联合动作始终作为对照分支存在。

### 3.2 探索与诊断记录：不需要倾向概率

选定候选 $\mathbf a_t^{c}$ 之后，照常施加现有在线流程的探索（`select_actions_eval` 中的 $\varepsilon$-greedy 或顺序探索），得到最终执行动作 $\mathbf a_t$。探索机制本身不做任何修改。

本算法**不计算、不存储、不使用行为倾向概率**。原因有两条：

1. QMIX 的 TD 更新本来就是离策略更新，普通 TD 损失可以直接消费这些转移，不需要重要性修正；
2. 版本末的退出判定使用在轨评估臂，不使用逆概率加权（见 5.4），因此没有任何地方需要倾向概率。

每条训练步只需要记录两个诊断量：

$$
\mathrm{source}_{t}\in\{\mathrm{none},\mathrm{offline},\mathrm{online}\},
\qquad
\mathrm{conflict}_{t}=\delta_t.
$$

它们只进日志，用于回答"提议器是否还在被使用、离线候选是否还与在线候选不同"，**不进入 replay，不进入任何损失**。

### 3.3 每条训练轨迹需要记录的量

$$
\mathcal T_m=\Big\{(h_{m,t},\ \mathbf a_{m,t},\ r_{m,t},\ h_{m,t+1})\Big\}_{t=0}^{T_m-1},
\qquad
\text{诊断量}\ \Big\{(\mathrm{source}_{m,t},\mathrm{conflict}_{m,t})\Big\}_{t=0}^{T_m-1}.
$$

replay 中只放前一组，诊断量可以只落盘不进 replay。

---

## 4. 第二部分：用新采集轨迹执行普通在线更新

所有新轨迹进入当前在线 replay $\mathcal D_{\mathrm{on}}$，**不混入原离线数据集**。QMIX 的 TD 损失保持原样：

$$
\mathcal L_{\mathrm{TD}}(\theta)
=
\mathbb E_{\mathcal D_{\mathrm{on}}\,,t}
\Big[\big(y_t-Q_{\mathrm{tot}}(h_t,\mathbf a_t;\theta)\big)^2\Big],
$$

$$
y_t
=
r_t+\gamma(1-d_t)\,
Q_{\mathrm{tot}}\!\left(h_{t+1},\mathbf a_{t+1}^{\ast};\theta^-\right),
$$

其中 $Q_{\mathrm{tot}}$ 与 $\theta^-$ 全部属于在线模型，目标动作按普通 QMIX 目标计算，并由 mixer 的单调性分解为逐 agent 贪心：

$$
\mathbf a_{t+1}^{\ast}
=
\Big(\arg\max_{a\in\mathcal A_i}Q_i^{\mathrm{on}}(h_{t+1},a;\theta^-)\Big)_{i=1}^{N}.
$$

整个算法中**不存在**以下对象：

$$
Q^{\mathrm{off}}\ \text{进入 TD 目标},\qquad
\lambda\mathcal L_{\mathrm{offline}},\qquad
\text{离线数据优先级 / 优先回放},\qquad
\text{额外的 pairwise ranking loss}.
$$

同样**不存在重要性修正项**：QMIX 的 TD 更新本来就是离策略更新，$0.5/0.5$ 提议随机化与后续探索产生的转移可以直接进入普通 TD 批次，不需要倾向概率，也不需要对损失做任何加权（见 3.2）。

### 离线行为如何被"吸收"

若某个离线联合动作在当前队友与当前在线策略下仍产生较高回报，执行它得到的转移会被在线 TD 反复学习，其价值最终由在线 $Q$ 自己表示；若它不再有效，真实回报不会支持它，在线模型也不需要解除一个持续存在的离线约束。这是**吸收**，不是把离线参数继续叠加在在线模型上。

---

## 5. 第三部分：版本化联合价值差与永久退出

### 5.1 版本边界的评估协议

每个在线训练版本 $v$ 结束时，**冻结**当前在线参数 $\theta$，运行两条互相独立的在轨评估臂。对 $m=1,\dots,M$：

1. 取评估种子 $s_m$，各 $m$ 互相独立。
2. 用种子 $s_m$ 新建环境，从初始 RNN 隐状态开始，以 $\pi^{\mathrm{off}}$ 闭环跑完一局（$\varepsilon=0$，不施加 $0.5/0.5$ 提议随机化，不施加训练探索），得到折扣回报 $G_m^{\mathrm{off}}$。
3. 用**相同种子** $s_m$ 重新新建环境，从初始 RNN 隐状态开始，以冻结的 $\pi_k^{\mathrm{greedy}}$ 闭环跑完一局（同样 $\varepsilon=0$），得到折扣回报 $G_m^{\mathrm{on}}$。

$$
G_m^{\mathrm{off}}
=
\sum_{t=0}^{T_m^{\mathrm{off}}-1}\gamma^{t}r_{m,t}^{\mathrm{off}},
\qquad
G_m^{\mathrm{on}}
=
\sum_{t=0}^{T_m^{\mathrm{on}}-1}\gamma^{t}r_{m,t}^{\mathrm{on}}.
$$

两条评估臂都必须：

- 从各自的初始 RNN 隐状态开始闭环展开，用**各自上一步的动作**更新策略历史；
- 不施加 $0.5/0.5$ 提议随机化，不施加训练探索；
- 不写入在线 replay，不产生任何梯度更新；
- 计入算法的总环境交互预算（每版本 $2M$ 条轨迹）。

相同种子只在**同一版本、同一编号的两条评估轨迹之间**使用；两条轨迹的长度可以不同，不要求逐时间步对齐。

### 5.2 配对估计量与统计性质

$$
\Delta_m=G_m^{\mathrm{off}}-G_m^{\mathrm{on}},
\qquad
\widehat\tau_v=\frac{1}{M}\sum_{m=1}^{M}\Delta_m,
\qquad
s_v^{2}=\frac{1}{M-1}\sum_{m=1}^{M}\big(\Delta_m-\widehat\tau_v\big)^{2}.
$$

定义两个单臂估计 $\widehat J_v(\pi)=\frac{1}{M}\sum_{m=1}^{M}G_m^{\pi}$，离线联合行为的剩余价值为：

$$
\boxed{\;
\widehat\tau_v
=
\widehat J_v(\pi^{\mathrm{off}})
-
\widehat J_v(\pi_k^{\mathrm{greedy}})
\;}
$$

统计性质：

- 每条评估臂都在**自己的目标策略轨迹分布**上采样，两个单臂估计分别无偏：
  $$
  \mathbb E\big[G_m^{\mathrm{off}}\big]=J(\pi^{\mathrm{off}}),
  \qquad
  \mathbb E\big[G_m^{\mathrm{on}}\big]=J(\pi_k^{\mathrm{greedy}}).
  $$
- 作差仍然无偏：
  $$
  \mathbb E[\Delta_m]=J(\pi^{\mathrm{off}})-J(\pi_k^{\mathrm{greedy}})=\tau_v.
  $$
- 共享种子**只降低配对差值的方差，不改变估计目标**。两条臂因共享初始状态与环境随机性而正相关，故 $\operatorname{Var}(\Delta_m)$ 小于两臂独立时的方差。
- $\Delta_m$ 在不同 $m$ 之间独立同分布（各 $m$ 用独立种子），因此上面的均值与标准差构成合法的配对统计量。

它回答的不是"离线教师总体上好不好"，而是：

> 在当前在线版本已经发生变化以后，**完整离线联合行为**是否仍然比当前可部署的在线贪心联合行为带来更高的可实现团队回报。

### 5.3 退出判定与不可逆退出

$$
\text{退出}\iff
\Big(M\ge M_{\min}\Big)\ \wedge\
\Big(\underbrace{\widehat\tau_v+z_{1-\alpha}\,\frac{s_v}{\sqrt M}}_{\text{单侧上界}}\ \le\ 0\Big),
\qquad \alpha=0.05.
$$

即：当 $\widehat\tau_v$ 的 $95\%$ 单侧上界已经不大于 $0$，且评估样本量足够时，永久关闭离线提议器。之后训练完全回到普通在线 MARL：离线网络不再被读取，$0.5/0.5$ 提议随机化被移除，探索与 TD 更新与纯在线算法逐式相同。**不再重新开启**离线机制。

每个版本的 $\widehat\tau_v$ 只由该版本边界新采集的 $2M$ 条评估轨迹决定，不累计历史版本；历史版本的数值与显著性不进入后续决策。这落实了"组合价值随版本变化"，但不保存或复用不稳定的 agent/pair 排名。

### 5.4 为什么不使用逆概率加权

本算法的前一版用行为倾向概率 $\mu_k$ 做 IPW。选定在轨评估臂后，整套 IPW 机制被删除，理由是：

- **自归一化与权重截断不能恢复支撑。** 若长轨迹上目标策略的权重几乎全为零，自归一化只能重新分配已有的非零权重，截断只能限制极大权重，两者都无法产生缺失的轨迹；截断还会引入额外偏差，使判据从"离线行为已无可测正收益"退化为"截断后的估计没有正收益"。
- **有效支撑门槛不能完成判定。** 门槛能防止 $\widehat\tau_v\approx0$、$s_v\approx0$ 导致第一版本立即退出，但门槛通过后仍需要一个有效的 $\widehat\tau_v$；若持续没有足够支撑，算法只能一直保持提议器开启。它解决"不能错误退出"，没有解决"如何最终判断可以退出"。
- **在轨评估不需要分布修正。** 两个目标策略各自跑自己的轨迹，行为策略与目标策略之间的分布差异根本不存在，因此既没有支撑退化，也不需要倾向概率。这是三条中唯一同时保留判据统计有效性、实现简单性与明确退出语义的路线。

### 5.5 随机性与预算管理

- 评估种子与训练种子**分开管理**：评估使用独立 RNG（例如按版本派生的 `default_rng(eval_seed_base + v)`），避免评估过程改变后续训练的随机序列。
- 每版本 $2M$ 条评估轨迹计入总环境交互预算；预算紧张时应减小 $M$ 或增大 $L_v$，而不是把评估轨迹排除在预算之外。
- 冻结在线参数期间不做任何梯度更新；两条臂跑完后立即恢复训练。

---

## 6. 算法伪代码

```text
输入: 离线 checkpoint（冻结 πoff）, 在线 QMIX (θ), p=0.5,
      版本长度 L_v, 每臂评估轨迹数 M, M_min, α=0.05, 评估种子基 eval_seed_base

初始化:
    θ  ← 当前离线到在线流程的初始化（非本算法新增）
    D_on ← ∅                       # 只放新采集轨迹，不放离线数据
    proposer_on ← True

while proposer_on and t_env < T_max:
    # ---- 阶段一：采集并训练一个版本窗口 v ----
    for each env step in version v:
        h_t ← 当前联合观测
        a_off ← JointGreedy(offline_net, h_t)     # 离线 RNN 按自身历史闭环展开
        a_on  ← JointGreedy(online_net,  h_t)
        if a_off == a_on:
            c ← a_on;  source ← none;  conflict ← 0
        else:
            z ~ Bernoulli(p);  c ← a_off if z else a_on
            source ← offline if z else online;  conflict ← 1
        a_t ← Explore(c)                           # 现有探索机制，不做修改
        r_t, h_{t+1} ← env.step(a_t)
        D_on.add(h_t, a_t, r_t, h_{t+1})           # 不存倾向概率
        log(source, conflict)                      # 只进日志
        θ ← θ - η ∇ L_TD(sample(D_on))             # 原版 QMIX 更新
    # ---- 阶段二：版本末双在轨评估臂 ----
    冻结 θ
    rng ← default_rng(eval_seed_base + v)          # 评估种子独立于训练种子
    for m = 1, ..., M:
        s_m ← rng 产生的独立种子
        G_off[m] ← OnPolicyReturn(env.reset(seed=s_m), πoff,        ε=0)
        G_on[m]  ← OnPolicyReturn(env.reset(seed=s_m), θ_frozen,    ε=0)
    解冻 θ
    Δ ← G_off - G_on
    τ̂_v ← mean(Δ);  s ← std(Δ);  ub ← τ̂_v + z_{1-α}·s/√M
    log(τ̂_v, ub, conflict_rate, offline_select_rate, M)
    if M ≥ M_min and ub ≤ 0:
        proposer_on ← False                        # 不可逆

退出后: 执行普通纯在线 MARL（无提议随机化，无评估臂）
```

---

## 7. 接入现有 QMIX 流程的实现步骤

1. **加载冻结离线教师。** 复用 `ovmse/ovm.py::load_offline_for_ovm` 的加载路径得到 `offline_q_network`（连同 `offline_mixer` 一并冻结），但**不使用** `_ovm_td_loss` 与任何 OVM 目标混合。离线数据集不进入在线 replay。

2. **离线 RNN 改为闭环展开。** `qmix/policy.py::reset_eval_hidden` 已分配 `eval_offline_rnn`。需把 `_offline_q_and_greedy` 中 `last_actions[agent] = executed_action` 改为写回离线策略**自己**上一步的贪心动作，使 $\pi^{\mathrm{off}}$ 是闭环策略。现有"喂入执行动作"的口径是诊断用反事实展开，保留在 trace 字段中，不作为提议器输入。

3. **联合提议函数。** 新增 `select_actions_joint_proposal(...)`：在同一 $h_t$ 上取两个联合贪心动作；冲突时以 $p$ 抽样；返回 `actions`、`proposal_source ∈ {none, offline, online}`、`conflict`。

4. **不引入倾向概率。** replay 批次、损失、目标网络都不新增字段；QMIX 的 TD 更新直接消费这些离策略转移。诊断字段只进日志，不进 replay。

5. **采集侧零改动。** `training/rollout.py::collect_parallel` 的 `step` dict 不需要 `mu` 字段；如需诊断可追加 `source`/`conflict`，但不写入 replay。

6. **更新侧零改动。** `macal/learner.py` 的 `learn` 与损失、目标网络、混合网络完全不变。

7. **版本边界的双评估臂。** 在 online runner 加版本边界（每 $L_v$ 环境步，$L_v$ 待定）：用独立评估 RNG 生成 $M$ 个种子；对每个种子分别以 $\pi^{\mathrm{off}}$ 与 $\pi_k^{\mathrm{greedy}}$（均 $\varepsilon=0$、初始隐状态、各自闭环、从该种子重建环境）跑完整一局，取折扣回报；计算 $\Delta_m,\widehat\tau_v,s_v$ 与单侧上界；落盘 `tau_v`、`tau_v_ub`、`conflict_rate`、`offline_select_rate`、`n_eval_episodes_per_arm`。整个评估期间冻结 $\theta$、不写 replay、不做梯度更新。满足退出条件后置 `proposer_on=False`，提议随机化与评估臂都关闭，并可卸载离线网络权重。

8. **配置项。** `proposer.enabled / proposer.p / proposer.version_len / proposer.eval_episodes_per_arm / proposer.min_eval_episodes / proposer.alpha / proposer.eval_seed_base`。

9. **回归检查。** `offline_select_rate` 只在**冲突决策集合内**统计时才应接近 $p$（即 $\mathbb E[z_t\mid\delta_t=1]=p$），必须与 `conflict_rate` 一起记录。两条评估臂在相同种子与相同参数下应可复现。退出后不再读取离线网络，训练方程与纯在线算法逐式相同；但此时在线参数、replay、优化器状态与随机数状态都已受到过渡期影响，因此最终回报与训练曲线**允许**与纯在线基线不同，不以逐点重合作为验收标准。退出决策在固定种子下可复现。

---

## 8. 训练流程（压缩为六行）

```text
初始化：加载离线策略与在线网络，离线数据不进入在线 replay
for 每个在线训练版本 v:
    用 πoff 与当前 πgreedy 构造联合动作提议
    在冲突决策处以 0.5 / 0.5 执行其中一个联合动作
    将新轨迹写入在线 replay，执行原始在线 TD 更新
    用共享种子的两条在轨评估臂（ε=0）估计 τv
    若离线联合行为已无可测正收益，则永久退出过渡期
退出后：执行普通纯在线 MARL
```

---

## 9. 参数：算法确定与实现待定

**由算法确定（不是可调超参数）**

| 参数 | 取值 | 来源 |
|---|---|---|
| $p$ | $0.5$ | 随机化比较协议，使两个联合候选被选概率已知且相等 |
| 目标策略 | $\pi^{\mathrm{off}}$ 与 $\pi_k^{\mathrm{greedy}}$，均 $\varepsilon=0$ | 退出判定比较的是可部署策略 |
| 评估协议 | 共享环境种子的两条独立在轨臂，各 $M$ 条 | 无 IPW、无倾向概率、无支撑退化 |
| 判定数据 | 仅当前版本边界新采集的 $2M$ 条轨迹 | 不累计历史版本 |
| 退出 | 不可逆 | 退出后不重启离线机制 |

**实现待定（本规范不设定默认值）**

| 参数 | 说明 |
|---|---|
| $L_v$ | 一个在线训练版本的长度。**尤其不能未经论证地取 $10^6$**：现有 5m_vs_6m 在线预算为 $3\times10^6$ 步、2s3z 为 $1\times10^6$ 步，$L_v=10^6$ 意味着整个在线阶段只有一个版本边界，"版本化重新判定"与"有限交接"实际上不会发生。$L_v$ 应取在线总预算的一个较小比例，使训练期内出现多次版本边界；具体值待定。 |
| $M$ | 每臂评估轨迹数；每版本总成本为 $2M$ 条环境轨迹，计入总交互预算。 |
| $M_{\min}$ | 退出判定所需最少评估轨迹数。 |
| $\alpha$ | 单侧置信水平（常规取 $0.05$，$z_{1-\alpha}=1.645$）。 |
| $\gamma$（评估 $\tau_v$） | 评估回报折扣，与训练一致。 |
| `eval_seed_base` | 评估种子基，须与训练种子流分离。 |

---

## 10. 与 Sim2O 的关键区别

| | Sim2O | 本算法 |
|---|---|---|
| 候选来源 | 逐 agent 从 $\{a_\beta^i,a_\theta^i\}$ 组合，共 $2^N$ 个 | 只有两个**完整**联合动作候选：$\mathbf a^{\mathrm{off}}$、$\mathbf a^{\mathrm{on}}$ |
| 候选筛选 | 用当前集中 mixer $Q_{\mathrm{tot}}$ 打 Boltzmann 分 + beam 剪枝 | 不用 mixer 判断离线知识是否可信；$0.5/0.5$ 随机执行 |
| 可信度判定 | mixer 数值（本身正在被在线重建） | 共享环境种子的两条在轨评估臂（均 $\varepsilon=0$）的配对回报差 $\widehat\tau_v$ |
| 估计量 | 无（推理期组合搜索） | 每版本 $2M$ 条在轨轨迹，配对无偏，无需倾向概率 |
| 离线数据 | 与在线数据按 $\rho$ 混合回放 | 完全不回放离线数据 |
| 离线策略寿命 | 全程冻结存在，作为永久动作来源 | 具有不可逆完成条件，$\widehat\tau_v$ 失去可测正值后整体退出 |

其中第 2、3 行是本质差异：当前 mixer 正处于在线重建过程中，用它给离线候选打分等于让被检验的对象执行检验。本算法把检验交给真实环境回报，并且用两条各自在轨的评估臂来读这个回报，而不是用行为倾向概率反推。

---

## 11. 实验结论到设计落点

| 实验结论 | 在算法中的具体落点 |
|---|---|
| 离线价值依赖队友和在线版本 | 每个版本重新计算 $\widehat\tau_v$，不继承历史结论 |
| 单体贡献不可加，pair 排名不可复用 | 不学习 agent 级离线权重，直接比较完整联合行为 |
| 联合动作排序比数值更可利用 | 离线 Q 只生成联合贪心动作，不进入 TD 目标 |
| TD、density、return 不能替代数据真实效用 | 不设置离线数据排序与优先回放，取消固定离线回放 |
| 必须保留随机或在轨对照 | 训练时在线联合动作始终是 $0.5$ 概率的对照分支；版本末另有在线贪心评估臂作为对照 |
| 离线知识可能只在过渡阶段有益 | $\widehat\tau_v$ 失去可测正值后，离线提议器整体退出 |

关于"TD、density、return 不能替代数据真实效用"这一行需要特别说明：Insight E **不是**必须通过一个"数据效用预测网络"才算被使用。它当前最强的结论是现有代理指标不能承担数据选择，因此算法直接取消固定离线回放、让真实回报决定哪些行为被保留，是更符合证据的设计。

---

## 12. 适用范围与不做什么

**适用范围。** QMIX、VDN、QTRAN 等离策略值分解方法：新采集轨迹可直接进入原有更新，本算法不要求任何倾向概率。Actor-critic：需要与行为策略相容的离策略修正（例如 V-trace 或截断重要性比），此时倾向概率是**该方法自身**的需要，不是本算法退出判据的需要；版本末的两条评估臂对 actor-critic 同样适用。**未经修正的 on-policy PPO / MAPPO 不能直接消费这些轨迹**，因此不能把这套行为生成过程不加修改地套到 MAPPO 上。

**明确不做。** 不做离线数据优先级与优先回放；不把 $Q^{\mathrm{off}}$ 数值写入 TD 目标或价值锚定；不加 pairwise ranking loss；不学 agent 级或 pair 级离线权重；不计算、不使用行为倾向概率；不做逆概率加权估计；退出后不重启离线提议器；不维护跨版本的排名缓存。
