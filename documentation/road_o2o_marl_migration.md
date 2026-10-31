# ROAD 迁移到 QMIX 在线微调框架（O2O MARL）

> 原文：*ROAD: Adaptive Data Mixing for Offline-to-Online Reinforcement Learning via Bi-Level Optimization*（Yang et al., arXiv:2605.14497）。
> 本文档是迁移规范：把单智能体 ROAD 改写为 MARL 版本，并给出可直接接入 `training/online.py` 的实现步骤、参数、测试和实验计划。
>
> 标注 **[待核对]** 的条目在原文 HTML 版中描述不完整，实现前需对照 PDF 正文/附录确认；标注 **[迁移设计]** 的条目是原文没有、为 MARL 场景新增的决定。

---

> 注：本文写作时以 FHJP 实现为基准。FHJP 已于 2026-10-31 移除，文中 `fhjp/` 路径、`configs/fhjp_*.yaml`、`scripts/launch_fhjp.sh` 与 `tests/test_fhjp.py` 仅作历史参考。

## 0. 一句话定义

**ROAD 把"每个 batch 中离线数据占多少"从固定超参数变成一个在线元决策。** 在线训练被切成若干 period，每个 period 开始时用滑动窗口 UCB 从候选集 $\Lambda$ 中选一个混合比例 $m^k$，并在整个 period 内固定使用。period 结束时，用当前 critic 计算替代奖励 $R_q=\Delta_{\text{off}}-\kappa\Delta_{\text{on}}$ 来更新 bandit。

迁移到本框架后只有两处变化：
- 每个 gradient step 的 `step_mix`（离线 episode 占比）由 ROAD 控制器给出；
- 每个 period 末多做两次无梯度 QMIX 前向，用来计算 $\Delta_{\text{off}}$ 和 $\Delta_{\text{on}}$。

离线阶段、learner、损失函数、mixer 都**不改**。

---

## 1. 原算法要点

### 1.1 动机

固定混合比例（RLPD / Cal-QL 的 50/50、固定 10%、线性递减、Balanced Replay）会导致两种失败：
- 离线占比过高，在线改进被拖住；
- 离线占比过低，出现 catastrophic unlearning，即微调初期性能塌陷。

原文进一步指出一个**目标错位**：内层 Q 学习最小化的是 Bellman 误差，但我们真正关心的是策略回报。所以混合比例 $g$ 应该按"对回报的影响"来选，而不是按 TD loss 来选。

### 1.2 双层形式

内层是在混合分布 $d_g^{\text{sample}}$ 上做 FQI：

$$
f^{k+1}\leftarrow\arg\min_{f\in\mathcal F}\ \mathbb E_{d_g^{\text{sample}}}\big[\mathcal Tf^k(s,a)-f(s,a)\big]^2
$$

外层选择混合方式 $g$，使由内层解导出的策略回报最大：

$$
\max_{g\in\mathcal G}\ J\Big(\pi\big(\arg\min_f\mathcal J^k_{\text{in}}(g,f)\big)\Big)
$$

外层目标 $J$ 无法直接求导，也不便在线评估。ROAD 做了两步近似：
- 用替代奖励 $R_q$ 代替 $J$；
- 把 $g$ 离散化为候选比例集 $\Lambda$，用 bandit 求解。

### 1.3 替代奖励

$$
\Delta_{\text{off}}=\mathbb E_{s\sim\mathcal D_{\text{off}},\,a\sim\pi_\theta}[Q_\phi(s,a)]-\mathbb E_{(s,a)\sim\mathcal D_{\text{off}}}[Q_\phi(s,a)]
$$

$$
\Delta_{\text{on}}=\mathbb E_{s\sim\mathcal D_{\text{on}},\,a\sim\pi_\theta}[Q_\phi(s,a)]-\mathbb E_{(s,a)\sim\mathcal D_{\text{on}}}[Q_\phi(s,a)]
$$

$$
R_q=\Delta_{\text{off}}-\kappa\,\Delta_{\text{on}}
$$

两项的含义：
- $\Delta_{\text{off}}$：当前策略相对离线行为策略的估计优势。它衡量在离线数据的"信任域"内，策略改进了多少。
- $\Delta_{\text{on}}$：在线数据近似 on-policy，所以真实的 $\Delta_{\text{on}}$ 应接近 0。若它偏大，说明 critic 高估了它"更想选"的动作，这是高估和遗忘的信号。
- $\kappa$ 权衡两者。原文 Table 2 的消融显示，$\kappa=0$ 和 $\kappa=\infty$ 都不如 $\kappa=1$。

### 1.4 滑动窗口 UCB

$$
m^k=\arg\max_{m\in\Lambda}\Big\{\bar R_{q,m}+\sqrt{\tfrac{c\log(k\wedge\tau)}{N_k(\tau,m)}}\Big\}
$$

- $N_k(\tau,m)$：最近 $\tau$ 个 period 内臂 $m$ 被选中的次数。
- $\bar R_{q,m}$：窗口内该臂 $R_q$ 的均值。
- 窗口用来应对非平稳性：critic 在变，同一个比例在不同阶段的效果也不同。

### 1.5 原文超参数

| 参数 | 值 |
|---|---|
| $\Lambda$（离线占比候选） | $\{0.1,0.2,0.3,0.4,0.5\}$ |
| $\kappa$ | 1 |
| $c$ | 2.0 |
| $\tau$ | 1000 个 period（Antmaze 另设） |
| 基座算法 | 以 IQL 为主；另测试 CQL、Cal-QL、PEX，即插即用 |
| period 长度 | "episode 或固定步数" **[待核对：具体步数、每 period 的梯度步数]** |
| $R_q$ 归一化 | **[待核对]**，HTML 版未见 |
| $R_q$ 取值方式 | 取 period 末的水平值，还是相对 period 初的增量 **[待核对]** |

---

## 2. 符号（MARL 版本）

| 符号 | 含义 |
|---|---|
| $N$ | 智能体数；联合动作 $\mathbf a_t=(a^1_t,\dots,a^N_t)$ |
| $\tau_t^i$ | agent $i$ 的局部历史，即 RNN 隐状态，输入为 obs + 上一步动作 + agent id |
| $Q^i_\theta(\tau^i_t,a^i)$ | 共享参数的 `MarlcRNN` 输出（`qmix/networks.py:14`） |
| $Q_{\text{tot}}(\boldsymbol\tau_t,\mathbf a_t,s_t)$ | 单调 `QMixer`（`qmix/networks.py:44`）对 $\{Q^i\}$ 的混合 |
| $\mathbf a^\star_t$ | 联合贪心动作。因为 mixer 单调，它等于逐 agent 在合法动作集上取 argmax |
| $\mathcal D_{\text{off}}$ | `EpisodeDataset`（vault 数据集），按整条 episode 采样 |
| $\mathcal D_{\text{on}}$ | `online_buffer`，即 `deque(maxlen=buffer_size)` |
| $m$ | 训练 batch 中**离线 episode** 的占比，与现有 `step_mix` 同义 |
| $k$ / $P$ | period 序号 / 每个 period 包含的梯度步数 |
| $e_t$ | 时间步 $t$ 是否有任一 agent 执行了 ε 探索动作 |

---

## 3. 迁移设计：从单智能体到 QMIX

### 3.1 什么保持不变

ROAD 只是外层数据调度器，内层"基座算法在混合数据上的更新"原样复用：

| 原文组件 | 本框架对应 |
|---|---|
| 离线预训练 $f^{\text{off}},\pi^{\text{off}}$ | 已有 OVMSE/CQL-QMIX 离线 checkpoint，例如 `checkpoints/ovmse_batch_20261014/<map>_offline/ovmse_paper_seed1_best.pt` |
| 在线 FQI 更新 | `PaperQMIXLearner.train_on_batch`（`ovmse/learner.py:124`），即 TD(λ) + 可选在线 CQL |
| 混合 batch | `training/online.py:534-556` 的 `step_mix` → `offline_bs/online_bs` → `merge_offline_online` |
| soft-greedy 行为策略 | ε-greedy 的 `select_actions_eval`（`qmix/policy.py`） |

### 3.2 联合动作的 argmax **[迁移设计]**

单智能体里，$a\sim\pi_\theta$ 来自 actor（IQL 的 AWR 策略）或 critic 的 argmax。QMIX 没有显式 actor，策略就是对 $Q_{\text{tot}}$ 的贪心。

- **求解方式。** 联合动作空间是 $\prod_i|\mathcal A_i|$，但 `QMixer` 对每个 $Q^i$ 单调，所以 $\arg\max_{\mathbf a}Q_{\text{tot}}=(\arg\max_{a^i}Q^i)_i$ 是精确解，不需要搜索。若以后换成非单调 mixer（例如 QPLEX 的 attention 部分或非单调 critic），这一步需要改成近似搜索，可参考 `sim2o/cbs.py`。
- **合法动作。** argmax 必须用 `avail_actions` 屏蔽非法动作，做法与 `ovmse/learner.py` 中 `mac_sel` 相同（填 `-9999999`）。
- **用贪心而非 soft-greedy。** 部署与评估都用 ε=0 的贪心策略，所以 $\Delta$ 中的 $\pi_\theta$ 取贪心策略。由此有一个有用的性质：

$$
\Delta_{\text{off}}\ge 0,\qquad \Delta_{\text{on}}\ge 0
$$

原因是数据中的合法动作的 $Q_{\text{tot}}$ 不会超过贪心动作的 $Q_{\text{tot}}$。$R_q$ 是两个非负量之差。

### 3.3 RNN / 部分可观测下的 $\Delta$ **[迁移设计]**

Q 依赖历史，而历史中包含上一步动作。对 $\mathbf a^\star_t$ 的评估采用**单步偏离口径**：历史 $\boldsymbol\tau_t$ 沿用数据中实际发生的动作，只在第 $t$ 步把数据动作替换为贪心动作。

这与 `train_on_batch` 计算 `target_max_q` 的方式一致：`_forward_qs` 用 batch 中的动作构造 `_augment_obs`。因此一次 `_forward_qs(batch)` 前向就能同时得到 $Q_{\text{tot}}(\mathbf a^\star_t)$ 和 $Q_{\text{tot}}(\mathbf a^{\mathcal D}_t)$，无需额外的 rollout。

$$
\widehat\Delta(\mathcal B)=\frac{\sum_{b,t} w_{b,t}\big[Q_{\text{tot}}(\boldsymbol\tau_{b,t},\mathbf a^\star_{b,t},s_{b,t})-Q_{\text{tot}}(\boldsymbol\tau_{b,t},\mathbf a^{\mathcal D}_{b,t},s_{b,t})\big]}{\sum_{b,t}w_{b,t}}
$$

$w_{b,t}$ 的定义：
- 对离线 batch，$w$ 就是训练用的 `mask`（`filled` 且未在终止之后）；
- 对在线 batch，$w$ 在此基础上再乘 $(1-e_{b,t})$，见 3.4。

两个 $Q_{\text{tot}}$ 都用**当前网络**（`q_network` + `mixer`），不用 target 网络，与原文的 $Q_\phi$ 一致。

### 3.4 探索对 $\Delta_{\text{on}}$ 的污染 **[迁移设计，必须处理]**

原文假设在线数据"近似 on-policy"，但本框架的在线数据来自 ε-greedy，而且现有配置 `eps_start=1.0` 会让初期数据接近均匀随机。对被探索的步，$Q_{\text{tot}}(\mathbf a^\star)-Q_{\text{tot}}(\mathbf a^{\mathcal D})$ 反映的是"随机动作有多差"，与高估无关。这会让 $\Delta_{\text{on}}$ 系统性偏大，并随 ε 衰减而漂移。

处理方式：

1. 在 `qmix/policy.py::select_actions_eval` 的探索分支（`explore_here=True`）记录一个 per-agent 的 `explored` 标志。由 `training/rollout.py::collect_parallel` 写入 step dict，再由 `stack_episodes` 堆成 `batch["explored"] [B,T,1]`，表示该步是否有任一 agent 探索。
2. 计算 $\Delta_{\text{on}}$ 时屏蔽 $e_t=1$ 的时间步。

屏蔽之后，$\Delta_{\text{on}}$ 的语义收敛为：**当前 critic 想选的联合动作与"当时的贪心动作"之间的价值差**，即策略漂移乘以价值差。对近期数据，它在 critic 稳定时接近 0；critic 高估某些新动作时它会增大。这正是 ROAD 想用来惩罚的信号。

`explored` 字段只用于诊断，不进入训练损失。离线数据没有这个字段，`merge_offline_online` 只合并两边都有的 key（`training/rollout.py:178`），因此不受影响。

### 3.5 episode 级采样与 transition 级占比 **[迁移设计]**

原文的 $m$ 是 transition 占比，本框架按整条 episode 采样。离线 episode 与在线 episode 的长度分布不同：Medium 数据中的胜负局长度不一，而在线策略变强后 episode 会变短。所以"episode 占比 $m$"并不等于"有效 transition 占比"。

- 臂的定义仍按 episode 占比，与现有 `mixing_ratio` 语义一致，避免改动 batch 构造。
- 额外记录有效 transition 占比：`road/eff_offline_frac = filled_off.sum() / filled_all.sum()`，用于事后解释。

### 3.6 period 与时间尺度 **[迁移设计]**

本框架的节奏是：每轮 `collect_parallel` 采集 `batch_size_run` 条 episode，然后做 1 次梯度步（`training/online.py:521`）。

以 `5m_vs_6m` 为例，`t_max=3M`，`batch_size_run=8`，在线 episode 平均长度约 25–40 步，所以整个训练约有 $10^4$ 量级的梯度步。照搬"1 个 episode = 1 个 period"或 $\tau=1000$ 都不合适：

- 每个 period 只有 1 个梯度步时，比例对 critic 的影响远小于 $R_q$ 的估计噪声，bandit 学不到东西；
- $\tau=1000$ 时，窗口几乎覆盖全程，失去非平稳适应能力。

建议（第 9 节列为待定参数）：
- **period = $P$ 个梯度步**，默认 $P=10$，约 80 条在线 episode、2–3k 环境步；
- **$\tau$ 以 period 计**，默认 $\tau=200$，约覆盖最近 2000 个梯度步；
- 全程约 $10^3$ 个 period，每个臂平均被尝试约 200 次。

### 3.7 $R_q$ 的尺度 **[迁移设计]**

原文 $c=2.0$ 隐含了 $R_q$ 的量级。SMAC 的回报被归一化到约 20，$Q_{\text{tot}}$ 的量级随地图和训练阶段变化，所以直接用原始 $R_q$ 时，$c$ 的含义会随地图漂移。默认做法：

$$
\tilde R_q^k=\frac{R_q^k-\mu_W}{\sigma_W+\epsilon}
$$

$\mu_W$ 和 $\sigma_W$ 是滑动窗口内所有臂的 $R_q$ 的均值和标准差，UCB 使用 $\tilde R_q$。消融版本可以用原始 $R_q$。日志同时记录原始值和归一化值。

### 3.8 和本框架既有机制的关系

| 机制 | 关系 |
|---|---|
| FSR（`fsr/learner.py::support_mix_ratio`） | 同样动态决定 `step_mix`，但依据是 flow 支撑度。**与 ROAD 互斥**，同时启用时直接报错 |
| FHJP（`fhjp/`） | 改变的是行为策略，ROAD 改变的是训练 batch，两者正交。v1 先**互斥**，避免实验因素混杂 |
| OVM（`transfer.mode: ovmse`） | OVM 改 TD 目标，ROAD 改数据。主实验用 `transfer.mode: full_td`，即纯 TD，对应原文的"基座 + ROAD"；OVM + ROAD 作为扩展 |
| 在线 CQL（`online_cql_weight`） | 对应原文的 CQL 基座。默认 0；设为大于 0 时，CQL 项只作用于同一混合 batch |
| PEXM（`training/pexm.py::ThompsonReturnSelector`） | 也是 bandit，但选择的是 episode 级行为策略，奖励是回报。实现可参考其写法，但不共用代码 |
| fork / resume（`fork_at` 等） | v1 不支持 ROAD 的 bandit 状态快照，同时启用时报错 |

---

## 4. 算法伪代码（MARL-ROAD）

```text
输入: 离线 checkpoint (θ, ψ) → q_network / mixer；离线数据 D_off；候选 Λ；κ, c, τ, P, B_Δ
初始化: online_buffer ← ∅；bandit ← SlidingWindowUCB(Λ, c, τ)；k ← 0；m ← bandit.select()

while t_env < t_max:
    episodes ← collect_parallel(envs, learner, ε(t_env))      # 每步记录 explored
    online_buffer.extend(episodes)
    if |online_buffer| < batch_size: continue

    # ---- 内层：基座算法在混合 batch 上更新（ratio 在 period 内固定）----
    offline_bs ← round(batch_size · m); online_bs ← batch_size − offline_bs
    batch ← merge(D_off.sample(offline_bs), online_buffer.sample(online_bs))
    learner.train_on_batch(batch, ...)
    steps_in_period += 1

    # ---- 外层：period 末更新 bandit ----
    if steps_in_period == P:
        with no_grad:
            B_off ← D_off.sample(B_Δ);  B_on ← online_buffer.sample(B_Δ)
            Δ_off ← masked_mean(Qtot(a*) − Qtot(a_D) ; B_off, mask)
            Δ_on  ← masked_mean(Qtot(a*) − Qtot(a_D) ; B_on,  mask·(1−explored))
        R_q ← Δ_off − κ·Δ_on
        bandit.update(m, normalize(R_q))
        k ← k+1; steps_in_period ← 0
        m ← bandit.select()               # 未被尝试过的臂优先；否则取 SW-UCB argmax
        log(road/*, force=True)
```

$\Delta$ 计算的 PyTorch 骨架如下。它只复用 learner 的已有方法，不引入新网络。

```python
@torch.no_grad()
def joint_advantage_gap(learner, batch, extra_mask=None):
    mac_out = learner._forward_qs(batch, target=False)              # [B,T,N,A]
    avail = batch["avail_actions"]
    greedy = mac_out.masked_fill(~avail, -9999999.0).argmax(dim=3, keepdim=True)
    q_star = torch.gather(mac_out, 3, greedy).squeeze(3)            # [B,T,N]
    q_data = torch.gather(mac_out, 3, batch["actions"].long()).squeeze(3)
    gap = learner.mixer(q_star, batch["state"]) - learner.mixer(q_data, batch["state"])
    mask = batch["filled"].float()
    mask[:, 1:] = mask[:, 1:] * (1.0 - batch["terminated"][:, :-1].float())
    if extra_mask is not None:
        mask = mask * extra_mask
    return float((gap * mask).sum() / mask.sum().clamp(min=1.0))
```

> 实现时需核对 `mixer` 的输入输出形状（`QMixer.forward(agent_qs [B,T,N], states [B,T,S]) -> [B,T,1]`）以及 mask 与 `train_on_batch` 是否完全一致。`train_on_batch` 用的是 `[:, :-1]` 切片，这里不需要 TD 目标，可以用整条序列，但填充槽必须被 `filled` 屏蔽。

---

## 5. 接入步骤（文件级）

### 5.1 新增包 `road/`

| 文件 | 内容 |
|---|---|
| `road/__init__.py` | 导出 `build_road_controller` |
| `road/bandit.py` | `SlidingWindowUCB(arms, c, tau, seed)`，方法有 `select() -> float`、`update(arm, reward)`、`state_dict()` / `load_state_dict()`。窗口为 `deque(maxlen=tau)`，元素是 `(arm, reward)`；先轮询未尝试的臂；同分时用固定 rng 打破平局 |
| `road/surrogate.py` | `joint_advantage_gap(learner, batch, extra_mask)`（见第 4 节）和 `surrogate_reward(d_off, d_on, kappa)` |
| `road/controller.py` | `RoadController`：属性 `ratio`；`after_train_step(offline_data, online_buffer, t_env) -> Optional[dict]` 计数，到 period 末计算 Δ、更新 bandit、切换比例，并返回日志行；`summary()` 给出各臂累计选择次数。`build_road_controller(cfg, learner)` 在 `road.enabled` 为假时返回 None，并在其中做互斥检查（FSR / proposer / fork） |

### 5.2 修改 `training/online.py`（预计改动 +30 行左右）

1. **构造**：在 FHJP 控制器构造之后（约 L387-416）调用 `road = build_road_controller(cfg, learner)`，并打印 Λ、κ、c、τ、P。
2. **比例**：在 `step_mix = mixing_ratio` 之后（L523）加入 `if road is not None: step_mix = road.ratio`。其后的 `offline_bs/online_bs` 逻辑不变，`step_mix > 0` 始终成立，因为 $\Lambda$ 中最小为 0.1。
3. **训练后钩子**：在 `logger.write(train_logs)` 之后（L594）加入：
   ```python
   if road is not None:
       road_row = road.after_train_step(offline_data, online_buffer, t_env)
       if road_row is not None:
           logger.write(road_row, force=True)
   ```
   **注意 RNG**：Δ 估计用的采样使用控制器自己的 `np.random.default_rng(seed + road.seed_offset)`，不能使用主循环的 `rng`。否则启用 ROAD 会改变训练 batch 的随机序列，与固定比例基线不可比。
4. **每步日志**：在 `train_logs` 中加入 `road/ratio` 和 `road/eff_offline_frac`（后者在构造 batch 时由 `filled` 计算）。
5. **checkpoint**：最终保存时（约 L650-661）在 `extra` 中加入 `"road": road.state_dict()`，内容为 bandit 窗口、各臂累计次数和 period 计数。

### 5.3 修改 rollout 记录 `explored`

- `qmix/policy.py::select_actions_eval`：`collect_parallel`（`training/rollout.py:47`）不传 `trace`，但会传入每条 episode 自己的 `state`，即 `new_episode_state()` 返回的 `{"rnn","last"}`。所以当 `state is not None` 时，把逐 agent 的 `explore_here` 写入 `state["explored"]`（`list[bool]`，按排序后的 agent 顺序）。FHJP 的 `JointProposalPolicy._explore`（`fhjp/proposer.py:217`）目前与 ROAD 互斥，暂不需要同步修改。
- `training/rollout.py::collect_parallel`：构造 step dict 时（约 L62）加入 `step["explored"] = np.asarray(states[i].pop("explored", [False]*len(agents)), dtype=bool)`。
- `training/rollout.py::stack_episodes`：若 step 中有 `explored`，堆叠为 `batch["explored"] [B,T,1]`，表示该步任一 agent 探索过；缺失时不生成该 key。
- `pv/`、`fhjp/` 以及 snapshot 读写都不受影响，因为新字段只是可选的。

### 5.4 配置

新增 `configs/road_5m_vs_6m.yaml` 和 `configs/road_2s3z.yaml`。基于 `configs/fhjp_*.yaml` 修改：
- 删除 `proposer`；
- 设置 `transfer.mode: full_td` 和 `system.mixing_ratio: 0.0`（该值被 ROAD 覆盖，仅作为保险）；
- 新增以下块：

```yaml
road:
  enabled: true
  arms: [0.1, 0.2, 0.3, 0.4, 0.5]   # 离线 episode 占比
  kappa: 1.0
  ucb_c: 2.0
  window: 200            # τ，单位 period
  period_steps: 10       # P，每 period 的梯度步数
  delta_batch: 32        # B_Δ，计算 Δ 的 episode 数（离线、在线各一份）
  normalize_reward: true
  mask_explored: true
  seed_offset: 7
```

### 5.5 启动脚本

复制 `scripts/launch_fhjp.sh` 为 `scripts/launch_road.sh`。保留 SEEDS/GPUS/MAPS/PLAN_ONLY 约定、共享离线 checkpoint 检查和"已完成即跳过"逻辑。跳过判定的文件名为 `ovmse_paper_online_full_td_seed<N>.pt`（learner 仍是 `PaperQMIXLearner`），输出目录改为 `checkpoints/road_<map>/seed<N>`。

### 5.6 测试 `tests/test_road.py`

沿用 `tests/test_fhjp.py` 的写法：只在 CPU 上运行，用假网络和假环境，并带 `__main__` 入口。

| 测试 | 断言 |
|---|---|
| bandit 先轮询 | 前 $|\Lambda|$ 次 `select()` 恰好各返回一个臂 |
| bandit 偏好高奖励 | 固定奖励（某臂最高）下，足够多轮后该臂的选择占比超过 0.5 |
| 滑动窗口遗忘 | 前 $\tau$ 轮臂 A 最优、之后臂 B 最优，再过 $\tau$ 轮后 B 成为主选 |
| Δ 非负 | 随机 Q 和随机合法数据动作下，`joint_advantage_gap >= 0` |
| Δ 为零 | 数据动作等于逐 agent 合法 argmax 时，`gap == 0` |
| 非法动作屏蔽 | 非法动作的 Q 极大时，argmax 不会选中它 |
| explored 屏蔽 | 只有被标记探索的步存在 gap 时，`mask_explored` 后 $\Delta_{\text{on}}=0$ |
| period 内比例固定 | 连续 $P$ 次 `after_train_step` 内 `ratio` 不变，第 $P$ 次返回日志行并可能切换 |
| RNG 隔离 | 启用 ROAD 前后，主循环 `rng` 抽取的 online 下标序列相同 |
| 互斥 | `road.enabled` 与 `learner_type: fsr`、`proposer.enabled`、`fork_at` 任一同时出现时报错 |

---

## 6. 日志字段

| 键 | 频率 | 含义 |
|---|---|---|
| `road/ratio` | 每梯度步 | 本步使用的离线 episode 占比 $m^k$ |
| `road/eff_offline_frac` | 每梯度步 | 有效离线 transition 占比 |
| `road/period` | period 末 | $k$ |
| `road/delta_off`, `road/delta_on` | period 末 | $\widehat\Delta_{\text{off}}$, $\widehat\Delta_{\text{on}}$ |
| `road/r_q`, `road/r_q_norm` | period 末 | 原始与归一化的 $R_q$ |
| `road/ucb_<m>`, `road/count_<m>` | period 末 | 各臂的 UCB 值与窗口内计数 |
| `road/explored_frac_on` | period 末 | $\Delta_{\text{on}}$ batch 中被屏蔽的时间步比例 |

`JsonlLogger` 只写入 `force=True` 的行，所以 period 行必须带 `force=True`。每梯度步的字段只进入 wandb；离线分析时用 period 行重建 $m^k$ 的时间序列即可。

---

## 7. 实验计划

**设置**
- 地图：`5m_vs_6m`（主图）和 `2s3z`。
- 种子：至少 3 个。
- 所有臂共享同一离线 checkpoint（`ovmse_batch_20261014/<map>_offline/ovmse_paper_seed1_best.pt`）。
- 其余参数与 `configs/fhjp_*.yaml` 一致：`t_max`、`batch_size=128`、`buffer_size`、ε 调度、`target_update_interval`。

**对照组**

| 组 | 配置 |
|---|---|
| Online-only | `mixing_ratio: 0.0`，与 FHJP 的基线相同 |
| Fixed-m | `mixing_ratio ∈ {0.1, 0.3, 0.5}` |
| Decreasing | 0.5 线性降到 0.1。需要一个小钩子，可复用 ROAD 控制器的"固定调度"模式 |
| ROAD | 本文默认配置 |
| ROAD κ=0 | 只看 $\Delta_{\text{off}}$ |
| ROAD κ=∞ | 只看 $-\Delta_{\text{on}}$，实现上取 `kappa: 1e6`，或令 $R_q=-\Delta_{\text{on}}$ |
| ROAD +0 臂 | $\Lambda=\{0,0.1,\dots,0.5\}$，检验 MARL 下是否会主动退到纯在线 |
| ROAD 无 explored 屏蔽 | 验证 3.4 节的必要性 |
| OVMSE（参考） | 现有 `configs/5m_vs_6m.yaml` 的在线阶段 |

**指标**
- 训练曲线：`evaluation/win_rate` 和 `evaluation/mean_episode_return`。
- 最终 640 局评估：`scripts/eval_640.py --episodes 640 --chunk 32`。
- 初期遗忘深度：前 200k 步内胜率的最低点相对 t=0 的下降。
- ROAD 机制诊断：$m^k$ 的时间序列与各臂的选择分布；$\Delta_{\text{off}}$、$\Delta_{\text{on}}$ 曲线；`q_bias`（`training/q_metrics.py`）与 $\Delta_{\text{on}}$ 的相关性，用来检验 $\Delta_{\text{on}}$ 能否作为高估代理。

**预期与判据**
- ROAD 的初期遗忘应不深于 Fixed-0.5。
- ROAD 的最终胜率应不低于 Online-only 和最佳 Fixed-m（配对种子差的 95% 置信区间）。
- 若 bandit 长期均匀分布，即各臂计数差异不显著，说明 $R_q$ 噪声大于臂间差异。此时应先调大 $P$ 和 $B_\Delta$，而不是调 $c$。

---

## 8. 风险与已知局限

1. **$R_q$ 噪声。** $\Delta$ 是两个 $Q_{\text{tot}}$ 之差的均值，样本量为 $B_\Delta$ 条 episode。臂之间的真实差异可能小于估计噪声。应先记录 $\widehat\Delta$ 的 bootstrap 标准差，再决定 $P$ 和 $B_\Delta$。
2. **信用滞后。** period $k$ 末的 critic 状态累积了此前所有臂的影响，$R_q^k$ 并不只归因于 $m^k$。原文同样存在这个问题，滑动窗口只能部分缓解。可以对比"水平值"与"相对 period 初的增量"两种 $R_q$（见 1.5 的 [待核对]）。
3. **Δ_off 被 CQL 预训练抬高。** CQL 会压低数据外的动作，离线 critic 在数据上的 $\mathbf a^{\mathcal D}$ 与 $\mathbf a^\star$ 往往重合，所以 $\Delta_{\text{off}}$ 初期接近 0，之后随在线更新上升。这符合"在信任域内改进"的语义，但量级变化很大，因此 3.7 的归一化是默认开启的。
4. **单调 mixer 的假设。** 精确的联合 argmax 依赖 QMIX 的单调性。迁移到 MAC-Flow（连续动作、VDN 双 critic）或 Sim2O（OMIGA 骨干）时，需要分别给出 $\mathbf a^\star$ 的定义，例如 flow 策略的采样，或 CBS 搜索出的动作。这些不在 v1 范围内。
5. **离散候选集。** 原文局限中提到的线性混合与离散搜索在这里同样存在。
6. **计算开销。** 每个 period 增加 2 次 $B_\Delta$ 条 episode 的无梯度 unroll，相对 $P$ 次训练步（每步 128 条 episode，前向加反向）的开销小于 10%。

---

## 9. 参数：原文给定与实现待定

| 参数 | 状态 | 值 / 候选 |
|---|---|---|
| $\Lambda$ | 原文 | $\{0.1,\dots,0.5\}$；扩展加入 0 |
| $\kappa$ | 原文 | 1 |
| $c$ | 原文 | 2.0（配合归一化的 $R_q$） |
| $\tau$ | 待定 | 默认 200 个 period；扫描 {50, 200, 1000} |
| $P$ | 待定 | 默认 10 个梯度步；扫描 {1, 10, 50} |
| $B_\Delta$ | 待定 | 默认 32 条 episode；扫描 {16, 32, 64} |
| $R_q$ 归一化 | 待定 | 窗口 z-score（默认）或原始值 |
| $R_q$ 取值方式 | 待核对原文 | period 末水平值（默认）或增量 |
| explored 屏蔽 | 迁移设计 | 默认开启 |
| 基座 | 迁移设计 | `full_td`（默认），以及 `online_cql_weight > 0` |

---

## 10. 范围与不做什么

- 不改离线阶段、learner、损失函数、网络结构和 checkpoint 格式（只在 `extra` 中追加字段）。
- 不做 per-agent 的混合比例。原文只有一个全局比例；per-agent 混合会破坏 episode 级的联合轨迹一致性，与 QMIX 的中心化训练不兼容。
- 不做连续化的 $g$（例如梯度型元学习），只实现原文的离散 bandit。
- v1 不与 FSR、FHJP、fork/resume 组合。
