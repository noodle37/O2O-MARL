# MaSO2：SO2 迁移到 QMIX 的 O2O MARL 实现

> 原文：*A Perspective of Q-value Estimation on Offline-to-Online Reinforcement Learning*（Zhang et al., arXiv:2312.07685）。方法名为 **SO2**（Smoothed Offline-to-Online）。
> 本文档说明怎样把单智能体、连续动作的 SO2 接到本仓库的离散 QMIX O2O 流程里。迁移后的方法记为 **MaSO2**。
>
> 代码包 `so2/`、配置 `configs/so2_*.yaml`、日志前缀 `so2/*`、启动脚本 `scripts/launch_so2.sh` 使用短名。
>
> 标注 **[迁移设计]** 的条目是原文没有、为了离散多智能体场景补上的决定。

## 0. 一句话定义

SO2 认为离线预训练留下的 Q 有偏差、排序也不准，在线微调因此又慢又不稳。它在 SAC + Q ensemble 上只加两件事：

1. **Perturbed Value Update (PVU)**：TD 目标里的下一步动作改成 \(a'+\epsilon\)，\(\epsilon\sim\mathrm{clip}(\mathcal N(0,\sigma),-c,c)\)。
2. **提高更新频率 \(N_{upc}\)**：每采集一次数据，对 Q（以及策略）连做 \(N_{upc}\) 次梯度更新。

MaSO2 把这两件事接到现有的在线 QMIX 上。离线阶段不重训，起点是各地图共用的 OVMSE checkpoint。其它 learner 的损失、网络和「每次采集更新 1 次」都不变。

## 1. 原文在做什么

在线阶段的伪代码（Algorithm 1）是：

1. 用当前策略与环境交互，把转移放进在线 buffer。
2. 重复 \(N_{upc}\) 次：
   - 从 \(\mathcal D^{\text{off}}\cup\mathcal D^{\text{on}}\) 抽一个 mini-batch；
   - 用扰动后的下一步动作算 TD 目标，更新每个 ensemble Q；
   - 用重参数化更新策略；
   - 对 target 网络做 Polyak 平均。

MuJoCo 默认 \(\sigma=0.3\)、\(c=0.6\)、\(N_{upc}=10\)。Q ensemble 的个数是底座（EDAC / SAC-\(N\)）的超参，不是 SO2 新加的项。原文在附录里把 PVU 和 \(N_{upc}\) 接到 OFF2ON、PEX 上，说明这两项可以换底座。

目标公式是

\[
\mathcal T Q(s,a)=r+\gamma\big(\hat Q(s',a'+\epsilon)-\beta\log\pi(a'\mid s')\big),\quad a'\sim\pi(\cdot\mid s').
\]

\(\beta\log\pi\) 是 SAC 的熵项。QMIX 没有随机 actor，MaSO2 不搬这一项。

## 2. 哪些东西保持原样

| 原文 | 本框架 |
|---|---|
| 离线预训练的 SAC-\(N\) | 已有 OVMSE QMIX checkpoint，`checkpoints/ovmse_batch_20261014/<map>_offline/ovmse_paper_seed1_best.pt` |
| 在线价值更新 | `PaperQMIXLearner.train_on_batch`：TD(\(\lambda\))，在线 CQL 权重为 0，不用 OVM |
| 行为策略 | 原有 \(\varepsilon\)-greedy 采集（`qmix/policy.py`） |
| target 网络 | 仍按 episode 间隔硬更新（2s3z 为 50，5m_vs_6m 为 200） |
| 其它算法 | `learner_type` 不是 `so2` 时不构造 `SO2Learner`，`n_upc` 默认为 1 |

`ovmse/learner.py` 里唯一的共享改动是把 bootstrap 动作从写死的 `argmax` 换成 `_select_bootstrap_actions`。基类实现仍是 `argmax`，返回的日志里不带 `so2/*`。ACKF、UOT、FSR、MaCal-QL、MAC-Flow、MaROAD 都不走这个覆盖。

## 3. 迁移设计

### 3.1 底座换成 QMIX，不引入 Q ensemble **[迁移设计]**

原文的 ensemble 是 SAC-\(N\) 的底座。本仓库的离线 checkpoint 是单个 `MarlcRNN` + `QMixer`。把同一个 checkpoint 复制成 10–50 份 ensemble 会改变参数量、优化器和评估路径，也没法和现有臂共用起点。

MaSO2 只迁移 SO2 自己的两项（PVU、\(N_{upc}\)）。这和原文附录里「SO2 可以叠到别的 O2O 算法上」的用法一致：底座用本框架的在线 QMIX（`transfer.mode: full_td`，关掉 OVM 和在线 CQL）。

### 3.2 离散 PVU **[迁移设计]**

连续动作里，每一步都会把 \(a'\) 挪开再读 Q，所以尖峰会被抹平。SMAC 的动作（移动、停止、攻击某个敌人）没有欧氏坐标，不能加 \(\mathcal N(0,\sigma)\)。

若把同样的噪声加到 Q 上再 argmax，尖峰反而最不容易被换掉：噪声尺度按 Q 的间隔归一化之后，换掉贪心动作的概率与间隔无关，而且在 \(\sigma=0.3,c=0.6\) 下这个概率接近 0。那会让 PVU 变成空操作。

因此 MaSO2 把 \(\sigma\) 读成**每个 agent 替换贪心 bootstrap 动作的概率**：

- 贪心动作仍是在线 Q 在合法动作上的 argmax（double Q，和原来一样）；
- 以概率 \(\sigma\) 换成「在其余合法动作上均匀抽一个」；只有一个合法动作时不替换；
- target 网络在这个动作上取值，再送进原来的 mixer 和 TD(\(\lambda\))。

\(\sigma=0\) 时与原来的 QMIX 目标逐元素相同。默认 \(\sigma=0.3\)。

\(c=0.6\) 在原文里是连续动作噪声的截断（动作大约在 \([-1,1]\)，这个截断已经覆盖大半个邻域）。离散动作没有这个球，MaSO2 **保留并记录** \(c\)，不用它去否决替换。否决大价值差会再次保护尖峰，和 PVU 的目的相反。日志字段是 `so2/clip`。

实现：`so2/perturb.py` 的 `perturbed_bootstrap_actions`，由 `SO2Learner._select_bootstrap_actions` 调用。扰动用 learner 自己的 CPU 随机数发生器，不碰采集和 batch 采样用的 NumPy RNG。

### 3.3 \(N_{upc}\) **[迁移设计]**

原文是「采集 1 条转移，更新 10 次」。本框架是「并行采 `batch_size_run` 条 episode，更新 1 次」。MaSO2 保持采集节奏，把更新次数改成每次采集后 \(N_{upc}\) 次，默认 10。每一次都重新抽 batch，对应 Algorithm 1 里循环内部的抽样。

10 次更新里，只有第 1 次传入 `episode_delta=batch_size_run`。后面 9 次传 0。这样 target 网络仍按「每 200（或 50）条 episode 硬更新」前进，不会因为 \(N_{upc}\) 被提前刷新。策略在 QMIX 里就是 Q 的贪心，所以这 10 次同时更新了价值和行为，对应正文里「Q 和策略一起提高频率」的做法。

`n_upc` 只写在 `SO2Learner` 上。`training/online.py` 用 `getattr(learner, "n_upc", 1)`，其它算法得到 1，循环体与改前的单次更新相同。ROAD、SAPR、MU-APR 的周期逻辑放在循环外面，避免被数成 10 个 period。SO2 与这三个控制器同时打开会直接报错。

### 3.4 离线 / 在线混合 **[迁移设计]**

原文从 \(\mathcal D^{\text{off}}\cup\mathcal D^{\text{on}}\) 均匀抽样。Vault 的 episode 数远大于在线 buffer（上限 5000），按容量均匀抽时在线样本几乎进不了 batch。

MaSO2 用固定的 **episode 混合比例 `mixing_ratio: 0.5`**：每个 mini-batch 一半离线 episode、一半在线 episode。这是「两边都进更新」的直接实现，没有再加 OFF2ON 那种 online-ness 重加权。有效 transition 占比仍会随两边 episode 长度变化，训练日志里的混合方式和其它算法一样走 `merge_offline_online`。

### 3.5 不改的在线细节

- 探索仍是去中心化 \(\varepsilon/N\)，\(\varepsilon\) 从 1.0 降到 0.05，衰减 100000 环境步。
- 评估仍是贪心，\(\varepsilon=0\)。PVU 只出现在训练的 TD 目标里，不出现在采集和评估里。
- 2s3z 的在线预算、并行环境数、梯度裁剪、target 间隔与该地图的 OVMSE / MaROAD 配置一致（1M 步，`batch_size_run=2`，clip 1.0，interval 50）。
- 5m_vs_6m 同样对齐（3M 步，`batch_size_run=8`，clip 10.0，interval 200）。

## 4. 代码入口

| 位置 | 作用 |
|---|---|
| `so2/perturb.py` | 离散 PVU |
| `so2/learner.py` | `SO2Learner`，覆盖 bootstrap 动作，带 `n_upc` |
| `training/factory.py` | `learner_type: so2` 时构造 `SO2Learner`，其余参数与 QMIX+CQL 工厂相同 |
| `training/online.py` | 仅当 learner 有 `n_upc` 时重复梯度步 |
| `configs/so2_2s3z.yaml`、`configs/so2_5m_vs_6m.yaml` | 在线配置 |
| `scripts/launch_so2.sh` | 一张卡一张图，nohup 由调用方加 |

checkpoint 文件名是 `so2_online_full_td_seed<seed>.pt`。

## 5. 怎么跑

单图（先 `export SC2PATH`）：

```bash
python train.py --config configs/so2_5m_vs_6m.yaml phase=online seed=1
```

两张卡各一个 seed、各一张图（2 号卡 2s3z，3 号卡 5m_vs_6m，seed 都是 1）：

```bash
nohup bash scripts/launch_so2.sh > logs/so2_batch_master.log 2>&1 &
```

日志在 `logs/so2_batch/so2_<map>_seed1.out`，指标在 `checkpoints/so2_<map>/seed1/online.jsonl`。评估行里会有 `so2/perturb_frac_interval_mean` 和 `so2/n_upc_interval_mean`。
