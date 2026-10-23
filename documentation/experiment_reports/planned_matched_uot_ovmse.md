# 匹配协议下的 UOT / No-Ranking / Vanilla O2O / OVMSE 对照实验方案

日期：2026-10-20
状态：预注册（先写方案，后跑实验）
代码快照：HEAD `b5da88c` + 工作区改动，`git diff | sha256sum = 89f11ce6e3f53a034d384cfea917aa68649d9eacf33cf23c95da39760084f1e4`。
实验期间不修改训练相关代码；新增脚本只读取 checkpoint 或启动训练，不改变训练路径。

---

## 0. 一句话目标

现有 UOT（15.4/0.53）与 OVMSE（17.9/0.78）的 3M 终点差距，同时混杂了**离线初始化/教师 checkpoint、replay mixing、OVM、代码版本**四个因素。本方案先统一起点，再用单变量对照回答：差距来自排序监督本身，还是来自 OVM / 离线回放 / 起点质量。

---

## 1. 当前发现（审计结论）

### 1.1 三条关键链路使用了三个不同的离线 checkpoint

| 用途 | 路径 | t_env | 离线 best_eval_return | 备注 |
|---|---|---:|---:|---|
| UOT 参考实验初始化 + 冻结教师 Q̄ | `checkpoints/ovmse_paper_5m_vs_6m/ovmse_paper_seed1.pt` | 10,002,211（final） | — | UOT 日志第 36 行确认 |
| OVMSE 参考实验初始化 | `checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt` | 5,076,958 | 15.223 | OVMSE 日志第 1518 行确认 |
| 排序可靠性测量教师 Q̄ | `checkpoints/ovmse_paper_5m_vs_6m/ovmse_paper_seed1_best.pt` | 9,159,389 | 14.987 | task-2 的测量对象 |

因此：

- UOT 的 Q̄ 是 10M final，不是可靠性报告里的 9.16M best；
- OVMSE 从 5.08M best 出发，比 UOT 的起点高约 2.7 return（离线 32 集日志口径）；
- 现有“UOT vs OVMSE”不是同一初始化、同一教师下的比较。

### 1.2 已确认的协议差异

| 项目 | UOT 参考 | OVMSE 参考 | 是否匹配 |
|---|---|---|---|
| 起点 | 10M final | 5.08M best | 否 |
| 教师 Q̄ | 10M final（UOT 自冻结） | 5.08M best（OVM 查询） | 否 |
| replay mixing | `0.0` | `0.1`（每 128 条 batch 约 13 条离线 episode） | 否 |
| OVM | off | on，`lambda_memory_end=0.1`，anneal 200k | 否 |
| online CQL | `online_cql_weight=0.0` | `0.0` | 是（不是差异） |
| 学习率 / TD(λ) / batch / envs | 1e-3 / 0.6 / 128 / 8 | 1e-3 / 0.6 / 128 / 8 | 是 |
| 在线步数 | 3M | 3M | 是（约 3.0M） |
| 梯度步数（在线部分） | ≈14,818 | ≈14,103 | 差约 5% |
| 探索 | per-agent ε：1.0→0.05 / 100k | 同 | 名义相同 |
| 评估 | 32 集曲线 + 640 集终点 | 32 集曲线 + 640 集终点 | 终点口径相同 |

补充：

- per-agent ε 使联合动作的随机率在 ε=1.0 时约为 `1-(1-1/5)^5≈0.672`，在 ε=0.05 时约为 `0.049`；名义 ε 相同不等于联合探索完全相同，但两组一致。
- 两条链路都从 checkpoint 恢复 Adam 优化器状态（`training/online.py:_load_learner_weights`），起点不同意味着优化器矩也不同。
- UOT 生产路径与 `verify_rank_reliability.py` 的测量路径不一致：测量脚本把 `live_q=q` 传进邻域构造，而生产在线时 live Q 是当前在线网络；且测量用了 `n_full_cand=44`，生产 K≤8。现有“翻转率”不能直接解释在线决策。
- `qmix/policy.py` 的 `evaluation/greedy_action_disagree_rate` 只认 `offline_q_network`；UOT 存的是 `frozen_q_network`，所以该指标在 UOT 上恒为 0，无法用于在线/教师分歧分析。

### 1.3 性能差距的时间结构

32 集 binned 曲线（win / return）：

| 区间 | UOT | OVMSE | 差 |
|---|---|---|---|
| 0–0.2M | 0.012 / 6.46 | 0.069 / 9.51 | +0.057 / +3.05 |
| 0.2–0.5M | 0.019 / 8.35 | 0.228 / 11.99 | +0.209 / +3.64 |
| 0.5–1M | 0.250 / 12.17 | 0.359 / 13.39 | +0.109 / +1.22 |
| 1–1.5M | 0.456 / 14.56 | 0.478 / 14.76 | +0.022 / +0.20 |
| 1.5–2M | 0.542 / 15.46 | 0.621 / 16.30 | +0.079 / +0.84 |
| 2–2.5M | 0.615 / 16.16 | 0.671 / 16.83 | +0.056 / +0.67 |
| 2.5–3M | 0.618 / 16.21 | 0.708 / 17.20 | +0.090 / +0.99 |

差距在 0.2–0.5M 最大，而不是“后期才失效”。这提示先检查起点与早期适应，而不是只盯着 3M 终点的排序损失。

现有 640 集终点：

| 实验 | win | return |
|---|---:|---:|
| UOT 主线（10M final 起点） | 0.617 | 16.23 |
| UOT control `none`（10M final 起点） | 0.620 | 16.20 |
| OVMSE 参考（5.08M best 起点） | 0.781 | 17.92 |

### 1.4 Ranking 分叉的因果证据

同一 trainer 快照（t≈1.009M）分叉 continue / off / decay，各续训约 2M 步，3 个 seed，640 集终点：

| seed | continue | off | decay |
|---|---:|---:|---:|
| 1 | 0.4219 | 0.5953 | 0.5766 |
| 2 | 0.6531 | 0.6578 | 0.6328 |
| 3 | 0.6687 | 0.6453 | 0.6078 |
| mean | 0.5812 | 0.6328 | 0.6057 |

off−continue 均值 +0.0516，decay−continue +0.0245，seed 间方向不一致。该实验能排除“关闭排序会立刻崩”的强假设，但 3 个 seed、约 0.12 的 95% CI 无法检测 0.05 量级效应，不能据此判定“排序无害”或“排序有害”。

### 1.5 排序可靠性证据

固定教师、分支回放、轨迹聚类 95% CI：

| 口径 | 一致率 |
|---|---:|
| 全部动作对 | 0.614 [0.582, 0.646] |
| algorithm pairs | 0.616 [0.581, 0.652] |
| top-1 vs top-2 | 0.576 [0.467, 0.686] |
| long（最高 vs 最低） | 0.799 [0.735, 0.864] |
| small_gap | 0.519 |
| large_gap | 0.490 |
| greedy wave | 0.641 |
| eps-greedy wave | 0.587 |
| early phase | 0.548 |
| late phase | 0.670 |
| |ΔQ̄|≥20 | 0.743 |

同时，algorithm pairs 的 |ΔG| 中位数 0.426；23.5% 的 pair |ΔG|<0.1，61.6%<0.5，7.4%≥2。决策遗憾重建显示中位数约 0.22–0.38、均值约 0.88–1.15，20–28% 的 probe 决策遗憾 >1.0。也就是说：**翻转既包含大量近等价动作，也存在不可忽略的大回报差尾部**，不能整体忽略，也不能整体归罪。

时间演化：13 个在线检查点一致率 0.605–0.681，不随训练步数系统漂移；主线相对 teacher 的配对差从 +0.008 到 +0.063，方向为正但多数在噪声内。

### 1.6 其他测量缺口

- 32 集评测的噪声：同一 10k 检查点内胜率可在 0.44–0.78 间摆动，不能用于判断 0.05–0.1 的机制效应；本方案终点与关键时间点一律用 640 集。
- 现有 fork 只有 3 个 seed，且分叉点在 1M；无法回答“从初始化开始就关闭排序”的效应。
- 现有 vanilla baseline（`pure_online_5m_vs_6m`）使用 lr 3e-4、ε_start 0.2、120k 步、9.16M best，且已崩溃，不可作为匹配的 Vanilla O2O。
- OVM 机制已有 on/off 对照（3s5z），但 5m_vs_6m 上没有“同起点 + 同 mixing”的 3M 因果对照。
- mixing 在 5m_vs_6m 上没有单独消融；OVMSE 的 mixing=0.1 与 OVM 总是同时出现。

---

## 2. 待回答的问题

- **Q1（起点）**：三个候选 checkpoint 在统一 640 集协议下差多少？现有 UOT/OVMSE 差距中有多少只是起点差？
- **Q2（排序）**：同一 checkpoint、同一在线配方下，UOT ranking（eta=0.1）相对 eta=0 的 Vanilla O2O 是帮助、无害还是有害？
- **Q3（OVM）**：同起点、同 mixing 下，OVM on/off 的 3M 终点差多少？
- **Q4（mixing）**：同起点、同 OVM off 下，mixing 0.1/0.0 的 3M 终点差多少？
- **Q5（决策层）**：在线策略实际选择的联合动作中，有多少与教师 Q̄ 的偏好相反？这些冲突动作的真实回报差是多少？排序损失是否正在把策略从更好的动作推回更差的动作？

---

## 3. 实验设计

### 3.0 公共协议（所有匹配臂）

| 项目 | 设定 |
|---|---|
| 起点 | Step 0 选出的单一 checkpoint（全部臂相同） |
| seed | 1（主实验）；2、3 用于复现 |
| 在线步数 | 3,000,000 env steps（不早停） |
| batch | 128 episodes / gradient step |
| parallel envs | 8 |
| lr | 1e-3 |
| TD(λ) | 0.6 |
| target sync | 每 200 episodes |
| ε | 1.0 → 0.05，100k steps，per-agent |
| buffer | 5000 episodes |
| online CQL | 0 |
| BC / behaviour loss | 不启用 |
| 评估曲线 | 每 10k 步 32 集（只画曲线，不做结论） |
| 终点评估 | 640 集，固定 seed 1，Wilson 95% CI |
| 中间评估 | 每 500k 保存 periodic checkpoint，之后统一跑 640 集 |
| 代码 | 本节冻结的同一工作区快照 |

说明：

- 所有臂从同一 checkpoint 加载，因此 Adam 一阶/二阶矩、`training_step_ctr`、target 网络初值完全一致。
- 所有臂保持 `transfer.mode` 只负责“是否加载 OVM 教师”，不改变 TD 路径。已验证 `UOTLearner.train_on_batch` 与 `PaperQMIXLearner.train_on_batch` 在 OVM off 时使用相同的 QMIX mixing、相同的 `build_td_lambda_targets`、相同的 target sync 与梯度裁剪。
- OVM 臂使用 OVMSE 参考配方：`lambda_memory_end=0.1`、`lambda_anneal_steps=200000`、`use_ovm=true`。
- OVM off 臂仍会加载离线网络用于诊断（PaperQMIX）或冻结教师（UOT），但不进入 Bellman target；这不改变梯度。

### 3.1 Step 0：统一评估三个候选 checkpoint（先做，约 10 分钟）

对以下三个 checkpoint 各跑 640 集，同一 `configs/5m_vs_6m.yaml`、同一 env seed 1、贪心策略：

1. `checkpoints/ovmse_paper_5m_vs_6m/ovmse_paper_seed1.pt`（10M final）
2. `checkpoints/ovmse_paper_5m_vs_6m/ovmse_paper_seed1_best.pt`（9.16M best）
3. `checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt`（5.08M best）

输出：`checkpoints/protocol_audit_20261020/eval640_*.jsonl`，日志 `logs/step0_ckpt_eval.out`。

**实际结果（2026-10-20 14:05 完成）**：

| checkpoint | t_env | 640 集 win | Wilson 95% CI | 640 集 return |
|---|---:|---:|---|---:|
| 10M final | 10,002,211 | 0.2391 | [0.2076, 0.2736] | 12.250 |
| 9.16M best | 9,159,389 | 0.2672 | [0.2344, 0.3028] | 12.436 |
| 5.08M best | 5,076,958 | 0.2422 | [0.2106, 0.2769] | 12.271 |

三个起点的 640 集胜率 CI 两两重叠，统计上不可区分。离线训练日志里的 `best_eval_return`（14.987 / 15.223）在 640 集统一协议下没有复现，说明它们是 32 集噪声选出的“best”，不是真实起点优势。**结论：现有 UOT/OVMSE 终点差不是“OVMSE 起点更高”造成的；起点只值约 12.3 return，而两条在线终点分别约 16.2 / 17.9。**

按上述预注册规则，选择 **5.08M best** 作为全部匹配臂的共同起点。

**一步等价性检查（2026-10-20 14:07）**：同一 checkpoint、同一 128-episode batch、同一优化器状态下，`UOTLearner(eta=0)` 与 `PaperQMIXLearner(use_ovm=False)` 各做一次 TD 更新：

```
uot   loss=2.189168691635132 grad_norm=104.94000244140625
paper loss=2.189168691635132 grad_norm=104.94000244140625
param max diff: q=0.0, mixer=0.0
EQUIVALENCE: PASS
```

因此臂 B 是臂 A 的干净 no-ranking 对照，也是与 C/D/E 同实现的 Vanilla O2O 对照。

**预注册选择规则**（在三个结果全部产出前固定）：

1. 若三个 checkpoint 的 640 集胜率 Wilson 95% CI 两两重叠，视为统计上不可区分，取 OVMSE 参考使用的 5.08M best，以保证与既有参考实验可比；
2. 否则取胜率最高者；
3. 极端情况下胜率完全相同，取平均 return 更高者。

选定后写入结果文档，后续所有臂只使用该 checkpoint。该规则避免用 640 集噪声在三个相近起点中挑选“看起来最好”的那个。

### 3.2 Step 1：决策层诊断（与 Step 2 并行，不占训练卡）

目标：把“排序翻转率”转换成“策略实际选择 + 真实回报差”。

数据来源：现有 fork 三个 3M checkpoint（seed 1）：

- `checkpoints/uot_fork_5m_vs_6m_seed1_formal/uot_online_fork_resume_continue_seed1.pt`
- `.../uot_online_fork_resume_off_seed1.pt`
- `.../uot_online_fork_resume_decay_seed1.pt`

**教师选择**：fork 三个臂在训练时实际使用的 Q̄ 是 10M final checkpoint，因此主诊断用 10M final 作教师；同时用可靠性报告测量过的 9.16M best 作敏感性参考。不要用新的 5.08M best 作 fork 的教师，否则会把“教师换了”混入“排序是否影响决策”。

方法：

1. 用每个 checkpoint 在固定 seed 下各跑一批贪心 rollout，记录其实际访问的状态、合法动作、执行动作、live Q、教师 Q̄。
2. 在同一批状态上计算：
   - **D1 决策分歧率**：任意两臂贪心联合动作不同的状态比例；
   - **D2 冲突率**：`argmax live Q ≠ argmax Q̄` 的状态比例；
   - **D3 冲突方向**：在冲突状态上，forced-return 比较 live 选择与 Q̄ 选择，统计 live 更好/更差/持平；
   - **D4 决策遗憾**：每臂所选联合动作相对该状态最佳 forced 动作的回报差（同一 continuation 下）；
   - **D5 候选覆盖**：生产候选池是否包含 live greedy、Q̄ greedy；live greedy 不在池中的比例。
3. Forced-action 协议：回放记录的前缀，在 probe 步强制某个联合动作，之后用**同一参考 continuation 策略**（分别用 teacher greedy 与 continue 3M greedy 各跑一遍，检查结论对 continuation 的敏感性）完成 episode，计算折扣回报。
4. 只在双方都合法、且 Q̄ 差值超过 margin 的动作对上统计，避免近等价动作噪声。

输出：`logs/decision_conflicts_seed1.json`、`documentation/experiment_reports/task7_decision_conflicts.md`。

判定：

- 若 D3 中 live 更好占比显著高于 50%，说明排序监督在决策层把策略推向更差动作，ranking 本身有问题；
- 若 D3 接近 50% 或 live 更好占比不显著，且 D4 中位数 <0.1，则翻转主要落在近等价动作，ranking 不是主因；
- 若 D5 很高（live greedy 常不在候选池），问题在候选池/支持集实现，而不是排序信号本身。

### 3.3 Step 2：匹配臂 A–E

所有臂的差异严格限制在下表开关；其余全部相同。

| 臂 | 方法 | ranking | OVM | mixing | learner | 关键 override |
|---|---|---|---|---|---|---|
| A | UOT | eta=0.1, joint support | off | 0.0 | `uot` | `system.uot_eta=0.1` |
| B | UOT w/o ranking = Vanilla O2O | eta=0 | off | 0.0 | `uot` | `system.uot_eta=0.0` |
| C | OVMSE 协议 | off | on | 0.1 | `ovmse_paper` | `transfer.mode=ovmse`, `system.mixing_ratio=0.1` |
| D | OVM only | off | on | 0.0 | `ovmse_paper` | `transfer.mode=ovmse`, `system.mixing_ratio=0.0` |
| E | Mixing only | off | off | 0.1 | `ovmse_paper` | `transfer.mode=none`, `system.mixing_ratio=0.1` |

补充说明：

- A、B 使用 `UOTLearner`；B 的 eta=0 仍会在 `no_grad` 下计算排序项用于日志，但不产生梯度，与 A 共享完全相同的代码路径。这样 A−B 只改变排序权重。
- C、D、E 使用 `PaperQMIXLearner`；C/D 的 `transfer.mode=ovmse` 决定 `use_ovm=true`，E 的 `transfer.mode=none` 保证 OVM off。C 同时改变 OVM 与 mixing，D/E 用于把 C 的联合效应拆开。
- 在正式跑之前做一个 **一步等价性检查**：同一 checkpoint、同一 batch、同一随机种子下，`UOTLearner(eta=0)` 与 `PaperQMIXLearner(use_ovm=False)` 各做一次优化器 step，比较 TD loss、梯度范数与更新后参数最大差。若差异非零，先修实现再跑（该检查不进入训练日志）。
- 不在本轮加入“ranking + OVM”组合臂：当前 `UOTLearner` 明确忽略 `use_ovm`，要实现该组合必须改训练代码，会把“方法对比”变成“新方法开发”。等 A–E 结果出来后再决定是否值得做。

### 3.4 分析与统计

**主终点**：3M 时 640 集 win rate 与 mean return，Wilson 95% CI。

**预注册对比**（按优先级）：

1. `A − B`：排序监督的净效应；
2. `D − B`：OVM 的净效应；
3. `E − B`：mixing 的净效应；
4. `C − B`：OVMSE 协议相对 Vanilla 的总差；
5. `C − (D + E − B)`：OVM 与 mixing 的交互（描述性，不做强推断）；
6. `A − C`：排序 vs OVMSE 全协议。

**中间曲线**：0.5M、1.0M、1.5M、2.0M、2.5M、3.0M 的 640 集 win/return，用同一评估协议；重点看 0–1M，因为现有差距主要在那里。

**重复**：seed 1 跑完 A–E 后，至少对 A、B、C 追加 seed 2、3（9 个 run 中的 6 个）；若 D/E 的 seed-1 效应 >0.08，再补 D/E 的 seed 2、3。5 个 seed 不可行时，3 个 seed 是解释方向性的最低要求。

**统计口径**：

- 640 集 Wilson CI 约 ±0.037（p≈0.6），只能可靠检测 ≥0.08 的胜率差；
- 训练 seed 间 SD 参考 fork 实验约 0.06–0.10；3 seeds 的 SE 约 0.04–0.06，因此 <0.08 的差异一律记为“不确定”，不写成“无效应”；
- 不把 32 集曲线上的单点波动当作结论；
- 所有臂共享起点与代码，不做事后挑选 checkpoint；终点固定为 3M 的最后一个 checkpoint（periodic 只用于曲线）。

### 3.5 运行与资源

- Step 0 在 GPU 3（当前空闲）串行跑 3×640 集。
- Step 2 优先使用 GPU 3；GPU 2 当前仅有 PEX-M 的 736 MiB 进程、利用率低，可作为第二张卡共享。每卡并发 2–3 个训练进程，按显存与 CPU 负载调整。
- 全部用 `nohup` 启动，日志写入 `checkpoints/matched_protocol_20261020/<arm>/train.out` 与 `logs/`；不持续监控，只在启动、每 500k checkpoint、结束时检查。
- 预计单个 3M run 在共享卡上约 20–30 小时；seed 1 五臂约 2–3 天完成，复现臂随后追加。
- 训练脚本：`scripts/run_matched_protocol_5m_vs_6m.sh`；配置：`configs/matched_protocol_5m_vs_6m.yaml`；分析脚本：`scripts/analyze_matched_protocol.py`。

---

## 4. 预注册解读规则

| 观测 | 解读 |
|---|---|
| A−B ≈ 0（CI 含 0）且 D/E 有明显正效应 | 排序不是主因，差距主要来自 OVM/mixing；保留/改造 ranking 应集中在决策相关的大 margin 对上 |
| A−B 显著为负（CI 排除 0，且 ≤−0.08） | 排序监督本身有害；下一步做“只在 |ΔQ̄| 大且 live 贪心参与时施加”的门控 |
| A−B 显著为正 | 排序监督有价值；应保留，并解释为何现有 UOT 仍落后（起点/OVM/mixing） |
| D−B 显著为正 | OVM 是主要因素；5m_vs_6m 上复制 3s5z 的 on/off 结论 |
| E−B 显著为正 | 离线回放是主要因素；优先做 replay/mixing 协议，而不是改排序损失 |
| C−B 明显但 D/E 都不明显 | 存在 OVM×mixing 交互，需要因子设计扩展 |
| Step 1 D3 显示 live 更好占比 >50% | 排序信号在决策层确实把策略推向更差动作；即使 A−B≈0，也应改排序目标的门控/权重 |
| Step 1 D4 中位数 <0.1 | 翻转的决策代价有限；把重点转向起点与早期适应 |

所有解释以 3M 640 集终点为主，32 集曲线只用于描述时间结构。若 seed 1 与 seed 2/3 方向冲突，结论记为“不稳定”，不挑选有利 seed。

---

## 5. 产出清单

- `checkpoints/protocol_audit_20261020/eval640_*.jsonl`：Step 0 原始评估。
- `logs/decision_conflicts_seed1.json` + `documentation/experiment_reports/task7_decision_conflicts.md`：Step 1 决策层诊断。
- `scripts/summarize_decision_conflicts.py`：对 Step 1 原始记录做 |ΔQ̄| 分箱、阶段分解与按 map 聚类的 bootstrap；输出 `logs/decision_conflicts_seed1_summary.json`。
- `configs/matched_protocol_5m_vs_6m.yaml`、`scripts/run_matched_protocol_5m_vs_6m.sh`：Step 2 可复现配置与启动脚本。
- `scripts/eval_matched_protocol.sh`、`scripts/analyze_matched_protocol.py`：终点/periodic 640 集评估与 A–E 因子对比（含配对 chunk bootstrap、Newcombe CI、机制检查）。
- `checkpoints/matched_protocol_20261020/<arm>/`：训练日志、periodic checkpoint、640 集终点评估。
- `documentation/experiment_reports/task8_matched_protocol.md`：最终对照报告（A–E、seed 复现、与现有 UOT/OVMSE 参考的差异归因）。

---

## 6. 运行记录

截至 2026-10-20 14:15（启动后约 10 分钟状态）：

- Step 0 已完成；选定共同起点 `checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt`（5.08M best）。
- Step 2 五臂已用 `nohup` 启动：
  `OFFLINE_CKPT=<上述 ckpt> ARMS="A B C D E" A_GPU=3 B_GPU=3 C_GPU=3 D_GPU=2 E_GPU=2 nohup bash scripts/run_matched_protocol_5m_vs_6m.sh > logs/matched_protocol_master_seed1.log 2>&1 &`
  主进程 2111232，子进程 2111241/2111244/2111248/2111253/2111258；启动日志已确认各臂开关与离线 checkpoint 一致。
- Step 1 决策层诊断（8 maps × 2 base arms × 3 probes）已用 `nohup` 启动：
  `SC2PATH=/home/jxy/StarCraftII TMPDIR=/tmp nohup python scripts/analyze_decision_conflicts.py --config configs/matched_protocol_5m_vs_6m.yaml --device cpu --maps 8 --max-probes 3 --out logs/decision_conflicts_seed1.json > logs/decision_conflicts_seed1.out 2>&1 &`
  PID 2122812；SC2 已正常启动。
- 按预注册规则不持续监控；下一次检查点为各臂每 500k periodic checkpoint（或 Step 1 完成）。
