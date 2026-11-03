# 离线到在线多智能体强化学习框架说明

## 1. 整体设计

这个仓库做多智能体离线到在线（offline-to-online，O2O）强化学习实验。离线数据用 OG-MARL 的 vault，在线训练按 PyMARL / PyMARL2 的做法写。两个阶段共用同一个训练入口和同一套张量布局，用配置里的 `phase` 字段切换。文中路径都相对仓库根目录。

离线阶段不跑环境，batch 从 vault 里按 episode 采样；只有评估时才进环境。在线阶段开 `batch_size_run` 个并行环境采 episode，放进 buffer，训练时按 `mixing_ratio` 掺入离线数据。两阶段靠 checkpoint 衔接：在线启动前必须找到本轮离线训练产出的 checkpoint（`training/online.py` 的 `_resolve_offline_checkpoint`），同时会额外冻结一份离线网络给 OVM 用。

## 2. 代码结构

代码分四层：共享组件在 `qmix/`，算法实现在 `ovmse/`、`macal/`、`macflow/`、`fsr/`、`uot/`，训练流程在 `training/`，数据与环境在 `data/`、`envs/`。`utils/` 放路径、checkpoint 和日志，`configs/` 放全部实验配置，`scripts/` 放串联离线-在线-评估的脚本。

| 目录 | 内容 |
|---|---|
| `qmix/` | QMIX 系列共用的组件：`MarlcRNN` agent network、`QMixer`、episode unroll 与 obs 增广（`core.py`）、epsilon-greedy 评估策略（`policy.py`） |
| `ovmse/` | OVMSE 复现：TD(λ) target（`targets.py`）、MACQL / CFCQL 保守损失（`cql.py`）、behaviour cloning（`behaviour.py`）、离线价值记忆 OVM（`ovm.py`）、迁移消融（`transfer.py`）、诊断（`diagnostics.py`） |
| `macal/` | MaCal-QL：Monte Carlo return-to-go（`returns.py`）、Cal-QL 校准保守损失（`calql.py`）、learner（`learner.py`） |
| `macflow/`、`fsr/`、`uot/` | 其它研究算法（联合流策略蒸馏、流支撑重放、UOT），本文不展开 |
| `training/` | 训练流程：`factory.py` 按配置构造 learner / 环境 / logger，`offline.py` 和 `online.py` 是两个阶段的循环，`rollout.py` 负责并行采集，`evaluation.py` 和 `q_metrics.py` 做评估与 Q 值诊断 |
| `data/` | vault 下载与 episode 数据集加载 |
| `envs/` | 环境工厂和各环境 wrapper（SMAC v1/v2、MAMuJoCo、MPE、RWARE、Flatland 等） |
| `utils/` | 路径解析、checkpoint 读写、JSONL / WandB logger |

## 3. 数据格式

所有阶段都用 PyMARL 的 `[B, T, N, ...]` 布局。一个 dataset 或 batch 包含七类张量：`obs`、`actions`、`state`、`avail_actions`、`reward`（团队奖励，取各 agent 奖励的均值）、`terminated`、`filled`。episode 会 pad 到固定长度 `T = episode_limit + 1`：多出来的一步是 bootstrap slot，复制前一时刻的 obs、state 和 legal actions，`terminated` 置 1、`filled` 置 0，这样 TD target 在 episode 末尾还能从下一状态 bootstrap（`data/vault_dataset.py`）。

loss 和统计一律用 `filled[:, :-1]` 做 mask，`terminated` 只在真正结束时为 1。SMAC 打满步数上限的超时局由 `training/rollout.py` 的 `td_terminated_flag` 单独处理：环境停掉这一局，但 TD 标记仍为 0，下一状态照常 bootstrap。这点对 OVM target 影响很大，详见 7.4。

agent network 在所有 agent 间共享参数：只有一个 `MarlcRNN` 实例，靠 obs 末尾拼的 agent-id one-hot 区分身份（`add_agent_id_to_obs`，默认打开），每个 agent 各有一个 GRU hidden state。obs 增广顺序和 PyMARL 一致：obs、上一动作 one-hot、agent-id one-hot（`qmix/core.py` 的 `_augment_obs`）。

## 4. 离线训练

入口是 `python train.py --config <yaml>`，默认 `phase: offline`。流程如下：

1. `load_vault_dataset` 先看 `vaults/` 下有没有数据，没有就从 HuggingFace 下载（支持 `HF_ENDPOINT` 镜像），然后加载并打印统计；
2. `build_learner` 按 `learner_type` 构造 learner，可选 `ovmse_paper`、`macal_ql`、`mac_flow`、`fsr`、`uot`；
3. 开了 `moderate_lambda` 时（OVMSE 的 CFCQL 需要 behaviour policy），主循环前先训 behaviour cloning 网络，步数上限由 `max_behaviour_filled_steps` 控制，结果缓存到 `behaviour_cache_dir`；
4. 主循环每跨过 `test_interval` 个 env step，就在真实环境里用 greedy 策略评估 `num_eval_episodes` 局并写日志；平均回报刷新纪录就存 best checkpoint。之后从数据集采 `batch_size` 条 episode 做一次梯度更新，`t_env` 按这批 episode 的填充步数累加。设 `count_updates: true` 时改为按梯度步数计数，用于以 update 数为单位的实验协议；
5. `t_max` 用完做最后一次评估，存 best 和 final 两个 checkpoint。

checkpoint 由 `utils/checkpoint.py` 写出，包含 `q_network`、`mixer`、两个 target network、optimizer 状态和元信息。离线文件名是 `{stem}_seed{seed}_best.pt` 和 `{stem}_seed{seed}.pt`，stem 由 learner 类型决定（OVMSE 是 `ovmse_paper`，MaCal-QL 是 `macal_ql`）。

## 5. 在线训练

入口是 `python train.py --config <yaml> phase=online`。`train.py` 会把 `online:` 块里的 `t_max`、`test_interval`、`log_path` 提到顶层。流程如下：

1. 重新加载 vault（混 batch 要用离线 episode）并构造 learner；
2. 加载离线 checkpoint：网络、target network、optimizer、训练步计数。`transfer.mode: scratch` 时跳过这一步，在线网络从零开始；冻结的离线网络仍会加载，供 OVM 和诊断使用（`load_offline_for_ovm`）；
3. `apply_transfer`（`ovmse/transfer.py`）按需做迁移消融：`ovmse` / `full_td` / `fsr` 不动权重；`rank_only` 做保序的 affine 缩放并重置 mixer；`rank_break` 打乱每个 agent Q 头动作行的顺序来破坏排序；`shift_plus` / `shift_minus` 平移 Q 头 bias，往 Q 值里注入量级偏差；`none` 只加载权重、不做处理。`transfer` 块没写时默认 `mode=ovmse`，也就是默认开 OVM；
4. 主循环：`batch_size_run` 个并行环境（线程池）按 ε-greedy 采整局 episode，放进 `deque` buffer。ε 从 `eps_start` 线性降到 `eps_min`，斜率由 `eps_decay_steps` 控制，每个 agent 的实际探索概率是 ε/N。buffer 里的 episode 数达到 `batch_size` 后，每次并行 rollout 做一次梯度更新，训练 batch 按 `mixing_ratio` 混入离线 episode，在线阶段的 CQL 权重换成 `online_cql_weight`；
5. 定期评估（含 OVM 诊断和 Q bias 指标），结束时存 `{stem}_online_{mode}_seed{seed}.pt`。

在线阶段还有两个实验用的机制：`ovm_diag` 块支持固定样本诊断和周期 checkpoint；`fork_at` / `resume_fork` 支持在指定 env step 冻结训练器状态（网络、optimizer、buffer、RNG），再用不同的 ranking loss 分支恢复，做受控对照。日常训练用不到。

## 6. 网络结构

`MarlcRNN`（`qmix/networks.py`）就是 PyMARL2 的 `n_rnn`：ReLU 全连接层、`GRUCell`、线性 Q 头，GRU 后面不加激活。`QMixer` 是单调超网络：一支 hypernetwork 从 global state 生成第一层权重并取绝对值保证非负，加上 state 相关的 bias 和 value stream，把各 agent 的 Q 混合成 `Q_tot`。这个结构满足 IGM 条件，联合动作的 greedy 选择可以拆成每个 agent 各自取 argmax，MaCal-QL 的 target 构造靠的就是这一点。target network（`target_q_network`、`target_mixer`）按 episode 数刷新，间隔是 `target_update_interval` 条 episode；有 `batch_size_run` 个并行环境时，实际每 `batch_size_run` 条 episode 走一步。

## 7. OVMSE：原理与 5m_vs_6m 运行参数

### 7.1 算法组成

OVMSE（arXiv:2410.19450）是在 QMIX 上做 offline-to-online 的方法，实现在 `ovmse/`，分四块：

- **TD(λ)**：λ_TD = 0.6 的多步 target，沿时间反向构造（`ovmse/targets.py`）；
- **保守正则**：默认用 CFCQL 的反事实形式，加 moderate-λ：先用 behaviour cloning 网络和各 agent 反事实 Q 的 KL，按 `softmax_temp` 做温度 softmax，在 agent 间分配 CQL 惩罚（`ovmse/cql.py`、`ovmse/behaviour.py`）。`raw_cql: true` 时退化成 MACQL 的联合动作 logsumexp 形式；
- **离线价值记忆（OVM）**：在线阶段冻结离线 `Q_tot`，target 取 `max(Q_offline, TD target)`，loss 是 `(1 - λ_memory)` 倍普通 TD loss 加 `λ_memory` 倍 OVM loss（`ovmse/ovm.py` 的 `_ovm_td_loss`）。`λ_memory` 从 1.0 线性退火到 `lambda_memory_end`，时长为 `lambda_anneal_steps`；
- **数据混合**：在线 batch 按 `mixing_ratio` 掺入离线 episode，在线阶段 CQL 权重换成 `online_cql_weight`。

### 7.2 5m_vs_6m 完整参数

下面是 `configs/5m_vs_6m.yaml`（5m_vs_6m，Medium）的完整参数，也就是对齐论文后的现行配置。表中的 env step 都是填充步（`t_env` 按 batch 的填充步数累加）。5m_vs_6m 平均每个 episode 约 27 步，128 条一批大约 3,463 个填充步，所以 10M 离线预算约合 2,900 次梯度更新；按 10,000 步的评估间隔，大约 960 次评估（一次 32 局）。

**任务与预算**

| 参数 | 取值 | 备注 |
|---|---|---|
| `task.source` / `env` / `scenario` / `dataset` | `cfcql` / `smac_v1` / `5m_vs_6m` / `Medium` | CFCQL 先验工作的 vault |
| `seed` | 1 | |
| `t_max`（离线） | 10,000,000 | env step |
| `online.t_max` | 3,000,000 | 在线微调预算 |
| `batch_size` | 128 | 每次梯度更新的 episode 数 |
| `batch_size_run` | 8 | 在线并行环境数 |
| `buffer_size` | 5000 | 在线 episode buffer 容量 |
| `test_interval` | 10,000 | 评估间隔（env step），在线阶段用同一值 |
| `num_eval_episodes` | 32 | 每次评估局数，greedy，ε = 0 |

**网络与优化**

| 参数 | 取值 |
|---|---|
| `learning_rate` | 0.001（Adam） |
| `recurrent_layer_dim` | 64 |
| `mixer_embed_dim` / `mixer_hyper_dim` | 32 / 64 |
| `discount` | 0.99 |
| `td_lambda` | 0.6 |
| `target_update_interval` | 200（按 episode 计） |
| `grad_norm_clip` | 10.0 |
| `add_agent_id_to_obs` | true |

**保守正则与 behaviour cloning**

| 参数 | 取值 | 备注 |
|---|---|---|
| `raw_cql` | false | 用 CFCQL 形式 |
| `moderate_lambda` | true | 训 behaviour 网络做温和分配 |
| `cql_weight` | 50.0 | 离线 CQL 权重 |
| `softmax_temp` | 0.5 | agent 间惩罚分配温度（对齐项，见 7.4） |
| `num_ood_actions` | 100 | OOD 候选动作数 |
| `max_behaviour_filled_steps` | 500,000 | behaviour cloning 步数上限，约 144 个 batch |
| `behaviour_cache_dir` | `<checkpoint_dir>/behaviour` | behaviour 网络缓存 |

**在线迁移与探索**

| 参数 | 取值 | 备注 |
|---|---|---|
| `transfer.mode` | ovmse | 开启 OVM |
| `online.use_ovm` | true | 显式标注，实际开关是 `transfer.mode` |
| `mixing_ratio` | 0.1 | 在线 batch 中离线 episode 占比 |
| `online_cql_weight` | 0.0 | 在线阶段关掉保守正则 |
| `lambda_memory_end` | 0.1 | λ_memory 退火终点 |
| `lambda_anneal_steps` | 200,000 | 退火时长（env step） |
| `eps_start` / `eps_min` / `eps_decay_steps` | 1.0 / 0.05 / 100,000 | 去中心化 ε/N 探索 |
| `device` | cuda | |

### 7.3 运行命令

手动跑单个阶段（SMAC 需要 StarCraft II，先 `export SC2PATH=<SC2 安装路径>`）：

```bash
cd /home/jxy/multiAgentRL

# 离线预训练
python train.py --config configs/5m_vs_6m.yaml \
  checkpoint_dir=checkpoints/ovmse_5m_vs_6m \
  log_path=checkpoints/ovmse_5m_vs_6m/offline.jsonl

# 在线微调，自动读取 online.t_max，开启 OVM
python train.py --config configs/5m_vs_6m.yaml \
  phase=online \
  checkpoint_dir=checkpoints/ovmse_5m_vs_6m_online \
  checkpoint_path=checkpoints/ovmse_5m_vs_6m/ovmse_paper_seed1_best.pt \
  log_path=checkpoints/ovmse_5m_vs_6m_online/online.jsonl
```

一条命令跑完离线、在线和 640 局终评：

```bash
bash scripts/run_ovmse_5m_vs_6m.sh Medium          # Medium 数据集
bash scripts/run_ovmse_5m_vs_6m.sh Medium-Replay    # Replay 数据集，脚本会自动覆盖
                                                    # system.mixing_ratio=0.0
                                                    # system.lambda_memory_end=0.05
```

终评 checkpoint 在 `checkpoints/<run_dir>/ovmse_paper_online_ovmse_seed1.pt`，640 局结果写到 `eval640_online_final.jsonl`。

### 7.4 与论文对齐的三处修改

早期版本在 5m_vs_6m Medium 上 640 局 greedy 胜率 45.47%，对齐后 76.09%（论文多种子中位数 69.53%）。一共改了三处（`documentation/ovmse_5m_vs_6m_alignment.md`）：

1. **超时局的 TD 终止标记**：SMAC 打满 70 步上限时，`terminated` 不再写 1，TD(λ) 继续 bootstrap。原来的写法会把 OVM target 里的在线项整段压低；
2. **学习率与 batch**：统一用 yaml 里的 1e-3 和 batch 128，之前脚本覆盖成了 5e-4 和 32；
3. **`softmax_temp` 从 5 改成 0.5**：0.5 让 CFCQL 惩罚在 agent 间分配得比较平均。

## 8. MaCal-QL：原理与运行

### 8.1 算法组成

MaCal-QL（`macal/`，实现说明见 `documentation/macal_ql_implementation_and_correctness.md`）把 Cal-QL 的 calibrated conservative Q-learning 搬到 QMIX 上。TD loss 用一步 backup，target 的 max 按 IGM 拆成 target network 各 agent 的 greedy 动作。保守项作用在联合 `Q_tot` 上，候选联合动作来自由各 agent 逐动作最大值构造的去中心化 ε-greedy 提议分布，并用 importance weight 修正提议分布和均匀分布的偏差。Cal-QL 的校准 clamp 把提议组的 log 动作值抬到不低于该 episode 的 Monte Carlo return-to-go `G_t`：只有出现过真正终止的 episode 才提供有效锚，被时间上限截断的 episode 整段放弃。权重 α 默认用对偶梯度学习（`dual_alpha: true`），当推低和推高的差超过 `target_action_gap` 时持续增大。

### 8.2 配置与运行

两组配置。`configs/macal_5m_vs_6m.yaml` 是 SMAC 5m_vs_6m Medium：`target_action_gap: 0.8`、`num_ood_actions: 10`、`cql_weight: 1.0`、`alpha_lr: 1e-4`、`alpha_init: 1.0`，在线 3M 步、`transfer.mode: none`。截至 2026-10 这个配置还没实际跑过。`configs/macal_2halfcheetah.yaml` 是 MAMuJoCo 2halfcheetah Good，离线与在线都跑过，产物在 `checkpoints/macal_2halfcheetah_{offline,online}/`。

命令和 OVMSE 一样：

```bash
python train.py --config configs/macal_2halfcheetah.yaml \
  checkpoint_dir=checkpoints/macal_2halfcheetah_offline \
  log_path=checkpoints/macal_2halfcheetah_offline/offline.jsonl

python train.py --config configs/macal_2halfcheetah.yaml \
  phase=online \
  checkpoint_dir=checkpoints/macal_2halfcheetah_online \
  checkpoint_path=checkpoints/macal_2halfcheetah_offline/macal_ql_seed1_best.pt \
  log_path=checkpoints/macal_2halfcheetah_online/online.jsonl
```

OVMSE 和 MaCal-QL 在 2halfcheetah 上的对照实验直接跑 `bash scripts/run_ovmse_macal_2halfcheetah.sh`：两个算法各占一块 GPU，各自跑完离线-在线链，离线权重另存一份 `offline_pretrained_frozen.pt`。

## 9. 切换地图

### 9.1 task 字段与数据矩阵

换地图只改配置里的 `task` 块，四个字段：`source`（数据来源）、`env`（环境族）、`scenario`（地图名）、`dataset`（vault 里的数据集名，比如 `Medium`、`Good`、`Medium-Replay`）。`data/vault_download.py` 的 `VAULT_INFO` 里登记了所有能自动下载的组合：

| source | env | scenario |
|---|---|---|
| `og_marl` | `smac_v1` | 3m、8m、5m_vs_6m、2s3z、3s5z_vs_3s6z |
| `og_marl` | `smac_v2` | terran_5_vs_5、terran_10_vs_10、zerg_5_vs_5 |
| `og_marl` | `mamujoco` | 2halfcheetah、2ant |
| `og_marl` | `gymnasium_mamujoco` | 2ant、2halfcheetah、2walker、3hopper、4ant、6halfcheetah |
| `og_marl` | `flatland` | 20_trains、30_trains |
| `cfcql` | `smac_v1` | 6h_vs_8z、3s_vs_5z、5m_vs_6m、2s3z |
| `alberdice` | `rware` | small / tiny 系列的 2ag、4ag、6ag |
| `omar` | `mpe` / `mamujoco` | simple_spread 等 / 2halfcheetah |
| `omiga` | `smac_v1` / `mamujoco` | 2c_vs_64zg、6h_vs_8z、5m_vs_6m、corridor / 6halfcheetah、2ant、3hopper |

表里没有的组合不会自动下载。可以自己转换一份 `.vlt` 放到 `vaults/<source>/<env>/<scenario>.vlt`。

### 9.2 切换步骤

建议复制一个语义最接近的配置再改 `task` 块。下面几点要一起处理：

1. **改产物路径**：`checkpoint_dir`、`checkpoint_path`、`log_path`、`behaviour_cache_dir` 还指着旧地图的文件名，要同步换掉。在线阶段会按 stem + seed 推断默认文件名（比如 `ovmse_paper_seed1_best.pt`），找不到本轮的离线 checkpoint 会直接报错；
2. **登记回合上限**：`data/vault_dataset.py` 顶部的 `SMAC_EPISODE_LIMIT` 已经包含 2s3z: 120、5m_vs_6m: 70、6h_vs_8z: 150、3s5z_vs_3s6z: 170、2halfcheetah: 1000 等场景。新场景不在里面要补一行，否则加载时报错；
3. **换算预算**：`t_env` 按填充 env step 计数，一次梯度更新消耗的步数是 `batch_size × episode 长度`。不同地图的 episode 长度差很多：2halfcheetah 每局固定 1000 步，128 条一批是 128,000 步，而 5m_vs_6m 只有约 3,463 步。所以要按比例放大 `t_max`、`test_interval`、`eps_decay_steps`、`lambda_anneal_steps`、`max_behaviour_filled_steps`，才能保持 gradient update 数、评估次数和退火比例不变。`target_update_interval` 按 episode 计数，要结合 `batch_size_run` 折算刷新节奏（5m_vs_6m 是 200/8 = 25 次更新刷一次，2halfcheetah 写成 50/2 = 25）。`configs/2halfcheetah.yaml` 开头的注释有一个完整的换算例子；
4. **数据放哪**：大一点的 vault（比如 3s5z_vs_3s6z）设 `system.data_device: cpu`，避免整个数据集常驻显存；
5. **动作模式**：2halfcheetah 默认走离散化，每个 agent 27 档力矩（`envs/discrete_mujoco_actions.py`）；`envs/__init__.py` 对 `scenario=2halfcheetah` 且 `action_mode != continuous` 的组合会自动选 `DiscreteMAMuJoCo`。要连续动作就显式设 `task.action_mode: continuous`，目前只有 MAC-Flow 等 learner 支持。

### 9.3 切换示例

从 5m_vs_6m 换到 3s5z_vs_3s6z（已经有 `configs/3s5z_vs_3s6z.yaml`），关键差异：

```yaml
task:
  source: og_marl      # 5m_vs_6m 用的是 cfcql 源
  scenario: 3s5z_vs_3s6z
t_max: 9966990         # Medium vault 996,699 个填充步的十倍
batch_size: 32
batch_size_run: 1
test_interval: 2500000
num_eval_episodes: 16
system:
  data_device: cpu
  target_update_interval: 200
online:
  t_max: 1000000
  test_interval: 200000
```

换到 2halfcheetah 时，按上面的预算换算把各个步数常量放大约 37 倍（128,000 / 3,463），参考 `configs/2halfcheetah.yaml` 和 `configs/macal_2halfcheetah.yaml`。

### 9.4 环境依赖

SMAC 需要 StarCraft II 和 `SC2PATH` 环境变量；MAMuJoCo 用 `envs/custom/multiagent_mujoco`；MPE、RWARE、Flatland 各有自己的 wrapper，在 `envs/` 下按需导入。所有环境都通过 `envs.get_environment(source, env, scenario, seed, action_mode)` 构造，`env` 和 `source` 的合法组合见 `envs/__init__.py`。部分组合有专用封装，比如 `source=omiga` 的 smac_v1 走 `SMACv1OMIGA`。

## 10. 评估、checkpoint 与日志

评估（`training/evaluation.py`）用 ε = 0 的 greedy 策略跑 `num_eval_episodes` 局。除了 mean / max / min return 和 SMAC 胜率（`battle_won`），每个决策都记录在线 Q 和冻结的离线 Q，经 `training/q_metrics.py` 汇总成 Q bias 诊断：`mean_q_tot`、`mean_return_to_go`、`q_bias`、`q_overestimate_rate`、`q_kendall_tau`（Q 和 return 的排序一致性）、`greedy_action_disagree_rate`（在线与离线 greedy 动作的不一致率）等。日志默认写 JSONL（配置 `log_path`；没配就走 WandB，项目名 `wandb_project`）。

checkpoint 命名：离线 `{stem}_seed{seed}_best.pt` / `{stem}_seed{seed}.pt`；在线 `{stem}_online_{transfer_mode}_seed{seed}.pt`。`utils/checkpoint.py` 还提供整训练器快照（含在线 buffer 和 RNG 状态），给 fork / resume 对照实验用。

## 11. 常用命令

```bash
# OVMSE（SMAC，需要 SC2PATH）
python train.py --config configs/5m_vs_6m.yaml                      # 离线
python train.py --config configs/5m_vs_6m.yaml phase=online         # 在线，开 OVM
bash scripts/run_ovmse_5m_vs_6m.sh Medium                            # 离线 + 在线 + 640 局终评

# MaCal-QL
python train.py --config configs/macal_2halfcheetah.yaml            # 离线
python train.py --config configs/macal_2halfcheetah.yaml phase=online

# 两算法对照（2halfcheetah，各占一块 GPU）
bash scripts/run_ovmse_macal_2halfcheetah.sh

# 终评任意 checkpoint
python scripts/eval_640.py --config <yaml> --checkpoint <ckpt.pt> \
  --episodes 640 --chunk 32 --out <out.jsonl> --tag <tag>

```

覆盖任意配置项用 Hydra 风格的点路径，比如 `python train.py --config configs/5m_vs_6m.yaml seed=2 task.dataset=Medium-Replay system.mixing_ratio=0.0`。
