# 5m_vs_6m：OVMSE 对齐 MARLC 后的实现改动

`5m_vs_6m` Medium 上，改之前的完整运行按论文 Table 1 做 640 局贪心测试，胜率是 45.47%。论文报告的多种子中位数是 69.53%。对齐下面三处之后，同一口径是 76.09%（487/640，95% 区间 72.6%–79.2%），平均回报 17.75。OVM 损失、`λ_memory` 退火和去中心化顺序探索没有改。

结果在 `checkpoints/ovmse_5m_vs_6m_medium_aligned/eval640_online_final.jsonl`。对齐前的对照是 `checkpoints/ovmse_5m_vs_6m_medium_run2`。

## 1. 超时局的 TD 终止标记

SMAC 在打满步数时 `done=True`，并在 info 里标 `episode_limit`。`5m_vs_6m` 上限是 70 步，大量对局这样结束。

PyMARL2 和 MARLC 的 `parallel_runner` 会停掉这一局，但写入 buffer 的 `terminated` 只有在不是 `episode_limit` 时才为 1。超时局的 TD(λ) 仍从下一状态自举。

改之前，`envs/smacv1.py` 把所有 `done` 都交给 `training/rollout.py` 写成终止。超时局的回报在最后一步被截成 0。OVM 目标是 `max(Q_offline, 在线 TD)`，在线那一项因此整段偏低。

改动：

- `envs/smacv1.py`：把 SMAC 的 `episode_limit` 放进 `info`。
- `training/rollout.py`：`td_terminated_flag` 在超时结束时返回 0，只有真正结束才返回 1。环境仍然停止。

## 2. 学习率和 batch

论文沿用 PyMARL2。MARLC 的 `ovmse_5m_vs_6m.yaml` 与 `qmix.yaml` 是 Adam、学习率 `1e-3`、batch 128。

`configs/5m_vs_6m.yaml` 里已经是这组值。实际启动脚本 `scripts/run_ovmse_5m_vs_6m.sh` 用命令行覆盖成了学习率 `5e-4`、batch 32。离线和在线都用了被改小的那组。

改动：删掉脚本里的这两项覆盖，离线和在线都用 yaml 里的 `1e-3` 和 batch 128。

## 3. 离线 CFCQL 温度

CFCQL 的 moderate-λ 用行为策略和反事实 Q 的 KL，再乘温度做 softmax，在智能体之间分配 CQL 惩罚。MARLC 能打出论文胜率的离线配置是 `softmax_temp: 0.5`（运行名 `slsoftmaxkl_0.5`）。

改之前 yaml 里是 `softmax_temp: 5`。温度 5 会把惩罚集中到 KL 最大的那个智能体；0.5 更接近均匀分配。OVM 会把这份离线 Q 冻住，带进在线目标。

改动：`configs/5m_vs_6m.yaml` 的 `softmax_temp` 从 5 改为 0.5。离线步数仍是 yaml 里的 10M，没有改成 MARLC 配置文件里的 50 万。

## 这次没有改的部分

OVM 公式、`λ_memory` 从 1.0 退火到 0.1（200k 步）、在线 CQL 权重 0、mixing ratio 0.1、去中心化 ε/N、Adam、γ=0.99、TD(λ)=0.6、目标网络间隔 200、梯度裁剪 10，都保持原样。三处是一起改的，没有单因素消融。
