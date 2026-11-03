# MARL_code — OVMSE / CFCQL (O2O MARL)

论文复现：QMIX + CFCQL/MACQL + 在线 OVMSE（arXiv:2410.19450），离线数据来自 OG-MARL Vault，在线在 SMAC 上微调。

## 环境准备

1. Python 3.10+，建议使用独立虚拟环境或 Conda。
2. 安装依赖：

```bash
cd MARL_code
pip install -r requirements.txt
```

3. **StarCraft II**（SMAC 必需）：安装 SC2 并设置 `SC2PATH`。参见 [SMAC 文档](https://github.com/oxwhirl/smac)。

4. 可选环境（仅在配置中使用对应 `task.env` 时需要）：
   - `smacv2` → 安装 SMACv2
   - `flatland` → `flatland-rl`
   - `mamujoco` / `gymnasium_mamujoco` → MuJoCo 与对应包
   - `mpe` / `rware` → 见 `envs/custom/` 内说明

## 目录说明

| 路径 | 说明 |
|------|------|
| `train.py` | 训练入口 |
| `configs/` | 场景 yaml（`2s3z`、`5m_vs_6m` 等） |
| `ovmse/` | 算法：QMIX、CQL、OVM、策略 |
| `training/` | 离线/在线训练循环 |
| `data/` | Vault 下载与 episode 数据集 |
| `envs/` | 多环境封装（默认 SMACv1） |
| `utils/` | 路径、checkpoint、日志 |
| `vaults/` | 自动下载的离线数据（git 忽略） |
| `checkpoints/` | 模型保存（git 忽略） |

相对路径（如 `checkpoints/...`）均解析到 **项目根目录**，与当前工作目录无关。

## 训练

离线（首次会自动从 Hugging Face 下载 Vault）：

```bash
python train.py --config configs/5m_vs_6m.yaml
```

在线（需先完成离线并得到 checkpoint）：

```bash
python train.py --config configs/5m_vs_6m.yaml phase=online
```

Hydra 风格覆盖示例：

```bash
python train.py --config configs/2s3z.yaml seed=2 system.device=cpu
```

## WandB

默认项目名见 yaml 中的 `wandb_project`。首次使用需 `wandb login`。
