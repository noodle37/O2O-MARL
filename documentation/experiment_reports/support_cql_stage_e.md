# 队友支持度门控（TCVM）阶段 E：支持分箱过估计验证（5m_vs_6m）

日期：2026-10-28。分析脚本：`scripts/analyze_support_cql_stage_e.py`。本阶段**不训练任何模型**，仅加载 Stage D 的终点检查点，在 CPU 上运行额外评估，把每一步按离线支持度偏离 `D` 分箱，比较各臂的过估计率。

## 1. 目的

阶段 D 发现 `anti`（把 CQL 惩罚放在老队支持**低**的步）比 `uniform`（全局均匀惩罚）尾段回报高 +0.40，3/3 seed 一致。阶段 E 要检验这个收益是否有对应的一阶机制信号：`anti` 是否确实在 D 高的步上降低了 Q 值过估计。

## 2. 方法

- 加载 Stage D 四个臂（`nocql`、`uniform`、`gated`、`anti`）每臂 3 个 seed 的终点检查点。
- 每个检查点跑 128 局贪心评估（固定种子 424242，与 Stage D 一致），在 CPU 上重放离线网络以计算每步的
  ```
  D_t = -(1/N) Σ_i log π_off,i(a_i,t | τ_t)
  ```
- 把每步的 `Q_tot` 与真实回报 `G_t` 对比，得到过估计率 `P(Q_tot > G_t)` 与有符号偏差 `Q_tot - G_t`。
- 按 `D` 的三分位数分箱：bin 0 为低偏离（老队支持高），bin 2 为高偏离（老队支持低）。

## 3. 结果

### 3.1 各臂 pooled 结果（3 seed 均值）

| 臂 | 平均 D | pooled 过估计率 | pooled 偏差 |
|---|---|---|---|
| `nocql` | 1.26 | 0.455 | −0.172 |
| `uniform` | 1.18 | 0.817 | +1.052 |
| `gated` | 1.17 | 0.806 | +0.976 |
| `anti` | 1.18 | 0.736 | +0.573 |

CQL 三个臂的过估计率都远高于 `nocql`，这与该指标在 20 万步区间的行为一致：不加保守项时 Q 反而轻微低估（负偏差），加 CQL 后 Q 转为高估。阶段 D 中 `uniform` 的回报远高于 `nocql`，说明“过估计率”在此不是简单的坏事指标，而是 Q 函数活跃度的反映。

### 3.2 按 D 分箱的过估计率（3 seed 均值 ± 标准误）

| 臂 | 低 D（高支持） | 中 D | 高 D（低支持） |
|---|---|---|---|
| `nocql` | 0.543 ± 0.020 | 0.398 ± 0.043 | 0.424 ± 0.026 |
| `uniform` | 0.688 ± 0.052 | 0.908 ± 0.010 | 0.854 ± 0.037 |
| `gated` | 0.619 ± 0.081 | 0.903 ± 0.024 | 0.896 ± 0.020 |
| `anti` | 0.582 ± 0.026 | 0.835 ± 0.020 | **0.790 ± 0.021** |

关键模式：

- **高 D  bin（预登记关注区）**：`anti < uniform < gated`，且 `anti` 在 3/3 seed 上都低于 `uniform`（−0.065，per-seed [−0.099, −0.032, −0.060]）。这正是放置假设预测的方向：把惩罚集中在低支持区域，会在那里产生更低的过估计。
- **低 D bin**：`anti` 同样低于 `uniform`（−0.106，2/3 seed）。这里 `gated` 也低于 `uniform`，但 `anti` 是最低的。单纯按“在哪里多罚”预测，低 D  bin 应该是 `gated < anti < uniform`，因为 `gated` 在多处罚；实际观察到 `anti < gated < uniform`。
- **中 D bin**：`anti` 明显低于 `uniform` 和 `gated`。

按 D 分箱的偏差（bias）呈现同一排序：高 D bin 中 `anti`（+0.999）< `uniform`（+1.230）< `gated`（+1.120）。

## 4. 解释：放置效应与总质量效应不可完全分离

Stage D 的训练日志显示三个 CQL 臂的实际平均权重不同（`support/cql_weight_mean_interval_mean`）：

| 臂 | 实际平均权重（≈ online_cql_weight 的乘数） |
|---|---|
| `uniform` | 1.000（按构造） |
| `gated` | 0.40–0.45 |
| `anti` | 0.57–0.65 |

因此 `anti` 比 `uniform` 少用了约 35–43% 的 CQL 总质量，又比 `gated` 多用了约 30–60%。这解释了部分结果：

- `anti` 在各 D bin 的过估计率普遍低于 `uniform`，可能部分来自“更少 CQL 质量”而非“更好放置”。
- 但 `anti` 的过估计率也低于 `gated`，而 `gated` 的总质量更低，这说明**放置方向确实起作用**。

最支持放置假设的证据来自**高 D bin**：`gated`（少罚低支持区）在该 bin 过估计率最高（0.896），`anti`（多罚低支持区）最低（0.790），`uniform` 居中（0.854）。三者的排序恰好与“在低支持区施加的惩罚强度”相反。

低 D bin 的偏离（`anti` 最低而非 `gated` 最低）提示：在该 bin 上，总质量效应压过了放置效应；或者 `uniform` 的均匀惩罚在低支持区造成了某种过校正，而 `anti` 通过减少该处惩罚避开了它。

## 5. 结论

1. **预登记的高 D bin 信号成立**：`anti` 在 D 最高的 bin 上系统性地降低过估计（vs `uniform`：−0.065，3/3 seed；vs `gated`：−0.106），支持“在低队友支持区集中保守惩罚”的放置方向。
2. **总 CQL 质量是未被完全控制的混淆**：`anti` 的实际平均权重（0.57–0.65）低于 `uniform`（1.0），因此其全局收益不能唯一归因于放置。
3. **最干净的机制陈述**：与 `gated` 相比，`anti` 用更多总质量却取得更低的高 D 过估计，说明把惩罚从“老队高支持区”移到“老队低支持区”确实改变了 Q 函数的误差分布。
4. **对后续工作的建议**：若要写成一个无混淆的顶会机制，需要把 `gated` / `anti` 的平均 CQL 质量归一化到与 `uniform` 相同（例如让门控权重均值为 1），再重跑 Stage D/E。当前结果支持假设方向，但效应大小受质量差异调制。

## 6. 数据位置

- 原始每步数据：`checkpoints/support_cql_stage_d/5m_vs_6m/stage_e_summary.json`
- 分析脚本：`scripts/analyze_support_cql_stage_e.py`
- 运行日志：`checkpoints/support_cql_stage_d/5m_vs_6m/stage_e.log`

## 7. 复现

```bash
/home/jxy/miniconda3/envs/multiAgentRL/bin/python scripts/analyze_support_cql_stage_e.py \
  --episodes 128 --out checkpoints/support_cql_stage_d/5m_vs_6m/stage_e_summary.json
```

默认 CPU 运行；每臂 3 seed × 128 局约需 1–1.5 小时。
