# FJR 与 fork continue 的可比性说明

> 结论：FJR（fresh 3M）与 fork continue（1M 快照 + 2M 续接 + 环境序列重开）不是同一个实验的两次重复，不能把两者的差值归因于 ranking 效应。

## 1. 为什么不能当作重复实验

FJR 和 fork continue 共享算法、超参数和（seed1 的）离线 checkpoint，但至少同时改变了四件事：

| 差异 | FJR fresh 3M | fork continue |
|---|---|---|
| 在线生命周期 | 一个进程连续跑 3M | 1M 后保存快照、退出、新进程续 2M |
| 环境 episode 序列 | 连续的第 0→N 局 | 从第 0 局重新开始 |
| 探索随机流 | 自己的一条未播种流 | 续接父运行在 1M 步的流 |
| replay buffer / 优化器历史 | 自己前 1M 步的历史 | 父运行前 1M 步的历史 |
| 离线 checkpoint（seeds 2/3） | 各自不同 | 各自不同 |

这意味着：它们不是"同一个处理（η=0.1）的两个重复"，而是两个不同的实验条件。观察到的 0.195 差距（FJR 0.617 对 fork continue 0.422）是这四五个差异的混合，无法拆出"ranking 造成了多少"。

## 2. 用实验设计的语言

FJR 和 fork continue 不是同一个因子的两个水平；它们是两个不同的处理组合。把它们放在一起，可以问"这两个条件的结果差多少"，但不能问"ranking 的效应是多少"。

更具体地说：

- **FJR** 回答的问题是：从同一离线模型出发，一个全新的 3M 在线运行能学到什么。
- **fork continue** 回答的问题是：给定一个已经训练了 1M 步的学习器快照，在重开的环境序列上继续训练 2M 步会发生什么。
- 两个问题都合法，但它们的条件不同；一个条件下的结果不能当作另一个条件的重复测量。

## 3. 哪些比较仍然有效

| 比较 | 是否可比 | 原因 |
|---|---|---|
| fork continue vs fork off vs fork decay | 是，内部有效 | 同一快照、同一新环境序列、同一恢复的 RNG、同一 buffer/优化器；唯一差别是 η |
| A vs B vs C vs D vs E | 是，内部有效 | 同一 5.08M best、同一 fresh-run 协议、同一 seed、同一 3M 预算；差别是 ranking/OVM/mixing |
| FJR vs TDR（均为 10M final、fresh 3M） | 是，同一 checkpoint 下的单 seed 对比 | 两者都是 fresh run；但探索未播种，且只有单 seed |
| FJR vs fork continue | 否，不能当作重复或方法对比 | 生命周期、环境序列、探索流、buffer/优化器历史都不同 |
| TDR vs fork-decay | 不干净 | 不只是"早衰减 vs 晚衰减"，还混入了 fresh vs 续接、环境重开 |
| A–E vs 历史 FJR/TDR/OVMSE | 否 | 离线 checkpoint 不同（5.08M best 对 10M final） |

## 4. 640 局评估能做什么、不能做什么

640 局评估让**测量**可比：

- 同样的 640 局、同样的环境 seed、同样的贪心策略；
- Wilson 区间量化的是评估抽样误差；
- 所有臂的数字在同一把尺子上。

但它没有让**实验条件**可比：

- 它不会把 fork 的环境重开变成 fresh run；
- 它不会把未播种的探索变成可复现；
- 它不会把不同离线 checkpoint 变成同一个起点；
- 它不会把单次训练变成多 seed。

因此，640 数字可以放在同一张表里，但这不等于这些实验可以互相解释。
