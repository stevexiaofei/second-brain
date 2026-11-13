---
title: 流水线并行：GPipe、1F1B 与零气泡
type: concept
status: growing
tags: [pipeline-parallel, pp, gpipe, 1f1b, interleaved, zero-bubble, microbatch, distributed-training]
created: 2026-11-09
updated: 2026-11-09
source: GPipe / Megatron-LM / Zero Bubble 论文 + PyTorch 源码（torch/distributed/pipelining/）
---

# 流水线并行：GPipe、1F1B 与零气泡

## 一句话理解

> PP 把模型**按层切成多个 stage**，激活向下游传、输入梯度向上游传。多个 microbatch 错峰流动，把多张卡用起来；调度要在**气泡、激活驻留显存和通信**之间权衡。GPipe → 同步 1F1B 的主要收益是减少激活驻留，不应把两种分母不同的气泡公式当成性能提升。

## 为什么重要

- **它是模型跨节点扩展的重要方式**：相比层内 TP，PP 通常在较少的 stage 边界通信，更适合放到节点间；但实际代价仍取决于激活形状、microbatch 数、切分平衡和链路，不能断言总比其他轴便宜多少个数量级。
- **它展示了调度的多重目标**：GPipe 简单，1F1B 控制激活驻留，interleaved 缩短填充/排空的计算单元，zero-bubble 利用可延后的权重梯度计算填空。并非每一步都只在优化同一个气泡公式。

## 核心概念：stage、microbatch 与气泡

### 切法与通信

把 $L$ 层分给 $p$ 个 stage（理想均衡时每个约 $L/p$ 层）：

- **前向**：stage $i$ 接收上游激活 → 本地计算 → 发送输出给下游。
- **反向**：stage $i$ 接收下游梯度 → 本地反传 → 发送输入侧梯度给上游。

普通链式 PP 每个 stage 边界、每个 microbatch 前向发送一次激活，反向发送一次梯度。它不直接传权重，但通信仍随 microbatch 数与边界张量大小变化；非均匀切分、跳连也可能增加通信。

### GPipe：先完成前向，再完成反向

设有 4 个 stage、4 个 microbatch，用 $F_j$、$B_j$ 表示第 $j$ 个 microbatch 的前向和反向。一个合法的 GPipe 顺序是：

- 前向按 stage 0 → 1 → 2 → 3 流动，各 stage 依次处理 $F_1,F_2,F_3,F_4$。
- 末 stage 完成全部前向后，依次处理 $B_1,B_2,B_3,B_4$；其他 stage 收到对应的下游梯度后才能处理各自的 $B_j$。
- 因此同一 microbatch 的反向依赖是 stage 3 → 2 → 1 → 0，不能在下游反向完成前画出上游反向。不同实现可以选择不同的 microbatch 反向顺序，不能把某一种顺序当作 GPipe 的定义。

### 气泡：先固定分母

以下是**stage 均衡、通信开销忽略、每个 microbatch 的前向/反向时间固定**的简化模型。令 $t_f,t_b$ 为一个 stage 处理一个 microbatch 的前向/反向时长，则 GPipe 的理想耗时为：

$$
T_{\text{useful}} = m(t_f+t_b), \qquad
T_{\text{bubble}} = (p-1)(t_f+t_b), \qquad
T_{\text{total}} = (m+p-1)(t_f+t_b)
$$

两种常见口径必须区分：

$$
\text{bubble overhead} = \frac{T_{\text{bubble}}}{T_{\text{useful}}} = \frac{p-1}{m}
$$

$$
\text{bubble fraction} = \frac{T_{\text{bubble}}}{T_{\text{total}}} = \frac{p-1}{m+p-1}
$$

它们是**同一份空闲时间的两种分母**，不需要假设 $t_f=t_b$ 才能互换。当 $m\gg p$ 时数值接近，但不能拿前者代表 GPipe、后者代表 1F1B 来比较。

例如 $p=4,m=4$ 时，overhead 为 $3/4$，空闲占总时间为 $3/7$；若增至 $m=16$，后者为 $3/19$。不存在“只要 $m\ge p$，1F1B 就固定空闲 $(p-1)/p$”的结论。

### microbatch 数与全局 batch

令 $b$ 为每个 microbatch 的样本数，$d$ 为数据并行副本数：

$$
B_{\text{global}} = b\,m\,d
$$

- **固定 $b,d$，增加 $m$**：全局 batch 增大，可能需要调整优化超参数。
- **固定全局 batch 和 $d$，增加 $m$、相应减小 $b$**：不必改变全局 batch；代价主要是小矩阵效率下降、更多调度/通信调用，以及样本数整除约束。
- $b$ 最小通常受样本粒度限制，不能无限减小。所以 $m$ 也并非任意大。

## 调度演进

### 同步 1F1B：主要减少激活驻留

GPipe 在反向开始前保存多个 microbatch 的激活，每个 stage 的激活驻留通常随 $m$ 增长。1F1B 在 warmup 后交替前向和反向，尽早释放已反传的激活，最后 cooldown 排空。

例如 $p=4,m=4$ 时，首 stage 的一种操作顺序是 $F_1,F_2,F_3,F_4,B_1,B_2,B_3,B_4$（其间可能等待）；靠近末端的 stage 更早进入前后向交替。若只有 4 个 microbatch，就不会出现 $F_5,F_6$。

- 在上述理想模型下，**同步非交错 1F1B 与 GPipe 的气泡量级和公式相同**。
- 每个 stage 的激活驻留从通常的 $O(m)$ 降为至多 $O(p)$ 个 microbatch（当 $m$ 足够大时，上游驻留通常更多）。
- 这里讨论每步排空、每步更新一次参数的同步调度，不等同于原始 PipeDream 的异步/权重版本化训练语义。

### Interleaved 1F1B：每卡持有多个虚拟 stage

把模型分为 $vp$ 个更小的虚拟 stage，每个物理 rank 持有 $v$ 个，例如 rank $i$ 持有编号 $i,i+p,\ldots$ 的 stage。调度在同一卡上的多个 model chunk 间交错执行。

在虚拟 stage 等分、可行的 microbatch 调度、忽略通信等假设下，常见的理想近似为：

$$
\text{overhead}_{\text{interleaved}} \approx \frac{p-1}{vm}, \qquad
\text{fraction}_{\text{interleaved}} \approx \frac{p-1}{vm+p-1}
$$

可理解为填充/排空用的计算单元缩小到原来的 $1/v$。不能只给原有 fraction 的整个分母乘 $v$，否则混淆了有效计算和气泡的缩放关系。

代价是更多 stage 边界通信、复杂调度与额外激活管理。具体实现还可能对 microbatch 数有约束；增加 $v$ 不保证端到端更快。Megatron 的虚拟流水线配置属于这一类。

### Zero Bubble：优先输入梯度，延后权重梯度

反向可以拆成两部分：

- **输入梯度计算**：产生传给上游的梯度，在流水线依赖的关键路径上，应尽早完成。
- **权重梯度计算**：参数更新需要它，但上游继续反传通常不依赖它；可以推迟到空闲时隙执行。

以线性层 $Y=XW$ 为例，给定输出梯度 $G$，输入梯度 $GW^\top$ 让上游继续，权重梯度 $X^\top G$ 可稍后计算。**不是先算权重梯度再延迟发送输入梯度**。

Zero Bubble 调度利用这种拆分填充普通 1F1B 的空洞。代价包括保存中间状态、显存约束、调度与优化器依赖管理。理论零气泡是特定计算时间/内存模型下的性质，不能推成任意网络、任意 microbatch 数都无空闲。

本地 PyTorch 源码中：

- `ScheduleInterleavedZeroBubble` 的文档标注实现 ZB1P，用权重反向计算填补气泡。
- `ScheduleZBVZeroBubble` 要求每个 rank **恰有两个 stage**；其理想零气泡性质要求前向、输入反向、权重反向耗时相等。真实模型还要考虑不均衡、通信和优化器检查。

### 调度一览

下表的气泡统一采用**空闲时间 / 总时间**，均基于前述理想假设。

| 调度 | 气泡 fraction | 激活驻留 | 核心权衡 |
|---|---|---|---|
| GPipe | $\frac{p-1}{m+p-1}$ | 通常 $O(m)$ | 实现简单，先全部前向再反向 |
| 同步 1F1B | $\frac{p-1}{m+p-1}$ | 至多 $O(p)$ 个 microbatch | 主要省激活显存，不自动省气泡 |
| Interleaved 1F1B | 近似 $\frac{p-1}{vm+p-1}$ | 依赖虚拟 stage 数与调度 | 缩短填充/排空，但增加通信和管理成本 |
| Zero Bubble 系列 | 依赖拆分耗时、调度与内存约束 | 需要保存可延后计算的状态 | 用权重梯度计算填空，不能套统一的无条件零值 |

## 与其他并行轴的组合

- **PP × TP**：每个 stage 的层内部再 TP。边界传输可能是复制激活，也可能是序列分片等布局；须匹配发送和接收端的 TP/SP 约定，不能假定总传完整激活。
- **PP × DP**：纯 DP×TP×PP 网格下 $d=\text{world size}/(n_{tp}p)$，全局 batch 仍是 $bmd$，不额外乘 PP 或 TP 度数。
- **PP × FSDP**：每个 stage 内再沿数据并行组分片参数/梯度/优化器状态。逻辑可组合不代表没有调度、显存和通信资源竞争。

## PyTorch 侧实现

以下是**4 个进程、每进程一张 CUDA 卡、仅 PP**的一步训练示例，需支持 `torch.distributed.pipelining` 的 PyTorch。保存为脚本后用 `torchrun --standalone --nproc-per-node=4 pp_demo.py` 启动。API 已对照本地 PyTorch 源码核验，**未在四卡环境执行**。

```python
import os
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.pipelining import pipeline, SplitPoint, Schedule1F1B


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*[nn.Linear(16, 16) for _ in range(12)])

    def forward(self, x):
        return self.layers(x)


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        if dist.get_world_size() != 4:
            raise RuntimeError("this example requires exactly 4 ranks")
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.manual_seed(0)
        model = ToyModel().to(device)
        pipe = pipeline(
            model,
            mb_args=(torch.randn(2, 16, device=device),),
            split_spec={
                "layers.3": SplitPoint.BEGINNING,
                "layers.6": SplitPoint.BEGINNING,
                "layers.9": SplitPoint.BEGINNING,
            },
        )
        stage = pipe.build_stage(stage_index=rank, device=device)
        # Only retain this rank's stage for the training loop.
        del pipe, model
        schedule = Schedule1F1B(stage, n_microbatches=8, loss_fn=nn.MSELoss())
        optimizer = torch.optim.SGD(stage.submod.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        if rank == 0:
            schedule.step(torch.randn(16, 16, device=device))
        elif rank == 3:
            losses = []
            output = schedule.step(
                target=torch.randn(16, 16, device=device), losses=losses
            )
            print(output.shape, len(losses))
        else:
            schedule.step()
        optimizer.step()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

- `mb_args` 用**一个 microbatch** 建图；`step` 输入的是本次完整 batch，由调度器拆成 8 份。
- 提供 `loss_fn` 才启用这里的训练反向；label 用 `target=`，逐 microbatch loss 写入调用方传入的 `losses` 列表。末 stage 返回合并输出，其他 stage 返回 `None`，不是 `(losses, output)` 元组。
- 自动分割入口为 `pipeline()` 配 `split_spec` 或模型内 `pipe_split()` 标记；手工构造 stage 用 `PipelineStage`，不是不存在的 `ManualPipelineStage`。
- 本地源码导出 **6 种**调度：`ScheduleGPipe`、`Schedule1F1B`、`ScheduleInterleaved1F1B`、`ScheduleLoopedBFS`、`ScheduleInterleavedZeroBubble`、`ScheduleZBVZeroBubble`。多 stage 调度器需要相应的本地 stage 列表，不能只替换单 stage 示例的类名。
- 调度器管理 stage 间 P2P 收发与依赖，不是每个 microbatch 都做全局 barrier。组合训练可参考 [torchtitan](https://github.com/pytorch/torchtitan)，具体支持组合以对应版本为准。

## 我的理解

- **PP 是空间划分与时间调度的结合**：把层放到不同卡只是起点，让卡持续忙碌才决定吞吐。
- **1F1B 最重要的是控制在途激活**：更早反向、更早释放显存；不要把显存收益写成两个分母不同的气泡数值之差。
- **容量规划必须同时看 $p,m,b,d$**：固定全局 batch 时，增加 $m$ 可以通过减小 $b$ 实现，但可能损害单卡 kernel 效率。
- **声明式切分有边界**：PyTorch 的 trace 帮忙推导 stage，不代表所有 Python 控制流或动态形状都可追踪；不能自动 trace 的模型可以手工构造 stage。

## 常见误区

1. **把 bubble overhead 和 bubble fraction 混为一谈** → 前者除以有效计算时间，后者除以总时间；比较前先统一分母。
2. **认为加大 $m$ 必然增大全局 batch** → 只在固定 $b,d$ 时成立，固定全局 batch 可减小 $b$。
3. **认为 PP 可无成本使用慢链路** → P2P 是依赖路径的一部分，延迟、带宽与计算不均衡仍会形成等待。
4. **混淆 PP 切分与 DP 复制** → PP stage 处理同一样本的不同层，DP 副本处理不同样本。
5. **认为 zero-bubble 先算权重梯度、零空闲且无代价** → 输入梯度优先，权重梯度填空；零气泡依赖特定假设，显存和优化器依赖仍需处理。
6. **忽略跨 rank 调用顺序** → 所有 rank 必须参与匹配的 schedule 步骤，不能某个 stage 提前跳过一轮。

## Related

- [并行策略总览](./parallelism-overview.md) — PP 在多维并行中的位置与容量规划
- [张量并行：Megatron 式](./tensor-parallel-megatron.md) — stage 内部的层内切分
- [专家并行与 MoE](./expert-parallel-moe.md) — stage 内再按专家放置
- [集合通信原语](../pytorch/pytorch-collective-communication.md) — P2P 与集合通信的区别
- [PyTorch 分布式训练](../pytorch/pytorch-distributed.md) — ProcessGroup 与 torchrun
- [FSDP 源码阅读指南](../pytorch/pytorch-fsdp.md) — 可组合的状态分片
- [PyTorch 分布式训练性能分析](../../../engineering/profiling/07-pytorch-distributed-profiling.md) — 在 trace 里量出空闲与等待
- [千卡集群网络拓扑与 NCCL 排查](../../../engineering/mlops-platform/thousand-gpu-network-topology.md) — 跨节点通信的物理链路

## References

- GPipe: *Efficient Training of Giant Neural Networks using Pipeline Parallelism*（Huang et al., 2019）
- PipeDream: *Generalized Pipeline Parallelism for DNN Training*（Narayanan et al., SOSP 2019）— 注意与同步 flush 调度区分
- *Memory-Efficient Pipeline-Parallel DNN Training*（Narayanan et al., 2021）— PipeDream-2BW 等后续方案
- *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*（Narayanan et al., 2021）— interleaved 调度
- *Zero Bubble Pipeline Parallelism*（Qi et al., 2024, arXiv:2401.10241）
- PyTorch 源码：`torch/distributed/pipelining/_IR.py`（`SplitPoint`、`Pipe.build_stage`）、`schedules.py`（六种调度与 `step` 契约）、`_backward.py`（输入/权重反向拆分）、`stage.py`（`PipelineStage`）
