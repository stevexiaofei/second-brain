---
title: 张量并行：Megatron 式 TP
type: concept
status: growing
tags: [tensor-parallel, megatron, tp, all-reduce, dtensor, sequence-parallel, distributed-training]
created: 2026-11-06
updated: 2026-11-13
source: Megatron-LM 论文（Shoeybi et al., 2019; Narayanan et al., 2021）+ PyTorch 源码（tensor/parallel/style.py）
---

# 张量并行：Megatron 式 TP

## 一句话理解

> TP 把**单个权重矩阵**切开给多卡：Megatron 的核心结构是 Transformer 的 MLP 和 Attention 都能组织成“**列并行 → 行并行**”的两段矩阵乘。配合合适的输入/输出布局，中间可以不通信；集合通信主要出现在布局边界（f/g）——经典 dense Transformer 的这一实现每层前向 2 次、反向 2 次 all-reduce，具体次数会随融合和 SP/词表并行等变体变化。

## 为什么重要

- **它是"模型并行"里粒度很细、也最难放对位置的一根轴**：TP 切到矩阵内部，单卡参数规模通常随 TP 度数下降，但代价是高频通信。NVLink 等高带宽低延迟互联通常更适合承载 TP；跨节点并非数学上禁止，但性能取决于消息大小、拓扑和 overlap。
- **经典 TP 的 f/g 是读并行图的起点**：典型 Transformer block 前向 2 次、反向 2 次 AR；认出逻辑布局边界，再检查具体实现的 SP、融合和额外算子，而不是假定所有模型都有相同次数。

## 问题：为什么"随便切一刀"不行

线性层 $Y = XA$，想要 $A$ 切成两半、两卡各算一半。有两种切法：

**按行切权重时，输入也要匹配分块**：

$$
X=[X_0\;X_1],\qquad A=\begin{bmatrix}A_0\\A_1\end{bmatrix},\qquad
Y=X_0A_0+X_1A_1
$$

**按列切权重时，输出直接分块**：

$$
A=[A_0\;A_1],\qquad Y=[XA_0\;XA_1]
$$

- **行切**：若输入特征也按对应维度分片，每卡得到部分和 $Y_i$，正确结果需要**再求和** → 一次 all-reduce；若每卡拿的是完整 $X$，就不能直接把 $A$ 按行切后相加。
- **列切**：每卡直接得到 $Y$ 的一片，**不需要通信**——但输入 $X$ 必须人人一份完整拷贝，且输出是“碎片”形态。

单看一层怎么切都行，但 Transformer 是**逐层堆叠**的，切法必须让**层与层之间的数据形态衔接**：上一层的输出碎片要能直接当下一层的输入。Megatron 的答案是让两种切法**交替出现**，并且利用一个分块律性质让中间零通信。

## 核心机制：列并行 → 行并行，中间零通信

关键恒等式（分块矩阵乘的串联性质）：

$$
\underbrace{X \cdot A_0 \cdot B_0}_{\text{卡 0}} + \underbrace{X \cdot A_1 \cdot B_1}_{\text{卡 1}} = X \cdot (A_0 B_0 + A_1 B_1) = XAB
$$

如果**同一个矩阵被列切、紧接着的下一个矩阵被行切**（即 $A$ 列切成 $A_0,A_1$、$B$ 行切成 $B_0,B_1$），那么每卡本地算 $X A_i B_i$，最后把两卡的部分和**加一次**就得到完整 $XAB$。中间的 $A_i$ 输出（碎片形态）恰好就是 $B_i$ 需要的输入形态，**不需要任何通信**。

MLP 和 Attention 都精确符合这个模式：

### MLP：两层线性层（列切 → 行切）

```text
              Y = GeLU(XA) B          （A: h→4h, B: 4h→h）
        ┌─────────────────────────────┐
卡 i:   X ──→ [A_i 列切] ──→ GeLU ──→ [B_i 行切] ──→ 部分和 Y_i
        └─────────────────────────────┘
              ↑ X 需要完整（f: identity/AG）      ↓ 一次 all-reduce（g）
```

GeLU 是逐元素的，作用在碎片上等价于作用在完整张量上——这是**列切之后能接激活函数**的原因（换个逐 token 归约的层如 LayerNorm 就不行）。

### Attention：多头天然分块（列切 → 行切）

先看**单卡上的原始计算**。用 $B$ 表示 batch size、$T$ 表示序列长度、$d_{\text{model}}$ 表示模型宽度、$H$ 表示 head 数，且 $d_h=d_{\text{model}}/H$。输入 $X\in\mathbb{R}^{B\times T\times d_{\text{model}}}$ 经过 Q/K/V 投影并拆成 heads；每个 head 独立计算缩放点积注意力，再拼接并经过输出投影：

$$
Q,K,V\in\mathbb{R}^{B\times H\times T\times d_h},\qquad
S_h=\frac{Q_hK_h^\top}{\sqrt{d_h}}\in\mathbb{R}^{B\times T\times T},\qquad
\operatorname{head}_h=\operatorname{softmax}(S_h)V_h\in\mathbb{R}^{B\times T\times d_h}
$$

$$
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_H)\in\mathbb{R}^{B\times T\times d_{\text{model}}},\qquad
Z=\operatorname{Concat}(\cdots)W_O\in\mathbb{R}^{B\times T\times d_{\text{model}}}
$$

```text
单卡：  X [B,T,d_model] ─→ [W_Q,W_K,W_V] ─→ Q,K,V [B,H,T,d_h]
                                           ─→ scores [B,H,T,T] × V
                                           ─→ heads [B,H,T,d_h]
                                           ─→ 合并 heads [B,T,d_model]
                                           ─→ [W_O] ─→ Z [B,T,d_model]
```

这里所有 heads 和完整的 $W_O$ 都在一张卡上，直接得到完整输出 $Z$，没有 TP 通信。Megatron 式 TP 利用 head 之间的独立性，把 $H$ 个 heads 均匀分给 $N$ 张卡（假设 $H$ 可被 $N$ 整除）：每卡负责 $H_i=H/N$ 个 heads，故本地 Q/K/V 与注意力输出的形状分别是 $[B,H_i,T,d_h]$，合并后的本地 head 特征是 $[B,T,H_i d_h]$。对应地，QKV 投影权重按输出维切分；$W_O$ 按输入维行切，每卡权重片是 $[H_i d_h,d_{\text{model}}]$，产生部分和 $Z_i\in\mathbb{R}^{B\times T\times d_{\text{model}}}$，再由 all-reduce 合成完整 $Z$。因此 TP 下各卡的 $Z_i$ 是相同形状的部分和，不是最终输出的不同 head 分片。

```text
        ┌──────────────────────────────────────────────┐
卡 i:   X ──→ [W_QKV 列切] ──→ 自己的 heads 做注意力 ──→ [W_O 行切] ──→ Z_i
        └──────────────────────────────────────────────┘
              f                                       g（一次 AR）
```

### f / g 共轭算子：通信只出现在段的首尾

Megatron 把每段两端的通信包装成两个算子：

| 算子 | 前向（经典无 SP 版本） | 反向 |
|---|---|---|
| **f**（列切段入口） | identity，输入已在 TP 组复制 | **all-reduce**，累加各分支对输入的梯度贡献 |
| **g**（行切段出口） | **all-reduce**，合并部分和 | identity，逻辑上同一份输出梯度直接回传 |

**为什么反向是“共轭”的**：从单个逻辑张量的复制与求和来看，求和的伴随是把梯度复制给各分支，复制的伴随是把分支梯度求和。这里采用 Megatron 的**同一逻辑损失、复制梯度不重复计数**的约定，不能推广成任意原生 `all_reduce` 的 backward 都是 identity。SP 边界的 AG/RS 则构成另一对布局变换。DTensor 的实际 autograd 路径还要结合 placement 与梯度布局，见集合通信笔记的 backward 例外。

### 经典 TP：每层前向 2 次、反向 2 次 AR

一个 Transformer 层 = MLP + Attention，各有一对 f/g：

```text
前向：f_attn(identity) → QKV列切 → attn → O行切 → g_attn(AR)
      + f_mlp(identity) → up列切 → GeLU → down行切 → g_mlp(AR)     = 2 次 AR
反向：对称地，g 的反向是 identity，f 的反向是 AR                     = 2 次 AR
共 4 次 all-reduce（前向 2 + 反向 2）；每个 AR 通信量 2S·(N-1)/N（ring），S 为激活大小
```

这解释了 TP 的两条工程约束：

1. **优先放在高带宽、低延迟域内**。每层通信频繁且存在依赖，overlap 能缓解但不能保证隐藏所有开销；是否值得跨节点必须实测。NVLink 域大小取决于机器拓扑，8 不是 TP 的普适上限。
2. **TP 度数受模型形状与实现约束**。简单均匀 head 分片要求相应 head 数可整除；GQA/MQA 还需检查 KV head 分布。复制 KV head 或专门的不均匀切分需要模型实现支持，不能仅凭 ParallelStyle 假定自动完成。

## 变体与工程细节

### Sequence Parallel（序列并行，Megatron-SP）

本节保留 TP 语境下的逐步例子；SP 与 CP 的整体对照、CP 的 attention 数据流和实现边界见[独立专题](./sequence-context-parallelism.md)。

经典 TP 在 LayerNorm/Dropout 区间常复制激活。Megatron-SP 把行并行出口的 AR 改成 **reduce-scatter**，在下一列并行入口用 **all-gather**，让中间的 LayerNorm/Dropout 和残差路径沿序列维分片；基本模型下 RS+AG 的通信量与 AR 相同，省的是 token-local 区间的激活显存。

#### 用 2 张卡跟一次激活的布局

假设 TP 组有 2 个 rank，某层激活形状为 $[8, B, H]$（8 个 token）。行并行层的每张卡先算出部分贡献；对同一个 token，两卡的部分贡献要相加，才能得到正确的输出。

**经典 TP 在行并行出口做 all-reduce**：先把两卡的部分结果相加，再把完整结果复制给两卡。后面的 LayerNorm、Dropout 和残差路径因此都看到完整 8 个 token：

```text
行并行输出（各自是部分贡献）       all-reduce 后
rank 0: [8, B, H] 部分和  ─┐      rank 0: [token 0..7, B, H]
rank 1: [8, B, H] 部分和  ─┴─求和→ rank 1: [token 0..7, B, H]
                                   两卡都对完整序列做 LayerNorm / Dropout / 残差
```

**Megatron-SP 在这里做 reduce-scatter**：仍然先跨卡求和，但不把完整结果都留给每张卡，而是沿**序列维**切开归属。rank 0 得到 token 0–3 的归约结果，rank 1 得到 token 4–7 的归约结果：

```text
行并行输出（各自是部分贡献）       reduce-scatter（先求和，再按序列切分）
rank 0: [8, B, H] 部分和  ─┐      rank 0: [token 0..3, B, H]
rank 1: [8, B, H] 部分和  ─┴─求和→ rank 1: [token 4..7, B, H]
                                   各自在本地做对应 token 的 LayerNorm / Dropout / 残差
```

因此这段 token-local 区间里，每张卡持有的是 $[8/2, B, H]$，而不是 $[8, B, H]$。如果这些激活需要为反向保存，单卡上这部分重复激活大约降到原来的 $1/2$。残差相加要求两支布局一致：主分支和 shortcut 都必须是同一 rank 上对应的 token 片段，不能拿完整序列直接和半段相加。

但下一列并行入口需要完整序列，所以在进入它之前做 **all-gather**，把两个 rank 的 token 片段收齐，并让每张卡都恢复完整张量：

```text
rank 0: [token 0..3] ─┐
rank 1: [token 4..7] ─┴─ all-gather → 每张卡都得到 [token 0..7, B, H]
                                      再进入下一列并行计算
```

可以把布局变化记成：

```text
经典 TP：行并行 → all-reduce → Replicate(整段序列)
                             → token-local 操作（每卡都存整段）
                             → 下一列并行

Megatron-SP：行并行 → reduce-scatter → Shard(序列片段)
                                     → token-local 操作（每卡只存本片段）
                                     → all-gather → Replicate(整段序列)
                                     → 下一列并行
```

**为什么通信量相同？**设完整激活大小为 $S$、TP 组大小为 $N$。ring 口径下，all-reduce 每 rank 发送 $2S\frac{N-1}{N}$；reduce-scatter 与 all-gather 各发送 $S\frac{N-1}{N}$，合计仍是 $2S\frac{N-1}{N}$。例如 $N=2$ 时，AR 每 rank 发送 $S$；RS 和 AG 各发送 $S/2$，总共也是 $S$。所以 SP 不是减少总传输字节，而是把“完整副本”改成“中间分片”，在 LayerNorm/Dropout/残差等 token-local 区间降低每卡激活占用。字节数相同不保证运行时间完全相同：实际还受 collective 启动延迟、拓扑、实现和计算重叠影响。

注意这不表示整个模型的激活显存必然减半：all-gather 后完整激活又会暂时出现，attention 等需要跨 token 交互的计算也不能简单当成独立 token-local 操作。SP 节省的是**特定序列分片区间**的冗余激活；具体能省多少取决于模型中这些区间、保存的激活和实现策略。

PyTorch 的 `SequenceParallel`（`style.py`）只负责所包裹的 LayerNorm/RMSNorm/Dropout 的序列分片计算。普通 Tensor 输入会被当作**已经本地切好**的序列片段，不会自动把完整序列切开。要组合 TP+SP，还必须一致配置：行并行输出 `Shard(sequence_dim)`、下一列并行输入从该布局收集成 `Replicate`、残差两支使用同样的序列布局、attention 的形状与 head 处理匹配。不能只在纯 TP 计划里给某个 norm 加一条 `SequenceParallel()`。

**SP 不等于 CP**：这里的 SP 主要切 token-local 操作的激活；Context Parallel 则把 attention 的上下文/KV 分给不同 rank，需要额外 KV 交换（例如环式传输、AG 或 A2A），不能把两者都解释为同一种 KV 分片。

### 通信与计算的交织

- 可通过分块、异步 collective 与独立 stream 重叠通信和计算，但后续算子必须能消费已就绪的布局，不能在完整输入到达前直接执行依赖完整张量的运算。
- 经典 TP 反向入口的 AR 累加的是**激活输入梯度**，不是把不同 TP 分片的 $A_i,B_i$ 参数梯度相加。参数分片的副本梯度同步属于 DP 轴；其 bucket 合并不能与 TP 的两次输入梯度 AR 混为一谈。

### 与 DP/FSDP 的组合

TP 组内各 rank 持有不同的参数分片；DP 组则连接**不同 TP 组中同一 TP 坐标**的 rank，同步的是同一参数分片的不同数据副本梯度。FSDP 的参数 AG/梯度 RS 也沿 **DP 子 mesh** 发生，不是在 TP 组内发生。

#### 先看设备分组：4 卡 = 2 路 DP × 2 路 TP

下面用两个 MLP block 演示组合，不引入 Attention、SP 或混合精度，以免把模型结构与设备分组混在一起。每个进程绑定一张 GPU；本例 rank 排布如下：

```text
                       TP 坐标 0      TP 坐标 1
DP 坐标 0                rank 0         rank 1       ← 合作处理数据 A
DP 坐标 1                rank 2         rank 3       ← 合作处理数据 B

横向 TP 组：  [0, 1]、[2, 3]    不同参数分片，合作计算同一批样本
竖向 DP 组：  [0, 2]、[1, 3]    同一 TP 参数分片，处理不同批样本
FSDP 沿竖向分片/恢复参数、归约梯度；不是沿横向做这些操作。
```

`mesh["tp"]` 和 `mesh["dp"]` 取得的是**当前 rank 所在的子组**。例如 rank 0 的 TP 组为 `[0, 1]`，DP/FSDP 组为 `[0, 2]`。这里的 DP 表示数据并行维度，不意味着每张卡始终常驻一份完整的 TP 参数分片：FSDP 会进一步分摊存储，计算时再临时取齐。

下面的交互动画逐阶段展示 4 个 rank 的参数分片与通信方向。点选 GPU 查看它所在的 TP / DP 通信组；也可用播放、上一步、下一步控制训练阶段。动画用于解释设备分组与数据布局，不是 NCCL 包级时序仿真。

<TpFsdpDemo />

#### 完整训练示例

**运行条件**：单机 4 张可见 CUDA GPU、支持 NCCL 的 Linux PyTorch 环境，建议 PyTorch 2.8 或更新版本，并具有下面的公开 API。将以下代码保存为 `tp_fsdp_demo.py`，再通过后面的 `torchrun` 命令启动；不要直接用单进程 `python` 启动。

```python
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


class MLPBlock(nn.Module):
    def __init__(self, width=32, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        # TP 后：fc1 输出本地隐藏特征分片；GeLU 不需要完整隐藏维。
        # fc2 消费这个分片，归约部分和后恢复完整 width 维。
        return self.fc2(F.gelu(self.fc1(x)))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([MLPBlock() for _ in range(2)])
        # head 不做 TP，在 TP 组内复制；稍后仍沿 DP 轴做 FSDP。
        self.head = nn.Linear(32, 8, bias=False)

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return self.head(x)


def main():
    dp, tp = 2, 2
    if int(os.environ.get("WORLD_SIZE", "0")) != dp * tp:
        raise RuntimeError("请用 torchrun --standalone --nproc-per-node=4 启动")
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        raise RuntimeError("需要 CUDA GPU 和支持 NCCL 的 PyTorch 环境")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        mesh = init_device_mesh("cuda", (dp, tp), mesh_dim_names=("dp", "tp"))
        tp_mesh, dp_mesh = mesh["tp"], mesh["dp"]
        dp_rank = mesh.get_local_rank("dp")
        print(
            f"rank={rank}: TP={tp_mesh.mesh.tolist()}, "
            f"DP/FSDP={dp_mesh.mesh.tolist()}",
            flush=True,
        )

        # 1. 各 rank 从同一份模型初始化开始；不能在这里用 rank 改 seed。
        # 注意：此刻每个 rank 都完整构造了整个 TinyModel，这是临时完整副本。
        # TP/FSDP 应用后参数即被 DTensor 分片替换；大模型应改用 meta device
        # 先建空壳再 materialize 本地分片，避免初始化期单卡显存装下整个模型。
        torch.manual_seed(0)
        model = TinyModel().to(device)

        # 2. 先做 TP：每个 block 都是列并行 → GeLU → 行并行。
        tp_plan = {}
        for i in range(len(model.layers)):
            tp_plan[f"layers.{i}.fc1"] = ColwiseParallel()
            tp_plan[f"layers.{i}.fc2"] = RowwiseParallel()
        model = parallelize_module(model, tp_mesh, tp_plan)

        # 3. 再沿 DP 子 mesh 做 FSDP，按 block 组织参数恢复/释放。
        for block in model.layers:
            fully_shard(block, mesh=dp_mesh, reshard_after_forward=True)
        # 根模型接管未被 block 管理的 head 参数，不会重复分片 block。
        # 根 fully_shard 使用默认 root 调度；子 block 仍按上面的策略 reshard。
        fully_shard(model, mesh=dp_mesh)

        # 4. 参数布局确定后，再创建优化器。
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
        model.train()

        # 5. 同一 TP 组的数据相同，不同 DP 副本的数据不同。
        # 用独立生成器，避免数据生成依赖 TP 内部的随机数状态。
        data_rng = torch.Generator(device=device).manual_seed(1000 + dp_rank)
        for step in range(3):
            x = torch.randn(8, 32, device=device, generator=data_rng)
            target = torch.randn(8, 8, device=device, generator=data_rng)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x), target)
            loss.backward()
            optimizer.step()

            # 每个 DP 组都要参与通信，不能把 all_reduce 放进 rank == 0。
            # 这里只聚合日志；梯度已由 FSDP 归约，不手动再做一次。
            mean_loss = loss.detach().clone()
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM, group=dp_mesh.get_group())
            mean_loss /= dp
            if rank == 0:
                print(f"step={step}, dp_mean_loss={mean_loss.item():.6f}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

启动命令：

```bash
torchrun --standalone --nproc-per-node=4 tp_fsdp_demo.py
```

正常执行时，四个进程分别打印各自的 TP 与 DP/FSDP 组，rank 0 再打印 3 步的平均 loss。进程输出顺序不保证固定；具体 loss 以实际运行结果为准，这里不提供未经实测的数值。

**把代码与概念对上：**

1. **TP 计划与模型名一一对应**：`layers.0.fc1` 等是实际存在的子模块。`ColwiseParallel()` 默认要求输入在 TP 组复制、输出沿最后一维分片；`RowwiseParallel()` 默认接收该分片并输出复制布局。因此每个 block 的边界都是完整特征，不需要另外拼接本地张量。本例隐藏维均匀可分，默认返回本地 Tensor 足够表达中间布局。
2. **同模型、不同数据**：模型初始化 seed 在全部 rank 上相同；数据 seed 只依赖 DP 坐标。rank 0/1 使用相同输入与标签，rank 2/3 使用另一批。不能把数据 seed 写成 `1000 + rank`，否则同一 TP 组会合作计算不同样本。seed 一致还有另一层原因：`TinyModel().to(device)` 阶段每个 rank 都临时持有完整模型，TP/FSDP 之后才被分片替换——只有各 rank 的完整权重一致，各自切出的分片才能拼回同一份模型。真实大模型下这个"先完整后切分"会撑爆单卡初始化显存，工程做法是先在 `torch.device("meta")` 上建空壳、再让并行 plan 只 materialize 本地分片；Megatron-LM 原版则干脆按 rank 直接初始化分片权重，不走完整副本。
3. **FSDP 自底向上应用**：逐 block 的调用建立独立参数组；根模型调用处理剩余的 `head`。`head` 虽然不做 TP，但每个 TP 组中的输入、参数与梯度在本例中一致，因此无需额外沿 TP 组同步它的参数梯度。
4. **batch 不乘 TP 度数**：每个数据副本有 8 个样本，两个副本共处理 $8\times2=16$ 个样本，而不是 32 个。MSE 在各副本上取均值，FSDP 按默认语义平均 DP 梯度，不需要再除以 TP 度数；日志也只沿 DP 组求平均。
5. **这不是完整 Transformer 训练框架**：示例只演示 TP×FSDP 与一次训练循环，不包含 Attention、SP、数据集分片、checkpoint 或性能调优。合成随机数据用于检查流程，不保证 3 步 loss 单调下降。

不要把完整 `(dp, tp)` mesh 直接传给 `fully_shard`：它接收二维 mesh 时的约定是 **HSDP 的复制维 × 分片维**，不是 DP×TP；轴名叫 `"tp"` 不会使其自动识别为张量并行。这里必须传 `mesh["dp"]`，且 DP、TP 子 mesh 来自同一个父 mesh。

**验证边界**：API 与组合顺序已对照本地 PyTorch 源码和测试；这不等于四卡运行验证。参考 `torch/testing/_internal/common_fsdp.py` 的 `MLPStack.parallelize()` 与 `test/distributed/_composable/test_composability/test_2d_composability.py`：先沿 TP 子 mesh 分片，再沿 DP 子 mesh 自底向上 `fully_shard`。

## PyTorch 侧实现

PyTorch 的 TP 是 **DTensor + ParallelStyle 计划**式的（对比 Megatron 手写通信）。下面仅是适配单个 Transformer block 的**纯 TP 计划片段**，不是任意模型直接可跑的脚本：进程组/CUDA 设备须先配置，模块名须匹配，attention 的 head reshape 必须按本地 head 数处理，输入在 TP 组内复制。这里不混入 SP，以免隐藏边界布局要求。

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module, ColwiseParallel, RowwiseParallel,
)

tp_mesh = init_device_mesh("cuda", (8,))          # 1-D 子 mesh（2D mesh 需先切片）
plan = {
    "attention.wq": ColwiseParallel(),            # 列切
    "attention.wk": ColwiseParallel(),
    "attention.wv": ColwiseParallel(),
    "attention.wo": RowwiseParallel(),            # 行切
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w3": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(),
}
model = parallelize_module(model, tp_mesh, plan)   # 一次调用完成参数切分 + 输入输出布局
```

- `ColwiseParallel` / `RowwiseParallel`（style.py）把参数重排为 DTensor `Shard(0)`/`Shard(1)` 并准备输入输出布局；通信由 **DTensor 的 redistribute 自动插入**（见下表），用户不写任何 collective。
- PyTorch 用 **placement 约定**表达通信：列并行输入保留 `Replicate` 对应经典 f 的前向 identity；列并行的局部输出是特征 `Shard`，行并行产生的 `Partial → Replicate` 对应 g 的 AR。不要把 f 的 identity 直接等同于 `Replicate → Shard`（后者会实际切输入）。

**placement ↔ 通信的完整映射**（`tensor/_api.py:494-499` 的 `redistribute` 文档）：

| 转换 | 原语 | 对应 Megatron 概念 |
|---|---|---|
| `Replicate → Shard` | 本地 chunk（无通信） | 将已有副本切片，不是经典 f 的 identity |
| `Shard → Replicate` | all-gather | f 前向（需要完整输入时） |
| `Partial → Replicate` | all-reduce | g 前向 |
| `Partial → Shard` | reduce-scatter | g 前向（SP 变体） |
| `Shard(i) → Shard(j)` | all-to-all | 重分片，TP/CP 等布局变换也可能使用 |

## 我的理解

- **TP 的设计受算子结构约束**：列切→行切、逐元素激活和多头独立计算能减少布局转换。LayerNorm 的隐藏维归约不能简单独立地在特征分片上计算；可通信恢复统计量，但经典方案通常保持完整隐藏维，借 SP 分片序列。
- **f/g 对偶要从逻辑计算图理解**：在 Megatron 的复制梯度约定下，f/g 有 AR 与 identity 的配对；AG/RS 也体现分片/拼接的梯度关系。不要把它写成所有 collective 的通用 backward 法则。
- **经典 TP 的通信随层数反复发生**：每层前向 2 次、反向 2 次 AR 是基础模型的估算，而非所有实现的固定税率。工程上常先让 TP 匹配高速互联域，再用其他轴扩展；最终以显存、吞吐和通信重叠实测为准。
- **PyTorch TP 与 Megatron 的分工**：Megatron 把通信手写在层实现里（性能极致、与模型耦合），PyTorch 用 DTensor placement 把通信声明式化（与模型解耦、可被编译栈优化）。前者是"库"，后者是"语言"——学习时用 Megatron 理解通信结构，用 PyTorch 理解现代抽象。

## 常见误区

1. **认为 TP 把"整个模型"切给多卡** → TP 只切**单层内部的矩阵**（列切+行切），层数没变、每卡都要跑所有层。按层切的是 PP。
2. **认为切分方式与相邻算子无关** → 列切输出可直接被匹配的行切消费。若连续列切且后一层需要完整隐藏输入，就要补充布局转换；额外通信量取决于实现，不能简单断言翻倍。
3. **认为反向是把通信机械倒放** → 应从逻辑计算图与梯度布局推导。经典 Megatron f/g 有上述 identity/AR 配对，但不是所有 AR 的通用 backward 定义。
4. **认为 TP 跨节点绝对不行或完全没代价** → 高通信频率使其通常优先放在高速域内；跨节点是否合适要测计算/通信比和 overlap，8 不是硬上限。
5. **认为所有算子都能独立地在特征分片上算** → LayerNorm 的隐藏维统计需要完整特征或额外归约；Dropout 本身是逐元素的，但随机数与分片语义必须一致。词表 softmax/交叉熵也有需要全局统计的专门实现。
6. **忽略 TP 对 head 数/隐藏维的整除约束** → 简单均匀 head 切分需要整除；GQA 的 KV head 复制等逻辑由模型实现负责，不能声称 `style.py` 自动补齐。
7. **以为 `ColwiseParallel` 输出还是完整张量** → 默认输出是 `Shard(-1)` 碎片形态，正是为了让配对的 `RowwiseParallel` 免通信接手；在中间插一个需要完整张量的算子会触发一次意外的 AG。

## Related

- [并行策略总览](./parallelism-overview.md) — TP 在 5D 并行中的位置与 mesh 映射法则
- [流水线并行](./pipeline-parallelism.md) — 按层切分，常与 TP 组合
- [专家并行与 MoE](./expert-parallel-moe.md) — MoE 的 FFN 专家切分，与 TP 正交
- [集合通信原语](../pytorch/pytorch-collective-communication.md) — AR/AG/RS 的语义与通信量公式；f/g 对偶的伴随证明（"求和的伴随是 Replicate"）
- [PyTorch 分布式训练](../pytorch/pytorch-distributed.md) — DeviceMesh 多维拓扑与 DTensor 的位置
- [FSDP 源码阅读指南](../pytorch/pytorch-fsdp.md) — TP×FSDP 2D mesh 的组合实现
- [千卡集群网络拓扑与 NCCL 排查](../../../engineering/mlops-platform/thousand-gpu-network-topology.md) — TP 优先映射到高速互联域的物理依据

## References

- Megatron-LM: *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*（Shoeybi et al., 2019）— 列切/行切与 f/g 算子的原始论文
- *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*（Narayanan et al., 2021）— TP×PP×DP 3D 并行
- *Reducing Activation Recomputation in Large Transformer Models*（Korthikanti et al., 2022）— Sequence Parallel 与选择性激活重计算
- PyTorch 源码：`torch/distributed/tensor/parallel/style.py`（`ColwiseParallel` :45、`RowwiseParallel` :179、`SequenceParallel` :324）、`api.py`（`parallelize_module`）、`tensor/_api.py:494-499`（placement ↔ collective 映射表）
- Megatron-LM 开源仓库（NVIDIA/Megatron-LM）— 手写通信的实现参照
