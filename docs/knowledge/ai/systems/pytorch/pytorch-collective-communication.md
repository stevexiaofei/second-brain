---
title: 集合通信原语：all-gather、reduce-scatter 与它们的亲戚
type: concept
status: growing
tags: [pytorch, distributed, collective, nccl, all-gather, reduce-scatter, all-reduce, source-code]
created: 2026-11-03
updated: 2026-11-03
source: PyTorch 源码（distributed_c10d.py / _fsdp_collectives.py / placement_types.py）+ 个人整理
---

# 集合通信原语：all-gather、reduce-scatter 与它们的亲戚

## 一句话理解

> 集合通信（collective communication）是"一组 rank 一起调用的、语义确定的通信操作"：每个原语定义了**数据如何从 N 份变成 N 份**。训练里最常用的两个是 `all-gather`（把 N 份分片拼成 N 份完整拷贝）和 `reduce-scatter`（把 N 份完整数据归约成 N 份分片）——它们恰好是彼此的"逆操作"，FSDP 的前向用前者取回参数、反向用后者分发梯度，正是这一对逆操作的完美应用。

## 为什么重要

- **它是所有并行策略的词汇表**：DDP = 反向时 all-reduce 梯度；FSDP/ZeRO-3 = 前向 all-gather 参数、反向 reduce-scatter 梯度；TP = 前后向各一次 all-reduce（或 all-gather + reduce-scatter）；EP/MoE = all-to-all。看到一种并行范式，第一反应应该是"它用了哪些原语、各多少次"。
- **通信量公式直接决定设计选择**：ring all-reduce 与"reduce-scatter + all-gather"的通信量完全相同（$2S\frac{N-1}{N}$），这就是"ZeRO-1/2 = reduce-scatter + all-gather 拆解 all-reduce"成立的原因——拆开之后中间态的显存占用却完全不同。
- **性能分析要落到原语粒度**：[07 号 profiling 笔记](../../../engineering/profiling/07-pytorch-distributed-profiling.md)里"NCCL kernel 占比高"的下一步就是分清**是哪种原语、消息多大、有没有被计算掩盖**，而这些都建立在认识每个原语的语义与代价之上。

## 核心概念

### 一张表看全部原语

设 world size = $N$，每个 rank 的输入大小为 $S$ 字节（除特别注明外，输入同构）：

| 原语 | 输入（每 rank） | 输出（每 rank） | 每 rank 通信量（ring，标准实现） | 输出语义 |
|---|---|---|---|---|
| **broadcast** | $S$ | $S$ | $S$（树形，$\log N$ 步） | 根 rank 的数据人人有 |
| **scatter** | $NS$ | $S$ | $S$ | 根的数据切成 N 份，各拿一份 |
| **gather** | $S$ | $NS$ | $S$ | 所有数据汇聚到根一个 rank |
| **reduce** | $S$ | $S$ | $S$ | 归约结果只在根 rank |
| **all-reduce** | $S$ | $S$ | $2S\frac{N-1}{N}$ | 归约结果人人有 |
| **all-gather** | $S$ | $NS$ | $S\frac{N-1}{N}$ | 拼接后的完整数据人人有 |
| **reduce-scatter** | $NS$ | $S$ | $S\frac{N-1}{N}$ | 归约结果切 N 份，各拿一份 |
| **all-to-all** | $S$ | $S$ | $\sim S\frac{N-1}{N}$（但依赖均衡） | 每 rank 把自己的分片发给目标 rank，人人"换货" |

读表的三条规律：

1. **all-X = X + broadcast，reduce-scatter = reduce + scatter**，且都不必经过根 rank（ring 实现里没有根的概念，这是它高效的原因）。
2. **all-gather 与 reduce-scatter 互为逆操作**：前者输入是 N 份分片、输出 N 份全量；后者输入 N 份全量、输出 N 份分片。两者总通信量都只有 $S\frac{N-1}{N}$，**远低于朴素 gather/reduce 的 $S$**（ring 把数据切成 N-1 块流水传递，每步只传 $S/N$）。
3. **all-to-all 是唯一"重排数据"的原语**：其余原语对所有 rank 做同样的事，all-to-all 允许每个 rank 指定"我这块给谁"，是 MoE 专家分发、TP 重分片（`Shard(i) → Shard(j)`）的底层支撑。

### 一个 4-rank 的具体例子

设 $N=4$，每 rank 持有一个各不相同的 4 元向量：

```text
r0: [1,1,1,1]   r1: [2,2,2,2]   r2: [3,3,3,3]   r3: [4,4,4,4]
```

各原语（SUM）之后每 rank 手里的数据：

```text
broadcast(src=0)   每 rank: [1,1,1,1]          ← 只有 r0 那份，复制
reduce(dst=0)      r0: [10,10,10,10]，其余不变   ← 求和但只有 dst 拿到
all_reduce         每 rank: [10,10,10,10]       ← 求和 + 人人一份
scatter(src=0)     rank r 得 r0 输入的第 r 块    ← 切开分发
gather(dst=0)      r0 收齐 4 个向量，其余为空    ← 收集到一人
all_gather         每 rank: 16 元拼接向量        ← 人人持全量
reduce_scatter     rank r 得 [10] 的第 r 个元素  ← 求和 + 各拿一块
all_to_all         rank r 得「各 rank 输入的第 r 块」拼接 ← 重排，无算术
```

两组最容易混淆的对照：

- **broadcast vs all-gather**：broadcast 传播的是**同一份**数据（人人结果相同，信息量 = 1 份）；all-gather 收集的是 **N 份不同的**分片（人人结果相同，但信息量 = N 份）。FSDP 前向恢复参数用的是 all-gather（分片本来就在各 rank 手里），推理权重同步才是 broadcast。
- **reduce vs reduce-scatter**：reduce 的结果只落在 dst 一个 rank；reduce-scatter 把归约结果**切开**，每 rank 永久持有一块。DDP 场景下每 rank 反正只需给自己那份参数做更新，全量归约结果里"人人拿全"的那一半是冗余的——这正是 reduce-scatter 省通信的空间。

### 关键洞察：all-reduce = reduce-scatter + all-gather

ring all-reduce 的内部实现恰恰是两步：先 reduce-scatter（每 rank 得到全量归约结果的 $1/N$ 分片），再 all-gather（把分片拼成全量）。这个分解在三个地方反复出现：

- **DDP 的桶 all-reduce**：一次 ring all-reduce，内部即 RS+AG。
- **ZeRO-1/2 的优化**：DDP 梯度 all-reduce 时，**各 rank 只需要自己参数分片对应的梯度分片**（更新优化器状态用），所以只需要 reduce-scatter 半程；优化器步之后新参数再 all-gather 回去。
- **FSDP**：干脆把两半拆到两个时刻——反向 reduce-scatter 梯度，前向 all-gather 参数。**通信总量不变，但中间态从"全量归约结果"变成"$1/N$ 分片"，换来显存的大幅下降。**

这就是"通信量不变、显存换时间/时间换显存"的经典权衡，理解了它，ZeRO 的 0/1/2/3 阶段划分只是"哪些状态参与 all-gather / reduce-scatter"的列举。

### 通信量公式的直觉：为什么 ring 是 $\frac{N-1}{N}$

ring all-gather 把每 rank 的输入切成 $N$ 个 chunk（每份 $S/N$），沿环传 $N-1$ 步，每步每 rank 收发 $S/N$；总收发 $(N-1)\cdot\frac{S}{N} = S\frac{N-1}{N}$。要点：

- **带宽项**随 N 缓慢增长并趋近 $S$，而不是线性增长 $NS$——这是 ring 优于 tree/朴素实现的核心。
- **延迟项** $\approx 2(N-1) \cdot t_{latency}$ 与传输步数成正比，所以**小消息用 ring 划不来**（NCCL 对小消息自动切 Tree 协议，见 [07 号笔记](../../../engineering/profiling/07-pytorch-distributed-profiling.md)的 NCCL 调参表）。
- 公式里的 $S$ 是**参与该次通信的字节数**。bucket 越小、调用次数越多，延迟项占比越高——这是"不要把 bucket 调太小"的数学依据。

## PyTorch API 全景

### 三组 API 的分工

| API 家族 | 位置 | 特点 | 用途 |
|---|---|---|---|
| **传统 collective** | `torch.distributed.distributed_c10d` | 输入输出是**独立的 tensor 参数**，返回 work handle | 命令式调用，DDP/FSDP/手写通信 |
| **functional collectives** | `torch.distributed._functional_collectives`（`funcol`） | 通信是 **tensor 上的运算**：`tensor ALL_GATHER(group)` 返回新 tensor，可进 autograd 图、可被 torch.compile 捕获 | DTensor、torch.compile 下的分布式 |
| **Coalescing** | `with _coalescing_manager(group, work): ...` | 一个上下文里打包多个 collective，一次后端提交，摊薄 CPU 发起开销 | 小消息聚合、Gloo 优化 |

源码位置（`torch/distributed/distributed_c10d.py`）：`broadcast` :2785、`all_reduce` :2835、`reduce` :3005、`all_gather` :3791、`all_gather_into_tensor` :3891、`gather` :4130、`scatter` :4213、`reduce_scatter` :4318、`reduce_scatter_tensor` :4361、`all_to_all_single` :4490、`all_to_all` :4639、`barrier` :4761。

### ReduceOp 与后端支持

归约算子：`SUM` / `PRODUCT` / `MIN` / `MAX` / `AVG` / `PREMUL_SUM`。注意 **`AVG` 与 `PREMUL_SUM` 仅 NCCL 支持**，Gloo 只有前四种——CPU 上用 gloo 调试、GPU 上用 nccl 生产时，归约算子的可用面不同，切后端要检查。

### `all_gather` 家族：两个变体的差别

```python
# 变体 1：all_gather —— 输出是 tensor 列表（每 rank 一个元素）
tensor_list = [torch.zeros_like(x) for _ in range(world_size)]
dist.all_gather(tensor_list, x)          # tensor_list[r] = rank r 的 x

# 变体 2：all_gather_into_tensor —— 输出是单个拼接/堆叠 tensor（更快，无 Python 列表开销）
out = torch.empty(world_size * x.numel(), dtype=..., device=...)
dist.all_gather_into_tensor(out, x)      # out = cat([x_rank0, x_rank1, ...])
```

- 变体 2（`_allgather_base`，`ProcessGroup.hpp:339`）要求各 rank 输入**同形**，输出是连续缓冲区，NCCL 实现更高效——**训练热路径一律用它**；变体 1 适合收集元数据、指标等异构小张量。
- `all_gather_object` 是 PyObject 序列化版，仅限低频控制流（如收集评估结果），绝不要出现在训练 step 里。

### `reduce_scatter` 家族

```python
# 输入 = world_size × output，按 dim-0 切 N 份；对第 r 份做全 rank 归约，结果落在 rank r
out = torch.empty(numel // world_size, ...)
dist.reduce_scatter_tensor(out, inp, op=ReduceOp.AVG)
```

两个易错点：

1. **语义不是"scatter 我的结果"**，而是"对每个 chunk 归约，第 r 个 chunk 的归约结果只留在 rank r"。输入 $NS$ → 输出 $S$。
2. `op=ReduceOp.AVG` 与手动 pre-divide 有数值差异（见下文 FSDP 一节），且 NCCL 的 AVG 在 world_size=1 时有已知问题（PyTorch 源码注释明确警告）。

### all-to-all：唯一需要"每个 rank 都不同逻辑"的原语

```python
# all_to_all_single：input_split_sizes[r] 个元素发给 rank r，
# output_split_sizes[r] 个元素从 rank r 收来
out = dist.all_to_all_single(out_tensor, in_tensor,
                             output_split_sizes, input_split_sizes)
```

MoE 的 token 分发（token → 专家所在 rank）、TP 的 `Shard(i)→Shard(j)` 重分片都用它。注意 split 是**元素数**而非字节数，且各 rank 的 split 列表必须互相自洽，否则死锁或越界。

拆开看这次调用在每对 rank 之间做了什么（没有算术，纯数据搬运）：

- **发送侧**：本 rank 的 `input` 沿 **dim-0** 按 `input_split_sizes` 切成 N 段，第 $j$ 段（`input_split_sizes[j]` 个元素）发给 rank $j$；其余维度形状各 rank 必须一致——切分只发生在 dim-0。
- **接收侧**：`output` 沿 dim-0 按**来源 rank 顺序**拼接收到的 N 段，来自 rank $i$ 的那段长 `output_split_sizes[i]`。
- **"自洽"的确切含义**：对任意一对 $(i, j)$，rank $i$ 的 `input_split_sizes[j]` 必须等于 rank $j$ 的 `output_split_sizes[i]`——我发你的必须正好是你声明要从我这收的。等价地，两个列表的和分别等于 `input.size(0)` 与 `output.size(0)`。
- 两个列表都传 `None`（或空）时退化为**均匀切分**：`input.chunk(N)` 发一圈、按来源拼回，效果是沿 dim-0 的"转置"（docstring 里 `[0,1,2,3]→[0,4,8,12]` 那个例子）。它也等价于 N 次以各 rank 为根的 `scatter` 的集体化版本。
- 名字里的 **single** 指单 tensor 进出（列表版是 `all_to_all`）；它落到 `group.alltoall_base`，与 `all_gather_into_tensor` 一样是连续缓冲、NCCL 友好的路径。

关键关系一句话：**发送矩阵与接收矩阵互为转置**——本 rank 的 `output_split_sizes` 就是发送矩阵里指向自己的那一列。用官方文档的 4-rank 不均匀例子（矩阵每行 = 该 rank 发给各对端的元素数）：

```text
发送矩阵 input_split_sizes              接收侧 output_split_sizes = 对应"列"
        →r0 →r1 →r2 →r3
r0       2   2   1   1     (dim0=6)    r0: [2,3,2,2] → out = [0,1 | 10,11,12 | 20,21 | 30,31]
r1       3   2   2   2     (dim0=9)    r1: [2,2,1,2] → out = [2,3 | 13,14 | 22 | 32,33]
r2       2   1   1   1     (dim0=5)    r2: [1,2,1,2] → out = [4 | 15,16 | 23 | 34,35]
r3       2   2   2   1     (dim0=7)    r3: [1,2,1,1] → out = [5 | 17,18 | 24 | 36]
```

（可验证：r0 发给 r1 的 2 个元素 `[2,3]`，正是 r1 输出中"来自 r0"的那一段；列视角与行视角一致即自洽。）

实践中 split 大小往往**事先不可知**：MoE 的 router 决定了"我发给别人多少"，但"别人发给我多少"只有别人的 router 知道。所以标准做法是**两段式**——先做一次 tiny all-to-all 交换计数，再做真正的数据 all-to-all：

```python
# ① 元数据交换：每 rank 发出自己的 input_split_sizes（每对端 1 个数，天然均匀 split）
counts = torch.tensor(my_input_split_sizes, device=device)   # shape [N]
recv = torch.empty_like(counts)
dist.all_to_all_single(recv, counts)                         # recv[i] = rank i 要发给我的数量

# ② 有了 output_split_sizes 才能预分配 out、发起合法的数据 all-to-all
out = torch.empty((recv.sum().item(), *inp.shape[1:]), dtype=inp.dtype, device=device)
dist.all_to_all_single(out, inp, output_split_sizes=recv.tolist(),
                       input_split_sizes=my_input_split_sizes)
```

TP 的 `Shard(i)→Shard(j)` 重分片则是均匀切分：两个 split 列表都填 $S/N$（或干脆传 `None`），不需要两段式。

### 同步语义：async_op 与 stream

- `async_op=False`：CPU 阻塞直到通信入队完成（GPU 上只是入队，真正等齐仍由 stream 语义保证）。
- `async_op=True`：返回 work handle，之后 `work.wait()`。**等价但不推荐**——PyTorch 更惯用的做法是独立通信 stream + event 同步（FSDP2 与 DDP 都是这么做的），让 CPU 永不阻塞、通信与计算在 GPU 上自然重叠。
- 传统 API 之外还有 **functional collectives** 的 `AsyncCollectiveTensor`：返回一个"懒 tensor"，用到时才 `wait()`，DTensor 的 `redistribute` 就靠它把通信藏进算子图。

## 工作原理

### 底层：ProcessGroup 抽象

所有 Python API 最终落到 `c10d::ProcessGroup` 的虚函数（`torch/csrc/distributed/c10d/ProcessGroup.hpp`）：`broadcast`/`allreduce`/`reduce_scatter`/`alltoall_base`，以及 base 变体 `_allgather_base`（:339）、`_reduce_scatter_base`（:521）。后端（NCCL/Gloo/MPI/UCC）各自实现这些虚函数；NCCL 的实现基本是"把参数翻译成 `nccl*` 调用"，所以 **NCCL 层的算法选择（Ring/Tree）对 PyTorch 透明**——你能控制的是消息大小、调用次数与 stream 调度，算法/协议交给 `NCCL_ALGO`/`NCCL_PROTO` 干预（一般不需要）。

### NCCL ring 的一步流程（以 all-gather 为例）

```text
每 rank 输入 S 字节，切成 N 个 chunk（每份 S/N），缓冲区按"发送目标"轮转布置：
step k：每 rank 把自己持有的 chunk 发给右邻，同时从左邻收一个新 chunk
       （收发同时进行，流水线满载）
N-1 步后：每个 rank 都集齐了全部 N 份 chunk（自己那份一开始就有，无需传递）
```

对应到硬件：节点内走 NVLink 的 P2P（一跳直达），跨节点走 IB/RoCE/TCP。链路选择由 NCCL 自动探测（`nvidia-smi topo -m` 可提前看拓扑，见 [07 号笔记](../../../engineering/profiling/07-pytorch-distributed-profiling.md)）。

### FSDP2 如何消费这两个原语（源码实证）

FSDP2 的全部通信集中在 `torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py`，只有两个入口：

**前向参数 all-gather（`foreach_all_gather` :164）**：

```python
# ① copy-in：把组内各参数的分片 flat成一维输入（自定义算子 fsdp::all_gather_copy_in）
all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(...)
# ② 通信：单个 base 版 all-gather，输出连续缓冲
all_gather_work = dist.all_gather_into_tensor(
    output_tensor=all_gather_output,
    input_tensor=all_gather_input,
    group=group, async_op=async_op,
)
# ③ copy-out：把连续输出再切开还原成各参数的完整形状（foreach_all_gather_copy_in :274）
```

要点：**一个参数组一次 all-gather**（不是每参数一次）——把整层参数 flat 成一个大消息，摊薄延迟项；输入输出都是一维连续缓冲，通信结束后再切分还原。

**反向梯度 reduce-scatter（`foreach_reduce` :376）**：

```python
# ① copy-in：组内各参数梯度 flat 成一维输入（foreach_reduce_scatter_copy_in :571）
foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size)
# ② 除法因子处理（见下）后发起通信
dist.reduce_scatter_tensor(output=reduce_output, input=reduce_scatter_input,
                           group=reduce_scatter_group, op=reduce_scatter_op)
# ③ HSDP 时在独立 stream 再 all_reduce（节点内 RS + 节点间 AR，见下）
```

**梯度除法因子（`_get_gradient_divide_factors` :605）——数值正确性的细节**：

梯度归约要算 $(g_1 + \dots + g_N)/N$，FSDP2 的实现分三档：

| dtype | 做法 | 原因 |
|---|---|---|
| fp32 / bf16 | 直接 `ReduceOp.AVG`（或 premul $1/N$） | 动态范围大，归约中除不会溢出；**省掉单独的除法 kernel** |
| fp16 | pre-divide $\approx\sqrt{N}$ + post-divide $\approx\sqrt{N}$ | 避免 $N$ 个大梯度相加溢出 fp16 上限（$65504$） |
| 用户显式传 factor | 按用户的 factor 分摊 | MoE 等场景梯度路径不均匀时手动控制 |

这个细节解释了 [03 号显存笔记](../../../engineering/profiling/03-pytorch-memory-profiling.md)里"bf16 训练梯度为什么推荐 bf16 存储"的另一半原因：除了省显存，还省了归约路径上的除法 kernel。

**HSDP 的两段式归约（`foreach_reduce` :469 起）**：reduce-scatter 在**节点内** shard group 上做（省 NVLink 带宽），结果再 all-reduce 到**节点间** replica group 上（走 IB）。对应"数据并行两层嵌套"的拓扑——节点内分片、节点间复制。这就是 `HYBRID_SHARD` 策略的通信结构。

### DTensor：placement 语义到原语的自动翻译

[DTensor](./pytorch-distributed.md) 用三种 placement 描述"张量在 mesh 上怎么放"：`Replicate` / `Shard(i)` / `Partial`。任意两个 placement 之间的转换自动选原语（`torch/distributed/tensor/_redistribute.py` + `placement_types.py`）：

```text
Shard(i) → Replicate()   all-gather（_to_replicate_tensor, placement_types.py:227）
Partial  → Replicate()   all-reduce（_reduce_value,           placement_types.py:674）
Partial  → Shard(i)      reduce-scatter（_reduce_shard_value, placement_types.py:683）
Shard(i) → Shard(j)      all-to-all（_to_new_shard_dim,       placement_types.py:281）
Replicate → Shard(i)     本地切片，无通信
Replicate → Partial      本地除以 N（_partition_value，reduce 的共轭），无通信
```

```mermaid
flowchart LR
    RE["<b>Replicate</b><br/><small>每个 rank 持完整张量</small>"]:::step
    SH["<b>Shard(d)</b><br/><small>沿维度 d 切片</small>"]:::step
    PA["<b>Partial</b><br/><small>每 rank 持部分和（典型：本地梯度）</small>"]:::step
    AG["all-gather<br/><small>收集所有分片</small>"]:::action
    AR["all-reduce<br/><small>求和后人人一份</small>"]:::action
    RS["reduce-scatter<br/><small>求和后各持一块</small>"]:::action
    A2A["all-to-all<br/><small>换维度重切分</small>"]:::action
    L1["本地切分<br/><small>零通信</small>"]:::branchNo
    L2["本地除法<br/><small>reduce 的共轭，零通信</small>"]:::branchNo

    SH -->|"Shard → Replicate"| AG --> RE
    PA -->|"Partial → Replicate"| AR --> RE
    PA -->|"Partial → Shard"| RS --> SH
    RE -->|"Replicate → Shard"| L1 --> SH
    RE -->|"Replicate → Partial"| L2 --> PA
    SH -->|"Shard(d0) → Shard(d1)"| A2A --> SH

    classDef step     fill:#eef2ff,stroke:#c7d2fe,color:#312e81,stroke-width:1.5px
    classDef action   fill:#fff7ed,stroke:#fdba74,color:#7c2d12,stroke-width:1.5px
    classDef decide   fill:#fef3c7,stroke:#fcd34d,color:#78350f,stroke-width:1.5px
    classDef branchNo fill:#f0fdf4,stroke:#86efac,color:#166534,stroke-width:1.5px
    classDef branchYes fill:#eef2ff,stroke:#c7d2fe,color:#3730a3,stroke-width:1.5px
```

记住这张表，读任何 TP/FSDP 代码时就能从 placement 变化直接推出通信模式：

```mermaid
flowchart LR
    RE["<b>Replicate</b><br/><small>每个 rank 持完整张量</small>"]:::step
    SH["<b>Shard(d)</b><br/><small>沿维度 d 切片</small>"]:::step
    PA["<b>Partial</b><br/><small>每 rank 持部分和（典型：本地梯度）</small>"]:::step
    AG["all-gather<br/><small>收集所有分片</small>"]:::action
    AR["all-reduce<br/><small>求和后人人一份</small>"]:::action
    RS["reduce-scatter<br/><small>求和后各持一块</small>"]:::action
    A2A["all-to-all<br/><small>换维度重切分</small>"]:::action
    L1["本地切分<br/><small>零通信</small>"]:::branchNo
    L2["本地除法<br/><small>reduce 的共轭，零通信</small>"]:::branchNo

    SH -->|"Shard → Replicate"| AG --> RE
    PA -->|"Partial → Replicate"| AR --> RE
    PA -->|"Partial → Shard"| RS --> SH
    RE -->|"Replicate → Shard"| L1 --> SH
    RE -->|"Replicate → Partial"| L2 --> PA
    SH -->|"Shard(d0) → Shard(d1)"| A2A --> SH

    classDef step     fill:#eef2ff,stroke:#c7d2fe,color:#312e81,stroke-width:1.5px
    classDef action   fill:#fff7ed,stroke:#fdba74,color:#7c2d12,stroke-width:1.5px
    classDef decide   fill:#fef3c7,stroke:#fcd34d,color:#78350f,stroke-width:1.5px
    classDef branchNo fill:#f0fdf4,stroke:#86efac,color:#166534,stroke-width:1.5px
    classDef branchYes fill:#eef2ff,stroke:#c7d2fe,color:#3730a3,stroke-width:1.5px
```

一个有意思的细节：**backward 时 `Replicate → Partial` 被跳过**（`_redistribute.py:249` 的注释）——梯度保持 Replicate 不变，因为"转成 Partial 再归约回来"比"保持 Replicate"更贵，DTensor 在这里做了调度层面的偷懒优化。这是"placement 推导必须考虑 autograd 上下文"的典型例子。

## 与训练范式的对应关系（速查）

| 范式 | 通信模式 | 原语序列（每 step） |
|---|---|---|
| **DDP** | 梯度 all-reduce | 反向：$B$ 个桶各一次 all-reduce（内部 = RS+AG） |
| **ZeRO-1/2** | 梯度 reduce-scatter，参数 all-gather | 反向 RS 梯度分片 → optimizer step → 前向 AG 新参数 |
| **FSDP/ZeRO-3** | 参数 all-gather + 梯度 reduce-scatter | 每层：前向 AG 参数 → 反向 RS 梯度（+ 预取） |
| **HSDP** | 节点内 FSDP + 节点间 DDP | RS(节点内) → AR(节点间) |
| **TP（Megatron 式）** | 每层 2 次集合通信 | f: AG/identity → g: AR；反向对偶 |
| **PP** | 邻居间点对点 | send/recv（activation & gradient），非集合通信 |
| **EP/MoE** | token 重路由 | all-to-all ×2（dispatch + combine） |
| **序列/上下文并行** | KV 分片交换 | all-gather / all-to-all（实现各异） |

> 通信原语是**名词**，训练范式是**句子**。读论文里的并行图时，先把它翻译成"这个箭头是哪种原语、消息多大、在哪个 mesh 维度上"，就完成了从"看懂结构"到"看懂代价"的跨越。

## 我的理解

- **原语选择 = 显存/带宽/延迟的三方议价**：all-reduce 一步到位但中间态全量；拆成 RS+AG 通信量相同、中间态只有 $1/N$；而 FSDP 再把 RS 和 AG 拆到反向与前向两个时刻，用"参数常驻显存 $1/N$"换取通信频次翻倍。同一套原语，调度不同就是不同范式。
- **"base 版 API 存在的理由"是连续大消息**：`all_gather_into_tensor` / `reduce_scatter_tensor` 把"列表 + 拼接"的工作交给 NCCL 在一块连续缓冲上做，省掉多次小 kernel 的启动开销与 Python 层拼接成本。凡是训练热路径，都应该走 base 版 + 独立 stream。
- **除法因子是容易被忽略的正确性细节**：fp16 pre/post-divide 的 $\sqrt{N}$ 技巧、bf16 直接 AVG、world_size=1 时 NCCL AVG 的坑——这类知识不出现在任何并行策略图里，却决定训练会不会 NaN。
- **DTensor 的 placement 表是"通信语法书"**：与其记"TP 用几次 all-reduce"，不如记 `Shard↔Replicate = AG`、`Partial→Shard = RS`、`Shard(i)↔Shard(j) = A2A`，任何并行组合的通信代价都能自己推出来。

## 常见误区

1. **认为 all-gather 每 rank 收 $N \times S$ 字节所以通信量是 $NS$** → ring 实现下每 rank 收发只有 $S\frac{N-1}{N}$；"$N\times S$"是**接收缓冲区大小**，不是链路上的流量。
2. **把 reduce-scatter 理解成"先 reduce 后 scatter 到根再分发"** → ring 实现里没有根，归约与分发是同一圈轮转的两个视角；也正因如此它才能与 all-gather 对偶。
3. **在热路径用 `all_gather`（列表版）或 `all_gather_object`** → 前者有逐元素调度开销，后者走序列化 + gloo 通道，两者都比 base 版慢一个量级；只有控制流才配用它们。
4. **手动 pre-divide 与 `ReduceOp.AVG` 混用** → AVG 是归约内嵌除法（省一次 kernel），手动除是独立 kernel；重复除会静默把梯度缩小 $N$ 倍，loss 曲线"看起来正常但发散得莫名其妙"。
5. **fp16 直接用 AVG 或 SUM** → 大梯度相加可能溢出 fp16 上限；要么 pre/post-divide $\sqrt{N}$（FSDP2 的做法），要么用 bf16/fp32 归约。
6. **认为 FSDP 每个参数各做一次 all-gather** → 是每个**参数组**一次：组内参数 flat 成一条大消息，这正是逐层 `fully_shard` 要自底向上调用的原因（组越小消息越小，延迟项占比越高）。
7. **async_op=True 就自动与计算重叠了** → async 只是让 CPU 不等；真正的重叠靠独立通信 stream + event 同步（DDP/FSDP2 内部做法），手写代码若在发起后立刻在默认 stream 上用结果，照样串行。
8. **在 `Shard(i)` 未经 padding 时直接 reduce-scatter 非整除维度** → $S$ 不整除 $N$ 时输出形状对不上；DTensor 会自动 pad（`placement_types.py:210`），手写代码要自己处理。
9. **分支里跳过一次集合操作** → 所有 rank 必须以**相同顺序**发起相同的集合调用；某个 rank 走了分支没调 all_reduce，其余 rank 永远等在环上——分布式"挂死"十有八九源于此（排查流程见 [07 号笔记](../../../engineering/profiling/07-pytorch-distributed-profiling.md)）。
10. **`async_op=True` 忘记 `work.wait()`** → 句柄被 GC 或直接读输出缓冲都会读到未定义数据，表现为"偶发读到旧值"类 bug；funcol 的 `AsyncCollectiveTensor` 同理，读之前必须 `wait_tensor()`。
11. **在通信完成前改写输入缓冲** → NCCL 异步读它。同 stream 写是安全的（排队序），别的 stream 写必须先同步；FSDP2 源码里大量 `stream.wait_stream(...)` 就是在维护这条纪律。
12. **`reduce_scatter_tensor` 的尺寸契约不满足** → `input.numel()` 必须恰好等于 `output.numel() × N`；FSDP2 的 `_get_dim0_padded_size` 就是为此存在的。

## Related

- [torch.distributed 分布式训练](./pytorch-distributed.md) — 集合通信所处的整体栈：ProcessGroup/Store/Reducer 四层
- [FSDP 源码阅读指南](./pytorch-fsdp.md) — 这两个原语最大的消费者：钩子驱动状态机与通信调度
- [Reducer 类设计详解](./pytorch-reducer.md) — DDP 的 all-reduce 消费者：桶调度、按序归约、comm hook
- [PyTorch 分布式训练性能分析](../../../engineering/profiling/07-pytorch-distributed-profiling.md) — 原语粒度之上的性能方法论：占比、重叠、nccl-tests
- [GPU 算子优化方法论](../gpu-kernel-optimization-methodology.md) — "通信"维度的理论框架，本笔记是其分布式特化
- [分布式系统核心理论基础](../../../distributed-systems/distributed-systems-foundations.md) — 集合通信在更广义分布式系统中的位置（注意区分：那里讲的是消息传递与共识，这里是数据面通信）

## References

- 源码：`torch/distributed/distributed_c10d.py`（collective API 层）、`torch/csrc/distributed/c10d/ProcessGroup.hpp`（`_allgather_base` :339、`_reduce_scatter_base` :521）
- FSDP2 通信：`torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py`（`foreach_all_gather` :164、`foreach_reduce` :376、`_get_gradient_divide_factors` :605）
- DTensor 重分片：`torch/distributed/tensor/_redistribute.py`、`placement_types.py`（各 `_to_*`/`_reduce_*` 方法）
- functional collectives：`torch/distributed/_functional_collectives.py`（`all_gather_tensor`、`reduce_scatter_tensor`、`AsyncCollectiveTensor`）
- NVIDIA NCCL 文档（Ring/Tree 算法、协议选择）与 [nccl-tests](https://github.com/NVIDIA/nccl-tests)
