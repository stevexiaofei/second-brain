---
title: FlashAttention Kernel 细节补充
type: concept
status: growing
tags: [AI, CUDA, FlashAttention, Kernel, GPU]
created: 2026-08-18
updated: 2026-08-18
---

# FlashAttention Kernel 细节补充

## 一句话理解

FlashAttention 的 kernel 不是“一个大矩阵乘法”，而是由 tile 级 forward / backward 子程序组成的流式计算系统：
forward 负责在线 softmax 与输出累积，backward 负责基于 `softmax_lse` 和 dropout 语义重建 `dQ / dK / dV`，而 split-KV 与 sequence-parallel 则是为 GPU 并行度服务的调度增强。

## 为什么单独补这部分

前面的 kernel 笔记已经建立了整体结构，但继续往下看代码后，最值得单独记下来的，是这些更细的实现事实：

- forward / backward 都不是单 kernel 逻辑，而是一组 tile 子程序
- `softmax_lse` 在 forward / backward / split combine 之间反复充当桥梁
- dropout 的随机性需要可复现到 tile 级
- backward 里存在专门的 sequence-parallel 路径和 accumulation 缓冲设计
- kernel specialization 不是细节，而是性能的主来源之一

## forward kernel 的关键事实

### 1. `compute_attn(...)` 只是外层入口

在 `flash_fwd_kernel.h` 里，`compute_attn(...)` 本身并不直接做全部工作，它只是拿到 `m_block / bidb / bidh` 之后，调用更细的 `compute_attn_1rowblock(...)`。

这说明 forward kernel 的基本单位是：

- 一个 query row block
- 一组对应的 K/V tiles
- 该 block 内维护的 online softmax 状态

### 2. dropout RNG 的位置语义非常细

源码里明确写到：

- forward 和 backward 要生成相同的 dropout pattern
- Philox RNG 的 `offset` 存 batch / head / lane id
- `subsequence` 存 attention matrix 中 16×32 block 的位置

这意味着 dropout 不是“整块随机一次”，而是**tile 定位可复现**。

这也是 FlashAttention backward 能够精确重建 forward dropout 路径的基础。

### 3. split-kv forward 有独立入口

`compute_attn_splitkv(...)` 说明 split-KV 不是在主路径里塞一个 if 分支，而是一个独立 kernel 路径。

它会根据：

- 是否 split
- 是否 append KV
- 当前 `blockIdx`
- `n_split_idx` / `num_n_splits`

来决定当前 CTA 处理哪一段 query / key 工作。

这就是 FA2 更强调的 work partitioning：不是单次算完，而是把工作切给更多 CTA。

### 4. combine kernel 的核心任务是合并多个 split 的 LSE 和 O

`combine_attn_seqk_parallel(...)` 读的是：

- `softmax_lseaccum_ptr`
- `softmax_lse_ptr`
- `oaccum_ptr`

它做了两件很关键的事：

1. 把各 split 的 LSE 做 log-sum-exp 合并
2. 用合并后的 scale 去重标定各 split 的 `O` 累积

换句话说，split path 的正确性不只是“把结果加起来”，而是必须先按 softmax 语义重新归一化。

## backward kernel 的关键事实

### 1. backward 的核心入口是 `compute_dq_dk_dv(...)`

在 `flash_bwd_kernel.h` 里，`compute_dq_dk_dv(...)` 会根据当前 `n_block_max` 决定：

- 只有一个 K/V block 时，直接走单块路径
- 否则先从最后一个 block 向前迭代，再处理中间块，再处理第一个块

这种顺序设计一方面适配 block 处理逻辑，另一方面也可能有寄存器/实现上的考虑。

### 2. `compute_dq_dk_dv_1colblock(...)` 是真实工作单元

和 forward 一样，backward 的真正工作也下沉到更细粒度的 block 子程序。

它负责：

- 某个 batch / head / K-block 的局部梯度计算
- `dQ / dK / dV` 的块内累积
- dropout / causal / local / alibi / softcap 的条件处理
- 可能的 sequence-parallel 参与

### 3. `compute_dq_dk_dv_seqk_parallel(...)` 说明 backward 有并行化分支

这个函数表明 backward 并不总是单一串行 block loop。
当 sequence-parallel 路径启用时，它会让多个 block 沿 `blockIdx.x` 分摊 K 方向的块。

其代码结构也说明：

- `blockIdx.y` / `blockIdx.z` 定位 batch / head
- `blockIdx.x` 在 K blocks 之间分摊
- 每个 block 调 `compute_dq_dk_dv_1colblock(..., /*Seq_parallel=*/true)`

这就是 backward 里专门为并行度设计的路径。

## forward / backward 的共同组织原则

### 1. 统一由 params 驱动

forward / backward kernel 都不是直接吃很多单独参数，而是从 `Params` 里拿：

- 指针
- stride
- `seqlen_q / seqlen_k`
- `h / h_k`
- causal / local / dropout / softcap
- `softmax_lse`
- `rng_state`
- `deterministic`

所以 kernel 设计的前提是：**所有复杂性都已经压缩到 params 里了。**

### 2. 计算状态以 tile 为单位维护

无论 forward 还是 backward，最重要的并行粒度都是 tile / block：

- 一个 CTA 处理一个 row block 或 column block
- shared memory 保存局部 tile
- registers 保存中间累积
- 全局内存只做必要写回

### 3. 边界条件尽量前置到 launch / params 层

kernel 内部仍会检查 `Is_even_MN`、`Is_even_K`、mask 条件，但大方向上：

- shape 预处理在 C++ / ATen 层完成
- kernel 内只保留少量 predicate 和 mask

这使得热路径更纯。

## 我从代码里得到的几个更细的认识

### 1. `softmax_lse` 是一种跨阶段状态，而不是单纯副产物

它不只是 forward 的可选输出，而是：

- backward 重建 softmax 的依据
- split combine 重标定的桥梁
- varlen / unpadded 路径中保持语义正确的关键

所以 `softmax_lse` 更像“attention 的流式状态摘要”。

### 2. dropout 的可复现要求很强

为了让 forward / backward dropout pattern 一致，代码把 RNG 语义绑定到了：

- batch
- head
- lane
- tile 位置信息

这比一般的“保存一个随机种子”精细得多。

### 3. backward 的 accumulation 设计反映了内存压力

ATen 侧已经明确避免大尺寸 `dq_accum`，而 kernel 侧的 sequence-parallel / split accumulation 也表明：

- backward 的瓶颈不仅是算力
- 还包括如何组织梯度累积缓冲
- 特别是长序列时，临时 buffer 的尺寸会影响整体吞吐

### 4. split-KV 不是补丁，而是调度策略

它和在线 softmax 的关系很紧：

- split 让并行度更高
- combine 负责把分片结果重新变回 exact attention 的语义

所以它不是近似，而是“分治 + 重新归一化”。

## 与 launch template 的关系

launch template 决定：

- 用哪个 head dim 特化
- 是否启用 dropout / causal / local / alibi / softcap
- 是否走 split path
- 是否 deterministic
- 是否 sequence-parallel

kernel 本体则把这些选择落实到：

- block mapping
- shared memory 布局
- register allocation
- predicate masking
- atomic / accumulation 策略

换句话说：

- **launch template 负责选路**
- **kernel 负责执行**

## 和论文的对应关系

| 论文/概念 | kernel 细节对应 |
|---|---|
| IO-aware exact attention | tile 化计算、避免 materialize score matrix |
| online softmax | `m / l` 状态的流式更新 |
| exact backward | `softmax_lse` + dropout 语义重建 |
| FA2 parallelism | split-KV、sequence-parallel |
| work partitioning | `compute_attn_splitkv` / `compute_dq_dk_dv_seqk_parallel` |
| deterministic backward | 分 split accumulation / 多 buffer 设计 |
| dropout | Philox tile 级可复现 RNG |
| local / causal | mask 与 tile 边界处理 |

## 我自己的理解

1. **FlashAttention 的 kernel 不是“更少内存的 attention”，而是“以 tile 为状态单元的 attention runtime”**
2. **forward 和 backward 共享同一种块化思维，但 backward 的状态恢复更复杂**
3. **split-KV 并不是简单并行，而是和 softmax 语义一起设计的**
4. **dropout 的可复现粒度精细到 block / lane，这就是它能 exact backward 的底层条件之一**
5. **kernel specialization 本质是在用编译期复杂度换运行期性能**

## 相关阅读

- [FlashAttention Kernel 与 Launch 机制](./flash-attention-kernel-and-launch.md)
- [FlashAttention PyTorch ATen 接入层](./flash-attention-pytorch-aten-integration.md)
- [FlashAttention 源码精读](./flash-attention-source-reading.md)

## References

- `third_party/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h`
- `third_party/flash-attention/csrc/flash_attn/src/flash_bwd_kernel.h`
- `third_party/flash-attention/csrc/flash_attn/src/flash_fwd_launch_template.h`
- `third_party/flash-attention/csrc/flash_attn/src/flash_bwd_launch_template.h`
