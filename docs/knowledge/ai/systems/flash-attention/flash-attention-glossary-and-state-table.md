---
title: FlashAttention 术语表与关键状态表
type: concept
status: growing
tags: [AI, FlashAttention, Glossary, State Table, CUDA, PyTorch]
created: 2026-08-18
updated: 2026-08-18
---

# FlashAttention 术语表与关键状态表

## 一句话理解

这篇笔记把 FlashAttention 里最容易混淆的术语和最关键的运行时状态放在一起：
前半部分帮我认词，后半部分帮我认状态。

## 为什么要单独做这篇

FlashAttention 的理解门槛不在“看不懂单个函数”，而在于：

- 很多词看起来像功能，其实是布局问题
- 很多 flag 看起来像开关，其实会改变 kernel 路径
- 很多中间量看起来像副产物，其实是 backward / split combine / inference 的核心桥梁

所以这篇笔记的作用不是讲故事，而是做索引：
以后我看到一个名词，就能快速知道它属于哪一层、服务什么流程、和哪些状态一起出现。

## 先记住的总原则

1. **术语 ≠ 状态**
   - 术语描述概念
   - 状态描述运行时需要保存和传递的东西

2. **训练态最重要的是 backward 可重建**
   - `softmax_lse`
   - `rng_state`
   - `dq / dk / dv`

3. **推理态最重要的是 cache 可持续增长**
   - `cache_seqlens`
   - `cache_batch_idx`
   - `block_table`
   - rotary / ALiBi / window 语义

4. **很多“功能词”其实都在约束数据布局**
   - packed / varlen
   - paged cache
   - MQA / GQA
   - split-KV

---

## 术语表

| 术语 | 一句话解释 | 常见位置 | 记忆点 |
|---|---|---|---|
| **IO-aware** | 把注意力重写成尽量少读写 HBM 的块化计算 | 论文、总图 | 重点是减少内存流量，不是改公式 |
| **Exact attention** | 仍然计算完整 softmax attention，不做近似 | 论文、kernel | FlashAttention 快，但不是近似注意力 |
| **Online softmax** | 在遍历 K/V tile 时在线更新 max 和 sum | forward kernel | 这是“省显存但 exact”的核心技巧 |
| **Tile / block** | 把矩阵切成 GPU 友好的小块 | launch / kernel | attention 的基本工作单位 |
| **CTA** | Cooperative Thread Array，CUDA block 级并行单元 | launch template | 决定一块 CTA 处理哪一段工作 |
| **Split-KV** | 把 K/V 方向切成多个分片并行算 | FA2、launch | 主要为并行度服务 |
| **Sequence-parallel** | 沿序列维度分摊 backward / combine 工作 | backward kernel | 常见于长序列或确定性路径 |
| **Packed QKV** | Q/K/V 已经打包在一个 tensor 中 | Python API | 更适合某些投影和训练路径 |
| **Varlen** | 变长 batch，使用 `cu_seqlens_*` 描述边界 | Python API / C++ / kernel | 不靠 padding，而靠前缀和定位 |
| **MQA / GQA** | Q 的头数多于 KV 的头数 | 接口 / params / backward | 通过 `h / h_k` 支持共享 KV 头 |
| **Paged KV cache** | KV cache 以 page/block 组织，而不是连续大张量 | 推理路径 | 解决长上下文和碎片问题 |
| **Rotary embedding** | 位置编码的一种旋转实现 | `flash_attn_with_kvcache` / `mha.py` | 推理时常与 cache 更新合并 |
| **ALiBi** | 线性位置偏置 | 接口 / params | 是 bias，不是 mask |
| **Softcap** | 对 attention score 做截断/压缩风格处理 | 接口 / params | 代码里通常和 dropout 互斥 |
| **Causal** | 只看当前位置及之前的 token | API / kernel | 右侧窗口常被压成 `0` |
| **Sliding window / local attention** | 只看局部窗口内的 token | API / params | 是 `window_size_left/right` 控制的 |
| **Philox RNG** | CUDA 可复现随机数生成器状态 | dropout / backward | 用于重建相同 dropout mask |
| **Autograd Function** | PyTorch 自定义前后向封装 | Python 接口层 | 训练态上下文的载体 |
| **ATen backend** | PyTorch 内部算子接入层 | `aten/src/ATen/...` | 让 FlashAttention 成为 PyTorch 原生路径 |
| **Kernel specialization** | 用模板按 head dim / feature 特化 kernel | launch template | 性能来自 specialization，代价是编译复杂 |

---

## 关键状态表

### 1. 训练态关键状态

| 状态 | 由谁创建 | 由谁消费 | 作用 | 什么时候重要 |
|---|---|---|---|---|
| `softmax_lse` | forward kernel / C++ bridge | backward kernel、split combine | 保存每一行 softmax 的 log-sum-exp | 训练态最关键的桥梁 |
| `rng_state` | forward 的 dropout 路径 | backward 的 dropout 重建 | 复现同一张 dropout mask | 只要 `dropout_p > 0` 就重要 |
| `out` | forward kernel | backward | 输出张量，也作为 backward 输入之一 | 所有训练路径 |
| `dq / dk / dv` | backward kernel | 上层 autograd | 反向梯度结果 | backward 最终产物 |
| `dq_accum` | backward C++ / kernel | sequence-parallel / accumulation 路径 | 减少巨大临时张量尺寸 | 长序列和 deterministic 路径 |
| `dk_expanded / dv_expanded` | backward C++ | backward 后 reduce | 支持 MQA / GQA 的展开计算 | KV 头数少于 Q 头数时 |
| `softmax_d` | backward C++ / kernel | backward 中间计算 | 存储 softmax 导数相关中间量 | 反向推导必需 |
| `deterministic` | API / params | backward launch / accumulation | 决定是否用更保守的可复现策略 | 调试、实验复现 |
| `seqlenq_ngroups_swapped` | C++ bridge | launch / kernel | decode 场景下的布局优化标志 | `seqlen_q == 1` 时常见 |

### 2. 变长 / 布局状态

| 状态 | 由谁创建 | 由谁消费 | 作用 | 什么时候重要 |
|---|---|---|---|---|
| `cu_seqlens_q` | Python / dataloader | varlen forward/backward | 描述每个 query 序列的前缀和边界 | 变长 batch |
| `cu_seqlens_k` | Python / dataloader | varlen forward/backward | 描述每个 key/value 序列的前缀和边界 | 变长 batch |
| `seqused_k` | Python 侧输入 | kernel | 限定每个 batch item 实际用到多少 K | 截断 / 稀疏 / cache 场景 |
| `block_table` | 推理框架 / cache 管理器 | paged KV kernel | page 到实际 KV block 的映射 | paged KV cache |
| `cache_leftpad` | 推理框架 | kvcache 路径 | cache 的左侧起始偏移 | 左填充布局 |
| `cache_batch_idx` | 推理框架 | kvcache 路径 | batch item 到 cache row 的映射 | 多 batch / 动态复用 cache |
| `cache_seqlens` | 推理框架 | kvcache 路径 | 当前 cache 已写入长度 | 增量解码时最关键 |
| `rotary_cos / rotary_sin` | rotary embedding 模块 | kvcache / rotary 路径 | 位置编码查表 | 推理常见 |
| `alibi_slopes` | 模型层 / wrapper | attention kernel | ALiBi 的 bias 参数 | 需要线性位置偏置时 |

### 3. kernel 调度状态

| 状态 | 含义 | 常见位置 | 记忆点 |
|---|---|---|---|
| `num_splits` | K/V 被切成几份并行算 | split-KV 路径 | 0 常表示“自动决定” |
| `head_size_rounded` | head dim 对齐后的大小 | C++ bridge / launch | 用来满足 kernel 对齐和向量化 |
| `seqlen_q_rounded` | query 长度对齐后的大小 | C++ bridge / launch | 让分块更容易 |
| `seqlen_k_rounded` | key/value 长度对齐后的大小 | C++ bridge / launch | 和 `p`、split path 配合 |
| `window_size_left/right` | 局部注意力窗口 | API / params | 大于序列长度时常转成 `-1` |
| `is_causal` | 是否启用因果掩码 | API / params | 右窗口经常被压成 0 |
| `softcap` | score 压缩 / 截断开关 | API / params | 与 dropout 常有约束关系 |

---

## 术语和状态的对应关系

有些词一看像概念，但其实背后直接对应到状态：

- **online softmax** → `softmax_lse`
- **dropout** → `rng_state`
- **varlen** → `cu_seqlens_q / cu_seqlens_k`
- **paged KV cache** → `block_table`
- **incremental decoding** → `cache_seqlens`
- **MQA / GQA** → `h / h_k`、`dk_expanded / dv_expanded`
- **deterministic backward** → `dq_accum` 的多 split 存储策略
- **rotary** → `rotary_cos / rotary_sin`
- **ALiBi** → `alibi_slopes`

这也是为什么我会把“术语表”和“关键状态表”放在一篇里：
很多时候它们根本不是两类东西，而是同一条数据流的两个视角。

## 一眼看懂 FlashAttention 时的检查清单

当我以后重新读源码，如果看到下面这些词，就应该立刻想到它属于哪一类：

### 训练态优先看

- `softmax_lse`
- `rng_state`
- `dq / dk / dv`
- `dq_accum`
- `deterministic`
- `softmax_d`

### 推理态优先看

- `cache_seqlens`
- `cache_batch_idx`
- `cache_leftpad`
- `block_table`
- `rotary_cos / rotary_sin`
- `alibi_slopes`
- `seqlenq_ngroups_swapped`

### 布局/调度优先看

- `cu_seqlens_q / cu_seqlens_k`
- `num_splits`
- `head_size_rounded`
- `seqlen_q_rounded / seqlen_k_rounded`
- `window_size_left/right`
- `h_h_k_ratio`

## 我自己的理解

1. **FlashAttention 里很多“功能词”其实是布局词**
   - 它们改变的是数据流和并行方式

2. **`softmax_lse` 是训练态最核心的状态摘要**
   - 没有它，exact backward 很难说清楚

3. **`cache_seqlens` 是推理态最核心的状态摘要**
   - 没有它，增量解码的定位就断了

4. **`block_table` 把“cache”从连续内存变成了可分页的数据结构**
   - 这也是长上下文推理能扩展的关键之一

5. **术语表和状态表放一起最有用**
   - 因为理解一个词最好的方式，就是知道它在系统里到底挂在哪个状态上

## 关联笔记

- [FlashAttention 阅读导览](./flash-attention-reading-guide.md)
- [FlashAttention 系统地图](./flash-attention-system-map.md)
- [FlashAttention 源码精读](./flash-attention-source-reading.md)
- [FlashAttention 接口与 Autograd](./flash-attention-interface-and-autograd.md)
- [FlashAttention Kernel 与 Launch 机制](./flash-attention-kernel-and-launch.md)
- [FlashAttention Kernel 细节补充](./flash-attention-kernel-details.md)
- [FlashAttention PyTorch ATen 接入层](./flash-attention-pytorch-aten-integration.md)

## References

- `third_party/flash-attention/README.md`
- `third_party/flash-attention/flash_attn/flash_attn_interface.py`
- `third_party/flash-attention/flash_attn/modules/mha.py`
- `third_party/flash-attention/csrc/flash_attn/flash_api.cpp`
- `third_party/flash-attention/csrc/flash_attn/src/flash.h`
- `third_party/flash-attention/csrc/flash_attn/src/flash_fwd_launch_template.h`
- `third_party/flash-attention/csrc/flash_attn/src/flash_bwd_launch_template.h`
- `third_party/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h`
- `third_party/flash-attention/csrc/flash_attn/src/flash_bwd_kernel.h`
- `aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp`
