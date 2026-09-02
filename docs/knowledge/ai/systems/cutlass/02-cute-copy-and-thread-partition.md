---
title: CUTLASS/CuTe 02：Copy Atom、TiledCopy 与线程分区
type: concept
status: seed
tags: [AI, CUDA, CUTE, CUTLASS, Copy, TiledCopy, Thread]
created: 2026-09-02
updated: 2026-09-02
source: third_party/cutlass/include/cute/atom/copy_atom.hpp、cute/algorithm/copy.hpp；知乎 CUTLASS 系列来源待逐篇核对
---

# CUTLASS/CuTe 02：Copy Atom、TiledCopy 与线程分区

## 一句话理解

`TiledCopy` 描述整个线程块如何协作搬运一个 tile；`get_thread_slice(tid)` 取出一个线程的局部分工；`partition_S` 和 `partition_D` 再分别生成 source 与 destination 的 Tensor view。它们是“谁搬哪些元素”的类型化表达，不是一次自动发生的拷贝。

## 从 Copy Atom 到 TiledCopy

可以把层次看成：

```text
Copy Atom      一次底层 copy 的元素类型、向量宽度和基本契约
线程布局       哪些线程参与、线程如何排布
value 布局     每个线程一次携带哪些值
TiledCopy      把上述模式平铺到完整 tile
thread slice   某个 tid 在这套模式中的局部视图
```

典型形态：

```cpp
using CopyAtom = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
auto tiled_copy = make_tiled_copy(CopyAtom{}, thread_layout, value_layout);
auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
```

`128` 在 `AutoVectorizingCopyWithAssumedAlignment<128>` 中表示最多按 128 bit（16 byte）对齐/向量化的假设，不表示 128 个线程。线程数由 thread layout 和 launch 配置决定。

## `partition_S` 与 `partition_D`

```cpp
Tensor tAgA = thr_copy.partition_S(gA);
Tensor tAsA = thr_copy.partition_D(sA);
```

读法是：

```text
tAgA：当前线程从 global A 读取的 source 槽位
tAsA：当前线程向 shared A 写入的 destination 槽位
```

其中 `S` 是 Source，`D` 是 Destination。两个 partition 使用同一套线程和值布局，使 source 与 destination 的逻辑槽位可以对应：

```cpp
for (每个被 tid 分配的 value) {
    tAsA[value] = tAgA[value];
}
```

真正的实现由 `cute::copy(tiled_copy, tAgA, tAsA)` 展开；partition 本身主要创建视图。

## 命名约定

CUTLASS 代码常用以下形式帮助读者恢复数据流：

```text
t + partition pattern + storage + logical object
```

例如：

| 名字 | 含义 |
|---|---|
| `gA` | global-memory A tile |
| `sA` | shared-memory A tile |
| `rA` | register A fragment |
| `tAgA` | `tA` copy partition applied to `gA` |
| `tAsA` | `tA` copy partition applied to `sA` |
| `tArA` | 与 `tA` 对应的 register fragment |

这套名字是约定而不是语言规则。遇到不熟悉的变量，必须结合构造 API、来源 Tensor 和后续操作确认含义。

## FlashAttention 中的 Q/K/V 实例

`flash_fwd_kernel.h` 中的代码是同一模式在实际 attention kernel 中的应用：

```cpp
typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
```

其中：

- `gQ/gK/gV` 是 global-memory 中当前 CTA 的 tile；
- `sQ/sK/sV` 是 shared-memory staging tile；
- `gmem_thr_copy_QKV` 是 QKV copy scheme 中当前线程的 slice；
- `tQgQ`、`tKgK`、`tVgV` 是当前线程要读取的 source 槽位；
- `tQsQ`、`tKsK`、`tVsV` 是当前线程要写入的 destination 槽位。

因此 Q 的搬运可读成：

```text
gQ --partition_S--> tQgQ --copy--> tQsQ <--partition_D-- sQ
```

K 和 V 完全类似。`Q/K/V` 是逻辑对象，`g/s` 是存储层次；不要把第二个字母误解成固定的 GEMM A/B/C 语法。

## 边界谓词：partition 不负责判断合法性

一个编译期 tile 可能覆盖运行时矩阵之外的尾行或尾列。常见做法是并行创建数据 view 与坐标 view：

```cpp
Tensor tOgO = thr_copy.partition_D(gO);
Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));
Tensor tOcO = thr_copy.partition_D(cO);
```

因为二者用了同一个 thread slice 和 destination partition，`tOcO` 的每个槽位与 `tOgO` 的目标地址一一对应。代码可以根据坐标判断行列是否小于实际尺寸，再调用 predicated copy。

职责分离如下：

```text
partition：决定本线程可能负责哪些槽位
identity tensor：告诉这些槽位的逻辑坐标
predicate：决定当前运行时尺寸下哪些槽位真正合法
copy：执行通过谓词检查的搬运
```

## 常见误区

1. **`partition_D` 会立刻搬数据**：错，它主要建立 destination view。
2. **所有线程平均各拿一个标量**：错，向量化 copy 中一个线程可能拥有连续多个元素，布局还可能跨多个 tile 重复。
3. **`128` 一定是 128 threads**：错，向量化对齐参数和线程数是两件事。
4. **global/shared 必须使用相同物理 layout**：错，逻辑对应可以相同，物理 layout 往往为合并访问、bank conflict 和 MMA 要求而不同。
5. **partition 名字决定 C++ 类型**：错，真正语义来自 layout、engine 和 API。

## 我的理解

TiledCopy 的难点不是“复制”本身，而是把一个 copy 操作拆成三个可验证问题：整个 CTA 的覆盖范围、当前线程的责任、当前 value 的合法坐标。读懂这三个层次后，复杂的 `partition_S/D` 就不再像隐藏的魔法索引。

## Related

- [CUTLASS/CuTe 01：Tensor、Layout 与坐标映射](./01-cute-tensor-and-layout.md)
- [CUTLASS/CuTe 03：TiledMMA 与 fragment](./03-cute-tiled-mma.md)
- [CUTLASS/CuTe 04：从 global → shared → MMA 串起 GEMM](./04-cute-gemm-pipeline.md)
- [CUTE 入门：FlashAttention 中的 GmemTiledCopyO 与边界处理](../flash-attention/cute-basics.md)

## References

- `third_party/cutlass/include/cute/atom/copy_atom.hpp`
- `third_party/cutlass/include/cute/algorithm/copy.hpp`
- `third_party/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h`
- `third_party/flash-attention/csrc/flash_attn/src/utils.h`
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [知乎 CUTLASS 系列来源](https://zhuanlan.zhihu.com/p/1937220431728845963)（当前环境无法稳定读取，篇目待核对）
