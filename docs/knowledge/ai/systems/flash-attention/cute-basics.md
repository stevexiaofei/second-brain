---
title: CUTE 入门：从 C CUDA 视角理解 flash-attn 的 kernel 写法
type: concept
status: growing
tags: [AI, CUDA, CUTE, CUTLASS, FlashAttention, Kernel]
created: 2026-08-25
updated: 2026-08-25
source: third_party/flash-attention/csrc/cutlass/include/cute/ + flash_fwd_kernel.h
---

# CUTE 入门：从 C CUDA 视角理解 flash-attn 的 kernel 写法

## 一句话理解

CUTE（CUDA Tensor Engine）是 CUTLASS 附带的**张量模板库**。它用 C++ 模板把"张量的形状、步长、线程分工、内存类型"全部编码进**类型**里，让你写 kernel 时**不写裸指针算术**，而是直接表达"我要对哪个视图做哪件事"。

如果你只写过 C 风格的 CUDA（`data[offset]`、`__syncthreads()`、手算索引），这篇笔记帮你把 flash_fwd_kernel.h 里看到的 cute 写法逐个翻译成你熟悉的语言。

> 目标读者：写过简单 C CUDA 算子、但没接触过 CUTLASS/cute 的人。

## 为什么 cute 会"劝退"新手

| C 风格 CUDA | cute 风格 | 初次感受 |
|---|---|---|
| `float *gO; gO[row * stride + col]` | `Tensor gO = make_tensor(ptr, shape, stride)` | 多一层包装 |
| 手写循环 + 手工算 offset | `local_tile` / `partition_D` 自动切块 | 抽象太"厚" |
| `int tid = threadIdx.x;` 自己规划 | `get_thread_slice(tidx)` 交给库 | 失去控制感 |
| 类型就是 `float*` | `Tensor<Engine, Layout>` 模板套模板 | 编译错误天书 |

**关键认知**：cute 没有引入新算法，只是把 C 里"你在心里做、在代码里手写"的索引数学，变成**类型和少量组合子**。先搞懂 3 个概念（Tensor / Layout / 线程切片），后面全是对它们的组合。

## 概念 0：一条主线 —— Tensor = Engine + Layout

cute 的 `Tensor<Engine, Layout>` 就两个成员，理解它一切就通了：

```
Tensor<Engine, Layout>
   Engine = 数据在哪 + 怎么访问    （全局指针 gmem / 共享内存 smem / 寄存器 regs）
   Layout = 形状(shape) + 步长(stride)，即"逻辑坐标 → 物理偏移"的映射
```

类比：`Tensor` 像一个"加了地图的指针"。`Engine` 是起点和内存类型，`Layout` 是这张地图（坐标怎么走到偏移）。

### 例 0：一个 3×2 行主序矩阵

```cpp
// 物理内存（C 视角）
// 地址 0..5 依次是 [a00 a01 a02 a10 a11 a12]
float raw[6] = {0,1,2,3,4,5};

// cute 视角
Tensor T = make_tensor(raw, make_shape(2, 3), make_stride(3, 1));
// 形状 (2, 3)：2 行 3 列
// 步长 (3, 1)：行索引每 +1 走 3 个元素，列索引每 +1 走 1 个元素
// T(1, 2)  == raw[1*3 + 2] == raw[5] == 5
```

**看懂两个工厂函数就入门一半**：

- `make_shape(2, 3)` → 形状元组 `(2, 3)`
- `make_stride(3, 1)` → 步长元组 `(3, 1)`

`make_tensor` 把三者打包：`make_tensor(指针, make_shape(...), make_stride(...))`。
还有配套的 `make_coord(1, 2)`（坐标元组）、`make_layout(shape, stride)`（合并成一个 Layout）。

## 概念 1：`_`（下划线）是"全取"通配符

你会在代码里到处看到 `mO(_, bidh, _)`。`_` 表示**这个维度全要**。

### 例 1：从 3 维切出"某个 head"

```cpp
// mO 形状 (seqlen_q, num_heads, head_dim)
Tensor mO = make_tensor(ptr, make_shape(seqlen_q, num_heads, head_dim),
                             make_stride(row_stride, head_stride, 1));

// C 视角：取第 bidh 个 head 的所有行 → (seqlen_q, head_dim)
// C 手写：需要自己算起始指针 + 行步长
// cute：直接说"我要第 bidh 个 head"
Tensor head = mO(_, bidh, _);   // 形状自动变成 (seqlen_q, head_dim)
```

固定一个维度（给具体数字）→ 该维消失；给 `_` → 该维保留。

## 概念 2：local_tile —— 把大张量切成"当前 CTA 的那块"

### 例 2：在大矩阵上切出一个 kBlockM × kBlockN 的块

```cpp
// gO：全局 O，形状 (seqlen_q, head_dim)，head_dim 已被上一节切掉
// 每个 CTA 负责第 m_block 个 Q-tile（kBlockM 行）
Tensor gO = local_tile(mO(_, bidh, _),
                       Shape<Int<kBlockM>, Int<kHeadDim>>{},  // 块大小
                       make_coord(m_block, 0));               // 块的坐标（第几行块，第几列块）
// 结果：一个 (kBlockM, kHeadDim) 的视图，指回全局内存对应位置
```

**等价理解**（C 视角）：

```cpp
// 相当于"声明我要处理的子矩阵起点"
float *block_ptr = mO_ptr + bidh*head_stride + m_block*kBlockM*row_stride;
// 然后所有访问都基于这个起点
```

`local_tile` 的第二个参数是块尺寸、第三个是块的索引，返回值就是你那块。**它是视图不是拷贝**——数据还在全局内存，只是帮你把坐标算好了。

> 题外话：`Shape<Int<kBlockM>, Int<kHeadDim>>{}` 这种写法表示"编译期已知的常量尺寸"，能让编译器全展开、常数折叠。这就是 cute 性能的来源之一。

## 概念 3：GmemTiledCopyO + partition_D —— 把"写全局"的任务分给 128 个线程

这是 flash_fwd_kernel.h 第 104-106 行的三连，也是最抽象的一段：

```cpp
typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;   // 1. 拷贝器（无状态，只带类型信息）
auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);  // 2. 我要的那份
Tensor tOgO = gmem_thr_copy_O.partition_D(gO);              // 3. 我的"目标槽位"视图
```

### 例 3：把 128×64 的 O 分给 128 个线程

设想一个简单 tiled copy：128 线程，每线程负责 8 列（每列 1 个元素），沿 M 维排布：

```
gO (128 行, 64 列)
  ↓ partition_D 按线程切
tOgO：每个线程看到 (1, 1, 8) 形状
      模式：thread 0 → 行 0，thread 1 → 行 1，... 每线程 8 列
```

- `partition_D` 的 **D = Destination**，即"我负责写入的目标"
- `tOgO` 是这个线程要写的全局位置视图，之后 `cute::copy` / 写操作都基于它

**为什么这么设计**：你不需要手动算"第 tidx 个线程写哪几列"——库按 tiled copy 里编码的线程布局自动分好。改线程布局只需改一处类型。

## 概念 4：寄存器缓冲 —— `make_tensor<Element>(shape)`

```cpp
Tensor tOrO = make_tensor<Element>(shape(tOgO));  // 在寄存器里开一块和 tOgO 同形状的缓冲
clear(tOrO);                                       // 全部填 0
```

`make_tensor<Element>(形状)` 与前面不同：**不给指针**，让 cute 在寄存器里分配（`Engine` 变成 `ArrayEngine`）。`tOrO` 的 r = register，是每个线程私有的"局部片段"。

### 例 4：先凑齐 0，再一次性写回

```cpp
// 我想把 gO 的某一块写成 0
Tensor tOrO = make_tensor<Element>(shape(tOgO));  // 寄存器里先凑 0
clear(tOrO);                                      // = 逐个置 0
copy(gmem_tiled_copy_O, tOrO, tOgO);              // 寄存器 → 全局（按线程分工 + 向量化）
```

> 为什么非要绕一圈寄存器：全局内存写要按向量宽度（比如一次 16B）才高效，寄存器里凑好连续数据再整块写，比逐元素写快得多。这也是"tiled copy"存在的意义。

## 概念 5：make_identity_tensor —— "存坐标"的张量，用来做边界判断

```cpp
Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));
// cO(i, j) 的"值"就是坐标 (i, j) 本身
```

普通张量存数据，`identity_tensor` 存的是**坐标**。配合**同一个** partition 切分，就能知道"我这个线程的每个槽位对应全局哪一行哪一列"。

### 例 5：判断该不该写（越界保护）

```cpp
// tOcO 和 tOgO 用同一 partition 切出来 → 一一对应
Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
// tOcO(0, m, 0) 返回坐标 (row, col)，get<0> 取行号
if (get<0>(tOcO(0, m, 0)) < actual_seqlen_q) {
    // 行号合法才写，否则跳过（避免越界写）
}
```

**这正是 flash_fwd_kernel.h 第 122-126 行干的事**：用坐标张量拿到真实行号，决定 `gLSE(row) = INFINITY`。

## 回到 flash_fwd_kernel.h 第 104-127 行：连起来看

现在整段就"读得懂"了。这段在 early-exit 分支（本 CTA 没有可算的 KV），作用是把 O 写 0、LSE 写 +∞：

| 行 | cute 概念 | 在干什么 |
|---|---|---|
| 104 | tiled copy | 声明"写全局 O 的方式"（线程布局+向量宽） |
| 105-106 | get_thread_slice / partition_D | 算我这个线程负责的全局槽位 `tOgO` |
| 107-108 | 寄存器张量 + clear | 寄存器里凑一份全 0 |
| 110-112 | identity_tensor + 同 partition | 造"坐标地图" `tOcO`，用于边界判断 |
| 113-117 | bool 掩码 | 算 K 维（head_dim）越界标记 |
| 119-121 | FLASH_NAMESPACE::copy | 循环 + 边界 mask，把 0 向量化写回全局 |
| 122-126 | 坐标张量 | 每行写 `gLSE = INFINITY` |

**核心心智模型**（一句话）：cute 用"**数据视图 + 坐标视图 + 同一套线程分区**"三个东西，把"谁、在哪、能不能写"全部显式化；你不再手写索引，而是声明式地表达意图。

## 接下来怎么看更多代码

1. 遇到 `make_tensor/make_shape/make_stride` → 在构造视图（概念 0）
2. 遇到 `_(...)` → 切维/固定维（概念 1）
3. 遇到 `local_tile` → 当前 CTA 的块（概念 2）
4. 遇到 `partition_S/D` → 线程分工（S=源/D=目标）（概念 3）
5. 遇到 `copy` / `clear` → 按 tiled copy 在内存间搬运（概念 4）
6. 遇到 `identity_tensor` → 边界判断用的坐标地图（概念 5）

## 我的理解

- cute 的价值不在"快"，而在**可组合性**：换 tile 大小、换线程数、换内存层次，改类型参数即可，索引数学交给库
- 对只写过 C CUDA 的人来说，最大的障碍是"代码没有显式的 for 循环"——其实循环被 `#pragma unroll` + 类型折叠"藏"进模板展开里了，跑的还是同样的指令
- 建议的练习路径：用 cute 写一个"矩阵清零 kernel"（就是例 4 的完整版），再写一个"取子块相加"，就基本能读懂 flash_fwd_kernel.h 的前半段了

## Open Questions

- `Swizzle`（swizzle 布局）如何与 bank conflict 对应？
- `TiledMMA` 的 `AtomLayout` 与 warp 内 MMA 分工怎么映射到硬件？
- cute 的 `#pragma unroll` 展开对寄存器压力/occupancy 的影响

## Related Knowledge

- [FlashAttention 源码精读](./flash-attention-source-reading.md) — kernel 上层的完整链路
- [FlashAttention Kernel 与 Launch 机制](./flash-attention-kernel-and-launch.md) — tile、launch 与 split-KV
- [FlashAttention Kernel 细节补充](./flash-attention-kernel-details.md) — backward 与 RNG
- [源码阅读方法](../../../learning/code-reading/)

## References

- `third_party/flash-attention/csrc/cutlass/include/cute/tensor.hpp`、`layout.hpp`、`tensor_impl.hpp`
- `third_party/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h`
- CUTLASS 官方文档与 cute 教程：https://github.com/NVIDIA/cutlass
