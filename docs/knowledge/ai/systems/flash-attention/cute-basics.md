---
title: CUTE 入门：从 C CUDA 视角理解 flash-attn 的 kernel 写法
type: concept
status: growing
tags: [AI, CUDA, CUTE, CUTLASS, FlashAttention, Kernel]
created: 2026-08-25
updated: 2026-09-01
source: third_party/cutlass/include/cute/ + third_party/flash-attention/csrc/flash_attn/src/
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

## 概念 3：`GmemTiledCopyO` + `partition_D` —— 按当前 CTA 的线程布局拆分全局写回

这是 `flash_fwd_kernel.h` 里最抽象的三连：

```cpp
typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;        // 1. 整个 CTA 的拷贝方案
auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx); // 2. 取线程 tidx 的方案
Tensor tOgO = gmem_thr_copy_O.partition_D(gO);                    // 3. 取该线程的目标视图
```

可以把三层对象分别理解成：

```text
GmemTiledCopyO          全班座位表：整个 CTA 如何共同搬一个 tile
get_thread_slice(tidx)  找到第 tidx 个线程在座位表中的位置
partition_D(gO)         按该位置列出这个线程要写的所有 O 坐标
```

`partition_D` 的 `D` 是 **Destination**。它不搬数据，也不新建一份 `gO`；它只是从目标张量 `gO` 中生成一个线程私有的逻辑视图。真正的寄存器到全局内存写回发生在后面的 `cute::copy`。

### 为什么当前代码经常是 128 个线程

先分清两个问题：

1. **这个 kernel 的一个 CTA 有多少线程？**由 `Kernel_traits` 的 `kNWarps` 决定；
2. **这些线程如何拆分 copy tile？**由 `GmemTiledCopyO` 中的 thread layout 和 value layout 决定。

`kernel_traits.h` 中有：

```cpp
static constexpr int kNWarps = kNWarps_;
static constexpr int kNThreads = kNWarps * 32;
```

CUDA 一个 warp 固定有 32 个线程。许多前向 specialization 传入 `kNWarps_ = 4`，例如：

```cpp
Flash_fwd_kernel_traits<Headdim, 128, 64, 4, ...>
//                                            ↑ 4 个 warp
```

所以该 specialization 的 CTA 线程数为：

$$
kNThreads = kNWarps \times 32 = 4 \times 32 = 128
$$

因此，原先所说的「分给 128 个线程」只适用于这些 **4-warp specialization**。`partition_D` 本身没有「必须使用 128 个线程」的规则。例如代码中 head dimension 较大的某些 specialization 使用 8 个 warp，此时：

$$
kNThreads = 8 \times 32 = 256
$$

同一个 `partition_D` 接口仍然成立，只是它按照新的 256-thread layout 重新分工。

> 为什么选择 4 个而不是 2 个或 8 个 warp？这是 kernel 调优结果，而不是 CuTe 语义：更多线程可能增加并行度，却也可能增大寄存器、共享内存、同步和 occupancy 压力。源码甚至记录了某些形状下 8 warps 比 4 warps 更慢；另一些较大 head dimension 的形状则确实选用 8 warps。

### `GmemTiledCopyO` 到底编码了什么

前向 traits 中的定义可以拆成四步：

```cpp
static constexpr int kGmemElemsPerLoad =
    sizeof(cute::uint128_t) / sizeof(Element);

static constexpr int kGmemThreadsPerRow =
    kBlockKSmem / kGmemElemsPerLoad;

using GmemLayoutAtom = Layout<
    Shape<Int<kNThreads / kGmemThreadsPerRow>,
          Int<kGmemThreadsPerRow>>,
    Stride<Int<kGmemThreadsPerRow>, _1>
>;

using GmemTiledCopyO = decltype(make_tiled_copy(
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
    GmemLayoutAtom{},
    Layout<Shape<_1, _8>>{}
));
```

这几部分的职责不同：

| 组成 | 回答的问题 |
|---|---|
| `Copy_Atom<...>` | 一次底层 copy 使用什么元素类型和向量化方式？ |
| `GmemLayoutAtom` | CTA 中的线程沿 tile 的行、列方向怎样排列？ |
| `Layout<Shape<_1, _8>>` | 每个线程在一个 copy atom 中持有怎样的 value 布局？ |
| `make_tiled_copy(...)` | 如何把上述原子模式平铺到完整目标张量？ |

这里的 `128` 容易产生第二种误解：

```cpp
AutoVectorizingCopyWithAssumedAlignment<128>
```

它允许 copy 算法假定指针和动态 layout **最多具有 128-bit（16-byte）对齐**，并在源/目标的共同连续布局允许时选择不超过 128 bit 的向量宽度；它不是线程数。对这里连续的 8 个 FP16/BF16 元素，通常正好可以形成一次 128-bit copy。恰好 CTA 也常有 128 个线程，但两个「128」来自完全不同的维度：

```text
kNThreads = 128                         → 128 个 CUDA 线程
AssumedAlignment<128> = 128 bits = 16 B → 内存对齐/向量化粒度
```

### 例 3A：FP16、`kHeadDim = 64` 时为什么是 `Shape<16, 8>`

`Element` 是 FP16/BF16 时，每个元素为 2 byte。一份 128-bit 数据包含：

$$
kGmemElemsPerLoad = \frac{128\ \mathrm{bit}}{16\ \mathrm{bit/element}}
= \frac{16\ \mathrm{byte}}{2\ \mathrm{byte/element}} = 8\ \mathrm{elements}
$$

当 `kBlockKSmem = 64` 时：

$$
kGmemThreadsPerRow = \frac{64}{8} = 8
$$

若该 specialization 使用 4 warps，即 `kNThreads = 128`，那么：

$$
\frac{kNThreads}{kGmemThreadsPerRow} = \frac{128}{8} = 16
$$

于是 thread-layout atom 是：

```cpp
GmemLayoutAtom = Layout<Shape<_16, _8>, Stride<_8, _1>>;
//                                ↑ 8 个线程共同覆盖一行的 64 个元素
```

可以把一个 atom 画成下面这样。每个 `Tn` 表示线程 `n` 负责一个连续 8-element 向量，而不是只负责一个标量：

```text
逻辑行  0: T0   T1   T2   T3   T4   T5   T6   T7
            0-7  8-15 16-23 ...             56-63
逻辑行  1: T8   T9   T10  T11  T12  T13  T14  T15
逻辑行  2: T16  T17  T18  T19  T20  T21  T22  T23
...
逻辑行 15: T120 T121 T122 T123 T124 T125 T126 T127
```

所以准确说法不是「thread 0 写第 0 行、thread 1 写第 1 行」，而是：

- 8 个线程协作覆盖一行的 64 个 FP16/BF16 元素；
- 128 个线程一次覆盖 16 行；
- 每个线程在一行中负责一个连续的 8-element、16-byte 片段。

### 例 3B：完整 `128 × 64` tile 如何重复这个 atom

上面的 thread-layout atom 一轮覆盖 `16 × 64` 个元素，而 `gO` 有 `128 × 64` 个元素。因此这个模式会沿行方向重复：

```text
第 0 轮：128 个线程共同处理行   0-15
第 1 轮：128 个线程共同处理行  16-31
第 2 轮：128 个线程共同处理行  32-47
...
第 7 轮：128 个线程共同处理行 112-127
```

以 thread 3 为例，它的目标片段不是一整行，而是类似：

```text
(row 0,   col 24-31)
(row 16,  col 24-31)
(row 32,  col 24-31)
...
(row 112, col 24-31)
```

所以 `partition_D(gO)` 之后的 `tOgO` 通常是一个 rank-3 视图，而不是简单的一维数组。可以用以下**心智模型**理解它的模式：

```text
tOgO(CPY, m, k)
      │    │  └─ copy atom 沿 K 方向平铺后的第 k 组
      │    └──── copy atom 沿 M 方向平铺后的第 m 组
      └───────── 一次 copy 内的 value mode（这里通常是 8 个元素）
```

确切的 mode 大小由 CuTe 对 layout 的合成结果决定，不应把所有 specialization 都硬记成某个固定 shape；真正稳定的是：`tOgO` 枚举了当前线程在整个 `gO` tile 中负责的全部目标位置。

### 例 3C：`kHeadDim = 32` 时线程分工会变化

当 `kBlockKSmem = 32`、元素仍是 FP16/BF16 时：

$$
kGmemThreadsPerRow = \frac{32}{8} = 4
$$

4-warp specialization 中：

$$
\frac{128}{4} = 32
$$

因此 thread layout 变为概念上的 `Shape<32, 4>`：

```text
每行：4 个线程 × 每线程 8 个元素 = 32 个元素
一轮：32 行 × 4 个线程 = 128 个线程
```

这说明即使 CTA 仍是 128 个线程，每行参与写回的线程数也会随 copy tile 的列宽变化。`partition_D` 不是按一个固定公式「每线程一行」切，而是服从 `GmemTiledCopyO` 编码的布局。

### 例 3D：8-warp specialization 中为什么变成 256 个线程

假设 `kNWarps = 8`，并且 `kBlockKSmem = 64`：

$$
kNThreads = 8 \times 32 = 256, \qquad
kGmemThreadsPerRow = \frac{64}{8} = 8
$$

那么：

$$
GmemLayoutAtom.shape = \left(\frac{256}{8}, 8\right) = (32, 8)
$$

即 8 个线程协作覆盖一行，一轮可以覆盖 32 行。调用代码完全不变：

```cpp
auto thr = gmem_tiled_copy_O.get_thread_slice(tidx);
Tensor tOgO = thr.partition_D(gO);
```

变化发生在 `GmemTiledCopyO` 的类型内部。这正是 CuTe 类型化布局的价值：kernel 主体声明「按本 specialization 的 copy layout 分目标」，而不用分别手写 128-thread 和 256-thread 的索引公式。

### 例 3E：尾行 —— 最后一个 Q tile 不满 `kBlockM`

假设 `kBlockM = 128`、实际 Query 长度为 150。第 0 个 CTA 处理逻辑行 `0-127`，第 1 个 CTA 的局部 tile 仍然按编译期 shape 描述 128 行，但其中只有前 22 行合法：

$$
max\_M = 150 - 1 \times 128 = 22
$$

`partition_D` 仍会把完整的 `128 × kHeadDim` 逻辑 tile 分给线程；它不负责判断运行时边界。边界由同样分区的坐标张量和 `copy` helper 判断：

```cpp
Tensor cO   = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));
Tensor tOcO = gmem_thr_copy_O.partition_D(cO);

if (get<0>(tOcO(0, m, 0)) < max_M) {
    // 只写局部行 0-21
}
```

例如在前面的 `Shape<16, 8>` 布局中，某线程会每隔 16 行拿到一个片段；若它从局部行 6 开始，则 `tOgO` 包含局部行 6、22、38、...。同位置的 `tOcO` 会给出这些行号；只有行 6 通过，22 及之后被跳过。这样既保留规则的编译期 tile，又避免越界写全局内存。

### 例 3F：尾列 —— 实际 `params.d` 小于编译期 `kHeadDim`

kernel specialization 的 `kHeadDim` 是编译期上界。例如实际 `params.d = 80` 可选择 `kHeadDim = 96` 的 specialization。接口要求 head dimension 是 8 的倍数，所以向量边界与每线程 8-element value 布局对齐。列谓词来自：

```cpp
Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
for (int k = 0; k < size(tOpO); ++k) {
    tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
}
```

结果是：

```text
列  0-79  → predicate = true，允许写入
列 80-95  → predicate = false，跳过写入
```

这里再次能看到职责分离：

- `partition_D` 决定**本线程可能负责哪些槽位**；
- `tOcO` 告诉代码**这些槽位的逻辑坐标是什么**；
- 行/列 predicate 决定**当前运行时尺寸下哪些槽位真的合法**。

### 为什么数据张量和坐标张量必须使用同一个 `partition_D`

这两行是一个成对设计：

```cpp
Tensor tOgO = gmem_thr_copy_O.partition_D(gO); // 目标数据位置
Tensor tOcO = gmem_thr_copy_O.partition_D(cO); // 同位置的逻辑坐标
```

因为 `gO` 与 `cO` 具有相同的逻辑 shape，又使用同一个 thread slice 和同一个 destination partition，所以二者逐槽对应：

```text
tOgO(_, m, k)  ← 真正要写的全局地址
   ↕ 同一线程、同一 (m, k)
tOcO(_, m, k)  ← 该地址对应的 (row, col)
```

如果两者使用不同的 partition，程序就可能拿 A 槽位的坐标去判断 B 槽位的地址，边界 mask 会失去意义。

### 把整个流程翻译成 C 风格 CUDA

CuTe 写法：

```cpp
auto thr = gmem_tiled_copy_O.get_thread_slice(tidx);
Tensor dst   = thr.partition_D(gO);
Tensor coord = thr.partition_D(cO);
copy(gmem_tiled_copy_O, src, dst, coord, predicate_K, max_M);
```

概念上等价于：

```cpp
for (每个由 copy layout 分给 tidx 的 row_group) {
    int row = /* 由 thread layout 和平铺次数计算 */;
    int col = /* 由 thread-in-row 和 value layout 计算 */;

    if (row < max_M && col < params.d) {
        // 一次写当前线程负责的连续向量；FP16/BF16 常为 8 个元素
        vector_store_16B(&gO[row][col], zero_vector);
    }
}
```

不同之处在于，CuTe 把 `row`、`col`、平铺次数和向量值布局编码在类型中，使编译器可以常量折叠并展开循环，也让同一份 kernel 主体适配不同的 tile/warp specialization。

### 小结：不要混淆三个层次

| 层次 | 当前典型值 | 谁决定 |
|---|---:|---|
| CTA 线程数 | 128（4 warps）或 256（8 warps） | `kNWarps × 32` |
| 每行参与 copy 的线程数 | 4 或 8 | `kBlockKSmem / kGmemElemsPerLoad` |
| 每线程每个 copy atom 的元素数 | FP16/BF16 下通常为 8 | 128-bit copy 与 value layout |

最重要的结论是：**`partition_D` 不等于「平均切成 128 份」**。它真正做的是：依据当前 `TiledCopy` 的线程布局和值布局，把目标张量中属于线程 `tidx` 的所有坐标组织成一个线程私有视图。

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

**这正是 early-exit 分支末尾在做的事**：用坐标张量拿到局部 tile 的真实行号，决定 `gLSE(row) = INFINITY`。

## 回到 `flash_fwd_kernel.h` 的 early-exit 分支：连起来看

现在整段就"读得懂"了。这段在 early-exit 分支（本 CTA 没有可算的 K/V），作用是把 O 写 0、LSE 写 +∞：

| 代码阶段 | cute 概念 | 在干什么 |
|---|---|---|
| 声明 `GmemTiledCopyO` | tiled copy | 声明"写全局 O 的方式"（线程布局+向量宽） |
| `get_thread_slice` / `partition_D` | 线程切片 | 算当前线程负责的全局槽位 `tOgO` |
| 寄存器张量 + `clear` | register fragment | 寄存器里凑一份全 0 |
| identity tensor + 同一 partition | 坐标视图 | 造"坐标地图" `tOcO`，用于边界判断 |
| `tOpO` | bool 掩码 | 算 K 维（head dimension）越界标记 |
| `FLASH_NAMESPACE::copy` | predicated copy | 循环 + 边界 mask，把 0 向量化写回全局 |
| 最后的行循环 | 坐标张量 | 每行由拥有逻辑列 0 的唯一线程写 `gLSE = INFINITY` |

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
- cute 的 `#pragma unroll` 展开对寄存器压力/occupancy 的影响

## Related Knowledge

- [CUTE TiledMMA：一条 MMA 指令如何"复制"成大 tile](./cute-tiled-mma.md) — TiledMMA 的线程/数值复制、片段分配与 gemm 展开（回答了本页原先的 TiledMMA Open Question）
- [FlashAttention 源码精读](./flash-attention-source-reading.md) — kernel 上层的完整链路
- [FlashAttention Kernel 与 Launch 机制](./flash-attention-kernel-and-launch.md) — tile、launch 与 split-KV
- [FlashAttention Kernel 细节补充](./flash-attention-kernel-details.md) — backward 与 RNG
- [源码阅读方法](../../../learning/code-reading/)

## References

- `third_party/cutlass/include/cute/tensor.hpp`、`layout.hpp`、`tensor_impl.hpp`
- `third_party/cutlass/include/cute/atom/copy_atom.hpp`
- `third_party/cutlass/include/cute/algorithm/copy.hpp`
- `third_party/flash-attention/csrc/flash_attn/src/kernel_traits.h`
- `third_party/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h`
- `third_party/flash-attention/csrc/flash_attn/src/flash_fwd_launch_template.h`
- CUTLASS 官方文档与 cute 教程：https://github.com/NVIDIA/cutlass
