---
title: CUDA 初学者学习路径与最小实验
type: concept
status: seed
tags: [CUDA, GPU Programming, Beginner, Warp, SM, Memory Hierarchy, GEMM, Softmax, FlashAttention]
created: 2026-08-21
updated: 2026-08-21
source:
  - https://chatgpt.com/share/6a8827ee-1f08-83e8-a600-0e5dc9dc2a4d
related:
  - ./cuda-hardware-and-programming-model-map.md
  - ../knowledge/ai/systems/flash-attention/flash-attention-reading-guide.md
---

# CUDA 初学者学习路径与最小实验

> 这是一段 ChatGPT 对话的整理稿。对话的核心建议是：现有 CUDA 研究地图的方向正确，但初学者不应从高级硬件名词开始，而应从最小代码、运行现象和逐步实验建立硬件直觉。

## 对话上下文

- 用户请求：为 CUDA 硬件与编程模型研究地图补充适合初学者理解的细节。
- 对话判断：原地图覆盖了硬件、执行模型、内存、同步、PTX/SASS、性能和学习路线，但概念密度较高，容易出现“需要先懂名词才能看懂名词”的问题。
- 主要建议：不推翻原地图，而是在每个概念下补充“是什么、为什么、最小例子、硬件上发生什么、常见误区、怎么实验”。

## 1. 先理解 CUDA 要解决什么问题

CUDA 的第一层直觉不是记住 GPC、TPC 或 TMA，而是理解如何把一个大问题拆成大量相似的小任务。

例如向量加法：

```text
C[i] = A[i] + B[i]
```

CPU 可以通过循环依次处理每个 `i`，而 GPU 更适合让不同线程分别负责不同的 `i`：

```text
线程 0  → C[0] = A[0] + B[0]
线程 1  → C[1] = A[1] + B[1]
线程 2  → C[2] = A[2] + B[2]
...
```

因此，一个 CUDA kernel 不是“执行一次普通函数”，而是启动一批线程，让每个线程执行同一段 kernel 代码并处理自己的数据。

CUDA 程序通常同时包含两个世界：

```text
CPU / Host
    │
    │ kernel launch
    ▼
GPU / Device
    ├── Thread
    ├── Block
    ├── Grid
    └── GPU Memory
```

`kernel<<<grid, block>>>(...)` 看起来像函数调用，实际表达的是一次 GPU 工作提交。

## 2. 从 Thread、Block、Grid 开始

推荐先从最简单的启动方式建立层次：

```cpp
hello<<<1, 1>>>();
```

表示：

```text
1 个 Grid
└── 1 个 Block
    └── 1 个 Thread
```

改为：

```cpp
hello<<<1, 4>>>();
```

就是一个 block 中的 4 个线程执行同一个 kernel。

### `threadIdx`：线程在 Block 中的位置

```cpp
__global__ void test() {
    printf("%d\n", threadIdx.x);
}
```

启动 `test<<<1, 4>>>()` 时，线程分别看到 `threadIdx.x = 0, 1, 2, 3`。

### `blockIdx`：当前 Block 在 Grid 中的位置

启动 `test<<<3, 4>>>()` 时，Grid 中有 3 个 block，每个 block 有 4 个线程。`threadIdx.x` 在每个 block 内都会从 0 开始，因此需要结合 `blockIdx.x` 得到全局下标：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

这个公式表达的是：

> 把 `(block, thread)` 的层次坐标映射成一维数组中的全局位置。

一个最小向量加法 kernel 是：

```cpp
__global__ void vector_add(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```

典型启动配置：

```cpp
int block_size = 256;
int grid_size = (N + block_size - 1) / block_size;
vector_add<<<grid_size, block_size>>>(A, B, C, N);
```

这里的 `if (i < N)` 是边界保护：线程总数通常向上取整，因此最后一个 block 可能包含超出数组范围的线程。

## 3. Warp：从线程抽象连接到硬件

CUDA 源码以 thread 为主要抽象，但 NVIDIA GPU 通常以 warp 为硬件调度和 SIMT 执行单位。一个 warp 通常包含 32 个线程：

```text
Block = 256 Threads
256 / 32 = 8 Warps

Warp 0 → Thread 0~31
Warp 1 → Thread 32~63
...
Warp 7 → Thread 224~255
```

一个 warp 中的线程可以处理不同数据，但通常执行相同指令，例如：

```text
Thread 0  → A[0]  + B[0]
Thread 1  → A[1]  + B[1]
...
Thread 31 → A[31] + B[31]
```

这也是 GPU 适合大量相似任务的原因之一。

### Warp divergence

如果同一 warp 内的线程进入不同分支：

```cpp
if (threadIdx.x < 16) {
    A();
} else {
    B();
}
```

硬件可能先执行 A 路径，再执行 B 路径，并在每条路径上屏蔽不参与的线程。

重要区分：

- 有 `if` 不等于一定慢；
- 如果整个 warp 选择同一分支，代价通常较小；
- 边界 mask 是正确性所必需的；
- 真正需要测量的是 warp 内控制流是否分歧，以及分歧是否成为瓶颈。

## 4. SM 与 resident 工作

可以先把 SM 理解为 GPU 中承载和调度 CUDA 工作的资源区域：

```text
SM
├── Warp Scheduler
├── CUDA Core
├── Tensor Core
├── Load/Store Unit
├── Register File
├── Shared Memory
├── L1 Cache
└── SFU
```

一个 block 在执行期间通常由一个 SM 承载，但一个 GPU 上的 SM 数量可能少于 Grid 中的 block 数。例如，100 个 SM 不代表 1000 个 block 能同时执行；只有满足资源约束的 block 会成为 resident，前面的 block 完成后，后续 block 才会被调度。

因此，初学者需要把以下概念连接起来：

```text
Grid 很大
    ↓
不代表所有线程同时执行
    ↓
resident CTA 数量受 SM 资源约束
    ↓
occupancy 描述 active warps 与上限的关系
```

## 5. 内存层次的初步直觉

先使用一个粗略但有用的模型：

```text
Register
    ↓
Shared Memory / L1
    ↓
L2
    ↓
HBM / GDDR
```

### Register

Register 是线程私有的高速存储，适合临时变量、索引、累加器和 Tensor Core fragment。寄存器不是无限的，使用过多可能降低 occupancy，甚至发生 spill。

### Shared Memory

Shared memory 是 block 内线程共享的显式管理存储。它的核心价值不只是“更快”，而是让一个 block 中的线程复用从 global memory 搬入的数据：

```text
Global Memory
      ↓
Shared Memory tile
      ↓
多个线程重复使用
```

使用 shared memory 时要同时考虑容量、bank conflict、数据布局和同步。

### L1、L2 与 Global Memory

Cache 可以减少部分对 HBM/GDDR 的访问，但 cache 命中不是性能保证。真正需要观察的是访问局部性、合并访问、缓存行为和数据复用。

一个更准确的优化问题是：

> 如何让数据尽可能长时间留在离计算单元更近的层级，同时不让寄存器、shared memory 或同步成本抵消收益？

## 6. 用 GEMM 串起知识体系

矩阵乘法是很好的综合学习案例：

```text
Grid
 ↓
CTA / Block
 ↓
Warp
 ↓
Thread
 ↓
Tile
 ↓
Global Memory
 ↓
Shared Memory
 ↓
Register
 ↓
Tensor Core
```

一个高性能 GEMM 通常需要理解：

- 一个 CTA 如何负责输出矩阵的局部 tile；
- 多个 warp 如何分工；
- global memory 数据如何搬到 shared memory；
- shared memory 数据如何进入寄存器；
- MMA/Tensor Core 如何执行矩阵乘加；
- 结果如何累加和写回。

建议按以下顺序递进：

```text
CPU GEMM
  → naive CUDA GEMM
  → tiled CUDA GEMM
  → shared-memory GEMM
  → register blocking
  → Tensor Core GEMM
  → CUTLASS GEMM
```

## 7. 学习 CUDA → PTX → SASS

初学者不宜一开始研究 SASS。推荐顺序是：

```text
CUDA C++
    ↓
PTX
    ↓
SASS
```

可以用向量加法逐层提问：

- CUDA C++：我要对 `A[i]` 和 `B[i]` 做加法；
- PTX：需要 load、add、store 等虚拟指令；
- SASS：针对 `sm_80`、`sm_90` 或 `sm_100`，最终选择什么机器指令。

CUDA 源码决定线程和数据映射，PTX 表达较稳定的虚拟 ISA，SASS 才是具体目标架构上的机器指令。研究性能时，三层都要看，但不应跳过高层算法和数据布局。

## 8. 分层学习顺序

对初学者，可以把原地图中的概念分成以下层级：

```text
Level 1
Thread / Block / Grid / threadIdx / blockIdx / Warp

Level 2
SM / Register / Shared Memory / L1 / L2 / Global Memory

Level 3
Coalescing / Bank Conflict / Occupancy / Divergence / Roofline

Level 4
Tensor Core / WMMA / MMA / CUTLASS

Level 5
cp.async / Pipeline / Barrier / TMA

Level 6
WGMMA / Warpgroup / Cluster / Distributed Shared Memory

Level 7
PTX / SASS / Microarchitecture / Profiler
```

核心原则是：

> 不要跳级。先让低层概念通过代码和实验变得具体，再进入高级指令和微架构。

## 9. 最小实验路线

### 实验一：Vector Add

依次比较：

1. 一个 thread；
2. 多个 thread；
3. 多个 block；
4. 不同 block size；
5. 不同 memory access pattern；
6. 使用 profiler 观察差异。

每次只改变一个变量，并同时确认结果正确。

### 实验二：矩阵转置

依次实现：

1. naive transpose；
2. shared-memory tiled transpose；
3. bank-conflict-aware transpose。

这个实验适合观察“算法相同但 kernel 性能差异很大”的原因。

### 实验三：Reduction

从 `sum(A)` 开始，逐步比较：

```text
Thread
 ↓
Warp
 ↓
Block
 ↓
Shared Memory
 ↓
Warp Shuffle
 ↓
Atomic
```

这个实验会同时涉及并行归约、同步、shared memory、warp primitive 和多阶段结果合并。

### 实验四：GEMM

沿着 CPU → naive CUDA → tiled → shared memory → register blocking → Tensor Core → CUTLASS 的顺序实现和测量。

### 实验五：Softmax

Softmax 是连接 CUDA 基础与 AI kernel 的桥梁：

$$
\operatorname{softmax}(x_i)=
\frac{e^{x_i}}{\sum_j e^{x_j}}
$$

实现时会遇到：

1. max reduction；
2. 减去最大值以保持数值稳定；
3. `exp`；
4. sum reduction；
5. 除法；
6. memory access；
7. warp reduction；
8. shared memory；
9. 数值精度和稳定性。

最后再把 FlashAttention 放到这条路径上：

```text
CUDA 基础
  → GPU 硬件
  → 内存层次
  → 性能优化
  → 并行算法
  → GEMM / Tensor Core
  → 异步 Pipeline
  → Hopper / Cluster
  → Softmax / Attention
  → FlashAttention
```

## 10. 五问法：把名词变成理解

研究任何 CUDA 概念时，至少回答：

1. **它是什么？**
2. **为什么存在？**
3. **谁可以访问或使用它？**
4. **它有什么限制？**
5. **怎么通过实验验证它的价值？**

以 shared memory 为例：

- 是什么：block 内线程共享的片上存储；
- 为什么存在：提高数据复用，减少 global memory 访问；
- 谁可以访问：同一个 block 中的线程；
- 有什么限制：容量、bank conflict、同步和资源占用；
- 怎么验证：比较 naive kernel 与 shared-memory tiled kernel，并用 CUDA Event、Nsight Compute 观察 latency、bandwidth、occupancy 和 bank conflict。

## 11. 学习原则

不要按下面的方式堆概念：

```text
概念 → 概念 → 概念 → 概念
```

更有效的循环是：

```text
概念
  ↓
最小代码
  ↓
运行
  ↓
观察结果
  ↓
改变一个变量
  ↓
测量性能
  ↓
解释为什么
  ↓
再学习底层硬件
```

例如学习 warp 时，可以：

1. 写一个 32-thread kernel；
2. 观察 `threadIdx`；
3. 增加到 64、128、256 threads；
4. 理解 warp 数量；
5. 加入分支制造 divergence；
6. 用 profiler 观察 stall；
7. 再回头研究 warp scheduler。

## 当前理解与待实践

这份对话给现有 CUDA 地图补充了一个重要视角：地图不应只按硬件名词展开，还需要按“最小代码 → 可观察现象 → 性能测量 → 底层解释”组织学习。

下一步可以把每个阶段都变成可运行实验，并记录：

- 环境：GPU 型号、compute capability、CUDA/toolkit 版本；
- 代码：kernel、launch configuration、编译选项；
- 正确性：结果校验和边界条件；
- 性能：kernel 时间、端到端时间、带宽、occupancy、寄存器、shared memory；
- 现象：warp stall、bank conflict、spill、cache 行为和分支发散；
- 解释：观测是否符合执行模型和内存模型。

## Related

- [NVIDIA CUDA 硬件与编程模型研究地图](./cuda-hardware-and-programming-model-map.md) — 完整的硬件、执行、内存、同步、PTX/SASS 和性能地图
- [FlashAttention 阅读指南](../knowledge/ai/systems/flash-attention/flash-attention-reading-guide.md) — 将 CUDA 基础连接到 attention 和 FlashAttention
- [FlashAttention 系统地图](../knowledge/ai/systems/flash-attention/flash-attention-system-map.md) — 连接算法、kernel、框架和系统层次
