---
title: NVIDIA CUDA 硬件与编程模型研究地图
type: concept
status: seed
tags: [CUDA, NVIDIA GPU, GPU Architecture, Programming Model, SM, Warp, Memory Hierarchy, Tensor Core]
created: 2026-08-21
updated: 2026-08-21
source:
  - https://docs.nvidia.com/cuda/cuda-c-programming-guide/
  - https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
  - https://docs.nvidia.com/cuda/parallel-thread-execution/
  - https://docs.nvidia.com/cuda/ampere-tuning-guide/
  - https://docs.nvidia.com/cuda/hopper-tuning-guide/
  - https://docs.nvidia.com/cuda/blackwell-tuning-guide/
---

# NVIDIA CUDA 硬件与编程模型研究地图

> 这是一份面向系统学习的 Inbox 研究地图，不是已经完全消化的稳定知识。目标是建立一条从 **GPU 硬件 → CUDA 抽象 → PTX / SASS → 性能分析 → Kernel 实践** 的连续路径。

## 一句话理解

CUDA 编程的核心不是“给 GPU 写一个函数”，而是把一个问题映射成一组可并行的线程层次，再让这些线程以适合 NVIDIA GPU 硬件层次的方式访问数据、协作和执行：

```text
问题空间
  ↓ 映射
Grid → Thread Block / CTA → Warp → Thread
  ↓ 绑定与调度
GPU → GPC / TPC → SM → Warp Scheduler → CUDA Core / Tensor Core / LDST
  ↓ 数据移动
Register → Shared Memory / L1 → L2 → HBM / Global Memory
```

真正的性能通常由三件事共同决定：

1. **工作如何分配**：足够的并行度、合理的 tile 和负载均衡；
2. **数据如何移动**：减少高代价层级的数据传输，让访问合并并提高复用；
3. **指令如何发射**：让 warp scheduler、Tensor Core、内存管线和异步拷贝管线持续有工作。

## 研究范围

本地图分成六层：

1. **硬件组织**：GPU、GPC、TPC、SM、执行单元和片上存储；
2. **执行模型**：SIMT、warp、线程块、grid、调度和发散；
3. **内存模型**：寄存器、shared memory、L1、L2、HBM、constant / texture；
4. **协作模型**：同步、原子操作、memory fence、Cooperative Groups、cluster；
5. **专用加速路径**：Tensor Core、WMMA、WGMMA、异步拷贝、TMA；
6. **性能模型**：occupancy、带宽、计算吞吐、延迟隐藏、roofline 和 profiling。

本轮只建立地图，不把每个概念都当作已经掌握；带有“待验证 / 待实践”的内容需要后续通过文档、源码、实验和 profiler 继续确认。

---

## 1. 先建立硬件地图

### 1.1 GPU 不是“很多 CUDA Core 的平面集合”

一个 NVIDIA GPU 可以粗略看成多个层级的资源集合：

```text
GPU
├── 多个 GPC（Graphics Processing Cluster）
│   └── 多个 TPC（Texture / Processor Cluster）
│       └── 一个或多个 SM（Streaming Multiprocessor）
│           ├── Warp Schedulers / Dispatch Units
│           ├── CUDA Cores / FP32、INT 等执行管线
│           ├── Tensor Cores
│           ├── Load / Store Units
│           ├── Register File
│           ├── Shared Memory / L1 Cache
│           └── Special Function Units
├── L2 Cache
└── HBM / GDDR Global Memory
```

不同 GPU 世代的命名、数量和具体微架构会变化，因此学习时要区分：

- **跨世代稳定的概念**：SM、warp、register、shared memory、L2、global memory、thread block；
- **架构相关的能力**：Tensor Core 数据类型、异步拷贝、TMA、WGMMA、thread block cluster、最大 resident blocks 等；
- **具体 SKU 的规格**：SM 数、显存容量、带宽、时钟、缓存容量和功耗。

不能把某一张显卡的规格表当成 CUDA 编程模型本身。

### 1.2 SM 是资源容器与 warp 执行调度中心

SM（Streaming Multiprocessor）是 CUDA 程序在硬件上的重要执行承载单元。一个 thread block / CTA 会被分配到某个 SM 上执行，并在其生命周期内使用该 SM 的资源。

一个 SM 可以驻留多个 CTA，但数量取决于资源约束：

- 每个 CTA 的线程数和 warp 数；
- 每个线程使用的寄存器数量；
- 每个 CTA 使用的 shared memory；
- SM 支持的最大 resident threads、warps、blocks；
- 具体 GPU 架构的限制。

可以用一个简化模型理解：

$$
\text{resident CTA 数}
\approx
\min\left(
\frac{\text{SM 可用线程数}}{\text{每 CTA 线程数}},
\frac{\text{SM 可用寄存器}}{\text{每 CTA 寄存器用量}},
\frac{\text{SM 可用 shared memory}}{\text{每 CTA shared memory}},
\text{架构上限}
\right)
$$

这只是上界估算，实际还会受到资源分配粒度和编译器寄存器分配的影响。

**重要区分：**

- 一个 grid 中可以有很多 CTA；
- 一个 SM 可以同时驻留多个 CTA；
- resident 不等于每个 CTA 都在同一时刻占用独立的执行单元；
- CTA 中的 warp 会共享 SM 的执行管线、寄存器文件、shared memory 和调度资源。

### 1.3 CUDA Core、Tensor Core 和 SM 的关系

“CUDA Core”不是一个等同于 CPU core 的完整处理器。它更接近 SM 内的一类标量 / 向量算术执行资源。一个 SM 还包含：

- warp scheduler 和指令发射逻辑；
- load/store 单元；
- Tensor Core；
- special function unit；
- 寄存器文件和 shared memory；
- 用于线程块、同步和内存操作的硬件支持。

因此不能只用“CUDA Core 数量”判断 kernel 性能。一个 kernel 可能受限于：

- global memory 带宽；
- L2 或 shared memory 流量；
- 指令发射吞吐；
- Tensor Core 利用率；
- 寄存器压力和 occupancy；
- 同步或原子操作；
- 指令依赖造成的延迟。

### 1.4 Tensor Core 是矩阵乘加专用路径

Tensor Core 面向小矩阵块的乘加计算，典型形式可以抽象为：

$$
D = A \times B + C
$$

CUDA 层可以通过 WMMA、MMA、WGMMA、CUTLASS 或更高层框架使用它们。不同架构支持的：

- 输入和累加数据类型；
- tile 形状；
- warp / warpgroup 协作方式；
- layout 和 shared-memory 要求；
- 异步或同步语义；

并不相同。

理解 Tensor Core 不能只停留在 API 名称，需要继续追踪：

```text
高层矩阵乘法
  → tile / layout
  → warp 或 warpgroup 指令
  → MMA / WGMMA
  → Tensor Core pipeline
  → 寄存器与 shared memory 数据流
```

---

## 2. CUDA 的抽象编程模型

### 2.1 Kernel、Grid、Block、Warp、Thread

CUDA 把一个 kernel 的执行组织成层次：

```text
一次 kernel launch
└── Grid
    └── Thread Blocks / CTAs
        └── Warps
            └── Threads
```

- **Kernel**：由 CPU 发起、在 GPU 上执行的函数；
- **Grid**：一次 kernel launch 产生的全部线程块；
- **Thread Block / CTA**：可以在同一个 SM 上协作的线程集合；
- **Warp**：通常由 32 个线程组成的硬件调度单位；
- **Thread**：程序员视角的单个逻辑执行实体。

线程的 `threadIdx`、线程块的 `blockIdx`、块维度 `blockDim` 和 grid 维度 `gridDim` 共同把逻辑坐标映射到数据坐标。

一个典型的一维映射是：

$$
i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

二维或三维问题则将多个坐标组合起来，常用于图像、张量和空间网格。

### 2.2 Block 是协作边界，Grid 通常不是同步边界

同一个 thread block 中的线程可以使用：

- `__syncthreads()` 做 block 级屏障；
- shared memory 交换数据；
- block 内原子操作和 memory fence；
- Cooperative Groups 提供的更细粒度协作。

不同 block 通常可以独立执行，调度到哪个 SM、何时执行并不由程序直接指定。因此默认 CUDA kernel 不能依赖普通 block 之间的全局同步。

需要跨 block 协作时，常见选择是：

1. 拆成多个 kernel launch，让 launch 边界承担全局顺序；
2. 使用原子操作和算法级协议；
3. 使用 cooperative launch / grid synchronization（需要硬件、启动方式和资源满足条件）；
4. 在支持的架构上使用 thread block cluster 等更高层协作机制。

### 2.3 Warp 是真正的硬件调度粒度

CUDA 源码写的是线程，但 SM 通常以 warp 为单位调度指令。一个 warp 的线程执行相同的指令流，只是各线程拥有自己的寄存器、谓词和数据。

这带来几个重要结论：

- block size 通常最好是 32 的倍数，但不是绝对规则；
- warp 内线程执行相同路径时效率较高；
- warp 内条件分支不同会发生 divergence；
- warp-level primitive 可以利用硬件调度粒度做快速协作；
- 一个线程的低效可能以 warp 的吞吐损失体现。

### 2.4 SIMT 不等于传统 SIMD

CUDA 常用 SIMT（Single Instruction, Multiple Threads）描述执行模型：程序员看到许多独立线程，硬件以 warp 为单位组织执行。

它和 SIMD 有相似处，也有重要区别：

- SIMD 通常强调一条向量指令操作多个 lane；
- SIMT 保留线程级索引、寄存器和控制流抽象；
- 硬件仍需处理 warp 内控制流分歧；
- CUDA 编程模型让线程看起来比底层执行更独立。

理解 SIMT 是理解 warp divergence、coalescing、warp shuffle 和 occupancy 的前提。

### 2.5 Warp divergence

当一个 warp 中的线程对条件分支选择不同路径时，硬件可能序列化执行不同路径，并让不参与当前路径的线程被屏蔽：

```text
同一个 warp：
线程 0~15 → 分支 A
线程 16~31 → 分支 B

执行时可能变成：
先执行 A（屏蔽 16~31）
再执行 B（屏蔽 0~15）
```

这不意味着所有分支都必须消灭：

- 如果分支在 warp 内一致，通常代价较小；
- 边界 mask 是必要的正确性机制；
- 过度追求无分支可能增加计算量或破坏可读性；
- 应通过 profiler 判断分支是否真的是瓶颈。

### 2.6 Grid 与硬件资源的关系

Grid 中的 CTA 数量应该覆盖足够多的 SM，并让每个 SM 有机会驻留多个 CTA。但“CTA 越多越好”不成立：

- CTA 太大，可能减少每个 SM 的 resident CTA 数；
- CTA 太小，可能增加调度和边界开销；
- 寄存器太多，可能降低 occupancy；
- shared memory 太多，可能限制并发；
- 工作不均衡会造成尾部 CTA 和 SM 空转。

因此 launch configuration 是算法设计的一部分，而不是最后随便填写的参数。

---

## 3. CUDA 内存层次

### 3.1 从快到慢的粗略视角

```text
每线程：register
    ↓
每个 block：shared memory / L1 相关缓存
    ↓
整个 GPU：L2 cache
    ↓
GPU 外部显存：HBM / GDDR global memory
    ↓
主机：host memory / unified memory 关联的系统内存
```

这不是严格的固定延迟排序；实际延迟和带宽取决于架构、访问模式、缓存命中、并发度和数据传输路径。但它提供了一个重要的设计直觉：

> 高性能 kernel 往往不是减少算术操作，而是让数据尽可能在更近的层级被复用。

### 3.2 Register

寄存器是线程私有的最快存储之一，适合保存：

- 标量临时值；
- 地址和索引；
- 累加器；
- Tensor Core fragment；
- 循环状态。

代价是寄存器是有限的。单线程寄存器使用量过高可能：

- 降低一个 SM 可驻留的 warp / CTA 数；
- 触发 spill，把局部变量溢出到 local memory；
- 增加访存流量和延迟。

因此“把更多数据放寄存器”并不总是更快。

### 3.3 Shared memory

Shared memory 是 block 内线程显式协作的片上存储，典型用途是：

1. 从 global memory 合并加载一个 tile；
2. 多个线程复用该 tile；
3. 计算完成后写回结果。

它的性能取决于：

- 访问是否产生 bank conflict；
- tile layout 是否适合线程映射；
- 使用量是否限制 resident CTA 数；
- 是否与异步拷贝、Tensor Core pipeline 协同。

FlashAttention 的 tile 复用、CUTLASS 的矩阵布局和很多 GEMM kernel 都依赖 shared memory 设计。

### 3.4 Global memory、coalescing 与带宽

Global memory 通常对应 HBM 或 GDDR。单个线程的访问看似合理，并不代表整个 warp 的访问高效。关键是 warp 的地址是否能合并成少量内存事务。

常见目标是：

- 相邻线程访问相邻或规则分布的数据；
- 对齐并减少不必要的 transaction；
- 让读取后的数据尽量被多个计算步骤复用；
- 避免每个线程随机访问大范围地址。

内存合并访问不是“线程索引连续”这么简单，还与：

- 数据类型大小；
- 对齐；
- stride；
- cache line / transaction 规则；
- layout 和边界；

有关，需要用 profiler 和具体架构文档验证。

### 3.5 L1、L2、constant 与 texture

- **L1**：靠近 SM，通常与 shared memory 在片上资源和配置上存在关系，具体组织依架构而变；
- **L2**：跨 SM 共享的更大缓存，常影响 global memory 访问和跨 kernel 数据复用；
- **constant memory**：适合广播性质强、访问模式符合约束的只读数据；
- **texture / read-only path**：历史上提供特定缓存和采样语义，现代用途需要结合架构与 API 判断。

不要把“用了 cache”当成性能保证；cache 命中率、访问局部性和并发行为必须通过测量确认。

### 3.6 Unified Memory 与显式拷贝

CUDA 的统一内存可以简化 CPU/GPU 地址空间管理，但不等于数据永远驻留在 GPU，也不等于省去了数据迁移。

需要继续理解：

- page migration；
- memory prefetch；
- access hint；
- page fault；
- CPU/GPU 并发访问和同步；
- pinned host memory 与异步 memcpy。

对于性能敏感路径，显式管理数据位置和传输通常更容易建立可预测的模型。

---

## 4. 同步、内存一致性与异步执行

### 4.1 同步解决什么问题

同步不是为了“让程序更安全”这么抽象，而是为了表达：

- 其他线程产生的数据何时可见；
- 一个阶段何时可以开始；
- 多个线程是否共同完成了 shared memory 的填充；
- 多个 producer / consumer 如何交换 buffer ownership。

常见层次包括：

- warp-level 同步和 shuffle；
- block-level `__syncthreads()`；
- memory fence；
- atomic 操作；
- cooperative groups；
- grid 或 cluster 级协作。

### 4.2 `__syncthreads()` 不是万能锁

`__syncthreads()` 主要表示 block 内线程到达屏障。使用时必须保证参与线程的控制流满足要求，否则可能死锁或行为未定义。

它通常需要和 shared memory 的生产/消费关系一起理解：

```text
线程组 A：写 shared tile
        ↓ __syncthreads()
线程组 B：读 shared tile
        ↓ __syncthreads()
复用 shared tile
```

如果只需要 warp 内协作，block 级屏障可能过重；如果需要跨 block 协作，`__syncthreads()` 又不够。

### 4.3 Atomic 与确定性

原子操作保证某些更新不会发生数据竞争，但不一定保证：

- 执行顺序；
- 浮点结果完全确定；
- 高吞吐；
- 跨多个复合步骤的事务语义。

例如多个 CTA 对同一个浮点输出做 atomic add，结果可能受执行顺序影响。训练和科学计算中要区分：

- 数值正确性；
- 可重复性；
- bitwise determinism；
- 性能代价。

### 4.4 异步 copy、pipeline 与 TMA

现代 CUDA kernel 常把数据搬运和计算重叠：

```text
global memory
     │ 异步拷贝
     ▼
shared memory tile 0 ──► Tensor Core / ALU 计算 tile 0
shared memory tile 1 ──► 同时预取 tile 1
```

这类设计通常包含：

- producer / consumer 阶段；
- double buffering 或 circular buffering；
- barrier / token；
- 异步 copy；
- shared memory layout；
- 计算与搬运之间的依赖管理。

Ampere、Hopper、Blackwell 等世代的可用指令和最佳实践不同。Hopper 的 TMA 更强调由硬件处理多维张量数据搬运和边界 / layout 相关工作，但学习时应区分：

- CUDA C++ 中的编程接口；
- PTX 指令语义；
- SASS 最终实现；
- 具体架构的吞吐和限制。

---

## 5. 从 CUDA C++ 到 PTX / SASS

CUDA 代码不是直接在 CUDA Core 上执行的。一个简化编译链是：

```text
CUDA C++ / device API
        ↓ nvcc 前端
PTX（虚拟 ISA / 中间表示）
        ↓ 架构相关编译
SASS（目标 GPU 的机器指令）
        ↓
GPU 执行
```

- CUDA C++ 提供线程、内存和同步抽象；
- PTX 提供相对稳定的虚拟指令集和架构能力表达；
- SASS 是具体 GPU 架构的机器级指令；
- `-arch` / `-code` 等编译选项影响生成的目标和兼容性。

研究一个性能问题时，可以逐层问：

1. C++ 代码表达了什么线程和数据映射？
2. 编译器生成了什么 PTX？
3. 目标架构最终选择了什么 SASS？
4. 寄存器、shared memory、同步和内存指令是否符合预期？
5. profiler 中实际瓶颈是什么？

不能只看 CUDA 源码推断最终硬件行为，也不能只看 SASS 忽略算法和数据布局。

---

## 6. 性能模型：从 occupancy 到 roofline

### 6.1 Occupancy 是手段，不是目标

Occupancy 常指一个 SM 上 active warps 与最大可支持 warps 的比例。较高 occupancy 可能帮助隐藏内存延迟，但不保证性能最高。

低 occupancy 可能来自：

- 寄存器使用过高；
- shared memory 使用过高；
- block 线程数过大；
- 架构限制。

高 occupancy 也可能伴随：

- 每线程寄存器太少，导致 spill 或重复加载；
- tile 太小，数据复用不足；
- Tensor Core 或内存管线没有被充分利用。

更准确的问题是：**当前 kernel 是否有足够的并发 warp 来隐藏它的主要延迟，同时保留足够资源做数据复用？**

### 6.2 带宽受限与计算受限

粗略地说：

- **memory-bound**：算力还有空余，但数据搬运跟不上；
- **compute-bound**：数据供应足够，但算术或矩阵指令吞吐成为瓶颈；
- **latency-bound**：并发不足或依赖链过长，无法隐藏延迟；
- **synchronization-bound**：屏障、原子或 producer/consumer 依赖限制吞吐。

算术强度可以写成：

$$
\text{Arithmetic Intensity}
=
\frac{\text{执行的 FLOPs}}{\text{移动的字节数}}
$$

Roofline 模型用算术强度把理论峰值算力和内存带宽放到同一张图上，帮助判断优化应该优先减少数据移动还是提高计算吞吐。

### 6.3 一个实用的性能排查顺序

```text
先确认结果正确
  ↓
测量 kernel 时间和端到端时间
  ↓
判断 memory-bound / compute-bound / latency-bound
  ↓
检查 grid / block / warp 映射
  ↓
检查 global memory 合并与复用
  ↓
检查 shared memory bank conflict 与容量
  ↓
检查 register、spill、occupancy
  ↓
检查分支、同步、atomic
  ↓
再看 Tensor Core、异步 pipeline 和架构特化
```

工具方向：

- Nsight Systems：看 CPU/GPU 时间线、kernel launch、并发和传输；
- Nsight Compute：看单个 kernel 的吞吐、内存、occupancy、warp 和指令指标；
- `ptxas` 输出：看寄存器和 shared memory 使用；
- CUDA events：测量 GPU 时间；
- CUDA Occupancy API / Calculator：估算资源限制；
- `cuobjdump` / `nvdisasm`：查看 cubin 和 SASS；
- 源码与架构 tuning guide：确认特性和限制。

---

## 7. 学习路线

### 第一阶段：建立抽象

目标：能读懂一个简单 CUDA kernel 的 launch 和索引。

1. CPU 与 GPU 的异构执行；
2. kernel launch；
3. grid / block / warp / thread；
4. `threadIdx`、`blockIdx`、`blockDim`、`gridDim`；
5. block 内同步和 shared memory；
6. 向量加法、矩阵转置、reduction。

### 第二阶段：建立硬件直觉

目标：能解释为什么同一个 kernel 有快慢差异。

1. SM 与 warp scheduler；
2. resident CTA、occupancy 和资源限制；
3. global memory transaction 与 coalescing；
4. shared memory bank conflict；
5. register pressure 与 spill；
6. warp divergence；
7. L1 / L2 / HBM 数据路径。

### 第三阶段：建立异步和矩阵计算模型

目标：能读懂现代高性能 GEMM / attention kernel 的骨架。

1. tile 与 data layout；
2. double buffering；
3. asynchronous copy；
4. `cuda::pipeline`、barrier 和 producer/consumer；
5. WMMA / MMA 与 Tensor Core；
6. CUTLASS 的 tiled MMA 抽象；
7. Hopper 的 TMA、WGMMA 和 warpgroup；
8. Blackwell 新增的架构能力及其编程接口。

### 第四阶段：进入性能工程

目标：能通过测量定位瓶颈，而不是凭术语猜测。

1. CUDA events 与可靠 benchmark；
2. Nsight Systems 时间线；
3. Nsight Compute kernel 分析；
4. PTX / SASS 对照；
5. occupancy 与 roofline；
6. 不同 GPU 架构的 tuning guide 对比；
7. 将优化结论写成可复现的实验记录。

### 第五阶段：连接现有知识

- [FlashAttention 术语表与关键状态表](../knowledge/ai/systems/flash-attention/flash-attention-glossary-and-state-table.md)：CTA、tile、split-KV、shared memory、warp；
- [FlashAttention Kernel 与 Launch 机制](../knowledge/ai/systems/flash-attention/flash-attention-kernel-and-launch.md)：work partitioning、split-KV、kernel specialization；
- [FlashAttention Kernel 细节补充](../knowledge/ai/systems/flash-attention/flash-attention-kernel-details.md)：CTA 工作单元、sequence parallel、累积和布局；
- [FlashAttention 论文精读](../knowledge/ai/systems/flash-attention/flashattention-paper-series.md)：IO-aware、sliced-Q、CTA / warp 映射；
- [RVV 算子开发必备基础知识](../knowledge/engineering/rvv-operator-development.md)：比较 CPU SIMD / RVV 与 GPU SIMT 的抽象差异；
- [PyTorch 源码理解](../knowledge/ai/systems/pytorch/)：观察 CUDA kernel 如何从框架、ATen 和编译栈被调用。

---

## 8. 容易混淆的概念

| 概念 | 不要简单理解成 | 更准确的理解方向 |
| --- | --- | --- |
| SM | 一个 CPU 核心 | 承载多个 warp / CTA 的资源和调度单元 |
| CUDA Core | 一个独立线程处理器 | SM 内的一类算术执行资源 |
| CTA | 一个 warp | 一组可在 block 边界内协作的线程，通常就是 thread block |
| Warp | 32 个完全独立的线程 | 硬件调度和 SIMT 执行的基本单位 |
| Occupancy | 越高越快 | 隐藏延迟的资源指标，需要与复用和吞吐一起看 |
| Shared memory | 自动缓存 | block 显式管理的片上协作存储 |
| Unified Memory | 免费的 GPU 显存 | 统一地址空间抽象，仍可能发生迁移和缺页 |
| Tensor Core | 更快的 CUDA Core | 面向矩阵乘加的专用执行路径 |
| Kernel launch | GPU 立即开始执行 | CPU 提交工作，实际调度受 stream、资源和硬件状态影响 |
| `__syncthreads()` | 全 GPU 同步 | block 内屏障 |
| PTX | GPU 最终机器码 | 虚拟 ISA / 中间表示，最终还要生成架构相关 SASS |

---

## 我的当前理解

CUDA 最值得建立的心智模型不是“记住 API”，而是：

> **程序员定义线程层次和数据流，编译器把它们翻译成指令，SM 以 warp 为单位调度，内存层次决定数据移动成本，资源分配决定能有多少工作同时驻留。**

FlashAttention 的很多优化都可以还原成这套模型：

- tile 化：把问题映射成可复用的局部工作；
- CTA / warp 划分：把工作分给硬件调度和执行单元；
- shared memory：提高片上数据复用；
- online softmax：避免写回完整中间矩阵；
- split-KV：在并行度不足时增加 CTA 工作量；
- kernel specialization：把运行时条件转成编译期形状；
- sequence parallel：在正确性、累积和并行度之间做折中。

这部分仍是基于文档、现有 FlashAttention 笔记和概念推理形成的研究框架。具体架构参数、指令吞吐、bank conflict 规则和 TMA / WGMMA 限制，必须在对应 GPU tuning guide、PTX 文档和 profiler 实验中逐项确认。

## 待继续回答的问题

1. 从 Volta、Ampere、Hopper 到 Blackwell，SM 的关键变化如何影响 CUDA 编程模型？
2. resident CTA、warp occupancy、register allocation 的精确计算规则是什么？
3. shared memory bank conflict 在不同架构中的具体规则如何变化？
4. `cp.async`、TMA、WGMMA 的异步依赖和可见性模型分别是什么？
5. CUTLASS 如何把 thread / warp / warpgroup / MMA / layout 组合成可复用模板？
6. Tensor Core 的 tile layout 如何映射到寄存器 fragment 和 shared memory？
7. 如何用 Nsight Compute 指标判断一个 attention kernel 的主要瓶颈？
8. CUDA Graph、stream、event、host callback 与 kernel launch overhead 如何共同影响端到端性能？
9. GPU memory consistency model 与 C++ memory model 的边界是什么？
10. 如何设计一组从向量加法、转置、reduction、GEMM 到 attention 的最小实验？

## 下一步实验建议

1. 写一个向量加法：比较不同 block size 和访存模式；
2. 写矩阵转置：比较 naive、shared-memory tiled 和 bank-conflict-free 版本；
3. 写 reduction：比较 divergent branch、warp shuffle 和多阶段 reduction；
4. 写 tiled GEMM：观察 tile、register blocking、shared memory 和 Tensor Core 的作用；
5. 用 Nsight Compute 分析每个版本，并记录时间、带宽、occupancy、寄存器、shared memory 和 warp stall；
6. 将实验结果与 FlashAttention 的 tile / CTA / warp 设计对照。

## Related

- [AI Infra 方向论文地图](./ai-infra-papers-map.md) — CUDA kernel、Triton、FlashAttention、编译器和 serving 的上游路线
- [FlashAttention 术语表与关键状态表](../knowledge/ai/systems/flash-attention/flash-attention-glossary-and-state-table.md)
- [FlashAttention Kernel 与 Launch 机制](../knowledge/ai/systems/flash-attention/flash-attention-kernel-and-launch.md)
- [FlashAttention Kernel 细节补充](../knowledge/ai/systems/flash-attention/flash-attention-kernel-details.md)
- [RVV 算子开发必备基础知识](../knowledge/engineering/rvv-operator-development.md)
- [PyTorch 源码理解](../knowledge/ai/systems/pytorch/)
