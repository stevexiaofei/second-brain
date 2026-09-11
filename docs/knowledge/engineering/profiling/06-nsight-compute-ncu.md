---
title: Nsight Compute（ncu）：单 kernel 的硬件计数器分析
subtitle: 是带宽受限、算力受限，还是延迟 / occupancy 受限
type: guide
status: seed
tags: [engineering, profiling, ncu, nsight-compute, CUDA, kernel, roofline, occupancy]
created: 2026-09-11
updated: 2026-09-11
source: 个人实践整理 + NVIDIA Nsight Compute 用户手册与 Kernel Profiling Guide
---

# Nsight Compute（ncu）：单 kernel 的硬件计数器分析

## 一句话理解

`ncu` 通过**硬件计数器 + kernel 重放**，回答一个非常具体的问题：

```text
这【一个】kernel 为什么慢？
  ├─ SM % 高、Memory % 低     → 算力受限（compute bound）
  ├─ Memory % 高、SM % 低     → 带宽受限（memory bound）
  ├─ 两个都低                 → 延迟 / occupancy 受限（有 stall，藏着结构性浪费）
  └─ 两个都高                 → 已经接近该 kernel 的硬件上限，去改算法
```

**它的开销极高**（同一个 kernel 要重放多遍以采集不同计数器），所以**必须是排查链的最后一环**，而不是第一环。

## 为什么它必须放在最后

| 前置问题 | 为什么必须先回答 |
|---|---|
| 这个 kernel 是热点吗？ | 用 [torch.profiler](04-pytorch-latency-profiling.md) / [nsys](05-nsight-systems-nsys.md) 确认。优化一个只占 2% 时间的 kernel 是白费力气 |
| GPU 上有空洞吗？ | 有空洞说明 GPU 在等，此时**所有 kernel 的指标都是"被动变好"的假象**，优化它们没有收益 |
| kernel 的输入布局对吗？ | 布局不对造成的低效（见下文 `Sectors Per Request`）应该先由算法/数据布局层面解决，而不是靠调 occupancy 硬扛 |

---

## 一、基本用法

### 1.1 采集

```bash
# 全量指标（慢，但信息最全）
ncu -o prof_out --set full python bench.py

# 只分析匹配的 kernel，只跑一次
ncu --set full -k "regex:gemm|attention" --launch-count 1 -o prof_out python bench.py

# 跳过前几次调用（避开 autotune / 首次编译）
ncu --launch-skip 5 --launch-count 1 --set full python bench.py

# 只采关心的 section（快很多）
ncu --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    --section WarpStateStats \
    python bench.py

ncu --list-sets        # 可用的 --set
ncu --list-sections    # 可用的 --section
```

### 1.2 关键开关

| 开关 | 作用 | 说明 |
|---|---|---|
| `-k, --kernel-name` | 按名字（支持正则）过滤 kernel | **必用**，否则会把所有 kernel 都分析一遍 |
| `-c, --launch-count` | 最多分析多少次启动 | 一般设 1~3 |
| `-s, --launch-skip` | 跳过前 N 次启动 | 避开 warmup |
| `--set` | 预置的 section 组合（`basic` / `full` / `roofline` …） | `full` 最慢 |
| `--section` | 精确挑选 section | 想快就自己挑 |
| `--cache-control` | 重放前是否清缓存（默认 `all`） | 改成 `none` 会更快，但部分指标偏乐观 |
| `--replay-mode` | `kernel`（默认）/ `application` / `range` | **NCCL / 多卡 / CUDA Graph 常需 `application`** |
| `--metrics` | 直接点名要哪个计数器 | 做脚本化回归时有用 |
| `-f, --force-overwrite` | 覆盖已存在的输出 | — |

### 1.3 查看

```bash
ncu-ui prof_out.ncu-rep            # GUI（推荐，图表最全）
ncu --import prof_out.ncu-rep --page details    # 终端打印
```

---

## 二、五步判读流程

拿到报告后，按这个顺序读，不要跳步。

### 第 1 步：Speed of Light —— 先定性

看两个百分比：**Compute (SM) Throughput** 与 **Memory Throughput**（相对于峰值）。

| SM % | Memory % | 结论 | 下一步 |
|---|---|---|---|
| 高 | 低 | **算力受限** | 看 ComputeWorkloadAnalysis：哪条流水线满了？能否减少指令？ |
| 低 | 高 | **带宽受限** | 看 MemoryWorkloadAnalysis：是 DRAM 还是 L1/L2？`Sectors Per Request` 正常吗？ |
| 都低 | 都低 | **延迟 / occupancy 受限** | 看 Occupancy 与 WarpStateStats：卡在什么 stall 上？ |
| 都高 | 都高 | 接近硬件上限 | 只能改算法（更少 FLOPs / 更好数据复用） |

> **"两个都低"是最容易被忽略也最有价值的一类。** 它说明硬件既没算满也没读满，时间都花在**等待**上了——通常是访存延迟没被并行度掩盖，或 occupancy 太低。

### 第 2 步：Memory Workload Analysis —— 带宽与访存形状

关注四类指标：

| 指标 | 含义 |
|---|---|
| **DRAM Throughput** | 是否吃满显存带宽 |
| **L1 / L2 Hit Rate** | 数据复用是否有效 |
| **Sectors Per Request** | ⭐ **每个访存请求实际拉了多少个 32 字节 sector** |
| Memory Chart | L1 / L2 / DRAM 之间的流量与命中关系图 |

⭐ **`Sectors Per Request` 是把 ncu 和硬件理论连起来的那根线。**

对一个"每线程请求 4 字节、warp 32 线程"的全局读：

$$\text{理想 sector 数} = \frac{32\ \text{线程} \times 4\ \text{B}}{32\ \text{B/sector}} = 4.0$$

```text
显示 4.0   →  完全合并，没有浪费
显示 8.0   →  一半 sector 是本线程用不到的 → 实际访存量翻倍
```

**这正是 [GPU 全局内存访存模型](../../ai/systems/gpu-memory-access-model.md) 里 Tiled MMA "50% GMEM 访存是多余的" 在 ncu 里的表现形式。** 遇到这种情况，修的是**数据布局**（thread/value layout、向量化宽度、对齐），而不是 occupancy。

### 第 3 步：Compute Workload Analysis —— 哪条流水线满了

看各类 pipe 的利用率：FMA / ALU / FP16 / FP32 / Tensor / LSU 等。

| 现象 | 可能原因 |
|---|---|
| Tensor pipe 高、其他低 | 正常（这就是想要的） |
| ALU / FMA 高但 Tensor 低 | 地址计算、边界判断、类型转换占了大量指令 → 减少冗余计算 |
| LSU（访存指令单元）高 | 访存指令太多 → 向量化、减少重复加载 |

### 第 4 步：Occupancy —— 有没有被资源限制住

| 指标 | 含义 |
|---|---|
| **Theoretical Occupancy** | 由 launch 配置与资源用量算出的**上限** |
| **Achieved Occupancy** | 实际达到的（通常低于理论值） |

若理论值就低，看**是什么限制了它**：

```text
寄存器用量过大          → __launch_bounds__ / 减少每线程状态
SMEM 用量过大           → 减小 tile / 分块更细
block 大小不合适         → 调整 blockDim
```

> **Occupancy 低 ≠ 一定有问题**：如果 kernel 已经是带宽瓶颈且访存合并良好，低 occupancy 也可能达到峰值带宽。**occupancy 是"是否存在优化空间"的提示，不是目标本身。**

### 第 5 步：Warp State Statistics —— 到底在等什么

这是"两个都低"时唯一能给出答案的地方。看 stall 原因的占比：

| Stall 原因 | 含义 | 典型对策 |
|---|---|---|
| **Long Scoreboard** | 等全局内存（L1 miss / 到 DRAM） | 提高数据复用（分块）、增加并行 warp、预取 |
| **Short Scoreboard** | 等共享内存 / MIO 操作 | 减少 bank conflict、减少 SMEM 访问次数 |
| **Wait** | 等固定延迟指令（依赖链） | 展开循环提高 ILP、重排指令 |
| **Barrier** | 等 `__syncthreads()` | 减少同步次数、减小 block 内的负载不均 |
| **MIO Throttle** | MIO 指令队列满（SMEM / 特殊函数指令过多） | 减少 SMEM 指令、用寄存器缓存复用值 |
| **LG Throttle** | 全局/本地访存指令队列满 | 向量化、减少访存指令条数 |
| **Not Selected** | 有就绪 warp，但调度器选了别人 | **不是问题**，说明并行度已足够 |
| **Math Pipe Throttle** | 算术流水线饱和 | 不是问题（算力已经打满） |

**判读要点**：`Not Selected` 与 `Math Pipe Throttle` 占比高是**好现象**；`Long Scoreboard` 高说明访存延迟没被掩盖；`Barrier` / `MIO Throttle` 高说明是结构性设计问题。

---

## 三、Roofline：把"受限类型"画成一张图

Roofline 模型用两个轴判断上限：

```text
                                  峰值算力
                                    │
   性能                             ├─────────────  ← 平台段（compute bound）
    (FLOP/s)                        │            ╱
                                    │          ╱    ← 斜坡段（memory bound）
                                    │        ╱
                                    └──────╱─────────────
                                         ╱
                                    算术强度 (FLOP/Byte)
```

- **落在斜坡段** → 带宽受限，提升算术强度（数据复用、分块）有效；
- **落在平台段** → 算力受限，只能减少 FLOPs 或换更快的指令；
- **离曲线很远** → 既不是带宽也不是算力限制，**是延迟 / occupancy 问题**。

ncu 的 GUI 里可以直接看 Roofline 图（需要相应 section）。

**这与 [GPU 算子优化方法论](../../ai/systems/gpu-kernel-optimization-methodology.md) 的三维度是对应的**：Roofline 把"计算"与"通信"两个维度量化在了同一张图上。

---

## 四、怎么构造一个"能被 ncu 分析"的场景

**在完整训练 job 上直接跑 ncu 基本不可行**（慢几个数量级，还会因为在集合通信上重放而挂住）。正确做法有三种：

| 做法 | 适用 |
|---|---|
| **写 microbenchmark**：把同一个 kernel 用同样的 shape / dtype 单独跑 | 最推荐。可复现、可回归、不干扰分布式 |
| `--launch-skip N --launch-count 1 -k <name>` | 能在真实程序里跑，但只抓稳态的某一次 |
| `--replay-mode application` | kernel replay 不可用时（NCCL / 多卡 / CUDA Graph）。会**重放整个程序**，极慢 |

> **分布式场景的现实建议**：把要分析的 kernel 剥离成单卡可运行的 microbenchmark（固定 shape），在单卡上做 ncu，再回到分布式环境用 nsys 验证整体收益。**不要试图在 8 卡 NCCL 训练里直接 ncu。**

---

## 五、常见误区

1. **在完整训练 job 上跑 ncu** → 慢到不可用，且分布式下容易挂住。
2. **只看 SM% / Memory% 就下结论** → "两个都低"的情况必须继续看 `WarpStateStats`，否则会得出"没有瓶颈"的错误结论。
3. **用 ncu 测端到端时间** → 重放彻底改变了执行时序，ncu 的耗时数字**不可用于性能结论**，只能用相对指标。
4. **不加 `-k` / `--launch-count` 过滤** → 分析了几百个 kernel，跑一晚上还没结束。
5. **不跳过 warmup 的那次启动** → 第一次启动可能与稳态选择不同的算法（autotune）或不同的数据状态。
6. **忽略 `--cache-control` 的影响** → 改成 `none` 虽然快，但重放之间的缓存残留会让命中率类指标偏乐观。
7. **把 occupancy 当目标** → occupancy 低不等于慢；带宽瓶颈的 kernel 低 occupancy 也可能打满带宽。
8. **`Sectors Per Request` 异常时去调 occupancy** → 那是**布局问题**，要从 `thread/value layout` 和向量化宽度上修。

## Related

- [Profiling 专题总览](./) — 工具地图与决策树
- [GPU 全局内存访存模型：向量化与合并访存](../../ai/systems/gpu-memory-access-model.md) — `Sectors Per Request` 与 sector/transaction 的理论依据
- [GPU 算子优化方法论：计算、通信、存储](../../ai/systems/gpu-kernel-optimization-methodology.md) — 判定受限类型后该往哪优化
- [Nsight Systems（nsys）](05-nsight-systems-nsys.md) — 上一层：确认这个 kernel 值得分析
- [PyTorch latency profiling：torch.profiler](04-pytorch-latency-profiling.md) — 上上层：框架视角的热点定位
- [CUTLASS / CuTe 03：TiledMMA 与 fragment](../../ai/systems/cutlass/03-cute-tiled-mma.md) — 用 layout 解释为什么会出现 `Sectors Per Request` 异常
- [CUTLASS / CuTe 09：TiledCopy 核心原理](../../ai/systems/cutlass/09-cute-tiled-copy-principle.md) — 修布局问题的具体手段

## References

- NVIDIA, [Nsight Compute 用户手册](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- NVIDIA, [Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)（Speed of Light、Memory Workload Analysis、Warp State 的定义）
- NVIDIA, [Nsight Compute CLI 选项](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- NVIDIA, [CUDA C++ Programming Guide — Arithmetic Intensity / Roofline](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
