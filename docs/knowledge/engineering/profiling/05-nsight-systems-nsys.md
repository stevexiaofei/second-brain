---
title: Nsight Systems（nsys）：系统级时间线与 GPU 空洞诊断
subtitle: 回答"这几十毫秒里 CPU、GPU、通信各在干什么"
type: guide
status: seed
tags: [engineering, profiling, nsys, nsight-systems, CUDA, timeline, NCCL, NVTX]
created: 2026-09-11
updated: 2026-09-11
source: 个人实践整理 + NVIDIA Nsight Systems 用户手册
---

# Nsight Systems（nsys）：系统级时间线与 GPU 空洞诊断

## 一句话理解

`nsys` 是**系统级时间线录制器**。它把 CUDA API、GPU kernel、NVTX 标注、OS 线程调度、同步原语、NCCL 通信按**同一条时间轴**画出来，用来回答：

```text
这几十毫秒里，CPU 在干什么？GPU 在干什么？它们在互相等吗？
GPU 上那些空洞（没有 kernel 的时间）是被谁造成的？
```

与 [ncu](06-nsight-compute-ncu.md) 的分工可以一句话概括：**nsys 回答"问题在时间轴上的哪一点"，ncu 回答"那一点为什么慢"。**

## 为什么重要

- 它补上了 `torch.profiler` 看不到的部分：**OS 线程、系统调用、跨进程/跨卡行为**。
- **"GPU 利用率低"的根因诊断只能在这里做。** nvidia-smi 只告诉你"有洞"，nsys 告诉你是数据加载、是 Python、是同步，还是通信在制造这个洞。
- 分布式训练的"**通信与计算是否重叠**"是一个纯粹的时间轴问题——只有时间线工具能回答。

## 与 torch.profiler 的取舍

| | [`torch.profiler`](04-pytorch-latency-profiling.md) | `nsys` |
|---|---|---|
| 视角 | 框架视角（op / `nn.Module` / Python 栈） | **系统视角**（CUDA API / OS 线程 / 跨进程） |
| 独有信息 | Python 调用栈、shape、op 级显存 | 线程调度、同步等待、`osrt` 阻塞、NCCL、跨卡 |
| 对上业务的难度 | 低（自带 op 名） | 高（需要 NVTX 标注） |

**推荐组合**：先用 `torch.profiler` 找到可疑阶段，再用 `nsys` 把这个阶段的系统行为看清楚。

---

## 一、基本用法

### 1.1 采集

```bash
nsys profile \
  -o prof_out \
  --force-overwrite true \
  -t cuda,nvtx,osrt,cublas,cudnn,nccl \
  --cuda-memory-usage=true \
  python train.py
```

生成 `prof_out.nsys-rep`（文本报告）与 `prof_out.sqlite`（可用 SQL 查询）。

### 1.2 关键开关速查

| 开关 | 作用 | 建议 |
|---|---|---|
| `-t, --trace` | 要采集哪些域：`cuda`、`nvtx`、`osrt`（OS runtime）、`cublas`、`cudnn`、`nccl`、`syscalls` | 分布式至少带 `cuda,nvtx,osrt,nccl` |
| `-s, --sample` | CPU 回栈采样：`process-tree`（启动应用时的默认值）/ `system-wide` / `none` | 只关心 GPU 时用 `none`；注意它**不会**关掉上下文切换采集 |
| `--cpuctxsw` | CPU 上下文切换采集（默认开启） | 与 `--sample=none` **成对使用**才能完全静默 CPU 侧采集 |
| `--cuda-memory-usage=true` | 记录显存分配事件 | 排查"什么时候分配了显存"时开 |
| `--capture-range` | 只在指定区间录制（见第三节） | **强烈推荐**，否则文件会非常大 |
| `-d, --duration` / `--delay` | 只录一段时间 | 简单场景的替代方案 |
| `--cuda-graph-trace=node` | 展开 CUDA Graph 内部节点 | 用了 CUDA Graph 时必开（否则只看到一个巨大的 graph 节点） |
| `--trace-fork-before-exec` | 让"只 fork 不 exec"的子进程也被 trace（默认不 trace） | DataLoader worker / `multiprocessing` fork 子进程场景必开 |
| `--wait` | 采集结束时等谁退出：`primary`（被启动的主进程）等 | 常驻服务 / 有守护子进程时用 `primary`，避免永远等不齐 |
| `--kill` | 采集结束时给应用进程组发什么信号：`none` / `sigterm`（默认）/ `sigkill` / 信号号 | 不想让 nsys 杀掉被 profile 的进程时用 `none` |
| `-f, --force-overwrite` | 覆盖已存在的报告文件（默认拒绝） | 脚本化 / 反复重跑时设 `true` |
| `--stats=true` | 采集完自动跑一遍统计报告 | 懒人友好 |

### 1.3 逐条解释：每个开关在控制什么

这些开关按"控制什么"分成五组，理解了分组就不会记混。

#### A. 录什么：`-t / --trace`

每个域是一类独立的采集器，各自有开销和文件体积：

| 域 | 录到的内容 | 用来回答 |
|---|---|---|
| `cuda` | CUDA runtime / driver API、kernel、memcpy/memset、同步 | 时间线的骨架：谁在什么时候 launch 了什么 |
| `nvtx` | 你自己打的 CPU 侧标注（`range_push` / `record_function`） | "这段 GPU 活动属于哪个业务阶段" |
| `osrt` | OS runtime 库调用（pthread 的 mutex/cond、文件 IO 的 read/write 等） | **CPU 侧阻塞归因**：空洞期间 CPU 在等什么 |
| `cublas` / `cudnn` | 库 API 层调用 | "这批 kernel 是哪一次库调用发出来的" |
| `nccl` | 集合通信的 API 与 kernel | 通信占比、是否与计算重叠 |
| `syscalls` | 系统调用（比 osrt 更细） | 更底层的阻塞归因 |

截图里的 `--trace=cuda,nvtx,cudnn` 是一个**单卡、面向库调用、轻量**的选择：没有 `nccl`（单卡不需要）、没有 `osrt`（省开销）。代价见 B 组末尾。

#### B. CPU 侧两个采集器：`-s / --sample` 与 `--cpuctxsw`

这是最容易被误解的一对，官方文档里有一条明确的交互说明：

- `--sample`（`-s`）：**CPU IP/回栈采样**——周期性抓各线程的调用栈，产出 CPU 侧火焰图与函数热点。取值 `process-tree`（启动应用时的默认值）/ `system-wide`（部分平台需要 root）/ `none`。
- `--cpuctxsw`：**上下文切换采集**——记录线程何时被调度器切下 CPU、切去等待什么。时间线上是独立的一行，用来找 CPU 争抢 / 抢占 / 锁等待。默认开启。
- **关键交互**：`--sample=none` **只关回栈采样，不关上下文切换采集**；要完全静默 CPU 侧，必须再显式加 `--cpuctxsw=none`。

这正是截图里 `--sample=none --cpuctxsw=none` 成对出现的原因。这两个是 CPU 侧开销最大的采集器，关掉能显著减小开销与文件体积。

> **代价**：两者都关、且不带 `osrt` 时，时间线仍然能告诉你"空洞发生在何时"，但**无法告诉你空洞期间 CPU 在 Python/系统层面做什么**——只能看到 CUDA API 层的调用。要归因 CPU，就得重新带 `osrt` 或 `--sample=process-tree` 采一次。

#### C. 哪些进程在会话里：`--trace-fork-before-exec`

机制（NVIDIA 工程师在论坛的说明）：nsys 会在每个被 trace 的进程里创建辅助线程；而 POSIX 规定**多线程进程 `fork()` 出的子进程在 `exec` 之前只能执行 async-signal-safe 操作**。因此 nsys 的默认策略是：

```text
子进程 fork 之后    →  立即禁用 trace
子进程调用 exec 后  →  重新启用 trace
```

由此得到一个反直觉但重要的结论：

- **只 fork、从不 exec 的子进程，默认是 trace 不到的** —— 典型就是 **fork 启动方式的 DataLoader worker**、`multiprocessing` 的 fork 子进程、gpu_burn 式的 worker 池；
- **fork 后又 exec 的子进程**（torchrun / mpirun 拉起的 python worker、shell 包装脚本）默认**能**被 trace（exec 后恢复）；
- `--trace-fork-before-exec=true` 强制 trace 那些"从不 exec"的子进程（接受上述 POSIX 限制带来的风险）。

所以"多进程场景录不全"的常见真相**不是没跟住 launcher，而是漏掉了 fork-only 的 worker**。

#### D. 会话怎么结束：`--wait` 与 `--kill`

- `--wait`：采集结束时**等谁退出**再收尾出报告。`primary` = 只等被启动的主进程；（部分版本还支持等整棵被 trace 的进程树）。对常驻服务或有守护子进程的程序，"等全部"可能永远等不齐，所以服务端 profile 常见 `--wait=primary`。默认值随版本不同，以 `nsys profile --help` 为准。
- `--kill`：采集结束时（`--duration` 到点、capture-range 结束、Ctrl-C）给**应用进程组**发什么信号。取值 `none` / `sigterm`（默认）/ `sigkill` / 具体信号号。文档行为：从命令行启动的应用在采集完成时**默认会被终止**，除非指定 `--kill none`；例外是当采集区间由 NVTX / `cudaProfilerStart/Stop` / 热键控制时，应用默认继续运行。
- `--kill=none` 的语义就是"**只观测，不动进程**"：profile 一个线上服务、或希望采集窗口结束后训练继续跑时必加。

C + D 合起来回答了三个问题：**谁在会话里、何时收尾、收尾时杀不杀进程。**

#### E. 输出处理：`--force-overwrite` 与 `--stats`

- `-f / --force-overwrite=true`：允许覆盖同名 `.nsys-rep`。默认拒绝覆盖，脚本重跑会直接报错；迭代调试时设 `true`。
- `--stats=true`：采集结束后自动执行 `nsys stats`，把摘要表打到终端。不想开 GUI 先看一眼热点时很有用（报告明细见第五节）。

#### 把截图里那条命令逐行读一遍

```bash
nsys profile \
  --force-overwrite=true \          # 允许覆盖旧报告，重跑不报错
  --trace=cuda,nvtx,cudnn \         # 只录 CUDA 活动 / 自定义标注 / cuDNN 调用
  --sample=none \                   # 关 CPU 回栈采样
  --cpuctxsw=none \                 # 关上下文切换采集（与上一行成对，才能完全静默 CPU 侧）
  --trace-fork-before-exec=true \   # 连"只 fork 不 exec"的子进程也 trace（DataLoader worker 等）
  --wait=primary \                  # 主进程退出即收尾出报告，不等可能常驻的子进程
  --kill=none \                     # 采集结束不给应用发信号，服务/训练继续跑
  --stats=true \                    # 采集完立刻打印统计摘要
  python train.py
```

一句话读法：**GPU 为中心、低开销、但不漏 fork 子进程、且不碰被 profile 的进程** —— 典型的"对长跑服务/训练进程做观测"的配置。

### 1.4 查看

```bash
nsys-ui prof_out.nsys-rep                  # GUI（功能最全，推荐）
nsys stats prof_out.nsys-rep               # 终端统计报告
```

---

## 二、时间线上要看的五件事

打开 GUI 后，按这个顺序看，基本不会跑偏：

### ① GPU 空洞（Gap）

```text
GPU:   [kernel A]        ←— 空洞 —→        [kernel B]
CPU:   ████████ 数据加载 / Python ████████
```

**空洞 = GPU 在等。** 顺着空洞往上看 CPU 行，通常能直接看到原因：

| 空洞上方对应什么 | 结论 |
|---|---|
| `osrt` 行里的 `read` / `pread` / 文件系统调用 | 数据加载瓶颈（磁盘 / 解码） |
| 大量 `aten::` / Python 相关的 CPU 工作 | CPU 侧算子或 Python 逻辑 |
| `cudaStreamSynchronize` / `cudaDeviceSynchronize` | 显式或隐式同步打断（`.item()`、`.cpu()`、`print`） |
| `cudaMalloc` 长时间占用 | 显存分配器在向驱动申请（碎片或首次扩张） |
| 通信 API（`ncclAllReduce` 等） | 通信没有与计算重叠 |

### ② kernel 之间是否背靠背

- **背靠背、无空隙** → GPU 侧是瓶颈，去 [ncu](06-nsight-compute-ncu.md)；
- **有细微空隙** → 可能是 kernel 之间的依赖、launch 延迟，或 stream 没有并行起来。

### ③ Memcpy 的位置与数量

时间线上的 `Memcpy` 行值得单独盯：

| 现象 | 问题 |
|---|---|
| 频繁的小 H2D/D2H 传输 | 通常是 `.cpu()`、`.item()`、CPU 侧张量拼接造成的，会强制同步 |
| 每步一次大 D2H（为了记日志） | 考虑降低频率或改成 GPU 侧聚合 |
| H2D 与计算串行 | 数据没有提前预取（`pin_memory` + 非阻塞传输 + 双缓冲） |

### ④ 通信是否与计算重叠（分布式）

在时间线上同时看**计算 kernel 行**和 **NCCL kernel 行**：

```text
重叠良好：  计算 ██████████████
            通信      ████████            ← 与计算并行存在

重叠失败：  计算 ██████████████
            通信               ████████   ← 严格串行，总时间 = 计算 + 通信
```

**重叠失败是分布式训练最常见的"白拿不到的性能"**。典型修法是让梯度通信在反向传播过程中就按 bucket 启动（而非等整个 backward 结束）。

### ⑤ CPU 侧的 CUDA API 分布

在 `CUDA API` 行里按耗时排序，看是不是被某几类调用占满：

| API | 含义与对策 |
|---|---|
| `cudaLaunchKernel` 次数极多 | kernel 太碎，考虑算子融合 / `torch.compile` |
| `cudaStreamSynchronize` 频繁 | 同步打断（见 ① 的第四行） |
| `cudaMalloc` 频繁 | 分配器在反复向驱动申请 → 碎片或分配模式问题 |
| `cudaMemcpy` 频繁 | 见 ③ |

---

## 三、`capture-range`：只录稳态的那几步

**这是 nsys 最重要的使用技巧。** 直接录整个训练 job 会让文件涨到几 GB，而且真正关心的稳态段淹没在启动噪声里。

### 3.1 用 CUDA Profiler API 圈定范围

```bash
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -o prof_out python train.py
```

```python
import torch

for step, batch in enumerate(loader):
    if step == 5:
        torch.cuda.cudart().cudaProfilerStart()     # 从这里开始录
    train_step(batch)
    if step == 10:
        torch.cuda.cudart().cudaProfilerStop()      # 到这里结束
        break
```

### 3.2 用 NVTX 圈定范围

```bash
nsys profile --capture-range=nvtx --nvtx-capture="train_step@main" -o prof_out python train.py
```

配合代码里的 `torch.cuda.nvtx.range_push("train_step")` / `range_pop()`。

### 3.3 为什么必须这样做

```text
第 1 步    cuDNN autotune、Triton/NVRTC 编译、分配器首次扩张   ← 不代表稳态
第 2-3 步  各种懒初始化                                        ← 不代表稳态
第 4 步起  稳态                                                 ← 只有这里有意义
```

---

## 四、NVTX：让时间线可读

没有标注的 nsys 时间线上只有 `cudaLaunchKernel` 和 kernel 名，看不出业务含义。加标注：

```python
import torch

torch.cuda.nvtx.range_push("data_load")
batch = next(loader)
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("forward_backward")
loss = train_step(batch)
torch.cuda.nvtx.range_pop()
```

也可以直接复用 `torch.profiler` 的标注（两边都能看到）：

```python
from torch.profiler import record_function

with record_function("attention"):
    out = attn(x)
```

**标注的粒度建议**：step 级 → 模块级（forward / backward / optimizer / data）→ 关键子模块。再细就会让时间线过载。

---

## 五、`nsys stats` 常用报告

```bash
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum prof_out.nsys-rep
```

| 报告名 | 内容 |
|---|---|
| `cuda_gpu_kern_sum` | **GPU kernel 总耗时排序**（谁最吃 GPU 时间） |
| `cuda_api_sum` | CUDA API 调用耗时与次数 |
| `cuda_gpu_mem_time_sum` | memcpy 的耗时统计 |
| `cuda_gpu_mem_size_sum` | memcpy 的字节量统计 |
| `nvtx_sum` | 按 NVTX 区间聚合（把 GPU 时间归到你的业务块上） |
| `osrt_sum` | OS runtime 调用统计（找 IO / 锁） |

**`nvtx_sum` 是性价比最高的一个**：它把 GPU 时间按你标注的区间（`forward` / `backward` / `optimizer`）聚合，直接给出"各阶段占多少"。

---

## 六、分布式场景

1. **多 rank 一起录，并留意 fork 子进程**：nsys 默认会 trace 被启动的进程树，但**只 `fork` 不 `exec` 的子进程默认不被 trace**（fork 启动方式的 DataLoader worker、`multiprocessing` fork 子进程属于此类），需要加 `--trace-fork-before-exec=true`（见 §1.3-C）；fork+exec 的子进程（torchrun / mpirun 拉起的 worker）默认能被 trace。
2. **关注 NCCL kernel**：在 kernel 行里找 `ncclDevKernel_*` / `nccl:*` 开头的名字，它们就是通信本体。
3. **对齐各 rank 的时间轴**：每个 rank 一份报告，比较"谁的空洞最长"——**straggler 往往是整个 job 的瓶颈**。
4. **注意采样开销**：`-s cpu` 在多 rank 同时开启时开销叠加更大，必要时用 `-s none`。

---

## 七、常见误区

1. **从头录到尾** → 文件几个 GB，且稳态段被启动噪声淹没。用 `--capture-range`。
2. **不加 NVTX 就分析** → 时间线上只有 kernel 名，无法映射回业务模块。
3. **把第一步的动作当稳态** → 会把 autotune 和编译开销当成真实瓶颈。
4. **看到 GPU 空洞就去做 kernel 优化** → 空洞意味着 GPU 在等，问题在 CPU / IO / 通信侧。
5. **只看 GPU 行不看 CPU 行** → 空洞的原因全部写在上面的 CPU 行里。
6. **用了 CUDA Graph 却没加 `--cuda-graph-trace=node`** → 时间线上只有一个大节点，看不到内部细节。
7. **多进程场景录不全** → 常见真相是漏掉了"只 fork 不 exec"的子进程（DataLoader worker 等），加 `--trace-fork-before-exec=true`（见 §1.3-C）。
8. **以为 `--sample=none` 就够了** → 上下文切换采集仍然开着；要完全静默 CPU 侧还需 `--cpuctxsw=none`。
9. **不看 `osrt` 行** → 会漏掉最容易被忽略的`数据加载 / 文件系统 / 锁`类阻塞。

## Related

- [Profiling 专题总览](./) — 工具地图与决策树
- [Nsight Compute（ncu）](06-nsight-compute-ncu.md) — 确认 GPU 侧是瓶颈之后的下一步
- [PyTorch latency profiling：torch.profiler](04-pytorch-latency-profiling.md) — 上一层的框架视角
- [系统层观测：htop、nvidia-smi 与 IO / 网络](01-system-level-observability.md) — 更便宜的一层
- [GPU 全局内存访存模型：向量化与合并访存](../../ai/systems/gpu-memory-access-model.md) — 解释时间线上为什么会出现大量小 memcpy
- [CUTLASS / CuTe 专题](../../ai/systems/cutlass/) — 被观察的 GEMM kernel 内部结构

## References

- NVIDIA, [Nsight Systems 用户手册](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- NVIDIA, [`nsys profile` 命令行选项](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#nsys-profile-command-options)（`--sample` / `--cpuctxsw` / `--kill` / `--wait` 的取值与默认值表）
- NVIDIA, [Nsight Systems 报告（`nsys stats`）](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#nsys-stats-command-options)
- NVIDIA 论坛, [`--trace-fork-before-exec` 机制说明](https://forums.developer.nvidia.com/t/generating-cupti-tables-with-nsys/236268/25)（fork 后禁用 trace、exec 后恢复的原因）
- NVIDIA 论坛, [`-s none` 不关闭上下文切换采集](https://forums.developer.nvidia.com/t/nvhpc-25-7-nsys-fails-when-collecting-a-profile-for-simple-fortran-program/347031/2)
- RTP-LLM, [Performance Profiling](https://rtp-llm.ai/build/en/references/profiling.html)（服务端 nsys 命令行的真实示例：`--wait=primary --kill=none --trace-fork-before-exec=true`）
- PyTorch, [`torch.cuda.nvtx`](https://pytorch.org/docs/stable/generated/torch.cuda.nvtx.range_push.html) 与 [`torch.autograd.profiler.emit_nvtx`](https://pytorch.org/docs/stable/generated/torch.autograd.profiler.emit_nvtx.html)
