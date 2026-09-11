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

### 1.2 关键开关

| 开关 | 作用 | 建议 |
|---|---|---|
| `-t, --trace` | 要采集哪些域：`cuda`、`nvtx`、`osrt`（OS runtime）、`cublas`、`cudnn`、`nccl`、`syscalls` | 分布式至少带 `cuda,nvtx,osrt,nccl` |
| `-s, --sample` | CPU 采样模式（`cpu` / `none`） | 只关心 GPU 时用 `-s none` 大幅降低开销与文件大小 |
| `--cuda-memory-usage=true` | 记录显存分配事件 | 排查"什么时候分配了显存"时开 |
| `--capture-range` | 只在指定区间录制（见第三节） | **强烈推荐**，否则文件会非常大 |
| `-d, --duration` / `--delay` | 只录一段时间 | 简单场景的替代方案 |
| `--cuda-graph-trace=node` | 展开 CUDA Graph 内部节点 | 用了 CUDA Graph 时必开（否则只看到一个巨大的 graph 节点） |
| `--stats=true` | 采集完自动跑一遍统计报告 | 懒人友好 |

### 1.3 查看

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

1. **多 rank 一起录**：用 `torchrun` / `mpirun` 时要注意 nsys 是否跟住了子进程；`nsys profile` 默认只跟当前进程，多进程启动方式需要确认 fork/child 相关选项（见 `nsys profile --help`），否则只会录到 launcher。
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
7. **多进程启动时只录到 launcher 进程** → 检查 nsys 的子进程跟踪选项。
8. **不看 `osrt` 行** → 会漏掉最容易被忽略的`数据加载 / 文件系统 / 锁`类阻塞。

## Related

- [Profiling 专题总览](./) — 工具地图与决策树
- [Nsight Compute（ncu）](06-nsight-compute-ncu.md) — 确认 GPU 侧是瓶颈之后的下一步
- [PyTorch latency profiling：torch.profiler](04-pytorch-latency-profiling.md) — 上一层的框架视角
- [系统层观测：htop、nvidia-smi 与 IO / 网络](01-system-level-observability.md) — 更便宜的一层
- [GPU 全局内存访存模型：向量化与合并访存](../../ai/systems/gpu-memory-access-model.md) — 解释时间线上为什么会出现大量小 memcpy
- [CUTLASS / CuTe 专题](../../ai/systems/cutlass/) — 被观察的 GEMM kernel 内部结构

## References

- NVIDIA, [Nsight Systems 用户手册](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- NVIDIA, [`nsys profile` 命令行选项](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#nsys-profile-command-options)
- NVIDIA, [Nsight Systems 报告（`nsys stats`）](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#nsys-stats-command-options)
- PyTorch, [`torch.cuda.nvtx`](https://pytorch.org/docs/stable/generated/torch.cuda.nvtx.range_push.html) 与 [`torch.autograd.profiler.emit_nvtx`](https://pytorch.org/docs/stable/generated/torch.autograd.profiler.emit_nvtx.html)
