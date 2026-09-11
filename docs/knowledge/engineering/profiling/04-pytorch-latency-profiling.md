---
title: PyTorch latency profiling：torch.profiler 与时间线三模式
subtitle: op 级热点、CPU/GPU 空心化与同步打断
type: guide
status: seed
tags: [engineering, profiling, pytorch, kineto, torch.profiler, nvtx, latency]
created: 2026-09-11
updated: 2026-09-11
source: 个人实践整理 + PyTorch profiler 官方文档与教程
---

# PyTorch latency profiling：torch.profiler 与时间线三模式

## 一句话理解

`torch.profiler`（底层是 **Kineto**）给你一张 **op 级时间线**。读它只需要回答三个问题：

```text
① 时间花在哪些 op 上？          →  看聚合表（key_averages）
② 这些时间是 CPU 还是 GPU 的？   →  看 Self CPU vs Self CUDA
③ GPU 在时间线上有没有空洞？      →  看 trace 时间线
```

**第三个问题最关键**：GPU 有空洞说明瓶颈根本不在 GPU 上（在数据加载、Python 逻辑或同步），此时去看 kernel 效率（ncu）是南辕北辙。

## 为什么重要

- 它填上了 [系统层工具](01-system-level-observability.md) 与 [单 kernel 工具](06-nsight-compute-ncu.md) 之间的空白：**"这一步里到底在干什么"**。
- 它能回答分布式训练里最要紧的两个问题：**通信占了多少**、**通信和计算重叠了吗**。
- 它自带 Python 调用栈与 shape 信息（nsys 没有），所以"是哪个模块申请的这次 kernel"这类问题在它这里更好回答。

---

## 一、标准骨架

```python
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=2, warmup=2, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./log/prof"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(loader):
        if step >= 20:
            break
        train_step(batch)
        prof.step()                      # ← 必须调用，schedule 靠它推进
```

### 1.1 `schedule` 的语义（最容易搞错的地方）

```text
wait=2     前 2 步：完全不记录（让 cudnn autotune / kernel 编译 / 分配器预热先过去）
warmup=2   接着 2 步：开启 profiler 但丢弃结果（让 profiler 自身也进入稳态）
active=3   接着 3 步：真正记录这 3 步
repeat=1   上述循环重复 1 次
```

**为什么必须绕过前几步**：第一次迭代包含 cuDNN benchmark 选算法、Triton / NVRTC 编译 kernel、显存分配器首次扩张。**这些都不代表稳态**，混进去会让报告完全失真。

`prof.step()` **不能漏**——`schedule` 是靠它逐 step 推进状态的；漏掉的话 profiler 会一直停在第一个状态。

### 1.2 三个值得开的开关

| 参数 | 作用 | 建议 |
|---|---|---|
| `record_shapes=True` | 记录每次 op 的输入形状 | 开。能区分"同一个 op 被不同 shape 调用" |
| `profile_memory=True` | 记录 op 级显存变化 | 开。可以顺带看显存大头 |
| `with_stack=True` | 记录 Python 调用栈 | **排查时开，稳定后关**（开销明显变大） |

### 1.3 产出的两种视图

```python
# ① 聚合表：快速定位热点
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

# ② Chrome trace：看时间线（推荐用 Perfetto 打开）
prof.export_chrome_trace("trace.json")
# 打开 https://ui.perfetto.dev 或 chrome://tracing 加载 trace.json
```

`on_trace_ready=tensorboard_trace_handler(...)` 会直接写出 TensorBoard 可读的 trace，在 TensorBoard 的 `PROFILE` 页看更省事。

---

## 二、读聚合表：列的含义与一个陷阱

| 列 | 含义 |
|---|---|
| `Self CPU` | 该 op 自身在 CPU 上花的时间（不含子 op） |
| `CPU total` | 含子 op 的累计 CPU 时间 |
| `Self CUDA` | 该 op 自身在 GPU 上花的时间 |
| `CUDA total` | 含子 op 的累计 GPU 时间 |
| `# of Calls` | 调用次数 |
| `Self CUDA Mem` | 该 op 自身的显存变化 |
| `Input Shapes` | 需要 `record_shapes=True` |

> **陷阱：`cuda_time_total` 会重复计算。** 时间线上的 op 是**嵌套**的（`nn.Module.forward` 包含它内部的 `aten::mm`），把父与子的 CUDA 时间相加会重复计数。
>
> **所以做热点排序时优先看 `self_cuda_time_total`。** 只有当你想问"这个模块总共花了多少"时，`cuda_time_total` 才有意义。

另一个常见现象：`key_averages` 里会出现 `aten::*`、`cudaLaunchKernel`、`cudaStreamSynchronize` 这些底层名字。**它们不是"你的代码"，而是你的代码在框架里的投影**——要用 `with_stack=True` 或"按模块聚合"才能映射回去。

---

## 三、时间线三模式：一眼看出瓶颈在哪

这是本笔记最实用的一张表。拿到 trace 后先判断属于哪一类：

### 模式 A：CPU 密集（GPU 有空洞）

```text
GPU 时间线：  [kernel]     ……空洞……     [kernel]     ……空洞……
CPU 时间线：  ████数据加载████  ████Python 逻辑████
```

| 特征 | 常见原因 | 下一步 |
|---|---|---|
| kernel 之间有明显间隔，`utilization.gpu` 不高 | `DataLoader` 是瓶颈（解码、增强、Python 处理） | 加 worker、`pin_memory=True`、`prefetch_factor`、把预处理移到 GPU 或提前离线做 |
| 大量细小 kernel + 大量 CPU `aten::` 调用 | Python / 框架开销（逐元素 op、循环里调小算子） | 算子融合、`torch.compile`、合并小 op |
| CPU 上耗时集中在 `torch.nn.Module._call_impl` | 只是调用树的根，不是热点 | 看 **Self CPU** 排序 |

### 模式 B：GPU 密集（kernel 背靠背）

```text
GPU 时间线：  [kernel][kernel][kernel][kernel][kernel][kernel]
CPU 时间线：  （很闲，只是偶尔入队）
```

| 特征 | 含义 | 下一步 |
|---|---|---|
| GPU kernel 几乎无空隙，`utilization.gpu` 接近 100% | **GPU 侧真的是瓶颈** | 换到 [ncu](06-nsight-compute-ncu.md) 做单 kernel 分析：是带宽受限还是算力受限 |
| 某些 kernel 特别长 | 可能是低效算子（尤其自研 kernel） | 用 nsys 看它的占比，再用 ncu 看它为什么慢 |

**注意：这一模式才是"该去看 ncu"的信号。** 模式 A 下用 ncu 分析再多的 kernel 也没有收益。

### 模式 C：同步打断

```text
GPU:  [kernel]     (等待)      [kernel]     (等待)
CPU:  ... cudaStreamSynchronize ... cudaMemcpy(D2H) ...
```

| 特征 | 常见原因 | 下一步 |
|---|---|---|
| 时间线上频繁出现 `cudaStreamSynchronize` / `cudaDeviceSynchronize` / `cudaMemcpyAsync(D2H)` | 每步都在把 GPU 结果拉回 CPU：`loss.item()`、`print(tensor)`、`if tensor > 0`、`.cpu()`、`assert`、写日志 | 减少同步频率（每 N 步记一次）、用 `.detach()` 延后、把判断逻辑挪到 GPU |

**这类问题最隐蔽也最容易修**：把每步的 `loss.item()` 改成每 50 步一次，往往立刻见效。

### 一个专门的检测工具

```python
torch.cuda.set_sync_debug_mode("warn")     # 或 1
```

打开后，**任何会隐式同步的操作都会打印警告并给出调用栈**。这是找"意料之外的同步"最快的办法，比在 trace 里肉眼找强得多。

（`0` = 关闭，`1` = 警告，`2` = 报错终止。）

---

## 四、让 trace 可读：NVTX 与 `record_function`

默认 trace 上全是 `aten::` 和 kernel 名，看不出业务含义。加标注即可：

```python
from torch.profiler import record_function

with record_function("data_load"):
    batch = next(loader)

with record_function("forward"):
    out = model(x)
```

或使用 NVTX（同时在 nsys 里可见）：

```python
torch.cuda.nvtx.range_push("optimizer.step")
optimizer.step()
torch.cuda.nvtx.range_pop()
```

要在 **nsys** 里也看到 autograd 的 op 边界，可在 `nsys` 运行时加：

```python
torch.autograd.profiler.emit_nvtx()
```

**这是把"性能数据"翻译成"业务语义"的关键一步**——不加标注的时间线，看半天也说不清是哪一块。

---

## 五、微基准：把 profiler 和"计时"分开

profiler 会改变时序，**绝对时间结论要用专门的计时工具**。

```python
# CUDA Event（异步友好）
start, end = torch.cuda.Event(True), torch.cuda.Event(True)
start.record()
out = model(x)
end.record()
torch.cuda.synchronize()              # ← 千万别忘
print(start.elapsed_time(end), "ms")

# torch.utils.benchmark：自带 warmup、自动处理异步、报告更规范
from torch.utils.benchmark import Timer
t = Timer(stmt="model(x)", globals={"model": model, "x": x})
print(t.timeit(100))
```

**规则**：任何"这个改动快了 X%"的结论，都要用上面这类工具同条件复测，而不是读 profiler 的 `CUDA total` 差值。

---

## 六、分布式训练

1. **每个 rank 各存一份**（文件名带 rank），再横向对比。**只看 rank 0 会漏掉 straggler。**
2. **通信单独看**：NCCL 的 kernel 在 trace 里表现为 `nccl:*` / `ncclDevKernel_*` / `AllReduce` 之类的名字。要回答两个问题：
   - 它占了多少总时间？
   - 它是否与计算**重叠**（时间线上是否与计算 kernel 同时存在）？
3. **重叠失败是常见病**：如果 AllReduce 与后续计算严格串行，总时间 ≈ 计算 + 通信；正确实现应该在反向传播过程中就启动梯度通信（bucket + hook）。
4. **注意开销放大**：`with_stack=True` 在每个 rank 上都有开销，分布式下会叠加，可能导致各 rank 时序漂移而互相等待。

---

## 七、`torch.profiler` 与 `nsys` 的分工

| | `torch.profiler` | [`nsys`](05-nsight-systems-nsys.md) |
|---|---|---|
| 粒度 | **op / kernel** | CUDA API / kernel / **OS 线程** / 跨进程 |
| 独有信息 | Python 调用栈、算子 shape、显存变化 | 线程调度、同步原语、NCCL、跨进程/跨卡、系统调用 |
| 与框架的关系 | 直接理解 `aten::` / `nn.Module` | 只看到 `cudaLaunchKernel`，需要 NVTX 才能对上业务 |
| 适用 | "我的模型哪一层慢" | "这台机器上 CPU / GPU / 通信怎么交错" |

**经验法则**：先用 `torch.profiler` 定位到"哪个模块 / 哪类 op"，再用 `nsys` 看系统级交错，最后用 `ncu` 看单个 kernel 的硬件指标。

---

## 八、常见误区

1. **不做 warmup 就录** → 把 cuDNN autotune 和 kernel 编译算成稳态开销。
2. **忘了调 `prof.step()`** → `schedule` 不推进，实际什么也没录到（或只录了第一步）。
3. **用 `cuda_time_total` 排序下结论** → 嵌套 op 会重复计数，应看 `self_cuda_time_total`。
4. **全程开 profiler** → 开销大、trace 巨大，而且找不到想看的那一段。
5. **GPU 有空洞却去做 kernel 级优化** → 瓶颈在 CPU / 数据加载，优化 kernel 收益为零。
6. **忘了 `torch.cuda.synchronize()` 就计时** → 测到的是 CPU 入队时间而不是 GPU 执行时间。
7. **不加 NVTX / `record_function`** → 时间线全是 `aten::xxx`，无法映射回业务模块。
8. **只在 rank 0 上 profile** → 看不到 straggler，也看不到通信与计算的重叠情况。
9. **相信单次运行的结论** → 至少重复测量，观察抖动；分布式下更要多次。

## Related

- [Profiling 专题总览](./) — 工具地图与决策树
- [Nsight Systems（nsys）](05-nsight-systems-nsys.md) — 从 op 级升级到系统级时间线
- [Nsight Compute（ncu）](06-nsight-compute-ncu.md) — 确认是 GPU 瓶颈之后的下一步
- [PyTorch 显存 profiling](03-pytorch-memory-profiling.md) — `profile_memory=True` 的进阶版本
- [系统层观测：htop、nvidia-smi 与 IO / 网络](01-system-level-observability.md) — 判断"GPU 利用率低"的粗筛
- [GPU 算子优化方法论：计算、通信、存储](../../ai/systems/gpu-kernel-optimization-methodology.md) — 判定瓶颈维度后的优化方向
- [PyTorch 专题](../../ai/systems/pytorch/) — profiler 在框架中的位置

## References

- PyTorch, [`torch.profiler` 文档](https://pytorch.org/docs/stable/profiler.html)（`schedule`、`ProfilerActivity`、`tensorboard_trace_handler`）
- PyTorch Tutorial, [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- PyTorch, [`torch.cuda.set_sync_debug_mode`](https://pytorch.org/docs/stable/generated/torch.cuda.set_sync_debug_mode.html)
- PyTorch, [`torch.utils.benchmark`](https://pytorch.org/docs/stable/benchmark_utils.html)
- [Perfetto UI](https://ui.perfetto.dev) — 打开 Chrome trace 的推荐工具（比 `chrome://tracing` 强）
