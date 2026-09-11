---
title: PyTorch 显存 profiling：三个口径、memory snapshot 与 OOM 归因
subtitle: 区分"需求大 / 碎片 / 泄漏 / 峰值超限"
type: guide
status: seed
tags: [engineering, profiling, pytorch, CUDA, memory, OOM, caching-allocator]
created: 2026-09-11
updated: 2026-09-11
source: 个人实践整理 + PyTorch CUDA memory 官方文档与 Understanding GPU Memory 教程
---

# PyTorch 显存 profiling：三个口径、memory snapshot 与 OOM 归因

## 一句话理解

排查 PyTorch 显存问题，先要分清**三个不同的口径**——混用它们是这件事上最常见的错误：

```text
torch.cuda.memory_allocated()   张量真正占用的字节            ← "我要多少"
torch.cuda.memory_reserved()    缓存分配器向驱动申请的总量      ← "我要到了多少"（含碎片与池化）
nvidia-smi 的 memory.used       reserved + CUDA context/库 + 其他进程  ← "这张卡被占了多少"
```

**三者依次增大，差值各有含义。** 搞清楚差值，OOM 的根因基本就定位了一半。

而 OOM 的根因只有四类：

| 根因 | 特征（怎么认出来） | 对策 |
|---|---|---|
| **需求真的太大** | `allocated` 峰值就超过了卡容量 | 降 batch / 梯度累积 / checkpointing / 分片 |
| **碎片化** | `num_alloc_retries > 0`，`reserved ≫ allocated` | `expandable_segments`、调整分配顺序 |
| **泄漏** | `allocated` 随 step 单调上升，不回落 | 找持引用的地方（见第五节） |
| **峰值瞬时超限** | 平均很低但某个瞬间爆掉 | 定位峰值来源（snapshot 时间线） |

## 为什么重要

- 显存是训练里最容易"突然死掉"的资源：OOM 会直接终止 job，而且往往在跑了几小时之后。
- 显存问题的排查**几乎不能靠猜**——`nvidia-smi` 只给你一个总数，只有 PyTorch 自己的分层 API 与 memory snapshot 才能告诉你"是谁在哪一行申请的"。
- 显存与速度经常冲突（checkpointing 用时间换显存、更大 batch 提高吞吐但要更多显存），所以**必须先测准，才能做取舍**。

---

## 一、三个口径与关键 API

### 1.1 基本读取

```python
import torch

device = torch.cuda.current_device()

torch.cuda.memory_allocated(device)        # 当前张量占用（字节）
torch.cuda.max_memory_allocated(device)    # 历史峰值
torch.cuda.memory_reserved(device)         # 缓存分配器持有的总量
torch.cuda.max_memory_reserved(device)     # 历史峰值

torch.cuda.reset_peak_memory_stats(device) # 重置峰值（重要！否则会看到上个阶段的峰值）
```

> **必踩的坑**：`max_memory_allocated()` 记录的是**整个进程生命周期**的峰值。如果你先跑了一个大 batch 再换成小 batch 测试，读到的仍是那个大峰值。**测每一段之前先 `reset_peak_memory_stats()`。**

### 1.2 `memory_summary()`

```python
print(torch.cuda.memory_summary())
```

它会同时打印 `allocated` / `reserved` / `active` / `inactive` 的当前值与峰值，并按**分配块大小**分桶。最有用的两行是：

```text
Allocated memory        ...    ← 实际用量
Reserved memory         ...    ← 分配器持有量
```

**`reserved - allocated` 的差额就是"被池化但没在用"的部分，它包含真正的碎片。** 这个差额长期很大（比如超过 1 GB）就值得进一步查碎片。

### 1.3 `memory_stats()`：碎片与重试的直接证据

```python
s = torch.cuda.memory_stats()
print(s["num_alloc_retries"])        # > 0 → 缓存分配器失败过，被迫重新 cudaMalloc
print(s["num_ooms"])                 # 本进程 OOM 次数
print(s["allocated_bytes.all.peak"])
print(s["reserved_bytes.all.peak"])
print(s["inactive_split_bytes.all.current"])   # 被切碎、暂时闲置的块 → 碎片规模
```

**`num_alloc_retries > 0` 是一个非常硬的信号**：说明曾经出现过"总量够但找不到合适大小的块"，即**碎片化**（或刚好卡在边界）。

---

## 二、memory snapshot：定位"是谁分配的"

这是 PyTorch 显存排查的**核心武器**。它记录缓存分配器的每一次分配/释放事件与调用栈。

### 2.1 采集

```python
import torch

# 开始记录（放在训练循环之前）
torch.cuda.memory._record_memory_history(max_entries=100000)

for step, batch in enumerate(loader):
    train_step(batch)
    if step == 10:
        break

# 落盘
torch.cuda.memory._dump_snapshot("snapshot.pickle")

# 停止记录（否则一直有额外开销）
torch.cuda.memory._record_memory_history(enabled=None)
```

要点：

- **`max_entries` 要够大**，否则只保留最近的若干条，看不到早期分配；
- **只录几步**（比如稳态的 5~10 步），既够定位又不会让文件过大；
- 这是**私有 API**（`torch.cuda.memory._`），跨版本可能变化，但它是官方推荐做法。

### 2.2 可视化

把 `snapshot.pickle` 拖到 <https://pytorch.org/memory_viz>（纯前端，文件不会上传）。三种视图各有分工：

| 视图 | 用途 |
|---|---|
| **Active Memory Timeline** | 一张"显存随时间"的曲线 + 每次分配的方块。**看峰值出现在哪一步、由哪些分配构成** |
| **Allocation Flame Graph** | 按调用栈聚合。**看是哪个模块/函数申请的**（需要记录 stacks） |
| **Allocator State History** | 看分配器的 OOM / retry 事件发生在什么状态 |

> 记录 stacks 需要 `_record_memory_history(stacks="all")`（默认在部分版本里就是 `"all"`）。**没有栈，火焰图就是空的**——这是第一次用最常见的困惑。

### 2.3 一次典型的阅读顺序

```text
1. 时间线上找到峰值点 → 对应到第几个 step
2. 看峰值点附近有哪些大块分配 → 是激活值？梯度？还是优化器状态？
3. 切到火焰图 → 找到申请它的代码路径
4. 如果是"逐 step 累积不回落" → 走泄漏排查（第五节）
```

---

## 三、碎片化：有总量却没有连续块

### 3.1 机制

缓存分配器会保留释放过的块以备复用，但**复用要求大小匹配**：

```text
初始： [-------------- 大块 4GB --------------]
中间： [激活 1GB][激活 1GB][激活 1GB][激活 1GB]      ← 被切碎
释放： [  空 1GB ][激活 1GB][  空 1GB ][激活 1GB]    ← 有 2GB 空闲，但没有连续 2GB
请求： 需要一个 2GB 的块  → 找不到 → retry → 仍失败 → OOM
```

**这就是为什么"显存明明还有但就是 OOM"。**

### 3.2 判据

```text
num_alloc_retries > 0                     → 已经发生过碎片导致的失败
reserved - allocated 长期很大（数 GB）      → 池里躺着大量无法复用的零碎块
nvidia-smi 还有余量但 PyTorch OOM          → 典型碎片症状
```

### 3.3 对策

```bash
# 方式一：环境变量（推荐，最省事）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py
```

`expandable_segments:True` 让分配器使用**可增长的虚拟段**，大幅缓解"块大小不匹配"造成的碎片。

其他手段：

| 手段 | 说明 |
|---|---|
| `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` | 限制块被切分的粒度，减少过小的碎块（缓解分裂，但可能浪费） |
| 在 step 开头**集中分配** | 避免"释放 → 申请 → 释放 → 申请"交错撕裂地址空间 |
| 避免变长输入 | seq len 剧烈变化会让分配大小频繁改变，加剧碎片 |
| 周期性 `torch.cuda.empty_cache()` | 只在**明确的阶段边界**用（例如 epoch 切换），不要每个 step 调，会拖慢 |

> `torch.cuda.empty_cache()` **不能解决真正的 OOM**：它只把缓存分配器里"空闲"的块还给驱动，正在使用的部分一个都不会释放。它只会让 `nvidia-smi` 的数字下降，并让随后的分配变慢。

---

## 四、泄漏：显存随 step 单调上升

### 4.1 典型模式（按出现频率排序）

| 模式 | 例子 | 修法 |
|---|---|---|
| **把带图的 tensor 累积进容器** | `losses.append(loss)` | 改成 `loss.item()` 或先 `.detach()` |
| **把中间结果挂到 module / self 上** | `self.cache = feats` | 用完置 `None`，或在 `forward` 里用局部变量 |
| **`retain_graph=True` 忘记关** | 自定义 backward | 只在需要多次 backward 时开，用完即关 |
| **未 detach 的图被闭包/hook 持有** | `register_forward_hook` 里存 output | hook 内 `.detach()` |
| **分布式下每个 rank 各记一份** | 只在 rank 0 记录时看起来"没涨" | 每个 rank 都监控 |
| **日志/评审工具持有引用** | wandb / tensorboard 记了 tensor | 记标量而不是 tensor |

### 4.2 排查步骤

```text
1. 用 memory snapshot 的【时间线】确认"是否真的单调上升"
   （注意：优化器状态、EMA 等本来就会在第一 step 后稳定到一个平台，那不是泄漏）
2. 用【火焰图】看反复出现的大块分配来自哪条路径
3. 缩小范围：注释掉可疑模块，看曲线是否变平
4. Python 对象层用 objgraph / tracemalloc 交叉验证（见 02 号笔记）
```

> **判据**：真正的泄漏表现为"**每个 step 净增 X MB，永不回落**"。如果曲线是"阶段性地台阶式上升然后稳定"，那通常是不同阶段（前向/反向/优化器）的正常构成。

---

## 五、降显存的常见手段（按代价排序）

| 手段 | 省什么 | 代价 |
|---|---|---|
| 减小 micro-batch + 梯度累积 | 激活值（线性） | 步数不变，但每步多次前反向，吞吐略降 |
| 混合精度 / bf16 | 激活值、参数、优化器状态（约一半） | 数值稳定性需要调 |
| Gradient checkpointing | 激活值（从 $O(L)$ 降到 $O(\sqrt L)$ 量级） | 多一次前向，时间增加 |
| 优化器状态分片（ZeRO / FSDP） | 优化器状态、梯度、参数（按 shard 数） | 通信量上升，需要调 overlap |
| 及时释放中间量 | 峰值而非平均 | 需要审代码，容易漏 |
| 就地操作（inplace） | 中间副本 | 破坏 autograd 需要的中间值时有风险 |
| `expandable_segments` | 碎片 | 无（推荐默认开） |

**关键认识**：显存占用的大头通常是**优化器状态 + 梯度 + 激活值**，而不是参数本身。Adam 的 fp32 一阶/二阶矩就是参数量的 2 倍。

---

## 六、常见误区

1. **用 `nvidia-smi` 的数字解释 PyTorch OOM** → 口径差着 context、库和其他进程，会得出错误结论。
2. **`torch.cuda.memory_allocated()` 就是"PyTorch 用了多少显存"** → 它不含池化与碎片，`reserved` 才是分配器持有量。
3. **`torch.cuda.empty_cache()` 能解决 OOM** → 不能。它只归还空闲块，并带来额外开销。
4. **`del x` 之后显存立刻释放** → 张量内存会**归还给缓存分配器**（`allocated` 下降），但 `reserved` 通常不降（这正是池化的目的）。
5. **只看一个 step 的瞬时值** → 必须看峰值（`max_memory_allocated`），OOM 发生在峰值。
6. **忘记 `reset_peak_memory_stats()`** → 读到的峰值来自更早的实验阶段。
7. **snapshot 火焰图是空的** → 说明没记录调用栈（`stacks` 未开启）。
8. **"显存有平台期就不是泄漏"** → 要区分"正常平台"（优化器状态）与"台阶式持续上升"（泄漏）。
9. **只在 rank 0 上监控显存** → 各 rank 的峰值可能不同，straggler rank 才是 OOM 的那一个。

## Related

- [Profiling 专题总览](./) — 工具地图与决策树
- [PyTorch latency profiling：torch.profiler](04-pytorch-latency-profiling.md) — 显存解决后接着看时间
- [Python profiling：cProfile、py-spy、scalene 与 memray](02-python-profiling.md) — 内存泄漏的 Python 对象层排查
- [系统层观测：htop、nvidia-smi 与 IO / 网络](01-system-level-observability.md) — `nvidia-smi` 显存口径的正确读法
- [GPU 算子优化方法论：计算、通信、存储](../../ai/systems/gpu-kernel-optimization-methodology.md) — SMEM / 寄存器容量约束与 occupancy 的关系
- [PyTorch 专题](../../ai/systems/pytorch/) — 缓存分配器在框架中的位置

## References

- PyTorch, [torch.cuda.memory — CUDA memory management](https://pytorch.org/docs/stable/torch_cuda_memory.html)
- PyTorch Blog, [Understanding GPU Memory 1: Visualizing All Allocations over Time](https://pytorch.org/blog/understanding-gpu-memory-1/) / [2: Find and Remediate Memory Leaks](https://pytorch.org/blog/understanding-gpu-memory-2/)
- PyTorch, [CUDA memory snapshot 可视化工具](https://pytorch.org/memory_viz)
- PyTorch, [`torch.cuda.memory_stats`](https://pytorch.org/docs/stable/torch_cuda_memory.html#torch.cuda.memory_stats) 文档中的计数器含义
