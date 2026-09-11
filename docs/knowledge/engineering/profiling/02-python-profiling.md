---
title: Python profiling：cProfile、py-spy、scalene 与 memray
subtitle: 定位 Python 侧热点、阻塞与内存增长
type: guide
status: seed
tags: [engineering, profiling, python, cProfile, py-spy, scalene, memray, memory]
created: 2026-09-11
updated: 2026-09-11
source: 个人实践整理 + 各工具官方文档
---

# Python profiling：cProfile、py-spy、scalene 与 memray

## 一句话理解

在动手之前先分清**四种完全不同的"慢"**，它们需要不同的工具：

| 症状 | 本质 | 合适的工具 |
|---|---|---|
| CPU 在忙，但不知道忙在哪 | **CPU 热点** | `cProfile`、`line_profiler`、`py-spy` |
| CPU 不忙，程序却要等很久 | **阻塞等待**（IO / 锁 / 网络 / 同步） | `py-spy`（看栈）、`strace`、时间线工具 |
| 内存（或显存）持续上涨 / OOM | **内存增长** | `tracemalloc`、`memray`、`objgraph` |
| Python 代码很薄，时间在 C / CUDA 里 | **native 时间** | `scalene`、`torch.profiler`、`nsys` |

**用错工具比不测还糟**：拿 `cProfile` 去诊断"卡在等网络"，你只会得到一张"什么都没干"的火焰图。

## 工具选型

| 工具 | 机制 | 开销 | 优点 | 局限 |
|---|---|---|---|---|
| `cProfile` | 确定性插桩（每个函数调用都记录） | 中 ~ 高 | 精确的调用次数与嵌套关系；标准库自带 | 看不到阻塞；看不到 C 扩展内部；开销会扭曲时序 |
| `py-spy` | 采样（默认 100 Hz） | **极低** | **不改代码、可附加到运行中进程**、可线上、能看 native 栈 | 看不到比采样间隔更短的事件；只有统计意义 |
| `scalene` | 采样 + 行级 | 低 ~ 中 | **区分 Python 时间 / native 时间 / 内存 / GPU**；行级粒度 | 需要安装；对极短脚本意义不大 |
| `line_profiler` | 行级插桩 | 高 | 精确到行 | 需要加装饰器、改代码；开销大 |
| `tracemalloc` | 分配点采样（Python 层） | 低 | 标准库；能给出**分配点的文件名+行号** | 看不到原生分配（C / numpy 内部） |
| `memray` | 跟踪所有分配（含 native） | 中 | 火焰图 / 表 / 实时；可 attach | 需要安装；大程序 trace 文件大 |

---

## 一、cProfile：函数级热点的标准答案

### 1.1 两种用法

```bash
# 命令行：整个脚本
python -m cProfile -o out.prof script.py

# 交互式查看
python -m pstats out.prof
#   sort cumtime
#   stats 20
```

```python
# 代码内：只测某一段
import cProfile

with cProfile.Profile() as pr:
    main()

pr.print_stats(sort="cumtime")
pr.dump_stats("out.prof")
```

可视化（强烈推荐，比文本表直观得多）：

```bash
pip install snakeviz
snakeviz out.prof          # 打开浏览器看 icicle / sunburst 图
```

### 1.2 最关键的一件事：`tottime` vs `cumtime`

这是读 `cProfile` 结果时最容易错的地方：

| 列 | 含义 | 用途 |
|---|---|---|
| `tottime` | 该函数**自身代码**消耗的时间（**不含**子调用） | 找"真正在算"的函数 |
| `cumtime` | 该函数**及其所有子调用**的累计时间（**会重复计算**） | 找"入口/瓶颈分支" |
| `ncalls` | 调用次数 | 找"被调用太多次"的小函数 |
| `percall` | `tottime/ncalls` 或 `cumtime/ncalls` | 找"单次很贵"的函数 |

**判读方法**：

```text
cumtime 很高、tottime 很低   → 它只是"路过"的父函数，真凶在它的子调用里
cumtime 高、tottime 也高     → 它自己就慢，是热点
tottime 低但 ncalls 巨大      → 单次便宜但调用太多，考虑向量化 / 缓存 / 上移循环外
```

对宽而浅的模块（比如 `torch/nn/modules/module.py` 的 `_call_impl`）几乎总会出现在 `cumtime` 榜首，**那只是调用树的根，不是热点**。

### 1.3 局限（决定了它不能包打天下）

1. **看不到阻塞**：`cProfile` 统计的是 CPU 时间，函数里若在 `socket.recv` 上等了 10 秒，它几乎不显示。
2. **看不到 C 扩展内部**：`numpy` / `torch` 的算子会整体算作一次调用，内部细节测不到。
3. **开销本身会扭曲结果**：对"大量极小函数"的代码，插桩开销可能让相对时间完全变形。
4. **多进程不覆盖**：只测当前进程，`DataLoader(num_workers>0)` 的子进程不会被记录。

> **在 GPU 训练里用 `cProfile` 需要特别小心**：CUDA 调用是**异步入队**的，CPU 侧只是把 kernel 排进队列。所以 `cProfile` 会告诉你"时间花在 `cudaLaunchKernel` 上"，这句话既不假也无用——真正的时间在 GPU 上，要用 [torch.profiler](04-pytorch-latency-profiling.md) 或 [nsys](05-nsight-systems-nsys.md)。

---

## 二、py-spy：不改代码、可线上、低开销

`py-spy` 是采样型剖析器，直接读目标进程的内存来重建 Python 栈，**不需要修改代码、不需要重启、不需要 `--inspect` 参数**。

```bash
pip install py-spy

py-spy top --pid <PID>                     # 实时 top（类似 htop 的界面，但是函数级）
py-spy dump --pid <PID>                    # 抓一次当前所有线程的调用栈（最轻量）
py-spy record -o profile.svg --pid <PID> --duration 30
py-spy record -o profile.svg -- python train.py
```

常用开关：

| 开关 | 作用 |
|---|---|
| `--native` | 同时采样 C / C++ / Rust 栈（能看到 native 时间去哪了） |
| `--idle` | **包含 GIL 空闲的线程** —— 看"这个线程是在等什么" |
| `--subprocesses` | 跟踪子进程（`DataLoader` worker、`torchrun` 的 rank） |
| `-r/--rate` | 采样频率（默认 100 Hz） |
| `--threads` | 在输出里按线程拆分（默认合并） |

### 什么时候 `py-spy` 优于 `cProfile`

```text
程序已经跑起来了，不想重跑        → py-spy attach
程序太慢，加插桩会失真            → py-spy（采样）
怀疑卡在 IO / 锁 / 等待           → py-spy --idle（cProfile 根本看不到）
怀疑时间在 C 扩展里               → py-spy --native
```

**`py-spy dump --pid` 是排查"某个 rank 卡住不动"的利器**：对着卡住的进程 dump 一次，立刻知道它停在哪个调用栈上（经常是某次集合通信或一次 `cudaMemcpy` 同步）。

局限性：只有统计意义，看不到比采样间隔（默认 10 ms）更短的事件；采样意味着多次运行结果会有细微差别。

---

## 三、scalene：一份报告同时给 CPU、native、内存

`scalene` 的独特价值是**把"Python 解释器时间"和"native 时间"分开统计**，并且做到行级：

```bash
pip install scalene
scalene --html --outfile prof.html script.py
scalene --cli script.py            # 终端输出
scalene --memory --cpu --profile-all script.py
```

输出里会区分：

```text
% Python      解释器执行 Python 字节码的时间
% native      C / C++ / 系统调用时间（numpy、torch 的算子都归这里）
% system      内核时间
% GPU         （若启用）GPU 上的时间
内存增长      net / peak memory，按行给
```

**最有用的判读**：如果某行的 `native` 占比极高而 `Python` 很低，说明时间在库函数里（例如一次大的 `tensor.cuda()` 或 `np.argsort`）；如果 `Python` 占比高，说明瓶颈是自己的 Python 逻辑（循环、属性访问、类型判断）。

它还能检测"**跨线程复制数据的竞争**"和"**GIL 争用**"这类很难手工发现的问题。

---

## 四、line_profiler：精确到行

`cProfile` 只到函数级。要知道函数里**哪一行**慢：

```bash
pip install line_profiler
kernprof -l -v script.py
```

```python
@profile            # 名字固定为 profile，由 kernprof 注入
def hot_function(x):
    a = x * 2                           # 每行的 Hits / Time / % Time 会打印出来
    b = sum(a)
    return b
```

开销很大（**逐行**插桩），只用于已经被确认的热点函数，不要拿来测整个程序。

---

## 五、内存：tracemalloc 与 memray

### 5.1 tracemalloc：标准库，找"Python 对象的分配点"

```python
import tracemalloc

tracemalloc.start(25)                   # 25 = 每个分配点保留 25 帧栈

snap1 = tracemalloc.take_snapshot()
run_first_half()
snap2 = tracemalloc.take_snapshot()

for stat in snap2.compare_to(snap1, "lineno")[:10]:
    print(stat)                         # 直接给出 文件:行号 与增量
```

要点：

- **`compare_to` 才是正确用法**——只在一个时间点看快照，分不清"正常占用"和"泄漏"；
- `tracemalloc.start(n)` 里的 `n` 决定能回溯多少层栈，太小会定位不到源头；
- 只跟踪 Python 层分配，`numpy` / `torch` 的大块 native 内存**看不到**。

### 5.2 memray：能看 native 分配

```bash
pip install memray

memray run -o out.bin script.py
memray flamegraph out.bin            # 生成交互式火焰图（html）
memray table out.bin                 # 表格式
memray stats out.bin                 # 峰值 / 分配次数概览
memray attach <PID>                  # 附加到运行中的进程
memray run --live script.py          # 实时终端视图
```

`memray` 能捕获 C / C++ 层的分配，因此**能定位 numpy / pyarrow / C 扩展造成的内存增长**，这是 `tracemalloc` 做不到的。

### 5.3 确认"是不是真的有泄漏"

```python
import gc, objgraph

objgraph.show_most_common_types(limit=15)   # 当前最多的对象类型
objgraph.show_growth(limit=10)              # 与上次调用相比"增长最多"的类型
len(gc.get_objects())                       # 对象总数随时间是否单调上升
```

**最经典的泄漏原因**：把 tensor / loss 累积进一个 list（`losses.append(loss)` 而不是 `losses.append(loss.item())`）——它同时会造成显存泄漏，见 [PyTorch 显存 profiling](03-pytorch-memory-profiling.md)。

---

## 六、微基准：不要用 profiler 的结果当性能结论

profiler 会改变时序，**绝对时间结论必须用专门的微基准工具**：

| 工具 | 适用 |
|---|---|
| `time.perf_counter()` | 手动圈定代码段（记得多次重复取中位数） |
| `timeit` | 极短表达式的重复计时（自动选择重复次数） |
| `torch.utils.benchmark.Timer` | **CUDA 场景的正确选择**：自带 warmup、自动处理异步、能报多个测量口径 |

```python
# 常见错误：没有 sync 就计时（测到的是 CPU 入队时间）
t0 = time.perf_counter()
y = model(x)
print(time.perf_counter() - t0)      # 错：GPU 还没算完

# 正确
torch.cuda.synchronize()
t0 = time.perf_counter()
y = model(x)
torch.cuda.synchronize()
print(time.perf_counter() - t0)
```

---

## 七、多进程与分布式场景

1. **`cProfile` / `tracemalloc` 只覆盖当前进程。** `DataLoader(num_workers=k)` 的 k 个 worker、`torchrun` 的 N 个 rank 都要单独测。
2. **`py-spy --subprocesses`** 可以一次跟住整棵进程树，是分布式排查的首选。
3. **每个 rank 各自输出一份报告**（文件名里带上 rank 号），再做对比——**不要只看 rank 0**，straggler 往往不是它。
4. **注意时序漂移**：如果各 rank 的分析是分别启动的，先排除"不同步"带来的干扰。

---

## 八、常见误区

1. **用 `cProfile` 诊断"慢在等待"** → 它统计 CPU 时间，会把阻塞函数显示得几乎不耗时。
2. **只看 `cumtime` 排名就下结论** → 根函数必然排第一；要结合 `tottime` 与 `ncalls`。
3. **`ncalls` 里出现 `1/3` 这种写法以为是 bug** → 那是"外部调用 1 次、含递归共 3 次"的记法。
4. **在 GPU 训练代码上用 `cProfile` 看性能** → 会被 `cudaLaunchKernel` 之类的入队开销误导。
5. **`tracemalloc` 只看一个快照** → 必须 `compare_to` 才有意义。
6. **内存问题只看 Python 层** → 大块 native 分配要用 `memray`。
7. **采样器看不到某个函数就以为它不耗时** → 采样间隔（默认 10 ms）以下的调用会被漏掉，可提高 `--rate` 复查。
8. **时序结论来自 profiler 输出** → 要用 `torch.utils.benchmark` 之类的专用工具复测。

## Related

- [Profiling 专题总览](./) — 工具地图与决策树
- [系统层观测：htop、nvidia-smi 与 IO / 网络](01-system-level-observability.md) — 上一层：先确认不是机器不够用
- [PyTorch 显存 profiling](03-pytorch-memory-profiling.md) — `tracemalloc` 覆盖不到的显存问题
- [PyTorch latency profiling：torch.profiler](04-pytorch-latency-profiling.md) — Python 很薄、时间在 GPU 上时的正确工具
- [Nsight Systems（nsys）](05-nsight-systems-nsys.md) — 跨进程、跨线程的系统级时间线

## References

- Python 官方文档：[`cProfile` / `pstats`](https://docs.python.org/3/library/profile.html)、[`tracemalloc`](https://docs.python.org/3/library/tracemalloc.html)、[`timeit`](https://docs.python.org/3/library/timeit.html)
- [py-spy](https://github.com/benfred/py-spy) — Sampling profiler for Python programs
- [scalene](https://github.com/plasma-umass/scalene) — CPU / GPU / memory profiler with line granularity
- [memray](https://github.com/bloomberg/memray) — Python memory profiler (native allocations included)
- [line_profiler](https://github.com/pyutils/line_profiler)、[objgraph](https://github.com/mgedmin/objgraph)、[snakeviz](https://jiffyclub.github.io/snakeviz/)
