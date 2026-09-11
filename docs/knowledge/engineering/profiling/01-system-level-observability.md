---
title: 系统层观测：htop、nvidia-smi 与 IO / 网络
subtitle: profiling 的第一站——零成本排除"机器不够用"
type: guide
status: seed
tags: [engineering, profiling, htop, nvidia-smi, nvitop, iostat, observability, linux]
created: 2026-09-11
updated: 2026-09-11
source: 个人实践整理 + NVIDIA / sysstat 官方文档
---

# 系统层观测：htop、nvidia-smi 与 IO / 网络

## 一句话理解

这是**最便宜的一层**。它不能告诉你"为什么慢"，但能在几秒内回答一个更重要的问题：**机器的哪一类资源被打满了，以及是谁打的**。

如果连 CPU / 显存 / IO / 网络都没饱和，那就不是"资源不够"的问题，应该马上换到进程内工具（[py-spy / cProfile](02-python-profiling.md)）或时间线工具（[nsys](05-nsight-systems-nsys.md)），而不是继续在这里盯着看。

## 为什么它是第一站

| 理由 | 说明 |
|---|---|
| 成本几乎为零 | 不侵入程序、不需要重跑、可以在生产 job 上一直挂着 |
| 能直接排除一整类可能 | "是不是机器被别的任务抢了" —— 这是最常见的伪性能问题 |
| 提供后续分析的必要上下文 | 后边的 nsys / ncu 结果必须放回"当时机器是什么状态"才能解释 |
| 能抓偶发问题 | 长 job 里周期性采集（每 5~10 秒一行日志）能抓到网络抖动、CPU 抢占、显存缓慢爬升 |

---

## 一、CPU 与内存

### 1.1 htop 的常用操作

`htop` 相比 `top` 的价值在交互：可以排序、过滤、按树状看父子进程。

| 按键 | 作用 |
|---|---|
| `F5` / `t` | **树状视图**（看父子进程，分布式训练里找 straggler 很有用） |
| `F6` | 选择排序字段 |
| `F4` / `\` | 按关键字**过滤**（只看某个训练进程） |
| `u` | 按用户过滤 |
| `H` | 开关"是否把线程单独列出" |
| `P` / `M` | 按 CPU% / MEM% 排序 |
| `k` / `F9` | 发送信号（kill） |
| `F2` | 设置（可以把 `PERCENT_CPU`、`PERCENT_MEM`、`IO` 等列打开） |

> **多核计数的坑**：`htop` 里一个进程的 CPU% 是**所有核加起来**的百分比，单核跑满就是 `100%`，8 核跑满能显示到 `800%`。而顶部总的 CPU% 条是**平均**口径。两者口径不同，不要混读。

### 1.2 三个关键判据

**① load average 要对照逻辑核数看。**

`load average: 1.52, 2.10, 1.87` 表示过去 1 / 5 / 15 分钟的**平均可运行任务数 + 不可中断（D 状态）任务数**。判据：

```text
load ≈ 核数      → 刚好跑满
load > 核数      → 有排队，可能是 CPU 不够
load >> 核数 但 CPU% 不高  → 大概率卡在 IO（D 状态）
```

**要看 D 状态**：load 高而 CPU 空闲，通常意味着进程在等磁盘或网络。

**② 看是"单核打满"还是"全核均衡"。**

这是一个信息量极大的判据：

| 现象 | 常见原因 |
|---|---|
| 只有 1 个核 100% | Python GIL 限制、单线程 DataLoader、序列化/反序列化瓶颈 |
| 少数几个核打满 | 单进程多线程，但并行度不足 |
| 全核均衡打满 | CPU 侧确实是瓶颈（CPU 算子、数据预处理） |
| 全核都很闲，GPU 也很闲 | 有锁 / 同步 / 等待，去看时间线 |

**③ 内存要看 `available` 而不是 `free`。**

Linux 会把空闲内存用作 page cache，所以 `free` 低是正常的。真正要看：

```bash
free -h          # 关注 available 列与 swap 使用
vmstat 1         # 关注 si/so（swap in/out）；非 0 说明已经开始换页，性能断崖
```

### 1.3 进程级与每核级

```bash
pidstat -p <PID> 1        # 某个进程的 CPU / 内存 / IO（sysstat）
mpstat -P ALL 1           # 每个核的利用率，验证"是不是单核打满"
atop                      # 历史回看能力最强的综合工具
```

---

## 二、GPU

### 2.1 最常用的三条命令

```bash
# ① 看一眼（默认输出）
nvidia-smi

# ② 持续刷新
nvidia-smi -l 1

# ③ 适合写日志 / 事后分析的 CSV 形式
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
           --format=csv -l 5
```

查"哪块卡被哪个进程占了"：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

> 如果训练的 rank 数少于卡数，或者怀疑有残留进程，这条命令是必查项。

### 2.2 两个必须理解的口径问题

**① `utilization.gpu` 高 ≠ GPU 用得好。**

它只表示"采样窗口内有 kernel 在执行的时间占比"。两种极端都可能出现：

```text
util = 100%  但实际很慢   →  一堆积木一样的小 kernel 串行跑，没喂饱 Tensor Core
util = 0%   但程序在跑    →  CPU 侧在准备数据 / Python 开销 / 正在同步等待
```

**所以它是"有没有空洞"的粗筛，不是效率指标。** 真要判断效率，必须看时间线（[nsys](05-nsight-systems-nsys.md)）和单 kernel 指标（[ncu](06-nsight-compute-ncu.md)）。

**② `memory.used` 与 `torch.cuda.memory_allocated()` 不是一回事。**

```text
nvidia-smi 显示的显存
  = torch 缓存分配器 reserved 的部分
  + CUDA context / cuDNN / cuBLAS 等库占用的部分
  + 同卡上其他进程占用的部分
```

所以二者差几百 MB 到几 GB 都算正常。**排查 OOM 要看 [PyTorch 显存 profiling](03-pytorch-memory-profiling.md)，不要用 nvidia-smi 的数字做结论。**

### 2.3 频率与降频

性能"忽快忽慢"或"比预期慢"时，先确认不是被降频：

```bash
nvidia-smi -q -d PERFORMANCE      # 看 Clocks Throttle Reasons
nvidia-smi -q -d POWER            # 功耗是否顶到 power cap
```

常见的降频原因：达到功耗墙（`sw_power_cap`）、温度过高（`hw_thermal_slowdown`）、显存频率受限（`sw_memory_throttle`）。

### 2.4 多卡与拓扑

```bash
nvidia-smi topo -m        # 卡间连接方式：NVLink / PCIe / SYS
nvidia-smi nvlink -s      # NVLink 链路状态与速率
```

分布式训练里，**通信走 NVLink 还是跨 NUMA 走 PCIe，性能可以差一个量级**。看到 NCCL 很慢时这是第一个要查的地方。

### 2.5 更好用的替代品

| 工具 | 特点 |
|---|---|
| `nvitop` | 终端 GUI，直接显示**每个 GPU 上的进程、用户、显存、利用率历史曲线**，多卡场景比 `nvidia-smi -l` 好用得多 |
| `nvtop` | 类似的终端监视器 |
| `dcgmi dmon` | NVIDIA DCGM，适合多机多卡的**批量指标采集**（字段很多，可用 `dcgmi dmon -e <fieldId>` 挑选） |

---

## 三、磁盘 IO

DataLoader 瓶颈、checkpoint 写入、日志刷盘都会在这里显形。

```bash
iostat -x 1       # 关注 %util、await、r/s、w/s
iotop -oPa        # 按进程看 IO；-o 只显示有 IO 的，-P 按进程，-a 累计
df -h             # 容量
du -sh <dir>      # 目录占用
```

判读要点：

- **`%util` 接近 100%** → 设备饱和（注意：对 NVMe 多队列设备，`%util` 的参考价值有限）；
- **`await` 明显高于设备正常延迟** → 有排队；
- **大量小文件随机读** → 常见于数据加载，考虑打包成 webdataset / 提高顺序性；
- **checkpoint 周期性地打满写带宽** → 会阻塞训练步，考虑异步写或错峰。

---

## 四、网络（分布式训练必看）

```bash
sar -n DEV 1              # 每张网卡的收发速率
iftop -i <iface>          # 按连接看实时流量（定位是谁在传）
ip -s link                # 网卡累计计数，关注 errors / dropped
```

分布式训练的关注点：

1. **NCCL 走的是哪张网卡**（计算网 vs 存储网是否混用）；
2. **带宽是否打满**——若 AllReduce 已经吃满网卡，那它就是瓶颈，不是"通信效率低"；
3. **是否有 dropped / retransmit**——偶发抖动足以让整个 job 变慢，且在单机上看不出来。

---

## 五、长 job 的正确姿势：把它写成日志

单次 `htop` 只能拍到一瞬间。对长时间训练，**周期性采集**才有效：

```bash
# 每 10 秒一行 GPU 指标，落到文件里
nohup nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
                 --format=csv -l 10 > gpu_metrics.csv &
```

配合 CPU 侧（`pidstat -p <PID> 10`）与网络侧（`sar -n DEV 10`）一起记，训练结束或出问题时回看，能回答"是哪一步开始变慢的"。

> 这类采集的开销可以忽略，但价值很大：**它把"偶发问题"变成了"可复现的曲线"**。

---

## 六、常见误区

1. **"`nvidia-smi` 利用率 100% 说明 GPU 用得好"** → 错。它只表示"有 kernel 在跑"，不表示 kernel 高效。
2. **"利用率 0% 说明卡是空的"** → 不一定。可能是采样窗口恰好落在 CPU 准备阶段，或程序正在同步等待。
3. **"load average 高就是 CPU 不够"** → 要看 D 状态。IO 等待同样会把 load 推高。
4. **"内存 free 很低就是内存不够"** → 错，要用 `available`，并检查 swap 是否真的在被使用。
5. **"只用一台机器上看就够了"** → 分布式下必须每台都看，平均值会掩盖 straggler。
6. **"`htop` 里 CPU% 是 100% 就是占满机器了"** → 那是单核口径，多核机器上要看总数与 `mpstat`。
7. **"`nvidia-smi` 的显存数字可以直接用来排查 PyTorch OOM"** → 口径不同（见 2.2），要换成 PyTorch 自己的 API。

## Related

- [Profiling 专题总览](./) — 工具地图与决策树
- [Python profiling：cProfile、py-spy、scalene、memray](02-python-profiling.md) — 排除机器问题后的下一站
- [PyTorch 显存 profiling](03-pytorch-memory-profiling.md) — `nvidia-smi` 显存数字的正确替代品
- [PyTorch latency profiling：torch.profiler](04-pytorch-latency-profiling.md)
- [Nsight Systems（nsys）](05-nsight-systems-nsys.md) — 从"利用率高不高"升级到"时间线上有没有空洞"
- [Nsight Compute（ncu）](06-nsight-compute-ncu.md)
- [GPU 算子优化方法论：计算、通信、存储](../../ai/systems/gpu-kernel-optimization-methodology.md) — 为什么"资源饱和"只是表象

## References

- `htop`、`top`、`sysstat`（`pidstat` / `mpstat` / `iostat` / `sar`）、`iotop`、`atop` 手册
- NVIDIA, [nvidia-smi 文档](https://docs.nvidia.com/deploy/nvidia-smi/)
- NVIDIA, [DCGM 用户指南](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/)
- [nvitop](https://github.com/XuehaiPan/nvitop)、[nvtop](https://github.com/Syllo/nvtop)
