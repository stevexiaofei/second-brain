---
title: 磁盘 IO 压测：disk_io_bench.py 解读与存储基准方法论
subtitle: 观测告诉你“谁在用磁盘”，压测提供受控负载下的能力对比
type: guide
status: seed
tags: [engineering, profiling, disk, io, benchmark, fsync, page-cache, nfs, storage]
created: 2026-11-04
updated: 2026-11-09
source: tools/disk_io_bench.py 程序解读 + 个人整理（Linux man 页：fadvise/fsync/open）
---

# 磁盘 IO 压测：disk_io_bench.py 解读与存储基准方法论

## 一句话理解

> [01 号笔记](01-system-level-observability.md)的 iostat/iotop 回答“**现在谁在用磁盘、用得健不健康**”；这篇回答互补的另一半——“**在这组受控负载下，挂载点表现如何**”。`disk_io_bench.py` 用 6 个 phase 对比顺序/随机、读/写、批量同步/逐操作同步。它通过 `posix_fadvise(DONTNEED)` 请求驱逐客户端 page cache，但不是 O_DIRECT，也不保证每次读取都命中物理盘。

## 为什么重要

- checkpoint、日志刷盘、随机数据读取关注的指标不同，不能只用一个“带宽”数字概括存储能力。
- **先读计时边界，再读数字**：脚本的 buffered 写吞吐也包含最终 fsync，不是仅测“数据进内存”的速度。与逐操作同步的差距包含批处理、写合并、请求频率等因素。
- 现场没有 fio、没有 root 时，标准库脚本可快速横向对比挂载点；但不能替代设备极限、元数据或完整 DataLoader 基准。

## 核心概念：六个 phase

| Phase | 操作与计时边界 | 指标 | 回答的问题 |
|---|---|---|---|
| `seq_write_buf` | 1MiB 块顺序写，最终 fsync 也计入总耗时 | MiB/s | 批量写入加一次同步的端到端吞吐 |
| `seq_write_sync` | 每个 1MiB 块写后 fsync | MiB/s | 高频同步屏障对顺序写的影响 |
| `seq_read` | 对完整文件请求 cache 驱逐后顺序读 | MiB/s | 尽量减少客户端页缓存影响的顺序读表现 |
| `rand_write_4k_buf` | 4KiB 随机覆盖写，最终 fsync 计入总耗时 | IOPS | 批量随机覆盖写吞吐 |
| `rand_read_4k` | 对完整初始化文件请求 cache 驱逐后，4KiB 随机读 | IOPS、平均 ms | 该随机读模式下的单线程表现 |
| `rand_write_4k_sync` | 每次 4KiB 随机覆盖写后 fsync | IOPS、p99 ms | 写入加同步的逐操作尾延迟 |

带宽统一按二进制 MiB 计算；`--size-mb` 的单位也是 MiB，名称为兼容原命令保留。`fsync_lat_p99` 是历史指标名，实际计时包含 **pwrite + fsync**，不是纯 fsync 系统调用耗时。

**读表原则**：先按实际工作负载选指标，再比较相同条件下的挂载点。checkpoint 往往更接近“一批大写入后同步”，未必接近“每 1MiB fsync”；单个大文件随机读也不等于 DataLoader 的小文件读取与解码。

## 工作原理

### 1. cache 控制：尽力减少客户端页缓存，不保证物理读

```python
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
```

`POSIX_FADV_DONTNEED` 是给内核的**建议**。之前的写完成 fsync，有助于使页变为可回收的 clean 页；脏页通常不会直接丢弃。但建议可能不被完全执行，其他进程也可能重新读入这些页。

它不会清理 NFS 服务端、存储控制器或设备缓存。随机读重复访问同一 offset 时，后续访问仍可能命中重新装入的页。若文件是稀疏文件，读洞可直接得到零，即使客户端 cache 被清理，也未必产生后端数据读。

**写侧尤其不能误读**：`seq_write_buf` 和 `rand_write_4k_buf` 都把最终 fsync 放在计时区间内。小于 RAM 的文件也不只是测 page cache 的接收速度；但批处理/写合并、存储端缓存与持久化语义仍影响结果。文件大于两倍 RAM 可以改变缓存压力，却不保证获得“设备级真相”。

### 2. 随机读必须有完整后备数据

原先只 `ftruncate` 到目标大小，再写少量随机 offset，会留下大量稀疏洞。独立生成的读 offset 很可能读到未写位置，产生虚高 IOPS。

修正后的生命周期是：

1. 顺序阶段完整写入文件并 fsync。
2. 顺序读完成后，将**这个已完整写入的文件**重命名为随机测试文件。
3. 随机写只覆盖指定位置，不再 `O_TRUNC`；随机读面对的是完整初始化的数据。
4. 所有阶段结束后清理本次新建的 scratch 目录。

初始化和重命名不在随机阶段计时内；同时只需一个目标大小的文件，空间预估无需因为随机文件初始化而翻倍。free space 预检仍只是粗估，不能替代用户 quota 或实际写入结果。

### 3. 随机 payload 的作用与限制

```python
payload = os.urandom(block)
```

随机块降低全零块特殊优化与块内压缩对测试的影响。但**同一个 payload 会被重复写入**，所以不能宣称排除了跨块去重。压缩/去重型 NAS 上，正式测试应使用 fio 等工具配置足够变化的写入缓冲。

### 4. 延迟测量边界

单次同步随机写的计时覆盖完整写入和 fsync：

```python
s = time.perf_counter()
# 写满这次 4KiB；短写时继续写完剩余部分。
# 然后 os.fsync(fd)。
lat.append(time.perf_counter() - s)
```

- 每次延迟不包含 open/close；整体吞吐则还包含 Python 循环等开销。
- buffered 随机写的每次 latency 主要是本次写调用，不含最后统一 fsync，脚本不把这组 latency 作为落盘延迟输出。
- sync 随机写的 p99 是写入加同步的完整操作延迟，包含文件系统、网络与持久化路径影响。
- 随机读平均延迟也包含系统调用/内核路径；读到 EOF 会报错，不会把空读算成成功 I/O。

### 5. 4KiB offset 与可复现序列

```python
RAND_BLOCK = 4 * KIB
rng = random.Random(args.seed)
n_blocks = size_bytes // RAND_BLOCK
w_off = [rng.randrange(n_blocks) * RAND_BLOCK for _ in range(args.rand_ops)]
r_off = [rng.randrange(n_blocks) * RAND_BLOCK for _ in range(args.rand_ops)]
```

payload 和读缓冲也使用 `RAND_BLOCK`。因此每次起始位置按 4KiB 对齐，操作不越过文件尾。seed 固定使目标间的 offset 序列一致，但读取可重复命中同一块；这不等于完全冷读或独立样本。

### 6. 工程健壮性

| 设计 | 作用与边界 |
|---|---|
| Linux/API 检查，参数必须为正 | 不在缺少 positional/vectored I/O 的平台上静默运行；避免空样本和非法随机范围 |
| `tempfile.mkdtemp` | 创建唯一、本次拥有的 scratch 目录，不复用已有 PID/时间戳目录 |
| `finally: rmtree` | 正常退出、异常或 KeyboardInterrupt 时尝试清理；强制终止进程或清理失败仍可能留文件 |
| 权限/只读/空间错误分类 | 捕获 EACCES、EPERM、EROFS、ENOSPC、EDQUOT，提示并继续下一个目标；其他 I/O 错误不静默吞掉 |
| `du.free == 0` 时提示后尝试 | 某些网络导出可能存在报告/配额差异，但也可能真已满；实际写入仍可能失败 |
| `df -T -P` | 尽力探测文件系统类型，失败显示 `?` |
| 短写重试、短读重试/EOF 检查 | 只把完整传输的数据算入测量，不把未完成操作当成功 |
| `flush=True` 与 `perf_counter()` | 及时显示进度，使用单调高精度计时 |

## 结果怎么读：先看现象，再验证原因

| 现象 | 可以提出的假设与下一步 |
|---|---|
| `seq_write_buf` 高而 `seq_write_sync` 低 | 逐块同步屏障昂贵，批处理/合并有效；不能仅据此认定 buffered 未落盘 |
| 随机读 IOPS 很高 | 检查 SSD、后端 cache、重复 offset 和文件大小；不能直接与 MiB/s 数值比较 |
| `fsync_lat_p99` 超过 100ms | 写入加同步路径有长尾；结合网络、服务端负载与 iostat 定位 |
| 顺序吞吐接近网络线速 | 可能受网络限制；核对 MiB/s 与 MB/s 单位、协议开销和实际流量 |
| 同一挂载点两次差异很大 | 检查并发负载、文件/服务端 cache、空间分配与后端状态，再重复测试 |

## 局限与适用边界

1. **不是 O_DIRECT/裸设备基准**。fsync 保证的语义取决于文件系统与后端的持久化承诺；DONTNEED 只是建议，不清所有层级缓存。正式测试用 fio 明确 direct、缓存、同步和缓冲内容策略。
2. **单线程、同步调用，应用侧一次一个 I/O**。不能代表 NVMe 或并行文件系统在高并发、深队列下的极限；buffered 写入后内核仍可能并行回写。
3. **`pct()` 是索引近似**。对升序样本取 $\lfloor q(n-1)\rfloor$ 的零基索引。1000 个样本的 p99 取索引 989，即**第 990 小**（第 11 大）；300 个样本取索引 296，即**第 297 小 / 第 4 大**。少量样本的尾延迟不稳定。
4. **不是小文件元数据基准**。已有文件覆盖写仍可能有分配/文件系统开销，但完全没有覆盖大量文件 create/open/close/delete 的负载。参见[并行文件系统](../../distributed-systems/distributed-storage/parallel-file-systems.md)。
5. **初始状态不完全相同**。顺序阶段包含新建/截断，随机阶段为已初始化文件覆盖写，且阶段顺序会影响缓存；不能把不同阶段的差值解释成纯粹的单一硬件因素。
6. **固定 offset 不代表完整可复现**。后台负载、缓存、压缩/去重和存储策略都可能变化，严肃报告要记录环境并重复测量。

## 与 fio 的对照

| 需求 | disk_io_bench.py | fio |
|---|---|---|
| 无第三方依赖地快速对比挂载点 | 适合 | 需安装或编译 |
| direct、高队列深度、多 job | 不覆盖 | 支持 |
| 精细延迟分布与明确分位数算法 | 仅简单索引近似 | 支持 histogram 等统计 |
| 混合读写、缓冲内容与访问模式控制 | 选项有限 | 通过 job 配置 |
| 采购报告/正式性能档案 | 只能作初步证据 | 配合明确方法与重复实验更合适 |

## 我的理解

- **测量诚信首先是边界准确**：代码是否计入 fsync，比指标叫 buffered 还是 sync 更重要。文件是否真正写满，比“清缓存”注释更重要。
- **缓存不是单层开关**：客户端 clean page、服务端缓存、控制器缓存的语义不同；“请求驱逐”不能写成“保证读盘”。
- **选型应贴近真实负载**：checkpoint 的批量写、日志的同步小写、DataLoader 的文件读取/解码不是同一种模式，脚本只提供其中几种受限近似。
- **报告异常与实际约束需分开**：free space 预检、用户 quota 与实际写入可能不一致，因此既要报告，也要正确处理失败。

## 常见误区

1. **认为 buffered 写只测 page cache** → 本脚本最终 fsync 在吞吐计时内；但批处理和后端缓存仍有影响。
2. **把 qdepth=1 与高并发 fio IOPS 直接比较** → 并发与访问模式不一致，没有直接可比性。
3. **在训练同时压测并据此定设备上限** → 结果混入排队与共享负载，先用[系统层观测](01-system-level-observability.md)确认环境。
4. **认为 quick 的 p99 足够稳定** → 300 样本对应第 4 大值，只适合初步检查，不是可靠尾延迟档案。
5. **以为清 page cache 就排除了所有缓存** → 后端 cache 仍在，重复 offset 也会重新变热。
6. **拿单文件随机读代替海量小文件基准** → metadata 与应用处理路径没有被覆盖。

## Related

- [系统层观测：htop、nvidia-smi 与 IO / 网络](01-system-level-observability.md) — 测试前确认负载
- [并行文件系统](../../distributed-systems/distributed-storage/parallel-file-systems.md) — 条带化与 metadata 瓶颈
- [元数据与性能瓶颈](../../distributed-systems/distributed-storage/metadata-and-performance.md) — 小文件场景
- [面向 AI 训练与多云场景的存储工程地图](../../distributed-systems/distributed-storage/ai-training-and-multicloud-storage-map.md) — 数据管线选型
- [Git on NFS](../git-on-nfs.md) — 存储语义与预期差异
- [PyTorch 显存 profiling](03-pytorch-memory-profiling.md) — 控制变量与警惕缓存

## References

- 脚本：`tools/disk_io_bench.py`；回归测试：`tools/test_disk_io_bench.py`（运行命令：`python -I tools/test_disk_io_bench.py`）
- Linux man 页：`posix_fadvise(2)`、`fsync(2)`、`open(2)`、`readv(2)`
- [fio](https://github.com/axboe/fio)；[PyTorch 分布式训练性能分析](07-pytorch-distributed-profiling.md)中的硬件基准思想
