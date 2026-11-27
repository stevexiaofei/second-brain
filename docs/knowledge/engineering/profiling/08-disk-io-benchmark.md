---
title: 磁盘 IO 压测：disk_io_bench.py 解读与存储基准方法论
subtitle: 观测告诉你“谁在用磁盘”，压测提供受控负载下的能力对比
type: guide
status: seed
tags: [engineering, profiling, disk, io, benchmark, fsync, page-cache, nfs, storage]
created: 2026-11-04
updated: 2026-11-25
source: tools/disk_io_bench.py 程序解读 + 个人整理（Linux man 页：fadvise/fsync/open）
---

<script setup>
import DiskIoReadWriteDemo from '../../../.vitepress/components/DiskIoReadWriteDemo.vue'
</script>

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

## 一次文件读写到底经过哪些活动？

先建立一个比“调用了 `read()` / `write()`”更完整的分层模型。下面的路径描述的是**普通 buffered 文件 I/O**；它不是异步 I/O，也不是 `O_DIRECT` 绕过页缓存的路径。

```text
应用：编码 / 序列化，准备或消费用户缓冲区
  │
  ├── 内置 open() 的文件对象：可能还有 Python 用户态缓冲
  │       └── flush()：把这层缓冲向下提交，不代表持久化
  │
  └── 本脚本：os.open() + os.write/readv/pwrite/preadv()
          │ 无 Python 文件对象缓冲，但仍有内核 page cache
          ▼
     系统调用：用户态 → 内核态
          ▼
     fd 表 → 打开文件对象（访问模式、当前偏移、关联文件）
          ▼
     VFS + 具体文件系统 / 网络文件系统客户端
          │
          ├── 读：查 page cache
          │       ├── 命中：复制到用户缓冲区 → 返回
          │       └── 未命中：请求下层数据，填充缓存，再复制 / 返回
          │
          └── 写：复制到 page cache，标记 dirty
                  ├── 通常可先返回，随后后台回写
                  └── fsync：主动推进并等待文件的持久化边界
          │
          ▼ 只有需要访问后端的路径才继续往下走
     本地：文件块映射 → 块层 / 请求队列 → 驱动 → 设备
     网络：客户端协议 → 网络 → 服务端文件系统 / 对象层 → 后端设备
          ▼
     服务端 / 控制器 / 设备缓存，以及最终持久化介质
```

这是**职责分层图，不是 Linux 各版本通用的内核函数调用栈**。VFS（Virtual File System，虚拟文件系统）提供统一接口，具体文件系统实现读写、块映射和一致性策略；page cache 与这些实现协作，不是所有文件系统都严格按图中盒子逐层串行执行。

还要先区分三种“完成”：

| 完成层次 | 应用可以认为发生了什么？ | 不能据此推出什么？ |
|---|---|---|
| 用户态 `file.write()` 返回 | 文件对象接受了数据，可能仍在 Python 缓冲区 | 不能认定已调用内核 `write` |
| 普通 `os.write()` / `os.pwrite()` 成功返回 | 返回字节数对应的数据已被内核接受 | 不能认定已经持久化；也不保证一次写满全部请求 |
| `os.fsync()` 成功返回 | 文件系统报告该文件的数据和相关元数据已满足其同步持久化承诺 | 不能认定所有缓存被清空、父目录项也已同步，或不可靠硬件也能兑现承诺 |

**buffered / direct** 讨论是否经过内核页缓存；**普通写 / 同步持久化写** 讨论何时等待稳定存储；**同步调用 / 异步提交** 讨论调用返回与操作完成如何组织。这是不同维度：脚本的 `sync` 阶段仍是 buffered I/O；普通 buffered `write()` 也可能因为回写限速等原因阻塞。

### 先看 `open()`：路径如何变成 fd？

前置概念见 [inode：文件身份、元数据与数据定位](../../distributed-systems/distributed-storage/inode-file-metadata.md)。其中区分了磁盘目录记录、内核 dentry、inode、打开文件对象和 fd，也解释了硬链接、rename 与删除后空间未释放的原因。

例如 `os.open('/mnt/data/sample.bin', os.O_RDONLY)` 大致经历：

1. **解析路径与挂载点**：从根目录或当前目录出发查找路径分量；`/mnt/data` 对应的挂载决定请求交给 ext4、XFS、NFS、CephFS 等哪种文件系统，而不是由文件后缀决定。
2. **查找文件元数据**：可能命中 dentry（目录项缓存）和 inode（文件元数据对象）缓存，也可能需要读取本地目录/元数据，或联系网络文件系统服务端做查找、权限和一致性检查。元数据缓存与文件内容的 page cache 不是同一件事。
3. **执行打开语义**：校验权限和访问模式；`O_CREAT` 允许创建不存在的文件，`O_TRUNC` 会把已存在文件截断为零长度。它们不等于把目标大小的数据预先写好。
4. **建立打开文件对象并返回 fd**：fd 是当前进程 fd 表中的一个整数索引，不是磁盘地址。打开文件对象记录访问模式、当前文件偏移等，并关联具体文件和挂载。

此后 `read/write` 主要经 fd 找文件，不需要每次从 `/` 重新解析完整路径；网络文件系统仍可能做必要的一致性检查。两次独立 `open()` 通常有不同的当前偏移，但访问同一文件时仍可能共享文件内容页缓存；`dup()` 等方式则可以共享同一个打开文件对象和偏移。

#### 动画一：路径、inode、打开对象与 fd

先完整走一遍，再切换“再次独立 open / dup 共享打开对象”，跳到最后比较 fd 3 和 fd 4 的偏移。**同一 inode 不代表共享当前偏移；共享偏移主要取决于是否指向同一打开文件对象。** 每个角色盒子都可点击查看职责。

<DiskIoReadWriteDemo mode="open" />

### 一次读：不是只有“从磁盘取数据”

以 `os.readv(fd, [buf])` 为例，一次完整读操作可能包含这些活动：

1. **应用准备缓冲区**：Python 创建或复用 `bytearray`，并把它包装成 `memoryview`；这一步属于用户态内存准备，不是磁盘 I/O。
2. **进入内核**：Python 运行时通过系统调用进入内核，内核根据 `fd` 找到打开文件对象和当前文件偏移。
3. **权限与范围检查**：内核检查该 fd 是否可读、偏移是否合法、请求长度是否越过文件尾。
4. **查文件页缓存**：普通 buffered read 先查该文件对应的 page cache。
5. **缓存命中路径**：若所需页已在客户端内核缓存中，内核直接把数据复制到 `buf`，没有必要访问后端存储。
6. **缓存未命中路径**：文件系统把请求转换成块设备请求，或转换成 NFS / CephFS 等网络文件系统请求；请求可能在服务端、控制器或设备缓存中再次命中。
7. **等待数据可用并复制回用户态**：系统调用通常在所需数据可读后返回字节数；`readv` 允许一次把数据填入多个用户缓冲区，但**不会自动变成异步 I/O、DMA 或绕过 page cache**。
8. **推进文件偏移**：`readv` 使用并推进 fd 的当前文件偏移；`preadv` 则使用显式 `offset`，不会改变 fd 的共享文件偏移，更适合多个随机位置操作。
9. **应用解释数据**：实际业务还可能在系统调用返回后解码、反序列化、校验数据、构造张量；脚本不做这些应用处理，因此结果不包含它们的成本。

本地块设备读取时，驱动通常组织 DMA（Direct Memory Access，直接内存访问）把设备数据传到内核管理的内存，完成事件再推动等待线程继续运行；普通 buffered read 之后仍要把文件数据复制到用户缓冲区。**设备与内存间的 DMA，不等于应用与 page cache 之间没有复制**。具体完成机制可能使用中断或轮询，网络路径也有不同实现，不能把某一种驱动路径当成所有文件读取的固定步骤。

顺序访问还可能触发 **readahead（预读）**：内核提前读取尚未被应用请求的相邻范围。所以即使读循环开始前成功驱逐了旧缓存，循环中后面的某次 `readv` 也可能命中本轮预读刚填充的页。

`08-disk-io-benchmark.md` 中的 `seq_read` 采用 `os.readv(fd, [buf])`，每次请求 1 MiB，直到返回 `0` 表示 EOF。它的计时从第一次读调用前开始，到最后一次读返回 EOF 后结束，因此包含系统调用、内核路径、等待后端数据和用户态数据复制，但不包含 `open()` 和 `close()`，也不包含后续的数据解析。

#### 动画二：缓存命中、未命中与顺序预读

依次对照三个读取分支。缓存命中时，应看见**没有后端数据读**；未命中时，先取到内核缓存，再复制到用户缓冲区，而不是 DMA 直接填入这个普通 buffered I/O 的应用 buf。切换 `readv / preadv` 比较目标范围与最终偏移；再把后端切换为网络文件系统，观察“客户端未命中”为什么仍不保证物理读盘。

示例文件内容按四个 4 KiB 范围标成 A/B/C/D。预读分支固定使用连续的 `readv`，假设读 B 时顺带取 C；这个窗口是示意值，不是 Linux 实际预读策略。

<DiskIoReadWriteDemo mode="read" />

### 一次写：`write()` 返回不等于数据已经落盘

以 `os.write(fd, view)` 为例，一次普通 buffered 写可能包含：

1. **应用准备待写数据**：脚本用 `os.urandom()` 生成 payload，并用 `memoryview` 表示当前尚未写完的切片。
2. **进入内核并解析 fd**：内核检查 fd、当前文件偏移、访问模式和请求范围。
3. **更新 page cache**：数据通常先复制到内核页缓存，并将对应页标记为 dirty；必要时分配页、扩展文件大小、更新 inode 元数据。
4. **返回短写或完整写**：`write()` 返回实际接受的字节数。返回成功通常表示内核接受了数据，不是“数据已经到达物理介质”。脚本显式循环处理短写，直到当前块完整写入。
5. **异步回写或主动同步**：后台回写线程可能稍后提交 dirty 页；调用 `fsync(fd)` 会让应用等待内核推进该 fd 相关的文件数据和必要元数据。具体“稳定”边界仍取决于文件系统、网络文件系统和后端存储的承诺。
6. **设备或服务端完成请求**：本地路径可能经过块层和设备缓存；网络文件系统还会经过客户端协议、服务端文件系统及服务端存储。`fsync()` 不等于清空所有缓存层，也不自动把目录项变更作为独立对象同步。

“可以先返回”不等于“总是立即返回”：缺页和内存分配、文件锁、空间分配、文件系统事务、脏页过多导致的回写限速，以及网络文件系统处理等，都可能让 `write()` 本身阻塞。覆盖一个尚未缓存页的局部范围时，文件系统还可能需要先读入原数据以保留未被覆盖部分；并不是每次写都只有向下的数据流。

因此，`seq_write_buf` 的真正含义是：**一批 1 MiB buffered 写入，加一次最终 `fsync` 的端到端时间**；它不是单纯的 page cache 接收速度。`seq_write_sync` 则是每个 1 MiB 写入后同步一次，并在循环结束再次 `fsync`。

#### 动画三：write 返回、dirty 页、回写与持久化

用两块旧内容 A0/A1 和新内容 B0/B1，对照“写完两块再同步”与“每块写完同步”。在 `pwrite` 已返回的阶段暂停：**应用可以继续、新内容已进缓存、但应用仍未获得新版本的同步持久化保证**，这三个状态可以同时成立。

再选择“脏页压力使 write 阻塞”，看第二次调用为什么可能在返回前等待后台回写。图中的“上次同步记录”只是历史，不宣称真实设备必然等到 `fsync` 才开始写入；**原地覆盖后、再次同步前，崩溃恢复不保证完整旧版本或完整新版本**。

<DiskIoReadWriteDemo mode="write" />

### `fsync()` 在这里同步了什么？

在本脚本语境中，`fsync(fd)` 是一个**持久化边界请求**，不是“把 Python 缓冲区 flush 一下”：

- 脚本使用的是 `os.open()` 和 `os.write()`，没有 Python 文件对象，因此不存在需要调用 `file.flush()` 的 Python 用户态文本/缓冲层。
- `fsync()` 主要针对该 fd 对应文件的数据，以及文件系统为保持文件状态所需的相关元数据。
- 它不负责把父目录的目录项更新单独同步到稳定存储；如果场景是“创建后必须保证文件名在断电后存在”，通常还要考虑对父目录 fd 做同步。
- 在 NFS、CephFS 等网络文件系统上，最终语义还受客户端、协议、服务端文件系统、服务端设备和挂载选项影响；不能把本地 ext4 上的直觉无条件套过去。

**同步与驱逐是两步不同的事**：`fsync` 后，没有被并发修改的已同步页通常变为 clean，但仍可驻留内存、被后续读取命中；`DONTNEED` 才是建议回收这些页。单纯驱逐又不能替代 dirty 数据的持久化。因此脚本先完整写入并 `fsync`，读前再发驱逐建议，而不是任选其中一个 API。

若业务使用的是 Python 文件对象，常见的同步顺序则是：

```python
with open(path, 'wb') as f:
    f.write(data)          # 可能仍有数据留在 Python 用户态缓冲
    f.flush()             # 先把该缓冲向操作系统提交
    os.fsync(f.fileno())  # 再同步内核已接收的文件数据和相关元数据
```

这是文件内容同步示例，不是包含临时文件、原子替换和目录同步的完整崩溃安全更新协议。正常 `with` 退出会关闭文件对象并处理其缓冲，但不能据此认为自动执行了 `fsync`。

#### 动画四：flush、fsync、DONTNEED 与 close

先观察 Python 文件对象缓冲 → `flush` → dirty page → `fsync` 成功 → clean 但仍驻留的顺序。随后切换驱逐成功/仍驻留两个分支，它们都是建议调用之后的合法结果；再看“未同步就建议驱逐”和“只 flush + close”，不要把回收建议或关闭句柄当作持久化屏障。

这个动画同时对照文件对象路径和 `os.write` 路径；脚本自身没有 Python 文件对象的那一层缓冲。

<DiskIoReadWriteDemo mode="sync" />

### 这个脚本中一次 I/O 的调用链

以各 phase 的实际代码为准：

| 阶段 | Python/API 层 | 每次操作的系统调用语义 | 计时范围 | 关键结论 |
|---|---|---|---|---|
| `seq_write_buf` | `os.open` → 多次 `os.write` → `os.fsync` → `os.close` | 顺序写入 1 MiB 块，最后同步一次 | 不含 `open` / payload 生成；包含循环、最终 `fsync` 和 `close` | 批量写 + 一次持久化边界 |
| `seq_write_sync` | `os.open` → 多次（写满一块 + `os.fsync`）→ 末尾 `os.fsync` → `os.close` | 每个 1 MiB 块都建立同步边界 | 同上；包含每次 `fsync`，以及循环末尾额外的一次 | 高频同步屏障的代价 |
| `seq_read` | `os.open` → `os.posix_fadvise` → 多次 `os.readv` → `os.close` | 顺序读，复用单个 1 MiB 用户缓冲区 | 从读循环前到 EOF `readv` 返回 0；不含 `open`、驱逐建议、缓冲分配和 `close` | `DONTNEED` 不保证物理读 |
| `rand_write_4k_buf` | `os.open` → 多次 `os.pwrite` → 最后 `os.fsync` → `os.close` | 4 KiB、显式 offset 的随机覆盖写 | 总时间包含循环、最后 `fsync` 和 `close`；不含 `open` | 应用逐次调用，但内核可并发回写 |
| `rand_read_4k` | `os.open` → `os.posix_fadvise` → 多次 `os.preadv` → `os.close` | 4 KiB、显式 offset 的随机读 | 每次 latency 包含该次读满循环；总时间含整个读循环，不含 `open`、驱逐建议、缓冲分配和 `close` | 不推进共享 fd offset；缓存可重新变热 |
| `rand_write_4k_sync` | `os.open` → 每次写满 4 KiB → 每次 `os.fsync` → `os.close` | 每个随机覆盖写后同步 | 每个 latency 包含完整写入和 `fsync`，不含 `close`；总时间还含循环和 `close`，不含 `open` | p99 是“写 + 同步”，不是纯 `fsync` 延迟 |

其中 `os.pwrite()` / `os.preadv()` 的显式 offset 很重要：这些操作不依赖也不推进打开文件对象的当前偏移，但**不会因此防止多个写入覆盖同一范围**。名字里的 `p` 表示 positional（指定位置），`v` 表示 vectored（一次提供多个用户缓冲区）；它们不是“物理 I/O”和“设备并行”的标记。

脚本下一次调用要等上一次返回，所以**应用一次只有一个未返回的读写调用**；但 buffered 写返回后内核仍能并行回写，顺序读也可能预读。因此不能据此断言块设备实际队列深度始终为 1。

同样，一次应用 4 KiB I/O 不等于一次物理设备操作：缓存命中可能没有后端读；文件系统可能拆分、合并请求；日志、元数据、RAID 和副本也可能带来额外 I/O。脚本输出的是**应用逻辑 IOPS**，不是直接读取设备计数器得到的物理 IOPS。

#### 动画五：六个 phase 的计时到底包住什么？

选择 phase 后，逐步观察 API/伪代码与两个计时状态：**总吞吐计时**和随机阶段的**单次 latency**。特别检查下面三个边界：

1. buffered 写的最终 `fsync` 仍在总计时内，随机 buffered 写的单次 latency 不含最终统一同步。
2. `seq_write_sync` 每块同步之外，循环末尾还有一次 `fsync`；`rand_write_4k_sync` 没有对应的末尾额外同步。
3. 写函数先执行 `finally: close` 再计算总耗时；读函数先在 `return` 中捕获耗时再执行 `finally`。所以总耗时对 `close` 的包含关系不同。

示意循环只展开一个代表性操作，不按虚构耗时画比例条，也不把一次应用 I/O 等同于一次物理设备操作。

<DiskIoReadWriteDemo mode="benchmark" />

### API 与“活动”的对应关系

| API | 它负责的活动 | 它不保证 / 不负责的事情 |
|---|---|---|
| `os.open()` | 建立 fd，解析路径并获得文件对象；`O_CREAT` / `O_TRUNC` 影响创建和截断 | 不读取全部文件；不代表后续写入已持久化 |
| `os.write()` | 按 fd 当前偏移写入一段字节，可能产生短写 | 不保证物理落盘；不提供随机 offset |
| `os.readv()` | 按 fd 当前偏移，把数据填入一个或多个缓冲区 | 不保证绕过 page cache；不改变为异步读；脚本只传一个 buffer |
| `os.pwrite()` | 在指定 offset 写入，不改变 fd 当前偏移 | 不保证持久化；短写仍需循环处理 |
| `os.preadv()` | 从指定 offset 读入多个缓冲区，不改变 fd 当前偏移 | 不保证 cache miss 或物理设备访问；脚本没有使用额外 flags |
| `os.posix_fadvise(..., DONTNEED)` | 向内核建议：这些文件页近期不需要，尽力回收 clean page | 不清理服务端/控制器缓存；不等于 `O_DIRECT`；不是强制清缓存 |
| `os.fsync()` | 请求把文件数据和必要文件状态推进到文件系统承诺的稳定边界 | 不等于清空所有缓存；不单独保证父目录项持久化；网络后端语义可能不同 |
| `os.close()` | 释放 fd，可能触发文件系统收尾和资源回收 | 不是通用持久化屏障；写阶段总计时包含它，读阶段总计时不包含它，单次 latency 都不含它 |
| `os.rename()` | 在同一文件系统内更新目录项，把已写完的 `seq.bin` 改名为 `rand.bin` | 不是随机阶段 I/O；脚本不把它计入随机 phase；rename 的崩溃持久化要求仍需目录同步语义 |
| `os.readv` / `os.preadv` | “vectored” 只表示一次调用可提供多个缓冲区 | 不表示向量化磁盘请求、并行提交、DMA 或高队列深度 |

### 从应用到物理介质：哪些活动被本笔记测到了？

这篇笔记实际覆盖的是：

- Python 调用系统调用的开销；
- 内核 fd、VFS、文件系统和 page cache 路径；
- 对本地或网络挂载点提交请求后的等待时间；
- `fsync()` 所触发并等待的同步路径；
- 单线程、同步调用下的批量顺序和 4 KiB 随机模式；
- 在可观察范围内由 `time.perf_counter()` 看到的端到端耗时。

它没有单独分离出：

- `open()`、`rename()`、目录创建和删除的耗时；`close()` 没有单列指标，且写阶段总耗时包含它，读阶段总耗时不包含它；
- Python payload 生成、随机 offset 生成和 scratch 初始化耗时；
- 用户态解码、压缩、反序列化和 DataLoader worker 调度；
- 设备内部 NAND/磁盘介质动作、控制器缓存命中与后台 GC；
- 网络文件系统服务端 CPU、网络 RPC、服务端 page cache、OSD/RAID 内部各阶段；
- 并发 I/O、队列深度、多个 worker/rank 的争用；
- 小文件 create/open/close/delete 的元数据压力。

所以读结果时要使用“**这个调用边界下，应用等待了多久、完成了多少逻辑字节/操作**”的表述，不要直接把它写成“物理盘读写速度”或“裸设备极限”。

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
