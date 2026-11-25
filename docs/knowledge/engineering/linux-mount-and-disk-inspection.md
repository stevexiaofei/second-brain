---
title: Linux 挂载与磁盘查看指南
type: guide
status: growing
tags: [Linux, Filesystem, Mount, NFS, Storage, Disk]
created: 2026-11-25
updated: 2026-11-25
---

# Linux 挂载与磁盘查看指南

## 一句话理解

Linux 挂载不是复制数据，而是把一个文件系统接入本机目录树，让用户和程序可以通过某个目录访问它。排查存储问题时，需要区分：**当前实际挂载关系、文件系统容量、本地块设备结构，以及开机持久化配置**。

## 什么是挂载？

挂载（`mount`）是将一个文件系统接入 Linux 目录树，使用户和程序可以通过某个目录访问其中的文件。

```text
本机目录树                         文件系统
/data  ───────── 挂载 ──────────>  /
                                  ├── image.jpg
                                  └── dataset/
                                      └── labels.json
```

挂载后，可以通过 `/data/image.jpg` 访问文件。挂载不等于复制数据：`/data` 是访问文件系统的入口。

- **挂载源（source）**：文件系统的来源，例如本地分区 `/dev/nvme1n1p1`，或 NFS 服务端共享目录。
- **挂载点（mount point）**：文件系统接入本机目录树的位置，例如 `/data`。
- **文件系统类型（filesystem type）**：组织和访问文件的机制，例如本地的 `ext4`、`xfs`，以及网络文件系统 `nfs`。

程序使用普通路径访问文件；内核根据挂载关系，将文件操作交给对应的文件系统处理。

## NFS 示例：远程目录如何出现在本机

在 `training109` 上查询到的关系如下（实际状态可能变化）：

```text
本机目录树                                 NFS 服务端
/f900  ───────── NFS 挂载 ─────────────>  powerscale.bst.loc:/ifs/f900/
  ├── Bev_annotations/                      ├── Bev_annotations/
  └── ...                                   └── ...
```

访问 `/f900/Bev_annotations/xxx` 时，内核通过 NFS 请求远端文件；客户端可能缓存读取的数据，但这不意味着文件已完整复制到本地硬盘。

同一次查询还发现 `/f900/APS` 是独立挂载点，源为 `powerscale.bst.loc:/ifs/f900/APS`。访问其下路径时使用该挂载；`/f900/Bev_annotations` 则不是独立挂载点，而是通过 `/f900` 访问。

> 挂载源相同或相近，并不能直接说明底层物理磁盘如何组织。要了解后端设备，还需要在存储服务端或存储系统管理面继续排查。

### 在非空目录上挂载

如果挂载前 `/data` 里已有 `old.txt`，挂载后，访问 `/data` 看到的是新文件系统的内容，原文件会暂时被遮住，并非自动删除。卸载后原文件通常重新可见。

因此，在非空目录上挂载需谨慎：优先选择专用且为空的挂载点，避免误以为原有文件已经被删除，或把本地文件与远端文件混在一起。

卸载（`umount`）是断开文件系统与挂载点的连接，不等于删除其中的数据。但卸载会影响正在使用它的程序，不要随意卸载共享训练盘。

## 常用查看命令

```bash
findmnt             # 查看挂载关系、来源和参数
findmnt -T /path    # 查看包含该路径的文件系统
mountpoint /path    # 判断路径本身是否是挂载点
df -hT /path        # 查看所属文件系统的容量和类型
lsblk -f            # 查看本地块设备、分区和文件系统
findmnt --fstab     # 查看 /etc/fstab 中的挂载配置
```

可以先记住它们的分工：

- `findmnt` 看当前挂载关系。
- `df` 看文件系统报告的空间。
- `lsblk` 看本地块设备结构。
- `mountpoint` 判断某个路径本身是不是挂载点。
- `findmnt --fstab` 看 `/etc/fstab` 中声明的配置。

NFS 是网络文件系统，一般不会出现在 `lsblk` 的本地块设备列表中。

## `findmnt`：查询路径所属的挂载

```bash
findmnt -T /f900/Bev_annotations -o TARGET,SOURCE,FSTYPE,OPTIONS
```

示例来自 `training109`，并经过简化：

```text
TARGET  SOURCE                        FSTYPE  OPTIONS
/f900   powerscale.bst.loc:/ifs/f900/ nfs     rw,...,vers=3,...,hard,proto=tcp,...
```

| 列 | 含义 |
|---|---|
| `TARGET` | 所属文件系统的挂载点，此处为 `/f900` |
| `SOURCE` | 挂载来源，此处为 NFS 服务端和导出路径 |
| `FSTYPE` | 文件系统类型，此处为 `nfs` |
| `OPTIONS` | 当前生效的挂载选项 |

`-T` 查的是**包含该路径的文件系统**，不代表传入的路径本身就是挂载点。

常见选项：

- `rw` / `ro`：读写 / 只读挂载。`rw` 不代表所有用户都有写权限，还需检查目录权限及服务端设置。
- `vers=3`：使用 NFSv3。
- `proto=tcp`：NFS 数据传输使用 TCP。
- `hard`：请求失败时持续重试；网络或服务端不可用时，进程可能长时间等待。
- `relatime`：按相对时间规则更新文件访问时间，以减少部分元数据写入。

## `mountpoint`：判断路径本身是否是挂载点

```bash
mountpoint /f900/Bev_annotations
mountpoint /f900/APS
```

在上述机器的查询中，前者显示 `is not a mountpoint`，后者显示 `is a mountpoint`。

如果需要按路径前缀查看当前挂载，可以筛选挂载表：

```bash
findmnt -rn -o TARGET,SOURCE,FSTYPE | grep '^/f900/'
```

`findmnt -R /f900` 是按内核挂载树查找后代，结果不一定等同于路径以 `/f900/` 开头的所有挂载。需要按路径排查时，可使用上面的全表筛选，并注意路径边界。

## `df`：查看文件系统容量

```bash
df -hT /f900/Bev_annotations
```

示例输出（容量会随时间变化）：

```text
Filesystem                    Type  Size  Used  Avail  Use%  Mounted on
powerscale.bst.loc:/ifs/f900/ nfs   5.2P  4.0P  1.3P   77%  /f900
```

| 列 | 含义 |
|---|---|
| `Filesystem` | 文件系统来源 |
| `Type` | 文件系统类型 |
| `Size` | 文件系统报告的总容量 |
| `Used` | 已使用量 |
| `Avail` | 当前报告的可用空间 |
| `Use%` | 使用率 |
| `Mounted on` | 挂载点 |

这里显示的是**整个文件系统报告的容量**，不是指定目录本身的大小。共享存储的用户或目录配额还可能限制实际可写额度。

查看目录占用可用：

```bash
du -sh /path
```

`du` 需要遍历目录，大目录在 NFS 上可能耗时很长并增加共享存储负载；统计结果也不等同于 `df` 的文件系统容量：

- `df` 关注文件系统层面还剩多少空间。
- `du` 关注目录树中被文件占用了多少空间。
- 已删除但仍被进程打开的文件，可能导致 `df` 与 `du` 的结果不一致。
- 配额、快照和稀疏文件等机制，也可能让用户看到的数字与直觉不同。

## `lsblk`：查看本地块设备

```bash
lsblk -f
```

示例（并非当前机器实际输出）：

```text
NAME        FSTYPE  LABEL  UUID                                  MOUNTPOINTS
nvme0n1
├─nvme0n1p1 vfat           XXXX-XXXX                             /boot/efi
└─nvme0n1p2 xfs            xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx  /
nvme1n1     ext4           xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx  /data
```

- `nvme0n1`、`nvme1n1` 是块设备。
- `p1`、`p2` 是分区。
- `FSTYPE` 是文件系统类型。
- `UUID` 是文件系统标识符。
- `MOUNTPOINTS` 是当前挂载位置。

挂载点为空不一定意味着磁盘不可用：它可能是还没有分区的磁盘、包含尚未挂载的分区，或只是当前没有接入目录树。不要在未确认数据状态前执行格式化、分区或 `mkfs`。

## 当前状态与持久化配置

```bash
findmnt -T /f900    # 当前实际挂载
findmnt --fstab     # 当前 /etc/fstab 中的配置
less /etc/fstab     # 直接查看配置文件
```

`/etc/fstab` 的典型格式是：

```text
来源  挂载点  文件系统类型  挂载选项  dump  fsck顺序
```

例如 `training109` 查询时有：

```fstab
powerscale.bst.loc:/ifs/f900/  /f900  nfs  defaults  0 0
```

配置不等于当前状态：手动挂载、开机后修改配置或 systemd 尚未重新加载等，都可能导致二者不同。

排查时建议先确认实际状态，再检查配置来源：

1. 用 `findmnt -T /path` 确认当前路径实际使用的文件系统。
2. 用 `mountpoint /path` 判断路径本身是否是挂载点。
3. 用 `df -hT /path` 确认容量、类型和挂载点。
4. 只有需要解释开机行为或自动挂载时，再检查 `/etc/fstab` 和 systemd 状态。

## 一套安全的排查顺序

当程序报告“磁盘满了”“路径不存在”或“写入很慢”时，可以按下面顺序进行：

```bash
# 1. 先确认路径到底落在哪个文件系统
findmnt -T /path -o TARGET,SOURCE,FSTYPE,OPTIONS

# 2. 确认路径自身是否是挂载点
mountpoint /path

# 3. 查看文件系统容量
df -hT /path

# 4. 如果是本地存储，再看块设备和分区
lsblk -f

# 5. 需要统计目录大小时，再谨慎使用 du
du -sh /path
```

在共享 NFS 上，优先确认挂载源和文件系统类型，再决定是否遍历大目录。不要一遇到空间问题就递归执行 `du`，也不要把 NFS 路径当作本地块设备处理。

## 常见误区

### 1. “挂载就是把远端数据下载到本地”

不是。挂载建立的是访问关系；读取时可能经过网络和多级缓存。只有显式执行复制、缓存或同步操作，才是在把数据复制到本地。

### 2. “`rw` 就代表当前用户一定能写”

不是。`rw` 只是挂载层允许读写，还要检查本地权限、ACL、NFS 服务端导出规则、UID/GID 映射、配额和只读快照等条件。

### 3. “`df` 显示的空间就是这个目录能用的空间”

不一定。`df` 是文件系统层面的报告；共享存储还可能存在用户、组或目录配额。

### 4. “路径不是挂载点，所以它没有使用挂载”

不一定。`findmnt -T /path` 查询的是包含该路径的文件系统。一个普通子目录可以通过祖先目录的挂载访问远端文件。

### 5. “`lsblk` 看不到，就说明存储不存在”

不一定。`lsblk` 主要展示本地块设备；NFS 等网络文件系统通常应使用 `findmnt` 和 `df` 查看。

### 6. “`/etc/fstab` 里有配置，所以当前一定已经挂载”

不一定。`fstab` 是持久化配置，不是当前内核状态。当前状态应以 `findmnt`、`mountpoint` 等结果为准。

## 我的理解

> `mount` 解决“程序用哪个路径访问”；导出和权限解决“谁能访问”；网络解决“请求怎么到达”；后端存储决定“字节实际保存在哪里”。这四层不能混为一谈。

这份笔记主要沉淀命令语义和排查方法；其中 `training109`、`powerscale.bst.loc` 和容量数字属于查询时的示例，实际环境会变化，不能作为当前基础设施状态的保证。

## 相关知识

- [NFS：部署、挂载与计算节点连接](../distributed-systems/distributed-storage/nfs-deployment-and-mount.md) — 从服务端导出、客户端挂载到训练节点访问的完整链路
- [CephFS：并行访问、部署、挂载与网络](../distributed-systems/distributed-storage/cephfs-deployment-and-mount.md) — 分布式文件系统的部署与挂载
- [并行文件系统](../distributed-systems/distributed-storage/parallel-file-systems.md) — 元数据、条带化和并发访问
- [磁盘 I/O Benchmark](./profiling/08-disk-io-benchmark.md) — 区分客户端缓存、服务端缓存和实际设备 I/O
- [Git 在 NFS 上的性能问题](./git-on-nfs.md) — 元数据密集型负载为什么对网络往返敏感
- [PV、PVC、StorageClass 与 CSI](./kubernetes/08-storage-pv-pvc-and-csi.md) — 宿主机能访问，不等于训练 Pod 已挂载
