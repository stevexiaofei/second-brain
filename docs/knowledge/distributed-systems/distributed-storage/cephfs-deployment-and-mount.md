---
title: CephFS：并行访问、部署、挂载与网络
type: concept
status: growing
tags: [CephFS, Ceph, Parallel File System, RADOS, MDS, OSD, Network, Mount]
created: 2026-11-20
updated: 2026-11-20
---

<script setup>
import NetworkStorageDemo from '../../../.vitepress/components/NetworkStorageDemo.vue'
</script>

# CephFS：并行访问、部署、挂载与网络

## 一句话理解

**CephFS 是构建在 Ceph RADOS 之上的分布式文件系统，支持多个客户端并行访问；客户端通过 MON 获得集群视图，通过 MDS 协调文件系统元数据，再直接向 OSD 读写文件内容。**

如果还不清楚“导出”和“挂载”的区别，先读 [NFS 部署与挂载](./nfs-deployment-and-mount.md)。本文在该心智模型上解释：为什么共享文件系统不一定需要所有数据都经过同一台文件服务器。

想通过实际执行的 C++ 代码观察文件切对象、主副本复制与故障修复，可读 [Mini CephFS 教学工程](./cephfs-cpp-demo.md)。它是单进程内存模拟器，不是本文部署的真实 CephFS，也不兼容 Ceph 协议。

::: warning 事实、示意和验证范围
概念及命令格式参考 Ceph 官方 Squid 分支文档；其他版本应重新核对。地址、机器数量、对象分布和阶段时序是教学示意，没有在真实 Ceph 集群执行部署、故障注入或吞吐测试。以下不是复制即得的生产部署脚本，也不包含数据盘初始化的安全审批和备份流程。
:::

## 1. 是两种“并行”吗？先拆开三个维度

### 1.1 分布式和并行：同一系统的不同属性

| 概念 | 回答的问题 | 在 CephFS 中的表现 |
|---|---|---|
| 分布式 | 数据和服务部署在哪里？ | 多台机器共同保存数据、提供服务和容错 |
| 存储并行 | 哪些 I/O 能同时工作？ | 多个客户端、请求、对象可以使用不同 OSD 和磁盘 |
| 计算并行 | 谁同时计算模型？ | 多 CPU/GPU 运行数据并行、张量并行、流水线并行等 |

**CephFS 是具体系统，“并行文件系统”是类别；分布式与并行不是互斥分类，也不是两套需要分别挂载的文件系统。**

存储并行给多个训练进程供给数据，计算并行负责模型计算，两者可以配合，但挂载 CephFS 不会自动启用分布式训练。

### 1.2 四种“多”不要混淆

| 机制 | 同时处理什么 | 不代表什么 |
|---|---|---|
| 数据分布、条带化 | 不同对象或文件部分分散到不同 OSD | 每个小文件或每次 `read()` 都访问多台机器 |
| 多客户端并发 | 多个计算节点各自发起 I/O | 一个客户端一定跑满整个集群 |
| 多 active MDS | 不同元数据工作负载可分摊 | 部署两个 MDS 就一定两个都 active |
| 多副本 | 同一个对象保存多份，提供容错 | 一次普通读取自动汇聚所有副本带宽 |

**一个 active MDS 加一个 standby MDS 仍然可以有并行的数据访问；数据并行并不要求多个 active MDS。**

## 2. NFS 与 CephFS：协议和系统不要直接画等号

常见单服务器 NFS：

```text
应用 → NFS 客户端 → NFS 服务器 → 后端文件系统 / 存储
```

CephFS 原生客户端：

```text
应用 → CephFS 客户端
          ├── MON：认证、集群映射等
          ├── MDS：路径、权限、文件布局、缓存协调等
          └── OSD：直接读写 RADOS 对象
```

也可以组合：

```text
NFS 客户端 → NFS-Ganesha 网关 → CephFS → RADOS
```

最后一种场景中，计算节点挂载的是 NFS 接口，不是原生 CephFS；网关会改变数据路径和性能边界。NFS 后端不一定是单机，NFSv4.1 也有 pNFS 机制，因此不能把“NFS”一概定义为“非并行、一次只做一个 I/O”。

## 3. 四个核心角色及其持久化关系

| 角色 | 全称 | 做什么 | 不做什么 |
|---|---|---|---|
| MON | Monitor | 维护关键集群状态、映射和仲裁，参与认证 | 不作为普通文件内容的统一转发网关 |
| MGR | Manager | 管理、指标和编排等 | 不是应用文件读写的必经中转站 |
| MDS | Metadata Server | 管理命名空间、权限、布局、客户端 capabilities 与缓存协调 | 不转发所有文件内容，也不只靠内存保存元数据 |
| OSD | Object Storage Daemon | 管理存储设备和 RADOS 对象读写、复制与恢复 | 不等同于一台物理服务器 |

一台物理服务器通常有多个 OSD，常见配置是一块数据盘对应一个 OSD。MDS 管理的元数据也会持久化到 RADOS 元数据池；CephFS 文件内容存放在数据池中。

**这里的底层对象是 RADOS 对象，不意味着客户端要通过 S3/RGW 读取 CephFS 文件。**

## 4. 物理连接：计算节点连谁，走哪张网络？

### 4.1 教学拓扑

| 主机 | public 地址 | cluster 地址 | 角色 |
|---|---|---|---|
| ceph-s1 | 10.20.0.11 | 10.30.0.11 | MON、MGR、MDS、OSD |
| ceph-s2 | 10.20.0.12 | 10.30.0.12 | MON、MGR、MDS、OSD |
| ceph-s3 | 10.20.0.13 | 10.30.0.13 | MON、OSD |
| gpu-c1 | 10.20.0.21 | 不需要 | CephFS 客户端、训练程序 |
| gpu-c2 | 10.20.0.22 | 不需要 | CephFS 客户端、训练程序 |

```text
GPU 节点 1 ──网卡──┐
                  ├──[public 网络交换机]──┬── ceph-s1 客户端网卡
GPU 节点 2 ──网卡──┘                     ├── ceph-s2 客户端网卡
                                         └── ceph-s3 客户端网卡

ceph-s1 后端网卡 ──┐
ceph-s2 后端网卡 ──┼──[cluster 网络交换机]
ceph-s3 后端网卡 ──┘
```

这是角色混部的小型示例，不是容量或性能建议。生产可以将 MDS 等角色独立部署。

### 4.2 public 与 cluster 网络

| 网络 | 常见流量 | 计算节点要连接吗？ |
|---|---|---|
| public network | 客户端与 MON、MDS、OSD 通信；也有部分服务间通信 | 要，或者至少有到所公布地址的可用路由 |
| cluster network | 配置独立后端网络后，OSD 间复制、恢复、心跳等 | 通常不需要 |
| 管理网络 | SSH、运维、监控等 | 按管理需求 |
| GPU 训练通信网络 | GPU 节点间梯度、参数和激活通信 | 按训练方案 |

**public 不表示公网。** 这些通常都是受控内网；public/cluster 网络也不是“GPU 计算并行的两张网络”。

Ceph 可以只用一张网络。拆分两张网络是为了隔离流量和故障影响，不是启动 CephFS 的必要条件。仅配置两个网段或 VLAN，不能保证独立物理带宽。

### 4.3 最重要的连接条件：不能只通 MON

客户端需要访问集群实际公布的 **MON、MDS、OSD public 地址**。MON 是引导入口，不是数据代理。只让客户端连上 MON，再把所有 OSD 藏在不可达的私网后面，不能完成原生 CephFS 读写。

常见端口是 MON TCP 3300/6789，OSD、MDS 等通常使用 TCP 6800–7568 范围内的端口，最终按所选版本和实际配置开放。SSH 是管理链路，不能替代这些数据服务连通性。

需要同时检查服务监听地址、节点间路由、防火墙和多网卡选址。不要只验证 `ping` 或 SSH 成功。

## 5. 动画一：挂载后，文件读取为什么不经过 MDS 转发？

演示选择一条客户端数据缓存未命中的读取路径：先取得集群视图和文件访问能力，再直接向两个 OSD 读取两个不同对象。

<NetworkStorageDemo mode="ceph-read" />

应观察的三个不变量：

1. **MON 提供集群视图，不代替客户端转发所有 I/O。** 已缓存的映射会更新，并非每次 `read()` 都重新联系 MON。
2. **MDS 协调元数据和 capabilities，文件数据直接走客户端与 OSD。** 元数据和访问能力有效时，也不必每次重新走完整路径解析。
3. **两个并行读取对应不同对象，不是同时读取同一对象的所有副本。** 对象放置只是示意，不是简单轮询算法。

客户端根据文件布局把文件偏移映射为 RADOS 对象，再结合 OSDMap、CRUSH 和 PG 映射等定位对象的服务 OSD。这里 PG 是 Placement Group（放置组），用于对象放置和管理；CRUSH 是决定数据放置的算法体系。

小文件可能只涉及一个数据对象，串行请求也不自动形成高并发。需要参考真实布局、I/O 深度和访问模式判断性能。

## 6. 动画二：写入为什么有两类网络流量？

演示假设：客户端已经完成认证并获得写权限；选择一个对象；健康的三副本池在三个主机上保存该对象；集群配置了独立 cluster 网络。

<NetworkStorageDemo mode="ceph-write" />

应区分：

- **客户端 → 主 OSD**：走 public 网络，提交对象写入。
- **主 OSD → 副本 OSD**：走配置的后端网络，复制的是同一个对象。
- **副本确认 → 主 OSD 汇集条件 → 客户端完成**：是健康三副本场景下的简化存储层过程，不是通用的“任意两份确认就成功”公式。

::: warning 存储层完成不等于任意 `write()` 的返回时刻
普通 buffered `write()` 可能先写入客户端缓存并返回，之后再回写。需要应用可依赖的持久性边界时，应正确使用 `fsync()` 等机制，并考虑错误处理。对象写入、CephFS 文件数据、元数据与应用 checkpoint 提交不是同一个层次；动画没有模拟全部协议细节、重试或降级写入。
:::

多副本主要解决可靠性问题，也会消耗额外存储和网络资源；不能把它当作读取性能必然按副本数倍增的机制。

## 7. cephadm 部署：从机器到文件系统

### 7.1 前置条件

在三台存储节点准备：

- 稳定主机名、IP、名称解析和时间同步。
- 对应 Ceph 稳定版本要求的 Python、Podman/Docker、LVM2 等依赖。
- 安装所选稳定版 cephadm；不要混用未经验证的镜像和客户端版本。
- 已核对身份、备份和用途的空白数据盘；不要选择系统盘。
- public/cluster 网卡及路由、防火墙配置。

**计算节点只需客户端，不需要运行 OSD，也不需要把自己的系统盘贡献给集群。**

### 7.2 在 ceph-s1 引导集群

```bash
sudo cephadm bootstrap \
  --mon-ip 10.20.0.11 \
  --cluster-network 10.30.0.0/24
```

这会创建初始 MON、MGR 等，并生成集群配置和管理凭据。若使用单网络，应省略独立后端网段配置。多网卡环境还要核对 public 网络及服务公布地址是否符合规划。

进入管理 shell：

```bash
sudo cephadm shell
```

下文到客户端挂载之前的 `ceph ...` 命令都在管理环境执行。

### 7.3 配置编排 SSH 并添加存储节点

```bash
ceph cephadm get-pub-key
```

将该公钥安全地安装到其他存储节点的指定管理账户；默认管理账户与自定义 SSH 用户的要求，应对照部署版本确认。不要把这里的 SSH 身份和后文的 CephX 客户端身份混在一起。

```bash
ceph orch host add ceph-s2 10.20.0.12
ceph orch host add ceph-s3 10.20.0.13
ceph orch host ls

ceph orch apply mon --placement="ceph-s1,ceph-s2,ceph-s3"
ceph orch apply mgr --placement="ceph-s1,ceph-s2"
```

主机名应与节点实际主机名一致。三个 MON 的多数派仲裁与文件数据的三副本是两件不同的事。

### 7.4 核对空盘，再部署 OSD

```bash
ceph orch device ls
```

::: danger 磁盘操作边界
OSD 部署会将指定设备交给 Ceph 使用。必须先核对稳定磁盘标识、盘上已有数据和备份，再替换下方占位符。这里故意不用“自动使用所有可用设备”的命令，也不包含清盘命令。
:::

下面是不可原样使用的设备占位示例：

```bash
ceph orch daemon add osd \
  ceph-s1:/dev/disk/by-id/EXAMPLE_EMPTY_DISK_S1
ceph orch daemon add osd \
  ceph-s2:/dev/disk/by-id/EXAMPLE_EMPTY_DISK_S2
ceph orch daemon add osd \
  ceph-s3:/dev/disk/by-id/EXAMPLE_EMPTY_DISK_S3

ceph osd tree
ceph -s
```

示例每主机一个 OSD，只用于解释结构。实际 OSD 数量、盘型、内存和网络要一起规划。

### 7.5 创建 CephFS 并部署 MDS

把文件系统创建和 MDS 放置分开写，便于分别检查：

```bash
ceph fs volume create trainingfs
ceph orch apply mds trainingfs \
  --placement="ceph-s1,ceph-s2"
ceph fs set trainingfs max_mds 1
```

volume 创建接口会创建文件系统及相应池，并可通过编排部署 MDS。随后显式规定两个 MDS 的放置位置，并保持一个 active rank，预期为一个 active、一个 standby，而不是两个 active。

```bash
ceph fs ls
ceph fs status trainingfs
ceph orch ps
ceph osd pool ls detail
ceph osd tree
ceph -s
```

验收时必须检查：

- 元数据池和数据池的副本数及 `min_size`（允许 I/O 的最少副本条件）。
- CRUSH 规则是否按预期的主机故障域分散副本。
- 三个 MON 是否形成仲裁，MDS 是否有预期 active/standby。
- PG 和集群健康情况，是否仍有未解决的警告。

**三副本参数不自动保证副本跨三台主机；三台机器也不等于生产高可用已全部验收。** 不要通过随意降低副本数或 `min_size` 来掩盖部署问题。

## 8. 计算节点：凭据、挂载与验证

### 8.1 创建最小范围的客户端身份

管理端示例：

```bash
ceph fs authorize trainingfs client.train / rw
ceph auth get client.train
ceph auth get-key client.train
ceph fsid
```

这是便于学习的文件系统根读写授权。真实环境应按任务、目录和只读/读写用途收紧，且不要把 `client.admin` 密钥发到计算节点。

::: tip 修改权限不是简单重跑一次 authorize
不要把给已有身份执行更窄的 `ceph fs authorize` 当成“自动撤销旧权限”。该接口可能保留或扩展既有 capabilities。变更后必须检查完整权限；撤销或替换应使用对应版本支持的权限管理流程。
:::

### 8.2 准备每台计算节点

Ubuntu/Debian：

```bash
sudo apt update
sudo apt install ceph-common
sudo mkdir -p /mnt/cephfs
```

通过安全通道把 `ceph auth get-key client.train` 的输出保存到客户端：

```text
/etc/ceph/ceph.client.train.secret
```

文件只包含密钥值，不是完整 keyring 内容。确认归 root 所有并限制权限：

```bash
sudo chown root:root /etc/ceph/ceph.client.train.secret
sudo chmod 600 /etc/ceph/ceph.client.train.secret
```

不要把密钥放进 Git、命令行参数、截图或训练日志。

### 8.3 使用内核 CephFS 客户端挂载

```bash
FSID='替换为 ceph fsid 输出的实际集群标识'

sudo mount -t ceph \
  "train@${FSID}.trainingfs=/" \
  /mnt/cephfs \
  -o "mon_addr=10.20.0.11:6789/10.20.0.12:6789/10.20.0.13:6789,secretfile=/etc/ceph/ceph.client.train.secret"
```

| 字段 | 含义 |
|---|---|
| `train` | CephX 身份 `client.train` 在设备名中的简写 |
| `FSID` | Ceph 集群唯一标识，不是文件系统名 |
| `trainingfs` | 文件系统名称 |
| `=/` | 挂载此文件系统的根目录 |
| `/mnt/cephfs` | 客户端本地挂载点 |
| `mon_addr` | 初始联系的 MON 地址，示例以 `/` 分隔 |
| `secretfile` | 仅含该客户端密钥的本地文件 |

该设备名格式需要兼容的内核及 `mount.ceph` 工具。示例使用 MON 6789 端口，不应把它理解为所有现代集群的唯一选择；仅启用 msgr2 的集群应使用相应端口和客户端配置。CephX 认证不等于所有链路默认已经加密，传输安全要单独设计。

### 8.4 验证并按需持久化

```bash
findmnt /mnt/cephfs
df -hT /mnt/cephfs
ls /mnt/cephfs
```

在第二台计算节点重复配置后，二者看到同一文件系统。共享挂载不是文件同步软件，不会为每个节点制作独立完整副本。

手动验证通过后，才考虑增量添加 `/etc/fstab`，下行中的 `ACTUAL_FSID` 必须替换为真实值：

```text
train@ACTUAL_FSID.trainingfs=/ /mnt/cephfs ceph mon_addr=10.20.0.11:6789/10.20.0.12:6789/10.20.0.13:6789,secretfile=/etc/ceph/ceph.client.train.secret,_netdev,nofail,x-systemd.automount 0 0
```

这是一种 systemd 按需挂载配置，不是所有任务必须采用的策略。启动训练前应确认实际 CephFS 挂载已经成功；`nofail` 允许主机继续启动，不代表存储故障可忽略。

### 8.5 两层权限要分别检查

```text
CephX 身份 / capabilities 允许吗？
                  ↓
Linux UID/GID、权限位和 ACL 允许吗？
```

`client.train` 不等于一名叫 `train` 的 Linux 用户。普通用户能否创建文件仍受 POSIX 权限影响；多计算节点需要一致的身份管理。若将不可信客户端接入共享文件系统，还需单独设计 root 权限约束、客户端可信边界等，不能只依赖密钥文件存在。

## 9. 常见错误与排查路径

| 症状或误解 | 应检查或纠正 |
|---|---|
| MON 通了，但挂载后 I/O 挂住 | 客户端是否能访问 MDS 和全部相关 OSD 的 public 地址 |
| 以为所有字节都经过 MDS | MDS 管元数据，客户端直连 OSD 读写内容 |
| 两个 MDS 就是元数据双活 | 检查 active rank 与 standby；不是只数进程 |
| 三副本就是三倍读取吞吐 | 副本是可靠性机制；读取并行看不同对象和请求 |
| 副本数为三就一定跨三机 | 核对 CRUSH 规则和主机故障域 |
| 能挂载但普通用户不能写 | 分别检查 CephX capabilities、UID/GID、权限位与 ACL |
| `rw` 改成 `r` 再 authorize 就完成撤权 | 检查旧 capabilities 是否仍然存在 |
| 故障恢复时训练突然变慢 | 后台恢复和前台 I/O 是否争抢网络、磁盘和 CPU |
| 海量小文件加载慢 | MDS、目录热点、元数据往返、客户端缓存和数据组织 |
| 挂载成功就等于 GPU 直接读盘 | 普通训练通常仍经过主机内存、解码预处理与主机到 GPU 传输 |

三节点教学集群不代表高吞吐生产集群。应分别测试正常、降级、恢复、客户端重连与应用 checkpoint 持久性；副本和快照都不能无条件替代独立备份。

## 我的理解

> 我把它分成三条路径：MON 帮客户端了解集群，MDS 协调文件系统语义，OSD 承载字节读写。再从两个方向看扩展：不同对象的分布支持并行，同一对象的多副本支持容错。它们协作，但不是同一回事。

这是基于机制形成的理解，不是一次生产部署或压测的经验结论。

## 参考与相关知识

以下使用固定的 Squid 文档入口，避免把滚动的开发版文档直接当成当前环境保证；安装其他版本时需要对应切换。

- [Cephadm 安装与 bootstrap](https://docs.ceph.com/en/squid/cephadm/install/)
- [Cephadm MDS 服务部署](https://docs.ceph.com/en/squid/cephadm/services/mds/)
- [Ceph 网络配置](https://docs.ceph.com/en/squid/rados/configuration/network-config-ref/)
- [CephFS volumes 与 subvolumes](https://docs.ceph.com/en/squid/cephfs/fs-volumes/)
- [使用内核驱动挂载 CephFS](https://docs.ceph.com/en/squid/cephfs/mount-using-kernel-driver/)
- [CephFS 客户端授权](https://docs.ceph.com/en/squid/cephfs/client-auth/)
- [官方文档源码：挂载](https://github.com/ceph/ceph/blob/squid/doc/cephfs/mount-using-kernel-driver.rst)与[客户端授权](https://github.com/ceph/ceph/blob/squid/doc/cephfs/client-auth.rst) — 本次核对采用的官方源码入口
- [NFS：部署、挂载与计算节点连接](./nfs-deployment-and-mount.md)
- [并行文件系统](./parallel-file-systems.md) — 文件数据分布、条带化与并发访问
- [Ceph、Lustre、MinIO 入门概览](./ceph-lustre-minio-overview.md) — Ceph 平台、CephFS、RBD 与对象接口的定位
- [副本、纠删码与放置](./replication-erasure-coding-placement.md) — 容错与故障域
- [元数据与性能瓶颈](./metadata-and-performance.md) — 小文件与热点
- [PV、PVC、StorageClass 与 CSI](../../engineering/kubernetes/08-storage-pv-pvc-and-csi.md) — 裸机挂载与训练 Pod 挂载是两层配置
