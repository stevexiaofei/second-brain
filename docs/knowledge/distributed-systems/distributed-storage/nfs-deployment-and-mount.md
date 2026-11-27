---
title: NFS：部署、挂载与计算节点连接
type: concept
status: growing
tags: [NFS, File Storage, Mount, Network, Linux, AI Training]
created: 2026-11-20
updated: 2026-11-20
---

<script setup>
import NetworkStorageDemo from '../../../.vitepress/components/NetworkStorageDemo.vue'
</script>

# NFS：部署、挂载与计算节点连接

## 一句话理解

**存储服务器把目录通过 NFS 导出；计算节点把这个远端目录挂载到自己的目录树；应用仍然用普通文件路径访问，由操作系统把相关操作转换为网络请求。**

本文讨论 Linux 上常见的单服务器 NFSv4 部署。NFS 是文件访问协议，不是磁盘、网线或某一种后端存储；后端也可以是分布式文件系统。先理解这种基本结构，再读 [CephFS 的部署与挂载](./cephfs-deployment-and-mount.md)。

::: warning 示例边界
以下命令面向 Ubuntu/Debian 教学环境，没有在真实存储集群执行验证。IP、设备、目录、权限和防火墙规则都要按实际环境核对；不要直接覆盖已有配置。本文只解释一份数据集共享，不包含磁盘格式化、RAID、备份或生产高可用部署。
:::

## 1. 三件不同的事：接网络、导出、挂载

| 层次 | 谁来做 | 做什么 |
|---|---|---|
| 网络连接 | 网卡、交换机、路由和防火墙 | 让计算节点能访问存储服务器 |
| 服务端导出（export） | NFS 服务端 | 声明哪些目录允许哪些客户端访问 |
| 客户端挂载（mount） | 每个计算节点的操作系统 | 把远端文件系统接到本地挂载点 |

**网络通了，不代表目录已经导出；目录已导出，不代表客户端已经挂载；挂载成功，也不代表当前用户有写权限。**

## 2. 物理上怎么连接？

```text
存储服务器 10.20.0.10
  数据盘 → 本地文件系统 → /srv/nfs/datasets → NFS 服务
                                                │ 网卡
                                                │ 网线或光纤
                                            [交换机]
                                            /      \
                                         网卡      网卡
                                          │         │
                                      gpu-c1      gpu-c2
                                    10.20.0.21  10.20.0.22
```

- 计算节点访问服务器，不需要通过另一台计算节点转发。
- 不必都处于同一个交换机或网段，只要路由可达、访问策略允许。
- NFSv4 常见访问入口是服务端 TCP 2049；NFSv3 通常还涉及其他 RPC 服务和端口，不能直接套用这套防火墙假设。
- SSH 用于管理服务器，不是应用读取 NFS 文件的传输方式。
- 存储、管理和 GPU 间训练通信可以共用网络，但共用链路就可能争抢带宽。VLAN 隔离不等于增加物理带宽。

## 3. 动画：挂载不是把文件全部下载下来

逐步观察：**未挂载时本地目录只是一个目录；挂载建立路径映射；真正读取时才按需发起网络 I/O。** 演示选择缓存未命中的读路径，但服务端可能从缓存取得数据，并非每次都读盘。

<NetworkStorageDemo mode="nfs" />

假设服务端目录是：

```text
/srv/nfs/datasets/
└── train/
    └── image001.jpg
```

客户端挂载后，路径关系是：

```text
计算节点：/mnt/datasets/train/image001.jpg
                        │ NFS
                        ▼
服务端：  /srv/nfs/datasets/train/image001.jpg
```

两台计算节点挂载同一个导出目录，访问的是同一份远端数据，不是两份独立副本。客户端可能缓存数据和属性，但缓存不等于完整离线复制。

::: tip 挂载点已有文件会怎样？
挂载到非空目录时，原有本地内容会被暂时遮住，不是被删除。应优先选择空挂载点，避免把“本地目录里的文件”和“挂载后的远端文件”混淆。
:::

## 4. 服务端：准备存储并导出目录

### 4.1 安装服务

在存储服务器 `10.20.0.10` 上：

```bash
sudo apt update
sudo apt install nfs-kernel-server
sudo mkdir -p /srv/nfs/datasets
```

如果使用独立数据盘，应先完成并验证该盘的本地挂载，再导出其目录。可以检查：

```bash
findmnt -T /srv/nfs/datasets
```

不要把“数据盘没挂上、目录实际落在系统盘”误当成部署成功。

### 4.2 配置 `/etc/exports`

查看现有配置后，增量加入下面这一行：

```text
/srv/nfs/datasets 10.20.0.0/24(ro,sync,no_subtree_check,root_squash,fsid=0)
```

| 配置 | 含义 |
|---|---|
| `/srv/nfs/datasets` | 服务端实际共享目录 |
| `10.20.0.0/24` | 允许访问的客户端网段，应按实际需要缩小 |
| `ro` | 数据集只读；需要写入时才按用途改为 `rw` |
| `sync` | 对允许的写入采用同步导出语义；不替代应用层持久性设计 |
| `no_subtree_check` | 不做子目录树检查 |
| `root_squash` | 将客户端 root 映射为匿名身份，而不是赋予服务端 root 权限 |
| `fsid=0` | 将此目录设为这套 NFSv4 导出的伪文件系统根 |

这里把数据集设成只读，是示例的最小权限选择。checkpoint 若需要写入，应单独设计导出范围和权限，而不是默认让所有训练节点都可改训练数据。

**`fsid=0` 会影响客户端看到的路径，而且同一 NFSv4 导出命名空间应有明确的根。已有 NFS 服务不能不检查现有根导出就追加此配置。**

应用并查看导出：

```bash
sudo exportfs -rav
sudo systemctl enable --now nfs-kernel-server
sudo exportfs -v
```

最后确认服务端防火墙允许选定计算节点访问 TCP 2049；不要向公网直接开放。

## 5. 客户端：挂载到本地目录

### 5.1 每台计算节点都要挂载

在 `gpu-c1`、`gpu-c2` 分别执行：

```bash
sudo apt update
sudo apt install nfs-common
sudo mkdir -p /mnt/datasets

sudo mount -t nfs -o vers=4,ro \
  10.20.0.10:/ \
  /mnt/datasets
```

为什么这里是 `10.20.0.10:/`，不是服务端完整路径？

因为前面明确把 `/srv/nfs/datasets` 设置成了 `fsid=0`，所以客户端看到的远端根 `/` 就对应这个目录。**这是本示例的导出布局，不是所有 NFS 服务器都必须使用 `:/`。**

验证：

```bash
findmnt /mnt/datasets
ls /mnt/datasets
```

文件的 Linux 权限仍然有效。常见 `sec=sys` 环境依赖客户端提供的 UID/GID，应统一身份管理，并确保服务端文件确实允许目标用户读取。不要用 `chmod 777` 代替权限设计。

### 5.2 自动挂载

手动验证成功后，再检查并增量修改客户端 `/etc/fstab`：

```text
10.20.0.10:/ /mnt/datasets nfs vers=4,ro,_netdev,nofail,x-systemd.automount 0 0
```

该示例适用于使用 systemd 的系统：

- `_netdev`：标记为依赖网络的文件系统。
- `x-systemd.automount`：访问时触发挂载。
- `nofail`：挂载失败不阻塞正常启动；**不等于训练任务可以忽略存储故障**。

不建议为了“故障时快点返回”随意给需要可靠 I/O 的任务加 `soft`；它会改变失败处理语义。应按工作负载设计超时、故障处理与作业依赖。

::: warning 防止误写到本地目录
远端未挂载时，挂载点可能仍是一个普通本地目录。可写共享目录尤其要防止任务把 checkpoint 写到本地系统盘却误以为已持久化到共享存储。任务启动前应确认实际文件系统类型和来源；使用按需挂载时，还要核对真正的远端挂载已成功。
:::

## 6. 文件读取和 GPU 计算怎么衔接？

普通 NFS 路径可以理解为：

```text
训练进程 / DataLoader
  → Linux 文件系统层 / NFS 客户端
  → 客户端网卡 → 网络 → NFS 服务端
  → 服务端缓存或文件系统 / 磁盘
  → 数据返回客户端主机内存
  → 解码、预处理、组 batch
  → 传输到 GPU 显存
```

这是解释缓存未命中时数据流的简化模型：客户端命中缓存时可能不需要远端数据读取，服务端命中缓存时可能不访问磁盘。挂载 NFS 本身不会开启 GPU 直接存储访问。

## 7. 故障和性能应该按哪一层排查？

| 症状 | 优先检查 |
|---|---|
| 服务器连不上、挂载超时 | IP、路由、防火墙、NFS 服务、2049 端口 |
| 提示拒绝访问 | 导出规则、客户端来源地址、NFSv4 远端路径 |
| 能挂载但不能读写 | `ro/rw`、UID/GID、权限位、ACL、root squash |
| 能列目录，但大量小文件启动很慢 | 元数据往返、客户端缓存、文件数量；不只看带宽 |
| 多个训练节点同时读就变慢 | 服务端网卡、CPU、磁盘和请求并发是否饱和 |
| 写 checkpoint 后找不到或只在某台节点看到 | 是否真正挂载、实际写入路径、是否写到了本地目录 |

**单服务器 NFS 的流量最终汇聚到这台服务器，但 NFS 并不是“一次只能处理一个请求”。** 问题在于整体资源上限，而不是协议天然把所有 I/O 串行化。

更大规模可考虑并行文件系统、本地缓存或对象存储等方案，取决于访问模式；不是节点多了就必须无条件换 CephFS。

## 我的理解

> 挂载解决“程序用哪个路径访问”；导出和权限解决“谁能访问”；网络解决“请求怎么到达”；后端存储决定“字节实际保存在哪里”。这四层不能混为一谈。

这是本笔记形成的理解模型，不是一次真实部署或性能测试的经验记录。

## 参考与相关知识

- [Linux NFS 导出配置：exports(5)](https://man7.org/linux/man-pages/man5/exports.5.html)
- [Linux NFS 客户端选项：nfs(5)](https://man7.org/linux/man-pages/man5/nfs.5.html)
- [CephFS：并行访问、部署、挂载与网络](./cephfs-deployment-and-mount.md) — 从单入口共享扩展到原生客户端直接访问多个数据节点
- [并行文件系统](./parallel-file-systems.md) — 元数据、条带化和并发访问的基础概念
- [Git 在 NFS 上的性能问题](../../engineering/git-on-nfs.md) — 元数据密集型负载为什么对网络往返敏感
- [磁盘 I/O Benchmark](../../engineering/profiling/08-disk-io-benchmark.md) — 区分客户端缓存、服务端缓存和实际设备 I/O
- [PV、PVC、StorageClass 与 CSI](../../engineering/kubernetes/08-storage-pv-pvc-and-csi.md) — 宿主机能访问，不等于训练 Pod 已挂载
