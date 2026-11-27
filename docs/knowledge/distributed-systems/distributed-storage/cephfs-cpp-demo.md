---
title: Mini CephFS：用最少 C++ 看懂对象、副本与修复
type: project
status: growing
tags: [CephFS, C++, RADOS, MDS, OSD, Replication, Teaching]
created: 2026-11-27
updated: 2026-11-27
---

# Mini CephFS：用最少 C++ 看懂对象、副本与修复

## 1. 这个工程解决什么问题

[CephFS 部署、挂载与网络](./cephfs-deployment-and-mount.md) 解释的是实际系统如何协作；本工程让这些角色变成可以单步调试的 C++ 对象：**MDS 解析名字，Client 将文件拆成对象，主 OSD 保存并复制字节，故障后从副本读取或修复。**

::: warning 教学模型，不是真实 CephFS 实现
这是从零写的单进程、内存版模拟器，不调用 `libcephfs`，不兼容 Ceph 协议，不支持挂载。目录、文件、对象和副本确实由代码维护，但进程结束后全部丢失。下文的故障、修复和日志均指这个模型，不是对真实 Ceph 集群的验证结果。
:::

源码工程位于仓库根目录下的 `tools/cephfs_demo/`。只有一个核心模型头文件、一个演示入口、一个测试文件；构建不需要第三方 C++ 库。

```text
tools/cephfs_demo/
  CMakeLists.txt
  README.md
  src/
    mini_cephfs.hpp   OSD / Rados / Monitor / MDS / Client
    main.cpp          默认完整场景
  tests/
    tests.cpp         自包含回归测试
```

## 2. 一张调用图

```text
用户的文件操作
      |
    Client ---- connect ----> Monitor：静态集群视图
      |
      +------ 元数据 ------> MDS：目录项 -> inode / size
      |
      +------ 文件布局 ----> ObjectId = (inode, object index)
                               |
                             Rados：计算固定副本集合
                               |
                            主 OSD ------> 副本 OSD
                         实际内存字节      同一对象的副本
```

这里的箭头是同步 C++ 函数调用，不是网络 RPC。Rados 是对象层的教学门面，不是额外引入一台中心数据服务器；MDS 不接收、缓存或转发文件内容。

| 类 | 保存/处理什么 | 刻意不做什么 |
|---|---|---|
| `OSD` | 对象 ID 到字节数组、在线状态、主副本复制 | 磁盘、网络协议、后台工作线程 |
| `Rados` | 固定放置、对象读写、已知对象清单、显式修复 | CRUSH/PG、动态成员、重平衡 |
| `Monitor` | 连接时提供静态配置视图 | CephX、Paxos、动态 map 下发 |
| `MDS` | 目录树、递增 inode、文件大小 | 元数据池持久化、日志、caps、HA |
| `Client` | 路径查询、偏移拆分、读取拼接、写后更新大小 | fd、缓存回写、多客户端并发协调 |

## 3. 在本机运行

需要 **C++17 编译器和 CMake 3.20+**。从仓库根目录执行：

```sh
cmake -S tools/cephfs_demo -B tools/cephfs_demo/build -DCMAKE_BUILD_TYPE=Debug
cmake --build tools/cephfs_demo/build --config Debug
ctest --test-dir tools/cephfs_demo/build -C Debug --output-on-failure
```

Linux / Ninja / MinGW 等单配置生成器运行：

```sh
./tools/cephfs_demo/build/cephfs_demo
```

Windows Visual Studio 多配置生成器运行：

```powershell
.\tools\cephfs_demo\build\Debug\cephfs_demo.exe
```

Visual Studio 生成器的配置由 `--config Debug` 决定，`CMAKE_BUILD_TYPE` 不生效；可以省略该选项。MinGW/Ninja 则需在配置时选择并设置相应编译器，Windows 文件名带 `.exe`。无需管理员权限，也不会格式化磁盘或访问真实 Ceph。

默认场景使用 **4 个 OSD、每对象 3 副本、8 字节对象**。对象大小故意设小，便于看到一个短字符串拆成多个对象；它不是 CephFS 的默认或推荐布局参数。

## 4. 文件偏移怎样变成对象

设文件中的字节偏移为 $p$，每个对象的固定大小为 $B$，对象序号为 $j$，对象内偏移为 $r$。所有偏移和大小均以字节计，$p\geq 0$、$B>0$：

$$
j=\left\lfloor\frac{p}{B}\right\rfloor,\qquad r=p\bmod B
$$

模型使用 **inode 与对象序号**共同标识对象，不把路径字符串当作文件数据的身份。inode 是文件身份；路径只是通过目录项找到 inode 的名字，详见 [inode：文件身份、元数据与数据定位](./inode-file-metadata.md)。

例如，一个 inode 为 4 的文件写入 20 字节 `abcdefghijklmnopqrst`，对象大小为 8 字节：

| 对象 | 文件字节区间（左闭右开） | 有效内容 | 本例固定放置：主、副本、副本 |
|---|---|---|---|
| `ino-4.obj-0` | [0, 8) | `abcdefgh` | OSD0、OSD1、OSD2 |
| `ino-4.obj-1` | [8, 16) | `ijklmnop` | OSD1、OSD2、OSD3 |
| `ino-4.obj-2` | [16, 20) | `qrst` | OSD2、OSD3、OSD0 |

最后一个对象在内存中仍占 8 字节，未使用尾部填零；文件大小是 20，不是 24，读取不能把填充当作文件内容。**这是布局示例，不是默认 demo 字符串的逐字节输出。**

- **切对象/条带化的简化**：不同对象的主位置不同。
- **复制**：同一个对象在三个目标上保存相同字节。
- 放置采用 inode、对象序号和 OSD 数量的确定性轮转计算；**不等于 CRUSH 或 PG 映射**。

## 5. 最值得单步调试的两条路径

### 5.1 写：先准备，再提交，最后更新大小

在 `Client::write()` 上断点：

1. 检查已连接、路径指向文件、偏移不越过 EOF、加法不溢出。
2. 通过 MDS 取得 inode 与原文件大小。
3. 遍历本次涉及的全部对象：读取旧对象以保留邻接字节，在临时缓冲区覆盖本次数据，检查每个副本目标都可写。
4. **全部对象检查通过后**，才逐对象发送给主 OSD；由主 OSD 的 `replicate()` 路径保存自身并写入副本。
5. 全部对象完成后更新 MDS 文件大小。

尤其应测试：第一个对象可写，第二个对象的某个副本离线。如果边写边检查，就可能修改前半段后才报错；本例通过先准备整个写入，避免这类**模型内可预见故障**造成半次覆盖。

这不是分布式事务。它依赖单线程、节点状态在一次调用内不变化、不注入内存分配失败或执行中崩溃等前提；不能推导出真实 CephFS 跨对象原子写保证。

### 5.2 读：从对象副本拼回文件

在 `Client::read()` 与 `Rados::read()` 上断点：

1. MDS 返回 inode 和文件大小。
2. 先以文件大小截断请求；读到/超过 EOF 返回成功的空字节数组。
3. 对每个涉及对象，按固定副本集合寻找在线且有完整对象的节点。
4. 提取对象内需要的片段，按文件偏移顺序拼接。

对象不存在或全部副本不可读时返回失败，**不是把缺失数据当成零**。末尾填充与文件损坏是两种不同情形。

查询 API 使用 `std::optional`：有值但长度为零是成功，`std::nullopt` 是失败。`Client::error()` 提供最近一次失败的原因，并在下一次客户端文件操作开始时清空，避免旧错误误导调试。

## 6. 故障注入与修复为什么这样简化

| 操作/状态 | 字节是否保留 | 能否读取 | 能否成为写入目标 |
|---|---|---|---|
| 正常 OSD | 保留 | 可以 | 可以 |
| `stop_osd()` | 保留在该模拟对象中 | 不从该节点读；可读其他副本 | 不可以 |
| `start_osd()`，之前只是停止 | 保留 | 可以 | 可以；本例停机期间不接受涉及它的写入 |
| `replace_osd()` | 清空该模拟节点的对象表 | 从其他存活副本读取 | 修复前不可以 |
| 替换后 `stop/start` | 仍缺数据 | 不会凭空恢复 | **不能绕过恢复状态** |
| `repair()` 完成 | 从存活副本补齐 | 可以 | 可以 |

本例不允许本次目标集合中的降级写，所以没有“停机期间写了新版，旧节点回归却被误读”的版本分叉问题。故障节点若不属于本次写入涉及对象的副本集合，不必阻止该写入。

`repair()` 的前提是所有配置 OSD 已在线，每个已知对象至少有一份存活且可读的副本。对象清单只用于本进程的教学恢复：先检查所有源，再补齐缺失副本，最后接纳恢复节点写入。全部副本丢失时无法恢复，必须报错。

真实 Ceph 还要解决动态 acting set、peering、版本、日志、后台恢复、网络分区和崩溃重启等问题。这个保守模型不是 Ceph `size/min_size` 策略或任意多数派确认规则的替代品。相关基础见 [副本、纠删码与放置](./replication-erasure-coding-placement.md)。

## 7. 默认演示入口：直接引用源码

以下代码由 VitePress 在构建时直接读取工程文件，避免在文档里维护另一份副本：

::: details 展开 main.cpp：创建、读写、故障、修复与核对
<<< ../../../../tools/cephfs_demo/src/main.cpp{cpp}
:::

阅读核心模型时，建议按 **OSD → Rados → MDS → Client** 顺序；Monitor 在这个版本里只是连接入口。演示输出的 `LOOKUP`、`PUT`、`REPLICATE`、`GET`、`REPAIR` 来自实际调用。控制台打印顺序不是网络并发或持久化时序的证据。

## 8. 测试覆盖与验证口径

测试程序不使用会在 Release 中被禁用的 `assert()`。CTest 注册一个可执行程序，程序内部有七组测试：

1. 目录树、规范路径、文件/目录类型、会话检查与空目录查询。
2. 多对象读写、局部覆盖、追加、EOF、零长度、二进制数据及文件隔离。
3. 参数合法性、确定性放置、对象缺失与偏移溢出检查。
4. 主 OSD 离线、副本读取、保留数据的重启、空节点替换与修复。
5. **后续对象不可写时，先前对象、文件大小和对象清单均不变化**。
6. 全部副本丢失，读取和修复必须失败，空节点不得被误标成健康。
7. 实际操作产生元数据查询和对象复制/读取日志。

构建和测试命令见上文。编译通过、demo 成功退出、Debug/Release 测试通过，是不同层面的验证；本站文档构建通过不能代替它们。可用 GCC/Clang 的 AddressSanitizer/UndefinedBehaviorSanitizer 进一步检查内存与算术问题，但是否可运行取决于本地工具链。

## 9. 真实系统与本例不能混淆的地方

- **元数据持久化**：真实 MDS 元数据与日志存 RADOS；本例目录树、大小及对象清单均只在内存。
- **持久性**：本例没有磁盘、真实 `fsync()` 或重启恢复；三份内存副本不能证明持久化完成。
- **访问能力与一致性**：没有 CephX、UID/GID、权限位、caps、缓存回收、租约或并发一致性协议。
- **分布式性**：没有 TCP、真实 MON 共识、MDS HA 或多进程，顺序函数调用不能用于评估吞吐扩展。
- **文件接口**：只做创建、读写、查询；无 fd/open/close、删除、rename、硬链接、稀疏文件或完整 POSIX 语义。
- **故障边界**：故障在操作之间注入，不模拟操作中的宕机、分区、静默损坏或任意异常原子性；直接修改内部测试接口不属于真实客户端权限模型。

**学习上的价值是把“文件 → inode → 对象 → 副本”连成可运行的因果链，而不是把复杂的生产系统描述成几个容器就能完整替代。**

## Related

- [CephFS：并行访问、部署、挂载与网络](./cephfs-deployment-and-mount.md) — 真实系统的角色、拓扑和部署边界
- [inode：文件身份、元数据与数据定位](./inode-file-metadata.md) — 名字与文件身份
- [并行文件系统](./parallel-file-systems.md) — 条带化、并发与并行访问
- [副本、纠删码与放置](./replication-erasure-coding-placement.md) — 冗余与故障域
- [一致性、共识与故障恢复](./consistency-consensus-failure.md) — 本模型没有实现的分布式难题
