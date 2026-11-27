# Mini CephFS：核心机制教学模拟器

面向学习 CephFS 内部数据路径的读者：**只用 C++17 标准库，把 Client、MON、MDS、RADOS 和多个 OSD 放在一个进程中，实际创建目录、切分对象、复制、读写、模拟节点故障和修复。**

它不依赖真实 Ceph，也不是 `libcephfs` 客户端、可挂载文件系统或兼容 Ceph 的实现。所有内容只在进程内存中，退出即丢失，不操作宿主机数据盘。

## 构建、测试、运行

需要 CMake 3.20+（用于 `ctest --test-dir`）和支持 C++17 的编译器。下面从本工程目录执行；**不需要安装 Ceph、Docker 或任何第三方 C++ 库**。

### Linux / macOS / MinGW / Ninja 等单配置生成器

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
./build/cephfs_demo
```

在 Windows 中使用 MinGW 时可显式选择 `-G "MinGW Makefiles"`；使用 Ninja 时选 `-G Ninja`，均需要先配置好对应编译器。Windows 单配置程序的文件名带 `.exe`。

### Windows / Visual Studio 多配置生成器

在有 Visual Studio C++ 工具链的开发者终端运行：

```powershell
cmake -S . -B build
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
.\build\Debug\cephfs_demo.exe
```

测试也应在 Release 下执行，确认检查不会被 `NDEBUG` 禁用：多配置用 `--config Release` 和 `-C Release`；单配置在另一个目录执行 `cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release` 后构建、测试。

没有 CMake 时也可直接编译（测试和演示分别是独立程序）：

```sh
mkdir -p build
g++ -std=c++17 -Wall -Wextra -Wpedantic -Isrc src/main.cpp -o build/cephfs_demo
g++ -std=c++17 -Wall -Wextra -Wpedantic -Isrc tests/tests.cpp -o build/cephfs_tests
./build/cephfs_tests
./build/cephfs_demo
```

上述直接编译命令使用 POSIX/Git Bash 语法，输出保留在已忽略的 `build/` 中。Windows 控制台应使用 UTF-8，以正确显示演示中的中文。

## 先读哪几个文件

| 文件 | 读什么 |
|---|---|
| [src/main.cpp](src/main.cpp) | 从一次完整演示理解调用链；默认 4 个 OSD、3 副本、8 字节对象 |
| [src/mini_cephfs.hpp](src/mini_cephfs.hpp) | 唯一核心模型：OSD → Rados → Monitor → MDS → Client |
| [tests/tests.cpp](tests/tests.cpp) | 路径/会话、二进制与边界、放置、故障、整次写入预检查、永久数据丢失、实际调用日志 |
| [CMakeLists.txt](CMakeLists.txt) | 两个可执行程序及一个 CTest 入口 |
| [完整知识指南](../../docs/knowledge/distributed-systems/distributed-storage/cephfs-cpp-demo.md) | 对象映射、故障策略及与真实 CephFS 的差异 |

```text
Client --连接--> MON（静态集群视图）
  |
  +--目录 / inode / 文件大小--> MDS（仅元数据）
  |
  +--文件偏移拆成对象--> Rados（对象路由）--> 主 OSD --> 副本 OSD
```

日志来自实际执行的查询、GET、PUT、REPLICATE 和 REPAIR 路径，不是预设剧本；这里的箭头是本进程函数调用，不是网络消息。

## 演示能看到什么

运行后依次创建目录和文件、跨对象写入与读取、局部覆盖、停止主 OSD 后从副本读取、拒绝受影响的写入、替换为空节点并修复、重新写入，以及全部副本不可读时的明确报错。每阶段有内容核对，失败退出码为 1。

API 支持 `connect`、`mkdir`、`create`、`write`、`read`、`stat`、`ls`。修改返回 `bool`，查询返回 `std::optional`；空文件/空目录是成功值，`std::nullopt` 是失败。调用失败后查看 `Client::error()`；下一次客户端文件操作会清除旧错误。

## 最重要的简化

- **仅同步串行调用**，故障在操作之间注入，没有线程、TCP 或真实 I/O 并行。
- 固定对象大小，确定性轮转式放置；**不是 CRUSH、PG 或动态 OSDMap**。
- **不接受降级写**：本次写入涉及对象的全部目标都要可写，旧对象副本必须完整且相同。先预检查整次写入，再修改对象，最后更新文件大小。无关对象的节点故障不必阻止一次写入。
- 节点停止保留内存；替换清空该模拟节点的内存。恢复前只允许从存活副本读取；显式 `repair()` 要求所有配置节点在线且每个已知对象至少有一份可读副本。全部副本丢失不可恢复。
- 不支持稀疏写：偏移不能超过 EOF；读长度会截断到 EOF。仅接受规范绝对路径，不支持 `.`、`..`、重复斜线或尾斜线（根目录除外）。
- **MDS 元数据和对象清单也只在内存中**；真实 CephFS 会将元数据及日志存于 RADOS。本例没有进程重启恢复、磁盘持久化、`fsync()`、CephX、caps/缓存一致性、共识、MDS HA、重平衡或完整 POSIX 操作。
- 不模拟写入途中的进程崩溃、内存分配失败、静默损坏、网络分区或并发写；预检查不是通用事务协议。直接修改内部 OSD/MDS 状态属于测试接口，不提供真实客户端隔离或权限边界。

本工程的“完整”是一个**核心机制闭环**，不是用几百行代码替代生产 CephFS。

## 当前交付的验证状态（2026-11-27）

- 配套 VitePress 指南已完成站点构建；这不代表 C++ 已通过验证。
- 当前 Windows 会话中执行 `cmake` 返回 `command not found`；通过 `where.exe` 未在 PATH 找到 CMake、GCC、Clang 或 MSVC，WSL 也不可用。
- **因此尚未编译本工程、执行 demo 或运行 Debug/Release 测试。** 上述七组测试是已编写的覆盖范围，不是已通过的测试结果。请在配置好工具链的环境按前述步骤验收；本次没有自动安装工具。
