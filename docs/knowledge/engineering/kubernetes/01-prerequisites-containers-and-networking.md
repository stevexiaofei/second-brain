---
title: Kubernetes 01：容器、网络与前置知识
type: concept
status: seed
tags: [Kubernetes, Containers, Docker, Linux, Networking]
created: 2026-08-31
updated: 2026-08-31
source: Kubernetes 官方文档与 roadmap.sh Kubernetes Roadmap
---

# Kubernetes 01：容器、网络与前置知识

## 一句话理解

> 学 Kubernetes 之前，不需要先成为 Linux 专家，但要能回答三个问题：容器是什么、进程如何通信、应用为什么需要被调度和持续管理。

## 为什么重要

Kubernetes 负责编排容器，而不是替容器本身提供隔离。若不了解镜像、进程、端口、DNS 和 Linux 资源，就容易把「应用启动失败」「Service 访问失败」「Pod 被驱逐」都归咎于 Kubernetes。

可以把系统分成三层：

```text
应用进程：HTTP 服务、Worker、数据库
    ↓ 被打包进
容器：隔离的进程 + 文件系统视图 + 网络/资源约束
    ↓ 被 Kubernetes 编排
集群：调度、服务发现、发布、自愈、权限、存储
```

## 容器的正确直觉

容器不是一台轻量级虚拟机。典型容器主要是宿主机上的一个或一组受隔离和限制的进程：

- **Namespaces** 隔离进程、网络、挂载点、用户等视图；
- **cgroups** 限制和统计 CPU、内存等资源；
- **镜像** 提供分层的只读文件系统，容器运行时在其上增加可写层；
- 容器通常共享宿主机内核，因此与虚拟机的隔离边界不同。

容器的生命周期通常与主进程相关。主进程退出，容器就结束；把多个完全无关的服务塞进一个容器，通常会损失独立发布和故障隔离能力。

## 镜像、容器与仓库

| 概念 | 含义 |
|---|---|
| Image | 不可变的分层应用模板，包含文件系统和启动配置 |
| Container | 某个镜像的运行实例，有自己的生命周期和可写层 |
| Registry | 保存和分发镜像的服务，如 OCI Registry、Docker Hub、私有仓库 |
| Tag | 人类可读的版本标记；可移动，不等同于不可变版本 |
| Digest | 镜像内容的摘要；按 digest 拉取可获得更强的内容确定性 |

生产环境不要把 `latest` 当作发布版本。更可控的做法是使用明确版本，甚至在发布清单中固定 digest，并配合镜像扫描、签名和供应链策略。

## 需要掌握的 Linux 基础

### 进程与信号

理解 PID 1、退出码、`SIGTERM` 和 `SIGKILL`。Kubernetes 删除 Pod 时通常先请求优雅终止，再在超时后强制结束；应用若不处理终止信号，滚动发布可能造成请求中断。

```bash
ps aux
kill -TERM <pid>
kill -KILL <pid>  # 最后手段，不给应用清理机会
```

### 文件系统与权限

容器镜像中的文件不等于持久数据。容器重建后，写入容器可写层的内容可能丢失；需要持久化时，要使用卷或外部存储。还要理解 UID/GID、文件权限以及只读根文件系统对应用的影响。

### 资源与 OOM

Linux 的内存不足可能由内核 OOM Killer 处理；Kubernetes 还会依据 requests、limits 和节点压力进行调度与驱逐。`OOMKilled` 是结果，不一定等于「代码有内存泄漏」，也可能是容器 limit 太小。

## 需要掌握的网络基础

### 监听地址与端口

服务监听 `127.0.0.1` 时只能接受本地回环请求；要接受来自 Pod 网络命名空间或外部的请求，通常监听 `0.0.0.0`（实际安全边界由网络策略和 Service 控制）。端口只是进程监听的位置，不等于 Kubernetes 的 Service 端口。

### TCP/IP、HTTP 与 DNS

至少应理解：

- TCP 连接由源/目的 IP 与端口等信息区分；
- HTTP 请求经过代理或负载均衡时，Host、路径和头部可能参与路由；
- DNS 把名称解析为地址，但解析成功不代表端口上有健康服务；
- `localhost` 在容器中指向当前容器/Pod 的网络命名空间，不指向你的笔记本或另一个 Service。

```bash
curl -v http://example.com
getent hosts example.com
ss -lntp
```

### Kubernetes 网络的预期

Kubernetes 网络模型通常要求：

1. 每个 Pod 拥有集群内可路由的 IP；
2. 同一集群中的 Pod 可以直接通信（具体限制受 NetworkPolicy 影响）；
3. 节点上的 kubelet 能访问 Pod；
4. Pod 不应依赖另一个 Pod 的固定 IP。

CNI 插件实现网络接口、地址分配和路由；Kubernetes API 本身不负责转发每个数据包。

## Docker 与 Kubernetes 的关系

Docker 是一套容器构建/运行工具；Kubernetes 是一套分布式编排 API 和控制系统。现代 Kubernetes 节点通过 CRI 对接 containerd、CRI-O 等容器运行时，Docker Engine 不再是 Kubernetes 必须依赖的运行时。

```text
Dockerfile → 镜像（OCI）→ Registry → 节点容器运行时 → Pod 中的容器
```

学习时可以用 Docker 构建镜像，但不要把 `docker ps` 看到的容器列表当作 Kubernetes 的真实状态来源。Kubernetes 的期望状态和对象状态应通过 API/`kubectl` 观察。

## 入门前自测

- 能解释镜像和容器的区别；
- 能构建并运行一个监听 HTTP 端口的镜像；
- 知道容器重启后可写层为什么不适合保存数据库数据；
- 能用 `curl`、DNS 查询和端口检查区分「名称解析失败」与「进程未监听」；
- 知道 `localhost`、Pod IP、Service 名称不是同一个概念。

## 常见误区

- **容器等于虚拟机**：容器共享内核，隔离模型和运维边界不同。
- **镜像 tag 不变所以内容不变**：tag 可以被重新指向；digest 更适合内容固定。
- **Service 端口就是容器端口**：Service 的 `port`、`targetPort`、NodePort 可能分别承担不同职责。
- **Pod IP 可以写进配置**：Pod 会被替换，应用应通过 Service 或其他稳定发现机制通信。
- **Kubernetes 会自动修复任何应用**：它能重启/替换符合声明的工作负载，但无法修复错误的镜像、配置或业务逻辑。

## 我的理解

容器解决的是「如何把一个进程及其依赖交付并隔离」，Kubernetes 解决的是「大量这样的进程在多台机器上如何持续满足期望状态」。因此学习顺序应该先建立进程和网络直觉，再理解对象和控制器，而不是从记忆 YAML 字段开始。

## Related

- [02 集群架构与声明式控制循环](./02-architecture-and-control-loop.md)
- [04 Pod 与多容器模式](./04-pods-and-container-patterns.md)
- [分布式系统核心理论基础](../../distributed-systems/distributed-systems-foundations.md)

## References

- [Kubernetes Concepts](https://kubernetes.io/docs/concepts/)
- [Container Runtimes](https://kubernetes.io/docs/setup/production-environment/container-runtimes/)
- [Kubernetes Network Model](https://kubernetes.io/docs/concepts/services-networking/)
- [Kubernetes Roadmap](https://roadmap.sh/kubernetes)
