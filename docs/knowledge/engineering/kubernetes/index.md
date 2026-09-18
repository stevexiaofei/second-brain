---
title: Kubernetes 入门 Wiki
type: map
status: seed
tags: [Kubernetes, K8s, Cloud Native, Container Orchestration, DevOps]
created: 2026-08-31
updated: 2026-08-31
source: roadmap.sh Kubernetes Roadmap 与 Kubernetes 官方文档
---

# Kubernetes 入门 Wiki

## 一句话理解

> Kubernetes（K8s）是一个通过 API 管理容器化工作负载的分布式控制系统：你声明「希望系统是什么样」，控制器持续观察实际状态并采取行动，使集群趋近目标状态。

它不是「更强的 Docker」，而是把**调度、服务发现、滚动发布、配置、存储、权限和自愈**统一到一套 API 与控制循环中。

## 这套 Wiki 要解决什么问题

[roadmap.sh Kubernetes Roadmap](https://roadmap.sh/kubernetes) 信息覆盖面很广，直接照着节点逐个搜索容易陷入名词堆积。本系列把它压缩为一条可执行的入门路径：

```mermaid
flowchart LR
    A[容器与网络基础] --> B[集群架构与控制循环]
    B --> C[对象、YAML 与 kubectl]
    C --> D[Pod 与 Workload]
    D --> E[Service / Ingress / DNS]
    E --> F[配置与存储]
    F --> G[资源、调度与扩缩容]
    G --> H[安全、Helm、可观测性]
    H --> I[CRD / Operator / 生产实践]
```

目标是学完核心章节后，能够：

- 看懂一个 Kubernetes 集群和请求链路；
- 编写并应用最小的 Deployment、Service、ConfigMap、Secret、PVC；
- 理解 Pod 为什么被调度、为什么重启、为什么访问不到；
- 用 `kubectl` 按层次排查应用、网络、节点和控制面的故障；
- 知道 Helm、HPA、RBAC、CRD、Operator 解决什么问题，以及下一步该学什么。

## 推荐阅读顺序

### 第一阶段：建立模型

1. [01 容器、网络与 Kubernetes 前置知识](./01-prerequisites-containers-and-networking.md)
2. [02 集群架构与声明式控制循环](./02-architecture-and-control-loop.md)
3. [03 kubectl、对象与 YAML](./03-kubectl-objects-and-yaml.md)

### 第二阶段：部署并访问应用

4. [04 Pod 与多容器模式](./04-pods-and-container-patterns.md)
5. [05 Workload：Deployment、StatefulSet 与 Job](./05-workloads-deployments-and-jobs.md)
6. [06 Service、Ingress、DNS 与网络策略](./06-services-networking-ingress-and-dns.md)

### 第三阶段：让应用可配置、可持久化、可扩展

7. [07 ConfigMap 与 Secret](./07-configuration-configmap-and-secrets.md)
8. [08 PV、PVC、StorageClass 与 CSI](./08-storage-pv-pvc-and-csi.md)
9. [09 资源、调度与自动扩缩容](./09-resources-scheduling-and-autoscaling.md)

### 第四阶段：运维与扩展

10. [10 RBAC、ServiceAccount 与 Pod 安全](./10-security-rbac-and-pod-security.md)
11. [11 Helm 与 Kubernetes 包管理](./11-helm-and-package-management.md)
12. [12 可观测性与故障排查](./12-observability-and-troubleshooting.md)
13. [13 CRD、Operator 与生产下一步](./13-crd-operators-and-production-next-steps.md)

不必一开始学习所有生态项目。先完成一条「部署 Web 服务 → 暴露服务 → 注入配置 → 挂载存储 → 观察与排障」的主线，再按工作需要进入安全、Helm、Operator 或 GitOps。

## Roadmap 到本系列的映射

| roadmap 主题 | 本系列章节 | 入门时应掌握的结果 |
|---|---|---|
| Containers、Linux、Networking | 01 | 明白 K8s 管理的对象和网络前提 |
| Cluster、Nodes、Namespaces | 02、03 | 能区分集群、节点、命名空间和 API 对象 |
| kubectl、YAML、Labels、Selectors | 03 | 能声明、查询、修改和删除对象 |
| Pods、Init Containers、Probes | 04 | 能解释 Pod 生命周期和健康检查 |
| Deployments、ReplicaSets | 05 | 能滚动发布、扩容和回滚无状态服务 |
| StatefulSets、DaemonSets、Jobs、CronJobs | 05 | 能按工作负载语义选择控制器 |
| Services、Ingress、DNS、NetworkPolicy | 06 | 能让 Pod 被稳定发现和访问 |
| ConfigMaps、Secrets | 07 | 能把配置与镜像解耦，并认识安全边界 |
| Volumes、PV/PVC、StorageClass、CSI | 08 | 能为有状态应用请求持久存储 |
| Requests/Limits、Scheduling、HPA | 09 | 能理解调度依据与容量边界 |
| RBAC、ServiceAccounts、Pod Security | 10 | 能按最小权限运行应用 |
| Helm | 11 | 能安装、升级、回滚一组 Kubernetes 资源 |
| Logging、Monitoring、Troubleshooting | 12 | 能用证据而非猜测定位故障 |
| CRD、Controllers、Operators | 13 | 能理解 Kubernetes 的扩展模型 |

## 贯穿实验：一个最小 Web 服务

建议准备一个可访问的容器镜像（以下用 `nginx:1.27` 作为示例），按下面的顺序逐步增加能力：

```text
Deployment → Service → Ingress
                    ↘ ConfigMap / Secret
                    ↘ PVC（如果应用需要持久数据）
                    ↘ requests / limits / HPA
                    ↘ 日志、事件、探针与排障
```

开始前准备一个本地集群，例如 kind、minikube、Docker Desktop Kubernetes 或云厂商集群。不同发行版的 LoadBalancer、Ingress Controller、存储类和权限默认值不同；本系列的 YAML 是学习骨架，不应未经审核直接用于生产。

```bash
kubectl version --client
kubectl get nodes
kubectl create namespace k8s-lab
kubectl config set-context --current --namespace=k8s-lab
```

删除实验资源：

```bash
kubectl delete namespace k8s-lab
```

## 核心心智模型

### 1. API 是入口

用户、CI/CD、控制器都通过 API Server 读写对象。`kubectl` 只是最常用的客户端，不是 Kubernetes 的核心本体。

### 2. 对象是意图

YAML 中的 `spec` 表达期望状态，`status` 记录观察到的状态。控制器不断比较两者并执行修正。

### 3. Pod 是调度单位，容器不是

调度器把 Pod 放到节点上；同一 Pod 内的容器共享网络命名空间、IP 和可选的卷。多数应用不直接管理 Pod，而是由 Deployment、StatefulSet 等 Workload 管理。

### 4. Service 是稳定入口

Pod 会被重建、IP 会变化。Service 通过 selector 找到一组符合条件的 Pod，为客户端提供稳定的虚拟入口。

### 5. 控制面与数据面分离

控制面决定「应该运行什么、放在哪里、怎样暴露」；节点上的 kubelet、容器运行时、网络和存储插件负责让这些决定真正发生。

## 先记住的词

| 词 | 最短解释 |
|---|---|
| Cluster | Kubernetes 管理的一组控制平面与工作节点 |
| Node | 实际运行 Pod 的机器或虚拟机 |
| Namespace | 在同一集群内隔离和组织对象的逻辑边界 |
| Pod | 一个或多个紧密协作容器的最小调度单元 |
| Workload | 管理 Pod 副本和生命周期的资源 |
| Service | 面向一组 Pod 的稳定网络入口 |
| Ingress | HTTP/HTTPS 路由规则，通常需要 Ingress Controller |
| ConfigMap / Secret | 非敏感 / 敏感配置数据对象 |
| PV / PVC | 存储资源 / 对存储的请求 |
| Controller | 根据 spec 与 status 推动实际状态变化的控制循环 |
| CRD | 给 Kubernetes API 增加新资源类型的声明 |
| Operator | 用控制器管理特定应用生命周期的模式 |

## 学习出口

完成本系列后，按实际目标选择下一条路线：

- **平台运维**：集群升级、备份恢复、etcd、节点生命周期、网络插件、策略和多租户；
- **应用交付**：镜像构建、Helm、Kustomize、GitOps、部署策略和发布观测；
- **平台开发**：client-go、Controller Runtime、CRD、Operator、Webhook 和测试；
- **云原生网络**：CNI、Service 流量路径、NetworkPolicy、Gateway API、服务网格；
- **云原生存储**：CSI、快照、备份、容量管理和有状态工作负载；可结合 [面向 AI 训练与多云场景的存储工程地图](../../distributed-systems/distributed-storage/ai-training-and-multicloud-storage-map.md) 学习。

## Related

- [分布式系统核心理论基础](../../distributed-systems/distributed-systems-foundations.md) — 理解控制面、共识、故障与一致性的理论背景
- [面向 AI 训练与多云场景的存储工程地图](../../distributed-systems/distributed-storage/ai-training-and-multicloud-storage-map.md) — K8s 存储、CSI、Operator 与 AI 数据场景
- [Engineering 知识索引](../index.md)

## References

- [Kubernetes Roadmap — roadmap.sh](https://roadmap.sh/kubernetes)
- [Kubernetes Concepts](https://kubernetes.io/docs/concepts/)
- [Kubernetes Tutorials](https://kubernetes.io/docs/tutorials/)
- [Kubernetes API Reference](https://kubernetes.io/docs/reference/kubernetes-api/)
