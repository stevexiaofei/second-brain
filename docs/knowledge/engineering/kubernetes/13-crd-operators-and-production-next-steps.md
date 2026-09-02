---
title: Kubernetes 13：CRD、Operator 与生产下一步
type: concept
status: seed
tags: [Kubernetes, CRD, Operator, Controller, GitOps, Production]
created: 2026-08-31
updated: 2026-08-31
source: Kubernetes 官方扩展文档与 roadmap.sh Kubernetes Roadmap
---

# Kubernetes 13：CRD、Operator 与生产下一步

## 一句话理解

> CRD 给 Kubernetes API 增加新的资源类型，Controller 负责调谐它，Operator 则把某个应用领域的运维知识编码进这套控制循环。

## Kubernetes 为什么可以扩展

内置 Deployment 控制器知道如何管理无状态副本，却不知道怎样升级 PostgreSQL 集群、扩容 Ceph、轮换证书或恢复某个业务系统。Kubernetes 允许用自定义对象表达领域意图：

```text
用户提交 DatabaseCluster（CR）
       ↓
API Server 按 CRD 校验和保存
       ↓ watch
Database Operator / Controller
       ↓ reconcile
StatefulSet、Service、PVC、Secret、备份任务、外部资源
       ↓
更新 DatabaseCluster status
```

## CRD 与 CR

- **CustomResourceDefinition（CRD）**：定义新 Kind、API group/version、schema、作用域等；
- **Custom Resource（CR）**：该类型的具体对象，例如 `DatabaseCluster/my-db`；
- CRD 只让 API Server 能保存和校验对象，不会自动产生业务行为；行为来自 Controller。

概念示例：

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: websites.example.com
spec:
  group: example.com
  scope: Namespaced
  names:
    plural: websites
    singular: website
    kind: Website
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                image:
                  type: string
                replicas:
                  type: integer
                  minimum: 1
```

生产 CRD 还要设计必填字段、默认值、status subresource、版本转换、打印列、删除 finalizer 和兼容策略。

## Controller 的调谐循环

一个健壮的 Controller 通常：

1. watch 关注对象及其子资源；
2. 读取期望 spec 和实际状态；
3. 幂等地创建/更新/删除资源；
4. 写入 status conditions；
5. 对临时失败做退避重试；
6. 用 ownerReferences 或 finalizers 管理生命周期；
7. 记录事件、日志和指标。

```text
reconcile(key):
  desired = read spec
  actual = observe cluster/external system
  diff = compare(desired, actual)
  apply(diff)        # 重复执行应安全
  update status
```

控制器可能因重试、进程重启或 watch 重连多次处理同一对象，所以幂等性比「只执行一次」假设更可靠。

## Operator 与普通自动化脚本的区别

Operator 通常以持续控制循环工作，并把领域状态暴露为 Kubernetes API；脚本多是一次性步骤。Operator 适合复杂、长期存在、需要持续修复的生命周期：

- 安装/配置；
- 扩缩容与成员变更；
- 版本升级；
- 备份和恢复；
- 证书/凭据轮换；
- 故障检测和自动修复。

不是所有应用都需要 Operator。简单 Deployment + Service + Helm 就能清晰管理时，新增 CRD 会引入 API 兼容、权限、升级和控制器运维成本。

## CSI 与 Operator 的联系

CSI 把存储的创建、挂载等通用能力标准化；存储 Operator 往往管理完整存储集群的生命周期。例如：

```text
CSI：Pod 如何申请并挂载卷
Operator：Ceph/数据库等系统如何部署、扩容、升级、自愈
```

两者可以协作，但职责不同。更多存储背景见 [分布式存储系统知识地图](../distributed-storage-knowledge-map.md)。

## 学完入门后的生产路线

### 1. 集群可靠性

- 控制面高可用、etcd 备份与恢复演练；
- 节点池、故障域、升级与版本偏差策略；
- PodDisruptionBudget、拓扑分布和优雅终止；
- 集群容量、配额和成本管理。

### 2. 交付与变更

- Helm/Kustomize 的可审查清单；
- CI 中 schema、策略和安全扫描；
- GitOps Controller 持续调谐 Git 与集群；
- 金丝雀、蓝绿发布和自动回滚依据；
- 数据库迁移与应用版本兼容策略。

### 3. 安全治理

- 身份联邦、最小 RBAC、审计日志；
- Pod Security、策略引擎、镜像签名/准入；
- Secret 静态加密、外部密钥系统和轮换；
- NetworkPolicy、egress 控制和供应链安全。

### 4. 可观测性与 SRE

- metrics/logs/traces/events 统一采集；
- 基于用户体验的 SLI/SLO 和告警；
- 控制面、节点、CNI、CSI、Ingress 的 runbook；
- 混沌/故障演练与容量压测。

### 5. 网络与存储进阶

- CNI 数据路径、eBPF、Gateway API、服务网格；
- CSI、卷快照、备份恢复、拓扑感知；
- StatefulSet 与数据库/存储 Operator；
- 多集群和跨区域流量/数据一致性。

## 一个合理的实践项目

为一个小型 API 构建以下闭环：

1. Deployment + Service + Ingress/Gateway；
2. ConfigMap/Secret + requests/limits + probes；
3. HPA + PodDisruptionBudget + topology spread；
4. RBAC + Pod Security + NetworkPolicy；
5. Helm/Kustomize 打包并进入 Git；
6. 日志/指标/告警和故障 runbook；
7. 若确有领域状态，再设计一个小 CRD/Controller；不要为了展示技术而强行 Operator 化。

## 何时开始写 Controller

先确认以下问题：

- 是否存在需要持续观察和纠偏的领域状态？
- 是否能保证 reconcile 幂等？
- API schema 将如何版本化和兼容？
- 删除时外部资源如何清理，finalizer 失败怎么办？
- status conditions 如何让用户判断进度和失败？
- Controller 需要哪些最小 RBAC 权限？
- 如何测试重试、重启、并发和部分失败？

常用开发生态包括 client-go、controller-runtime 和 Kubebuilder，但应先理解 API 与控制循环，再使用脚手架。

## 常见误区

- **创建 CRD 就有控制逻辑**：CRD 只定义和存储类型，Controller 才执行行为。
- **Operator 是高级 Helm Chart**：Helm 主要渲染/提交清单，Operator 持续观察并调谐领域状态。
- **所有应用都应该 Operator 化**：简单系统可能不值得承担自定义 API 和控制器成本。
- **Controller watch 到一次就结束**：重试、重连和状态变化会多次触发，应设计幂等。
- **finalizer 是删除保护开关**：它会阻塞对象最终删除，必须有可靠清理和故障恢复策略。
- **进生产只差一个 YAML**：可靠性、安全、备份、观测、升级和组织流程同样重要。

## 我的理解

CRD/Operator 揭示了 Kubernetes 最核心的可复用思想：把运维意图建模成 API，把操作经验变成可重复的控制循环。但扩展 API 也是长期承诺；只有领域状态确实需要持续调谐时，复杂度才值得。

## Related

- [02 集群架构与声明式控制循环](./02-architecture-and-control-loop.md)
- [08 PV、PVC、StorageClass 与 CSI](./08-storage-pv-pvc-and-csi.md)
- [11 Helm 与 Kubernetes 包管理](./11-helm-and-package-management.md)
- [12 可观测性与故障排查](./12-observability-and-troubleshooting.md)
- [分布式存储系统知识地图](../distributed-storage-knowledge-map.md)

## References

- [Custom Resources](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)
- [CustomResourceDefinition](https://kubernetes.io/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/)
- [Operator Pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)
- [Controllers](https://kubernetes.io/docs/concepts/architecture/controller/)
- [Kubebuilder Book](https://book.kubebuilder.io/)
