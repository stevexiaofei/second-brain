---
title: Kubernetes 09：资源、调度与自动扩缩容
type: concept
status: seed
tags: [Kubernetes, Resources, Scheduling, Requests, Limits, HPA, Autoscaling]
created: 2026-08-31
updated: 2026-08-31
source: Kubernetes 官方资源管理与调度文档、roadmap.sh Kubernetes Roadmap
---

# Kubernetes 09：资源、调度与自动扩缩容

## 一句话理解

> requests 影响 Pod 能否被放到某个节点，limits 约束容器可使用的上限；调度决定「放哪儿」，Autoscaler 决定「要多少副本或节点」。

## requests 与 limits

```yaml
resources:
  requests:
    cpu: "100m"
    memory: "128Mi"
  limits:
    cpu: "500m"
    memory: "512Mi"
```

- **CPU request**：调度器用来计算节点可分配容量；CPU 超过 limit 通常会被 throttling；
- **内存 request**：调度器的容量依据；
- **内存 limit**：容器可使用的上限，超出时可能被 OOMKilled；
- 未设置资源不能简单理解为无限制，实际还受节点、LimitRange、配额和运行时行为影响。

request 应接近正常或目标负载，limit 应根据延迟、峰值和故障行为评估。对延迟敏感应用，过低 CPU limit 可能造成 throttling；对内存，盲目把 limit 设很大可能让节点在压力时更晚暴露问题。

## QoS 类别

Kubernetes 会根据 Pod 中容器的资源声明归类 QoS：Guaranteed、Burstable、BestEffort。它影响节点内存压力下的驱逐排序，但不是「服务等级协议」，也不替代合理的容量规划。

## 调度流程

```text
Pending Pod
  ↓ filter：资源、taint、亲和性、拓扑等是否满足
候选节点
  ↓ score：按优先级和策略评分
绑定 Node
  ↓ kubelet 创建 Pod
```

```bash
kubectl get pod <pod-name> -o wide
kubectl describe pod <pod-name>  # 查看 FailedScheduling 事件
kubectl describe node <node-name>
```

Pod 一直 Pending 时，先看事件中的具体约束，而不是直接删除重建。

## 选择节点的常用机制

### nodeSelector / nodeAffinity

```yaml
spec:
  nodeSelector:
    kubernetes.io/os: linux
```

更复杂的条件用 node affinity，例如优先选择带 SSD 标签的节点。标签必须由可信的节点管理流程维护，否则调度约束可能被误用。

### Taints 与 Tolerations

Taint 是节点发出的「不要把普通 Pod 放到这里」信号；Toleration 表示 Pod 可以承受该 taint，但不等于它一定会被调度到该节点。

### Pod affinity / anti-affinity 与拓扑分布

它们可以让副本靠近依赖或分散到不同节点/可用区。强约束提高可用性或局部性，但也可能在容量不足时让 Pod 永远 Pending。生产配置要在可靠性与可调度性之间取平衡。

## Namespace 配额

ResourceQuota 限制命名空间总资源，LimitRange 为容器设置默认或最小/最大资源。二者能防止单个团队耗尽集群，但要配合监控和清晰的资源归属。

```bash
kubectl -n k8s-lab get resourcequota,limitrange
```

## HPA：按指标调整副本

Horizontal Pod Autoscaler 通常根据 CPU、内存或自定义指标调整 Workload 的 replicas：

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web
  minReplicas: 2
  maxReplicas: 10
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

CPU 利用率通常相对于 request 计算；没有合理 request，HPA 可能无法得到预期结果。HPA 还依赖 Metrics Server 或其他指标适配器，扩容也受节点容量和应用启动速度限制。

```bash
kubectl -n k8s-lab get hpa
kubectl -n k8s-lab describe hpa web
```

## VPA 与 Cluster Autoscaler

- **VPA** 调整 Pod 的资源建议或请求，可能需要重建 Pod；不应与 HPA 对同一资源指标无规划地同时控制；
- **Cluster Autoscaler** 在 Pod 因资源不足无法调度时增加节点，在空闲时缩减节点；它依赖云 provider 或节点组能力；
- 三者解决不同层次问题：Pod 副本、Pod 资源规格、集群节点数量。

```text
业务负载 → HPA（副本数）→ 调度器 → Cluster Autoscaler（节点数）
                 ↑
              VPA（资源规格，需协调）
```

## 常见误区

- **request 是应用保证得到的实际 CPU**：它是调度容量和计量依据，不保证应用永远获得该性能。
- **limit 越大越好**：可能掩盖内存问题、破坏节点可预测性或导致更大范围 OOM。
- **有 HPA 就能无限扩容**：上限、节点容量、配额、镜像拉取和下游瓶颈都会限制扩容。
- **toleration 把 Pod 放到 tainted 节点**：它只移除一个排斥条件，不提供偏好。
- **CPU 70% 是固定事实**：阈值必须结合业务延迟、吞吐和启动时间调优。
- **调度失败靠重启解决**：应从 FailedScheduling 事件找出资源或约束冲突。

## 我的理解

资源与调度把「应用想要什么」翻译成「集群如何分配有限资源」。Autoscaling 不是魔法，而是反馈控制：指标有延迟，扩缩有成本，应用启动和下游容量也有滞后。稳定系统需要同时设计资源声明、调度约束、指标质量和容量上限。

## Related

- [02 集群架构与声明式控制循环](./02-architecture-and-control-loop.md)
- [05 Workload：Deployment、StatefulSet 与 Job](./05-workloads-deployments-and-jobs.md)
- [12 可观测性与故障排查](./12-observability-and-troubleshooting.md)

## References

- [Resource Management for Pods and Containers](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)
- [Assign Pods to Nodes](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/)
- [Taints and Tolerations](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/)
- [Horizontal Pod Autoscaling](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Vertical Pod Autoscaling](https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler)
