---
title: Kubernetes 12：可观测性与故障排查
type: experience
status: seed
tags: [Kubernetes, Observability, Troubleshooting, Logs, Metrics, Events]
created: 2026-08-31
updated: 2026-08-31
source: Kubernetes 官方排障与可观测性文档、roadmap.sh Kubernetes Roadmap
---

# Kubernetes 12：可观测性与故障排查

## 一句话理解

> Kubernetes 排障的关键不是记命令，而是沿「对象意图 → 控制器 → 调度 → 节点 → 容器 → 网络/存储 → 业务」逐层收集事件、状态、日志和指标。

## Environment

以下命令假设：

- 已确认 `kubectl config current-context` 指向正确集群；
- 对目标命名空间有只读或调试权限；
- 集群可能使用不同的 CNI、CSI、Ingress Controller 和监控系统，组件名应按实际环境替换。

生产排障先保留证据，再做重启/删除。直接删除 Pod 可能暂时恢复服务，却会丢失现场并掩盖根因。

## 四类证据

| 信号 | 回答的问题 | 常用入口 |
|---|---|---|
| 状态 | 对象现在是什么状态 | `kubectl get -o wide/yaml` |
| 事件 | 控制面/节点最近尝试了什么 | `kubectl describe`、`kubectl get events` |
| 日志 | 应用或组件说了什么 | `kubectl logs` |
| 指标 | 资源和行为随时间如何变化 | Metrics Server、Prometheus、平台监控 |
| Trace | 一个请求经过哪些服务 | OpenTelemetry/Tracing 系统 |

事件有保留期限，日志也可能随容器重建而丢失；生产系统需要集中采集，而不是事后只依赖 `kubectl`。

## 通用排障路径

### 1. 确认范围和目标

```bash
kubectl config current-context
kubectl config view --minify
kubectl -n k8s-lab get deployment,rs,pods,service,endpointslice,ingress -o wide
```

明确是单 Pod、单节点、单命名空间、单服务还是整个集群。范围判断错误会浪费大量时间。

### 2. 看对象状态与事件

```bash
kubectl -n k8s-lab describe pod <pod-name>
kubectl -n k8s-lab get events --sort-by=.lastTimestamp
kubectl -n k8s-lab get pod <pod-name> -o yaml
```

关注 Conditions、Reason、Message、容器状态、restartCount、调度节点、探针和 ownerReferences。

### 3. 看当前和上一次容器日志

```bash
kubectl -n k8s-lab logs <pod-name> -c <container>
kubectl -n k8s-lab logs <pod-name> -c <container> --previous
kubectl -n k8s-lab logs deployment/web --all-containers --tail=200
```

`--previous` 对 CrashLoopBackOff 特别重要，因为当前容器可能还没有输出根因。

### 4. 在容器或临时调试环境验证

```bash
kubectl -n k8s-lab exec -it <pod-name> -c <container> -- sh
kubectl -n k8s-lab port-forward service/web 8080:80
kubectl -n k8s-lab run curl --rm -it --image=curlimages/curl:8.10.1 -- sh
```

镜像可能遵循最小化原则而没有 shell/debug 工具。不要为了排障永久把大量工具装进生产镜像；可使用临时 Pod 或 ephemeral container，并受 RBAC/审计约束。

## 症状到路径

### Pod Pending

1. `describe pod` 查 `FailedScheduling`；
2. 检查 requests、节点 allocatable、taints/tolerations、affinity、PVC；
3. 检查 ResourceQuota、LimitRange 和 Cluster Autoscaler；
4. 不要通过反复删除 Pod 期望资源凭空出现。

### ImagePullBackOff

检查镜像名/tag/digest、registry 网络、认证 Secret、ServiceAccount 的 `imagePullSecrets`、节点 DNS/代理和仓库限流。

### CrashLoopBackOff

```bash
kubectl logs <pod> -c <container> --previous
kubectl describe pod <pod>
```

检查命令/参数、退出码、配置文件、Secret、依赖、权限、探针和 OOM。CrashLoopBackOff 是退避状态，不是根因。

### Running 但 Service 不通

```text
进程监听地址/端口
  → readiness
  → Pod labels
  → Service selector
  → EndpointSlice
  → Service port/targetPort
  → NetworkPolicy
  → Ingress/Gateway/DNS
```

逐跳测试 Pod IP（仅诊断）、Service 名称和 Ingress host，确定在哪一层断开。

### PVC Pending 或挂载失败

检查 PVC event、StorageClass、PV、CSI controller/node plugin、节点拓扑、访问模式、attach/mount event 和后端存储状态。

### OOMKilled / CPU 延迟高

```bash
kubectl top pod -n k8s-lab
kubectl top node
kubectl describe pod <pod>
```

`kubectl top` 是当前快照，不替代历史指标。检查 limit、request、应用内存曲线、GC、并发和节点压力；CPU limit 还可能带来 throttling。

## 从 Workload 向下追 owner

```bash
kubectl -n k8s-lab get pod <pod> -o jsonpath='{.metadata.ownerReferences}'
kubectl -n k8s-lab get rs
kubectl -n k8s-lab describe deployment web
kubectl -n k8s-lab rollout status deployment/web
```

不要只修 Pod 表象。Pod 由 ReplicaSet 管、ReplicaSet 由 Deployment 管；真实配置源在上层模板或 Git/Helm Release。

## 可观测性基础架构

```text
应用 stdout/stderr ─→ 节点日志 ─→ 日志采集器 ─→ 日志存储/查询
应用/组件 metrics ─→ Prometheus 等 ─→ 告警 / Dashboard
请求 trace        ─→ Collector    ─→ Trace 后端
Kubernetes events ─→ API / 导出器 ─→ 事件存档
```

重点指标包括：

- Deployment 可用副本、Pod restart、调度失败；
- CPU/内存使用与 requests/limits；
- API Server 延迟/错误、etcd 延迟和容量；
- 节点 NotReady、磁盘/内存/PID 压力；
- Service 请求率、错误率、延迟和饱和度；
- HPA 当前指标、目标和副本变化。

业务 SLI/SLO 比「Pod 是绿色」更接近用户体验。

## Root Cause 记录模板

排障后不要只写「重启后恢复」。建议记录：

- **Environment**：集群/命名空间/版本/工作负载；
- **Symptoms**：开始时间、影响范围、可观察信号；
- **Investigation**：按时间保存关键事件、日志和指标；
- **Root Cause**：哪个条件触发了什么机制；
- **Solution**：立即缓解和永久修复；
- **Why**：为什么此前策略/测试/告警没发现；
- **Lessons Learned**：新增的监控、测试、runbook 和防护。

## 常见误区

- **CrashLoopBackOff 是错误原因**：它只是 kubelet 的重启退避结果。
- **先重启再看日志**：可能破坏现场；先保存状态、事件和 previous logs。
- **`kubectl get pods` 足够**：Running 不代表 Ready、可路由或业务健康。
- **只看应用日志**：调度、CNI、CSI、kubelet 和控制器故障可能发生在应用启动前。
- **指标正常就没有故障**：采集可能缺失，聚合也可能隐藏单个节点/租户问题。
- **临时修改 Pod 是永久修复**：上层控制器或 GitOps 会恢复原状态。

## Lessons Learned

最有效的 Kubernetes 排障习惯是「缩小范围、沿对象关系走、用事件解释状态、用日志解释进程、用指标解释时间」。命令只是收集证据的方式；保持资源 owner、labels、变更记录和统一日志比多背命令更重要。

## Related

- [05 Workload：Deployment、StatefulSet 与 Job](./05-workloads-deployments-and-jobs.md)
- [06 Service、Ingress、DNS 与网络策略](./06-services-networking-ingress-and-dns.md)
- [08 PV、PVC、StorageClass 与 CSI](./08-storage-pv-pvc-and-csi.md)
- [09 资源、调度与自动扩缩容](./09-resources-scheduling-and-autoscaling.md)

## References

- [Troubleshooting Applications](https://kubernetes.io/docs/tasks/debug/debug-application/)
- [Troubleshooting Clusters](https://kubernetes.io/docs/tasks/debug/debug-cluster/)
- [Observability](https://kubernetes.io/docs/concepts/cluster-administration/observability/)
- [Logging Architecture](https://kubernetes.io/docs/concepts/cluster-administration/logging/)
- [Resource Metrics Pipeline](https://kubernetes.io/docs/tasks/debug/debug-cluster/resource-metrics-pipeline/)
