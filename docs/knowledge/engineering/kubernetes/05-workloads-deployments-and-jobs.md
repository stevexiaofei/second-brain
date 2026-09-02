---
title: Kubernetes 05：Workload、Deployment、StatefulSet 与 Job
type: concept
status: seed
tags: [Kubernetes, Workload, Deployment, StatefulSet, DaemonSet, Job]
created: 2026-08-31
updated: 2026-08-31
source: Kubernetes 官方 Workloads 文档与 roadmap.sh Kubernetes Roadmap
---

# Kubernetes 05：Workload、Deployment、StatefulSet 与 Job

## 一句话理解

> Workload 资源把「一组 Pod 应该如何运行」编码成控制策略；Deployment 管理无状态副本，StatefulSet 管理有身份的副本，Job 管理有终点的任务。

直接创建裸 Pod 只能启动一次实例，不能可靠表达副本、更新、回滚和失败重试。生产应用通常由 Workload 控制器间接管理 Pod。

## Workload 选择表

| 资源 | 适合的问题 | 关键语义 |
|---|---|---|
| Deployment | 无状态 Web/API/Worker | ReplicaSet、滚动更新、回滚 |
| StatefulSet | 有稳定身份或持久卷的副本 | 有序创建、稳定网络身份、每副本存储 |
| DaemonSet | 每个节点需要一个 Pod | 节点级 agent、日志、网络插件 |
| Job | 一次性任务 | 成功次数、失败重试、完成后结束 |
| CronJob | 周期性任务 | 按计划创建 Job |
| ReplicaSet | 直接维持 Pod 副本 | 通常由 Deployment 间接管理 |

## Deployment：无状态服务的默认入口

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app.kubernetes.io/name: web
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    metadata:
      labels:
        app.kubernetes.io/name: web
    spec:
      containers:
        - name: web
          image: nginx:1.27
          ports:
            - name: http
              containerPort: 80
          readinessProbe:
            httpGet:
              path: /
              port: http
```

Deployment 控制器通过 ReplicaSet 管理 Pod。修改 `spec.template`（例如镜像）会创建新的 ReplicaSet，并按更新策略逐步替换旧 Pod。

```bash
kubectl -n k8s-lab apply -f deployment.yaml
kubectl -n k8s-lab rollout status deployment/web
kubectl -n k8s-lab set image deployment/web web=nginx:1.28
kubectl -n k8s-lab rollout history deployment/web
kubectl -n k8s-lab rollout undo deployment/web
kubectl -n k8s-lab scale deployment/web --replicas=5
```

滚动更新不会自动证明业务正确。readiness、错误率、延迟和关键业务指标都应参与发布判断。

## ReplicaSet 与 selector

ReplicaSet 的职责是「符合 selector 的 Pod 数量达到 replicas」。不要手工创建一个与 Deployment selector 重叠的 ReplicaSet，否则两个控制器可能争抢同一批 Pod。Deployment 通常是用户管理无状态发布的抽象，ReplicaSet 是其实现层。

## StatefulSet：身份比副本更重要

StatefulSet 适合需要以下语义的应用：

- 稳定且可预测的 Pod 名称，如 `db-0`、`db-1`；
- 每个副本对应独立的持久卷；
- 有序创建、更新或终止；
- 集群成员需要知道彼此身份。

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: web
spec:
  serviceName: web-headless
  replicas: 2
  selector:
    matchLabels:
      app.kubernetes.io/name: web
  template:
    metadata:
      labels:
        app.kubernetes.io/name: web
    spec:
      containers:
        - name: web
          image: nginx:1.27
          volumeMounts:
            - name: data
              mountPath: /var/lib/example
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 1Gi
```

StatefulSet 不会自动把数据库变成高可用数据库。复制协议、故障转移、备份、恢复和数据一致性仍由应用或 Operator 负责。

## DaemonSet：每节点一份

DaemonSet 会在符合条件的节点上运行一个 Pod，常用于节点日志 agent、监控 agent、网络插件和存储插件。节点 taint、容忍度、资源压力和升级策略仍可能影响实际运行数量。

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-agent
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: node-agent
  template:
    metadata:
      labels:
        app.kubernetes.io/name: node-agent
    spec:
      containers:
        - name: agent
          image: example/agent:1.0
```

## Job 与 CronJob：有终点的控制器

Job 关注「成功完成多少次」，不是永远保持 Pod 在线：

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: data-migration
spec:
  backoffLimit: 3
  completions: 1
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: migrate
          image: example/migrator:1.0
          command: ["/app/migrate"]
```

CronJob 按 schedule 创建 Job：

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: nightly-report
spec:
  schedule: "0 2 * * *"
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
            - name: report
              image: example/report:1.0
```

任务必须考虑幂等性、超时、重复执行、时区和失败重试。CronJob 触发成功不等于任务业务成功。

## Workload 的通用调试路径

```bash
kubectl -n k8s-lab get deployment,rs,pods
kubectl -n k8s-lab describe deployment web
kubectl -n k8s-lab get pods -l app.kubernetes.io/name=web
kubectl -n k8s-lab rollout status deployment/web
kubectl -n k8s-lab get events --sort-by=.lastTimestamp
```

看不懂一个 Pod 为什么存在时，沿着「Workload → ReplicaSet → Pod」向下追；看不懂它为什么被创建时，回到模板和 selector。

## 常见误区

- **所有应用都用 Deployment**：有稳定身份、每节点副本或任务终点时应选择不同 Workload。
- **StatefulSet 自带数据库高可用**：它提供身份和编排语义，不实现数据库复制协议。
- **扩容 replicas 就一定有更多可用实例**：资源不足、探针失败、镜像拉取失败都可能阻止可用。
- **更新镜像后 rollout 完成就安全**：还要观察业务指标和数据迁移兼容性。
- **CronJob 是精确计时器**：控制面故障、并发策略和任务执行时间会影响实际行为。
- **手动编辑 Pod 解决 Deployment 问题**：应修改 Deployment 模板，否则控制器会覆盖临时改动。

## 我的理解

Workload 的核心是把应用的生命周期语义显式化：Deployment 表达「可替换的同质副本」，StatefulSet 表达「有身份的副本」，Job 表达「最终要完成的工作」。资源类型选错，后续补再多 YAML 也很难得到正确行为。

## Related

- [04 Pod 与多容器模式](./04-pods-and-container-patterns.md)
- [06 Service、Ingress、DNS 与网络策略](./06-services-networking-ingress-and-dns.md)
- [08 PV、PVC、StorageClass 与 CSI](./08-storage-pv-pvc-and-csi.md)
- [12 可观测性与故障排查](./12-observability-and-troubleshooting.md)

## References

- [Workloads](https://kubernetes.io/docs/concepts/workloads/)
- [Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [StatefulSets](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/)
- [DaemonSets](https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/)
- [Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
- [CronJob](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/)
