---
title: Kubernetes 04：Pod 与多容器模式
type: concept
status: seed
tags: [Kubernetes, Pod, Containers, Probes, Init Container]
created: 2026-08-31
updated: 2026-08-31
source: Kubernetes 官方 Pod 文档与 roadmap.sh Kubernetes Roadmap
---

# Kubernetes 04：Pod 与多容器模式

## 一句话理解

> Pod 是 Kubernetes 的最小调度和部署单元：一个或多个必须共同运行的容器共享网络命名空间、IP、端口空间和声明的卷。

## 为什么不是直接调度容器

调度器需要把一组紧密耦合的容器放在同一节点、同一生命周期中。例如主应用和日志代理可能需要共享文件卷；主应用和本地代理需要通过 `localhost` 通信。Pod 把这些容器及其共享资源包装成一个调度单位。

```text
Node
└── Pod（一个 IP / 一个生命周期）
    ├── app container
    ├── sidecar container（可选）
    └── shared volumes（可选）
```

多数场景每个 Pod 一个主容器即可。不要因为「可以多容器」就把整个系统塞进一个 Pod。

## Pod 的网络模型

同一 Pod 内的容器：

- 共享一个网络命名空间；
- 共享 Pod IP；
- 可以通过 `localhost` 互相访问；
- 不能监听相同端口；
- 对外访问通常通过 Service，而不是直接暴露 Pod IP。

```yaml
containers:
  - name: app
    image: example/app:1.0
    ports:
      - containerPort: 8080
  - name: proxy
    image: example/proxy:1.0
    # proxy 可通过 localhost:8080 访问 app
```

Pod 内共享网络不等于跨 Pod 共享网络；跨 Pod 通信要经过 Pod 网络和策略。

## Pod 生命周期与状态

典型状态包括 `Pending`、`Running`、`Succeeded`、`Failed`、`Unknown`。`Running` 只表示 Pod 已绑定节点且至少有容器被启动，不代表业务请求一定成功。

```bash
kubectl get pod web-xxx -o wide
kubectl describe pod web-xxx
kubectl get pod web-xxx -o jsonpath='{.status.phase}{"\n"}'
```

容器状态还包括 `Waiting`、`Running`、`Terminated`，退出码和 reason 对排障很重要：

- `CrashLoopBackOff`：容器反复退出，kubelet 使用退避时间重启；
- `ImagePullBackOff`：镜像拉取失败并退避重试；
- `OOMKilled`：容器被内存限制或节点压力相关机制终止；
- `Completed`：一次性容器正常退出，Job 场景中可能是成功。

## Init Container

Init container 在应用容器之前按顺序运行，全部成功后才启动主容器。适合：

- 等待或检查依赖服务；
- 生成初始配置；
- 执行一次性的准备工作；
- 以不同权限完成初始化，再让主容器使用较小权限运行。

```yaml
spec:
  initContainers:
    - name: check-config
      image: busybox:1.36
      command: ["sh", "-c", "test -f /config/app.conf"]
      volumeMounts:
        - name: config
          mountPath: /config
  containers:
    - name: app
      image: example/app:1.0
      volumeMounts:
        - name: config
          mountPath: /config
  volumes:
    - name: config
      configMap:
        name: app-config
```

Init container 失败会阻止主容器启动；应让检查逻辑有明确超时和可观察输出，避免永久 Pending。

## Probes：告诉系统如何判断健康

### Startup probe

给启动慢的应用准备时间。startup probe 成功前，liveness/readiness probe 不会接管判断。

### Readiness probe

判断 Pod 是否应该接收流量。失败通常会让 Pod 从 Service endpoints 中暂时移除，但不一定重启容器。

### Liveness probe

判断应用是否需要被重启。配置过于激进会把短暂的高负载或下游故障放大成重启风暴。

```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: http
  initialDelaySeconds: 10
  periodSeconds: 10
readinessProbe:
  httpGet:
    path: /ready
    port: http
  periodSeconds: 5
startupProbe:
  httpGet:
    path: /startup
    port: http
  failureThreshold: 30
  periodSeconds: 5
```

健康端点应反映正确的语义：readiness 可以检查应用是否准备接流量；liveness 不应简单地把所有外部依赖故障都判成「进程死了」。

## 多容器模式

| 模式 | 关系 | 例子 |
|---|---|---|
| Sidecar | 辅助主容器 | 日志转发、代理、配置刷新 |
| Ambassador | 代表主容器访问外部 | 本地代理、协议适配 |
| Adapter | 把输出转换为统一格式 | 指标格式转换 |
| Init | 只在启动阶段运行 | 初始化数据、等待依赖 |

多容器会共享资源和发布节奏，也会增加 CPU、内存、日志和排障成本。若两个组件可以独立扩缩容、独立发布，通常应该拆成不同 Pod，通过 Service 通信。

## Pod 模板与不可变性

Deployment、StatefulSet 等 Workload 的 `spec.template` 是 Pod 模板。修改模板通常会触发新 ReplicaSet 或新 Pod；直接修改正在运行的 Pod 往往不可持续，因为控制器会按模板恢复它。

## 最小 Pod 示例

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: debug-shell
spec:
  containers:
    - name: shell
      image: busybox:1.36
      command: ["sh", "-c", "sleep 3600"]
```

实验后删除：

```bash
kubectl apply -n k8s-lab -f pod.yaml
kubectl exec -n k8s-lab -it debug-shell -- sh
kubectl delete -n k8s-lab pod debug-shell
```

生产应用通常不直接创建裸 Pod，而应交给 Workload 资源管理。

## 常见误区

- **Pod 是虚拟机**：它是共享节点内核的一组容器，不提供完整 VM 边界。
- **Pod 永远存在**：Pod 是相对短暂的对象，故障或发布时会被替换。
- **Running 等于可用**：还要看 readiness、Service endpoints、业务请求和依赖。
- **liveness 检查越多越好**：错误探针会造成重启循环。
- **多容器就是微服务**：微服务通常以多个 Pod/Service 独立演进；Pod 多容器强调强耦合共址。
- **直接修改 Pod 能持久生效**：由 Workload 管理时，应修改模板源对象。

## 我的理解

Pod 的关键不是「里面装了几个容器」，而是它定义了一个共享的故障和调度边界。把必须同生共死的东西放进 Pod，把需要独立生命周期的东西拆开，是 Kubernetes 建模的第一道架构判断。

## Related

- [03 kubectl、对象与 YAML](./03-kubectl-objects-and-yaml.md)
- [05 Workload：Deployment、StatefulSet 与 Job](./05-workloads-deployments-and-jobs.md)
- [06 Service、Ingress、DNS 与网络策略](./06-services-networking-ingress-and-dns.md)

## References

- [Pods](https://kubernetes.io/docs/concepts/workloads/pods/)
- [Pod Lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)
- [Init Containers](https://kubernetes.io/docs/concepts/workloads/pods/init-containers/)
- [Configure Liveness, Readiness and Startup Probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
