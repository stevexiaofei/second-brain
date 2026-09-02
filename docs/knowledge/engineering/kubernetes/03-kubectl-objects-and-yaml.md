---
title: Kubernetes 03：kubectl、对象与 YAML
type: concept
status: seed
tags: [Kubernetes, kubectl, YAML, API, Labels, Namespaces]
created: 2026-08-31
updated: 2026-08-31
source: Kubernetes 官方对象与 kubectl 文档、roadmap.sh Kubernetes Roadmap
---

# Kubernetes 03：kubectl、对象与 YAML

## 一句话理解

> Kubernetes 对象是 API 中的结构化意图；YAML 是一种表达对象的格式，`kubectl` 是读写这些对象的客户端。

## 一个对象的骨架

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
  namespace: k8s-lab
  labels:
    app.kubernetes.io/name: web
spec:
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
          ports:
            - name: http
              containerPort: 80
```

- `apiVersion`：这个资源使用的 API 组和版本；
- `kind`：资源类型；
- `metadata`：名字、命名空间、标签、注解、UID 等身份和附加信息；
- `spec`：用户期望的配置；
- `status`：Kubernetes 或控制器观察到的当前状态，通常不手写。

## API Group、Version 与 Kind

`apps/v1` 中 `apps` 是 API group，`v1` 是版本；`Deployment` 是 Kind。核心 API 组通常写成 `v1`（省略 group），例如 Pod、Service、ConfigMap。

```bash
kubectl api-resources
kubectl explain deployment.spec
kubectl explain deployment.spec.template.spec.containers
kubectl api-versions
```

API 版本不是随意的字符串。新资源应查当前集群支持的稳定版本；不要把旧教程中的 `extensions/v1beta1` 直接复制到新集群。

## `kubectl` 的日常分类

### 查状态

```bash
kubectl get pods
kubectl get pods -o wide
kubectl get deployment web -o yaml
kubectl get all
kubectl get events --sort-by=.lastTimestamp
```

### 创建与变更

```bash
kubectl apply -f deployment.yaml
kubectl diff -f deployment.yaml
kubectl label pod <pod-name> purpose=debug
kubectl annotate deployment web owner=platform
```

`apply` 的重点是让对象趋近文件中声明的状态；它适合把 YAML 纳入版本控制。直接使用 `kubectl create` 是一次性命令式创建，适合实验或生成初始模板，但不应替代可审查的清单。

### 删除与调试

```bash
kubectl delete -f deployment.yaml
kubectl describe pod <pod-name>
kubectl logs <pod-name> -c web
kubectl exec -it <pod-name> -c web -- sh
```

删除对象前确认命名空间和 context，避免在错误集群中操作。

## Context 与 Namespace

`kubectl` 会根据 kubeconfig 的 current-context 决定连接哪个集群、使用哪个身份和默认命名空间：

```bash
kubectl config get-contexts
kubectl config current-context
kubectl config use-context kind-dev
kubectl get namespaces
kubectl -n k8s-lab get pods
```

安全习惯：

- 在破坏性命令前显式写 `--context` 和 `--namespace`；
- 给 shell prompt 显示当前 context；
- 不把生产 kubeconfig 和实验 kubeconfig 混在无保护的位置；
- 使用 RBAC 限制身份，而不是依赖「小心一点」。

## Labels、Selectors 与 Annotations

### Labels：可筛选的身份

Labels 是键值对，用于分组和选择对象：

```yaml
metadata:
  labels:
    app.kubernetes.io/name: web
    app.kubernetes.io/part-of: checkout
    app.kubernetes.io/version: "1.0"
```

```bash
kubectl get pods -l app.kubernetes.io/name=web
kubectl get pods -l 'app.kubernetes.io/name in (web,api)'
```

### Selector：控制器和 Service 的连接点

Deployment 的 `spec.selector` 必须匹配 Pod template 的 labels。Service 的 selector 决定它把流量送到哪些 Pod。标签设计错误时，对象可能都显示 Running，但 Service 没有 endpoints 或 Deployment 无法管理预期 Pod。

### Annotations：给工具的非筛选元数据

Annotations 适合存放构建信息、Ingress 配置提示、拥有者链接等较大或不用于选择的元数据。不要把高频变化数据塞进 annotations；它们会增加对象更新和审计负担。

## 声明式清单的工作流

```text
编辑 YAML → kubectl diff → review → kubectl apply → get/describe/events → git 记录
```

典型命令：

```bash
kubectl apply --dry-run=client -f deployment.yaml -o yaml
kubectl diff -f deployment.yaml
kubectl apply -f deployment.yaml
kubectl rollout status deployment/web
```

`--dry-run=client` 主要检查客户端侧的对象结构和生成结果；它不证明服务器准入、镜像存在、调度资源充足或应用可用。

## 一个最小 Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
  labels:
    app.kubernetes.io/name: web
spec:
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
          ports:
            - name: http
              containerPort: 80
```

保存为 `deployment.yaml` 后：

```bash
kubectl create namespace k8s-lab
kubectl apply -n k8s-lab -f deployment.yaml
kubectl -n k8s-lab get deployment,pods --show-labels
```

## 常见误区

- **所有 YAML 都能直接 apply**：API 版本、字段、准入策略和集群版本可能不兼容。
- **`containerPort` 自动开放端口**：它主要是元数据/声明，网络可达性还要由应用监听、Service、策略等共同决定。
- **标签随便写**：selector 是控制器关系的关键，修改 selector 可能破坏资源管理。
- **`kubectl get all` 等于所有资源**：它只覆盖一组常见资源，不是 API 中所有 Kind。
- **`status` 应由用户维护**：status 通常由控制器或 kubelet 写回。
- **apply 成功等于应用成功**：apply 只说明 API 接受了对象，需继续看 rollout、探针、日志和业务请求。

## 我的理解

YAML 不是 Kubernetes 的本质，API 对象才是；`kubectl` 也不是魔法命令，而是对 API 的可视化入口。掌握 `spec/status`、labels/selectors 和 context，就拥有了阅读大多数 Kubernetes 配置的通用语法。

## Related

- [02 集群架构与声明式控制循环](./02-architecture-and-control-loop.md)
- [04 Pod 与多容器模式](./04-pods-and-container-patterns.md)
- [05 Workload：Deployment、StatefulSet 与 Job](./05-workloads-deployments-and-jobs.md)

## References

- [Objects In Kubernetes](https://kubernetes.io/docs/concepts/overview/working-with-objects/)
- [Object Management](https://kubernetes.io/docs/concepts/overview/working-with-objects/object-management/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [Labels and Selectors](https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/)
