---
title: Kubernetes 06：Service、Ingress、DNS 与网络策略
type: concept
status: seed
tags: [Kubernetes, Service, Ingress, DNS, Networking, NetworkPolicy]
created: 2026-08-31
updated: 2026-08-31
source: Kubernetes 官方网络文档与 roadmap.sh Kubernetes Roadmap
---

# Kubernetes 06：Service、Ingress、DNS 与网络策略

## 一句话理解

> Pod 是会变化的后端实例，Service 提供稳定的服务发现入口，Ingress 或 Gateway 把外部 HTTP 路由到 Service，DNS 让这些入口可以用名称访问。

## Pod IP 为什么不够

Pod 可能因更新、故障或驱逐被替换，IP 也随之变化。客户端不应维护 Pod IP 列表；Service 根据 selector 选择后端，并提供稳定的虚拟 IP 和 DNS 名称。

```text
client → Service（稳定入口）→ EndpointSlice → ready Pod（动态后端）
```

Service 只把流量送到符合 selector 且通常已通过 readiness 的后端。若 Service 没有 endpoints，先检查 selector、Pod labels、Pod readiness 和端口。

## Service 的端口关系

```yaml
apiVersion: v1
kind: Service
metadata:
  name: web
spec:
  type: ClusterIP
  selector:
    app.kubernetes.io/name: web
  ports:
    - name: http
      port: 80
      targetPort: http
```

- `port`：Service 对内提供的端口；
- `targetPort`：转发到 Pod 的端口，可引用容器的 named port；
- `nodePort`：NodePort 类型在节点上暴露的端口；
- `containerPort`：Pod 模板中的端口声明，不单独创建可达路径。

常见类型：

| 类型 | 典型用途 |
|---|---|
| ClusterIP | 集群内部稳定访问，默认类型 |
| NodePort | 通过每个节点的端口暴露服务，常用于简单实验 |
| LoadBalancer | 请求云或集群实现分配外部负载均衡 |
| ExternalName | 用 DNS CNAME 把服务名映射到外部名称，不创建代理端点 |

```bash
kubectl -n k8s-lab get service web
kubectl -n k8s-lab get endpointslice -l kubernetes.io/service-name=web
kubectl -n k8s-lab describe service web
```

## 集群 DNS

集群 DNS 通常为 Service 创建类似以下名称：

```text
web.k8s-lab.svc.cluster.local
```

同一命名空间内通常可直接使用 `web`；跨命名空间应使用 `web.k8s-lab` 或完整名称。Headless Service（`clusterIP: None`）不提供单一虚拟 IP，而是返回后端 Pod 地址，常与 StatefulSet 的稳定身份配合。

DNS 解析成功只代表名称有记录，不代表应用健康：继续检查 endpoints、端口、网络策略和应用日志。

## Ingress：HTTP 路由规则

Ingress 描述将 HTTP/HTTPS 请求按 host/path 路由到 Service 的规则，但必须有 Ingress Controller 才会真正处理流量：

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web
spec:
  ingressClassName: nginx
  rules:
    - host: web.example.test
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: web
                port:
                  name: http
```

Ingress 不是自动获得公网 IP 的魔法资源。需要安装并配置 controller、入口地址、TLS 证书和 DNS；新项目也应了解 Gateway API，它对流量角色和协议扩展表达得更清晰。

## NetworkPolicy：限制 Pod 流量

NetworkPolicy 通过 selector 描述允许的 ingress/egress 流量，实际执行依赖支持 NetworkPolicy 的 CNI 插件：

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-only-from-web
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: api
  policyTypes: [Ingress]
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app.kubernetes.io/name: web
      ports:
        - protocol: TCP
          port: 8080
```

一旦某方向进入隔离状态，未被规则允许的流量可能被拒绝。应先画清流量矩阵，再逐步收紧策略，并验证 DNS、监控和运维通道不会被意外阻断。

## 端到端访问链路

```text
浏览器
  ↓ DNS / LoadBalancer
Ingress Controller
  ↓ host/path 路由
Service ClusterIP
  ↓ selector / EndpointSlice
ready Pod IP:containerPort
  ↓
应用进程
```

排查时逐跳验证，不要直接修改一堆端口：

```bash
kubectl -n k8s-lab get ingress,service,endpointslice,pods -o wide
kubectl -n k8s-lab run curl --rm -it --image=curlimages/curl:8.10.1 -- sh
# 在临时调试 Pod 内：curl -v http://web
```

## 常见误区

- **创建 Service 就能从公网访问**：ClusterIP 只在集群内可达；外部入口还需要 NodePort、LoadBalancer、Ingress 或 Gateway。
- **Ingress 自带 controller**：Ingress 是规则对象，controller 是执行规则的实现。
- **Service selector 选中了 Deployment**：selector 通常匹配 Pod labels，不是 Deployment 名称。
- **DNS 通了就说明服务通了**：还要验证 endpoints、端口、协议和策略。
- **NetworkPolicy 默认全部生效**：没有兼容实现的网络插件时，策略可能不执行；不同方向和命名空间规则也要分别分析。
- **`targetPort` 必须是数字**：可以使用 Pod 模板中定义的 named port，命名更不易错。

## 我的理解

Service 解决的是「后端实例变化但调用方不变」，Ingress 解决的是「多 HTTP 路由共享外部入口」，DNS 解决的是「用稳定名称找到入口」，NetworkPolicy 解决的是「谁允许和谁通信」。它们分属发现、路由和安全三个层次，不应混成一个概念。

## Related

- [04 Pod 与多容器模式](./04-pods-and-container-patterns.md)
- [05 Workload：Deployment、StatefulSet 与 Job](./05-workloads-deployments-and-jobs.md)
- [07 ConfigMap 与 Secret](./07-configuration-configmap-and-secrets.md)
- [10 RBAC、ServiceAccount 与 Pod 安全](./10-security-rbac-and-pod-security.md)

## References

- [Service](https://kubernetes.io/docs/concepts/services-networking/service/)
- [Ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/)
- [DNS for Services and Pods](https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/)
- [Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
- [Gateway API](https://gateway-api.sigs.k8s.io/)
