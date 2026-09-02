---
title: Kubernetes 10：RBAC、ServiceAccount 与 Pod 安全
type: concept
status: seed
tags: [Kubernetes, Security, RBAC, ServiceAccount, Pod Security, NetworkPolicy]
created: 2026-08-31
updated: 2026-08-31
source: Kubernetes 官方安全文档与 roadmap.sh Kubernetes Roadmap
---

# Kubernetes 10：RBAC、ServiceAccount 与 Pod 安全

## 一句话理解

> Kubernetes 安全要分别控制「谁能调用 API」「Pod 以什么身份和 Linux 权限运行」「哪些网络连接被允许」，RBAC、SecurityContext 和 NetworkPolicy 解决的是不同层次。

## API 请求的安全链路

```text
客户端
  ↓ TLS / Authentication：你是谁？
API Server
  ↓ Authorization（RBAC）：你能做什么？
  ↓ Admission / Pod Security：这个请求允许创建吗？
对象被持久化或执行
```

认证、授权和准入不是同一个机制；只给一个用户 RBAC 权限，不会自动改变他们启动的容器权限。

## RBAC 四个核心对象

- **Role**：命名空间内允许哪些 API 动词和资源；
- **ClusterRole**：集群级权限，或可被命名空间绑定复用；
- **RoleBinding**：把 Role/ClusterRole 绑定给用户、组或 ServiceAccount；
- **ClusterRoleBinding**：把 ClusterRole 绑定到集群范围主体。

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
  namespace: k8s-lab
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: app-read-pods
  namespace: k8s-lab
subjects:
  - kind: ServiceAccount
    name: app
    namespace: k8s-lab
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

遵循最小权限：只授予需要的 verbs、resources、apiGroups 和 namespaces。不要为了让程序「先跑起来」直接绑定 `cluster-admin`。

```bash
kubectl auth can-i get pods --as=system:serviceaccount:k8s-lab:app -n k8s-lab
kubectl auth can-i create deployments --as=system:serviceaccount:k8s-lab:app -n k8s-lab
```

## ServiceAccount

Pod 中的应用可以使用 ServiceAccount 身份访问 API：

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: app
  namespace: k8s-lab
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
  namespace: k8s-lab
spec:
  replicas: 1
  selector:
    matchLabels: {app: app}
  template:
    metadata:
      labels: {app: app}
    spec:
      serviceAccountName: app
      automountServiceAccountToken: false
      containers:
        - name: app
          image: example/app:1.0
```

若应用不需要访问 Kubernetes API，关闭自动挂载 token 可减少暴露面。需要访问时，应明确它需要哪些资源和动作，并考虑 token 投射、轮换和外部身份方案。

## SecurityContext

```yaml
spec:
  securityContext:
    runAsNonRoot: true
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: app
      image: example/app:1.0
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop: ["ALL"]
```

实际应用可能需要写临时目录、监听低端口、特定 UID 或 Linux capability。安全配置应在测试中验证，而不是盲目复制后在启动失败时全部放开。

## Pod Security Admission

Pod Security Admission 通过命名空间标签等方式施加 Baseline/Restricted 等 Pod 安全标准。它控制 Pod 是否能被创建，和 RBAC 的 API 权限不同：一个身份可能有权限创建 Pod，但 Pod 的安全上下文仍会被准入策略拒绝。

```bash
kubectl label namespace k8s-lab \
  pod-security.kubernetes.io/enforce=baseline \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted
```

先在 warn/audit 模式观察工作负载，再逐步 enforce；生产切换前应清点所有镜像的 UID、权限、hostPath、hostNetwork 和 capability 需求。

## NetworkPolicy 的位置

RBAC 控制 API 调用；NetworkPolicy 控制 Pod 网络流量；它们互不替代：

```text
程序访问 API Server → RBAC
Pod 访问数据库 Service → NetworkPolicy + Service + 应用认证
容器获得 root / host namespace → SecurityContext + Pod Security
```

## 安全基线

1. 使用命名空间和清晰 labels 组织租户；
2. 所有服务账号采用最小 RBAC；
3. 应用尽量非 root、禁止不必要的提权和 capability；
4. Secret 配置静态加密、访问审计和轮换；
5. 对入口、东西向流量和 egress 逐步建立 NetworkPolicy；
6. 固定可信镜像来源和版本，做漏洞扫描与签名验证；
7. 保留审计日志，限制 kubeconfig 和节点访问；
8. 让安全策略进入 CI，避免只在生产集群临时发现。

## 常见误区

- **Role 是角色身份**：Role 是权限规则，RoleBinding 才把规则绑定给主体。
- **ServiceAccount 等于管理员账户**：它只是 Pod 可使用的 API 身份，权限由 RBAC 决定。
- **RBAC 能阻止容器里的 root**：RBAC 不限制 Linux 进程权限，应使用 SecurityContext 和 Pod Security。
- **Pod Security 能阻止网络访问**：安全准入与网络访问控制是不同层次。
- **给应用 cluster-admin 最省事**：这是高影响的越权设计，后续也难以收回。
- **NetworkPolicy 默认有实现**：策略是否执行取决于网络插件和集群配置。

## 我的理解

Kubernetes 安全不是一个开关，而是三条正交边界：API 身份与权限、工作负载进程权限、网络可达性。最小权限只有在三者都收紧、并能观察和轮换凭据时才真正成立。

## Related

- [06 Service、Ingress、DNS 与网络策略](./06-services-networking-ingress-and-dns.md)
- [07 ConfigMap 与 Secret](./07-configuration-configmap-and-secrets.md)
- [12 可观测性与故障排查](./12-observability-and-troubleshooting.md)

## References

- [Kubernetes API Authentication](https://kubernetes.io/docs/reference/access-authn-authz/authentication/)
- [Authorization Overview](https://kubernetes.io/docs/reference/access-authn-authz/authorization/)
- [Using RBAC Authorization](https://kubernetes.io/docs/reference/access-authn-authz/rbac/)
- [Service Accounts](https://kubernetes.io/docs/concepts/security/service-accounts/)
- [Configure a Security Context for a Pod or Container](https://kubernetes.io/docs/tasks/configure-pod-container/security-context/)
- [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
