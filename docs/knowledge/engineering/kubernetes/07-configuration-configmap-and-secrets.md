---
title: Kubernetes 07：ConfigMap、Secret 与应用配置
type: concept
status: seed
tags: [Kubernetes, ConfigMap, Secret, Configuration, Security]
created: 2026-08-31
updated: 2026-08-31
source: Kubernetes 官方配置文档与 roadmap.sh Kubernetes Roadmap
---

# Kubernetes 07：ConfigMap、Secret 与应用配置

## 一句话理解

> ConfigMap 和 Secret 把配置数据从容器镜像与代码中分离出来，使同一份镜像可以在不同环境运行；Secret 只表达敏感数据语义，不自动等于安全存储。

## 为什么把配置从镜像分离

如果每个环境都重新构建镜像，发布链路会变长，且容易把环境差异和凭据烘焙进镜像。Kubernetes 对象可以在部署时注入环境变量或文件：

```text
同一个镜像
   + dev ConfigMap/Secret → 开发行为
   + prod ConfigMap/Secret → 生产行为
```

镜像应包含程序和默认安全行为；环境对象提供部署差异。

## ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: web-config
data:
  LOG_LEVEL: info
  app.yaml: |
    timeout_seconds: 3
    feature_x: false
```

以环境变量注入：

```yaml
envFrom:
  - configMapRef:
      name: web-config
```

以文件挂载：

```yaml
volumeMounts:
  - name: config
    mountPath: /etc/web
volumes:
  - name: config
    configMap:
      name: web-config
```

ConfigMap 适合非敏感配置。把大二进制、不断变化的状态或凭据放入 ConfigMap 会降低可维护性和安全性。

## Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: web-credentials
type: Opaque
stringData:
  username: app
  credential: "<inject-at-deploy-time>"
```

读取：

```yaml
env:
  - name: DATABASE_CREDENTIAL
    valueFrom:
      secretKeyRef:
        name: web-credentials
        key: credential
```

`stringData` 便于写明文输入，API Server 会在处理时转换为 `data` 的 base64 表示。Base64 是编码，不是加密；生产环境还需配置静态加密、严格 RBAC、审计、外部密钥管理或 Secret Store CSI Driver 等措施。

不要把真实凭据提交到 Git。示例中的占位值只用于说明字段结构。

## 更新行为

- 通过环境变量注入的 ConfigMap/Secret 值通常不会在已运行进程中自动更新，常需重启 Pod；
- 通过卷挂载的文件可能最终被 kubelet 更新，但应用是否重新读取由应用决定；
- 使用 `subPath` 挂载时通常不会收到自动更新；
- 更新配置对象不一定自动触发 Deployment rollout。常见做法是给 Pod template 加配置 checksum annotation，或由配置刷新控制器管理。

```yaml
spec:
  template:
    metadata:
      annotations:
        config-checksum: "由部署工具根据配置计算的摘要"
```

真实部署中应由 Helm/Kustomize/CI 计算摘要，不要手工填写一个永远不变的占位值。

## 配置对象的边界

| 数据 | 更合适的位置 |
|---|---|
| 非敏感开关、URL、日志级别 | ConfigMap |
| 密码、token、证书私钥 | Secret 或外部密钥系统 |
| 稳定程序默认值 | 镜像/代码中的默认配置 |
| 用户数据、队列、数据库文件 | 持久存储，不是 ConfigMap/Secret |
| 高度动态配置 | 专门配置服务或控制器，评估缓存与一致性 |

## 通过命令生成对象

```bash
kubectl -n k8s-lab create configmap web-config \
  --from-literal=LOG_LEVEL=info \
  --dry-run=client -o yaml

kubectl -n k8s-lab create secret generic web-credentials \
  --from-literal=username=app \
  --from-literal=credential='<inject-at-deploy-time>' \
  --dry-run=client -o yaml
```

命令行参数可能进入 shell history、进程记录或 CI 日志，生产凭据应使用受控的 Secret 注入流程。

## 常见误区

- **Secret 的 base64 是加密**：它只是编码，拿到对象权限的人可能直接解码。
- **ConfigMap/Secret 更新后应用必然刷新**：注入方式、挂载方式和应用行为决定刷新效果。
- **把 Secret 放进私有 Git 就安全**：Git 历史、备份和日志可能长期保留泄露内容。
- **所有环境配置都写进镜像**：破坏镜像复用和发布可审计性。
- **环境变量适合所有配置**：大配置、证书、需要动态刷新或结构化文件通常更适合卷挂载或专门系统。

## 我的理解

ConfigMap 和 Secret 的价值首先是解耦，不是凭空提供密钥安全。真正的安全性来自完整链路：谁能读取 API 对象、etcd 如何加密、凭据如何轮换、应用是否会把值打进日志，以及部署历史是否可追溯。

## Related

- [06 Service、Ingress、DNS 与网络策略](./06-services-networking-ingress-and-dns.md)
- [08 PV、PVC、StorageClass 与 CSI](./08-storage-pv-pvc-and-csi.md)
- [10 RBAC、ServiceAccount 与 Pod 安全](./10-security-rbac-and-pod-security.md)
- [11 Helm 与 Kubernetes 包管理](./11-helm-and-package-management.md)

## References

- [ConfigMaps](https://kubernetes.io/docs/concepts/configuration/configmap/)
- [Secrets](https://kubernetes.io/docs/concepts/configuration/secret/)
- [Kubernetes Secrets Good Practices](https://kubernetes.io/docs/concepts/security/secrets-good-practices/)
- [Encrypting Confidential Data at Rest](https://kubernetes.io/docs/tasks/administer-cluster/encrypt-data/)
