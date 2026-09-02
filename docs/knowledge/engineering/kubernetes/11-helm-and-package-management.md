---
title: Kubernetes 11：Helm 与包管理
type: concept
status: seed
tags: [Kubernetes, Helm, Chart, Packaging, Deployment]
created: 2026-08-31
updated: 2026-08-31
source: Helm 官方文档、Kubernetes 官方文档与 roadmap.sh Kubernetes Roadmap
---

# Kubernetes 11：Helm 与包管理

## 一句话理解

> Helm 把一组带参数的 Kubernetes 清单打包成 Chart，并以 Release 为单位进行安装、升级和回滚；它管理清单交付，不替代 Kubernetes 控制器。

## Helm 解决什么问题

一个应用通常不仅有 Deployment，还包括 Service、Ingress、ConfigMap、ServiceAccount、RBAC、HPA 等。复制多份 YAML 给不同环境会造成漂移。Helm 用模板和 values 抽取变化点：

```text
Chart templates + values-dev.yaml  → dev Release → Kubernetes 对象
                + values-prod.yaml → prod Release → Kubernetes 对象
```

Helm 客户端渲染模板并向 Kubernetes API 提交对象；对象之后仍由各 Kubernetes controller 负责运行。

## Chart 的基本结构

```text
my-app/
├── Chart.yaml          # Chart 名称、版本、依赖
├── values.yaml         # 默认参数
├── templates/          # Kubernetes 模板
│   ├── deployment.yaml
│   ├── service.yaml
│   └── _helpers.tpl
└── charts/             # 打包后的依赖（可选）
```

`Chart.yaml` 中应区分：

- `version`：Chart 本身版本；
- `appVersion`：应用版本提示，不自动决定镜像 tag；
- dependencies：子 Chart 依赖，需要锁定和更新流程。

## 一个最小模板

`values.yaml`：

```yaml
replicaCount: 2
image:
  repository: nginx
  tag: "1.27"
service:
  port: 80
```

`templates/deployment.yaml` 的片段：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "my-app.fullname" . }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app.kubernetes.io/name: {{ include "my-app.name" . }}
  template:
    metadata:
      labels:
        app.kubernetes.io/name: {{ include "my-app.name" . }}
    spec:
      containers:
        - name: app
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
```

模板越灵活不一定越好。只把环境真实需要变化的参数暴露为 values，避免把整个 Kubernetes API 再抽象一遍。

## 常用命令

```bash
helm create my-app
helm lint ./my-app
helm template web ./my-app -n k8s-lab -f values-dev.yaml
helm install web ./my-app -n k8s-lab --create-namespace
helm list -n k8s-lab
helm upgrade web ./my-app -n k8s-lab -f values-prod.yaml --wait
helm history web -n k8s-lab
helm rollback web 1 -n k8s-lab
helm uninstall web -n k8s-lab
```

推荐在安装前先 `helm lint`、`helm template`，并对渲染后的 YAML 做审查或策略校验。`--wait` 会等待部分资源达到就绪条件，但仍不代表业务指标完全健康。

## Values 的优先级

通常由低到高：Chart 默认 `values.yaml` → `-f` 指定文件 → `--set`/`--set-string`。多个 values 文件叠加时，后面的覆盖前面的。

命令行 `--set` 适合少量临时参数；复杂结构和可审计环境配置优先放在版本控制的 values 文件中。不要把明文 Secret 直接提交到 values 文件或命令行历史。

## Release、升级与回滚

Release 是「某个 Chart + 某组 values 在一个命名空间中的安装实例」。升级时 Helm 计算并提交新的资源；Kubernetes Workload 再执行 rollout。

回滚只能恢复 Helm 记录的清单版本，不一定能逆转：

- 已执行的数据迁移；
- 外部数据库结构变化；
- PVC 中的数据；
- Hook 创建的外部资源；
- 不向后兼容的应用协议。

因此发布策略要把应用、数据和资源清单一起设计。

## Chart 依赖与仓库

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
helm search repo nginx
helm dependency update ./my-app
helm package ./my-app
```

安装第三方 Chart 前应检查：镜像来源、权限、SecurityContext、CRD、values 默认值、升级说明和维护状态。Chart 是可执行部署意图，不应只因来源流行就直接进入生产。

## Helm 与其他工具的边界

| 工具/机制 | 主要职责 |
|---|---|
| Helm | 模板化、打包、Release 生命周期 |
| Kustomize | 对 YAML 做声明式 overlay/patch，不使用 Helm 模板语言 |
| kubectl apply | 把对象意图提交给 API |
| GitOps Controller | 持续让集群状态与 Git 声明一致 |
| Kubernetes Controller | 管理对象实际运行状态 |

它们可以组合，但应避免多个系统同时修改同一字段而产生控制冲突。

## 常见误区

- **Helm 是 Kubernetes 包运行时**：Helm 负责渲染和提交资源，Pod 仍由 Kubernetes 运行。
- **Chart values 越多越通用**：过度模板化会让配置难以理解和测试。
- **`helm upgrade` 成功等于业务成功**：要看 rollout、探针、日志和业务指标。
- **回滚能恢复数据库**：清单回滚与数据回滚是不同问题。
- **Secret 放 values 就安全**：values 可能出现在 Git、CI 日志或 Release 数据中，需要专门密钥方案。
- **第三方 Chart 可以默认信任**：它可能创建高权限 RBAC、hostPath 或特权容器。

## 我的理解

Helm 的价值是把一组相关 Kubernetes 对象定义成可复用、可版本化的交付单元。它最危险的地方也在模板：如果抽象层隐藏了实际 YAML，团队会失去对权限、网络和更新行为的直觉。因此总要能查看并审查渲染结果。

## Related

- [03 kubectl、对象与 YAML](./03-kubectl-objects-and-yaml.md)
- [05 Workload：Deployment、StatefulSet 与 Job](./05-workloads-deployments-and-jobs.md)
- [07 ConfigMap 与 Secret](./07-configuration-configmap-and-secrets.md)
- [13 CRD、Operator 与生产下一步](./13-crd-operators-and-production-next-steps.md)

## References

- [Helm Documentation](https://helm.sh/docs/)
- [Charts](https://helm.sh/docs/topics/charts/)
- [Chart Template Guide](https://helm.sh/docs/chart_template_guide/)
- [Using Helm](https://helm.sh/docs/intro/using_helm/)
