---
title: Kubernetes 08：PV、PVC、StorageClass 与 CSI
type: concept
status: seed
tags: [Kubernetes, Storage, PV, PVC, StorageClass, CSI]
created: 2026-08-31
updated: 2026-08-31
source: Kubernetes 官方存储文档与 roadmap.sh Kubernetes Roadmap
---

# Kubernetes 08：PV、PVC、StorageClass 与 CSI

## 一句话理解

> Pod 描述「我要挂载什么」，PVC 描述「我需要怎样的存储」，PV 代表「实际的一块持久卷」，StorageClass 和 CSI 负责把请求动态连接到具体存储后端。

## 为什么需要这层抽象

Pod 会被替换，但数据库、上传文件和检查点不能随 Pod 一起消失。Kubernetes 把计算生命周期与存储生命周期分离：应用只请求能力，不必把云厂商磁盘 API 写进 Pod 模板。

```text
Pod → PVC（需求）→ StorageClass（供给策略）→ CSI Driver → 云盘 / NAS / 本地存储
                         ↓
                    PV（实际卷）
```

## Volume、PV 与 PVC

- **Volume**：Pod 内的挂载来源，可以是临时卷、ConfigMap、Secret 或持久卷；
- **PersistentVolume（PV）**：集群可使用的持久存储资源，描述容量、访问模式、回收策略等；
- **PersistentVolumeClaim（PVC）**：用户/应用对 PV 能力的请求；
- **StorageClass**：一类存储的模板和动态供给参数。

应用通常只引用 PVC：

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: web-data
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: standard
  resources:
    requests:
      storage: 1Gi
```

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: writer
spec:
  containers:
    - name: writer
      image: busybox:1.36
      command: ["sh", "-c", "echo hello > /data/message; sleep 3600"]
      volumeMounts:
        - name: data
          mountPath: /data
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: web-data
```

## 访问模式不是性能承诺

| 模式 | 语义 |
|---|---|
| ReadWriteOnce（RWO） | 一个节点以读写方式挂载；不是「一个 Pod」 |
| ReadOnlyMany（ROX） | 多节点只读挂载 |
| ReadWriteMany（RWX） | 多节点读写挂载 |
| ReadWriteOncePod（RWOP） | 由单个 Pod 使用（需相应支持） |

可用模式取决于存储后端和 CSI Driver。RWO 通常允许同一节点上的多个 Pod 使用同一卷，不能简单理解成「全局只能有一个 Pod」。

## StorageClass 与动态供给

```bash
kubectl get storageclass
kubectl get pvc,pv
kubectl describe pvc web-data
```

StorageClass 常包含 provisioner、参数、绑定模式和回收策略。用户创建 PVC 后，外部 provisioner 可能动态创建 PV 和后端磁盘。

- `Immediate`：PVC 创建时立即绑定/供给；
- `WaitForFirstConsumer`：等 Pod 的调度约束确定后再供给，避免拓扑不匹配；
- `Delete` / `Retain`：删除 PVC 后，PV/后端数据的处理策略；具体行为必须核对存储插件和集群策略。

## CSI：标准存储插件接口

CSI（Container Storage Interface）把 Kubernetes 与存储系统连接起来。常见组件承担不同职责：

- controller sidecar：创建、删除、扩容卷并协调快照；
- node plugin：在节点上挂载/卸载卷并把设备暴露给 Pod；
- external provisioner：观察 PVC 并请求供给 PV；
- external attacher：处理需要 attach 的块设备。

```text
Kubernetes controller → CSI controller → 存储 API
kubelet on node       → CSI node plugin → 节点挂载点
```

CSI 是接口，不是某个具体存储产品。故障排查要沿着 PVC → PV → CSI controller/node plugin → 后端卷逐层看。

## 有状态应用的基本判断

持久卷能保存字节，但不自动解决：

- 数据库复制和选主；
- 跨区域容灾；
- 备份、恢复和一致性快照；
- 文件锁和并发访问；
- 升级时的数据格式兼容。

StatefulSet、PVC 和数据库 Operator 往往需要一起设计，但三者职责不同。先明确应用的数据模型和恢复目标，再选择存储模式。

## 存储排障路径

```bash
kubectl -n k8s-lab get pvc
kubectl -n k8s-lab describe pvc web-data
kubectl get pv
kubectl get storageclass
kubectl -n k8s-lab get pod -o wide
kubectl -n k8s-lab describe pod <pod-name>
kubectl -n kube-system get pods | grep -i csi
```

常见状态：

- PVC `Pending`：没有匹配 PV、StorageClass 不存在、动态供给失败或拓扑/容量不满足；
- Pod `ContainerCreating`：可能正在挂载卷、拉取镜像或等待 CNI；
- 挂载成功但写入失败：检查文件系统权限、只读挂载、UID/GID、配额和后端故障。

## 回收与数据安全

`Delete` 可能在删除声明后清理后端资源，`Retain` 则把恢复/清理责任留给管理员。任何生产数据都不应只依赖 PV 的回收策略；应有独立备份、恢复演练和删除保护流程。

## 常见误区

- **PVC 就是硬盘**：PVC 是请求，实际能力来自绑定的 PV、StorageClass 和后端。
- **删除 Pod 会删除数据**：取决于卷类型和控制器；持久卷通常独立于 Pod，但不要用假设代替检查。
- **RWO 只能被一个 Pod 使用**：它通常限制节点级读写挂载，不是简单的 Pod 数量限制。
- **有 PVC 就有跨节点高可用**：存储后端、访问模式、复制和应用协议共同决定可用性。
- **CSI 负责保存业务数据语义**：CSI 主要负责卷生命周期和挂载，数据库一致性由更高层负责。
- **StorageClass 名称在所有集群都一样**：默认类、provisioner 和参数由集群环境决定。

## 我的理解

Kubernetes 存储的核心是「需求与实现解耦」：应用通过 PVC 描述所需能力，平台通过 StorageClass/CSI 选择实现。抽象降低了应用对基础设施的耦合，却不能消灭容量、拓扑、性能、备份和一致性这些真实约束。

## Related

- [05 Workload：Deployment、StatefulSet 与 Job](./05-workloads-deployments-and-jobs.md)
- [07 ConfigMap 与 Secret](./07-configuration-configmap-and-secrets.md)
- [09 资源、调度与自动扩缩容](./09-resources-scheduling-and-autoscaling.md)
- [分布式存储系统知识地图](../distributed-storage-knowledge-map.md) — 分布式存储、CSI 与 Operator 的工程背景

## References

- [Volumes](https://kubernetes.io/docs/concepts/storage/volumes/)
- [Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
- [Storage Classes](https://kubernetes.io/docs/concepts/storage/storage-classes/)
- [Container Storage Interface (CSI)](https://kubernetes.io/docs/concepts/storage/volumes/#csi)
