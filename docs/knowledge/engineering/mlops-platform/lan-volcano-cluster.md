---
title: 局域网多服务器搭建 Volcano 集群
type: guide
status: growing
tags: [kubernetes, volcano, gpu, distributed-training, mlops]
created: 2026-11-05
updated: 2026-11-05
source: 用户提供的部署草稿与官方文档入口；尚未在实际环境验证
---

# 局域网多服务器搭建 Volcano 集群

## 一句话理解

Volcano 不是独立于 Kubernetes 的集群，而是安装在 Kubernetes 上的批处理和调度扩展。先建立可用的 Kubernetes 集群，再安装 Volcano；GPU 工作负载还需要节点驱动、容器运行时配置和 GPU device plugin。

本文是**尚未在具体环境验证的实验指南**，不是生产环境操作手册。实际安装前应核对各组件版本、操作系统、网络和存储条件。

## 整体架构与前提

以三台局域网服务器为例，IP 均为示例：

| 服务器 | IP 示例 | 角色 |
|---|---|---|
| server-1 | `192.168.1.11` | Kubernetes 控制节点 |
| server-2 | `192.168.1.12` | 工作节点，可带 GPU |
| server-3 | `192.168.1.13` | 工作节点，可带 GPU |

```text
使用者 / 可选的 Dashboard
        │ kubectl / Kubernetes API
        ▼
server-1：控制节点
  ├── Kubernetes API Server 与控制组件
  └── Volcano 控制器和调度器（实际部署位置取决于配置）
        │
        ├──────────────────┐
        ▼                  ▼
server-2：工作节点       server-3：工作节点
  ├── containerd           ├── containerd
  ├── kubelet 与 Pod       ├── kubelet 与 Pod
  └── GPU（可选）          └── GPU（可选）
```

提交 Volcano Job 后，控制器创建任务 Pod，Volcano scheduler 根据资源和调度策略决定何时、在哪个节点运行；节点上的 kubelet 和容器运行时再启动容器。输出若需持久保存，必须显式配置相应存储或日志系统。

这是**单控制节点**实验方案：server-1 故障时控制面不可用。生产环境还需要高可用控制面、etcd 备份恢复、监控、权限与安全策略等。

## 一、准备机器和网络

每台机器需要兼容的 Linux 系统、互通且稳定的局域网地址、唯一 hostname、足够的 CPU/内存/磁盘、准确的系统时间，以及安装软件包和拉取镜像的能力。GPU 节点还需要 NVIDIA GPU 和兼容驱动。建议同步节点时间并规划必要的防火墙端口。

提前规划互不冲突且不与局域网、现有网络重叠的 **Pod CIDR** 和 **Service CIDR**。Pod CIDR 必须与所选 CNI 的配置一致；地址选择应结合实际网络和 CNI 文档。

从控制节点检查连通性：

```bash
ping 192.168.1.12
ping 192.168.1.13
ssh user@192.168.1.12 hostname
ssh user@192.168.1.13 hostname
```

若节点需要访问公网镜像仓库，还要验证各节点的访问能力；隔离环境应提前准备内部仓库、镜像和软件包。`ping` 可用于初步排查，但不能代替 API Server、CNI、存储和节点间各必要端口的检查。

## 二、先搭建 Kubernetes

常见组合是 **kubeadm + containerd + CNI**：kubeadm 初始化和加入集群，containerd 运行容器，kubelet 管理节点上的 Pod，CNI 提供 Pod 网络。基础概念可先读 [Kubernetes 入门 Wiki](../kubernetes/index.md)。

1. 在各节点安装并配置 containerd、kubelet、kubeadm；在需要操作集群的机器安装 kubectl。
2. 按官方文档核对内核、容器运行时、cgroup 和 swap 等设置；核对 Kubernetes、containerd、CNI 与 Volcano 的版本兼容性。
3. 在 server-1 初始化控制面。例如以下仅为模板，将 CIDR 替换为已规划的实际网段：

```bash
POD_CIDR='10.244.0.0/16'  # 示例：须先确认不冲突且与所选 CNI 一致
sudo kubeadm init \
  --apiserver-advertise-address=192.168.1.11 \
  --pod-network-cidr="$POD_CIDR"
```

Service CIDR 如需指定，应与实际配置保持一致。初始化后按输出配置 kubectl；在 server-2 和 server-3 分别执行生成的 `kubeadm join` 命令。join token 有有效期，完整命令属于敏感接入信息，不应公开。

4. 根据所选 CNI 的官方文档安装兼容版本；待节点加入并完成网络配置后检查：

```bash
kubectl get nodes -o wide
kubectl get pods -A
```

期望节点为 `Ready`、系统组件正常。先确认普通 Kubernetes 集群可运行 Pod，再安装 Volcano，便于隔离故障。

- [使用 kubeadm 创建集群](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/create-cluster-kubeadm/)
- [安装 kubeadm](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/)
- [容器运行时配置](https://kubernetes.io/docs/setup/production-environment/container-runtimes/)

## 三、安装 Volcano

根据集群 Kubernetes 版本查阅 [Volcano 官方安装文档](https://volcano.sh/en/docs/installation/)，选择兼容版本和 Helm、YAML 等安装方式。不同版本的安装资源与配置可能变化，避免将不同时期的命令拼接使用。

安装后检查：

```bash
kubectl get pods -n volcano-system
kubectl get crd | grep volcano
```

确认 scheduler、controller 等组件正常，以及所需 CRD 已注册；具体资源名依安装方式和版本而异。可创建示例 namespace：

```bash
kubectl create namespace team-vis
```

如果已存在，该命令会报已存在；可确认后继续使用。生产环境应按需求设置资源配额及最小权限 RBAC。

## 四、提交第一个 CPU Volcano Job

先不使用 GPU，以最小任务验证 API、控制器、调度器和容器运行链路。保存为 `hello-volcano.yaml`：

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: hello-volcano
  namespace: team-vis
spec:
  minAvailable: 1
  schedulerName: volcano
  tasks:
    - name: worker
      replicas: 1
      template:
        spec:
          restartPolicy: Never
          containers:
            - name: worker
              image: busybox:1.36
              command:
                - sh
                - -c
                - |
                  echo "Hello from Volcano"
                  hostname
                  sleep 10
              resources:
                requests:
                  cpu: "1"
                  memory: "256Mi"
                limits:
                  cpu: "1"
                  memory: "256Mi"
```

这里 `minAvailable: 1` 表示至少一个任务 Pod 满足调度要求；本例只有一个副本，因此**不能单独验证跨节点通信或多副本 gang scheduling**。镜像须可从执行节点拉取，内网可替换为可信的内部镜像。

```bash
kubectl apply -f hello-volcano.yaml
kubectl get vcjob -n team-vis
kubectl get pods -n team-vis -o wide
# 将 POD_NAME 换成上一步输出的实际 Pod 名称
kubectl logs -n team-vis "$POD_NAME"
```

看到 `Hello from Volcano` 说明最基本的提交和执行链路已打通。Pod 长时间 `Pending` 时先查看 `kubectl describe pod -n team-vis "$POD_NAME"` 的 Events；也可查看 Job、PodGroup 和 Volcano scheduler 日志，确认资源、队列、污点和调度条件。`ImagePullBackOff` 等镜像问题则检查镜像名、凭据和节点到仓库的网络。更多思路见 [Kubernetes 故障排查](../kubernetes/12-observability-and-troubleshooting.md)。

## 五、配置和验证 GPU

每个 GPU 工作节点需要：

1. 安装与 GPU、系统和框架需求兼容的 NVIDIA 驱动；
2. 安装并配置 NVIDIA Container Toolkit，使所用容器运行时能暴露 GPU；
3. 在集群中部署 NVIDIA Kubernetes device plugin，发现并向 Kubernetes 上报 GPU；
4. 选择与框架和宿主机驱动兼容的容器镜像。

先在 GPU 节点检查 `nvidia-smi`，再查看 Kubernetes 节点状态：

```bash
# 将 GPU_NODE_NAME 替换为 GPU 节点名
kubectl describe node "$GPU_NODE_NAME"
```

在 `Capacity` / `Allocatable` 中查找 `nvidia.com/gpu` 及其数量。之后可在 Volcano Job 的容器中申请 GPU：

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

这申请一个由 device plugin 暴露的 GPU 资源；**资源已上报不等于容器内 GPU 已验证可用**，还应运行兼容镜像，在实际 Pod 内验证 GPU。多节点训练还需正确配置进程启动、节点间网络、NCCL、存储和同步；安装 Volcano 不会自动完成这些工作。

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)
- [NVIDIA Kubernetes device plugin](https://github.com/NVIDIA/k8s-device-plugin)

## 六、存储与 Dashboard 接入

多机训练或评估应明确：数据集和 checkpoint 使用各节点及 Pod 可访问的 NFS、共享文件系统或对象存储；输出和日志有持久化位置；镜像仓库对所有工作节点可达；各节点本地高速盘上的临时 cache 不应误认为持久数据。宿主机能访问路径，并不代表 Pod 已挂载且有权限访问。相关概念见 [PV、PVC、StorageClass 与 CSI](../kubernetes/08-storage-pv-pvc-and-csi.md) 和 [AI 训练存储工程地图](../../distributed-systems/distributed-storage/ai-training-and-multicloud-storage-map.md)。

接入 Dashboard 时，应核实它实际通过 Kubernetes API 还是 SSH 网关执行 `kubectl`，配置对应凭据与 namespace，使用最小权限 RBAC，并检查 Job 镜像、存储挂载和 GPU 配置。不要仅凭示例推断某个 Dashboard 已配置完成或可以运行。

> **原始项目上下文，未在本知识库验证：**草稿提到某训练系统的 `--tmp_dir_root` 是 sequential cache 的本地临时目录，不同训练不能共用，中断后可能需要人工确认并清理；还提到某 Dashboard 通过 SSH 网关运行 `kubectl`、使用 `team-vis`，相关文件可能有合并冲突。这些属于特定项目的待核实情况，不是 Kubernetes 或 Volcano 的通用行为；不能假定此知识库包含该 Dashboard 或对应代码。

## 七、推荐验证顺序与待确认信息

1. 初始化 Kubernetes，先验证普通 CPU Pod；
2. 加入其他节点并确认网络、节点状态；
3. 安装 Volcano，运行 Hello Volcano Job；
4. 配置并验证单节点 GPU；
5. 挂载并验证共享数据和结果存储；
6. 验证多节点 GPU 任务、进程启动及通信；
7. 最后接入 Dashboard，并验证权限和任务模板。

制定实际方案前，确认 Linux 发行版和版本、服务器及 GPU 型号与数量、外网访问和镜像仓库、共享存储、控制面高可用需求，以及是否接入现有 Dashboard、训练系统或 CI。

## Related

- [Kubernetes 资源、调度与自动扩缩容](../kubernetes/09-resources-scheduling-and-autoscaling.md) — 调度资源基础
- [黑芝麻模型训练平台架构解析](./black-sesame-training-platform.md) — GPU 训练平台与存储分层实例
- [PyTorch 分布式训练](../../ai/systems/pytorch/pytorch-distributed.md) — 集群承载的典型训练任务
