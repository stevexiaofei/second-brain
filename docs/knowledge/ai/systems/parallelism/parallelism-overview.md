---
title: 并行策略总览：5D 并行的组合逻辑
type: concept
status: growing
tags: [distributed-training, parallelism, tensor-parallel, pipeline-parallel, expert-parallel, data-parallel, sequence-parallel, megatron]
created: 2026-11-06
updated: 2026-11-13
source: 个人整理 + Megatron-LM / DeepSpeed / PyTorch 文档
---

# 并行策略总览：5D 并行的组合逻辑

## 一句话理解

> 多维并行不是五种互斥的技术选项：DP 按数据分工，TP 切矩阵，PP 切层，EP 分配专家，SP/CP 分摊部分序列相关计算与激活。“5D”是理解这些策略的视角，不是五个独立 mesh 维度的统一标准：Megatron-SP 常复用 TP 组，EP 也可能复用非专家模块的 DP 设备集合。组合时既要核对**逻辑布局与通信组**，也要让高频通信尽量落在高带宽、低延迟的物理域内。

## 为什么需要并行：一个自顶向下的推导

不用背“5D 并行”这个名词，从下面四类需求就能理解策略的来源：

**约束 1：单卡显存放不下模型。** 70B 模型 bf16 权重 140 GB，加 Adam 状态（fp32 参数副本 + 动量 + 方差）约 840 GB，远超单卡 80 GB。可考虑三类分摊方式：

- **按层切（PP）**：每卡拿连续几层。简单，但通信只剩邻居 send/recv，代价是流水线气泡。
- **按矩阵内部切（TP）**：把单个权重矩阵切给多卡。粒度最细，但每层都要通信，代价是对带宽的苛刻要求。
- **按"参数没必要人人一份"切（ZeRO/FSDP）**：参数/梯度/优化器状态分片，计算时按需恢复参数，结束后释放临时全量副本。这是 DP 的状态分片路线，不等同于层内矩阵 TP。

**需求 2：处理更多训练数据、提高总吞吐。** **DP** 让多个模型副本处理不同样本，再同步梯度。它不自动解决单卡 batch 太小：固定全局 batch 时，增大 DP 反而会减小每个副本的 batch；固定副本 batch 时，全局 batch 才随 DP 增大。

**约束 3：长序列的激活与 attention 成本。** **Megatron-SP** 沿序列分片 LayerNorm/Dropout 等 token-local 区间，常复用 TP 组；**CP** 则分片 attention 上下文，需交换 KV 等信息。两者目的相关，但作用范围与通信并不相同。

**约束 4（MoE 时代新增）：参数量增长快于计算量。** MoE 把 FFN 替换为 $N$ 个专家，参数多了但每个 token 只算 Top-k 个 → **EP** 把不同专家放到不同卡，token 用 all-to-all 路由过去。

```text
单卡放不下 ──┬── 按层切 ──────────────→ PP（层间切分）
             ├── 按矩阵切 ────────────→ TP（层内切分）
             └── 按状态切 ────────────→ FSDP/ZeRO（DP 的进化）
算力要吃满 ──────────────────────────→ DP（数据维度）
激活显存爆炸 ────────────────────────→ SP/CP（序列维度）
参数增长 > 计算增长 ─────────────────→ EP/MoE（专家维度）
```

> 这些策略有不同的主要切分对象，因此可以组合，但绝非互不干扰。DP×TP×PP 的规则网格可按维度相乘；加入 EP、SP 时必须先确认它们是否复用已有组。不要把所有并行度机械相乘，也不必把每根轴都开满。

## 五根轴速查

| 轴 | 切什么 | 解决什么 | 每层/每步核心通信 | 通信量级 | 对硬件的要求 |
|---|---|---|---|---|---|
| **DP** | 数据（batch 维） | 提高样本吞吐 | 按桶梯度 all-reduce | ring 发送量近似梯度大小的 2 倍 | 可跨节点，仍受梯度规模与 overlap 影响 |
| **TP** | 权重矩阵（层内） | 分摊单层参数与计算 | 经典实现每层前向/反向各 2 次 AR | 每层高频，消息大小随激活形状变化 | 优先高带宽、低延迟域，非硬性限于 NVLink |
| **PP** | 网络层（层间） | 分摊深模型各层 | 邻居 send/recv | 按 microbatch 传边界激活及梯度 | 可跨节点，仍需关注依赖路径延迟 |
| **EP** | 专家（FFN 替换） | 参数容量 vs 计算量解耦 | 常见 A2A dispatcher 前向 dispatch + combine | 每 MoE 层前向 2 次 A2A，训练反向另计 | 依 token 流量与拓扑评估，跨节点可能代价大 |
| **SP/CP** | token-local 激活 / attention 上下文 | 激活显存、长上下文 | SP 常为边界 RS/AG；CP 交换 KV 等 | 随实现与序列形状变化 | SP 常复用 TP 组，CP 可跨节点 |

**组合时的黄金法则：通信频率越高的轴，越要放在带宽越高、延迟越低的物理拓扑内层。**

以 8 卡 × 8 节点 = 64 卡集群为例（每节点 NVLink 全互联）：

```text
物理拓扑（从内到外）：NVLink 域(8卡) → 节点间 IB/RoCE(64卡)
通信粒度：TP/SP 按层布局边界；EP 按 MoE 层；DP 按梯度桶；PP 按 microbatch 边界。
无法脱离层数、桶数与 microbatch 数给出所有策略的固定频率排序。

推荐映射：
  TP  → 优先 NVLink 域内（本例最多 8 卡），不是普适硬上限
  SP  → 与 TP 同组（Megatron 惯例：SP 是 TP 的伴生）
  DP  → 可跨节点，按梯度桶通信；与 PP 如何分组需共同决定
  PP  → 可跨节点，按 stage 边界传激活及梯度，仍需实测链路代价
  EP  → 视模型而定：dense 部分 TP×DP，MoE 部分 TP×EP
```

Megatron-LM 文档里的 `--tensor-model-parallel-size 8 --pipeline-model-parallel-size 4` 加上 world_size 反推 DP size，就是这个映射的命令行化身。

## 每篇分论（本系列）

| 策略 | 切分对象 | 通信模式 | 详细笔记 |
|---|---|---|---|
| TP（Megatron 式） | 权重矩阵（列切+行切） | 经典实现每层前向/反向各 2 次 AR | [TP：Megatron 式张量并行](./tensor-parallel-megatron.md) |
| PP | 网络层 | 邻居 send/recv + 调度 | [流水线并行](./pipeline-parallelism.md) |
| EP/MoE | 专家（token 路由） | dispatch/combine 两次 all-to-all | [专家并行与 MoE](./expert-parallel-moe.md) |
| DP/FSDP/ZeRO | 数据/优化器状态 | all-reduce / AG+RS | 见 [PyTorch 分布式训练](../pytorch/pytorch-distributed.md)、[FSDP 源码阅读指南](../pytorch/pytorch-fsdp.md) |
| SP/CP | token-local 区间 / attention 上下文 | SP 边界 RS/AG；CP 按实现交换或重排 KV 等 | [序列并行 SP 与上下文并行 CP](./sequence-context-parallelism.md) |

## 与物理拓扑的对照

本系列讲的是**计算侧的通信模式**（谁和谁通信、通信什么、多少次）。对应的**物理侧**（GPU → PCIe → HCA → Leaf/Spine 的链路怎么验证）见 [千卡集群网络拓扑与 NCCL 排查](../../../engineering/mlops-platform/thousand-gpu-network-topology.md)。两篇合起来才是完整的"千卡怎么跑通"：

```text
并行策略回答：逻辑上这 64 卡怎么分工？通信模式是什么？
网络拓扑回答：物理上这些通信走哪条链路？带宽够不够？如何验证？
```

排障时的顺序也正好是"逻辑 → 物理"：先确认并行维度与 mesh 分组符合预期（本系列），再看 NCCL 是否沿预期路径通信（网络拓扑笔记）。

## 我的理解

- **5D 并行的本质是"分而治之"在显存/计算/通信三维上的展开**：每根轴都是把某种"放不下或算不满"的资源切成 $1/N$，然后用一种特定的通信模式拼回来。选择并行策略就是选择"切什么、何时拼、拼的代价谁来付"。
- **可组合不等于互不影响**：TP/PP 分摊模型后可能腾出显存用于更大 microbatch，但缓冲区、驻留激活与参数预取又会占用空间；各组还会竞争计算和互联资源。所以多维并行不是“全开就好”，而要按瓶颈逐项验证。
- **"哪根轴放哪个 mesh 维度"比"用不用某根轴"更能体现工程功力**：同样开 TP=8、DP=8，TP 放 NVLink 内还是跨节点，性能可能差数倍。现代框架（PyTorch DeviceMesh、Megatron）都把"逻辑轴 → 物理拓扑"的映射显式化，正是为了让这个决策可见、可调。
- **范式正在从"规则驱动"走向"搜索驱动"**：手调 5D 并行的组合空间已经大到人力难以枚举（Alpa 等工作的动机），但理解每根轴的通信代价仍是看懂任何自动搜索结果的前提。

## 常见误区

1. **把 5D 当作五个互斥选项或五个独立设备维度** → 策略可以组合，也可能复用通信组；先核对各模块的参数与激活布局。
2. **把 FSDP 等同于层内矩阵 TP** → FSDP/ZeRO 沿 DP 组分片模型状态，各数据副本处理不同数据；FSDP 为本地计算临时恢复参数，TP 则让同一次算子的矩阵计算由多 rank 协作。
3. **认为"并行度越高越快"** → 每根轴都有通信税：TP 开太大，每层的 all-reduce 会吃掉计算收益；DP 开太大，全局 batch 变大可能伤收敛。并行度是显存约束下的优化问题，不是越大越好。
4. **把 TP 和 PP 的通信代价等量齐观** → TP 通常按层高频同步，PP 按 microbatch 传 stage 边界；二者的拓扑偏好不同，但不能据此推出跨节点 TP 一律错误。
5. **重复计算模型并行 rank 的 batch** → 设每 microbatch 有 $b$ 个样本、每次更新累积 $m$ 个 microbatch、DP 副本数为 $d$，则 $B_{\text{global}}=bmd$。TP/PP/CP 协作处理同一样本，不额外乘这些并行度。

## Related

- [集合通信原语](../pytorch/pytorch-collective-communication.md) — 本系列通信模式（AR/AG/RS/A2A/send-recv）的"名词层"详解，两篇互为名词与句子
- [张量并行：Megatron 式](./tensor-parallel-megatron.md) — 本系列 TP 分论
- [流水线并行](./pipeline-parallelism.md) — 本系列 PP 分论
- [专家并行与 MoE](./expert-parallel-moe.md) — 本系列 EP/MoE 分论
- [PyTorch 分布式训练](../pytorch/pytorch-distributed.md) — PyTorch 分布式栈全景：ProcessGroup/DeviceMesh/DDP/FSDP
- [FSDP 源码阅读指南](../pytorch/pytorch-fsdp.md) — DP 进化线的源码级实现
- [千卡集群网络拓扑与 NCCL 排查](../../../engineering/mlops-platform/thousand-gpu-network-topology.md) — 本系列通信模式的物理承载：链路与验证
- [PyTorch 分布式训练性能分析](../../../engineering/profiling/07-pytorch-distributed-profiling.md) — 用 profiler 验证本系列理论

## References

- Megatron-LM: *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*（Narayanan et al., 2021）— 3D 并行（TP×PP×DP）的原始论述
- DeepSpeed 文档：ZeRO 与 MoE 训练（数据/专家并行组合）
- PyTorch 文档：`DeviceMesh`、`torch.distributed.tensor.parallel`、`torch.distributed.pipelining`
- DeepSeek-V3 Technical Report（2024）— MoE 的大规模训练与通信/计算重叠实践
- Alpa: *Automating Inter- and Intra-Operator Parallelism*（Zheng et al., OSDI 2022）— 自动搜索并行策略
