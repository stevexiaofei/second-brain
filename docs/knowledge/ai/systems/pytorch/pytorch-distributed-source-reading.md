---
title: 从 5D 并行到 PyTorch 分布式源码：精读路线
type: guide
status: growing
tags: [pytorch, distributed, source-code, reading-guide, ddp, fsdp, dtensor, tensor-parallel, pipeline-parallel]
created: 2026-11-25
updated: 2026-11-25
source: 本地 PyTorch ba56102387ef21a3b04b357e5b183d48f0afefc7 静态核验 + 个人学习路线整理
---

# 从 5D 并行到 PyTorch 分布式源码：精读路线

> **当前起点**：已读完 [5D 并行系列](../parallelism/index.md)，下一步不再重复概念综述，而是把“切什么、何时拼回来”追到真实对象、调用链、状态变化和测试断言。
>
> **验证范围**：本页已核验源码入口和所引用测试；没有执行多卡训练、性能测量或完整 PyTorch 测试。下文实验均为待完成任务，不是已有实验结论。

## 1. 先固定版本与阅读边界

| 项目 | 本次基线 |
|---|---|
| 本地源码 | `D:/project/pytorch` |
| 提交 | [`ba56102387ef21a3b04b357e5b183d48f0afefc7`](https://github.com/pytorch/pytorch/tree/ba56102387ef21a3b04b357e5b183d48f0afefc7)，提交日期 2025-08-04 |
| 版本标记 | [`version.txt`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/version.txt)：`2.8.0a0`，不能据此当作正式版 2.8.0，也不是 2.9 |
| 工作区差异 | 核验时 `test/distributed/_composable/fsdp/test_fully_shard_state.py` 存在本地修改；阅读它时先区分上游实现与本地实验 |
| 实际运行环境 | 尚未核对；安装的 wheel、CUDA/NCCL 与该源码可能不是同一版本 |

本页源码链接固定到上述提交，路径均相对 PyTorch 根目录。SB 旧笔记来自不同时间的阅读，不能把旧标题里的版本号、行号或实时 `main` 文档当成本次基线。开始实验前，记录实际的 `torch.__version__`、`torch.version.git_version`、CUDA/NCCL 版本、设备数和后端。

**这轮先读训练运行时，不通读整个 PyTorch：**

- 主线：通信契约 → DeviceMesh → DDP → DTensor → FSDP2 → TP/SP → PP → 组合验证。
- autograd 只补 `AccumulateGrad`、张量/模块 hook、引擎最终回调；见 [autograd 笔记](./pytorch-autograd.md)。
- CP、EP 是分支：本版本的 CP 是实验接口；不能把 PyTorch 的底层原语当成完整 EP 训练框架。
- 先跳过 RPC、elastic 重启、分布式 checkpoint、全量 `torch.compile`、NCCL 内部 ring/tree 算法。遇到主线依赖再下钻。

## 2. 用四个问题贯穿所有并行策略

每读一个模块，都填写同一张表，而不是摘抄函数列表：

| 问题 | 必须写清的内容 |
|---|---|
| **谁参与？** | global rank、group rank、mesh 坐标、当前进程所属子组 |
| **数据是什么？** | 逻辑 shape、本地 shape、placement、dtype、参数/梯度/激活/通信缓冲的区别 |
| **谁触发？** | 用户调用、module hook、autograd hook、schedule、引擎回调 |
| **何时可用、何时可释放？** | 发起通信、消费结果、stream/event/Work 依赖、别名与 storage 生命周期 |

**阅读时使用的心智模型，不是一条统一调用栈：**

```text
DDP       梯度就绪事件 → bucket → 归约 → 写回 grad
FSDP2     模块/反向 hook → 参数换入换出 → AG / RS → 分片 grad
TP/SP     模块计划 → DTensor 算子布局推导 / redistribute → 所需通信
PP        schedule → stage 的 F/B 与边界 P2P → microbatch 生命周期

共同依赖：进程组、通信顺序、缓冲区生命周期、正确的结果消费依赖
DeviceMesh 负责逻辑分组；DTensor 负责张量布局；PP schedule 负责执行时序。
它们不是三种同义的“自动并行”。
```

AG 指 all-gather，RS 指 reduce-scatter，AR 指 all-reduce，P2P 指点对点收发。各原语的数据语义复习见 [集合通信原语](./pytorch-collective-communication.md)。

## 3. 推荐精读顺序与完成标准

### 第 0 站：通信契约——一行 all_reduce 到底承诺了什么？

**先读一个简单测试，再追一次普通 eager 调用。**

1. [`distributed_c10d.py`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/distributed/distributed_c10d.py#L2835)：`all_reduce`，沿 `group.allreduce` 看参数与 `Work` 返回。
2. [`c10d/init.cpp`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/csrc/distributed/c10d/init.cpp)：Python 绑定；再看同目录的 `ProcessGroup.hpp`、`Backend.hpp` 和 `Work.hpp`，分清进程组对象与具体后端。
3. [`ProcessGroupNCCL.cpp`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L3556)：`collective` 中的 stream 选择、输入依赖、通信入队与事件。
4. 同文件 [`WorkNCCL::wait`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L775)：读 `synchronizeStream` 和 CPU 阻塞条件，不要只读函数名。

需要得到的结论：

- Store 主要承担发现和初始化协调，不是训练张量的数据通道。
- `async_op=True` 不等于“消费者可以立刻读结果”，也不等于“必然产生重叠”。
- CUDA/NCCL 下，默认非 blocking-wait 路径的 `Work.wait()` 主要给**调用它的当前流**建立对通信完成的依赖；开启 blocking wait、显式 timeout 等会改变主机等待行为，barrier 还有特殊处理。不能推广为“CPU 一定等到 GPU 完成”。
- **本提交**的 `collective` 对 `asyncOp=False` 使用当前流，对 `True` 使用专用 NCCL 流并建立输入依赖；这是版本相关实现，不应把“所有 NCCL 调用永远在单独流上”写成不变量。
- 同一进程组的 collective 必须跨 rank 匹配；多进程组还要核对创建与执行顺序，见 [`new_group` 契约](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/distributed/distributed_c10d.py#L5177)。

**测试入口**：[`test_c10d_gloo.py`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/test/distributed/test_c10d_gloo.py#L510) 的 `test_allreduce_basics` 用于看数值契约。CPU/Gloo 不能替代 NCCL stream 语义验证。

**完成标准**：能分别说明“主机返回、通信入队、GPU 完成、消费者流可安全使用”的区别，并画出生产者 → 通信 → 消费者的依赖。

### 第 1 站：DeviceMesh——5D 的逻辑轴怎么变成通信组？

先用一个手算例子确认方向，避免“行/列”口头描述歧义：

```text
mesh = [[0, 1, 2, 3],
        [4, 5, 6, 7]]
mesh_dim_names = ("dp", "tp")

沿 dp 维变化：{0,4}、{1,5}、{2,6}、{3,7}
沿 tp 维变化：{0,1,2,3}、{4,5,6,7}
rank 5：坐标 (1,1)，dp 组 {1,5}，tp 组 {4,5,6,7}
```

读 [`device_mesh.py`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/distributed/device_mesh.py#L384) 的 `DeviceMesh`，依次定位 `_init_process_groups`、`__getitem__`、`get_group`。

**重点问题**：为什么各 rank 必须给出一致的 mesh？`mesh["tp"]` 如何找到“当前 rank 所属”的 TP 子组？逻辑 rank 排列是否真的符合节点/NVLink 拓扑？

**测试入口**：[`test_device_mesh_nd`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/test/distributed/test_device_mesh.py#L340) 对照多维 mesh 的子组成员。

**完成标准**：不运行代码也能列出上述每个子组；有设备后，再打印各 rank 坐标与组成员验证。给 mesh 维度起名不会自动优化物理网络路径。

### 第 2 站：DDP / Reducer——先读最清楚的梯度事件驱动运行时

详细内容复用 [Reducer 类设计详解](./pytorch-reducer.md)，这里仅给最短主线。

- Python 入口：[`nn/parallel/distributed.py`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/nn/parallel/distributed.py#L1145)，`_ddp_init_helper` → `_pre_forward` / `_post_forward`。
- 默认 eager C++ Reducer 主线：[`reducer.cpp`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/csrc/distributed/c10d/reducer.cpp#L650)。本版本也有 Python reducer 等分支，首轮不混读。

```text
AccumulateGrad hook
  → autograd_hook(parameter_index)
  → mark_variable_ready：定位 bucket，递减 pending
  → mark_bucket_ready：受 next_bucket_ 顺序约束
  → all_reduce_bucket / comm hook
  → 引擎最终回调 finalize_backward
  → 消费归约结果并完成 grad 写回
```

**先问为什么，再看优化**：

1. 为什么梯度先就绪的后续 bucket 也可能必须等？答案不是“通信要串行完成”，而是跨 rank **发起顺序必须一致**。
2. `bucket.pending` 与 `next_bucket_` 分别控制什么？
3. 归约的求平均发生在哪里？用户 comm hook 接管后谁负责缩放？
4. [`no_sync`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/nn/parallel/distributed.py#L1424) 为什么必须覆盖 forward，而不只是 backward？

**测试入口**：[`test_c10d_gloo.py`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/test/distributed/test_c10d_gloo.py#L2325) 的 `test_forward_backward` 和 `test_forward_backward_unused_parameters`：观察手动构造 Reducer、准备反向与未使用参数处理。

**完成标准**：用“4 个参数、2 个桶，其中后一个桶先就绪”解释每次计数器变化、何时发通信、何时写回；再对照笔记里的交互演示。

### 第 3 站：DTensor——布局如何变成实际通信？

先建立最小词汇，再进入策略推导：

| placement | 同一 mesh 维上的含义 | 不能混淆为 |
|---|---|---|
| `Replicate()` | 各 rank 持同一逻辑值的完整副本 | 各自算出的不同 partial 值 |
| `Shard(dim)` | 沿张量的指定维度分片 | 沿同编号 mesh 维度切分；两类维度不是一回事 |
| `Partial("sum")` | 各 rank 的同形局部贡献，求和后才得到逻辑值 | 已经完整可用的复制结果 |

首轮只追一个 `redistribute`，再追一个矩阵算子：

```text
显式布局变换：DTensor.redistribute
              → Redistribute.forward
              → redistribute_local_tensor
              → placement 的具体转换 → functional collective

算子执行：DTensor.__torch_dispatch__
          → OpDispatcher 的布局传播/必要的重分片
          → 本地算子 → 用输出 spec 包装结果
```

源码入口：`tensor/_api.py`、`tensor/_dispatch.py`、`tensor/_redistribute.py`、`tensor/placement_types.py`。先看 [`Redistribute.forward/backward`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/distributed/tensor/_redistribute.py#L283) 与 [`Shard` 转换实现](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/distributed/tensor/placement_types.py#L192)，再看算子策略注册。

**应能独立推出的常见变换**（沿同一 mesh 维，先限于普通 placement）：

- `Shard → Replicate`：AG。
- `Partial(sum) → Replicate`：AR。
- `Partial(sum) → Shard`：RS。
- `Replicate → Shard`：已有完整副本时可本地切分，不必 AG。
- `Shard(某维) → Shard(另一维)`：需要重排；继续追具体后端路径与 fallback，不只背 A2A 标签。

**测试入口**：[`test_partial_to_replicate_forward_backward`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/test/distributed/tensor/test_redistribute.py#L171) 对照前向 AR 与反向实际保留的梯度布局。

**完成标准**：给出同一个张量的全局 shape、本地 shape、mesh、placements；分别解释前向变换和实际梯度 placement。不要用“前向通信反过来就是反向通信”代替 `Redistribute.backward` 的源码分析。更详细的反向例外见 [集合通信笔记](./pytorch-collective-communication.md)。

### 第 4 站：FSDP2——重点精读参数生命周期，而不只是 AG + RS

详细内容复用 [FSDP 源码阅读指南](./pytorch-fsdp.md)。这站建议花最多精力，按三层对象读：

| 对象 | 追踪什么 |
|---|---|
| `FSDPState` | 模块 hook、root/lazy init、最终回调 |
| `FSDPParamGroup` | 通信分组、unshard/wait/reshard、预取与梯度归约 |
| `FSDPParam` | 分片参数与计算参数、DTensor、视图和 storage 生命周期 |

**入口不要走错**：本提交的主实现是 [`fsdp/_fully_shard/_fully_shard.py`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/distributed/fsdp/_fully_shard/_fully_shard.py#L86)；旧 `_composable/fsdp/fully_shard.py` 只是兼容导入。

```text
fully_shard（构造期：收集尚未被内层组管理的参数、分片、注册状态）
  → 首次 forward lazy init，共享通信上下文
  → pre_forward：unshard → wait_for_unshard → 使用计算参数
  → post_forward：按策略 reshard，给输出挂 pre-backward hook
  → pre_backward：必要时 unshard，安排反向预取
  → post_backward：保留梯度引用、按策略 reshard、foreach_reduce
  → root final callback：兜底并完成收尾
```

主要读 [`_fsdp_state.py`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/distributed/fsdp/_fully_shard/_fsdp_state.py#L229)、[`_fsdp_param_group.py`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py#L263)、同目录 `_fsdp_param.py` 和 `_fsdp_collectives.py`。

**必须回答的六个问题**：

1. 为什么通常逐层、自底向上 `fully_shard`，而不是仅包根模块？分组怎样影响通信粒度与峰值显存？
2. `unshard` 与 `wait_for_unshard` 为什么分开？“已发起”与“计算可用”之间有哪些依赖？
3. 前向隐式预取与显式指定下一模块分别做什么？隐式重叠不是“当前 hook 已知道并调用下一层”。
4. 为什么不能随意替换或提前释放 autograd 已引用的参数对象/storage？
5. `reshard_after_forward=None`、`True`、`False` 为什么产生不同 AG 次数？`None` 对 root 和非 root 的默认行为不同。
6. 优化器看到哪个参数和哪个 `.grad`？分片优化器状态是如何随分片参数自然产生的，而非 FSDP 手工逐一拆解所有 optimizer 对象？

**把通信次数推到测试断言**：[`test_fully_shard_communication_count`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/test/distributed/_composable/fsdp/test_fully_shard_comm.py#L318) 按 Transformer block 分组并包根模块，覆盖 `True`、`False`、整数与 `None` 配置下的前反向 AG/RS 次数。先预测，再读断言。

**重叠测试**：[`test_fully_shard_training_overlap`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/test/distributed/_composable/fsdp/test_fully_shard_overlap.py#L27) 用延时模拟通信/计算，只做相对非重叠基线的 sanity check；不能把它读成实际模型吞吐保证。

**完成标准**：对一个非根参数组，逐时刻记录“模块上注册的参数、计算用参数、完整 storage 是否驻留、梯度在哪里、正在进行哪次通信”，再用 profiler 验证而非只数函数调用。

### 第 5 站：TP / SP——把 Megatron 数学结构对照到 ParallelStyle

读 [`parallelize_module`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/distributed/tensor/parallel/api.py#L16) → [`style.py`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/distributed/tensor/parallel/style.py#L45) 的 `ColwiseParallel` / `RowwiseParallel` / `SequenceParallel`，再回 DTensor 路径看通信如何产生。

- 对 `nn.Linear`，存储的 weight 轴序是“输出特征、输入特征”：Colwise 切 weight 的 dim-0，Rowwise 切 dim-1。不要把论文右乘矩阵的轴号原样套进实现。
- 每种 style 分开看：参数 partition、输入布局声明/转换、输出目标布局、是否 `to_local()`。
- Rowwise 的局部矩阵结果是求和 partial；输出设为 Replicate 需要 AR，设为序列分片则可走 RS。
- `SequenceParallel` 主要处理 LayerNorm/Dropout 等 token-local 区间，不等于 attention 的 CP。
- 普通 Tensor 经 `from_local(..., run_check=False)` 被解释为某种布局时，调用者必须保证数据已符合声明；API 不会替错误语义兜底。

**测试入口**：[`test_sequence_parallel_style`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/test/distributed/tensor/parallel/test_tp_style.py#L342) 对照归一化/Dropout 区间的通信计数、序列分片输出和参数梯度 placement；同文件也有 Colwise/Rowwise 测试。

**完成标准**：给“两层 Linear + 激活”列出前向每个边界的参数/激活布局；再加入 SP，指出通信位置为什么变化。与 [TP 详解](../parallelism/tensor-parallel-megatron.md) 对照，不再重复推导一遍矩阵公式。

### 第 6 站：PP——从调度表追到 stage、buffer 和 P2P

先读 GPipe 与单 stage/rank 的 1F1B，最后再看 interleaved / zero-bubble。

1. [`schedules.py`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/distributed/pipelining/schedules.py#L498)：`PipelineScheduleSingle.step` 的拆 microbatch、调度、合并输出。
2. 同文件 `ScheduleGPipe` / `Schedule1F1B` 的 `_step_microbatches`：warmup、steady state、cooldown 的实际收发与等待。
3. [`stage.py`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/distributed/pipelining/stage.py#L1238)：`PipelineStage` 和基类里的 forward/backward chunk、缓存、收发信息。
4. 进阶才看 `_IR.py` 的自动切分、`_backward.py` 的输入/权重梯度拆分。

**重点问题**：microbatch 标号怎样对应缓存？stage 编号与 rank 是否总是一一对应？激活何时能释放？loss reduction、microbatch 数与梯度缩放是否匹配？

**测试入口**：[`test_grad_with_manual`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/test/distributed/pipelining/test_schedule_multiproc.py#L284) 用手动 stage 与参考模型比较输出、loss 和梯度。注意本提交 `Schedule1F1B` 的稳态代码按 1B1F 组织，名称不能代替逐操作追踪。

**完成标准**：手推“2 stage、4 microbatch”的 GPipe 与 1F1B 操作序列；对每个 receive 标明对应 send、消费它的计算以及保存/释放的激活。一个 mesh 维度叫 PP，不会自动生成这些依赖。已有动画和训练示例见 [流水线并行笔记](../parallelism/pipeline-parallelism.md)。

### 第 7 站：组合验证，而不是继续增加名词

用 [TP 笔记中的 TP×FSDP 示例](../parallelism/tensor-parallel-megatron.md)，对照 [`TestFullyShard2DTraining.test_train_parity_2d_mlp`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/test/distributed/_composable/test_composability/test_2d_composability.py#L118) 与 [`MLPStack.parallelize()`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/testing/_internal/common_fsdp.py#L926) 回答。这是本轮 FSDP2 组合的入口；不要混同于使用 FSDP1 wrapper 的 `test/distributed/fsdp/test_fsdp_tp_integration.py`：

1. 为什么此实现先 TP，再沿 DP 子 mesh 自底向上 FSDP？
2. FSDP all-gather 恢复的是“当前 TP 分片对应的计算参数”，还是整个全局模型参数？
3. 同一个参数的全局 shape、TP 本地 shape、FSDP 持久分片 shape 分别是什么？
4. 不同 DP 副本喂不同数据、同一 TP 组协作处理同一输入的约束在哪里落实？
5. 加入 PP 后，当前 rank 只持哪些 stage，DP 同步和 TP 通信分别发生在哪个子组？

**完成标准**：画清 rank 分组与参数生命周期；验证单步 loss、还原后的梯度/参数更新与参考实现一致，再讨论峰值显存与耗时。数值比较要声明随机种子、loss 缩放、dtype 和容差，不要求浮点结果逐 bit 相同。

## 4. CP / EP：明确源码能回答与不能回答的事

- **CP**：入口为 `torch.distributed.tensor.experimental.context_parallel`，实现集中于 `tensor/experimental/_attention.py`。重点追 buffer 序列切分、SDPA 接管、KV 轮转与结果合并、上下文退出恢复。具体接口与限制以本提交源码为准，见 [SP/CP 笔记](../parallelism/sequence-context-parallelism.md)。
- **EP**：从 `distributed_c10d.py` 的 `all_to_all_single` 可研究 split 和数据交换契约，但这不包含 router、token permutation、专家计算、combine 与辅助损失的完整系统。DTensor 重分片也不自动变成 MoE dispatcher，见 [专家并行笔记](../parallelism/expert-parallel-moe.md)。完整 EP 实现应另选一个固定版本的训练框架再开精读，不在本页臆造入口。

## 5. 每轮阅读如何沉淀到 SB

已有笔记是主题正文，本页只做跨主题路线；不要再建一套同义的 DDP/FSDP 综述。

| 收获类型 | 沉淀位置 |
|---|---|
| 通信语义、Work/stream 契约 | [集合通信原语](./pytorch-collective-communication.md) |
| bucket / hook / 归约顺序 | [Reducer](./pytorch-reducer.md) |
| 参数状态、预取、storage 生命周期 | [FSDP](./pytorch-fsdp.md) |
| TP/SP/CP/PP/EP 的概念与实现对照 | [5D 系列](../parallelism/index.md) 对应主题 |
| trace、显存、真实瓶颈 | [分布式 profiling](../../../engineering/profiling/07-pytorch-distributed-profiling.md) |
| 尚无专文且已形成独立理解的机制 | 搜索确认无重复后，再拆成原子笔记并注册导航 |

每次建议只深挖一个问题，留下下面六项；未运行的实验只写预测，不写成“我的经验”：

```text
问题：我现在无法解释的一个具体行为
基线：源码 commit / 安装版本 / 环境
源码证据：文件 + 符号 + 关键条件分支
我的解释：为什么这样设计，哪些条件下不成立
验证：输入、预期、实际、断言或 trace；未运行则明确写未运行
关联 / 待解：更新哪篇既有笔记，下一条需要验证什么
```

### 第一轮就从这里开始

- [ ] 手算第 1 站的 rank 分组，确认 mesh 维与 tensor 维不混用。
- [ ] 精读 `all_reduce` → 后端入队 → `WorkNCCL.wait`，建立异步语义。
- [ ] 精读 Reducer 主链，用两个 bucket 解释按序发起与最终写回。
- [ ] 最小掌握 DTensor 三种 placement，随后进入 FSDP2 生命周期。

**暂不标记后续阶段“已掌握”**：读过笔记、定位到源码、解释出不变量、运行验证，是四个不同进度。
