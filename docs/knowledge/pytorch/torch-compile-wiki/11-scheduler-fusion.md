---
title: torch.compile 调度器与融合
type: concept
status: seed
tags: [PyTorch, torch.compile, 调度器, 融合]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\pytorch-2.8.0\wiki\11_scheduler.html
---

# 十一、调度器与融合

*算子融合与执行顺序优化*

## 11.1 调度器概述

调度器是 Inductor 的优化核心，定义于 `scheduler.py#L2015`。它接收 Lowering 阶段产出的 IR 操作列表（`V.graph.operations`），执行 **融合 (Fusion)**、**重排 (Reorder)** 与 **图分区 (Partition)** 三类优化。

> **💡 提示：调度器的定位：**如源码注释所言——"A Scheduler is a graph of BaseSchedulerNodes. It is responsible for optimizations such as fusion, reorder, and graph partition." 它把 IR 操作图转换为 `BaseSchedulerNode` 图，在此基础上做融合与排序，最终交给代码生成器逐节点生成内核。

## 11.2 Scheduler.__init__ 流程

定义于 `scheduler.py#L2023`。整个初始化在 `dynamo_timed("Scheduler.__init__")` 计时下进行：

*torch/_inductor/scheduler.py#L2015*

```python
class Scheduler:
    """
    A Scheduler is a graph of BaseSchedulerNodes. It is responsible for
    optimizations such as fusion, reorder, and graph partition.
    """

    __dep_size_hint_cache: dict[Dep, int]

    def __init__(self, nodes: list[ir.Operation]) -> None:
        with dynamo_timed("Scheduler.__init__"):
            self._init(nodes)

    def _init(self, nodes: list[ir.Operation]) -> None:
        super().__init__()
        self.__dep_size_hint_cache = {}
        V.graph.scheduler = self
        self.backends: dict[torch.device, BaseScheduling] = {}
        self.post_grad_graph_id = next(_post_grad_graph_counter)
        self._graph_partition_counter = itertools.count()

        self.completed_operations: OrderedSet[str] = OrderedSet()
        self.available_buffer_names = OrderedSet([
            *V.graph.graph_inputs.keys(),
            *V.graph.constants.keys(),
            *V.graph.torchbind_constants.keys(),
        ])

        # 1. 将 IR 节点包装为 SchedulerNode
        self.nodes = [self.create_scheduler_node(n) for n in nodes]
        self.update_zero_dim_cpu_tensor()
        self.available_buffer_names.update(V.graph.constants.keys())

        # 2. 修剪已满足的依赖
        for node in self.nodes:
            node.prune_deps()

        # 3. 收集 donated buffers (内存复用)
        self.name_to_donated_buffer: dict[str, SchedulerDonatedBuffer] = (
            self.get_donated_buffers()
        )

        # 4. 构建名称查找表
        self.name_to_node: dict[str, BaseSchedulerNode] = {
            n.get_name(): n for n in self.nodes
        }
        self.name_to_buf: dict[str, SchedulerBuffer] = {
            buf.get_name(): buf
            for node in self.nodes
            for buf in node.get_outputs()
        }
        self.name_to_fused_node: dict[str, BaseSchedulerNode] = self.name_to_node.copy()

        # 5. 处理 mutation: 重命名被修改的缓冲区以防止依赖图成环
        self.mutation_real_name: dict[str, str] = {}
```

### 11.2.1 create_scheduler_node：IR → SchedulerNode

根据 IR 节点类型选择不同的调度节点包装：

*torch/_inductor/scheduler.py#L2183*

```python
def create_scheduler_node(self, node: ir.Operation) -> BaseSchedulerNode:
    assert node.get_origins() is not None, (
        "All nodes passed to scheduling must have an origin"
    )
    if node.is_no_op():
        return NopKernelSchedulerNode(self, node)
    elif isinstance(node, (ir.ComputedBuffer, ir.TemplateBuffer)):
        return SchedulerNode(self, node)
    elif isinstance(node, ir.ExternKernel):
        return ExternKernelSchedulerNode(self, node)
    else:
        raise NotImplementedError(node)
```

| IR 节点类型 | 调度节点类 | 说明 |
| --- | --- | --- |
| no_op | `NopKernelSchedulerNode` | 空操作，不生成代码 |
| `ComputedBuffer` | `SchedulerNode` | 常规计算缓冲（Pointwise/Reduction 物化） |
| `TemplateBuffer` | `SchedulerNode` | 模板内核（matmul/conv + epilogue） |
| `ExternKernel` | `ExternKernelSchedulerNode` | 外部算子回退（如 cuBLAS 调用） |

## 11.3 依赖追踪

调度器需要精确知道每个节点依赖哪些缓冲区、又被哪些节点消费，才能安全地融合与重排。

### 11.3.1 prune_deps：修剪已满足的依赖

初始化时对每个节点调用 `node.prune_deps()`，移除那些依赖项已在 `available_buffer_names`（图输入、常量、torchbind 常量）中的"已满足依赖"。这样后续分析只关注 **未满足** 的依赖，即必须等待某个内核产出后才能消费的真依赖。

### 11.3.2 name_to_buf：缓冲区查找表

构建 `name_to_buf` 字典，把每个输出缓冲区的名字映射到其 `SchedulerBuffer`：

*name_to_buf 构建*

```python
self.name_to_buf: dict[str, SchedulerBuffer] = {
    buf.get_name(): buf
    for node in self.nodes
    for buf in node.get_outputs()
}
```

由于一个 SchedulerNode 可能有多个输出（如 split、多输出模板），所以对每个节点的 `get_outputs()` 展开。该表是依赖图的核心：当节点 A 的 reads 中出现 `buf_x`，调度器通过 `name_to_buf["buf_x"]` 找到生产者节点，建立 A → 生产者的依赖边。

### 11.3.3 mutation_real_name：变异重命名

当某个缓冲区被 inplace 修改（如 `buf0` 在 `buf1` 的内核中被写入），为避免依赖图出现环，调度器把被修改的版本重命名：`mutation_real_name = {"buf0": "buf1"}`。此后所有对 `buf0` 的引用在依赖图中都映射为对 `buf1` 的引用，从而把"修改"转化为"生产"，保持依赖图的无环性。

> **⚠️ 注意：依赖类型：**调度器区分 `MemoryDep`（具体内存位置依赖，带 index/var_names/size/mode）与 `StarDep`（"整块"依赖，用于外部算子等不暴露内部索引的情况）。模板融合时还会把 `StarDep` 改写为 `MemoryDep` 以统一评分。

## 11.4 融合算法

融合的核心目标是把多个 **可融合的** SchedulerNode 合并为一个 `FusedSchedulerNode`，使它们在同一个内核中执行，中间结果留在寄存器中而不写回显存。

### 11.4.1 融合条件

两个节点能否融合，取决于多个维度：

- **设备一致：**所有节点必须在同一设备上（CPU/GPU 分开调度）。
- **形状兼容：**Pointwise 融合要求循环范围 (ranges) 一致或可广播。
- **依赖关系允许：**融合后不能引入环；通常 Producer-Consumer 关系且无其他冲突依赖时可融合。
- **读写冲突：**不能同时写同一缓冲区；读-写需通过依赖排序保证正确性。
- **算子类型匹配：**见 11.5 融合规则表。
- **性能评分：**`score_fusion_memory` 评估融合是否真正节省内存读写；若融合反而增加寄存器压力，可能放弃。

### 11.4.2 融合过程

调度器按拓扑顺序遍历节点，对每个节点尝试与已有融合组或其他节点合并：

1. 计算每个节点的读/写集合 (`read_writes`)。
2. 对每个待融合节点，找到其依赖的生产者节点，评估能否并入生产者的融合组。
3. 调用 `FusedSchedulerNode.fuse(node1, node2)` 合并，更新 `name_to_fused_node`。
4. 融合组的未满足依赖 = 各成员节点未满足依赖的并集。
5. 当某融合组的所有依赖都已满足（`unmet_deps` 为空），即可交给代码生成器生成内核。

## 11.5 FusedSchedulerNode 与调度节点类型

定义于 `scheduler.py#L1300`。`FusedSchedulerNode` 是一个"伪节点"，代表一组待融合的 SchedulerNode。

*torch/_inductor/scheduler.py#L1300*

```python
class FusedSchedulerNode(BaseSchedulerNode):
    """
    This is a "fake" scheduler node that represents a group of scheduler nodes
    that are meant to be fused together. The way it does this is by maintaining
    its unmet dependencies as the union of its constituent nodes.
    """

    snodes: list[BaseSchedulerNode]

    @classmethod
    def fuse(cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> FusedSchedulerNode:
        assert node1.scheduler is node2.scheduler
        assert isinstance(node1, (SchedulerNode, FusedSchedulerNode))
        ...
        # 模板 + 外部多输出: 把 StarDep 改写为 MemoryDep 以统一评分
        if node1.is_template() and isinstance(node2, ExternKernelSchedulerNode):
            assert isinstance(node2.node, MultiOutput)
            ...  # 改写依赖类型
        nodes = list(itertools.chain(node1.get_nodes(), node2.get_nodes()))
        return cls(node1.scheduler, nodes)
```

**FusedSchedulerNode 的职责：**

- **成员管理：**持有 `snodes` 列表，通过 `get_nodes()` 展开所有成员。
- **依赖合并：**未满足依赖取成员的并集，保证融合组在所有成员的依赖都满足后才执行。
- **FLOPs 估算：**`estimate_flops` 汇总成员中模板/外部算子的 FLOPs，但避免对融合方法重复计数。
- **循环重排：**`reorder_loops_by_dep_pair` 在融合后调整循环顺序以匹配依赖对的访问模式（模板节点除外）。

| 调度节点类 | 文件位置 | 职责 |
| --- | --- | --- |
| `BaseSchedulerNode` | L194 | 基类，管理依赖、缓冲区使用、融合接口 |
| `SchedulerNode` | L1003 | 单个 IR 操作（ComputedBuffer/TemplateBuffer）的调度节点 |
| `FusedSchedulerNode` | L1300 | 融合后的节点，包含多个 SchedulerNode |
| `GroupedSchedulerNode` | L1850 | 分组调度节点（如 split 的多输出） |
| `ExternKernelSchedulerNode` | — | 外部算子回退节点 |
| `NopKernelSchedulerNode` | — | 空操作节点，不生成代码 |

## 11.6 融合规则表

| 融合类型 | 条件 | 示例 |
| --- | --- | --- |
| Pointwise + Pointwise | 相同设备、相同形状（或可广播）、依赖关系允许 | `(a + b) * c` → 单个内核 |
| Reduction + Pointwise | 归约后接元素级操作，归约结果作为 pointwise 的输入 | `sum(x).relu()` |
| Pointwise + Reduction | 元素级操作后接归约（部分情况，需无其他消费者） | `(x * 2).sum()` |
| 模板 + Pointwise (epilogue) | matmul/conv 后接 pointwise，作为 epilogue 融入模板 | `matmul(x, w) + bias + relu` |
| 模板 + MultiOutput | 模板的多输出与外部多输出节点融合（StarDep→MemoryDep） | 多输出 matmul 的各输出后续 pointwise |
| Reduction + Reduction | 同维度归约，无数据依赖冲突 | `x.sum(0) + x.max(0)` |
| 不融合：跨设备 | 节点位于不同设备 | CPU 节点与 GPU 节点 |
| 不融合：形状冲突 | 循环范围不兼容且无法广播 | 不同 `ranges` 的 Pointwise |

## 11.7 内存复用 (Memory Reuse) 与 donated buffers

调度器通过 `donated buffers` 机制实现内存复用，减少峰值显存占用。定义于 `scheduler.py#L2151`：

*torch/_inductor/scheduler.py#L2151*

```python
def get_donated_buffers(self) -> dict[str, SchedulerDonatedBuffer]:
    name_to_donated_buf = {}
    for name in V.graph.graph_inputs_original:
        if isinstance(V.graph.graph_inputs_original[name], ir.DonatedBuffer):
            name_to_donated_buf[name] = SchedulerDonatedBuffer(
                self,
                V.graph.graph_inputs_original[name],
                defining_op=None,
            )
    return name_to_donated_buf
```

**donated buffer 的含义：**一个图输入缓冲区被标记为 `DonatedBuffer`，表示 **该输入在图执行完毕后不再被外部使用**，因此其存储可以被图内其他缓冲区复用（覆盖写入）。调度器把它们收集到 `name_to_donated_buffer`，在分配新缓冲区时优先复用这些 donated 存储。

- **典型场景：**反向图中，前向保存的某些激活在反向用完后即可 donate；前向图的临时中间结果在产出下游后也可 donate。
- **收益：**避免为每个新输出分配新显存，降低峰值内存。
- **安全保证：**只有确认输入在图外不再被引用（如非用户可见输出、非模型参数）才会被标记为 donated，避免误覆盖用户张量。

> **✨ 技巧：内存复用策略：**调度器在代码生成阶段会综合 donated buffers、缓冲区生命周期（通过依赖图推导）与 `buffer_to_padded_size`（inplace padding）信息，把多个不重叠生命周期的缓冲区映射到同一块物理内存，显著降低显存峰值。

## 11.8 融合前后对比

下面以具体 IR 节点展示融合前后的差异：

**融合前：4 个独立内核，4 次显存读写**

```text
IR 节点序列 (Lowering 产出):
  ComputedBuffer[buf1] = Pointwise(A, B):  C = A + B      → 内核1
  ComputedBuffer[buf2] = Pointwise(buf1):  D = C * 2      → 内核2
  ComputedBuffer[buf3] = Pointwise(buf2):  E = relu(D)    → 内核3
  ComputedBuffer[buf4] = Reduction(buf3):  F = sum(E)     → 内核4

执行流:
  内核1: load A, load B → C = A+B   → store buf1 (写显存)
  内核2: load buf1       → D = C*2   → store buf2 (读+写显存)
  内核3: load buf2       → E = relu  → store buf3 (读+写显存)
  内核4: load buf3       → F = sum   → store buf4 (读+写显存)

  显存读写: 4 次写 + 3 次读 = 7 次访存; 4 次内核启动
```

**▼ 调度器融合**

**融合后：1 个融合内核，中间结果留在寄存器**

```text
FusedSchedulerNode {
  SchedulerNode[Pointwise A+B]
  SchedulerNode[Pointwise *2]
  SchedulerNode[Pointwise relu]
  SchedulerNode[Reduction sum]
}

  → 代码生成器产出单个 Triton/C++ 内核:

  内核1: load A, load B
         ├─ tmp1 = A + B          (寄存器)
         ├─ tmp2 = tmp1 * 2       (寄存器)
         ├─ tmp3 = relu(tmp2)     (寄存器)
         └─ F = sum(tmp3)         (归约)
         → store F                (写显存)

  显存读写: 1 次写 + 2 次读 = 3 次访存; 1 次内核启动
  buf1/buf2/buf3 从未物化到显存 → 节省 3 个缓冲区

收益:
  - 内存带宽减少 ~57% (3/7)
  - 内核启动开销减少 ~75% (1/4)
  - 峰值显存降低 (中间缓冲不分配)
```

## 11.9 融合的收益

```text
未融合 (4 个内核, 4 次内存读写):
  内核1: load A, load B → C = A + B   → store C
  内核2: load C          → D = C * 2  → store D
  内核3: load D          → E = D.relu()→ store E
  内核4: load E          → F = E.sum() → store F

融合后 (1 个内核, 1 次内存读写):
  内核1: load A, load B → F = relu(A + B).sum()  → store F
         └─ 中间结果 C, D, E 留在寄存器中，无需写回内存 ─┘

收益: 内存带宽减少 ~75%, 内核启动开销减少 ~75%
```

对于 memory-bound 的算子（如大量 pointwise 堆叠），融合能带来数倍加速；对于计算密集型算子（matmul/conv），则主要通过 epilogue 融合与模板自动调优获益。融合后的 `FusedSchedulerNode` 列表随后进入 [第十二章 代码生成](./12-code-generation.md)，由 Triton/C++ 后端逐节点生成最终内核代码。

## Related

- [10 Lowering: FX → IR](./10-lowering-fx-to-ir.md) — Lowering 产出 IR 操作列表，调度器在此之上做融合与排序
- [12 代码生成](./12-code-generation.md) — 融合后的 FusedSchedulerNode 交给代码生成器生成内核
- [PyTorch 索引](../index.md)
