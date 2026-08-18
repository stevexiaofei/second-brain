---
title: 缓存机制
type: concept
status: seed
tags: [PyTorch, torch.compile, 缓存]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\pytorch-2.8.0\wiki\05_cache.html
---

# 缓存机制

> 多版本编译结果的存储与查找

Dynamo 的缓存机制允许同一个函数在不同输入条件下拥有多个编译版本，避免不必要的重编译。缓存以 **code object 为单位**组织，每个 code object 维护一条 `CacheEntry` 链表。核心实现位于 `torch/_dynamo/cache_size.py` 与 C 扩展 `torch/_C/_dynamo/eval_frame`。

## 💾 5.1 缓存结构

每个 Python 函数对应一个 `code object`，其 `co_extra` 暂存区挂载 Dynamo 的缓存链表。链表节点是 `CacheEntry`，封装了一次编译产物与其 Guard 检查器。

```text
Code Object (每个 Python 函数对应一个)
  │
  └─ co_extra 暂存区
       │
       ├─ cache_entry 链 (链表，按编译顺序排列)
       │    │
       │    ├─ CacheEntry[0]  (首次编译)
       │    │    ├─ GuardedCode
       │    │    │    ├─ code: 自定义字节码 (编译后的)
       │    │    │    └─ check_fn: Guard 检查函数 (C 树)
       │    │    ├─ guard_manager: GuardManagerWrapper (root + diff_root)
       │    │    ├─ extra_state: 额外状态 (Dynamo 运行时)
       │    │    └─ next: → CacheEntry[1]
       │    │
       │    ├─ CacheEntry[1]  (第一次重编译的结果)
       │    │    └─ GuardedCode { code, check_fn }, next → ...
       │    │
       │    └─ CacheEntry[N]  → next = NULL
       │
       └─ cache_size_info: 缓存大小追踪 (CacheSizeRelevantForFrame)
```

### CacheEntry 的关键字段

- **GuardedCode**：包含 `code`（Dynamo 重写后的自定义字节码）和 `check_fn`（Guard 检查入口）。
- **guard_manager**：`GuardManagerWrapper` 实例，持有 `RootGuardManager` 树与 `diff_guard_root`，是 C 层检查的真正主体。
- **extra_state**：Dynamo 在该缓存项上挂载的运行时状态，例如 CUDA Graph 句柄、自动调优缓存键等。
- **id_matched_objs**：记录此缓存项 ID_MATCH 的对象弱引用，用于缓存大小计数（见 5.4）。
- **next**：指向下一个 `CacheEntry`，构成链表。

## 🔗 5.2 CacheEntry 与 co_extra 的挂载

CPython 的 `PyCodeObject` 提供了 `co_extra` 暂存区，允许第三方存储自定义数据。Dynamo 在 C 层把缓存链表头指针写入该区域。这样帧评估钩子在拦截到该 code object 时，能 O(1) 拿到缓存链表。

> **💡 提示：** 为什么挂在 code object 上？Python 中函数是"一等公民"，但其 `__code__` 才是真正的字节码载体。多个函数/lambda 若共享同一 code object（例如闭包工厂产出的函数），它们也共享同一份缓存——这是缓存能跨函数实例复用的根本原因。

## 🔍 5.3 缓存查找流程

帧评估钩子触发后，C 层遍历 `CacheEntry` 链表，对每项执行 Guard 检查函数：

```c
# eval_frame.c 伪代码
PyObject* eval_frame_callback(frame, ...) {
    code = frame->f_code;

    # 取出 co_extra 上的缓存链表头
    CacheEntry* entry = code->cache_entry;
    while (entry != NULL) {
        # 执行 Guard 检查函数 (C 树, 无 pybind 开销)
        if (entry->guarded_code.check_fn(frame)) {
            # Guard 通过！使用此缓存的字节码
            return entry->guarded_code.code;
        }
        entry = entry->next;
    }

    # 无缓存命中 → 调用 Python 层 callback 进行编译
    return call_python_callback(frame);
}
```

### 缓存查找决策树

```text
帧评估钩子触发
    │
    ▼
取 code object 的 co_extra 缓存链表
    │
    ├─ 链表为空 (无缓存)?
    │   └─ YES → 调用 Python callback 进行首次编译
    │            → 新建 CacheEntry 写入链表头
    │
    ▼
遍历 CacheEntry 链表
    │
    ├─ 对每个 entry 执行 Guard 检查 (C 函数)
    │   │
    │   ├─ 通过 → 命中! 返回此 entry 的 out_code (快速路径)
    │   │
    │   └─ 失败 → 继续下一个 entry
    │
    ▼
所有 entry 都不命中?
    │
    ├─ 计算当前 cache_size (见 5.4)
    │   │
    │   ├─ 超过 accumulated_recompile_limit (默认 64)
    │   │   → 放弃编译, 回退 eager
    │   │
    │   ├─ 超过 recompile_limit (默认 8, 对相同 ID_MATCH 对象)
    │   │   → 放弃编译, 回退 eager
    │   │
    │   └─ 未超过限制
    │       → 重新编译, 创建新 CacheEntry 加入链表
    │
    ▼
执行 (编译后字节码 / eager)
```

## 📊 5.4 cache_size 追踪与 recompile_limit

为了防止无限重编译，Dynamo 对每个 code object 的缓存大小设限。追踪逻辑位于 `cache_size.py`。这里有一个关键设计：**双重限制**。

> **📝 备注：** 为什么需要两个限制？详见 `cache_size.py` 顶部注释。简言之：当存在图断裂时，Dynamo 会对 `nn.Module` 实例加 `ID_MATCH` Guard。如果一个模型有 16 个模块实例、且输入 batch size 各异，单是 forward 这一个 code object 就可能产生 32 个缓存项。若只有一个上限，要么设得很高（导致编译不友好的函数也要 32 次才回退），要么设得很低（模块实例场景提前失效）。两个限制解开了这个矛盾。

### 两个限制

- **`recompile_limit`**（默认 8）：对**同一组 ID_MATCH 对象**的最大缓存项数。即同一个模块实例最多重编译几次。
- **`accumulated_recompile_limit`**（默认 64）：单个 code object 上缓存项的**累计上限**，无论 ID_MATCH 对象是否相同。

追踪数据结构是 `CacheSizeRelevantForFrame`：

```python
@dataclass
class CacheSizeRelevantForFrame:
    """
    追踪与给定 frame 具有相同 id_match 对象的缓存项数量。
    """

    # Dynamo 链表中 CacheEntry 的总数
    num_cache_entries: int = 0

    # 与当前 frame 的 ID_MATCH 对象相同的 CacheEntry 数
    num_cache_entries_with_same_id_matched_objs: int = 0

    def will_compilation_exceed(self, limit: int) -> bool:
        # 任一限制达到都算超限 (所以用 >=)
        return (
            self.will_compilation_exceed_accumulated_limit()
            or self.will_compilation_exceed_specific_limit(limit)
        )

    def will_compilation_exceed_accumulated_limit(self) -> bool:
        return self.num_cache_entries >= config.accumulated_recompile_limit

    def will_compilation_exceed_specific_limit(self, limit: int) -> bool:
        return self.num_cache_entries_with_same_id_matched_objs >= limit
```

计算过程：`compute_cache_size` 遍历整个缓存链表，统计总数 `num_cache_entries`，并对每项调用 `_has_same_id_matched_objs` 比较 ID_MATCH 对象的弱引用是否一致，得到 `num_cache_entries_with_same_id_matched_objs`。`exceeds_recompile_limit` 据此判断是否应放弃编译。

> **✨ 技巧：** 示例：假设 `recompile_limit=2`、`accumulated_recompile_limit=32`，模型有 16 个 ID_MATCH 的模块实例。每个实例最多产生 2 个缓存项（受 `recompile_limit` 限制），而累计上限 32 足以容纳所有实例的缓存。但对于一个"编译不友好"的动态函数（无 ID_MATCH），只需 2 次重编译就会回退 eager——这正是双限制的价值。

## 🧹 5.5 缓存的失效与清理

缓存项的生命周期与 code object 绑定，但也会因以下原因被失效或清理：

- **对象回收：** 当 ID_MATCH 的 `nn.Module` 实例被垃圾回收，其弱引用失效。`guard_manager` 持有的弱引用变空，相关缓存项实际上不再可命中。这也是 `accumulated_recompile_limit` 检查之外的一道兜底。
- **guard_manager 失效：** 注释中明确提到："this check is needed in the case that the frame's cache doesn't grow and we keep recompiling. This can happen if the guard guard_manager becomes invalidated, e.g. due to guarded objects being freed." 即被守卫对象释放会导致 manager 失效，需要重编译。
- **code object 释放：** 当函数本身被回收，`co_extra` 随之释放，缓存链表整体回收。
- **显式重置：** `torch._dynamo.reset()` 清空所有缓存与编译上下文，用于调试或在运行时切换编译策略。

`extra_state` 上挂载的资源（如 CUDA Graph）也会在缓存项失效时一并释放，避免显存泄漏。

## 🧵 5.6 缓存与多线程

Dynamo 的缓存以 code object 为中心，而 code object 在 Python 中是**进程级共享**的不可变对象。这带来几个多线程层面的特点：

- **缓存天然共享：** 同一进程内多个线程执行同一函数时，访问的是同一个 `co_extra` 缓存链表。一个线程编译出的结果，其他线程可立即复用（只要 Guard 通过）。
- **编译串行化：** 为避免多线程并发触发同一 code object 的重复编译，Dynamo 使用编译锁（`CompileContext` / `CompileId`）串行化编译过程。首个线程编译，其余线程等待并复用结果。
- **Guard 检查可并发：** Guard 检查是只读操作，C 层 `check_nopybind` 无写共享状态，多线程可安全并发执行缓存查找。
- **计数非原子风险：** `compute_cache_size` 遍历链表是只读操作，但写新缓存项到链表头需要同步，由编译锁保证。

> **⚠️ 注意：** 实践建议：在 DataLoader 多进程（`num_workers > 0`）场景下，子进程会 fork/spawn 新解释器，缓存**不跨进程共享**。每个 worker 进程需要各自编译。若希望预热缓存以减少首次编译开销，可在主进程执行一次 warmup，但 worker 仍需各自编译——考虑使用 `torch._inductor.compile_fx` 的持久化缓存（`torch._inductor.config.cache_dir`）落盘共享。

## 📌 5.7 关键设计要点回顾

### per code object

缓存挂在 code object 的 `co_extra` 上，共享同一 code object 的函数共享缓存。

### 链表 + Guard

每个缓存项 = 字节码 + Guard 检查树。查找 = 遍历链表逐项检查 Guard。

### 双重限制

`recompile_limit`（同 ID_MATCH 对象）+ `accumulated_recompile_limit`（累计）兼顾复用与早回退。

### C 层快速路径

Guard 检查在 C 层 `check_nopybind` 完成，无需进入 Python，命中耗时微秒级。

> **💡 提示：** 缓存是 per code object 的：如果动态创建多个相同函数的副本（如 lambda），它们共享同一个 code object 和缓存。缓存管理代码位于 `cache_size.py`。

## Related

- [04 Guard 系统](./04-guard-system.md)
- [PyTorch 索引](../index.md)
