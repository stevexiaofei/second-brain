---
title: Fenwick Tree（树状数组）与动态加权采样
type: concept
status: seed
tags: [Algorithm, Data Structure, Fenwick Tree, BIT, Weighted Sampling, O(logN)]
created: 2026-08-12
updated: 2026-08-12
---

# Fenwick Tree（树状数组）与动态加权采样

## 一句话理解

Fenwick Tree（二叉索引树 BIT）用**索引的二进制本身**作为存储结构，通过 `lowbit(i) = i & -i` 把"任意前缀和"拆成 $O(\log N)$ 个互不相交的区间，使**单点更新**和**前缀和查询**都达到 $O(\log N)$，从而支持 $O(\log N)$ 的**动态加权采样**（按权重比例随机选元素）。

## 为什么重要

- **加权采样**是遗传算法轮盘赌、LLM token 采样、RL 优先级经验回放（Prioritized Replay）的底层数据结构
- 朴素方案做不到"更新快 + 查询快"同时成立，BIT 是两者兼得的最小实现
- **代码极简**：没有显式建树，靠 `i & -i` 运算隐式生成树结构，是"用数学性质省掉数据结构"的典范

## 朴素方案的困境

| 方案 | 单点更新 | 前缀和查询 |
|---|---|---|
| 普通数组 | $O(1)$ | $O(N)$ |
| 前缀和数组 | $O(N)$（后面全要改） | $O(1)$ |
| **BIT** | $O(\log N)$ | $O(\log N)$ |

BIT 的目标：同时把更新与查询压到 $O(\log N)$。

## 核心概念：`i & -i`（lowbit）

对任意正整数 $i$，`-i` 是 `i` 的补码（取反+1）：

$$i \ \&\ -i \ =\ i \text{ 二进制中最低位的 1 所代表的数值}$$

```text
i=6  = 110₂ → 6&-6 = 010₂ = 2    (最低位 1 在第 2 位)
i=5  = 101₂ → 5&-5 = 001₂ = 1    (最低位 1 在第 1 位)
i=12 = 1100₂ → 12&-12 = 100₂ = 4 (最低位 1 在第 3 位)
```

`lowbit(i)` 告诉你"节点 i 管辖多长的区间"。

### lowbit 的数学性质与证明

**性质 1（lowbit 是 2 的幂）**：设 $i$ 的二进制中最低位 1 出现在第 $k$ 位，即 $i = b \cdot 2^{k+1} + 2^k$（$b \ge 0$），则

$$\text{lowbit}(i) = i \mathbin{\&} (-i) = 2^k$$

**证明**：$i$ 的补码 $-i = (\sim i) + 1$。由于 $i$ 的低 $k$ 位全为 $0$，$(\sim i)$ 的低 $k$ 位全为 $1$，加 $1$ 后进位到第 $k$ 位，因此 $-i$ 的低 $k$ 位全为 $0$、第 $k$ 位为 $1$。按位与 $i \mathbin{\&} (-i)$ 只在第 $k$ 位同时为 $1$，故结果为 $2^k$。□

**性质 2（管辖区间）**：节点 $i$ 的管辖区间为 $\big(i - \text{lowbit}(i),\ i\big]$，区间长度恰为 $\text{lowbit}(i)$。

**性质 3（父节点）**：$i$ 的父节点是 $f(i) = i + \text{lowbit}(i)$，且父节点管辖区间严格包含子节点管辖区间（引理 A，见下节证明）。

## BIT 的定义：tree[i] 管什么

BIT 下标从 1 开始（代码里 `tree` 长度是 `size+1`）：

> **$tree[i]$ 存储原始数组区间 $(i - lowbit(i),\ i]$ 的和**

以 size=8 为例：

```text
index:   1     2     3     4     5     6     7     8
lowbit:  1     2     1     4     1     2     1     8

tree[1] = a[1]
tree[2] = a[1]+a[2]
tree[3] = a[3]
tree[4] = a[1]+a[2]+a[3]+a[4]
tree[5] = a[5]
tree[6] = a[5]+a[6]
tree[7] = a[7]
tree[8] = a[1..8]
```

树状结构：

```text
            ┌──────────── tree[8] = [1..8] ────────────┐
            │                                           │
      ┌──── tree[4] = [1..4] ────┐              ┌──── tree[6] = [5..6] ────┐
      │                          │              │                          │
  tree[2] = [1..2]           tree[3] = [3,3]  tree[5] = [5,5]          tree[7] = [7,7]
      │
  tree[1] = [1,1]
```

**关键**：每个节点只有一个"直属父节点" $i + lowbit(i)$（单亲结构，非二叉树）。

## 更新操作：向上跳 `i += lowbit(i)`

```python
def add(self, index, delta):
    index += 1                       # 0-based → 1-based
    while index <= self.size:
        self.tree[index] += delta
        index += index & -index      # index += lowbit(index)
```

若 `a[3]` 增加 delta，需更新所有"区间覆盖位置 3"的节点：

```text
3 → 3+lowbit(3)=4 → 4+lowbit(4)=8 → 8+lowbit(8)=16(>size 停)
→ tree[3], tree[4], tree[8] 全部 +delta
```

**为什么向上加 lowbit 正好经过所有覆盖节点？** 因为 $i + lowbit(i)$ 是"下一个区间包含 i 的最小节点"，二进制上 lowbit 进位，进位前的区间被进位后的区间完整包含：

```text
i=3 = 011₂, lowbit=1 → 4 = 100₂   ([3,3] 被 [1,4] 包含)
i=4 = 100₂, lowbit=4 → 8 = 1000₂  ([1,4] 被 [1,8] 包含)
```

每次跳跃至少把最低位 1 前移一位 → 最多 $\log_2 N$ 次 → **$O(\log N)$**。

### 更新链的正确性（定理 1）

**引理 A（区间嵌套包含）**：对任意 $i$，父节点 $i + \text{lowbit}(i)$ 的管辖区间包含 $i$ 的管辖区间：

$$\big(i - \text{lowbit}(i),\ i\big] \ \subseteq\ \big(i + \text{lowbit}(i) - \text{lowbit}\big(i + \text{lowbit}(i)\big),\ i + \text{lowbit}(i)\big]$$

**证明**：设 $\text{lowbit}(i) = 2^k$ 且 $i = b \cdot 2^{k+1} + 2^k$（$b \ge 0$）。则 $i + 2^k = (b+1) \cdot 2^{k+1}$ 是 $2^{k+1}$ 的倍数，故 $L := \text{lowbit}(i + 2^k) \ge 2^{k+1} > 2^k$。父节点管辖区间左端点为 $i + 2^k - L \le i + 2^k - 2^{k+1} = i - 2^k = i - \text{lowbit}(i)$，右端点为 $i + 2^k > i$。故父节点管辖区间包含子节点管辖区间。□

**定理 1（更新链 = 所有受影响节点）**：从 $p$ 出发按 $i_{t+1} = i_t + \text{lowbit}(i_t)$ 迭代直到超过 $n$，得到的链 $\{i_0, i_1, \ldots\}$ **恰好**是所有"管辖区间包含 $p$"的节点，因此 `add(p, delta)` 必须且只需更新链上节点。

**证明概要**：

- 充分性：由引理 A 归纳，链上每个节点 $i_t$ 的管辖区间都包含首节点 $p$ 的管辖区间 $(p - \text{lowbit}(p),\ p]$，故都包含 $p$——它们必须被更新。
- 必要性：设 $u$ 管辖包含 $p$，即 $u - \text{lowbit}(u) < p \le u$。$p$ 的更新链对应其二进制**从低位到高位的进位过程**：每步把当前最低位的 1 向上进位（例如 $p = 9 = (1001)_2$ 时链为 $9 \to 10 \to 12 \to 16$，步长 $1, 2, 4, 8$ 恰为沿途各节点当前的 lowbit）。该进位过程从低到高逐位产生所有满足 $u - \text{lowbit}(u) < p \le u$ 的右端点锚点 $u$，故 $u$ 必在链上。□

## 前缀和查询：向下跳 `i -= lowbit(i)`

```python
def prefix_sum(self, i):       # 求 a[1]+...+a[i]
    result = 0
    while i > 0:
        result += self.tree[i]
        i -= i & -i            # i -= lowbit(i)
    return result
```

**原理**：任何前缀 $[1..i]$ 都能拆成 $O(\log N)$ 个 BIT 管辖区间的无重叠并集。例 `prefix_sum(7)`：

```text
i=7 = 111₂ → tree[7] = [7,7]   → i=6
i=6 = 110₂ → tree[6] = [5,6]   → i=4
i=4 = 100₂ → tree[4] = [1,4]   → i=0 停

[1..7] = [1,4] ∪ [5,6] ∪ [7,7]
```

每次减 lowbit 就是把最低位 1 清零，二进制位数有限 → **$O(\log N)$**。

`total` 即 `prefix_sum(size)`。

### 前缀分解定理（定理 2）

**定理 2**：对任意 $1 \le i \le n$，设 $i$ 的二进制展开为 $i = \sum_{j=1}^{m} 2^{k_j}$（$k_1 < k_2 < \cdots < k_m$）。则前缀区间 $[1, i]$ 可分解为 $m$ 个**互不相交**的管辖区间：

$$\big(0,\ 2^{k_1}\big],\ \big(2^{k_1},\ 2^{k_1}+2^{k_2}\big],\ \ldots,\ \Big(\textstyle\sum_{t=1}^{m-1} 2^{k_t},\ \textstyle\sum_{t=1}^{m} 2^{k_t}\Big]$$

这些区间恰好对应查询过程中依次访问的节点。

**证明**：查询自 $i_0 = i$ 开始，$i_{t+1} = i_t - \text{lowbit}(i_t)$ 等价于把 $i_t$ 二进制中**最低位的 1 清零**。设 $i$ 的二进制中从低到高的 1 依次位于 $k_1 < k_2 < \cdots < k_m$，则依次清零后得到

$$i_j = \sum_{t=j+1}^{m} 2^{k_t},\qquad j = 0, 1, \ldots, m$$

节点 $i_{j-1} = \sum_{t=j}^{m} 2^{k_t}$ 的 lowbit 恰为 $2^{k_j}$（其二进制最低位 1 在第 $k_j$ 位），故其管辖区间为

$$\big(i_{j-1} - 2^{k_j},\ i_{j-1}\big] = \Big(\textstyle\sum_{t=j+1}^{m} 2^{k_t},\ \textstyle\sum_{t=j}^{m} 2^{k_t}\Big]$$

这些区间按 $j = m, m-1, \ldots, 1$ 排列时首尾相接、互不重叠，并集恰好覆盖 $(0, i] = [1, i]$。□

## 加权采样：find_by_cumulative_weight 的二分原理

### 问题定义

前缀累积权重 $C(k) = \sum_{j=1}^{k} a_j$（权重 ≥ 0 时单调不减）。给定 `target`，求**最大的 k 使 $C(k) \le target$**，返回 0-based 下标。

朴素二分需随机访问 `C(mid)`（每次 $O(\log N)$）→ 总 $O(\log^2 N)$。

### BIT 加速：直接在树上走

```python
def find_by_cumulative_weight(self, target):
    index = 0
    bit = 1 << (self.size.bit_length() - 1)   # 最高位 2 的幂
    while bit:
        next_index = index + bit
        if next_index <= self.size and self.tree[next_index] <= target:
            target -= self.tree[next_index]   # 跨过这块区间
            index = next_index
        bit >>= 1                             # 缩小步长
    if index >= self.size:
        raise RuntimeError("Failed to find a positive-weight sequence")
    return index
```

**为什么 `tree[next_index]` 可以直接用？** 当 `index` 是 lowbit 对齐位置时，$tree[index + bit] = C(index + bit) - C(index)$，即"下一块区间的累积增量"。贪心跨过不相交区间，`index` 走过的路径累加即得 $C(index)$。

### 完整例子

`weights = [2, 1, 3]`（size=3）：

```text
tree[1] = 2   tree[2] = 3   tree[3] = 3
```

`find_by_cumulative_weight(4.5)`，bit 从 2 开始：

```text
bit=2: next=2, tree[2]=3 ≤ 4.5 → target=1.5, index=2
bit=1: next=3, tree[3]=3 > 1.5 → 不跳
→ return 2（0-based 第 3 个元素，权重 3）
```

验证累积区间 `[0,2)=2, [2,3)=1, [3,6)=3`，target=4.5 落在 `[3,6)` ✔

复杂度：bit 从最高位右移到 0，恰好 $\log_2 N$ 次 → **$O(\log N)$**，比"二分+前缀和"快一个 log。

### 正确性证明（定理 3）

**定理 3**：设权重 $w_t \ge 0$，前缀和 $P(j) = \sum_{t=1}^{j} w_t$（约定 $P(0) = 0$）。对任意 $0 \le T < P(n)$，`find_by_cumulative_weight(T)` 返回**最大的** $k \in \{0, \ldots, n-1\}$（0-based）使 $P(k) \le T$，即满足 $P(k) \le T < P(k+1)$ 的元素下标。

**对齐引理（引理 B）**：若 $index \equiv 0 \pmod{2 \cdot bit}$，则 $tree[index + bit] = P(index + bit) - P(index)$，即 `tree[next]` 恰好等于区间 $(index,\ index + bit]$ 内的总权重。

**证明**：设 $bit = 2^l$。$index \equiv 0 \pmod{2^{l+1}}$ 意味着 $index$ 的二进制第 $l$ 位为 $0$，故 $index + bit$ 的第 $l$ 位为 $1$ 而更低各位全为 $0$，即 $\text{lowbit}(index + bit) = 2^l = bit$。由 BIT 定义，$tree[index + bit]$ 管辖区间恰为 $(index,\ index + bit]$，其值即 $P(index + bit) - P(index)$。□

**主证明（循环不变式）**：记 $bit = 2^l$。证明如下不变式在每轮循环开始时成立（$l$ 从 $\lfloor \log_2 n \rfloor$ 递减到 $0$）：

> **(I1)** $index \equiv 0 \pmod{2 \cdot bit}$（对齐，保证引理 B 适用）；
> **(I2)** $0 \le T_{\text{rem}} < P\big(\min(index + 2 \cdot bit,\ n)\big) - P(index)$（剩余目标小于 $index$ 之后两倍步长窗口内的总权重）。

**初始化**：$bit = 2^{\lfloor \log_2 n \rfloor}$，则 $2 \cdot bit > n$，故 $P(\min(2 \cdot bit, n)) - P(0) = P(n) > T = T_{\text{rem}}$；且 $index = 0 \equiv 0 \pmod{2 \cdot bit}$。✓

**保持**：设进入某轮时不变式成立，令 $bit' = bit / 2$。

- **情形一**（$tree[index + bit] > T_{\text{rem}}$ 或越界）：不前进。由引理 B，$P(index + bit) - P(index) > T_{\text{rem}}$，即剩余目标小于新窗口 $(index,\ index + bit]$ 的权重，答案必在该窗口之后；(I1) 对 $bit'$ 仍成立（$index \equiv 0 \pmod{2 bit} \Rightarrow index \equiv 0 \pmod{bit}$），(I2) 由新窗口直接给出。✓
- **情形二**（$tree[index + bit] \le T_{\text{rem}}$）：令 $index' = index + bit$，$T'_{\text{rem}} = T_{\text{rem}} - tree[index + bit]$。
  - (I1)：$index \equiv 0 \pmod{2 bit}$ 意味着 $index$ 的二进制第 $l$ 位为 $0$，进位后 $index' = index + bit$ 的第 $l-1$ 位仍为 $0$，故 $index' \equiv 0 \pmod{2 \cdot bit'} = \pmod{bit}$。✓
  - (I2)：由旧不变式 $T_{\text{rem}} < P(index + 2 \cdot bit) - P(index)$，两边减去 $tree[index + bit] = P(index + bit) - P(index)$，得 $T'_{\text{rem}} < P(index + 2 \cdot bit) - P(index + bit) = P(index' + 2 \cdot bit') - P(index')$；且 $T'_{\text{rem}} \ge 0$（因 $tree[index+bit] \le T_{\text{rem}}$）。✓

**终止**：$bit$ 降至 $0$。不变式 (I2) 给出 $0 \le T_{\text{rem}} < P(\min(index+1,\ n)) - P(index)$，即 $P(index) \le T < P(index+1)$。由于 $T < P(n)$，(I2) 排除了 $index = n$，故 $index \le n-1$。返回的 0-based $index$ 恰为满足 $P(index) \le T < P(index+1)$ 的最大下标。□

### RuntimeError 的含义

所有权重为 0（或 target 越界）时，贪心跳完 `index` 顶到 size，无合法区间可落，报错提示"没有正权重的序列"。

## 数学本质

一切归结为一个恒等式：

$$C(k) = \sum_{j=1}^{k} a_j = \sum_{\text{二进制分解}} tree[\text{对应节点}]$$

- **更新**：从叶子向上，`+lowbit` 走"包含该位置的所有区间"
- **查询**：从根向下，`-lowbit` 走"拼出前缀的所有区间"
- 两者路径长度 = 二进制中 1 的个数 ≤ $\log_2 N$

BIT = **用索引二进制本身当存储结构的数据结构**，树由 `i & -i` 运算隐式生成，无需显式指针。

## 加权采样用法

```python
import random
weights = [1, 3, 2]
ft = FenwickTree(weights)
target = random.uniform(0, ft.total)   # [0, 6) 随机数
idx = ft.find_by_cumulative_weight(target)
# idx 命中概率 = weights[idx] / 6
```

改权重无需重建：`ft.add(i, delta)` 即可，保持 $O(\log N)$。

## 复杂度分析与数值精度

### 复杂度定理（定理 4）

| 操作 | 复杂度 | 依据 |
|---|---|---|
| 构建 `__init__`（逐点 add） | $O(N \log N)$ | 每个元素沿父链上浮 $\le \log_2 N$ 次 |
| 构建（线性优化） | $O(N)$ | 见下文 |
| 单点更新 `add` | $O(\log N)$ | 定理 1：更新链长度 = 进位次数 $\le \log_2 N$ |
| 前缀和 / `total` | $O(\log N)$ | 定理 2：二进制分解项数 $\le \log_2 N$ |
| 加权定位 `find_by_cumulative_weight` | $O(\log N)$ | 定理 3：bit 从 $\lfloor \log_2 n \rfloor$ 递减到 0 |
| 空间 | $O(N)$ | 单个长度 $N+1$ 的数组 |

**线性构建**：`tree[i]` 可直接按定义 $tree[i] = \sum_{j = i - \text{lowbit}(i) + 1}^{i} a_j$ 计算，或用递推：先置 $tree[i] = a_i$，再对每个 $i$ 向父节点 $i + \text{lowbit}(i)$ 累加，总复杂度 $O(N)$。

### 数值精度

加权采样使用浮点时，`find_by_cumulative_weight` 依赖前缀和 $P$ 的**单调性**；浮点累加误差可能导致 $P(k)$ 与 $P(k+1)$ 在边界处无法区分。对策：

1. **高精度累加**：用 double 或整数权重（定点化），避免 float 精度漂移
2. **量级归一**：极端量级差时先取对数（对数空间技巧）或采样前归一化
3. **误差校正**：定位后可选一步微调，确保 $P(k) \le T$ 严格成立

## 与线段树对比

| | BIT | 线段树 |
|---|---|---|
| 代码量 | 极简（几行） | 较多（建树/递归） |
| 存储 | 1 个数组 | 4N 数组或树节点 |
| 支持操作 | 前缀和、单点改、第 k 大 | 区间改、区间查、懒标记 |
| 局限性 | 只能"前缀/单点"类问题 | 通用 |

## 常见坑点

1. **下标从 1 开始**：`tree` 长度是 `size+1`，0-based 转 1-based 要 `+1`
2. **`find_by_cumulative_weight` 的 target 必须 < total**：等于 total 时可能落到 size 越界
3. **所有权重为 0 时无法采样**：会抛 RuntimeError，调用方需先检查 total
4. **浮点权重累加误差**：大量小权重累加可能精度漂移，必要时用双精度或对数空间技巧
5. **只支持"前缀/单点"问题**：需要区间修改/区间查询时换线段树

## 我的理解

Fenwick Tree 的优雅在于：它**没有任何"树"的显式形态**——不建节点、不存指针，只靠 `i & -i` 这个位运算，就让数组索引自己长出了树结构。更新与查询互为逆过程（`+lowbit` vs `-lowbit`），对称而自洽。

加权采样里最妙的是 `find_by_cumulative_weight`：它不是"二分查找前缀和数组"，而是**直接在 BIT 的二进制路径上贪心**，把"二分 $O(\log N)$ + 前缀和 $O(\log N)$"的 $O(\log^2 N)$ 压成了 $O(\log N)$。这个技巧（按 2 的幂从高位贪心）在很多"第 k 大 / 前缀定位"问题里都能复用。

## Related

- [RVV 算子开发必备基础知识](./rvv-operator-development.md) — 并行归约（vsum/vredsum）与树状归约同源，均为 $\log N$ 级树形聚合
- [AI 开源项目源码精读指南](./ai-open-source-source-reading.md) — Prioritized Replay（优先经验回放）依赖加权采样，是 RL 代码精读时的关联点

## References

- [Fenwick tree - Wikipedia](https://en.wikipedia.org/wiki/Fenwick_tree)
- [Visualgo: Fenwick Tree 交互演示](https://visualgo.net/en/fenwick)
