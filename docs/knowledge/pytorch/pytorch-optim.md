---
title: torch.optim — 优化算法
type: concept
status: seed
tags: [pytorch, source-code]
created: 2026-08-11
updated: 2026-08-11
---

# torch.optim — 优化算法

## 一句话理解

> `torch.optim` 提供基于梯度的参数更新算法（SGD/Adam 家族等）与学习率调度器：优化器持有参数引用与每个参数的状态（动量、二阶矩等），在 `step()` 时读取 autograd 产出的 `.grad` 更新参数。

## 为什么重要

训练一个神经网络本质上是"前向算损失 → autograd 反向算梯度 → 优化器用梯度更新参数"的循环。autograd 只负责产出梯度，**如何用梯度更新**（学习率、动量、自适应缩放、二阶近似）决定了收敛速度、最终精度与显存占用——这正是 `torch.optim` 的职责。不同的优化器在大规模训练（LLM、扩散模型）中的选择直接影响成本与稳定性，理解其源码实现有助于调试训练异常、做 fused 优化器、集成 ZeRO/FSDP 分片状态。

## 核心概念

### Optimizer 基类

| 概念 | 说明 |
| ---- | ---- |
| 参数组（param_groups） | 优化器持有 `List[Dict]`，每个 dict 含一组 `params` 与该组超参（`lr`、`weight_decay`、`momentum` 等）。可对不同层用不同学习率 |
| `state` | `Dict[Parameter, Dict]`，按参数存优化器状态（动量、二阶矩、步数计数等）。FSDP/ZeRO 会把这部分分片 |
| `step(closure=None)` | 执行一次参数更新。部分算法（如 LBFGS）需 closure 重算损失 |
| `zero_grad(set_to_none=True)` | 清空参数 `.grad`。默认置 `None`（比填零省内存、更快） |
| `state_dict()` / `load_state_dict()` | 序列化优化器状态，用于检查点恢复 |
| 钩子 | `register_step_post_hook` 等允许在更新前后插入逻辑（如梯度裁剪、自定义正则） |

### 优化算法族

| 算法 | 关键思想 | 位置 |
| ---- | ---- | ---- |
| `SGD` | 带动量/Nesterov 的随机梯度下降，最基础 | `torch/optim/sgd.py` |
| `Adam` | 一阶+二阶矩自适应估计 + 偏差校正 | `torch/optim/adam.py` |
| `AdamW` | Adam + 解耦权重衰减（L2 正则不再扭曲动量） | `torch/optim/adamw.py` |
| `Adagrad` | 按参数累积梯度平方和，逐参数自适应学习率 | `torch/optim/adagrad.py` |
| `Adadelta` | Adagrad 的指数衰减版，无需全局学习率 | `torch/optim/adadelta.py` |
| `RMSprop` | 梯度平方指数移动平均做分母 | `torch/optim/rmsprop.py` |
| `Adamax` | Adam 的 $\infty$-范数变体 | `torch/optim/adamax.py` |
| `RAdam`/`NAdam` | Adam + Nesterov 或预热修正 | `torch/optim/radam.py`、`nadam.py` |
| `ASGD` | 平均随机梯度下降 | `torch/optim/asgd.py` |
| `Rprop` | 仅用梯度符号的弹性反向传播 | `torch/optim/rprop.py` |
| `LBFGS` | 拟牛顿法（线搜索 + 海森近似），小批量/全批 | `torch/optim/lbfgs.py` |
| `SparseAdam` | 面向稀疏梯度（Embedding）的 Adam 变体 | `torch/optim/sparse_adam.py` |
| `Adafactor` | 因式分解二阶矩省显存，常用于大 Transformer | `torch/optim/_adafactor.py` |

### 学习率调度器

`torch/optim/lr_scheduler.py` 定义 `LRScheduler` 基类与具体调度器：`StepLR`、`MultiStepLR`、`ExponentialLR`、`CosineAnnealingLR`、`ReduceLROnPlateau`（按指标触发）、`OneCycleLR`、`CyclicLR`、`SequentialLR`、`LambdaLR` 等。调度器在 `optimizer.step()` 之后调用 `scheduler.step()`，按规则改写每个 param_group 的 `lr`。

### 随机权重平均（SWA）

`torch/optim/swa_utils.py` 提供 `AveragedModel`（权重平均包装器）与 `SWALR`（SWA 专用学习率），在训练末期对多轮权重做平均，常能提升泛化。

### 函数式实现

`torch/optim/_functional.py` 提供"仅张量"的优化步骤实现（如 `_functional_adam`），把"参数张量 + 梯度 + 状态"作为入参，不依赖 `Optimizer` 对象——供 fused 内核、`torch.compile` 与分布式优化器复用同一份算法逻辑。

## 工作原理

### 训练循环中的位置

<div class="diagram">
  <div class="h-flow" style="flex-wrap:wrap; justify-content:center; align-items:stretch;">
    <span class="d-node d-node-start">① forward<br/><small style="opacity:0.75; font-weight:400;">算 <code>y = model(x)</code></small></span>
    <span class="d-arrow"></span>
    <span class="d-node">② 计算 loss</span>
    <span class="d-arrow"></span>
    <span class="d-node d-node-active">③ <code>loss.backward()</code><br/><small style="opacity:0.75; font-weight:400;">autograd 引擎把梯度写进 <code>.grad</code></small></span>
    <span class="d-arrow"></span>
    <span class="d-node">④ <code>zero_grad()</code><br/><small style="opacity:0.75; font-weight:400;">默认 <code>set_to_none=True</code></small></span>
    <span class="d-arrow"></span>
    <span class="d-node d-node-active">⑤ <code>optimizer.step()</code><br/><small style="opacity:0.75; font-weight:400;">读 <code>.grad</code> 更新参数 + 维护 state</small></span>
    <span class="d-arrow"></span>
    <span class="d-node">⑥ <code>scheduler.step()</code><br/><small style="opacity:0.75; font-weight:400;">调整每 <code>param_group</code> 的 <code>lr</code></small></span>
  </div>
  <div class="h-flow" style="justify-content:center; margin-top:14px; flex-wrap:wrap;">
    <span class="d-label" style="border-color:#c7d2fe; background:#eef2ff; color:#3730a3;">⑥ → 下一迭代回到 ①（按 step 或 epoch 调调度器）</span>
  </div>
  <div class="d-note">
    <b>常见坑：</b><code>zero_grad()</code> 在 <code>backward()</code> 前调（PyTorch 习惯）还是 <code>step()</code> 后调（TF 习惯）等价；<code>ReduceLROnPlateau</code> 的 <code>step(metric)</code> 与其他基于 epoch/step 的调度器语义不同，不可混用。
  </div>
</div>

### 关键算法的数学形式

**SGD with momentum**（`torch/optim/sgd.py`）：

$$v_t = \mu \, v_{t-1} + g_t$$

$$\theta_t = \theta_{t-1} - \eta \, v_t$$

Nesterov 变体先用动量前瞻一步再算梯度。

**Adam**（`torch/optim/adam.py`），其中 $g_t$ 为梯度、$m_t$ 一阶矩、$v_t$ 二阶矩、$t$ 步数：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

$$\theta_t = \theta_{t-1} - \eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

偏差校正 $\hat{m}_t, \hat{v}_t$ 解决初始阶段矩估计偏向 0 的问题。权重衰减在原 Adam 中以 $g_t \leftarrow g_t + \lambda \theta_{t-1}$ 实现（与动量耦合，会扭曲更新方向）；`AdamW` 改为解耦：直接对参数衰减 $\theta_t \leftarrow (1-\eta\lambda)\theta_t$，这是大模型训练的标准选择。

### step() 的执行流程

1. 遍历 `param_groups`，对每个参数：
2. 跳过 `grad is None` 的参数（避免未参与前向的参数被误更新）；
3. 取出/初始化该参数的 `state`（如 Adam 的 `exp_avg`、`exp_avg_sq`、`step`）；
4. 按算法更新状态与参数（通常调用 `_functional_*` 或 `torch._foreach_*` 批量算子以减少 kernel launch 开销）；
5. 触发 `step_post_hook`。

### state_dict 与检查点

`state` 以 `Parameter` 对象为键，序列化时转为参数索引。恢复时按 `load_state_dict` 重建映射。FSDP/ZeRO 会把 `state` 按分片重组，使得每个 rank 只持有自己那份参数的优化器状态——这是大模型省显存的关键。

## 我的理解

- 优化器与 autograd 的接口极简：**`.grad` 字段**。autograd 不关心谁消费梯度，优化器不关心梯度怎么来的。这种松耦合让 fused/compiled 优化器、分布式优化器都能透明接入。
- `foreach` API（`torch._foreach_add_` 等张量列表算子）是近年来 optim 性能优化的重点：一次 kernel 处理整个参数组，显著降低 launch 开销，大模型训练里这是实打实的吞吐收益。
- Adam 系列"自适应"的本质是把每个参数的学习率除以自身梯度的 RMS，使不同量级的参数获得相近的有效步长——这对 Embedding（稀疏大梯度）与卷积权重（密集小梯度）共存的网络尤其重要。`SparseAdam` 进一步只更新出现过的行。
- `zero_grad(set_to_none=True)` 默认置 `None` 而非填零，既省内存又让 `step()` 能用 `grad is None` 跳过未参与计算的参数——这是细节但有意义。
- `ReduceLROnPlateau` 是少数不基于 epoch 而基于指标的平台期检测调度器，内部维护 `best` 与 `num_bad_epochs`；它与基于 step 的调度器不可混用 `scheduler.step()` 语义（一个吃 `metrics`、一个不吃），是常见误用点。
- 大模型训练里优化器状态才是显存大头：Adam 每个参数要存 `exp_avg` + `exp_avg_sq`（FP32 下 8 字节/参数），比参数本身（BF16 2 字节）和梯度还大——这正是 ZeRO/FSDP 分片优化器状态的动机。

## Related

- [torch.autograd](./pytorch-autograd/) — 产出 `.grad` 供优化器消费
- [torch.nn](./pytorch-nn/) — `Module.parameters()` 是优化器的输入
- [分布式训练](./pytorch-distributed/) — ZeRO/FSDP 分片优化器状态、`zero_redundancy_optimizer`
- [torch.compile 编译栈](./pytorch-compile/) — fused/compiled 优化器、`apply_optimizer_in_backward`

## References

- `torch/optim/` 整个目录
- `torch/optim/optimizer.py` — `Optimizer` 基类
- `torch/optim/adam.py`、`torch/optim/adamw.py`、`torch/optim/sgd.py`
- `torch/optim/lr_scheduler.py` — 学习率调度
- `torch/optim/_functional.py` — 函数式实现
- `torch/optim/swa_utils.py` — 随机权重平均
