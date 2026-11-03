---
title: 轨迹预测的 Loss 与评测指标
type: concept
status: growing
tags: [Autonomous Driving, Trajectory Prediction, GMM NLL, BCE, minADE, MissRate, Loss Design]
created: 2026-11-02
updated: 2026-11-02
source: attention_interaction.html 动画讲解页（对齐 v2_5_12_f2_13 配置的真实代码库）
---

# 轨迹预测的 Loss 与评测指标

## 一句话理解

模型同时学两件事：**回归**报"圈子"（GMM NLL：位置 + 不确定度），**分类**报"哪条候选靠谱"（64 个独立 BCE，不是 softmax 多分类）；评测则是多峰友好的 min 口径（6 条里取最优）与工程约束指标（舒适、碰撞、沿导航进度）的组合。

## Loss：四项组合

每一层 Decoder 都输出 64 条轨迹，所以 **6 层各被监督一次**，最后把 6 层 loss 求平均。总损失：

$$
\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{reg} + 0.5 \cdot \mathcal{L}_{vel} + 0.5 \cdot \mathcal{L}_{head}
$$

| Loss | 权重 | 学什么 |
|---|---|---|
| 回归 · GMM NLL | 1.0 | 对 9 维输出里的 $x, y, \sigma_x, \sigma_y, \rho$ 做高斯混合负对数似然：既学位置，也学"我有多不确定" |
| 分类 · BCE | 1.0 | 64 个查询各自和 GT 算距离，最近的标正样本；学"这条候选靠不靠谱" |
| 速度 · L1 | 0.5 | 直接约束 $v_x, v_y$，避免"位置对但速度乱跳"，对 jerk 类指标有直接帮助 |
| 朝向 · cosine | 0.5 | 对 $\sin/\cos(\theta)$ 做余弦约束；用 sin/cos 表示避开 $\pm\pi$ 角度跳变 |

### 为什么是 GMM NLL：打靶类比

你说"会落在 10.0 米处，误差 ±0.3 米"，真实是 11.5 米——按你自己的说法差了 5 个 $\sigma$，罚得极重；若一开始就说 ±3 米，同样偏 1.5 米只是 0.5$\sigma$，罚得轻。**NLL 逼模型如实报告不确定度**：有把握报小，没把握报大。L1/L2 只看误差大小、不管自信程度。

代码里（`modules/utils/loss_utils.py:200-203`）在算什么：

- 位置误差除以自称的 $\sigma$——越不准又越自信罚越重
- $\log \sigma_x + \log \sigma_y$——专门惩罚"把 σ 吹大作弊"
- $\log(1-\rho^2)$——防止相关性过于自信
- $-2\rho \, dx \, dy / (\sigma_x \sigma_y)$——允许 x、y 误差相关（转弯时横纵向误差往往同时出现）
- σ 裁在 0.2~148 m、ρ 裁在 ±0.5，避免极端值炸掉 loss

GMM 的"混合"在哪？**在分类分支**——64 个 BCE 分数过 sigmoid 就是 64 个分量的权重 $\pi_k$，回归只对选中的那条算高斯 NLL。两者合起来才是完整的 GMM 训练。

### 为什么是 BCE 不是 softmax 多分类

先确认：是 **64 个独立的二分类**，不是一个 64 分类（`predictors/diffusion_bst.py:386-389`）。三个理由：

1. **候选不互斥**。KMeans 锚点里常有"偏左 0.3 m"和"偏左 0.5 m"这种近似重复，softmax 会把概率总和硬分成 1，逼模型把其中一条压到接近 0；BCE 允许它们同时拿高分。
2. **要和推理一致**。推理按绝对分数排序再 NMS 删重复。BCE 的 0.91 表示"这一条单独看有 91% 像"，删掉邻居后含义不变；softmax 的 0.55 表示"占全体的 55%"，删掉 20 条后这个数就不再成立。
3. **能跳过重复项**。BCE 逐项乘 mask 即可；softmax 是整体归一化，想忽略某些项得先填 $-\infty$，而分母会随场景里重复项的多少漂移。

什么时候才该用 softmax：候选真正互斥、必须且只能选一个时（例如"8 个路口出口走哪个"）。**softmax 回答"哪一条最像"，只能有一个答案；BCE 回答"每一条各自有多像"，可以好几个，也可以一个都没有。**

### 正样本怎么选（容易看漏的三步）

1. **确定 GT 终点**：取最后一个有效帧（`gt_final_valid_idx`），不是固定第 50 帧——轨迹被截断的场景会提前取值，避免拿填 0 的假终点去比。
2. **算 64 条候选各差多远**：第 1~2 层拿 KMeans 锚点比**终点**；第 3~4 层拿第 2 层预测、第 5~6 层拿第 4 层预测，比**整条轨迹**（逐帧距离累加 ≈ ADE）。
3. **先去重，再取最近**（`modules/utils/motion_utils.py:141-142`）：

```python
select_mask = batch_nms(...)            # 算出谁被淘汰
dist.masked_fill(~select_mask, 1e10)    # 被淘汰的距离设成无穷大
dist.argmin()                           # "最近"一定落在存活者里
```

为什么非要去重：假设 GT 终点附近有 A（0.5 分，离 GT 0.2 m）和 B（0.9 分，离 GT 0.3 m），两者只差 0.3 m。不去重时 argmin 会选中 A，等于告诉模型"你更看好的 B 是错的"；去重后 A 被 B 抑制，正样本变成 B——**那个簇里分数最高的代表**。

选出来之后用于三件事：① 分类 BCE 置 1（被淘汰的候选连当负样本的资格都没有：`loss_cls * select_mask`）；② 回归/速度/朝向只对这条算 loss（"正模式"的由来）；③ 作为 `mode_idx` 传给碰撞、停车等辅助损失。

### 当前未启用的项

- `aux_losses`（TTC / 碰撞 / 停车）权重全为 0
- `predict_control: false`
- 代码里另有一版 `compute_loss` 会在 6 层后用原始 GT 再监督一次（"fake layer 7"），但当前训练/验证走的都是 `get_decoder_loss`（6 层平均），那条分支没有调用点

## 评测指标：好与坏怎么量化

6 条候选里只要有一条接近 GT 就算"预测到了"，所以误差类指标都带 `min`（oracle 口径）。

| 指标 | 单位 | 算法 | 回答的问题 | 局限 |
|---|---|---|---|---|
| minADE | 米 | 有效帧逐点欧氏距离，6 条取最小再平均 | 平均偏离多少 | 单条轨迹内部抖动会被平均掉 |
| minFDE | 米 | 只看终点帧，6 条取最小（`gt_final_valid_idx`） | 最后停在哪 | 不看中间过程 |
| MissRate | % | 终点误差超阈值算 miss；阈值横向 1.8 m / 纵向 3.6 m，按初始车速 1.4~11 m/s 线性放大 | 开得越快允许误差越大 | 低速场景会很严格 |
| Comfort | jerk | 有限差分（dt=0.1s）求速度→加速度→jerk，统计横/纵向分量 | 坐着舒不舒服 | 只看运动学，不看是否撞上 |
| TrajectoryProgress | % | 预测轨迹投影到导航路线，看前进了多少、有没有开反 | 是否沿导航走 | 依赖参考线质量，线错指标也错 |
| CollisionRate | % | 自车与 agent 都按 OBB 处理，分离轴定理逐帧判重叠 | 几何上是否重叠 | 不含责任判定；当前 `skip_prediction: true` 下实际不计算 |

注意口径差异：**舒适性只取得分最高的那条轨迹**算，minADE/minFDE 是 6 条取最优（oracle），两者要一起看。另有 minAHE/minFHE、1s/3s/5s 分时段 minADE。

MissRate 阈值缩放代码：`metrics/miss_rate.py:22` — `base_lat=1.8, base_lon=3.6, alpha=(v-1.4)/(11-1.4)`。

## 我的理解

- **GMM NLL 的本质是"对自己诚实"的约束**：模型可以把 σ 报大来逃避位置惩罚，但 log σ 项罚这种作弊，最终收敛到"该自信时自信、该含糊时含糊"。
- **BCE vs softmax 的选择标准值得推广**：只要"候选近似重复且推理时按绝对分数+后处理删重"，就用独立二分类。这个模式在 detection 的 focal loss、多标签分类里是同一个道理。
- **正样本逐层演进（evolving anchors）把"assign 标签"变成了动态过程**，和 DETR 的匈牙利匹配动机相似但实现更简单：不是全局最优匹配，而是"先去重再取最近"的贪心。
- **评测的 oracle 口径（min）是行业惯例也是陷阱**：minADE 好看不代表 top-1 轨迹可用，工程上真正喂给下游的是 top-1，所以要同时盯 Confidence-ranked 指标。

## Limitations

- 数值与权重对齐 `v2_5_12_f2_13` 配置，随实验变化
- MissRate 的速度自适应阈值使不同车速段的 miss 不可直接横比
- CollisionRate 在 skip_prediction 下为 None，评测表里出现空值是预期行为

## Related Knowledge

- [六路 token 注意力编码](./six-modality-attention-encoder.md) — 被监督的模型本体
- [锚点扩散解码器](./anchor-diffusion-decoder.md) — 正样本逐层换的机制背景（evolving anchors）
- [自动驾驶训练数据管理](./data-management.md) — 指标驱动的 badcase 回流上游
