# Training Optimization

这一主题讨论训练时如何把损失函数的局部梯度转化为稳定、有效且可负担的参数更新：包括优化器、预条件、学习率、参数化约束、数值稳定性和训练动力学。

## Notes

### 统一框架

- [最速下降的范数对偶框架：SGD、SignSGD 与 Muon 是同一条公式](./steepest-descent-duality-map.md) — 最速下降方程 → 对偶范数 → 对偶映射；含 msign 的完整推导与"参数/梯度的类型错误"
- [谱范数与 RMS 几何：为什么 Muon 用算子范数，而 AdamW 的几何不匹配](./spectral-norm-rms-geometry.md) — RMS→RMS 算子范数推导、输出扰动上界、AdamW 的 $\ell_1\to\ell_\infty$ 与 NP-hard 论证

### 具体方法

- [Muon 优化器](./muon-optimizer.md) — 对矩阵动量更新做近似极分解/半正交化的优化器；算法总览、[Newton–Schulz 代数推导](./muon-optimizer.md)与 Moonlight 改型
- [Bi-Maxwell：Muon 的物理响应与双时间尺度动量](./bimaxwell-muon-physical-response.md) — 条件化的 Polar 动机、双时间尺度记忆核和 $124\text{M}$ 公开基准
- [平滑矩阵 Polar 谱梯度流](./smoothed-matrix-polar-spectral-gradient-flows.md) — Muon-type 连续时间模型的稳定性与局部谱方向条件

## 组织原则

- 优先区分**算法定义**、**实现近似**、**经验结果**和**机制解释**；它们的证据强度不同。
- 优化器通常只定义参数更新，不能脱离模型结构、参数分类、学习率策略、数据、batch size 和数值精度谈“更好”。
- 当一个方法只适用于一类张量时，记录其与其他参数优化器组成的混合方案，而不是把它描述为完整训练配方。

## Related

- [AI Foundations](../)
- [torch.optim — 优化算法](../../systems/pytorch/pytorch-optim.md) — PyTorch 中 SGD、Adam/AdamW 等优化器的对象模型和源码入口
