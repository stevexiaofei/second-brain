# Training Optimization

这一主题讨论训练时如何把损失函数的局部梯度转化为稳定、有效且可负担的参数更新：包括优化器、预条件、学习率、参数化约束、数值稳定性和训练动力学。

## Notes

- [Muon 优化器](./muon-optimizer.md) — 对矩阵动量更新做近似极分解/半正交化的优化器；作为算法总览与实验入口
- [Bi-Maxwell：Muon 的物理响应与双时间尺度动量](./bimaxwell-muon-physical-response.md) — 条件化的 Polar 动机、双时间尺度记忆核和 $124\text{M}$ 公开基准
- [平滑矩阵 Polar 谱梯度流](./smoothed-matrix-polar-spectral-gradient-flows.md) — Muon-type 连续时间模型的稳定性与局部谱方向条件

## 组织原则

- 优先区分**算法定义**、**实现近似**、**经验结果**和**机制解释**；它们的证据强度不同。
- 优化器通常只定义参数更新，不能脱离模型结构、参数分类、学习率策略、数据、batch size 和数值精度谈“更好”。
- 当一个方法只适用于一类张量时，记录其与其他参数优化器组成的混合方案，而不是把它描述为完整训练配方。

## Related

- [AI Foundations](../)
- [torch.optim — 优化算法](../../systems/pytorch/pytorch-optim.md) — PyTorch 中 SGD、Adam/AdamW 等优化器的对象模型和源码入口
