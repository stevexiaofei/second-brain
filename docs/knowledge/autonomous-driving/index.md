# 🚗 Autonomous Driving

## 笔记索引

- [端到端轨迹预测总览](./e2e-trajectory-diffusion-model.md) — 多模态注意力编码 + 锚点扩散解码的系统级总览（hub，下面 4 篇是它的原子展开）
- [自动驾驶训练数据管理最佳实践](./data-management.md) — 存储分层、数据版本化、标注质量、流水线闭环
- [BEV Data Manager：本地版本化列式存储设计解析](./bev-data-manager.md) — schema 不可变 + 原子发布 + 五道校验的 `.bevcol`/`.bevsession` 列存，特征导出落地
- [六路 token 注意力编码](./six-modality-attention-encoder.md) — 6 类异构信息拼成 377 token 走同一个注意力池：Q/K/V 开会类比、真实 tensor shape、代码映射
- [锚点扩散解码器](./anchor-diffusion-decoder.md) — 64 条 KMeans 锚点加噪后经 6 层 cross-attention 细化，NMS 出 6 个 mode；训练/推理不对称与 evolving anchors
- [轨迹预测的 Loss 与评测指标](./trajectory-loss-and-metrics.md) — GMM NLL 与 BCE 的分工、正样本怎么选、minADE/MissRate 等 oracle 口径的局限
- [端到端轨迹预测的训练闭环与数据管线](./e2e-training-pipeline.md) — 8 卡 DDP 训练循环、三阶段数据预处理（compact 列存 + rank shards）、代码地图

## Perception

- Camera
- LiDAR
- BEV
- 3D Detection
- Segmentation

## Prediction

- Trajectory Prediction
- Motion Forecasting
- Occupancy

## Planning

- Motion Planning
- Decision Making
- End-to-End Planning

## World Models

- Scene Representation
- Simulation
- Generative World Models

## Reinforcement Learning

- PPO
- GRPO
- Offline RL
- Imitation Learning
