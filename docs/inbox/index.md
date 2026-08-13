# 📥 Inbox

这里是第二大脑的入口。任何突然想到的东西、看到的文章、论文、代码经验、问题、想法，都可以先放这里。

不要一开始就纠结分类。

## Rules

1. 先记录
2. 后整理
3. 不要因为分类而停止记录
4. 定期让 TRAE 帮助整理

## Notes

- [Second Brain 使用工作流](./second-brain-workflow.md) — 从 capture 到 knowledge 的完整流程、目录用途、与 TRAE 协作的常用指令
- [ER 图（实体-关系图）](./er-diagram.md) — 数据库概念设计的图形化工具，描述实体、属性、关系和基数
- [UML 类图（Class Diagram）](./uml-class-diagram.md) — 面向对象设计的静态结构图，含六种类关系（依赖/关联/聚合/组合/实现/继承）
- [DFD（数据流图）](./dfd.md) — 描述数据流动与加工的结构化分析工具，含分层展开和平衡原则
- [DDIM (Denoising Diffusion Implicit Models)](./ddim-paper.md) — 扩散模型加速采样论文笔记，含公式 (11) 解释、KL 散度在变分推断中的角色、定理 1 完整证明
- [uv — 极速 Python 包管理工具](./uv-python-package-manager.md) — Astral 出品的 Rust 实现一体化 Python 工具链，替代 pip/venv/poetry/pyenv/pipx/twine，含常用命令、对比表、典型工作流
- [AI 开源项目源码精读指南](./ai-open-source-source-reading.md) — 12 个值得精读的 AI 开源项目、Top 5 学习路线、Design Pattern 对照表、源码精读方法
- [RVV 算子开发必备基础知识](./rvv-operator-development.md) — RISC-V Vector Extension 的核心概念、编程模型、intrinsics 命名和算子开发流程
- [Second Brain 迭代路线图](./second-brain-iteration-roadmap.md) — 现状盘点、四阶段迭代计划（内容闭环/外部打通/AI 深化/发布治理）与优先级建议
- [分布式存储系统知识地图](./distributed-storage-knowledge-map.md) — 面向 AI 训练与多云场景的存储知识体系：三大类型、一致性、对象存储、并行文件系统、多级缓存、存储网络协议、学习路径
- [RVV 算子设计大赛备考指南](./rvv-operator-challenge.md) — 基于大赛三张设计图（Local Buffer/Mode/Intrinsics/DAG）的 Softmax 实战，含标准 RVV 与 FP16 3D 版本实现

## Example

```text
今天发现 Docker 在 NFS 上执行 git status 特别慢。
可能和 inode / metadata / network filesystem 有关。
```
