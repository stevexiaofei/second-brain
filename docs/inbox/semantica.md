---
title: Semantica — AI Agent 的开源知识图谱基础设施
type: concept
status: seed
tags: [Knowledge Graph, AI Governance, Provenance, Decision Intelligence, Reasoning, RAG, MCP]
created: 2026-08-12
updated: 2026-08-12
source: "https://github.com/semantica-agi/semantica"
---

# Semantica — AI Agent 的开源知识图谱基础设施

## 一句话理解

Semantica 是开源的 **Graph-Native 基础设施**，定位是"AI Agent 的 Palantir"：把企业数据抽取成**知识图谱（Knowledge Graph）+ 上下文图（Context Graph）**，在 LLM/向量库之下提供**确定性的推理、溯源（provenance）与决策审计**层——回答的不是"AI 说了什么"，而是"**AI 为什么这么说**"。

## 为什么重要

- 大多数 AI Agent 只存 embedding，不存语义：上下文无法解释、决策无法审计
- 在信贷、医疗、法律、政府等强监管领域，Agent 的决策必须经得起"为什么"的追问
- Semantica 是**确定性基础设施**：建图、推理、溯源都不需要 LLM，可作为 LLM 之上的可信层
- 与纯 RAG（向量相似度）相比，它是**图遍历 + 语义搜索**，支持因果关系与时间回溯

## 对比：Semantica vs 传统方案

| 维度 | Vector DB + RAG | Plain LLM Memory | **Semantica** |
|---|---|---|---|
| 召回方式 | Embedding 相似度 | Token 窗口 | **图遍历 + 语义搜索** |
| 决策历史 | 不存储 | 不存储 | **一等公民，可查询对象** |
| 溯源（Provenance） | 无 | 无 | **W3C PROV-O，带源链接** |
| 推理 | 无 | 黑盒 | **前向链、Rete、Datalog、SPARQL** |
| 冲突检测 | 静默覆盖 | 静默覆盖 | **检测、标记、解决** |
| 时间旅行 | 无 | 无 | **图快照（bi-temporal）** |
| 合规导出 | 无 | 无 | **PROV-O、SHACL、OWL、RDF** |
| 策略执行 | 无 | 无 | **内置规则引擎 + SHACL** |
| 实体解析 | 无 | 无 | **分块 + 语义去重** |
| 多 Agent 上下文 | 各自独立 | 各自独立 | **单一共享智能层** |

## 核心能力

### 1. Context Graphs（上下文图）
结构化、可查询的图，覆盖 Agent 知道、决定和推理的一切。

### 2. Decision Intelligence（决策智能）
每个决策是**一等公民对象**：可追溯、可按先例搜索、可因果关联。

```python
from semantica.context import ContextGraph
graph = ContextGraph(advanced_analytics=True)

# 记录决策（带完整结构化上下文）
decision_id = graph.record_decision(
    category="vendor_selection",
    scenario="Choose cloud provider for HIPAA workload",
    reasoning="AWS offers BAA, mature HIPAA tooling, and existing team expertise",
    outcome="selected_aws",
    confidence=0.93,
)

# 查询"为什么会这样"
chain = graph.trace_decision_chain(decision_id)   # 完整因果祖先
similar = graph.find_similar_decisions("cloud vendor", max_results=5)  # 先例
impact = graph.analyze_decision_impact(decision_id)  # 下游影响图
ok = graph.check_decision_rules({"category": "vendor_selection"})  # 策略门
```

### 3. AI Governance & Ontology（治理与本体）
- SHACL 约束、冲突检测、合规规则、OWL 生成、SKOS 词汇管理（含可视化编辑器）

### 4. 确定性推理
前向链（forward chaining）、Rete 网络、Datalog、SPARQL——全可解释路径，非黑盒。

### 5. 完整审计
每个事实带 W3C PROV-O 溯源，审计轨迹可导出 JSON / CSV / RDF。

### 6. 知识流水线
多源摄取 → 实体感知分块 → NER/关系/事件抽取 → 图构建 → 语义去重 → 保留溯源的合并。

## 架构

```text
Sources → Ingest → Parse → Normalize → Split → Extract → Conflict Detection → Deduplication
 → Knowledge Graph → [ Ontology · Reasoning · Provenance · Decisions ] → Enriched KG
 → Vector Store + Polyglot Graph Store (RDF & LPG) → Export / Visualize / REST · MCP · CLI
```

- **Ingest**：文件、Web、数据库、Databricks、Snowflake、Google Drive、Elasticsearch、Kafka、Kinesis、Git、邮件、MCP
- **Extract**：NER、关系、事件、三元组；冲突事实先标记再合并
- **KG**：`GraphBuilder` 建图，bi-temporal 事实 + 图分析（中心性、社区发现、链接预测）
- **Intelligence**：SHACL/OWL 治理、Rete/Datalog/SPARQL 推理、PROV-O 溯源、决策记录
- **Storage**：多语言（polyglot）——RDF 三范式存储（Oxigraph/Blazegraph/Jena/RDF4J）+ 属性图（Neo4j/FalkorDB/AGE/Neptune）+ 向量存储，可互换
- **Outputs**：RDF/OWL/Parquet/Cypher/JSON-LD 导出、交互可视化、REST API、MCP server、CLI

## 集成

- 原生 **Agno** 支持
- 完整功能的 **MCP server**
- 综合 **CLI** 与 **REST API**
- 主流编辑器的插件

## 快速开始

```bash
pip install semantica
semantica doctor   # 环境自检
```

```python
from semantica.context import ContextGraph
graph = ContextGraph(advanced_analytics=True)
```

## 我的理解

Semantica 踩中的是当前 AI Agent 最被忽视的痛点：**可信度与可审计性**。RAG 解决"召回"，向量库解决"相似"，但都回答不了"为什么"。Semantica 用成熟的图数据库 + 确定性推理 + PROV-O 溯源，把 LLM 从"唯一真相源"降级为"语义层之一"——这正是它在强监管行业能立足的原因。

对 second brain 的启发：它把知识**从文档组织升级为图组织**（实体/关系/决策因果），和我迭代路线图里的"知识图谱页 + 回链自动化"是同一条路径的工业级版本。可以作为`AI 开源项目源码精读指南`清单里的一个候选研究对象（设计模式：Graph Builder、Pipeline、Plugin、多后端抽象）。

## Related

- [AI 开源项目源码精读指南](./ai-open-source-source-reading.md) — 12 个精读项目的扩展候选（Graph 构建/推理引擎）
- [Second Brain 迭代路线图](./second-brain-iteration-roadmap.md) — 知识图谱与回链自动化的路线参考
- [分布式存储系统知识地图](./distributed-storage-knowledge-map.md) — 图数据库多后端（Neo4j/FalkorDB/Neptune）属于存储选型范畴

## References

- [Semantica GitHub 仓库](https://github.com/semantica-agi/semantica)
- [ARCHITECTURE.md（Mermaid 图）](https://github.com/semantica-agi/semantica/blob/main/ARCHITECTURE.md)
- [W3C PROV-O](https://www.w3.org/TR/prov-o/)
