---
title: nanobot Providers Registry 源码精读
type: concept
status: seed
tags: [AI, Providers, Registry, Model Routing, OpenAI-Compatible, Nanobot, Source Code]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\nanobot\nanobot\providers\registry.py
  - d:\project\nanobot\nanobot\providers\base.py
---

# nanobot Providers Registry 源码精读

> 本文基于本地源码 `d:\project\nanobot\nanobot\providers\registry.py`（784 行）拆解。目标：看懂 nanobot 怎么用**纯数据表**支撑几十个 LLM 后端的识别、路由与参数适配。

## 一句话理解

`PROVIDERS` 是一个**元数据表**（`tuple[ProviderSpec, ...]`），每行描述一个 LLM 后端：名字、匹配关键词、API key 环境变量、默认 base URL、backend 实现类型、gateway/local/oauth 分类、以及各种推理参数适配。**顺序即优先级**，gateway 排最前。

## 1. 两个核心数据类

### `ProviderSpec`（[registry.py#L31-L136](file:///d:/project/nanobot/nanobot/providers/registry.py#L31-L136)）

字段按功能分五组：

| 分组 | 字段 | 作用 |
|---|---|---|
| 身份 | `name` / `keywords` / `env_key` / `display_name` | config 字段名、模型名匹配关键词、环境变量名 |
| 实现 | `backend`（"openai_compat"/"anthropic"/"azure_openai"/"openai_codex"/"xai_grok"/"github_copilot"/"bedrock"） | 用哪个 provider 实现类 |
| 识别 | `is_gateway` / `is_local` / `is_direct` / `detect_by_key_prefix` / `detect_by_base_keyword` | 按 key 前缀 / base URL 关键词自动识别 |
| 参数适配 | `thinking_style` / `reasoning_effort_remap` / `model_overrides` / `responses_models` / `implicit_reasoning_models` / `extract_thinking_blocks` | 各家推理参数方言 |
| 能力 | `supports_prompt_caching` / `supports_max_completion_tokens` / `is_transcription_only` | 能力标记 |

**关键点**：`env_extras` 值支持占位符 `{api_key}` 和 `{api_base}`（[registry.py#L34-L38](file:///d:/project/nanobot/nanobot/providers/registry.py#L34-L38)），如 Skywork 需要把 key 同时放进 `APIFREE_API_KEY`。

### `ProviderModelSpec`（[registry.py#L21-L28](file:///d:/project/nanobot/nanobot/providers/registry.py#L21-L28)）

给没有 model-list endpoint 的 provider（OpenAI Codex、xAI Grok）用的**手工维护模型清单**：

```python
@dataclass(frozen=True)
class ProviderModelSpec:
    id: str
    label: str = ""
    description: str = ""
    context_window: int | None = None
```

## 2. `PROVIDERS` 表的组织逻辑（顺序 = 优先级）

[registry.py#L143-L749](file:///d:/project/nanobot/nanobot/providers/registry.py#L143-L749)，从上到下依次是：

```text
1. Custom / Azure / Bedrock        ← is_direct，用户自供全部信息
2. 网关 Gateways                   ← is_gateway=True，按 key 前缀/api_base 识别（OpenRouter/OrcaRouter/
                                        EdenAI/OpenCode/HF/Skywork/AiHubMix/SiliconFlow/Novita/VolcEngine/BytePlus）
3. 标准 Provider                   ← 按模型名关键词匹配（anthropic/openai/deepseek/gemini/zhipu/
                                        dashscope/moonshot/minimax/mistral/stepfun/xiaomi_mimo...）
4. 本地部署                        ← is_local=True（vLLM/Ollama/LM Studio/Atomic Chat）
5. 辅助                            ← 语音转写等（Groq/AssemblyAI）
```

**为什么网关在最前**：网关能路由任何模型，所以匹配时它优先赢（`# Gateways can route any model, so they win in fallback.`，[registry.py#L187](file:///d:/project/nanobot/nanobot/providers/registry.py#L187)）。

## 3. 三种识别策略（重要）

| 策略 | 字段 | 例子 |
|---|---|---|
| 模型名关键词 | `keywords` | `("anthropic", "claude")`、`("deepseek",)` |
| API key 前缀 | `detect_by_key_prefix` | OpenRouter: `"sk-or-"`，HF: `"hf_"`，NVIDIA: `"nvapi-"` |
| base URL 关键词 | `detect_by_base_keyword` | `"openrouter"`、`"volces"`、`"11434"`(Ollama) |

一个 provider 可以组合多种：如 Ollama `keywords=("ollama",)` + `detect_by_base_keyword="11434"`。

## 4. 参数适配方言（这块最见功力）

不同家的 reasoning/thinking 控制方式完全不同，`ProviderSpec` 用一组字段抹平：

| 方言 | 注入方式 | 使用方 |
|---|---|---|
| `thinking_style="thinking_type"` | `{"thinking": {"type": "enabled"/"disabled"}}` | DeepSeek、VolcEngine、Xiaomi |
| `thinking_style="enable_thinking"` | `{"enable_thinking": true/false}` | DashScope、ModelScope |
| `thinking_style="reasoning_split"` | `{"reasoning_split": true/false}` | MiniMax |
| `gateway_reasoning_style="reasoning_effort"` | `{"reasoning": {"effort": ...}}` | OpenRouter |
| `reasoning_effort_remap` | 把 OpenAI 的 low/medium/high 映射成该家接受的词 | Mistral（只收 high/none） |
| `implicit_reasoning_models` | 该模型隐含推理，**不能传** reasoning_effort kwarg | Magistral |
| `extract_thinking_blocks` | 从 content block 列表提取 thinking → reasoning_content | Mistral |
| `strip_history_reasoning_content` | 回放历史时剥掉 reasoning_content | Mistral（严格 schema 校验） |
| `reasoning_as_content` | reasoning 字段才是真答案 | StepFun |
| `model_overrides` | 按模型覆盖参数（如 Kimi K2.7 必须 temperature=1.0） | Moonshot |

## 5. 查找帮助函数

[registry.py#L757-L784](file:///d:/project/nanobot/nanobot/providers/registry.py#L757-L784)：

```python
def find_by_name(name: str) -> ProviderSpec | None:
    """按 config 字段名查，如 "dashscope"。用 to_snake 归一化处理连字符。"""
    normalized = to_snake(name.replace("-", "_"))
    for spec in PROVIDERS:
        if spec.name == normalized:
            return spec

def create_dynamic_spec(name, *, display_name="", thinking_style=""):
    """为自定义用户 provider 动态生成 spec（is_direct=True + strip_model_prefixes）。"""
```

## 6. Provider 实现层（base.py，配合理解）

`LLMProvider`（[base.py#L302](file:///d:/project/nanobot/nanobot/providers/base.py#L302)）是抽象基类，核心接口：

```python
@abstractmethod
async def chat(self, messages, tools=None, **kwargs) -> LLMResponse   # 单次
async def chat_stream(...)                                            # 流式
async def chat_stream_with_retry(...)   # 带重试的流式（runner 用）
async def chat_with_retry(...)          # 带重试的普通（runner 用）
```

`LLMResponse`（[base.py#L255](file:///d:/project/nanobot/nanobot/providers/base.py#L255)）关键字段：`content` / `reasoning_content` / `thinking_blocks` / `tool_calls` / `finish_reason`（`stop|length|tool_calls|error|refusal|content_filter`）/ `usage` / `provider_state`。

## 7. 一次请求的 provider 路由链路

```text
config 里写了 model="deepseek-chat" 或 provider="deepseek"
  → factory.make_provider(config)（providers/factory.py）
  → 用 config 的 provider 字段匹配 ProviderSpec
  → 根据 spec.backend 实例化对应实现（openai_compat / anthropic / ...）
  → AgentLoop.runtime_resolver（ModelRuntimeResolver）持有 LLMRuntime
  → runner._request_model → provider.chat_with_retry / chat_stream_with_retry
```

## 8. 你该重点看的源码细节

| 问题 | 位置 |
|---|---|
| 网关为什么优先 | `PROVIDERS` 顺序注释（[registry.py#L186-L188](file:///d:/project/nanobot/nanobot/providers/registry.py#L186-L188)） |
| 自动识别三种策略 | `detect_by_key_prefix` / `detect_by_base_keyword` / `keywords` |
| 各厂 thinking 方言 | `thinking_style` / `gateway_reasoning_style`（[registry.py#L85-L96](file:///d:/project/nanobot/nanobot/providers/registry.py#L85-L96)） |
| 自定义 provider | `create_dynamic_spec`（[registry.py#L766-L784](file:///d:/project/nanobot/nanobot/providers/registry.py#L766-L784)） |
| provider 实现接口 | `LLMProvider`（[base.py#L302](file:///d:/project/nanobot/nanobot/providers/base.py#L302)） |

## 9. 常见坑点

1. **别在 keywords 里放太宽泛的词**：如 `("step",)` 这种会误匹配其他模型名——匹配是"先命中先赢"。
2. **reasoning 参数各家不通用**：`reasoning_effort` 在 Mistral 上直接传会 400，必须走 remap；Magistral 干脆不能传。
3. **网关 strip_model_prefix**：AiHubMix 不认识 `anthropic/claude-3`，要剥成 `claude-3`（`strip_model_prefix=True`）。
4. **新增 provider 只改两处**：文件头注释明确说"加一个 ProviderSpec + 加一个 config schema 字段就完事"——这是数据驱动设计的核心收益。

## Related

- [nanobot AgentLoop 与 AgentRunner 源码精读](./nanobot-agentloop-runner.md) — runner 如何调 provider
- [nanobot 源码阅读指南](./nanobot-source-reading-guide.md)
- [AI 索引](./index.md)
