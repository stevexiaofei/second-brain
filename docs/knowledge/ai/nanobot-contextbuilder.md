---
title: nanobot ContextBuilder 源码精读
type: concept
status: seed
tags: [AI, Agents, ContextBuilder, Prompt, Memory, Skills, Nanobot, Source Code]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\nanobot\nanobot\agent\context.py
---

# nanobot ContextBuilder 源码精读

> 本文基于本地源码 `d:\project\nanobot\nanobot\agent\context.py`（315 行）拆解，目标是让你看懂"发给模型的 messages 到底是怎么拼出来的"。

## 一句话理解

`ContextBuilder` 是 nanobot 的 **prompt 编译器**：把散落在 workspace、memory、skills、session、附件里的信息，编译成一次模型调用需要的 `[system] + [history] + [current]` 消息列表。

## 1. 类骨架与常量

[context.py#L52-L66](file:///d:/project/nanobot/nanobot/agent/context.py#L52-L66)：

```python
class ContextBuilder:
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md"]
    _SKIPPABLE_DEFAULTS = {"AGENTS.md", "USER.md"}
    _MAX_RECENT_HISTORY = 50       # 最近记忆最多取 50 条
    _MAX_HISTORY_TOKENS = 8_000    # 最近记忆 section 硬上限 8000 token

    def __init__(self, workspace, timezone=None, disabled_skills=None):
        self.memory = MemoryStore(workspace)   # 长期记忆
        self.skills = SkillsLoader(workspace, ...)  # 技能加载
```

注意两个边界常量：**记忆条数上限 50、token 上限 8000**——这是控制 system prompt 体积的硬护栏。

## 2. 四个核心方法（调用链）

```text
build_messages()          ← 顶层入口，返回完整 messages
  ├── build_system_prompt()   ← system prompt
  └── build_current_message() ← 当前轮消息
        └── build_user_content()  ← 文本 + 图片 → content
```

### 2.1 `build_messages()`：顶层装配

[context.py#L206-L265](file:///d:/project/nanobot/nanobot/agent/context.py#L206-L265)：

```python
def build_messages(self, history, current_message, *, media=None, channel=None, ...):
    # 1. 从当前消息中检测显式调用的 skill（如 @skill_name）
    active_skill_names = self.skills.get_explicitly_invoked_skills(current_message)

    messages = [
        {"role": "system", "content": self.build_system_prompt(
            active_skill_names=active_skill_names, ...)},
        *history,   # 历史消息原样展开
    ]
    current = self.build_current_message(current_message, media=media, ...)

    # 2. 关键细节：如果 history 最后一条与当前消息同 role，
    #    则合并 content 而不是 append（保证 role 交替）
    if messages[-1].get("role") == current_role:
        last["content"] = self._merge_message_content(last["content"], current["content"])
        ...
        messages[-1] = last
        return messages
    messages.append(current)
    return messages
```

**role 交替的两种形态**：
- 都是字符串 → `f"{left}\n\n{right}"` 简单拼接
- 有 multimodal block（list）→ `_to_blocks` 把两端都规范成 `{"type": ...}` block 列表再拼接（[context.py#L147-L167](file:///d:/project/nanobot/nanobot/agent/context.py#L147-L167)）

### 2.2 `build_system_prompt()`：system prompt 的六大 section

[context.py#L68-L127](file:///d:/project/nanobot/nanobot/agent/context.py#L68-L127)：

```python
parts = [
    self._get_identity(...),          # ① 身份：workspace 路径 + 平台 + 时区
]
bootstrap = self._load_bootstrap_files(root)   # ② AGENTS.md/SOUL.md/USER.md
parts.append(render_template("agent/tool_contract.md"))  # ③ 工具契约模板
if include_memory:
    parts.append(f"# Long-term Memory\n{memory}")         # ④ 长期记忆
if active_skills:
    parts.append(f"# Active Skills\n{...}")               # ⑤ 激活技能 + 技能摘要
if include_memory_recent_history:
    parts.append(f"# Recent History\n{...}")              # ⑥ 近期记忆（≤50 条 / ≤8000 token）
if session_summary:
    parts.append(f"[Archived Context Summary]\n{...}")    # ⑦ 压缩后的旧上下文摘要
return "\n\n---\n\n".join(parts)
```

最终 system prompt 结构：

```text
[Identity]                ← 你是谁、工作目录、平台、时区
---
[AGENTS.md / SOUL.md / USER.md]   ← 项目指令 + 个人画像
---
[Tool Contract]           ← 模型与工具的行为契约（模板渲染）
---
[Long-term Memory]        ← MEMORY.md 内容（用户自定义过才注入）
---
[Active Skills]           ← 常驻技能 + 本轮显式调用的技能
---
[Recent History]          ← dream cursor 之后的近期活动
---
[Archived Context Summary] ← session 压缩摘要（可选）
```

### 2.3 bootstrap 文件的加载细节（容易被忽略）

[context.py#L169-L196](file:///d:/project/nanobot/nanobot/agent/context.py#L169-L196)：

```python
sources = [
    ("AGENTS.md", project_root),   # 项目级：来自当前 workspace
    ("SOUL.md",   self.workspace), # agent 全局：来自 agent 根
    ("USER.md",   self.workspace), # agent 全局
]
for filename, root in sources:
    content = root / filename 读取
    # 跳过"模板原样"的默认文件（用户没自定义就不注入，省 token）
    if filename in _SKIPPABLE_DEFAULTS and self._is_template_content(content, filename):
        continue
```

**设计**：`AGENTS.md`/`USER.md` 若是默认模板内容则跳过（省 token），`SOUL.md` 若命中 legacy 模板则替换为 bundled 模板——即"只有用户真正自定义过的文件才进入 prompt"。

### 2.4 `build_user_content()`：图片 → base64 data URL

[context.py#L286-L315](file:///d:/project/nanobot/nanobot/agent/context.py#L286-L315)：

```python
def build_user_content(self, text, image_paths):
    if not image_paths:
        return text
    for path in image_paths:
        raw = p.read_bytes()
        mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]  # 从字节重新探测
        b64 = base64.b64encode(raw).decode()
        image_blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
            "_meta": {"path": str(p)},   # 记住原始路径，持久化时替换回占位符
        })
    return image_blocks + [{"type": "text", "text": text}]
```

**关键点**：
1. **从字节重探测 MIME** 而不是用扩展名——附件可能在路由后被修改。
2. `_meta.path` 被带到后端的 `_sanitize_persisted_blocks`（loop.py），落盘时把 data URL 替换回 `image_placeholder_text(path)`，**避免会话文件里存大 base64**。

## 3. 与 AgentLoop 的协作点

在 [loop.py#L734-L750](file:///d:/project/nanobot/nanobot/agent/loop.py#L734-L750) 的 `_build_initial_messages` 中调用：

```python
return self.context.build_messages(
    history=ctx.history,
    current_message=ctx.msg.content,
    media=ctx.msg.media if ctx.kind is TurnKind.USER and ctx.msg.media else None,
    channel=ctx.delivery.route.channel,
    session_summary=ctx.pending_summary,       # compact 阶段的产出
    workspace=scope.project_path,              # workspace scope 解析后的项目路径
    runtime_context_blocks=ctx.runtime_context_blocks,  # 每轮解析的动态上下文块
    include_memory=ctx.session.policy.persist, # session 策略决定是否带记忆
    include_memory_recent_history=not ctx.ephemeral,
    session_key=ctx.session.key,
    unified_session=self._unified_session,
)
```

几个联动点：
- `session.policy.persist=False` → 不注入长期记忆（临时会话省钱）
- `ephemeral=True` → 不带 recent history
- `runtime_context_blocks` 由 `_resolve_runtime_context_for_turn` 从工具和外部注册的 `RuntimeContextProvider` 动态解析

## 4. 你该重点看的源码细节

| 问题 | 位置 |
|---|---|
| 图片怎么进 prompt | [context.py#L286-L315](file:///d:/project/nanobot/nanobot/agent/context.py#L286-L315) |
| 默认文件怎么被跳过 | `_is_template_content`（[context.py#L198-L204](file:///d:/project/nanobot/nanobot/agent/context.py#L198-L204)） |
| role 交替怎么合并 | `_merge_message_content`（[context.py#L147-L167](file:///d:/project/nanobot/nanobot/agent/context.py#L147-L167)） |
| 技能怎么激活 | `skills.get_explicitly_invoked_skills` / `get_always_skills` |
| 记忆游标 | `memory.get_last_dream_cursor()`（Dream 之后才计入 Recent History） |

## 5. 常见坑点

1. **不要以为 system prompt 只有模板**：它由 identity/bootstrap/tool contract/memory/skills/recent history 六部分动态拼成，token 开销最大的通常是 skills 摘要和 recent history。
2. **图片 data URL 不进 session 文件**：靠 `_meta.path` 在持久化层替换成占位符，读源码时看到 `image_placeholder_text` 不要奇怪。
3. **同 role 合并是双刃剑**：合并省 token 且满足 provider 约束，但会丢失消息边界——理解 `build_messages` 里那个 `if` 分支很重要。

## Related

- [nanobot AgentLoop 与 AgentRunner 源码精读](./nanobot-agentloop-runner.md) — build 阶段的调用方
- [nanobot Tool Registry 源码精读](./nanobot-tool-registry.md) — tool contract 的 schema 来源
- [nanobot 源码阅读指南](./nanobot-source-reading-guide.md)
- [AI 索引](./index.md)
