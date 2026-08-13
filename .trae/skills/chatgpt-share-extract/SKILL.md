---
name: "chatgpt-share-extract"
description: "Extracts full conversation from a ChatGPT share link (chatgpt.com/share/...) via proxy download + turbo-stream decode. Invoke when user provides a ChatGPT share URL and asks to extract/organize/save the conversation content."
---

# ChatGPT Share Extractor

从 ChatGPT 分享链接（`chatgpt.com/share/...`）提取完整对话内容（角色 + 文本），保存为 JSON 或整理成笔记。该链接在国内网络通常无法直接访问，需要走代理。

## 触发条件

- 用户提供 `https://chatgpt.com/share/xxx` 链接，要求"提取/整理/保存"对话内容
- WebFetch 对 share 链接返回空/失败
- 浏览器访问 share 链接报 `ERR_CONNECTION_RESET`

## 完整流程

### Step 1: 先尝试直接访问

```powershell
# WebFetch 或 curl 直接访问
# 若成功（返回非空 HTML），跳过代理，直接进入 Step 4
```

大多数情况下 chatgpt.com 直接访问会失败，直接进入 Step 2。

### Step 2: 通过代理下载页面

**代理凭据必须在运行时向用户询问，禁止硬编码/落盘。**

向用户索要：`host:port` + `user:password`。密码含特殊字符（如 `@`）需 URL 编码：

```powershell
New-Item -ItemType Directory -Path "_tmp_screenshot" -Force | Out-Null
curl.exe -sS `
  -x "http://USER:URL_ENCODED_PASSWORD@HOST:PORT" `
  -L `
  -o "_tmp_screenshot/page.html" `
  -w "HTTP %{http_code} | size=%{size_download} | url=%{url_effective}\n" `
  -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36" `
  "TARGET_SHARE_URL" `
  --max-time 60
```

- 成功标志：`HTTP 200`，`size` 数百 KB
- `HTTP 407`：代理认证失败，检查密码编码
- `HTTP 403`：加 `-A` User-Agent

### Step 3: 确保 turbo-stream 可用

```powershell
npm install --no-save turbo-stream
```

### Step 4: 解码 turbo-stream 提取消息

**关键点：对话数据藏在 `<script>` 内的 `streamController.enqueue("...")` 调用中**，不是 `enqueue("...")`。enqueue 参数是一个带转义的字符串，需要先反转义，再交给 turbo-stream 解码。

```javascript
// _tmp_screenshot/extract.mjs
import { decode } from 'turbo-stream';
import fs from 'node:fs';

const html = fs.readFileSync('_tmp_screenshot/page.html', 'utf8');
const dataScripts = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map((m) => m[1]);

// 1. 找包含 streamController.enqueue 的 script（真正的 React Router 数据流）
let targetScript = dataScripts.find((s) => s.includes('streamController.enqueue'));
if (!targetScript) throw new Error('no streamController.enqueue script found');

// 2. 提取 enqueue("...") 的字符串参数（处理 \uXXXX / \" / \\ 转义）
const marker = 'streamController.enqueue("';
const start = targetScript.indexOf(marker);
let i = start + marker.length;
let raw = '';
while (i < targetScript.length) {
  const c = targetScript[i];
  if (c === '\\') {
    const next = targetScript[i + 1];
    if (next === '"') { raw += '"'; i += 2; continue; }
    if (next === '\\') { raw += '\\'; i += 2; continue; }
    if (next === 'n') { raw += '\n'; i += 2; continue; }
    if (next === 'r') { raw += '\r'; i += 2; continue; }
    if (next === 't') { raw += '\t'; i += 2; continue; }
    if (next === 'u') { raw += String.fromCharCode(parseInt(targetScript.slice(i + 2, i + 6), 16)); i += 6; continue; }
    raw += c; i++; continue;
  }
  if (c === '"') break;
  raw += c; i++;
}

// 3. turbo-stream 解码（用 ReadableStream 传入字符串）
const stream = new ReadableStream({ start(c) { c.enqueue(raw); c.close(); } });
const decoded = await decode(stream);

// 4. 关键：decoded 是【扁平数组】+【引用对象】结构
//    - 形如 [ {...}, "loaderData", {...}, "actionData", ... ]
//    - 对象引用 {"_键索引": "值索引"} 表示：键名在 decoded[键索引]，值在 decoded[值索引]
//    - linear_conversation 的值在标记 "linear_conversation" 的下一个数组元素

function resolve(idx, depth = 0, seen = new Set()) {
  if (depth > 10 || idx === undefined || idx === null || typeof idx !== 'number') return idx;
  if (seen.has(idx)) return undefined;
  seen = new Set(seen);
  seen.add(idx);
  const v = decoded[idx];
  if (Array.isArray(v)) {
    return v.map((x) => (typeof x === 'number' ? resolve(x, depth + 1, seen) : x));
  }
  if (v && typeof v === 'object') {
    const out = {};
    for (const k of Object.keys(v)) {
      if (k.startsWith('_')) {
        const keyName = decoded[Number(k.slice(1))];
        if (typeof keyName === 'string') out[keyName] = resolve(v[k], depth + 1, seen);
      } else {
        out[k] = resolve(v[k], depth + 1, seen);
      }
    }
    return out;
  }
  return v;
}

// 5. 定位 linear_conversation
let linearIdx = -1;
for (let idx = 0; idx < decoded.length - 1; idx++) {
  if (decoded[idx] === 'linear_conversation') { linearIdx = idx + 1; break; }
}
const linear = resolve(linearIdx);

// 6. 提取消息：node.message.author.role + node.message.content.parts
const messages = [];
for (const node of linear) {
  if (!node || typeof node !== 'object') continue;
  const msg = node.message;
  if (!msg) continue;
  const role = msg.author?.role;
  const parts = msg.content?.parts;
  if (!parts) continue;
  const text = parts.map((p) => (typeof p === 'string' ? p : JSON.stringify(p))).join('\n');
  messages.push({ role, text });
}

fs.writeFileSync('_tmp_screenshot/messages.json', JSON.stringify(messages, null, 2), 'utf8');
console.log(`Extracted ${messages.length} messages`);
```

### Step 5: 筛选与使用

```javascript
// 按角色过滤
const conversation = messages.filter((m) => m.role === 'user' || m.role === 'assistant');

// 按关键词筛选（如提取某主题相关消息）
const rvv = messages.filter((m) => /RVV|Softmax/i.test(m.text || ''));
```

终端直接打印中文可能乱码（PowerShell 编码问题），**优先写文件后用 Read 工具读取**。

### Step 6: 清理临时文件

```powershell
Remove-Item _tmp_screenshot/page.html, _tmp_screenshot/messages.json, _tmp_screenshot/extract.mjs -ErrorAction SilentlyContinue
```

## 常见问题

| 症状 | 原因 | 处理 |
|---|---|---|
| decoded 顶层无 `loaderData` 键 | 页面结构变了 | decoded 是扁平数组，改用 `streamController.enqueue` 定位 + 引用解析 |
| `linear_conversation` 值是数字数组 | 未解析引用 | 必须用 `resolve()` 递归解析引用对象 |
| 中文乱码 | 终端编码问题 | 写文件用 UTF-8，用 Read 工具读取 |
| `turbo-stream` decode 失败 | script 格式变化 | 检查 HTML 中是否还有其他 enqueue 调用，或手工 JSON 解析 |

## 安全规则

- **代理凭据运行时询问，禁止写入任何文件**（包括本 SKILL.md、临时文件、日志）
- 下载后立即解析、立即清理，不留中间文件
- 凭据只在单次 curl 命令中使用，用完即弃
