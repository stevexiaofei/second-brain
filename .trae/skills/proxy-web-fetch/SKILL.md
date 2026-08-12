---
name: "proxy-web-fetch"
description: "Fetches web content through an authenticated HTTP proxy when direct access fails. Invoke when WebFetch returns empty, browser reports ERR_CONNECTION_RESET, or user provides a URL that can't be reached directly."
---

# Proxy Web Fetch

When direct network access to a URL fails (WebFetch empty, browser ERR_CONNECTION_RESET, timeout), use an authenticated HTTP proxy + curl to download the page, then parse the content.

## When to Invoke

- `WebFetch` returns empty for a public URL
- Browser subagent reports `ERR_CONNECTION_RESET` / `ERR_ABORTED` / timeout
- User explicitly says to use a proxy
- User provides a URL from a service that may be blocked (chatgpt.com, google.com, etc.)

## Step-by-Step

### 1. Confirm direct access fails

First try `WebFetch` or `Invoke-WebRequest`. If it succeeds, no proxy needed.

### 2. Ask for proxy credentials

**Always ask the user for proxy credentials at runtime. Never hardcode or store credentials in files.**

If the user hasn't provided proxy info, ask:

> "直接访问失败。请提供代理地址和凭据（格式：`host:port` + `user:password`）"

### 3. Download with curl + proxy

After the user provides proxy credentials, construct the curl command:

```powershell
curl.exe -sS `
  -x "http://USERNAME:URL_ENCODED_PASSWORD@PROXY_HOST:PROXY_PORT" `
  -L `
  -o "_tmp_screenshot/page.html" `
  -w "HTTP %{http_code} | size=%{size_download} | url=%{url_effective}\n" `
  -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36" `
  "TARGET_URL" `
  --max-time 60
```

**CRITICAL: URL-encode special characters in the password before constructing the proxy URL.**
Common encodings: `@` → `%40`, `#` → `%23`, `:` → `%3A`, `/` → `%2F`, `?` → `%3F`

Example: if the password contains `@`, it must be encoded as `%40` in the proxy connection string, otherwise curl will interpret everything after `@` as the host.

If you see `HTTP 407`, the proxy requires authentication — check credentials and encoding.
If you see `HTTP 000`, the proxy itself is unreachable — check host/port.

### 4. Parse the downloaded HTML

The parsing strategy depends on the site:

#### General websites

Use Node.js to extract text:

```javascript
const fs = require('fs');
const html = fs.readFileSync('_tmp_screenshot/page.html', 'utf8');
// Strip tags, get text content
const text = html.replace(/<script[\s\S]*?<\/script>/g, '')
                 .replace(/<style[\s\S]*?<\/style>/g, '')
                 .replace(/<[^>]+>/g, '\n')
                 .replace(/\n{3,}/g, '\n\n')
                 .trim();
console.log(text);
```

#### ChatGPT share pages (chatgpt.com/share/...)

ChatGPT share pages use React Router turbo-stream format. The conversation data is embedded in `<script>` tags and requires `turbo-stream` to decode.

1. Install turbo-stream (if not already):
```powershell
npm install --no-save turbo-stream
```

2. Extract and decode:
```javascript
import { decode } from 'turbo-stream';
import fs from 'node:fs';

const html = fs.readFileSync('_tmp_screenshot/page.html', 'utf8');

// Find all <script> contents
const dataScripts = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);

// The largest script contains the turbo-stream data
// Look for the one with "enqueue" calls
let targetScript = dataScripts.find(s => s.includes('enqueue('));
if (!targetScript) {
  // Fallback: find the longest script
  targetScript = dataScripts.sort((a, b) => b.length - a.length)[0];
}

// Extract the enqueue argument (a JS-escaped string)
const start = targetScript.indexOf('enqueue("');
let i = start + 'enqueue("'.length;
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
    if (next === 'u') {
      const hex = targetScript.slice(i + 2, i + 6);
      raw += String.fromCharCode(parseInt(hex, 16));
      i += 6;
      continue;
    }
    raw += c; i++; continue;
  }
  if (c === '"') break;
  raw += c; i++;
}

// Decode via turbo-stream (pass string chunks via ReadableStream)
const stream = new ReadableStream({
  start(c) { c.enqueue(raw); c.close(); }
});
const decoded = await decode(stream);

// Navigate to conversation data
const data = decoded.loaderData['routes/share.$shareId.($action)'].serverResponse.data;
const linear = data.linear_conversation;

// Extract messages
const messages = [];
for (const node of linear) {
  if (!node || typeof node !== 'object') continue;
  const msg = node.message;
  if (!msg) continue;
  const role = msg.author?.role;
  const parts = msg.content?.parts;
  if (!parts) continue;
  const text = parts.map(p => typeof p === 'string' ? p : JSON.stringify(p)).join('\n');
  messages.push({ role, text, create_time: msg.create_time });
}

fs.writeFileSync('_tmp_screenshot/messages.json', JSON.stringify(messages, null, 2));
console.log(`Extracted ${messages.length} messages`);
```

3. Filter to user + assistant messages only:
```javascript
const conversation = messages.filter(m => m.role === 'user' || m.role === 'assistant');
```

### 5. Clean up

Delete downloaded HTML and intermediate files after extracting the needed content:

```powershell
Remove-Item _tmp_screenshot/page.html, _tmp_screenshot/messages.json -ErrorAction SilentlyContinue
```

## Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| `HTTP 407` | Proxy auth required | URL-encode password, check username |
| `HTTP 000` | Proxy unreachable | Check host/port, network |
| `HTTP 403` | Blocked by site | Add `-A` user agent header |
| Empty HTML | JS-rendered page | The page needs browser rendering; try browser subagent with `--proxy-server` flag |
| `turbo-stream` decode fails | Script format changed | Check ChatGPT version; inspect HTML manually for data patterns |

## Browser Subagent with Proxy

If the page requires JavaScript rendering and curl only gets a shell HTML, use the browser subagent with proxy:

```
browser_navigate to chrome://settings
# or launch Chrome with proxy flag:
# chrome.exe --proxy-server="http://PROXY_HOST:PROXY_PORT"
```

Note: Browser subagent proxy support depends on the browser launch configuration. If the browser subagent can't use a proxy, fall back to curl + manual parsing.

## Security Rules

**NO credentials (username, password, proxy address) should be hardcoded in this SKILL.md or any project file.**

- **Runtime-only**: Always ask the user to provide proxy host/port and credentials when the skill is invoked
- **No file storage**: Never write proxy credentials to any file (including temp files, memory files, or config files)
- **No logging**: Never include credentials in command output or log messages
- **Minimal scope**: Use credentials only for the specific download command, then discard them
- **Cleanup**: Delete all downloaded files immediately after extraction (Step 5)
