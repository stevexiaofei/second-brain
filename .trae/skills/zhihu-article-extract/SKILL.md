---
name: zhihu-article-extract
description: Extract full text, LaTeX formulas, and code from Zhihu articles or answers by browser rendering. Use when the user gives a zhihu.com link and WebFetch fails. Do not use for ChatGPT or WeChat pages.
---

# Zhihu Article Extract

Direct HTTP fetching of Zhihu does not work. `WebFetch` returns nothing usable, and `curl` gets a 403 or an anti-bot redirect shell. The content exists only in the rendered DOM, so render it in a browser and read the DOM.

## Scope

| URL form | Meaning |
|---|---|
| `zhuanlan.zhihu.com/p/<id>` | 专栏文章 — single article, content root `.Post-RichTextContainer .RichText` |
| `www.zhihu.com/question/<qid>/answer/<aid>` | 单个回答 — content root `.RichContent-inner .RichText` |
| `www.zhihu.com/question/<qid>` | 问题页 — multiple answers, only the first is expanded |

For a question page, ask which answer is wanted (or which ones) before extracting; do not silently return only the first.

## Procedure

1. `browser_navigate` to the URL and wait for the article body to render.
2. Scroll through the whole article, then scroll back. Link cards load lazily and their `.LinkCard-title` span stays empty until rendered; the script reads the `data-text` attribute on the card's `<a>` instead, which is populated on first paint, and warns only when a card yields no title at all.
3. If an expand control is present (`.ContentItem-more button`, or a button whose text contains 阅读全文 / 展开), click it first and wait for the body to grow. Question pages need every collapsed answer expanded separately.
4. Run `scripts/extract-zhihu.js` via `browser_evaluate`, passing the file content as the expression. It returns `{url, title, author, time, markdown, counts, warnings}`.
5. Check `warnings` and `counts`. Non-empty `warnings` means the read is incomplete — resolve it before reporting. If `error` is returned, the page is a login wall, a captcha, or an unlisted URL form; report that instead of guessing.
6. Report `title` / `author` / `time` and the markdown body. Preserve the markdown verbatim — do not summarize unless the user asked for a summary.

## Formulas are recoverable

Zhihu renders math as `span.ztext-math` carrying the **original LaTeX in the `data-tex` attribute**. The script converts these to `$...$`, and falls back to `img.eeimg[alt]` when `data-tex` is absent.

Never report a formula as an unreadable image without checking `data-tex` first — that is the difference between a usable extraction and a lossy one.

## Figures are pixels

Body illustrations are real `<figure>`/`<img>` elements. Their pixel content cannot become text. The script emits a `[图片]` placeholder with the alt text and figcaption. Say explicitly which parts of the article are figures when the argument depends on them.

## Known limitations

- Code block language tags are inferred from `class` attributes and are often missing. Verify before republishing code as a specific language.
- Answer ordering and "推荐" answers on question pages depend on runtime state; the extracted order is the DOM order, not necessarily the highest-voted.
- Deleted, folded, or paid-content articles return no content root. That is a real absence, not a fetch failure.

## Related skills

- `proxy-web-fetch` — only helps when the failure is network reachability. Zhihu failures are anti-bot, not network, so a proxy does not fix them.
- `chatgpt-share-extract` — separate mechanism (turbo-stream decode), do not substitute.
