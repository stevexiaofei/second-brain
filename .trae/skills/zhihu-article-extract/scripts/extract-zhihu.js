// Zhihu article/answer extractor. Run inside the page via browser_evaluate.
// Pass this whole file as the expression. Returns a JSON-serializable object.
(() => {
  const pick = (sels) => {
    for (const s of sels) {
      const e = document.querySelector(s);
      if (e && e.children.length) return e;
    }
    return null;
  };

  const root = pick([
    '.Post-RichTextContainer .RichText',
    '.RichContent-inner .RichText',
    '.RichText',
    '.RichContent-inner',
    '.QuestionAnswer-content',
  ]);

  if (!root) {
    return {
      error: 'content root not found',
      tried: [
        '.Post-RichTextContainer .RichText',
        '.RichContent-inner .RichText',
        '.RichText',
        '.RichContent-inner',
        '.QuestionAnswer-content',
      ],
      hint: 'Page may be a login wall, a captcha, or a URL form not covered here.',
      title: document.title,
    };
  }

  const BLOCK = new Set([
    'p', 'div', 'section', 'article', 'figure', 'pre', 'blockquote', 'table',
    'ul', 'ol', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr',
  ]);

  // Zhihu wraps outbound links in link.zhihu.com/?target=<urlencoded>.
  const unwrapLink = (href) => {
    if (href.includes('link.zhihu.com')) {
      const m = href.match(/[?&]target=([^&]+)/);
      if (m) {
        try { return decodeURIComponent(m[1]); } catch (e) { /* keep href */ }
      }
    }
    return href;
  };

  // Inline rendering: text nodes, math, links, emphasis, breaks, images.
  const inline = (node) => {
    let out = '';
    for (const n of node.childNodes) {
      if (n.nodeType === 3) {
        out += n.nodeValue.replace(/\s+/g, ' ');
        continue;
      }
      if (n.nodeType !== 1) continue;
      const tag = n.tagName.toLowerCase();

      // Math: Zhihu keeps the original LaTeX in data-tex on span.ztext-math.
      if (n.classList && n.classList.contains('ztext-math')) {
        const tex = n.getAttribute('data-tex') || '';
        if (tex) { out += '$' + tex + '$'; continue; }
      }
      if (tag === 'img') {
        const alt = n.getAttribute('alt') || '';
        // Fallback for math rendered as an image without data-tex.
        if (n.classList && (n.classList.contains('ztext-math') || n.classList.contains('eeimg'))) {
          if (alt) { out += '$' + alt + '$'; continue; }
        }
        if (n.closest && n.closest('span.ztext-math')) continue;
        out += '[图片' + (alt ? ' — ' + alt : '') + ']';
        continue;
      }
      if (tag === 'br') { out += '\n'; continue; }
      if (tag === 'code') { out += '`' + n.textContent + '`'; continue; }
      if (tag === 'a') {
        const txt = inline(n).trim();
        if (!txt) continue;
        const href = unwrapLink(n.getAttribute('href') || '');
        const noise = href.includes('zhida.zhihu.com') || href.includes('/search?');
        out += href.startsWith('http') && !noise ? '[' + txt + '](' + href + ')' : txt;
        continue;
      }
      if (tag === 'b' || tag === 'strong') { out += '**' + inline(n).trim() + '**'; continue; }
      if (tag === 'i' || tag === 'em') { out += '*' + inline(n).trim() + '*'; continue; }
      out += inline(n);
    }
    return out;
  };

  const renderCode = (pre) => {
    const codeEl = pre.querySelector('code') || pre;
    const cls = (codeEl.className || '') + ' ' + (pre.className || '') +
                ' ' + (pre.parentElement ? pre.parentElement.className || '' : '');
    let lang = '';
    let m = cls.match(/language-([\w+#-]+)/i);
    if (m) lang = m[1];
    if (!lang) {
      m = cls.match(/highlight[-\s]([\w+#-]+)/i);
      if (m && m[1] !== 'highlight') lang = m[1];
    }
    if (!lang) {
      const badge = pre.parentElement && pre.parentElement.querySelector('.lang, .code-lang');
      if (badge) lang = badge.textContent.trim().toLowerCase();
    }
    const text = (codeEl.innerText || codeEl.textContent || '').replace(/\n+$/, '');
    return '```' + lang + '\n' + text + '\n```';
  };

  const renderTable = (tbl) => {
    const rows = [...tbl.querySelectorAll('tr')].map((tr) =>
      [...tr.children].map((td) => inline(td).trim().replace(/\|/g, '\\|'))
    );
    if (!rows.length) return '';
    const width = Math.max(...rows.map((r) => r.length));
    const pad = (r) => { const c = [...r]; while (c.length < width) c.push(''); return c; };
    const head = pad(rows[0]);
    return [
      '| ' + head.join(' | ') + ' |',
      '| ' + head.map(() => '---').join(' | ') + ' |',
      ...rows.slice(1).map((r) => '| ' + pad(r).join(' | ') + ' |'),
    ].join('\n');
  };

  const lines = [];
  let loadingCards = 0;
  const walk = (el) => {
    const tag = el.tagName.toLowerCase();

    // Link cards are <a class="LinkCard"> wrappers around an image plus a title.
    // Without this branch the card's text is silently dropped.
    if (el.classList && el.classList.contains('LinkCard')) {
      const titleEl = el.querySelector('.LinkCard-title');
      // data-text on the <a> is populated before the card body renders, so it is
      // the only reliable source while .LinkCard-title still carries "loading".
      const fromSpan = titleEl ? titleEl.innerText.trim().split('\n')[0] : '';
      const fromAttr = (el.getAttribute('data-text') || '').trim();
      const title = fromAttr || fromSpan;
      const href = unwrapLink(el.getAttribute('href') || '');
      // Only a real gap: neither the fallback nor the rendered span produced a title.
      if (!title) loadingCards++;
      if (title || href) lines.push('[链接卡片] ' + title + ' -> ' + href);
      return;
    }

    switch (tag) {
      case 'h1': lines.push('# ' + inline(el).trim()); return;
      case 'h2': lines.push('## ' + inline(el).trim()); return;
      case 'h3': lines.push('### ' + inline(el).trim()); return;
      case 'h4': lines.push('#### ' + inline(el).trim()); return;
      case 'h5': lines.push('##### ' + inline(el).trim()); return;
      case 'h6': lines.push('###### ' + inline(el).trim()); return;
      case 'hr': lines.push('---'); return;
      case 'pre': lines.push(renderCode(el)); return;
      case 'blockquote': {
        const body = inline(el).trim();
        if (body) lines.push(body.split('\n').map((l) => '> ' + l).join('\n'));
        return;
      }
      case 'figure': {
        const img = el.querySelector('img');
        const alt = (img && img.getAttribute('alt')) || '';
        const cap = el.querySelector('figcaption');
        lines.push('[图片' + (alt ? ' — ' + alt : '') + ']' +
                   (cap ? '\n> ' + cap.textContent.trim() : ''));
        return;
      }
      case 'table': {
        const t = renderTable(el);
        if (t) lines.push(t);
        return;
      }
      case 'ul':
      case 'ol': {
        let i = 1;
        for (const li of el.children) {
          if (li.tagName.toLowerCase() !== 'li') continue;
          const body = inline(li).trim();
          if (body) lines.push((tag === 'ol' ? i++ + '. ' : '- ') + body);
        }
        return;
      }
      default: {
        const kids = [...el.children];
        const hasBlock = kids.some((k) =>
          BLOCK.has(k.tagName.toLowerCase()) || (k.classList && k.classList.contains('LinkCard'))
        );
        if (hasBlock) {
          kids.forEach(walk);
        } else {
          const t = inline(el).trim();
          if (t) lines.push(t);
        }
      }
    }
  };

  [...root.children].forEach(walk);

  const text = (sel) => {
    const e = document.querySelector(sel);
    return e ? e.innerText.trim() : '';
  };

  return {
    url: location.href,
    title: text('.Post-Title') || text('h1') || document.title,
    author: text('.AuthorInfo-name') || text('.AuthorInfo a.UserLink-link'),
    time: text('.ContentItem-time') || text('.Post-Time') || text('.ContentItem-status'),
    markdown: lines.filter(Boolean).join('\n\n'),
    counts: {
      formula: root.querySelectorAll('span.ztext-math').length,
      figure: root.querySelectorAll('figure').length,
      image: root.querySelectorAll('img').length,
      code: root.querySelectorAll('pre').length,
      table: root.querySelectorAll('table').length,
      linkCard: document.querySelectorAll('.LinkCard').length,
      block: root.children.length,
    },
    warnings: [
      loadingCards ? loadingCards + ' link card(s) yielded no title — reload and rerun' : '',
      document.querySelector('.ContentItem-more button') ? 'expand button still present — content may be truncated' : '',
      document.querySelector('.signFlowModal, .Modal-wrapper') ? 'login modal detected' : '',
      document.querySelector('.Captcha') ? 'captcha detected' : '',
    ].filter(Boolean),
  };
})()
