// Markdown for the chat: the text is escaped first, then formatted (the classic app's renderer, ported as it was).
// Output is an HTML string for dangerouslySetInnerHTML; nothing but this module's own tags (and, inside a code block, the highlighter's
// own <span class="hljs-..."> around text it has escaped) can appear in it.
import { highlight } from "./highlight"

const esc = (s: string) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!)

function inline(s: string): string {
  const codes: string[] = []
  s = s.replace(/`([^`\n]+)`/g, (_, c: string) => { codes.push(c); return `\u0000${codes.length - 1}\u0000` })
  s = esc(s)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<em>$2</em>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
  return s.replace(/\u0000(\d+)\u0000/g, (_, i: string) => `<code class="inline">${esc(codes[+i])}</code>`)
}

function codeBlock(lang: string, code: string, open = false): string {
  const coloured = highlight(lang, code, open)                  // null: no language, one that is not known, or too long: plain
  return `<div class="code-block"><div class="code-head"><span>${esc(lang || "code")}</span>` +
    `<button type="button" data-code-copy aria-label="Copy code">Copy</button></div>` +
    `<pre><code${coloured === null ? "" : ' class="hljs"'}>${coloured ?? esc(code)}</code></pre></div>`
}

function blocks(text: string): string {
  const out: string[] = []
  const lines = text.split("\n")
  let para: string[] = []
  let list: { tag: string; items: string[] } | null = null
  const flushPara = () => { if (para.length) out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = [] }
  const flushList = () => {
    if (list) out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`)
    list = null
  }
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i]
    let m: RegExpMatchArray | null
    if (!l.trim()) { flushPara(); flushList(); continue }
    if ((m = l.match(/^(#{1,6})\s+(.*)$/))) {
      flushPara(); flushList()
      const tag = m[1].length <= 2 ? "h3" : "h4"
      out.push(`<${tag}>${inline(m[2])}</${tag}>`)
      continue
    }
    if (/^\s*([-*_])\s*\1\s*\1(?:\s*\1)*\s*$/.test(l)) { flushPara(); flushList(); out.push("<hr>"); continue }
    if ((m = l.match(/^>\s?(.*)$/))) { flushPara(); flushList(); out.push(`<blockquote>${inline(m[1])}</blockquote>`); continue }
    if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1])) {
      flushPara(); flushList()
      const cells = (row: string) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()))
      let html = `<table><thead><tr>${cells(l).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`
      i += 2
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) html += `<tr>${cells(lines[i++]).map((c) => `<td>${c}</td>`).join("")}</tr>`
      i--
      out.push(html + "</tbody></table>")
      continue
    }
    if ((m = l.match(/^\s*(?:[-*+]|(\d+)[.)])\s+(.*)$/))) {
      flushPara()
      const tag = m[1] ? "ol" : "ul"
      if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] } }
      list.items.push(m[2])
      continue
    }
    if (list && /^\s{2,}\S/.test(l)) { list.items[list.items.length - 1] += " " + l.trim(); continue }
    flushList()
    para.push(l)
  }
  flushPara(); flushList()
  return out.join("")
}

export function markdown(text: string): string {
  let html = ""
  let rest = text
  for (;;) {
    const m = rest.match(/(^|\n)```([^\n`]*)\n/)
    if (!m) { html += blocks(rest); break }
    html += blocks(rest.slice(0, m.index))
    rest = rest.slice(m.index! + m[0].length)
    const end = rest.match(/(^|\n)```[ \t]*(\n|$)/)
    if (!end) { html += codeBlock(m[2].trim(), rest, true); break }          // still streaming
    html += codeBlock(m[2].trim(), rest.slice(0, end.index))
    rest = rest.slice(end.index! + end[0].length)
  }
  return html
}
