import { describe, expect, test } from "bun:test"
import { markdown } from "./markdown"

// Code in an answer is coloured like an IDE (highlight.js, see lib/highlight.ts). The colours are spans around the text and must
// never change the text or let anything in as markup: the answer is the model's, and the model reads web pages and files.
const fence = (lang: string, code: string, closed = true) => "```" + lang + "\n" + code + (closed ? "\n```" : "")
const inner = (html: string) => (html.match(/<pre><code[^>]*>([\s\S]*?)<\/code><\/pre>/) ?? [])[1] ?? ""
const plain = (htmlInner: string) => htmlInner.replace(/<[^>]+>/g, "").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&#x27;/g, "'").replace(/&amp;/g, "&")

describe("code is coloured", () => {
  test("Python: keywords, strings, numbers and calls get their own classes", () => {
    const html = markdown(fence("python", 'def area(r):\n    return 3.14 * r  # circle\nprint("x", area(2))'))
    const code = inner(html)
    expect(code).toContain('<span class="hljs-keyword">def</span>')
    expect(code).toContain("hljs-number")
    expect(code).toContain("hljs-string")
    expect(code).toContain("hljs-comment")
    expect(html).toContain('class="hljs"')
  })
  test("the usual languages and their short names are known", () => {
    for (const lang of ["py", "js", "ts", "tsx", "json", "bash", "sh", "powershell", "ps1", "c", "cpp", "c++", "cuda", "csharp", "java", "go", "rust", "rs", "sql", "yaml", "yml", "html", "xml", "css", "md", "diff", "toml", "dockerfile", "kotlin"]) {
      expect(markdown(fence(lang, "x = 1")), lang).toContain('class="hljs"')              // a known language: the block is marked as coloured
    }
  })
  test("a line of info after the language does not hide it", () => {
    expect(markdown(fence("python title=a.py", "def f(): pass"))).toContain("hljs-keyword")
  })
  test("no language, or one that is not known, stays plain text (no guessing)", () => {
    for (const lang of ["", "nonsense", "text", "plaintext"]) {
      const html = markdown(fence(lang, "def f(): pass"))
      expect(html, lang).not.toContain("hljs")
      expect(inner(html)).toBe("def f(): pass")
    }
  })
})

describe("what is shown is what was written", () => {
  const samples: [string, string][] = [
    ["javascript", 'const a = "<b>&</b>";\nif (x < 3 && y > 2) { document.write(\'<script>alert(1)</script>\') }'],
    ["html", '<div onclick="steal()"><script>alert(1)</script><img src=x onerror=alert(1)></div>'],
    ["python", 'print("<script>alert(1)</script>")  # </code></pre><script>x</script>'],
    ["bash", "echo '&lt;' > out.txt; cat <<EOF\n<tag>\nEOF"],
    ["", '<script>alert(1)</script> & "quotes" and \'single\''],
  ]
  test("the text is the same with the colours taken off", () => {
    for (const [lang, code] of samples) expect(plain(inner(markdown(fence(lang, code)))), lang).toBe(code)
  })
  test("no markup of the model's can get in: every < is one of the highlighter's own spans", () => {
    for (const [lang, code] of samples) {
      const body = inner(markdown(fence(lang, code)))
      expect(body.replace(/<\/?span( class="(hljs-[a-z_ -]+|language-[a-z]+)")?>/g, ""), lang).not.toContain("<")
      expect(body, lang).not.toMatch(/<(script|img|div|iframe|a)\b/i)         // (the words onclick, onerror may be shown: they are text)
    }
  })
  test("a block that closes the page's own tags cannot end the block early", () => {
    const html = markdown(fence("python", 'x = "</code></pre></div><h1>hi</h1>"'))
    expect(html.match(/<pre>/g)).toHaveLength(1)
    expect(html).not.toContain("<h1>")
  })
})

describe("while it is being written", () => {
  test("a block that is still open is coloured too, and closes into the same thing", () => {
    const open = markdown(fence("python", "def f():\n    return 1", false))
    expect(open).toContain("hljs-keyword")
    const closed = markdown(fence("python", "def f():\n    return 1", true))
    expect(plain(inner(open))).toBe(plain(inner(closed)))
  })
  test("a very long block is left plain, so a long answer stays quick", () => {
    const big = "x = 1\n".repeat(20_000)
    const html = markdown(fence("python", big))
    expect(html).not.toContain("hljs-")
    expect(plain(inner(html))).toBe(big)
  })
  test("a long block that is still open is plain until it closes (it would be coloured again at every word)", () => {
    const part = "y = 2\n".repeat(2_000)
    expect(markdown(fence("python", part, false))).not.toContain("hljs-")
    expect(markdown(fence("python", part, true))).toContain("hljs-")
  })
})

describe("the rest of the markdown is as it was", () => {
  test("text around a block, inline code and the copy button", () => {
    const html = markdown("Use `x` here:\n\n" + fence("py", "print(1)") + "\n\nDone.")
    expect(html).toContain('<code class="inline">x</code>')
    expect(html).toContain("data-code-copy")
    expect(html).toContain("<p>Done.</p>")
    expect(html).toContain("<span>py</span>")
  })
})
