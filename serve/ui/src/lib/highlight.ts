// Code in an answer is coloured like an IDE: highlight.js (BSD-3-Clause, see REFERENCES.md), its core with the languages people ask a
// model for. The result is HTML made of the library's own <span class="hljs-..."> around escaped text; lib/markdown.ts puts it inside
// <pre><code>. Nothing is guessed: a block with no language, or one that is not known, stays plain.
import hljs from "highlight.js/lib/core"
import bash from "highlight.js/lib/languages/bash"
import c from "highlight.js/lib/languages/c"
import cmake from "highlight.js/lib/languages/cmake"
import cpp from "highlight.js/lib/languages/cpp"
import csharp from "highlight.js/lib/languages/csharp"
import css from "highlight.js/lib/languages/css"
import diff from "highlight.js/lib/languages/diff"
import dockerfile from "highlight.js/lib/languages/dockerfile"
import go from "highlight.js/lib/languages/go"
import ini from "highlight.js/lib/languages/ini"
import java from "highlight.js/lib/languages/java"
import javascript from "highlight.js/lib/languages/javascript"
import json from "highlight.js/lib/languages/json"
import kotlin from "highlight.js/lib/languages/kotlin"
import lua from "highlight.js/lib/languages/lua"
import makefile from "highlight.js/lib/languages/makefile"
import markdown from "highlight.js/lib/languages/markdown"
import php from "highlight.js/lib/languages/php"
import powershell from "highlight.js/lib/languages/powershell"
import python from "highlight.js/lib/languages/python"
import ruby from "highlight.js/lib/languages/ruby"
import rust from "highlight.js/lib/languages/rust"
import sql from "highlight.js/lib/languages/sql"
import swift from "highlight.js/lib/languages/swift"
import typescript from "highlight.js/lib/languages/typescript"
import xml from "highlight.js/lib/languages/xml"
import yaml from "highlight.js/lib/languages/yaml"

const LANGUAGES = { bash, c, cmake, cpp, csharp, css, diff, dockerfile, go, ini, java, javascript, json, kotlin, lua, makefile, markdown, php, powershell, python, ruby, rust, sql, swift, typescript, xml, yaml }
for (const [name, def] of Object.entries(LANGUAGES)) hljs.registerLanguage(name, def)

// the names a model writes after the fence
const ALIAS: Record<string, string> = {
  py: "python", python3: "python", js: "javascript", jsx: "javascript", mjs: "javascript", node: "javascript", ts: "typescript", tsx: "typescript",
  sh: "bash", zsh: "bash", shell: "bash", console: "bash", ps1: "powershell", pwsh: "powershell", ps: "powershell",
  "c++": "cpp", cc: "cpp", cxx: "cpp", hpp: "cpp", h: "c", cuda: "cpp", cu: "cpp", "c#": "csharp", cs: "csharp", rs: "rust", golang: "go",
  yml: "yaml", html: "xml", svg: "xml", xhtml: "xml", vue: "xml", md: "markdown", patch: "diff", toml: "ini", conf: "ini", cfg: "ini",
  jsonc: "json", json5: "json", docker: "dockerfile", make: "makefile", rb: "ruby", kt: "kotlin",
}

// A long block is left plain so a long answer stays quick; one still being written has to be shorter, since it is coloured again at every
// word until it closes.
const MAX_CLOSED = 40_000
const MAX_OPEN = 8_000
const cache = new Map<string, string>()

/** The name a fence's language word stands for ("py", "python title=a.py", "C++"), or null when it is not one that is known. */
export function languageOf(word: string): string | null {
  const l = word.trim().toLowerCase().split(/[\s{,:]/)[0]
  const name = ALIAS[l] ?? l
  return name && hljs.getLanguage(name) ? name : null
}

/** The coloured HTML of `code` for the fence word `lang`, or null when it should stay plain (unknown language, too long, a failure). */
export function highlight(lang: string, code: string, open: boolean): string | null {
  const name = languageOf(lang)
  if (!name || code.length > (open ? MAX_OPEN : MAX_CLOSED)) return null
  const key = name + "\u0000" + code
  const hit = cache.get(key)
  if (hit !== undefined) return hit
  let html: string
  try { html = hljs.highlight(code, { language: name, ignoreIllegals: true }).value } catch { return null }
  if (cache.size >= 300) cache.delete(cache.keys().next().value!)         // the oldest goes
  cache.set(key, html)
  return html
}
