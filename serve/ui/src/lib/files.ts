// Pictures and text files for the chat: the attach button, dropping them on it, or pasting a picture (issue #30).
import type { Health } from "./api"
import type { Attachment } from "./chat"
import { toast } from "../components/toast"
import { t } from "./i18n"

const TEXT_EXT = /\.(txt|md|markdown|rst|tex|py|pyi|ipynb|js|mjs|cjs|ts|tsx|jsx|vue|svelte|json|jsonl|csv|tsv|log|ya?ml|toml|ini|cfg|conf|env|xml|html?|css|scss|less|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|go|java|kt|kts|swift|rb|php|pl|lua|r|jl|scala|sql|sh|bash|zsh|fish|ps1|psm1|bat|cmd|diff|patch|gradle|cmake|mk|dockerfile|gitignore|proto|graphql)$/i
const MAX_TEXT_FILE = 512 * 1024

const isTextFile = (f: File) =>
  f.type.startsWith("text/") || /json|xml|javascript|yaml|toml|x-sh|x-python/.test(f.type) ||
  TEXT_EXT.test(f.name) || /(^|[\\/])(makefile|dockerfile|readme|license)$/i.test(f.name)

const read = (f: File, as: "text" | "url") =>
  new Promise<string>((resolve, reject) => {
    const r = new FileReader()
    r.onload = () => resolve(String(r.result))
    r.onerror = () => reject(r.error)
    as === "text" ? r.readAsText(f) : r.readAsDataURL(f)
  })

/** The attachments a set of files makes; each refused file says why, in a toast. */
export async function readFiles(files: Iterable<File>, health: Health): Promise<Attachment[]> {
  const out: Attachment[] = []
  for (const f of files) {
    if (f.type.startsWith("image/")) {
      if (!health.images) { toast("warn", t("Pictures are off"), t("This model was set up for text only.")); continue }
      if (f.size > 20e6) { toast("warn", t("Picture too large"), t("{name} is over 20 MB.", { name: f.name })); continue }
      out.push({ kind: "image", name: f.name || t("pasted image"), url: await read(f, "url") })
      continue
    }
    if (!isTextFile(f)) {
      toast("warn", t("Not a text file"), health.images ? t("{name}: attach text files (code, notes, logs, data) or pictures.", { name: f.name }) : t("{name}: attach text files (code, notes, logs, data).", { name: f.name }))
      continue
    }
    if (f.size > MAX_TEXT_FILE) { toast("warn", t("File too large"), t("{name} is over 512 KB.", { name: f.name })); continue }
    const text = await read(f, "text")
    if (text.includes("\u0000")) { toast("warn", t("Not a text file"), t("{name} looks like a binary file.", { name: f.name })); continue }
    out.push({ kind: "file", name: f.name, text })
  }
  return out
}

export async function copyText(text: string) {
  try {
    await navigator.clipboard.writeText(text)
  } catch {                                       // http on another host: no async clipboard
    const ta = document.createElement("textarea")
    ta.value = text
    document.body.appendChild(ta)
    ta.select()
    document.execCommand("copy")
    ta.remove()
  }
  toast("success", t("Copied to clipboard"), "", 1800)
}
