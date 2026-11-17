// Pictures, text files and documents for the chat: the attach button, dropping them on it, or pasting a picture (issue #30).
// A document (PDF, Word, Excel, PowerPoint, OpenDocument, EPUB, RTF) goes as a data: URL; the server reads its text (#334).
import type { Health } from "./api"
import type { Attachment } from "./chat"
import { toast } from "../components/toast"
import { t } from "./i18n"

const TEXT_EXT = /\.(txt|md|markdown|rst|tex|py|pyi|ipynb|js|mjs|cjs|ts|tsx|jsx|vue|svelte|json|jsonl|csv|tsv|log|ya?ml|toml|ini|cfg|conf|env|xml|html?|css|scss|less|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|go|java|kt|kts|swift|rb|php|pl|lua|r|jl|scala|sql|sh|bash|zsh|fish|ps1|psm1|bat|cmd|diff|patch|gradle|cmake|mk|dockerfile|gitignore|proto|graphql)$/i
const MAX_TEXT_FILE = 512 * 1024
const DOC_EXT = /\.(pdf|docx|xlsx|pptx|odt|ods|odp|epub|rtf)$/i
const LEGACY_EXT = /\.(doc|xls|ppt)$/i
export const MAX_DOC_FILE = 30e6                    // serve/documents.py MAX_BYTES

const isTextFile = (f: File) =>
  f.type.startsWith("text/") || /json|xml|javascript|yaml|toml|x-sh|x-python/.test(f.type) ||
  TEXT_EXT.test(f.name) || /(^|[\\/])(makefile|dockerfile|readme|license)$/i.test(f.name)

export type FileKind = "image" | "doc" | "legacy" | "text" | "other"
/** What a file is to the attach button: by its name first (a browser often gives a document no type, and an .rtf or .docx a type that
 *  looks like text or XML), then its type. */
export function fileKind(f: { name: string; type: string }): FileKind {
  if (f.type.startsWith("image/")) return "image"
  if (DOC_EXT.test(f.name) || f.type === "application/pdf") return "doc"
  if (LEGACY_EXT.test(f.name)) return "legacy"
  return isTextFile(f as File) ? "text" : "other"
}

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
    const kind = fileKind(f)
    if (kind === "image") {
      if (!health.images) { toast("warn", t("Pictures are off"), t("This model was set up for text only.")); continue }
      if (f.size > 20e6) { toast("warn", t("Picture too large"), t("{name} is over 20 MB.", { name: f.name })); continue }
      out.push({ kind: "image", name: f.name || t("pasted image"), url: await read(f, "url") })
      continue
    }
    if (kind === "doc") {
      if (f.size > MAX_DOC_FILE) { toast("warn", t("File too large"), t("{name} is over 30 MB.", { name: f.name })); continue }
      out.push({ kind: "doc", name: f.name, url: await read(f, "url") })
      continue
    }
    if (kind === "legacy") { toast("warn", t("Old Office format"), t("{name}: save it as .docx, .xlsx or .pptx and attach it again.", { name: f.name })); continue }
    if (kind === "other") {
      toast("warn", t("This file cannot be attached"), health.images
        ? t("{name}: attach text files (code, notes, logs, data), documents (PDF, Word, Excel, PowerPoint, OpenDocument, EPUB, RTF) or pictures.", { name: f.name })
        : t("{name}: attach text files (code, notes, logs, data) or documents (PDF, Word, Excel, PowerPoint, OpenDocument, EPUB, RTF).", { name: f.name }))
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
