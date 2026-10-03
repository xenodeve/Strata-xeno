// `@file` in the prompt (issue #99): typing `@` and a few letters offers the files of the project's folders; the file that is picked is written as `@path`, and when the prompt is sent
// the text of each file that is mentioned goes with it, as an attachment (cut to a size; only inside the project's folders). The server does the looking (serve/files.py).
import { apiHeaders, url } from "./api"
import type { Attachment } from "./chat"

export interface AtQuery { start: number; query: string }          // start: where the `@` is

/** The `@name` the caret is in: an `@` at the start of the text or after a space, then what has been typed of the name (no space in it). */
export function atQuery(text: string, caret: number): AtQuery | null {
  const m = /(?:^|[\s(\[{"'])@([^\s@]*)$/.exec(text.slice(0, caret))
  return m ? { start: caret - m[1].length - 1, query: m[1] } : null
}

/** The text after a file is picked: `@path` and a space where the name was typed (what was left of that word after the caret goes too); the caret goes after the space. */
export function pickMention(text: string, at: AtQuery, caret: number, path: string): { text: string; caret: number } {
  const tail = text.slice(caret).replace(/^\S*/, "")
  const head = text.slice(0, at.start) + "@" + path + " "
  return { text: head + tail.replace(/^ /, ""), caret: head.length }
}

const MAX_MENTIONS = 8

/** The files a prompt mentions: each `@name` at the start of a word, once, without the punctuation that ends a sentence. A name with `..` or a leading slash is not one. */
export function mentionsOf(text: string): string[] {
  const out: string[] = []
  for (const m of text.matchAll(/(?:^|[\s(\[{"'])@([^\s@]+)/g)) {
    const name = m[1].replace(/[.,;:!?)\]}'"]+$/, "")
    if (!name || name.length > 300 || name.startsWith("/") || name.startsWith("\\") || /(^|[\\/])\.\.([\\/]|$)/.test(name) || /^[A-Za-z]:/.test(name) || out.includes(name)) continue
    out.push(name)
    if (out.length >= MAX_MENTIONS) break
  }
  return out
}

const query = (folders: string[]) => `path=${encodeURIComponent(folders[0] ?? "")}${folders.slice(1).map((f) => "&dirs=" + encodeURIComponent(f)).join("")}`

export interface FoundFile { path: string; folder: number }

/** The files that go with `q`; null when the server cannot be asked. */
export async function getFiles(folders: string[], q: string): Promise<FoundFile[] | null> {
  if (!folders.length) return []
  try {
    const r = await fetch(url(`agent/files?${query(folders)}&q=${encodeURIComponent(q)}`), { headers: apiHeaders() })
    return r.ok ? (((await r.json()) as { files?: FoundFile[] }).files ?? []) : null
  } catch { return null }
}

export async function getMention(folders: string[], rel: string): Promise<{ name: string; text: string; cut: boolean } | null> {
  try {
    const r = await fetch(url(`agent/mention?${query(folders)}&rel=${encodeURIComponent(rel)}`), { headers: apiHeaders() })
    if (!r.ok) return null
    const j = await r.json()
    return j?.ok === true ? { name: String(j.name), text: String(j.text), cut: !!j.cut } : null
  } catch { return null }
}

/** The files a prompt mentions, as attachments (the ones that are there; a name that is no file of the project is left as the words it is). */
export async function resolveMentions(folders: string[], text: string): Promise<Attachment[]> {
  if (!folders.length) return []
  const names = mentionsOf(text)
  if (!names.length) return []
  const got = await Promise.all(names.map((n) => getMention(folders, n)))
  return got.flatMap((g): Attachment[] => (g ? [{ kind: "file", name: g.name, text: g.cut ? g.text + "\n[cut here: the file is longer than is sent]" : g.text }] : []))
}
