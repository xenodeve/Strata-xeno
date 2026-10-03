// The folders of a path that is being typed (issue #96): from "C:/work/app-w" the folders in C:/work whose names go with "app-w", so a rough path is enough and
// one click finishes it. Pure text work; the folders themselves come from the server (GET /agent/folders).

export interface Typed { base: string; partial: string }       // base: what is typed up to and with the last separator, which is also what to ask the server to list

/** Splits what is typed at its last separator: the folder to list, and the part of a name typed after it. Null when there is nothing to complete (no separator yet). */
export function splitTyped(text: string): Typed | null {
  const s = text.replace(/^\s+/, "")
  if (!s.trim()) return null
  if (/^[A-Za-z]:$/.test(s)) return { base: s + "\\", partial: "" }               // "C:" is the drive's top folder
  if (s === "~") return { base: "~/", partial: "" }
  const i = Math.max(s.lastIndexOf("/"), s.lastIndexOf("\\"))
  return i < 0 ? null : { base: s.slice(0, i + 1), partial: s.slice(i + 1) }
}

/** The separator to finish a chosen folder with: the kind the path was typed with. */
export const sepOf = (base: string): string => (base.includes("\\") ? "\\" : "/")

/** The names that go with what was typed: those that start with it first, then those that contain it (a rough name is enough); any case. */
export function rank(names: string[], partial: string): string[] {
  const p = partial.trim().toLowerCase()
  if (!p) return names
  const starts = names.filter((n) => n.toLowerCase().startsWith(p))
  const has = names.filter((n) => !n.toLowerCase().startsWith(p) && n.toLowerCase().includes(p))
  return [...starts, ...has]
}

/** A name cut into the part that matches what was typed and the rest, for showing the match in bold. */
export function matchParts(name: string, partial: string): [string, string, string] {
  const p = partial.trim().toLowerCase()
  const at = p ? name.toLowerCase().indexOf(p) : -1
  return at < 0 ? [name, "", ""] : [name.slice(0, at), name.slice(at, at + p.length), name.slice(at + p.length)]
}
