// Many conversations kept in this browser (issue #92). Three kinds of key:
//   chats        the index: which conversations there are, which is open, the projects (folders) that group them
//   chat         the OPEN conversation's messages - the key the app always used and the classic app still reads
//   chat.<id>    every other conversation's messages
// The pure operations are here, over a small `Backing` (the app's `store`; a Map in the tests); the controller and the sidebar sit
// on top. A conversation is added when its first prompt is saved, and one that has been emptied is dropped, so there is never a
// blank item in the list. A write the browser refuses (its storage is full) is returned as `ok: false`, never swallowed.

export interface StoredMessage { role: string; text: string; time?: number; compact?: unknown; files?: { name: string }[]; images?: { name: string }[] }
export interface SessionMeta { id: string; title: string; time: number; project?: string; named?: boolean }
/** folders: where the coding tools of the project's chats work (issue #96). The first is the main one (commands run there, relative paths start there); the others are
 *  the project's other folders, such as other git worktrees of the same repository, where files are as free as in the main one. */
export interface Project { id: string; name: string; folders?: string[] }
export const FOLDERS_MAX = 20
export interface SessionIndex { active: string | null; items: SessionMeta[]; projects: Project[] }
export interface Backing { get<T>(k: string, d: T): T; set(k: string, v: unknown): boolean; remove(k: string): void }

export const ACTIVE_KEY = "chat"
export const INDEX_KEY = "chats"
export const slotKey = (id: string) => `chat.${id}`
const TITLE_MAX = 40
const NAME_MAX = 60

export const newId = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 6)
const EMPTY = (): SessionIndex => ({ active: null, items: [], projects: [] })

/** One line, no runs of spaces, cut at `max` with an ellipsis; null when nothing is left. */
function clean(text: string, max: number): string | null {
  const s = text.replace(/\s+/g, " ").trim()
  if (!s) return null
  return s.length > max ? s.slice(0, max).trimEnd() + "…" : s
}

/** A conversation's title: its first prompt, or the name of the file or picture it was (empty when there is no prompt). */
export function titleOf(messages: StoredMessage[]): string {
  const first = messages.find((m) => m.role === "user" && !m.compact)             // not the summary a compaction left in place of the first prompts
  if (!first) return ""
  return clean(first.text || "", TITLE_MAX) ?? first.files?.[0]?.name ?? first.images?.[0]?.name ?? ""
}

const isObj = (x: unknown): x is Record<string, unknown> => !!x && typeof x === "object" && !Array.isArray(x)
const FOLDER_MAX = 1000
const okFolder = (x: unknown): x is string => typeof x === "string" && x.trim().length > 0 && x.trim().length <= FOLDER_MAX
/** The folders that are text, cleaned, once each, in order, no more than FOLDERS_MAX. */
export function cleanFolders(x: unknown): string[] {
  const out: string[] = []
  for (const f of Array.isArray(x) ? x : []) if (okFolder(f) && !out.includes(f.trim())) out.push(f.trim())
  return out.slice(0, FOLDERS_MAX)
}

function sanitize(raw: unknown): SessionIndex | null {
  if (!isObj(raw) || !Array.isArray(raw.items)) return null
  const items: SessionMeta[] = []
  for (const x of raw.items) {
    if (!isObj(x) || typeof x.id !== "string" || !x.id) continue
    items.push({
      id: x.id, title: typeof x.title === "string" ? x.title : "", time: typeof x.time === "number" ? x.time : 0,
      ...(typeof x.project === "string" ? { project: x.project } : {}), ...(x.named === true ? { named: true } : {}),
    })
  }
  const projects: Project[] = []
  for (const p of Array.isArray(raw.projects) ? raw.projects : []) if (isObj(p) && typeof p.id === "string" && typeof p.name === "string") {
    const folders = cleanFolders(Array.isArray(p.folders) ? p.folders : [p.folder])        // a project made when it had one folder has `folder`
    projects.push({ id: p.id, name: p.name, ...(folders.length ? { folders } : {}) })
  }
  const active = typeof raw.active === "string" && items.some((i) => i.id === raw.active) ? raw.active : null
  return { active, items, projects }
}

/** The index. With none stored, a conversation the app already had (at `chat`) becomes the first one - nothing is lost. */
export function loadIndex(b: Backing, now: number): SessionIndex {
  const stored = sanitize(b.get<unknown>(INDEX_KEY, null))
  if (stored) return stored
  const msgs = b.get<unknown>(ACTIVE_KEY, [])
  if (!Array.isArray(msgs) || !msgs.length) return EMPTY()
  const last = msgs[msgs.length - 1] as StoredMessage
  const id = newId()
  const index: SessionIndex = { active: id, items: [{ id, title: titleOf(msgs as StoredMessage[]), time: typeof last?.time === "number" ? last.time : now }], projects: [] }
  b.set(INDEX_KEY, index)
  return index
}

export const persistIndex = (b: Backing, index: SessionIndex) => b.set(INDEX_KEY, index)

/** Saves the open conversation (`messages` as they are to be stored) and updates the index: the first prompt of a new one adds it (in `project`, when it was started in one), an
 *  emptied one is dropped, otherwise its time moves and its title follows the first prompt unless it was named by hand. */
export function saveActive(b: Backing, index: SessionIndex, messages: StoredMessage[], now: number, project?: string): { index: SessionIndex; ok: boolean } {
  const wrote = b.set(ACTIVE_KEY, messages)
  let next = index
  const open = index.active !== null && index.items.some((i) => i.id === index.active)
  if (!messages.length) {
    if (open) next = { ...index, active: null, items: index.items.filter((i) => i.id !== index.active) }
  } else if (!open) {
    const id = newId()
    const inProject = project && index.projects.some((p) => p.id === project) ? { project } : {}        // a conversation that was started in a project
    next = { ...index, active: id, items: [{ id, title: titleOf(messages), time: now, ...inProject }, ...index.items] }
  } else {
    next = { ...index, items: index.items.map((i) => (i.id === index.active ? { ...i, time: now, title: i.named ? i.title : titleOf(messages) || i.title } : i)) }
  }
  return { index: next, ok: persistIndex(b, next) && wrote }
}

/** Starts an empty conversation; the open one (`current`, its messages) is kept in the list. When it cannot be kept (storage full)
 *  the same index object comes back and nothing changed. */
export function newSession(b: Backing, index: SessionIndex, current: unknown[]): SessionIndex {
  if (index.active !== null && current.length && !b.set(slotKey(index.active), current)) return index
  if (index.active === null) return index
  const next = { ...index, active: null }
  b.set(ACTIVE_KEY, [])
  persistIndex(b, next)
  return next
}

/** Opens conversation `id`: the open one is kept, the other's messages come back. Null when there is no such conversation; `ok: false`
 *  (and nothing changed) when the open one could not be kept. */
export function openSession(b: Backing, index: SessionIndex, current: unknown[], id: string): { index: SessionIndex; messages: unknown[]; ok: boolean } | null {
  if (!index.items.some((i) => i.id === id)) return null
  if (id === index.active) return { index, messages: current, ok: true }
  if (index.active !== null && current.length && !b.set(slotKey(index.active), current)) return { index, messages: current, ok: false }
  const target = b.get<unknown[]>(slotKey(id), [])
  b.set(ACTIVE_KEY, target)
  b.remove(slotKey(id))                                    // the open one lives at `chat` only
  const next = { ...index, active: id }
  persistIndex(b, next)
  return { index: next, messages: target, ok: true }
}

export function removeSession(b: Backing, index: SessionIndex, id: string): { index: SessionIndex; clearedActive: boolean } {
  const clearedActive = index.active === id
  b.remove(slotKey(id))
  if (clearedActive) b.set(ACTIVE_KEY, [])
  const next = { ...index, active: clearedActive ? null : index.active, items: index.items.filter((i) => i.id !== id) }
  persistIndex(b, next)
  return { index: next, clearedActive }
}

// The rest change the index only (the caller saves it): the same object comes back when there is nothing to change.
export function renameSession(index: SessionIndex, id: string, title: string): SessionIndex {
  const t = clean(title, NAME_MAX)
  if (!t || !index.items.some((i) => i.id === id)) return index
  return { ...index, items: index.items.map((i) => (i.id === id ? { ...i, title: t, named: true } : i)) }
}

export function moveSession(index: SessionIndex, id: string, project: string | undefined): SessionIndex {
  if (!index.items.some((i) => i.id === id) || (project !== undefined && !index.projects.some((p) => p.id === project))) return index
  return { ...index, items: index.items.map((i) => { if (i.id !== id) return i; const { project: _was, ...rest } = i; return project === undefined ? rest : { ...rest, project } }) }
}

/** A project is a name and the folders its chats work in (the coding tools). The page asks for at least one; projects made before that have none. */
export function addProject(index: SessionIndex, name: string, id: string = newId(), folders?: string[]): SessionIndex {
  const n = clean(name, NAME_MAX)
  const f = cleanFolders(folders)
  return n ? { ...index, projects: [...index.projects, { id, name: n, ...(f.length ? { folders: f } : {}) }] } : index
}

export function renameProject(index: SessionIndex, id: string, name: string): SessionIndex {
  const n = clean(name, NAME_MAX)
  if (!n || !index.projects.some((p) => p.id === id)) return index
  return { ...index, projects: index.projects.map((p) => (p.id === id ? { ...p, name: n } : p)) }
}

/** The folders a project's chats work in (the coding tools), in order (the first is the main one); none clears them. The server checks that they are folders. */
export function setProjectFolders(index: SessionIndex, id: string, folders: string[]): SessionIndex {
  if (!index.projects.some((p) => p.id === id)) return index
  const f = cleanFolders(folders)
  return { ...index, projects: index.projects.map((p) => { if (p.id !== id) return p; const { folders: _was, ...rest } = p; return f.length ? { ...rest, folders: f } : rest }) }
}

export const foldersOf = (index: SessionIndex, project: string | undefined): string[] => (project ? index.projects.find((p) => p.id === project)?.folders ?? [] : [])

/** Deleting a project does not delete its conversations: they are unfiled. */
export function removeProject(index: SessionIndex, id: string): SessionIndex {
  return { ...index, projects: index.projects.filter((p) => p.id !== id), items: index.items.map((i) => { if (i.project !== id) return i; const { project: _was, ...rest } = i; return rest }) }
}
