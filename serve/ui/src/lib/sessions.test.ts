import { describe, expect, test } from "bun:test"
import {
  addProject, loadIndex, moveSession, newSession, openSession, removeProject, removeSession, renameProject, renameSession, saveActive, titleOf,
  type Backing, type SessionIndex,
} from "./sessions"

// Many conversations kept in the browser: an index (`chats`), the open conversation at `chat` (the key the classic app reads) and
// the others at `chat.<id>`. These are the pure operations; the controller and the sidebar sit on top of them.
function memory(limit = Infinity) {
  const data = new Map<string, string>()
  const b: Backing & { data: Map<string, string> } = {
    data,
    get: <T>(k: string, d: T) => (data.has(k) ? (JSON.parse(data.get(k)!) as T) : d),
    set: (k, v) => {
      const text = JSON.stringify(v)
      if (text.length > limit) return false
      data.set(k, text)
      return true
    },
    remove: (k) => { data.delete(k) },
  }
  return b
}
const u = (text: string, extra: object = {}) => ({ role: "user", text, time: 1, ...extra })
const a = (text: string) => ({ role: "assistant", text, time: 2 })

describe("title", () => {
  test("the first prompt, on one line, cut to about 40 characters", () => {
    expect(titleOf([u("  What is\n a   mixture of experts? ")])).toBe("What is a mixture of experts?")
    const long = titleOf([u("x".repeat(100))])
    expect(long.length).toBeLessThanOrEqual(41)
    expect(long.endsWith("…")).toBe(true)
  })
  test("a prompt of only a file or a picture is named after it, and no prompt gives an empty title", () => {
    expect(titleOf([u("", { files: [{ name: "notes.md" }] })])).toBe("notes.md")
    expect(titleOf([u("", { images: [{ name: "cat.png" }] })])).toBe("cat.png")
    expect(titleOf([])).toBe("")
  })
})

describe("loading", () => {
  test("nothing stored: no conversations, no open one", () => {
    const b = memory()
    expect(loadIndex(b, 5)).toEqual({ active: null, items: [], projects: [] })
  })
  test("the conversation the app had before becomes the first one, and keeps its key", () => {
    const b = memory()
    b.set("chat", [u("old question"), a("old answer")])
    const idx = loadIndex(b, 99)
    expect(idx.items).toHaveLength(1)
    expect(idx.items[0].title).toBe("old question")
    expect(idx.active).toBe(idx.items[0].id)
    expect(b.get("chat", [])).toHaveLength(2)                 // still where the classic app reads it
    expect(b.get<SessionIndex>("chats", null as never).items).toHaveLength(1)     // and the migration is saved
  })
  test("loaded again, it is the same, not migrated twice", () => {
    const b = memory()
    b.set("chat", [u("q")])
    const first = loadIndex(b, 1)
    expect(loadIndex(b, 2)).toEqual(first)
  })
  test("a damaged index is read as far as it can be", () => {
    const b = memory()
    b.set("chats", { active: "gone", items: [{ id: "a", title: "ok", time: 1 }, { nope: 1 }, "x"], projects: [{ id: "p", name: "P" }, 3] })
    const idx = loadIndex(b, 1)
    expect(idx.items.map((i) => i.id)).toEqual(["a"])
    expect(idx.projects).toEqual([{ id: "p", name: "P" }])
    expect(idx.active).toBeNull()
    b.set("chats", "garbage")
    expect(loadIndex(b, 1)).toEqual({ active: null, items: [], projects: [] })
  })
})

describe("saving the open conversation", () => {
  test("the first prompt makes a conversation, titled from it, and it is the open one", () => {
    const b = memory()
    const msgs = [u("hello there"), a("hi")]
    const { index, ok } = saveActive(b, loadIndex(b, 1), msgs, 10)
    expect(ok).toBe(true)
    expect(index.items).toHaveLength(1)
    expect(index.items[0]).toMatchObject({ title: "hello there", time: 10 })
    expect(index.active).toBe(index.items[0].id)
    expect(b.get("chat", [])).toEqual(msgs)
  })
  test("later saves keep the conversation and move its time; a title chosen by hand stays", () => {
    const b = memory()
    let idx = saveActive(b, loadIndex(b, 1), [u("first")], 10).index
    idx = renameSession(idx, idx.active!, "My title")
    idx = saveActive(b, idx, [u("changed first"), a("x")], 20).index
    expect(idx.items).toHaveLength(1)
    expect(idx.items[0]).toMatchObject({ title: "My title", time: 20 })
  })
  test("an unnamed title follows the first prompt when it is rewritten", () => {
    const b = memory()
    let idx = saveActive(b, loadIndex(b, 1), [u("first")], 10).index
    idx = saveActive(b, idx, [u("rewritten")], 20).index
    expect(idx.items[0].title).toBe("rewritten")
  })
  test("a conversation that has been emptied (its only prompt taken back) is gone, not left blank", () => {
    const b = memory()
    let idx = saveActive(b, loadIndex(b, 1), [u("only")], 10).index
    idx = saveActive(b, idx, [], 20).index
    expect(idx.items).toEqual([])
    expect(idx.active).toBeNull()
  })
  test("storage that is full is reported, and the index still says what is open", () => {
    const b = memory(200)
    const { ok } = saveActive(b, loadIndex(b, 1), [u("x".repeat(500))], 10)
    expect(ok).toBe(false)
  })
})

describe("switching", () => {
  function two() {
    const b = memory()
    let idx = saveActive(b, loadIndex(b, 1), [u("one"), a("1")], 10).index
    const first = idx.active!
    idx = newSession(b, idx, b.get("chat", []))
    idx = saveActive(b, idx, [u("two"), a("2")], 20).index
    return { b, idx, first, second: idx.active! }
  }
  test("a new conversation leaves the old one in the list, and the open one is empty until its first prompt", () => {
    const b = memory()
    let idx = saveActive(b, loadIndex(b, 1), [u("one")], 10).index
    const first = idx.active!
    idx = newSession(b, idx, b.get("chat", []))
    expect(idx.active).toBeNull()
    expect(idx.items.map((i) => i.id)).toEqual([first])
    expect(b.get("chat", ["x"])).toEqual([])
    expect(b.get(`chat.${first}`, [])).toHaveLength(1)
  })
  test("opening one restores its messages and the one that was open is kept", () => {
    const { b, idx, first, second } = two()
    const r = openSession(b, idx, b.get("chat", []), first)!
    expect(r.index.active).toBe(first)
    expect((r.messages as { text: string }[])[0].text).toBe("one")
    expect(b.get("chat", [])).toEqual(r.messages)
    const back = openSession(b, r.index, b.get("chat", []), second)!
    expect((back.messages as { text: string }[])[0].text).toBe("two")
    expect(b.get(`chat.${second}`, null)).toBeNull()          // the open one lives at `chat` only
  })
  test("an id that does not exist opens nothing", () => {
    const { b, idx } = two()
    expect(openSession(b, idx, b.get("chat", []), "nope")).toBeNull()
  })
})

describe("renaming, moving, deleting", () => {
  test("rename (an empty name is refused), move, and delete one conversation", () => {
    const b = memory()
    let idx = saveActive(b, loadIndex(b, 1), [u("one")], 10).index
    const id = idx.active!
    expect(renameSession(idx, id, "   ")).toBe(idx)
    idx = renameSession(idx, id, "  Kept  ")
    expect(idx.items[0]).toMatchObject({ title: "Kept", named: true })
    idx = addProject(idx, "Work", "p1")
    idx = moveSession(idx, id, "p1")
    expect(idx.items[0].project).toBe("p1")
    idx = moveSession(idx, id, undefined)
    expect(idx.items[0].project).toBeUndefined()
    const gone = removeSession(b, idx, id)
    expect(gone.index.items).toEqual([])
    expect(gone.index.active).toBeNull()
    expect(b.get("chat", ["x"])).toEqual([])
  })
  test("deleting a conversation that is not open removes its stored messages", () => {
    const b = memory()
    let idx = saveActive(b, loadIndex(b, 1), [u("one")], 10).index
    const first = idx.active!
    idx = newSession(b, idx, b.get("chat", []))
    idx = saveActive(b, idx, [u("two")], 20).index
    const gone = removeSession(b, idx, first)
    expect(gone.index.items).toHaveLength(1)
    expect(gone.index.active).toBe(idx.active)
    expect(b.get(`chat.${first}`, null)).toBeNull()
  })
  test("moving to a project that does not exist is refused", () => {
    const b = memory()
    const idx = saveActive(b, loadIndex(b, 1), [u("one")], 10).index
    expect(moveSession(idx, idx.active!, "nope")).toBe(idx)
  })
})

describe("projects", () => {
  test("create (names are cleaned, an empty one is refused), rename", () => {
    let idx: SessionIndex = { active: null, items: [], projects: [] }
    expect(addProject(idx, "   ", "x")).toBe(idx)
    idx = addProject(idx, "  My   work ", "p1")
    expect(idx.projects).toEqual([{ id: "p1", name: "My work" }])
    idx = renameProject(idx, "p1", "Home")
    expect(idx.projects[0].name).toBe("Home")
    expect(renameProject(idx, "p1", "")).toBe(idx)
  })
  test("deleting a project keeps its conversations, which are unfiled", () => {
    const b = memory()
    let idx = saveActive(b, loadIndex(b, 1), [u("one")], 10).index
    idx = addProject(idx, "Work", "p1")
    idx = moveSession(idx, idx.active!, "p1")
    idx = removeProject(idx, "p1")
    expect(idx.projects).toEqual([])
    expect(idx.items).toHaveLength(1)
    expect(idx.items[0].project).toBeUndefined()
  })
})
