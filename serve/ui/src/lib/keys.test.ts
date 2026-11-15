import { describe, expect, test } from "bun:test"
import { browserKeyAfterCreate, createKey, dismissFreshKey, getFreshKey, holdersOf, keysAnswer, logKind, showFreshKey, subscribeFreshKey, teamOf, waitingText, type KeysAction } from "./keys"

// The page's reading of the #310 routes: GET /keys (who is asking and whether the server is in team mode) and /metrics live (who holds
// the engine and who waits).
describe("keysAnswer / teamOf", () => {
  test("the owner's view, in team mode or not", () => {
    const a = keysAnswer(200, { keys: [{ name: "boss", role: "owner", created: 1, devices: [] }], log: [], retired: ["old"], team: true, keys_file: "strata-keys.json" })
    expect(a.kind).toBe("view")
    expect(teamOf(a)).toBe(true)
    const off = keysAnswer(200, { keys: [], team: false })
    expect(off).toEqual({ kind: "view", view: { keys: [], log: [], retired: [], team: false, keys_file: null } })
    expect(teamOf(off)).toBe(false)
  })
  test("403 forbidden is a guest of a team-mode server; a 403 with only a reason is someone who is not the owner of a server outside it", () => {
    expect(keysAnswer(403, { error: { type: "forbidden", message: "keys can be managed only with an owner key" } })).toEqual({ kind: "guest" })
    expect(teamOf({ kind: "guest" })).toBe(true)
    const not = keysAnswer(403, { error: { message: "only from this PC" } })
    expect(not).toEqual({ kind: "not-owner", message: "only from this PC" })
    expect(teamOf(not)).toBe(false)
  })
  test("no route, or an error: team mode is not known", () => {
    expect(keysAnswer(404, null)).toEqual({ kind: "none" })
    expect(teamOf(keysAnswer(404, null))).toBeNull()
    expect(keysAnswer(500, { error: { message: "broken" } })).toEqual({ kind: "error", message: "broken" })
    expect(keysAnswer(502, null)).toEqual({ kind: "error", message: "HTTP 502" })
    expect(teamOf(keysAnswer(500, null))).toBeNull()
  })
})

describe("who holds the engine and who waits", () => {
  test("the holder, alone", () => {
    expect(holdersOf({ holder: "ann", waiting_who: [] })).toEqual(["ann"])
    expect(holdersOf({ holder: null })).toEqual([])
  })
  test("with parallel: each busy slot's name, once", () => {
    expect(holdersOf({ holder: null, slots: [{ state: "decoding", who: "ann" }, { state: "idle", who: null }, { state: "prefill", who: "bob" }, { state: "decoding", who: "ann" }] })).toEqual(["ann", "bob"])
  })
  test("the waiting, in arrival order with their places, from waiting_who only", () => {
    expect(waitingText({ waiting_who: [{ who: "bob", since: 2 }, { who: "cat", since: 3 }] })).toBe("bob (1), cat (2)")
    expect(waitingText({ waiting_who: [] })).toBe("")
    expect(waitingText({})).toBe("")
  })
})

describe("logKind", () => {
  test("the server's words for each change", () => {
    expect(logKind("created")).toEqual({ kind: "created", arg: "" })
    expect(logKind("renamed from ann")).toEqual({ kind: "renamed", arg: "ann" })
    expect(logKind("role:guest")).toEqual({ kind: "role", arg: "guest" })
    expect(logKind("approved")).toEqual({ kind: "approved", arg: "" })
  })
})

// I1 (task 6 review): a new key's text lives in a store of its own, above the waiting screen, until it is dismissed; and the first key
// replaces a key this browser had stored when the server has no API key (that stored key was never checked, and is not valid after).
describe("the new key, shown once", () => {
  test("kept until dismissed, and the listeners hear both", () => {
    let heard = 0
    const off = subscribeFreshKey(() => { heard++ })
    showFreshKey({ secret: "sk-strata-x", name: "boss", browser: "stored" })
    expect(getFreshKey()).toEqual({ secret: "sk-strata-x", name: "boss", browser: "stored" })
    expect(getFreshKey()).toEqual({ secret: "sk-strata-x", name: "boss", browser: "stored" })     // reading it does not drop it
    dismissFreshKey()
    expect(getFreshKey()).toBeNull()
    expect(heard).toBe(2)
    off()
  })
})

describe("browserKeyAfterCreate: does this browser take the new key", () => {
  test("a key that is not the first: the browser keeps its own", () => {
    expect(browserKeyAfterCreate({ first: false, stored: "", serverHasKey: false })).toBe("keep")
    expect(browserKeyAfterCreate({ first: false, stored: "k", serverHasKey: true })).toBe("keep")
  })
  test("the first key, and the browser has none: it takes it", () => {
    expect(browserKeyAfterCreate({ first: true, stored: "", serverHasKey: false })).toBe("store")
    expect(browserKeyAfterCreate({ first: true, stored: "", serverHasKey: true })).toBe("store")
    expect(browserKeyAfterCreate({ first: true, stored: "  ", serverHasKey: null })).toBe("store")
  })
  test("the first key, and the browser has a stale key the server never asked for: it is replaced", () => {
    expect(browserKeyAfterCreate({ first: true, stored: "old", serverHasKey: false })).toBe("store")
  })
  test("the first key, and the browser has the server's API key (still valid), or the server could not be asked: it keeps it", () => {
    expect(browserKeyAfterCreate({ first: true, stored: "k", serverHasKey: true })).toBe("keep")
    expect(browserKeyAfterCreate({ first: true, stored: "k", serverHasKey: null })).toBe("keep")
  })
})

// Re-review of I1: /health "api_key" is the server's key_required(), true as soon as the keys file exists - so it must be read BEFORE the
// create. These run createKey against a fake server whose /health answers api_key false before the create and true after it.
describe("createKey: the server's state before the create decides", () => {
  const fakeServer = (o: { before: boolean | null; stored: string; keeps?: boolean }) => {
    const calls: string[] = []
    let created = false
    let kept = o.stored
    return {
      calls, kept: () => kept,
      io: {
        serverHasKey: async () => { calls.push("health"); return created ? true : o.before },
        post: async (b: KeysAction) => { calls.push("post:" + b.action); created = true; return { ok: true as const, key: "sk-strata-new" } },
        stored: () => kept,
        keep: (k: string) => { calls.push("keep"); if (o.keeps === false) return false; kept = k; return true },
      },
    }
  }
  test("a stale key on a server with no API key is replaced: /health is read before the create", async () => {
    const s = fakeServer({ before: false, stored: "stale" })
    const r = await createKey({ name: "boss", role: "owner" }, true, s.io)
    expect(s.calls).toEqual(["health", "post:create", "keep"])
    expect(r).toEqual({ fresh: { secret: "sk-strata-new", name: "boss", browser: "stored" } })
    expect(s.kept()).toBe("sk-strata-new")
  })
  test("the server's own API key is kept", async () => {
    const s = fakeServer({ before: true, stored: "k" })
    expect(await createKey({ name: "boss", role: "owner" }, true, s.io)).toEqual({ fresh: { secret: "sk-strata-new", name: "boss", browser: "unchanged" } })
    expect(s.kept()).toBe("k")
  })
  test("/health could not be read and the browser has a key: kept, and the card is told it may be wrong", async () => {
    const s = fakeServer({ before: null, stored: "maybe" })
    expect(await createKey({ name: "boss", role: "owner" }, true, s.io)).toEqual({ fresh: { secret: "sk-strata-new", name: "boss", browser: "uncertain" } })
    expect(s.kept()).toBe("maybe")
  })
  test("the browser refuses to keep it: failed", async () => {
    const s = fakeServer({ before: false, stored: "", keeps: false })
    expect(await createKey({ name: "boss", role: "owner" }, true, s.io)).toEqual({ fresh: { secret: "sk-strata-new", name: "boss", browser: "failed" } })
  })
  test("a later key: /health is not asked and the browser keeps its key", async () => {
    const s = fakeServer({ before: true, stored: "mine" })
    expect(await createKey({ name: "ann", role: "guest" }, false, s.io)).toEqual({ fresh: { secret: "sk-strata-new", name: "ann", browser: "unchanged" } })
    expect(s.calls).toEqual(["post:create"])
  })
  test("a refused create: its reason, nothing kept", async () => {
    const io = { serverHasKey: async () => false, post: async () => ({ error: "was used before" }), stored: () => "", keep: () => true }
    expect(await createKey({ name: "x", role: "owner" }, true, io)).toEqual({ error: "was used before" })
  })
})
