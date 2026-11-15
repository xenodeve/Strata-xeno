import { describe, expect, test } from "bun:test"
import { browserKeyAfterCreate, dismissFreshKey, getFreshKey, holdersOf, keysAnswer, logKind, showFreshKey, subscribeFreshKey, teamOf, waitingText } from "./keys"

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
