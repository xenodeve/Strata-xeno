import { describe, expect, test } from "bun:test"
import { holdersOf, keysAnswer, logKind, teamOf, waitingText } from "./keys"

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
