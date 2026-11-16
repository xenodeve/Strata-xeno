import { beforeEach, describe, expect, test } from "bun:test"

// This browser's runner (#310 part 2): the id of the runner on the engineer's own PC, kept as `strata.runner`; which machine a chat's coding tools
// work on; the `runner=` a page route carries; which machine ran a call; why a card was closed by the server.
const data = new Map<string, string>()
;(globalThis as Record<string, unknown>).localStorage = {
  getItem: (k: string) => (data.has(k) ? data.get(k)! : null),
  setItem: (k: string, v: string) => { data.set(k, String(v)) },
  removeItem: (k: string) => { data.delete(k) },
}

const { runnerId, setRunnerId, withRunner, machineOf, isCodingCall, chatMachine, closeReason, isSkillsCall, myPcCoding, blockText } = await import("./runner")
const ID = "r0123456789abcdef"

beforeEach(() => { data.clear() })

describe("runnerId", () => {
  test("round trip, kept JSON-encoded as strata.runner like the other strata.* values", () => {
    expect(runnerId()).toBe("")
    expect(setRunnerId(ID)).toBe(true)
    expect(runnerId()).toBe(ID)
    expect(data.get("strata.runner")).toBe(JSON.stringify(ID))
  })
  test("a value that is not a runner id is refused and not kept", () => {
    expect(setRunnerId("x")).toBe(false)
    expect(runnerId()).toBe("")
    data.set("strata.runner", JSON.stringify("x"))
    expect(runnerId()).toBe("")
    for (const bad of ["R0123456789abcdef", "r0123456789abcde", "r0123456789abcdefa", "r0123456789abcdeg", "r01234567 89abcdef"]) expect(setRunnerId(bad)).toBe(false)
  })
  test("a bare id (not JSON) set by another program is read too; blank clears", () => {
    data.set("strata.runner", ID)
    expect(runnerId()).toBe(ID)
    expect(setRunnerId("")).toBe(true)
    expect(runnerId()).toBe("")
    expect(data.has("strata.runner")).toBe(false)
  })
  test("pasted with spaces around it", () => {
    expect(setRunnerId(`  ${ID}\n`)).toBe(true)
    expect(runnerId()).toBe(ID)
  })
})

describe("withRunner", () => {
  test("adds runner= to a query, or starts one", () => {
    expect(withRunner("agent/git?path=a", ID)).toBe(`agent/git?path=a&runner=${ID}`)
    expect(withRunner("agent/git?path=a", ID).endsWith(`&runner=${ID}`)).toBe(true)
    expect(withRunner("mcp", ID)).toBe(`mcp?runner=${ID}`)
  })
  test("no runner: unchanged", () => {
    expect(withRunner("agent/git?path=a")).toBe("agent/git?path=a")
    expect(withRunner("agent/git?path=a", "")).toBe("agent/git?path=a")
  })
})

describe("machineOf", () => {
  test("machine my is My PC; anything else the Server PC", () => {
    expect(machineOf({ server: "my_agent", machine: "my" })).toBe("my")
    expect(machineOf({ server: "agent" })).toBe("server")
    expect(machineOf({ server: "my_fs" })).toBe("server")               // a server MCP may be called my_fs: the event's machine decides
    expect(machineOf({})).toBe("server")
  })
})

describe("isCodingCall / isSkillsCall: by the event's machine, not the name alone", () => {
  test("the server's coding group, and the runner's only on a call that ran on My PC (also renamed on a clash)", () => {
    expect(isCodingCall({ server: "agent" })).toBe(true)
    expect(isCodingCall({ server: "my_agent", machine: "my" })).toBe(true)
    expect(isCodingCall({ server: "my_agent_2", machine: "my" })).toBe(true)
    expect(isCodingCall({ server: "my_agentx", machine: "my" })).toBe(false)
    expect(isCodingCall({ server: "fs" })).toBe(false)
    expect(isCodingCall({})).toBe(false)
  })
  test("a server MCP literally named my_agent / my_skills (no runner, no machine) stays an MCP call", () => {
    expect(isCodingCall({ server: "my_agent" })).toBe(false)
    expect(isSkillsCall({ server: "my_skills" })).toBe(false)
    expect(isCodingCall({ server: "agent", machine: "my" })).toBe(false)         // a runner's call is never the server's group
  })
  test("skills of either machine", () => {
    expect(isSkillsCall({ server: "skills" })).toBe(true)
    expect(isSkillsCall({ server: "my_skills", machine: "my" })).toBe(true)
    expect(isSkillsCall({ server: "my_skills_2", machine: "my" })).toBe(true)
    expect(isSkillsCall({ server: "fs" })).toBe(false)
  })
})

describe("myPcCoding: a My PC chat never sends a prompt that silently goes without tools", () => {
  const ONLINE = { available: true, allowed: true, shell: "bash", tools: ["Read"], machine: "ann-pc", online: true }
  test("no runner id, or a runner the server does not know for this key (404): blocked, with the reason", () => {
    expect(myPcCoding("", undefined)).toMatchObject({ block: "no-runner", info: { allowed: false } })
    expect(myPcCoding(ID, "unknown")).toMatchObject({ block: "unknown", info: { allowed: false } })
  })
  test("known (online or offline), not answered yet, or one failed ask: sent with the runner, so the server says what is wrong", () => {
    expect(myPcCoding(ID, ONLINE)).toEqual({ info: ONLINE, block: null })
    expect(myPcCoding(ID, { ...ONLINE, online: false, allowed: false })).toMatchObject({ info: { allowed: true, online: false }, block: null })
    expect(myPcCoding(ID, undefined)).toMatchObject({ info: { available: true, allowed: true }, block: null })
    expect(myPcCoding(ID, null)).toMatchObject({ info: { available: true, allowed: true }, block: null })
  })
  test("the words: what is wrong and what to do, long and short", () => {
    expect(blockText("no-runner", "http://26.1.2.3:8091").full).toContain("Settings")
    const u = blockText("unknown", "http://26.1.2.3:8091")
    expect(u.full).toContain("python -m serve.runner --server http://26.1.2.3:8091 --key YOUR-KEY")
    expect(u.short).toBe("My PC: the runner is not connected")
    expect(blockText("no-runner", "x").short).toBe("My PC: this browser has no runner")
  })
})

describe("chatMachine: where a chat's coding tools work", () => {
  test("no runner id in this browser: the Server PC, as today, unless the project is on My PC", () => {
    expect(chatMachine({ project: undefined, serverAllowed: true, runner: "" })).toBe("server")
    expect(chatMachine({ project: undefined, serverAllowed: false, runner: "" })).toBe("server")
    expect(chatMachine({ project: "server", serverAllowed: true, runner: "" })).toBe("server")
    expect(chatMachine({ project: "my", serverAllowed: true, runner: "" })).toBe("my")          // never quietly the server's tools for a My PC project
  })
  test("a project keeps its machine", () => {
    expect(chatMachine({ project: "my", serverAllowed: true, runner: ID })).toBe("my")
    expect(chatMachine({ project: "server", serverAllowed: true, runner: ID })).toBe("server")
    expect(chatMachine({ project: "server", serverAllowed: false, runner: ID })).toBe("server")
  })
  test("a chat in no project: the Server PC for an owner, else My PC", () => {
    expect(chatMachine({ project: undefined, serverAllowed: true, runner: ID })).toBe("server")
    expect(chatMachine({ project: undefined, serverAllowed: false, runner: ID })).toBe("my")
  })
  test("not known yet (GET /agent unanswered or failed): the Server PC, never the other machine", () => {
    expect(chatMachine({ project: undefined, serverAllowed: undefined, runner: ID })).toBe("server")
    expect(chatMachine({ project: "my", serverAllowed: undefined, runner: ID })).toBe("my")
  })
})

describe("closeReason: a card the server closed, in words", () => {
  // The words the server really sends as `reason` (serve/runner_hub.py `_card`: abort.why = r.ended or "ended", "offline", "cancelled";
  // Runner.ended from Runner.end(why): "replaced" :697 :724, "revoked" :738, "device" :742, "closed" :694 :757). The server's own stream test
  // asserts `reason == "replaced"` (serve/test_runner_http.py :613, a second runner process on the same PC).
  test("each word the server sends", () => {
    expect(closeReason("replaced")).toBe("Your PC's runner restarted, so this was closed.")
    expect(closeReason("revoked")).toBe("The key of this chat was revoked, so this was closed.")
    expect(closeReason("device")).toBe("The owner refused or cleared your PC as a device, so this was closed.")
    expect(closeReason("closed")).toBe("The server is stopping, so this was closed.")
    expect(closeReason("ended")).toBe("Your PC's runner stopped, so this was closed.")
    expect(closeReason("offline")).toBe("Your PC went offline, so this was closed.")
  })
  test("cancelled is the answer that stopped; an unknown word gets a sentence, never the bare word; none is null", () => {
    expect(closeReason("cancelled")).toBe("The answer stopped before you answered.")
    expect(closeReason("something new")).toBe("The server closed this before you answered.")
    expect(closeReason(null)).toBe(null)
    expect(closeReason(undefined)).toBe(null)
    expect(closeReason("")).toBe(null)
  })
})

describe("the page's fetchers name the runner of a My PC project", () => {
  test("each GET carries runner= when given, and is today's URL without it", async () => {
    ;(globalThis as Record<string, unknown>).location ??= { pathname: "/next/", search: "" }
    const urls: string[] = []
    const real = globalThis.fetch
    globalThis.fetch = (async (u: string) => { urls.push(String(u)); return new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } }) }) as typeof fetch
    try {
      const { getAgent, getMcp, getFolders } = await import("./api")
      const { getGit, getGitDiff } = await import("./git")
      const { getMemory, getMemoryFile } = await import("./memory")
      const { getFiles, getMention } = await import("./mention")
      await getAgent(ID); await getMcp(ID); await getFolders("C:/w", ID)
      await getGit("C:/w", ID); await getGitDiff("C:/w", "a.txt", false, false, ID)
      await getMemory(["C:/w"], ID); await getMemoryFile(["C:/w"], "project:0", 0, ID)
      await getFiles(["C:/w"], "a", ID); await getMention(["C:/w"], "a.txt", ID)
      expect(urls.every((u) => u.endsWith(`&runner=${ID}`) || u.endsWith(`?runner=${ID}`))).toBe(true)
      expect(urls.length).toBe(9)
      urls.length = 0
      await getAgent(); await getMcp(); await getFolders("C:/w"); await getGit("C:/w"); await getFiles(["C:/w"], "a"); await getMemory(["C:/w"])
      expect(urls).toEqual(["/agent", "/mcp", "/agent/folders?path=C%3A%2Fw", "/agent/git?path=C%3A%2Fw", "/agent/files?path=C%3A%2Fw&q=a", "/agent/memory?path=C%3A%2Fw"])
    } finally { globalThis.fetch = real }
  })
})
