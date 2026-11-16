import { beforeEach, describe, expect, test } from "bun:test"

// This browser's runner (#310 part 2): the id of the runner on the engineer's own PC, kept as `strata.runner`; which machine a chat's coding tools
// work on; the `runner=` a page route carries; which machine ran a call; why a card was closed by the server.
const data = new Map<string, string>()
;(globalThis as Record<string, unknown>).localStorage = {
  getItem: (k: string) => (data.has(k) ? data.get(k)! : null),
  setItem: (k: string, v: string) => { data.set(k, String(v)) },
  removeItem: (k: string) => { data.delete(k) },
}

const { runnerId, setRunnerId, withRunner, machineOf, isCodingServer, chatMachine, closeReason, isSkillsServer } = await import("./runner")
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

describe("isCodingServer / isSkillsServer", () => {
  test("the server's coding group and the runner's (also renamed on a clash)", () => {
    expect(isCodingServer("agent")).toBe(true)
    expect(isCodingServer("my_agent")).toBe(true)
    expect(isCodingServer("my_agent_2")).toBe(true)
    expect(isCodingServer("my_agentx")).toBe(false)
    expect(isCodingServer("fs")).toBe(false)
    expect(isCodingServer(undefined)).toBe(false)
  })
  test("skills of either machine", () => {
    expect(isSkillsServer("skills")).toBe(true)
    expect(isSkillsServer("my_skills")).toBe(true)
    expect(isSkillsServer("my_skills_2")).toBe(true)
    expect(isSkillsServer("fs")).toBe(false)
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
  test("the reasons the server gives (runner_hub.py)", () => {
    expect(closeReason("offline")).toBe("Your PC went offline, so this was closed.")
    expect(closeReason("your PC's runner restarted")).toBe("Your PC's runner restarted, so this was closed.")
    expect(closeReason("the key was revoked")).toBe("The key of this chat was revoked, so this was closed.")
    expect(closeReason("the owner refused or cleared this PC's device")).toBe("The owner refused or cleared your PC as a device, so this was closed.")
    expect(closeReason("the server is stopping")).toBe("The server is stopping, so this was closed.")
    expect(closeReason("closed")).toBe("The server is stopping, so this was closed.")
    expect(closeReason("ended")).toBe("Your PC's runner stopped, so this was closed.")
  })
  test("cancelled is the answer that stopped; an unknown reason keeps the server's words; none is null", () => {
    expect(closeReason("cancelled")).toBe("The answer stopped before you answered.")
    expect(closeReason("something new")).toBe("This was closed: something new")
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
