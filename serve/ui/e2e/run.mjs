// The browser checks: what only a real browser can show (layout, overflow, motion, scroll, text in both languages), run
// against the mock server (serve/ui/dev/mock_server.py: no model, no GPU, never the daily server). Not part of `bun test`.
//   bun run e2e            run every check, exit 1 on a failed check or a console error
//   bun run e2e follow     only the checks whose name contains "follow"
// Needs a Chromium: set STRATA_E2E_CHROME to its executable, or install Playwright's (`npx playwright install chromium`).
// Python is `python` unless STRATA_E2E_PYTHON says otherwise.
import { chromium } from "playwright-core"
import { spawn } from "node:child_process"
import { existsSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs"
import os from "node:os"
import { fileURLToPath } from "node:url"
import path from "node:path"
import { checks } from "./checks.mjs"

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..")
const PY = process.env.STRATA_E2E_PYTHON || "python"

function chromePath() {
  const p = process.env.STRATA_E2E_CHROME || chromium.executablePath()
  if (!p || !existsSync(p)) {
    console.error("No Chromium found. Set STRATA_E2E_CHROME to a Chromium/Chrome executable, or run: npx playwright install chromium")
    process.exit(2)
  }
  return p
}

/** A mock server on a free port (the OS picks it through port 0 is not offered by the mock, so a high port is tried). */
async function startMock(env, port) {
  const child = spawn(PY, ["serve/ui/dev/mock_server.py", String(port)], { cwd: ROOT, env: { ...process.env, ...env, PYTHONIOENCODING: "utf-8" }, stdio: "ignore" })
  const base = `http://127.0.0.1:${port}`
  for (let i = 0; i < 60; i++) {
    try { if ((await fetch(base + "/metrics")).ok) return { base, stop: () => child.kill() } } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 250))
  }
  child.kill()
  throw new Error("the mock server did not start on " + port)
}

const only = process.argv[2]
const browser = await chromium.launch({ executablePath: chromePath() })
// fast: a short think and a quick read of the prompt; long: a long answer that streams for a while
const fast = await startMock({ STRATA_MOCK_THINK_MS: "2", STRATA_MOCK_PREFILL_TPS: "20000" }, 18771)
const long = await startMock({ STRATA_MOCK_LONG: "1", STRATA_MOCK_THINK_MS: "6", STRATA_MOCK_PREFILL_TPS: "20000" }, 18772)
// admin: a mock with a run config file that lists one real MCP server (serve/mcp_fake_server.py), for setting servers up from the page
const FAKE_MCP = path.join(ROOT, "serve", "mcp_fake_server.py")
const SECRET = "s3cret-e2e-value"
const adminConfig = path.join(mkdtempSync(path.join(os.tmpdir(), "strata-e2e-")), "strata-e2e.json")
writeFileSync(adminConfig, JSON.stringify({ model: "m", mcp_servers: { fake: { command: PY, args: [FAKE_MCP], env: { TOKEN: SECRET } } } }, null, 2))
// importer: a mock that reads a FAKE home folder (other apps' skills and MCP servers, issue #94); the real one is never read
const SECRET_IMPORT = "tok-imported-secret"
const fakeHome = mkdtempSync(path.join(os.tmpdir(), "strata-e2e-home-"))
const put = (rel, text) => { const f = path.join(fakeHome, rel); mkdirSync(path.dirname(f), { recursive: true }); writeFileSync(f, text) }
const skill = (rel, name, description) => put(`${rel}/${name}/SKILL.md`, `---\nname: ${name}\ndescription: ${description}\n---\n\n# ${name}\nsteps\n`)
skill(".claude/skills", "pdf-tools", "Read and write PDF files")
skill(".claude/skills", "shared-skill", "in two apps")
skill(".codex/skills", "codex-only", "only in Codex")
skill(".codex/skills", "shared-skill", "in two apps")
skill(".agents/skills", "agents-skill", "in the shared folder")
put(".cursor/mcp.json", JSON.stringify({ mcpServers: {
  fake: { command: PY, args: [FAKE_MCP] },
  quiet: { command: "node", args: ["q.js"], disabled: true },
  "secret-off": { command: "node", args: ["s.js"], env: { TOKEN: SECRET_IMPORT }, disabled: true },
} }))
put(".codex/config.toml", '[mcp_servers.off-here]\ncommand = "node"\nargs = ["off.js"]\nenabled = false\n\n[mcp_servers.nothing]\nenabled = true\n')
put(".gemini/settings.json", JSON.stringify({ mcpServers: { "g-sse": { url: "https://g.example.com/sse" } } }))
const importConfig = path.join(mkdtempSync(path.join(os.tmpdir(), "strata-e2e-imp-")), "strata-e2e-import.json")
writeFileSync(importConfig, JSON.stringify({ model: "m" }, null, 2))
const importer = { ...(await startMock({ STRATA_MOCK_THINK_MS: "2", STRATA_MOCK_PREFILL_TPS: "20000", STRATA_MOCK_MCP_CONFIG: importConfig, STRATA_HOME: fakeHome, APPDATA: path.join(fakeHome, "AppData") }, 18774)), config: importConfig, home: fakeHome, secret: SECRET_IMPORT }
const admin = { ...(await startMock({ STRATA_MOCK_THINK_MS: "2", STRATA_MOCK_PREFILL_TPS: "20000", STRATA_MOCK_MCP_CONFIG: adminConfig }, 18773)), config: adminConfig, py: PY, fakeMcp: FAKE_MCP, secret: SECRET }
let failed = 0
try {
  // one request in the history, so the request page has something to open
  await fetch(fast.base + "/v1/chat/completions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: "m", max_tokens: 5, messages: [{ role: "user", content: "hello" }] }) })
  for (const c of checks) {
    if (only && !c.name.includes(only)) continue
    const errors = []
    const results = []
    const t = { ok: (name, pass, detail = "") => results.push({ name, pass: !!pass, detail }) }
    const t0 = Date.now()
    try { await c.run({ browser, fast, long, admin, importer, t, errors }) } catch (e) { results.push({ name: "the check ran to its end", pass: false, detail: String(e).split("\n")[0] }) }
    if (errors.length) results.push({ name: "no console error or page error", pass: false, detail: errors.slice(0, 3).join(" | ") })
    for (const r of results) { console.log(`${r.pass ? "PASS" : "FAIL"} ${c.name.split(":")[0]}: ${r.name}${r.detail ? "  " + r.detail : ""}`); if (!r.pass) failed++ }
    console.log(`     (${c.name.split(":")[0]}: ${((Date.now() - t0) / 1000).toFixed(1)} s)`)
  }
} finally {
  await browser.close()
  fast.stop(); long.stop(); admin.stop(); importer.stop()
}
console.log(failed ? `\n${failed} check(s) failed` : "\nall browser checks passed")
process.exit(failed ? 1 : 0)
