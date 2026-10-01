// The browser checks: what only a real browser can show (layout, overflow, motion, scroll, text in both languages), run
// against the mock server (serve/ui/dev/mock_server.py: no model, no GPU, never the daily server). Not part of `bun test`.
//   bun run e2e            run every check, exit 1 on a failed check or a console error
//   bun run e2e follow     only the checks whose name contains "follow"
// Needs a Chromium: set STRATA_E2E_CHROME to its executable, or install Playwright's (`npx playwright install chromium`).
// Python is `python` unless STRATA_E2E_PYTHON says otherwise.
import { chromium } from "playwright-core"
import { spawn } from "node:child_process"
import { existsSync } from "node:fs"
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
    try { await c.run({ browser, fast, long, t, errors }) } catch (e) { results.push({ name: "the check ran to its end", pass: false, detail: String(e).split("\n")[0] }) }
    if (errors.length) results.push({ name: "no console error or page error", pass: false, detail: errors.slice(0, 3).join(" | ") })
    for (const r of results) { console.log(`${r.pass ? "PASS" : "FAIL"} ${c.name.split(":")[0]}: ${r.name}${r.detail ? "  " + r.detail : ""}`); if (!r.pass) failed++ }
    console.log(`     (${c.name.split(":")[0]}: ${((Date.now() - t0) / 1000).toFixed(1)} s)`)
  }
} finally {
  await browser.close()
  fast.stop(); long.stop()
}
console.log(failed ? `\n${failed} check(s) failed` : "\nall browser checks passed")
process.exit(failed ? 1 : 0)
