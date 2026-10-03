// Starts the demo backend: the mock server of the real Strata-xeno checkout (serve/ui/dev/mock_server.py).
//
//   STRATA_SRC=<path to a Strata-xeno checkout> npm run demo
//
// What is real: the web app (serve/ui), the server (serve/*.py), its routes and pages.
// What is fake: the model. A script answers, and the engine's statistics are fixtures.
// It binds to 127.0.0.1 only, uses a temporary history folder, and points the importers at an empty folder,
// so no chat, skill, MCP server or memory of this PC is shown. It never starts an engine and never touches a GPU.
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// the checkout is, in order: STRATA_SRC, or the repository this folder sits in (website/ is a folder of the Strata-xeno repository)
const here = dirname(fileURLToPath(import.meta.url));
const candidates = [process.env.STRATA_SRC, resolve(here, "..", "..")].filter(Boolean);
const src = candidates.find((p) => existsSync(join(p, "serve", "ui", "dev", "mock_server.py")));
if (!src) {
  console.error("No Strata-xeno checkout found. Set STRATA_SRC to the folder that holds serve/ui/dev/mock_server.py.");
  process.exit(1);
}
const port = process.env.DEMO_PORT || "8187";
const emptyHome = mkdtempSync(join(tmpdir(), "strata-demo-home-"));
const py = process.env.PYTHON || "python";

console.log(`[demo] Strata-xeno checkout: ${src}`);
console.log(`[demo] backend: http://127.0.0.1:${port}/next/   (the site proxies it at /demo/next/)`);
const child = spawn(py, ["serve/ui/dev/mock_server.py", port], {
  cwd: src,
  windowsHide: true,
  stdio: "inherit",
  env: {
    ...process.env,
    PYTHONDONTWRITEBYTECODE: "1",
    STRATA_MOCK_AGENT: "1", // the chat's coding tools run on a script: type "agent demo"
    STRATA_HOME: emptyHome, // the importers read an empty folder, never the real home
    STRATA_MOCK_PREFILL_TPS: "40000",
    STRATA_MOCK_THINK_MS: "6",
  },
});
child.on("exit", (code) => process.exit(code ?? 0));
for (const sig of ["SIGINT", "SIGTERM"]) process.on(sig, () => child.kill());
