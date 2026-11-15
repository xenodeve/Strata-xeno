# Strata Desktop: a team workstation, used from each engineer's own PC

Status: design, for the developer's approval (2026-11-15). Line references are to the deployed tree
`C:\Strata-exp\src-main` at `27d8d5d6` (merge of upstream v0.1.42); re-check them against `main` (`e5737a27`)
before planning.

## Intent

The developer runs Strata-xeno on one strong PC (the 5060 Ti + 4070 SUPER box) as a **workstation for a team of
software engineers**. Each engineer works from their own PC, over whatever reaches the workstation: LAN, Radmin VPN,
Tailscale or a Cloudflare Tunnel. The developer's words: "as versatile as possible for others to use", "a desktop
program that picks a running Strata-xeno through its API and a token and shows everything as normal", "skills and MCP
on both the remote PC and the host, as two separate parts", "permissions clearly separated", "show the machine's
status as before, so team work is convenient".

Today a second PC can reach the server (`"host": "0.0.0.0"` + `"api_key"`, set 2026-11-15, issue #310), but the web
app's coding tools, folder picker, git panel, `@file`, memory, hooks, MCP servers and skills all act on **the
workstation's** disk and processes. An engineer cannot work on their own repository.

## Goals

1. **Strata Desktop**, a program on each engineer's PC (Windows first). It keeps a list of servers (name, URL, token),
   connects to one, and shows the server's own web app: chat, Monitor, hardware status, history, settings - the same
   pages the workstation shows locally.
2. **Two tool groups in every chat**, labelled in the UI:
   - **My PC** (the engineer's machine): the project folder and the coding tools (Read, Write, Edit, Glob, Grep,
     NotebookEdit, Bash and its output/kill, TodoWrite), git panel, `@file`, project memory, checkpoints/rewind, and the
     MCP servers and skills installed on that PC.
   - **Server PC** (the workstation): its MCP servers and skills; for the owner also its folders and coding tools.
3. **Roles per key.** Each key has a name and a role:
   - `owner`: everything, as a key holder has today.
   - `guest`: may use the workstation's model, its MCP servers' tools and its skills. Gets none of the workstation's
     folders, coding tools, Bash, hooks, git/file/memory routes, rewind, and cannot change any server setting (MCP
     config, import, hooks, web access, helpers, run config, `/settings`).
4. **Team visibility.** Monitor shows which key name holds the engine now and who waits, at which place in the queue,
   with every slot count (one slot today, more with `"parallel"`).
5. **Works over every transport** that carries HTTP(S) from the engineer's PC to the workstation, including a
   Cloudflare Tunnel: the runner only ever makes outbound requests, and no request stays silent longer than ~90 s
   (Cloudflare's Proxy Read Timeout is an idle limit between reads, 125 s by default; the chat stream and
   `/agent/run` already send keep-alives).

## Non-goals

- Running the model on the engineer's PC, or a client-side tool loop. There is one tool loop, `run_with_mcp` on the
  server (`server.py:4820`); the runner only executes tool calls and page routes.
- A pure-web mode with no install (File System Access API). Possible later as a reduced "My PC" (no Bash, no git,
  Chrome/Edge over https only); not in this design.
- Hooks on the engineer's PC (the runner runs none in v1).
- Sharing chats between engineers. Chats stay in each browser's/webview's localStorage (`strata.chats`,
  `ui/src/lib/sessions.ts:11-19`), as today.
- macOS/Linux packages (the Python runner should stay portable, but only a Windows build is delivered).
- Turning on `"parallel"`. That is a separate measured decision (issue to open: RAM, 4070 VRAM, per-request and total
  tok/s, time to first token at N = 1, 2, 4 on Gyro-M D2x; `--batch` also disables slot save/restore,
  `server.py:3215`, and MIP, #284). The developer (2026-11-15): several slots only with the 262K config, since 512K
  is not expected to fit in memory; so that test runs on `strata-gyro-m-d2x.json` (262K), not the 512K one.

## Anti-goals (a failure even if it works)

- A guest reaching the workstation's files or a shell by any route (a tool, a page route, a hook, a config write).
- One engineer reading, steering, cancelling or answering the permission prompt of another engineer's run.
- A tool call or status that goes silent while the call travels to a runner: the web app's rule "always show what the
  model is doing now" (`AGENTS.md`) holds for remote calls too (`ui/src/lib/status.ts`, `agentStatus`).
- Two tool loops that drift apart.
- The default path changing: a server with no runner connected and a single `"api_key"` behaves byte-identically.

## Constraints

- `AGENTS.md`: a new feature is opt-in and the default stays byte-identical; `docs/BLUEPRINT.md` is updated in the
  same commit for every new route, config key, process and thread; issues are bilingual; every stage the timeline does
  not show gets a span (`serve/timeline.py`), and a change that adds latency names it.
- The latency the runner adds per tool call (one network round trip plus the long-poll pickup) must be measured with a
  timeline span (`runner.call`: queued -> picked up -> result) over Radmin and over a tunnel, and reported.
- Only the standard library on the server side, as `serve/mcp.py` already does.
- Remote-tool approval prompts can wait up to 600 s (`agent_run.py:138-157`); the runner channel must carry that wait
  without one HTTP request lasting longer than ~90 s.

## Approaches considered

1. **A runner on the engineer's PC that dials out to the server, and the server's one loop routes calls to it.**
   *Chosen.* Works over every transport (outbound HTTP only), one loop, runs survive a closed window (`serve/runs.py`),
   reuses `agent.py`, `shell.py`, `mcp.py`, `skills.py`, `harness.py`, `gitview.py`, `files.py`, `folders.py`,
   `memory.py`, `checkpoints.py` as they are. Cost: new server routes and per-identity routing.
2. A local program with its own tool loop that only calls `/v1/chat/completions`. No server change, but a second loop
   (permissions, auto mode, hooks, status events) that drifts from the first; server-side MCP and skills would not be
   in the same chat. Rejected by the developer's "both parts in one chat" and by the repo's no-debt rule.
3. The page relays tool calls to a helper on `127.0.0.1` of the engineer's PC. Stops when the page closes; Chrome's
   private-network rules block an https tunnel page from reaching localhost. Rejected.
4. Browser-only (File System Access API). No Bash/git, Chrome/Edge over https only. Kept as a possible later reduced
   mode (non-goal above).

Desktop shell: **Python + pywebview, frozen with PyInstaller** (builder's call to keep or replace with an equivalent).
The runner is Python and reuses the serve modules, so one language and one process; the window uses WebView2, which
Windows 11 ships. Electron or Tauri would still need a Python sidecar.

## Design

### Identity and roles (server)

- Named keys `{"name": str, "key": str, "role": "owner" | "guest"}` live in a keys file managed live from Settings
  (decision 3 below), with the approved devices of each key (decision 1). The existing `"api_key"` / `--api-key` /
  `STRATA_API_KEY` keys stay and count as `owner` with the name `owner`.
- `_authorized` (`server.py:5531`) returns the matched identity (`name`, `role`) and the request keeps it
  (`self.identity`). With no key at all, the identity is the loopback owner, as `may_edit` decides today
  (`mcp_admin.py:51-60`).
- Every `may_edit` caller (list in the appendix) asks `identity.role == "owner"` instead of "a key is set". The
  routes that today check only `_authorized` but change server state - POST `/config` (6119, 6355), `/settings` (6401),
  `/import` - become owner-only. `strata_mcp` use and `strata_skill` stay open to guests.
- **Ownership:** a run (`serve/runs.py`), a permission/question rid (`agent_run.Broker`) and a runner belong to the
  identity that created them. `/agent/run`, `/agent/steer`, `/agent/cancel`, `/agent/permission`, `/agent/question`,
  `/agent/rewind`, `/agent/checkpoints/forget` refuse another identity's id (today they take any id: `runs.py:82,122,129`,
  `server.py:5992,6009`). The owner may cancel anyone's run (to free the engine); nobody else may.

### The runner channel

A runner is the Desktop app's background part. It connects with its token and only makes outbound requests:

- `POST /runner/hello` - protocol version, machine name, `sys.platform`, shell description, home folder, and the tool
  list of its groups (its coding tools; its MCP servers' tools; its skills tools). Answers a `runner_id` bound to the
  identity, or 409 with the server's protocol version when they do not match (the app then says "update Strata
  Desktop").
- `GET /runner/next?id=&wait=25` - long-poll for jobs; returns at most after 25 s with nothing. The runner is "online"
  while it polls; it goes "offline" after 60 s without a poll, and its pending jobs fail with an error the model reads.
- `POST /runner/reply` - a job's result, or a message in the middle of a job (below).
- Jobs are of two kinds: **tool calls** from `run_with_mcp`, and **page routes** the server forwards (folders, git,
  files/mention, memory, rewind/forget) when the page asks about "My PC".
- **Calls back during a job.** A runner-side tool call needs the server for: the permission prompt / auto-mode judge
  (`ctx.ask`), the question card (`ctx.question`), events (`ctx.emit`, e.g. hook-free progress), the cancel signal, the
  vision flag, web settings. The runner's `AgentContext` (`agent.py:57-70`) gets proxies: `ask` posts
  `{"job", "ask": card}` with `/runner/reply` and waits for the answer on its next `/runner/next` polls; cancel arrives
  the same way. So no request outlives ~25 s, and a 600 s approval wait is many short polls.
- The permission gate (`permissions.decide`, which resolves real paths on disk), session state (`Session`: reads,
  todos, background shells, `agent.py:73-81`), checkpoints (`checkpoints.py:85-225`) and nested instruction files
  (`_with_nested`, `agent.py:301`) all run **on the runner**, inside its own `AgentServer`. The server keeps the
  Broker, the judge (`side_request`), and the run's event stream to the page.

### Routing tool calls (server)

- A runner registers in the hub as a duck-typed built-in server (fields `name`, `kind`, `status`, `tools`, `call`,
  `start`, `close`, `wants_context=True`; `agent.py:187-213`, `mcp.py:497-501`) whose `call` sends a job and waits
  for the reply.
- The hub's routes are global today (`McpHub.routes`, `mcp.py:520-535`). They become **per request**: a request sees
  the server's groups it may use (guest: MCP + skills; owner: also `agent`) plus the groups of **its own identity's
  runner** that the request names (`strata_agent.runner`). Another identity's runner is never in the map.
- **Names.** The chat's project lives on one machine, and the coding tools keep their plain names (`Read`, `Bash`, ...)
  for that machine: the runner's when the project is on My PC, the server's `agent` when an owner picks a server
  folder. A request never carries both coding groups. MCP and skills tools of both machines are offered together; the
  runner's get a distinct prefix (proposal: `my_<server>__<tool>`, and `my_skills` for the skills server) so no name
  clashes into `_2` (`mcp.py:530`). The event's `server` field carries the group so the page can label it.
- `subagent.py:94` hard-codes `"agent"` as the coding server; it takes the run's coding group instead.

### What moves to the runner in `start_agent_run` (`server.py:3117-3177`)

When the project is on My PC, the server does not touch its own disk for the run. The runner reports (at hello, and
per run on request) the facts the server needs and does the disk work itself:

| Today (server) | With a runner |
|---|---|
| `realpath`/`isdir` of `cwd`, `dirs` (3120-3128) | runner validates and returns the real paths |
| shell (`self.agent.shell`, 3138) | the runner's shell |
| `Policy(..., protected=[config_path, mcp_config_path])` (3139) | Policy built on the runner; protected = the runner's own config files |
| checkpoint scope (3142-3144) | runner's checkpoint store |
| `.git` lookup (3145-3153) | runner |
| `agent_prompt.build(..., sys.platform, git, ..., memory_blocks)` (3168) | prompt text built on the server from facts the runner returns (platform, shell, git, memory blocks from the runner's home and project) |
| hooks (`hooks_mod.Runner`, 3169-3176) | none for runner projects in v1; server hooks never run for a guest |
| web settings, helpers, judge, vision (3154-3166, 3170) | stay on the server |

### Page routes for My PC

`/agent/folders`, `/agent/git`, `/agent/git/diff`, `/agent/files`, `/agent/mention`, `/agent/memory`,
`/agent/memory/file`, `/agent/rewind`, `/agent/checkpoints/forget` take a `runner=<id>` parameter (or body field).
With it, the server checks the runner belongs to the caller and forwards the request as a job; without it, they act
on the workstation and are owner-only.

### The web app (`serve/ui`)

- **Knowing its runner.** Strata Desktop sets two localStorage values on the server's origin before the app starts:
  `strata.apikey` (the token; `api.ts:8-14` already sends it) and `strata.runner` (its runner id). A page with no
  runner (a plain browser) behaves as today.
- **Project folder picker** (`FoldersEditor.tsx`, `NewProjectDialog.tsx:89`, `AgentControls.tsx:65`): a machine
  choice, "My PC" / "Server PC" (the latter only for an owner). A project stores its machine with its folders
  (`sessions.ts`), and `agentRequest` (`lib/agent.ts:11-23`) sends `runner` with `cwd`/`dirs`.
- **Tool cards** (`Messages.tsx:117`, `AgentCall.tsx`) show which machine ran the call, from the event's `server`
  field.
- **Settings:** MCP, skills/import (`McpSettings.tsx`, `ImportSettings.tsx`) show two sections, "My PC" and "Server
  PC". The My PC section comes from the runner (its MCP servers and skills, its own config file); the Server PC
  section is read-only for a guest.
- **Status words** (`lib/status.ts`) gain the remote steps, sent by the server when they begin: *sending the call to
  your PC*, *running Bash on your PC*, *your PC is offline* (with what the model was told). A status that appears only
  when the call returns breaks the rule.
- **Monitor:** each running and waiting request shows its identity name and queue place.
- **Origin.** The app is loaded from the server's own URL, so `_own_page` / `/agent/run`'s Origin check
  (`server.py:5721`, `6328-6338`) passes as today. No `trusted_origins` change is needed.

### Strata Desktop (the program)

- **Window:** first screen = the server list (name, URL, token, "test" = `GET /health` then an authorized `GET /props`;
  shows role and model). Then a webview on the chosen server's URL, with the two localStorage values injected.
- **Runner:** starts with the app, keeps polling while the window is closed (tray icon: online / offline / which
  server), stops on "Quit". Its MCP servers are configured in its own config file (same format as the server's
  `mcp_servers`, `mcp.py` docstring) and it imports the PC's skills/MCP from Claude Code, Codex, etc. through
  `harness.py`, as the server does.
- **Tokens** are stored with Windows DPAPI (per user), not in plain text.
- **Packaging:** one PyInstaller build (`StrataDesktop.exe`) that bundles the serve modules it needs; Git for Windows'
  bash is found as `shell.py` finds it today, and its absence is reported in the app, not a crash.
- **Versions:** the page always comes from the server, so it matches the server. The runner protocol carries a
  version; a mismatch is refused at hello with a message.

## Security summary

- Every route the runner uses needs a valid key; runner ids and run ids are bound to the identity.
- A guest's tools that touch files or run commands only ever run on the guest's own PC, by the guest's own runner,
  behind the same permission gate as today (approvals on their own page).
- Server MCP servers are shared tools, but in team mode a guest gets only the servers the owner marks for guests
  (run-config `"guest_mcp"`, default empty; "Guests may use" per server in Settings). Superseded the earlier "notice
  only" call: see "Changes during part 1" below (R11).
- Hooks, web-access settings, helpers, run config and imports are owner-only.

## Decisions by the developer after the builder check (2026-11-15)

1. **Trust: the key holder and the server are trusted** (the runner is a thin executor; rules and mode come with the
   request, as written above), **plus device approval**: when a key is used from a device it has not been used from
   before, the request is held and a notification asks to approve or refuse that device.
   - A **device** is a Strata Desktop install (a device id and secret it creates on first run, kept with DPAPI) or a
     browser (an id kept in its localStorage, sent as a header by `apiHeaders`). The server stores approved devices
     per key, with name, first/last seen and address. Loopback requests on the workstation are approved implicitly.
   - Until approved, every route but `/health`, `/` and the static app answers 403 `device_pending` with the device's
     short code; the page shows "waiting for approval, code ABCD"; a runner from that device is not accepted.
   - **Who approves (designer's call):** the owner, from Settings and from a notification in the owner's web app and
     Desktop tray; and any device already approved for the same key, so an engineer approves their own second PC. The
     **first** device of a new key needs the owner, so a key that leaks before its holder uses it is still caught.
   - Refusing a device blocks it for that key until the owner clears it. Approving or refusing is logged.
2. **Monitor visibility:** a guest sees other people's requests as name, state, queue place and timings only; previews,
   kept prompts and detail records are visible to their own identity and to the owner. The owner sees everything.
   Claude Code / OpenAI calls (`/v1/*`) carry identity too, so their records are filtered the same way.
3. **Keys are managed live:** the owner adds, renames, changes the role of and revokes keys in Settings, and it takes
   effect at once without a restart. Revoking a key ends its runners and cancels its runs. Keys are stored in a file of
   their own beside the run config (not in the run config, so a key change never rewrites it), read at start and on
   change; `"api_key"` / `--api-key` keep working as the built-in owner key.

## Calls made by the designer (the developer may overturn any)

- **More owner-only routes:** `/unload`, `/load`, `/v1/vram`, `/slots/*` and `POST /metrics/keep` (`server.py:6106`)
  change the shared engine or what is recorded, so they become owner-only too.
- **Web tools (WebFetch, WebSearch) run on the server** for every project, My PC included, under the server's own
  switch (`serve/web.py`). The Brave key never leaves the server (`web.py:40`); when the server's web access is off,
  nobody's runs get the tools. The coding group is therefore split: file and shell tools on the project's machine, web
  tools on the server. Helpers (`subagent.py`) and the memory "apps switched on" setting likewise follow the server's
  config.
- **Plain HTTP:** allowed in v1. Strata Desktop warns when the server URL is `http://` to a non-loopback address that
  is not Radmin (26.0.0.0/8) or Tailscale (100.64.0.0/10), because the token and the files the runner returns would
  travel in clear; it recommends Tailscale or a tunnel. No TLS in the stdlib server in v1.
- **Prompt-cache contention between engineers** is measured and reported in v1, not handled. The gate is taken per
  model round (`server.py:4370-4400`), so two engineers' rounds alternate and a layer-split engine has one cache slot;
  the re-read cost is not measured. A timeline run with two runners alternating on D2x gives the number, and an issue
  carries it with the fix to consider (per-identity slots, holding the engine across a run's tool calls).
- **Cloudflare:** the poll limit above is kept at 25 s, well inside the 125 s idle limit.

## Changes during part 1 (controller rulings, 2026-11-15/16; the developer may overturn any)

Part 1 (team keys, branch `xeno/team-keys`) settled these; parts 2 and 3 plan from this text, not the older wording
above where they differ.

- **R2 - team mode is opt-in.** Team mode = the keys file exists (created by the owner's first key, from the
  workstation itself or with the legacy key). Without it the server behaves as before: no device approval, today's
  edit rules. So a server with only the single `"api_key"` gets no device approval until the first named key exists.
  To leave team mode: stop the server, delete the keys file, start it again.
- **R6 -** in team mode every request needs a key, the workstation's own browser included (as with an `api_key`
  today).
- **R8 -** the names of waiting requests are in `/metrics` `live.waiting_who: [{"who","since"}]`; `live.waiting` keeps
  its #465 meaning (a count, only with `"parallel"`).
- **R9 -** a key's name is never reused (revoked and renamed-from names are retired).
- **R11 - server MCP for guests is opt-in per server** (`"guest_mcp"`), because an imported filesystem or shell MCP
  server would otherwise give every guest the workstation's files or shell (anti-goal 1). Skills stay open to guests
  (read-only, fenced to each skill's folder).
- **R12 -** an owner-role key sees every run and record (decision 2); owner keys are not isolated from each other, so
  a team should have one owner.
- Guests' image inputs: no local paths or `file://`, and http(s) URLs only to public addresses (#311).

## Left to the builder

- Wire format of jobs and replies (JSON fields), poll timing within the ~90 s limit, and retry/backoff.
- Exact prefix spelling for My-PC MCP/skills tools, as long as names are unique, under 64 characters and grouped.
- How identity reaches `Monitor` and history records (`serve/history.py`, `telemetry.py`).
- Module layout (proposal: `serve/runner_hub.py` server side, `desktop/` for the app), tests (`serve/test_runner_*.py`
  with a fake runner over the real HTTP server, as `mcp_fake_server.py` does for MCP), and the order of work.
- Whether pywebview stays, if a measured problem appears (e.g. WebView2 missing on Windows 10 machines).

## Appendix: today's key gates (from the code map, `server.py` at `27d8d5d6`)

- `may_edit` callers, GET: `/agent` 5692, `/agent/hooks` 5700, `/agent/helpers` 5709, `/agent/run` 5718,
  `/agent/web` 5738, `/agent/folders` 5747, `/agent/git` 5757, `/agent/files` + `/agent/mention` 5769,
  `/agent/memory` 5781, `/mcp/config` (via `mcp_admin.view`).
- `may_edit` callers, POST: `/mcp/config` 5911, `/agent/rewind` + `/agent/checkpoints/forget` 5929, `/agent/steer`
  5951, `/agent/cancel` 5966, `/agent/question` 5981, `/agent/permission` 6000, `/agent/helpers` 6017, `/agent/web`
  6042, `/agent/hooks` 6067, `/import` 6093, `strata_agent` in a chat request 6493.
- Only `_authorized`: POST `/config` 6119/6355, `/settings` 6401, `strata_mcp` 6521, `strata_skill` 6481.
- No identity is recorded today; every key holder is an owner.
