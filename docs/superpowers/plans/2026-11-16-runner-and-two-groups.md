# The runner and the two tool groups ("My PC" / "Server PC") - Implementation Plan (part 2 of Strata Desktop)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let each engineer work on their own PC's repository from the workstation's chat: a runner on their PC dials out
to the server, executes the coding tools, its own MCP servers and skills, and the "My PC" page routes, while the
server keeps the one tool loop, the approvals, the judge, the web tools and the event stream.

**Architecture:** A new server module `serve/runner_hub.py` keeps the connected runners, their job queues and the
duck-typed hub groups (`my_agent`, `my_skills`, `my_<server>`) whose `call` sends a job and waits for the reply over
three routes (`POST /runner/hello`, `GET /runner/next`, `POST /runner/reply`). Each chat request gets its own route
view of the hub (`McpHub.with_servers`): the server's groups the identity may use plus its own runner's groups.
The runner program `python -m serve.runner` reuses `agent.py`, `shell.py`, `mcp.py`, `harness.py`, `checkpoints.py`
and a new `serve/page_routes.py` (the page routes' bodies, shared by the server and the runner, so they cannot drift).
The web app learns its runner id, lets a project live on "My PC" or "Server PC", and labels and narrates remote calls.

**Tech Stack:** Python 3.11 standard library on both ends (`http.server`, `urllib.request`, `threading`, `json`,
`hashlib`), unittest/pytest; React + TypeScript in `serve/ui` (bun test, vite build, `dist/` committed).

**Spec:** `docs/superpowers/specs/2026-11-15-strata-desktop-design.md` - "The runner channel", "Routing tool calls
(server)", "What moves to the runner in start_agent_run", "Page routes for My PC", "The web app (serve/ui)", "Security
summary", "Changes during part 1" (R2, R6, R11, R12), anti-goals 1-5. Part 1's plan
(`docs/superpowers/plans/2026-11-15-team-keys-and-devices.md`) is done on branch `xeno/team-keys` (head `01036713`).
Part 3 (pywebview window, tray, DPAPI, PyInstaller) is not in this plan.

## Global Constraints

- Default path byte-identical: a request without `strata_agent.runner` builds the same tool list (names, order,
  descriptions), the same prompt and the same events as before; the `call` event gains no key; `GET /agent` without
  `runner=` answers as before. A server no runner ever connected to differs only by the new routes existing.
- Standard library only, server and runner. Part 1's rules all hold: owner-only routes stay owner-only, 404 (not 403)
  for another identity's ids, device approval applies to the runner's requests like any other.
- Protocol: `RUNNER_PROTOCOL = 1`. Poll: `GET /runner/next` waits at most `POLL_MAX_S = 25` s (clamped 0..25);
  `OFFLINE_S = 60` (no poll held and none for 60 s = offline); no request on the channel may stay silent longer than
  ~90 s.
- Runner id: `"r" + sha256(f"{who}\n{device}").hexdigest()[:16]` - stable across runner and server restarts, so the
  page's stored id keeps working. Knowing an id grants nothing: every lookup is `RunnerHub.get_for(id, identity)`,
  which matches `who == identity.name` exactly - the owner role too (nobody runs commands on another person's PC).
- Group names on the server: the runner's coding group is `my_agent` (plain tool names, hidden from `GET /mcp`); its
  skills server `my_skills`; each of its MCP servers `my_<name>` (tools `my_<name>__<tool>` by the hub's existing rule).
  Runner groups are appended after the server's groups, so on a name clash the runner's tool gets the `_2`, never the
  server's. Every runner group has `machine = "my"`.
- A request never carries two coding groups: with `runner`, the server's `agent` is replaced in the view by its
  subset `WebFetch`, `WebSearch`, `Task` (they run on the server under the server's own switches, for guests too; the
  R11 `guest_mcp` filter does not remove this subset). Without `runner`, nothing changes.
- Guests may launch coding runs only with `strata_agent.runner` naming their own online runner; the server's own coding
  tools stay owner-only. Outside team mode a runner is an owner's (`_owner_only` on the runner routes).
- No hooks for runner projects; the runner never reads the server's config; the memory "apps switched on" list, web
  access and helpers come from the server's config in each job.
- Runner files: `runner_dir()` = `%LOCALAPPDATA%\StrataRunner` on Windows, `~/.strata-runner` elsewhere, holding
  `runner.json` (its run config: `mcp_servers` / `mcpServers`, `"skills"` switches, same format as the server's),
  `device.json` (`{"id", "name"}`, made once) and `checkpoints/`. `--key` may come from `STRATA_RUNNER_KEY`.
- Every process started here runs without a window (`CREATE_NO_WINDOW` for children, background shells); never touch
  port 8091, `C:\Strata-exp\src-main` or the `D:/Github/Strata` configs. Test servers use port 8099 or 0.
- `docs/BLUEPRINT.md` in the same commit as each task that adds a route, file, thread, config key or event
  (`tools/hooks/commit-msg`); issue comments English + Thai on `xenodeve/Strata-xeno` #310 with `--repo`.
- Python tests: `python -m pytest serve/<file> -q` from the repo root; tests set `runner_hub.POLL_MAX_S` /
  `OFFLINE_S` small and `RunnerHub.close()` wakes held polls so `httpd.shutdown()` never waits 25 s. UI:
  `cd serve/ui && bun test && bun run typecheck && bun run build`, then `python -m pytest serve/test_ui.py -q`, `dist/`
  committed. bun is `C:/Users/xenod/.bun/bin/bun.exe`.

## Review Focus

1. **The runner goes silent in the middle of a call** (laptop lid closed, Radmin drops after the job was picked up).
   The call must end within `OFFLINE_S` with text the model reads ("went offline while ... was running; whether it
   finished is unknown"), a `remote` event `offline`, and no fallback to the server's tools. Test in Task 2
   (`test_offline_after_pickup_fails_with_text`), and the not-yet-picked case (`test_offline_before_pickup`).
2. **A guest names another identity's runner** - in `strata_agent.runner`, in a page route's `runner=`, in
   `/runner/next?id=` or `/runner/reply`. Every one answers 404 and nothing reaches either PC. Tests in Task 1
   (`test_poll_and_reply_for_another_identity_are_404`), Task 4 (`test_guest_cannot_route_to_another_runner`) and
   Task 5 (`test_page_route_with_another_runner_is_404`).
3. **A 600 s approval through Cloudflare** (idle limit 125 s). The approval must be carried by many short polls: no
   single `/runner/next` outlives `POLL_MAX_S` + 1 s, and the answer reaches the runner on the next poll. Test in
   Task 2 (`test_long_approval_spans_many_polls`, `POLL_MAX_S = 0.5`, answer after 3 s).
4. **A key is revoked while its runner is polling and a job is open.** The job fails ("the key was revoked"), the
   runs stop, the next poll is 401, and the runner exits instead of retrying forever. Tests in Task 1
   (`test_revoke_ends_runner`), Task 2 (`test_revoke_fails_open_job`) and Task 6 (`test_runner_exits_on_401`).
5. **Two chats (or two windows) use one runner at once, and a second runner process starts on the same PC.** Jobs
   run concurrently, replies out of order reach the right call, each chat keeps its own session (todos, reads,
   background shells); the newer process replaces the older one (409 `runner_replaced` on the old one's poll) and the
   older one's open jobs fail with text. Tests in Task 2 (`test_two_chats_share_one_runner`) and Task 1
   (`test_second_instance_replaces_first`).

---

### Task 1: The runner channel - registration, polling, presence

**Files:**
- Create: `serve/runner_hub.py`; Test: `serve/test_runner_hub.py`, `serve/test_runner_http.py`
- Create: `serve/runner_fake.py` (a scripted runner over real HTTP for tests, as `serve/mcp_fake_server.py` is for MCP)
- Modify: `serve/server.py` - `Service.__init__` (beside `self.broker`/`self.runs`, ~3101: `self.runner_hub =
  runner_hub.RunnerHub()`), `do_GET` (`/runner/next`, `/runners`), `_do_post` (`/runner/hello`, `/runner/reply`),
  `_keys_post` (rename -> `runner_hub.move`; revoke -> `runner_hub.end_all`), `_devices_post` (refuse / clear ->
  `runner_hub.end_device`), shutdown path (`runner_hub.close()`)

**Interfaces:**
- Consumes: part 1 `Identity`, `self.identity`, `self.device_id`, `self.team`, `_owner_only`, `_acting()`.
- Produces (`serve/runner_hub.py`):
  - Constants `RUNNER_PROTOCOL = 1`, `POLL_MAX_S = 25.0`, `OFFLINE_S = 60.0`, `CANCEL_GRACE_S = 10.0`,
    `REQUEST_TIMEOUT_S = 30.0`.
  - `runner_id(who: str, device: str) -> str`.
  - `class Runner`: `id`, `who`, `device`, `instance`, `info: dict` (`machine`, `platform`, `shell`, `home`),
    `groups: list[dict]`, `skills: list[dict]`; `online(now: float | None = None) -> bool`;
    `take(wait: float) -> list[dict]` (blocks on a `threading.Condition` until a message or `wait`; marks each job it
    hands out picked up with `t_picked`); `deliver(msg: dict) -> bool` (Task 2 fills the job side); `fail_all(text:
    str) -> int`; `public(full: bool) -> dict`.
  - `class RunnerHub`: `hello(who, device, instance, info: dict, groups: list, skills: list) -> Runner` (same id and
    instance: update lists only; same id, new instance: the old one's jobs fail with "your PC's runner restarted" and
    its next poll is refused); `get_for(rid, viewer: Identity) -> Runner | None`; `move(old, new) -> int`;
    `end_all(who) -> int`; `end_device(who, device) -> int`; `view(viewer: Identity) -> list[dict]`; `close()`.
- Wire format:
  - `POST /runner/hello` body `{"protocol": 1, "instance": str 8-64, "machine": str <= 80, "platform": str, "shell":
    str | null, "home": str, "groups": [{"name", "kind": "coding" | "mcp" | "skills", "status", "error", "tools":
    [{"name", "description", "inputSchema"}]}], "skills": [{"name", "description", "app"}]}` (cap 4 MB) -> 200
    `{"runner": id, "who": name, "protocol": 1, "poll_s": 25}`; wrong protocol -> 409 `{"error": {"type":
    "runner_protocol", "server_protocol": 1, "message": "update Strata Desktop (the server speaks runner protocol
    1)"}}`.
  - `GET /runner/next?id=&instance=&wait=` -> 200 `{"messages": [...]}`; unknown or another identity's id -> 404;
    replaced instance -> 409 `{"error": {"type": "runner_replaced"}}`.
  - `POST /runner/reply` body `{"id", "instance", "job", ...}` (cap 32 MB: a Read image fits) -> 200 `{"ok": true}`,
    404 for an unknown or finished job.
  - `GET /runners` -> `{"runners": [Runner.public(...)]}`: own runners and, for the owner, all in full; a guest sees
    others' only as `{"who", "online"}`.
  - Gate of all four: `_authorized()`, then in team mode any identity (device approval already ran), outside it
    `_owner_only("a runner can connect")`. No `_own_page` (the runner is not a page) but JSON only on POST.
  - The device of a hello is `self.device_id` in team mode; outside team mode `_authorized` leaves it `""`, so hello
    takes `identity_mod.device_of(X-Strata-Device, X-Strata-Device-Name, User-Agent, peer)[0]` there - two PCs on a
    single-user server get two ids.
  - `_stop_identity` stays as it is (it also runs on a demotion to guest, and a guest keeps her runner); the revoke
    branch of `_keys_post` calls `runner_hub.end_all(name)` after it.

- [ ] **Step 1: Write the failing tests.** `serve/test_runner_hub.py` (no HTTP): `test_runner_id_is_stable`
  (same who+device -> same id, other who -> other id); `test_take_returns_at_wait_with_nothing` (wait 0.2 -> `[]` in
  0.2-0.5 s); `test_online_window` (`OFFLINE_S = 0.3`: online after hello, offline 0.4 s later, online while a `take`
  is held); `test_get_for_matches_who_exactly` (owner identity does not get ann's runner);
  `test_second_instance_replaces_first` (RF 5: second hello with a new instance -> same id; the old instance is
  refused); `test_close_wakes_held_polls`. `serve/test_runner_http.py` (part 1's `Base` start helper, team mode, guest
  ann and bob with approved devices): `test_hello_returns_id_and_who`; `test_protocol_mismatch_is_409` (`protocol: 2`
  -> 409 `runner_protocol`, `server_protocol == 1`); `test_pending_device_runner_is_refused` (403 `device_pending`);
  `test_poll_and_reply_for_another_identity_are_404` (RF 2: bob polls ann's id -> 404, replies -> 404);
  `test_runners_view_by_identity` (ann sees her runner in full, bob's as who/online; owner sees both in full);
  `test_revoke_ends_runner` (RF 4: owner revokes ann -> her runner is gone from `/runners`, her next poll 401);
  `test_demoted_owner_keeps_runner` (role owner -> guest: the runner stays online); `test_outside_team_mode_runner_is_owner`
  (no keys file, `api_key` set: remote request with the key -> 200; without key 401);
  `test_two_devices_outside_team_mode_get_two_ids` (no keys file, two `X-Strata-Device` values -> two ids, neither
  replaced).
- [ ] **Step 2: Run** `python -m pytest serve/test_runner_hub.py serve/test_runner_http.py -q` - Expected: FAIL
  (`ModuleNotFoundError: serve.runner_hub`).
- [ ] **Step 3: Implement** the module and the four routes as above; `runner_fake.FakeRunner(port, key, device,
  groups, on_job: Callable[[dict, Callable[[dict], None]], None])` runs hello + a poll thread and calls `on_job(msg,
  reply)` per message, `stop()` ends it (Tasks 2-5 script it).
- [ ] **Step 4: Run** the same command plus `serve/test_identity_http.py serve/test_security.py` - Expected: all pass.
- [ ] **Step 5: Commit** `feat(serve): the runner channel - hello, long-poll, reply, presence (#310 part 2)`;
  BLUEPRINT: the four routes, `serve/runner_hub.py` and `serve/runner_fake.py` in the file table, the runner as a new
  process in the process picture (row in "Not yet in the baseline").

### Task 2: Jobs and the calls back during a job

**Files:**
- Modify: `serve/runner_hub.py` (`Job`, `Runner.submit`, `Runner.request`, `RunnerGroup`, `Runner.servers`)
- Modify: `serve/mcp.py` `McpHub.call` (~553): optional `progress`
- Test: `serve/test_runner_hub.py`, `serve/test_runner_http.py`

**Interfaces:**
- Consumes: Task 1 `Runner`, `take`, `deliver`.
- Produces:
  - `Runner.submit(kind: str, payload: dict) -> Job`; `Job`: `id`, `t_queued`, `t_picked`, `inbox: queue.Queue`,
    `done: threading.Event`, `result`.
  - `Runner.request(kind: str, payload: dict, timeout: float = REQUEST_TIMEOUT_S) -> dict` - for `run.start` and
    `page` jobs: the result, or `{"error": "offline" | "timeout", "message": str}`.
  - `Runner.servers() -> dict[str, RunnerGroup]` keyed by the server-side names (`my_agent`, `my_skills`, `my_<x>`).
  - `RunnerGroup`: `name`, `kind = "runner"`, `status`, `error`, `info`, `tools`, `machine = "my"`,
    `wants_progress = True`; coding group also `wants_context = True`, `plain_names = True`, `hidden = True`;
    `call(tool, arguments, timeout, cancel, ctx=None, progress=None) -> dict` (MCP result shape `{"content",
    "isError"}`).
  - `McpHub.call(name, arguments, cancel=None, ctx=None, progress: Callable[[str], None] | None = None)`: passes
    `progress=` only to a server with `wants_progress`; nothing else changes.
- Messages server -> runner (in `/runner/next`): `{"job", "kind": "tool", "group": <runner-local name: "agent",
  "skills" or the MCP server's>, "tool", "arguments", "run": ctx.run_key, "session": ctx.session, "mode":
  ctx.policy.mode, "helper": ctx.in_helper, "vision": ctx.vision, "timeout_s"}`, `{"job", "kind": "run.start", ...}`
  (Task 3), `{"job", "kind": "page", ...}` (Task 5), `{"job", "answer": ...}`, `{"job", "cancel": true}`.
- Messages runner -> server (`/runner/reply`): `{"job", "result": {...}, "exec_ms": int}`, `{"job", "ask": card}`,
  `{"job", "question": questions}`, `{"job", "emit": event}`.
- `RunnerGroup.call` loop (0.25 s ticks, on the hub's worker thread): `progress("sending")` at submit,
  `progress("running")` once picked; an `ask` / `question` message is served inline with `ctx.ask(card)` /
  `ctx.question(qs)` and the answer queued as `{"job", "answer"}`; `emit` -> `ctx.emit(event)`; `cancel` set ->
  `{"job", "cancel": true}`, then up to `CANCEL_GRACE_S` for the runner's result, else
  `"The request was cancelled."`; offline -> `progress("offline")` and the error texts below. A coding call has no
  overall timeout (as today); an MCP or skills call waits `timeout + 30` s.
- Exact texts the model reads (`{m}` = machine name):
  - not picked up: `your PC ({m}) is offline, so {tool} was not run. Tell the user; do not retry it until they say
    their PC is back.`
  - picked up: `your PC ({m}) went offline while {tool} was running; whether it finished is unknown. Tell the user,
    and check the state before you repeat it.`
  - replaced: `your PC's runner restarted while {tool} was running; whether it finished is unknown.`
  - revoked: `the key of this chat was revoked, so {tool} was not finished.`
- Timeline, on the calling thread at the end of every job: `timeline.complete("runner.call", t_queued, t_end,
  exec_ms, chars)` and `timeline.complete("runner.call wait", t_queued, t_picked)`.

- [ ] **Step 1: Write the failing tests** (FakeRunner over HTTP unless noted): `test_tool_job_round_trip` (result text
  comes back through `hub.call`, `progress` saw `["sending", "running"]`); `test_ask_is_served_by_ctx_ask` (fake sends
  `ask`, a stub ctx answers `"allow"`, the fake receives `{"answer": "allow"}` and then replies);
  `test_question_and_emit_reach_ctx`; `test_cancel_reaches_runner` (cancel set mid-job -> fake sees `cancel`, call
  returns within grace); `test_offline_before_pickup` (RF 1: no poller, `OFFLINE_S = 0.3` -> the "is offline" text,
  `progress` ends with `"offline"`); `test_offline_after_pickup_fails_with_text` (RF 1: fake picks up then stops
  polling -> "went offline while" text within `OFFLINE_S + 1`); `test_long_approval_spans_many_polls` (RF 3:
  `POLL_MAX_S = 0.5`; ctx.ask answers after 3 s; the fake records every poll's duration: all <= 1.5 s, and the answer
  arrives); `test_two_chats_share_one_runner` (RF 5: two calls from sessions s1/s2 at once, the fake answers the
  second first; each call gets its own text); `test_revoke_fails_open_job` (RF 4: a job picked up, owner revokes ann
  -> the call returns the "revoked" text at once); `test_spans_written` (unit: with `timeline.configure(tmp)`, one job ->
  one `runner.call` and one `runner.call wait` record); `test_hub_call_without_progress_unchanged` (unit, `McpHub` with
  a plain fake server: same result dict as before).
- [ ] **Step 2: Run** `python -m pytest serve/test_runner_hub.py serve/test_runner_http.py serve/test_mcp.py -q` - Expected: FAIL.
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run** the same command - Expected: all pass.
- [ ] **Step 5: Commit** `feat(serve): runner jobs, approvals and questions over short polls, runner.call span (#310 part 2)`;
  BLUEPRINT: job/message format, the `runner.call` spans in Diagnostics.

### Task 3: `start_agent_run` for a project on My PC

**Files:**
- Modify: `serve/folders.py` (two helpers moved out of `start_agent_run`), `serve/server.py` `start_agent_run` (~3148),
  `serve/agent.py` `AgentContext` (~57), `serve/agent_run.py` `AgentRun.__init__` (~115)
- Test: `serve/test_folders.py`, `serve/test_runner_run.py` (Service level, fake `Runner.request`)

**Interfaces:**
- Produces:
  - `folders.resolve(cwd_raw, dirs_raw, max_dirs: int) -> tuple[str | None, list[str]]` and
    `folders.in_git(folder: str | None) -> bool` - today's lines 3151-3159 and 3172-3180, unchanged in behaviour; the
    server and the runner both call them.
  - `AgentContext.run_key: str = ""` (the runner-side run this context's calls belong to).
  - `AgentRun.hub = None` (the request's route view, Task 4), `AgentRun.coding = "agent"` (the coding group's name),
    `AgentRun.runner = None`.
  - `Service.start_agent_run(sa, messages, who="owner", runner: Runner | None = None) -> AgentRun`; raises
    `RunnerUnavailable(message: str)` (new, in `runner_hub`) when the facts job fails.
- `run.start` payload `{"cwd", "dirs", "mode", "allow", "deny", "session", "checkpoint", "memory_on"}` (the server's
  sanitised values; `memory_on` from `memory_mod.settings(conf)["on"]`); result `{"run": run_key, "cwd", "dirs",
  "shell", "platform", "git", "memory": blocks, "tools": [coding tool names]}`.
- With `runner`: no `os.path` call on `cwd`/`dirs` on the server, no `hooks_mod`, no `self.checkpoints`; the server's
  mirror `Policy(cwd=facts cwd, dirs=facts dirs, mode, allow, deny, protected=[])` serves the judge and the server-side
  web tools; `shell` = the runner's; `run.ctx.run_key`, `run.coding = "my_agent"`, `run.runner = runner`; web, helpers,
  vision and the judge exactly as today; `tools` for the prompt = the runner's coding names + `WebFetch`, `WebSearch`,
  `Task` where not hidden, `ExitPlanMode` only in plan mode; `agent_prompt.build(cwd, shell, mode, date,
  facts["platform"], facts["git"], None, tools, dirs, facts["memory"])`.

- [ ] **Step 1: Write the failing tests:** `test_resolve_matches_old_rules` and `test_in_git_walks_six_parents` in
  `serve/test_folders.py`; in `serve/test_runner_run.py`: `test_runner_facts_build_the_prompt` (fake facts with
  `cwd = "C:\\Users\\ann\\proj"` (absent on this disk), `platform = "linux"`, shell `"bash 5"`, git true, one memory
  block -> the prompt contains the path, `linux`, `bash 5`, the git line and the block's text; `run.coding ==
  "my_agent"`; `run.ctx.run_key == facts["run"]`); `test_runner_run_has_no_hooks` (a run config with a prompt hook ->
  `run.hooks is None`); `test_runner_run_touches_no_server_disk` (`os.path.isdir`/`realpath` patched to raise -> no
  error); `test_offline_runner_raises` (`RunnerUnavailable`); `test_no_runner_run_unchanged` (same `sa` without runner
  on a temp git folder -> `run.prompt == agent_prompt.build(realpath(tmp), describe(shell), mode, today, sys.platform,
  True, None, tools, [], svc.memory_blocks([realpath(tmp)]))`, `run.coding == "agent"`, `run.ctx.run_key == ""`).
- [ ] **Step 2: Run** `python -m pytest serve/test_folders.py serve/test_runner_run.py serve/test_agent_run.py -q` - Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command plus `serve/test_agent_chat.py serve/test_agent_prompt.py` - Expected: all pass.
- [ ] **Step 5: Commit** `feat(serve): a run on My PC takes its folder, shell, git and notes from the runner (#310 part 2)`;
  BLUEPRINT: "The life of a request" - the facts job.

### Task 4: Per-request routing, names and the remote steps

**Files:**
- Modify: `serve/mcp.py` (`McpHub.with_servers`), `serve/runner_hub.py` (`agent_subset`), `serve/server.py`
  (`_openai` ~6840-6905, `_still_may_run`, `run_with_mcp` ~4901, `GET /agent`), `serve/subagent.py` `execute` (~94)
- Test: `serve/test_runner_routing.py` (HTTP, ScriptedEngine from `serve/test_mcp.py`, `tool_call` from
  `serve/test_harness.py`, FakeRunner)

**Interfaces:**
- Consumes: Task 2 `Runner.servers()`, Task 3 `start_agent_run(..., runner=)`, `AgentRun.hub/coding/runner`.
- Produces:
  - `McpHub.with_servers(extra: dict[str, object], replace: dict[str, object] | None = None) -> McpHub` - a shallow
    view: the same settings and start threads, `servers` = the hub's in order with `replace` swapped in place, then
    `extra`; its own `_routes`. Never `status()`/`close()` on a view.
  - `agent_subset(agent_server, names: tuple[str, ...]) -> object` - name `"agent"`, the hub fields of `AgentServer`,
    `tools` filtered to `names`, `call` delegating.
  - `strata_agent.runner: str` (the request field). In `_openai`: a str `runner` -> `svc.runner_hub.get_for(id,
    self._acting())`; None -> 404 `{"error": {"type": "runner_not_found", "message": "there is no runner with that id
    for this key"}}`; offline -> 409 `{"error": {"type": "runner_offline", "message": "your PC ({m}) is offline:
    start the runner on it and send again"}}`; `RunnerUnavailable` -> the same 409. With a runner: no `_edit_answer`
    gate, no `svc.agent is None` 409; `arun.hub = svc.mcp.with_servers(runner.servers(), replace={"agent":
    agent_subset(svc.agent, ("WebFetch", "WebSearch", "Task"))})` (no `replace` when `svc.agent` is None); the skip
    rules use `arun.hub.servers`; "coding tools alone" keeps `arun.coding` and `"agent"`; R11 skips only
    `svc.mcp.servers` names other than `"agent"`; `run_with_mcp(svc, arun.hub, ...)`.
  - `_still_may_run(arun)`: a runner run passes for any identity whose key still matches and whose runner
    `get_for` still finds; a server run keeps the owner rule.
  - `run_with_mcp`: for a route whose server has `machine == "my"`, the `call` event carries `"machine": "my"`, a
    `{"event": "remote", "id", "state": "sending"}` follows it at once, and the `progress` states `running` /
    `offline` are yielded as `remote` events on the next 0.2 s tick (deduplicated). Other routes: unchanged events.
  - `subagent.execute`: `hub = parent.hub or svc.mcp`; allowed names from `hub.servers[parent.coding].tools` and the
    `"agent"` server's tools; template from `hub` skipping every server but `parent.coding` and `"agent"`;
    `helper.hub, helper.coding, helper.runner, helper.ctx.run_key` copied from the parent.
  - `GET /agent?runner=<id>` -> `{"available": true, "allowed": online, "reason": "" | "your PC is offline",
    "shell", "tools", "machine", "online"}` from the hello lists; another identity's id -> 404.

- [ ] **Step 1: Write the failing tests:** `test_guest_runs_tools_on_own_runner` (guest ann, scripted `Read` call ->
  the FakeRunner gets a `tool` job with `group == "agent"`; the server's `AgentServer.call` is patched to fail the
  test if reached; the answer streams); `test_guest_cannot_route_to_another_runner` (RF 2: ann names bob's id -> 404
  `runner_not_found`, no job at bob's fake); `test_guest_without_runner_still_403` (part 1 rule);
  `test_offline_runner_is_409`; `test_names_and_order` (owner with runner and server MCP fake `fs`: offered tools =
  server MCP names unchanged, then `Read`.. from the runner, `my_fs__x` for the runner's `fs`, `WebFetch` only when web
  access is on, no `Read_2`); `test_call_event_names_machine` (`call` has `server: "my_agent"`, `machine: "my"`, then
  `remote: sending`, then `remote: running`, then `result`); `test_helper_uses_runner_group` (helpers on, scripted
  `Task` -> the helper's `Read` reaches the fake with `helper: true`, `mode: "plan"`);
  `test_default_request_unchanged` (no runner, owner, a server MCP fake: the offered names in order equal
  `svc.mcp.template_tools(exclude=hidden, skip_servers=skip)` names; the `call` event's key set is exactly
  `{"event", "id", "name", "server", "tool", "arguments", "round"}`; no `remote` event in the stream; `GET /agent`
  answers `{"available", "allowed", "reason", "shell", "tools"}` only); `test_get_agent_with_runner`.
- [ ] **Step 2: Run** `python -m pytest serve/test_runner_routing.py -q` - Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `python -m pytest serve/test_runner_routing.py serve/test_mcp.py serve/test_agent_chat.py serve/test_identity_http.py serve/test_server.py -q` - Expected: all pass.
- [ ] **Step 5: Commit** `feat(serve): each request routes to its own runner's groups; My PC tools by plain names (#310 part 2)`;
  BLUEPRINT: the request path (route view), the `strata_agent.runner` field, the `remote` event, the `machine` key.

### Task 5: Page routes for My PC

**Files:**
- Create: `serve/page_routes.py`; Test: `serve/test_page_routes.py`, `serve/test_runner_http.py`
- Modify: `serve/server.py` GET `/agent/folders`, `/agent/git`, `/agent/git/diff`, `/agent/files`, `/agent/mention`,
  `/agent/memory`, `/agent/memory/file` (~5965-6022), POST `/agent/rewind`, `/agent/checkpoints/forget` (~6161),
  GET `/mcp` (~5910)

**Interfaces:**
- Produces:
  - `page_routes.answer(path: str, query: dict[str, list[str]], body: dict | None, *, home: str, memory_on:
    list[str], checkpoints: checkpoints_mod.Checkpoints) -> tuple[int, dict]` - the bodies of the nine routes moved
    here word for word (same answers, same 400/404 texts).
  - Handler `_runner_param(q: dict, body: dict | None) -> Runner | None | bool` - None: no `runner`; the Runner when
    it is the caller's (`get_for(id, self._acting())`); False after answering 404 (`runner_not_found`) or 409
    (`runner_offline`).
  - With a runner: the job `{"kind": "page", "path", "query", "body", "memory_on"}` -> result `{"status", "body"}`,
    sent back as `self._json(status, body)`; no owner gate. Without it: today's owner gate, then `page_routes.answer`
    with the workstation's home and `svc.checkpoints`.
  - `GET /mcp?runner=<id>` -> `{"servers": [...the runner's mcp and skills groups as McpHub.status() lists them,
    names with "my_"], "tools": n, "skills": [...], "machine"}` from the hello lists (no job).

- [ ] **Step 1: Write the failing tests:** `serve/test_page_routes.py`: `test_answers_equal_direct_calls` (on a
  temp git repo: `answer("/agent/git", {"path": [d]}, ...) == (200, gitview.info(d))`; the same for
  `folders.look`, `gitview.diff`, `files.find`, `files.read`, `memory.discover`'s listing, and `checkpoints.preview` /
  `forget`);
  `test_rewind_validation_texts`. `serve/test_runner_http.py`: `test_page_route_forwarded_to_runner` (guest ann,
  `GET /agent/git?path=X&runner=<own>` -> the fake receives `{"kind": "page", "path": "/agent/git"}` and its answer is
  the HTTP answer); `test_page_route_with_another_runner_is_404` (RF 2); `test_page_route_without_runner_stays_owner_only`;
  `test_rewind_with_runner_in_body`; `test_mcp_listing_for_runner`.
- [ ] **Step 2: Run** `python -m pytest serve/test_page_routes.py serve/test_runner_http.py -q` - Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same plus `serve/test_folders.py serve/test_gitview.py serve/test_files.py serve/test_memory.py serve/test_checkpoints.py serve/test_identity_http.py` - Expected: all pass.
- [ ] **Step 5: Commit** `feat(serve): folders, git, files, notes and rewind of My PC through the runner (#310 part 2)`;
  BLUEPRINT: `serve/page_routes.py`, the `runner=` parameter on the nine routes and `/mcp`.

### Task 6: The runner program `python -m serve.runner`

**Files:**
- Create: `serve/runner.py`; Test: `serve/test_runner.py` (in-process server from `serve()` + a real `serve.runner`
  instance on a thread)

**Interfaces:**
- Consumes: Tasks 1-5 wire format; `agent.AgentServer`, `shell.install`/`find_shell`/`describe`, `harness.Importer`,
  `mcp.hub_from_config`, `checkpoints.Checkpoints`, `folders.resolve`/`in_git`, `memory.collect`, `page_routes.answer`.
- Produces:
  - `runner_dir() -> Path`; `load_device(dir: Path) -> dict` (`{"id": 22 chars [A-Za-z0-9_-], "name": machine}`,
    made once).
  - `class Client(server: str, key: str, device: dict)`: `hello(info: dict) -> dict`, `next(rid, instance, wait) ->
    list[dict]`, `reply(rid, instance, msg: dict) -> None`; headers `Authorization: Bearer`, `X-Strata-Device`,
    `X-Strata-Device-Name: Strata runner on <machine>`; raises `Refused(kind: str, message: str)` for 401,
    `device_refused`, `runner_replaced`, `runner_protocol`; `Pending(code)` for `device_pending`; `Gone` for 404.
  - `class Executor(home: Path, config_path: Path, data_dir: Path)`: its own `AgentServer` with `shell.install` only
    (no web, no Task), `Importer`, `hub_from_config(cfg, None, builtins=importer.builtins())`, `Checkpoints(data_dir /
    "checkpoints")`; `info() -> dict`, `groups() -> list[dict]`, `skills() -> list[dict]`;
    `handle(msg: dict, link: JobLink) -> dict` for `run.start` / `tool` / `page`. Runs keep `{run_key: (Policy,
    Scope | None)}`, LRU 50; Policy `protected = [config_path, data_dir / "device.json"]`; a `helper` call uses a copy
    of the run's policy with `mode` from the job and `question` answering `{"answers": None}`.
  - `class JobLink(job: str, send: Callable[[dict], None])`: `ask(card) -> str`, `question(qs) -> dict | str`,
    `emit(event) -> None`, `cancel: threading.Event`; `ask`/`question` send the message and wait on the job's inbox for
    `answer` or `cancel` ("cancelled").
  - `main(argv: list[str] | None = None) -> int`: `--server URL --key KEY [--device-id ID] [--config PATH]`; prints
    once on connect `[runner] connected to <url> as <who>; this PC's runner id is <id> - paste it in the web app under
    Settings > This browser's runner`; one poll thread, one thread per job (at most 16 at once, the rest queued);
    re-hello when its groups change (an MCP server started) and on 404; backoff 1, 2, 4 ... 30 s on network errors;
    `Pending` -> print the code once, retry every 10 s; `Refused` -> print the reason and exit 3; Ctrl+C -> exit 0.

- [ ] **Step 1: Write the failing tests:** `test_end_to_end_write_with_approval` (team mode, guest ann; temp project
  folder; ScriptedEngine writes a `Write` call outside the auto-allowed rules; the page's `POST /agent/permission`
  allows it; the file exists in the temp folder with the content; the stream shows `machine: "my"`);
  `test_runner_page_routes` (`/agent/folders?runner=` lists the temp folder; `/agent/files` finds a file);
  `test_runner_mcp_group` (a runner config with `serve/mcp_fake_server.py` as a stdio server -> `my_fake__echo` is
  offered and answers); `test_runner_exits_on_401` (RF 4: revoke ann -> `main` returns 3 within 2 polls);
  `test_runner_waits_for_device_approval` (pending -> approved by the owner -> connects); `test_rehello_after_404`
  (server restarted -> same runner id); `test_rewind_on_runner` (a Write with a checkpoint, then rewind via the page
  route puts the file back).
- [ ] **Step 2: Run** `python -m pytest serve/test_runner.py -q` - Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `python -m pytest serve -q` - Expected: all pass (full suite once here).
- [ ] **Step 5: Commit** `feat(serve): python -m serve.runner - the My PC executor (#310 part 2)`; BLUEPRINT: the
  runner process, its files and flags, `STRATA_RUNNER_KEY`.

### Task 7: Web app - the runner id, the machine of a project, the two sections

**Files:**
- Create: `serve/ui/src/lib/runner.ts` + `runner.test.ts`, `serve/ui/src/components/RunnerSettings.tsx`
- Modify: `lib/sessions.ts` (`Project.machine`, `addProject`, new `projectMachine`), `lib/agent.ts` (`agentRequest`),
  `lib/api.ts` (`getAgent`, `getFolders`, `getMcp`), `lib/git.ts`, `lib/memory.ts`, `lib/mention.ts`, `lib/rewind.ts`
  (each fetcher takes `runner?: string`), `lib/chat.ts` (~756 passes `ctx.runner`), `lib/panel.ts:40`,
  `pages/chat/Chat.tsx` (~97, ~258, ~268), `pages/chat/Messages.tsx:117`, `pages/chat/AgentCall.tsx`,
  `components/NewProjectDialog.tsx`, `components/AgentControls.tsx`, `components/FoldersEditor.tsx` (via its `api`),
  `components/McpSettings.tsx`, `components/ImportSettings.tsx`, `pages/Settings.tsx` (section list),
  `lib/i18n.ts` (Thai for every new string), `serve/ui/dist/`

**Interfaces:**
- Produces: `runnerId(): string` / `setRunnerId(id: string)` (store key `runner`, i.e. localStorage
  `strata.runner`, validated `/^r[0-9a-f]{16}$/`); `withRunner(path: string, runner?: string): string` (adds
  `runner=` to a query); `machineOf(call: { server?: string; machine?: string }): "my" | "server"` (`machine === "my"`
  -> my); `Project.machine?: "my"` (absent = Server PC, so old projects are unchanged); `addProject(index, name, id,
  folders, machine?: "my")`; `projectMachine(index, project?: string): "my" | "server"`; `agentRequest(s, info,
  folders, session, rules, checkpoint?, runner?: string)` adds `runner` to `strata_agent` when given;
  `getAgent(runner?: string)`, `getMcp(runner?: string)`.
- Rules: the machine choice "My PC" / "Server PC" shows in NewProjectDialog and AgentControls only when this page has a
  runner id; "Server PC" only when `GET /agent` says `allowed`; a new project defaults to My PC when the runner is
  online. A chat in no project runs on the Server PC for an owner, else on My PC. Chat.tsx keeps both AgentInfos and
  picks by the project's machine. Tool cards: `server` `agent` or `my_agent` -> AgentCall; the card shows "My PC" or
  "Server PC" when a runner id is set. Settings: "This browser's runner" (paste the id, shows machine and online via
  `GET /agent?runner=`); MCP and Skills settings get a read-only "My PC" section from `GET /mcp?runner=` above the
  "Server PC" section (read-only for a guest, as part 1 made it).

- [ ] **Step 1: Write the failing tests** (`bun test`): `runner.test.ts` - `runnerId()` round trip and rejects
  `"x"`; `withRunner("agent/git?path=a", "r0123456789abcdef")` ends with `&runner=r0123456789abcdef`, no runner ->
  unchanged; `machineOf({server: "my_agent", machine: "my"}) === "my"`, `machineOf({server: "agent"}) === "server"`.
  `sessions.test.ts`: a project added with `"my"` keeps it through save/load; an old project loads as `"server"`.
  `agent.test.ts`: `agentRequest(..., "r0123456789abcdef").strata_agent.runner` is the id; without it the object
  equals today's. `rewind.test.ts`: the body carries `runner` when given.
- [ ] **Step 2: Run** `cd serve/ui && bun test` - Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `cd serve/ui && bun test && bun run typecheck && bun run build`, then `python -m pytest serve/test_ui.py -q` - Expected: all pass.
- [ ] **Step 5: Commit** `feat(ui): projects on My PC or the Server PC; this browser's runner; two sections (#310 part 2)` with `dist/`;
  BLUEPRINT: the `strata.runner` value and the web app rows.

### Task 8: Status words for remote steps, and runners in Monitor

**Files:**
- Modify: `serve/ui/src/lib/status.ts`, `status.test.ts`, `lib/chat.ts` (`onTool`, `ToolEvent`, `ToolCall`),
  `pages/chat/AgentCall.tsx` (offline mark), `pages/Live.tsx`, `lib/keys.ts` (`getRunners`), `lib/i18n.ts`, `dist/`

**Interfaces:**
- Produces: `ToolCall.machine?: "my"`, `ToolCall.remote?: "sending" | "running" | "offline"`; `onTool` sets `machine`
  from the `call` event and `remote` from `remote` events; `StatusTool` gains `machine?`, `remote?`; `StatusKind` gains
  `"sending" | "offline"`; `getRunners(): Promise<RunnerRow[] | null>` (`GET /runners`).
- Exact words (English / Thai): `Sending the call to your PC…` / `กำลังส่งคำสั่งไปที่เครื่องของคุณ…`;
  `Running {tool} on your PC…` / `กำลังรัน {tool} บนเครื่องของคุณ…`; `Your PC is offline…` /
  `เครื่องของคุณออฟไลน์อยู่…`; Live: `Runners: {list}` / `เครื่องที่เชื่อมต่อ: {list}`, each `name (machine)` with
  `online` / `offline`.
- Order inside `agentStatus` for the running call: `remote === "offline"` -> offline; `"sending"` -> sending;
  `machine === "my"` and running -> "Running {tool} on your PC…"; asking, hook and judging keep priority as today.

- [ ] **Step 1: Write the failing tests:** `status.test.ts` - a running `Bash` with `remote: "sending"` -> kind
  `sending`, label `Sending the call to your PC…`; with `remote: "running", machine: "my"` -> `Running Bash on your
  PC…`; `remote: "offline"` -> `Your PC is offline…`; asking still wins over sending; a call without `machine` -> the
  words of today. `chat.test.ts` - `call` then `remote` events set `machine` and `remote` on the ToolCall.
- [ ] **Step 2: Run** `cd serve/ui && bun test` - Expected: FAIL.
- [ ] **Step 3: Implement**; Live shows the runners line under the holder/waiting line, polling `GET /runners` with the
  page's existing Monitor interval.
- [ ] **Step 4: Run** `cd serve/ui && bun test && bun run typecheck && bun run build`, then `python -m pytest serve/test_ui.py -q` - Expected: all pass.
- [ ] **Step 5: Commit** `feat(ui): the status says when a call travels to your PC; runners in Monitor (#310 part 2)` with `dist/`.

### Task 9: The latency of a runner call, end to end over Radmin, and the record

**Files:**
- Modify: `tests/xeno/perf/timeline.py` (`load` tolerates a missing engine file; new `runner_report`; printed by
  `main` as "runner calls"), `tests/xeno/test_timeline.py`, `docs/BLUEPRINT.md` (revision row)

**Interfaces:**
- Produces: `runner_report(tl: Timeline) -> dict` - `{"calls": n, "total_ms": {"median", "p90", "max"}, "wait_ms":
  {...}, "exec_ms": {...}, "network_ms": {...}}` where `wait` = the `runner.call wait` span, `exec` = its `a`, and
  `network` = `runner.call` - `wait` - `exec`.

- [ ] **Step 1: Write the failing tests** in `tests/xeno/test_timeline.py`: `test_runner_report` (three synthetic
  `runner.call` + `runner.call wait` pairs -> the medians exact); `test_load_server_file_only` (only
  `<file>.server.json` exists -> loads).
- [ ] **Step 2: Run** `python -m pytest tests/xeno/test_timeline.py -q` - Expected: FAIL. **Step 3: Implement.**
  **Step 4: Run** again - Expected: PASS.
- [ ] **Step 5:** Start a test server from the branch in the background with no window: `STRATA_TIMELINE=<scratch>/rt.json
  python serve/server.py --engine mock --port 8099 --host 0.0.0.0 --config <temp run config naming a temp keys_file>
  --script <call> --script <call> ... (20 times) --script "</think>\n\ndone"` - one request whose 20 rounds each run
  one Bash call (`AGENT_ROUNDS = 100` allows it; `<call>` is the text of `serve/test_harness.tool_call("Bash",
  command="echo hi")`; MockEngine gives each round the next script and repeats the last); make a guest key; start `python -m serve.runner --server
  http://26.33.142.160:8099 --key <guest>` in the background (this PC, through its Radmin address); approve both
  devices; send that one chat request with `strata_agent.runner` and `allow: ["Bash(echo:*)"]`, so no prompt waits.
- [ ] **Step 6:** `python tests/xeno/perf/timeline.py <scratch>/rt.json` - record `runner_report`: median and p90 of
  total, pickup wait, exec and network per call. Expected: every call ok; a wait median well under 1 s (a held poll
  answers at once).
- [ ] **Step 7:** Stop both processes (`taskkill /T` each tree; check port 8099 is free).
- [ ] **Step 8:** Comment on #310 (English + Thai, `--repo xenodeve/Strata-xeno`): what landed, the commits, the test
  counts, the latency table with how it was measured (Radmin, same PC, mock engine, 20 calls), that a tunnel run and
  the two-engineer prompt-cache contention run are still to do, and that deploying is the developer's step.
- [ ] **Step 9: Commit** `docs: part 2 record - runner call latency over Radmin (#310)`; BLUEPRINT revision row
  ("Not yet in the baseline" until deployed).

---

## Self-review

- Spec coverage: the channel (T1-T2), routing and names (T4), start_agent_run moves (T3), page routes (T5), the
  runner (T6), the web app (T7-T8), the security summary (T1, T4, T5 tests), the latency span and measurement (T2,
  T9), revoke and device decisions ending runners (T1), subagent's coding group (T4). Left for part 3: the window,
  tray, DPAPI, the Desktop setting `strata.runner` itself.
- Decisions this plan makes beyond the spec: the stable runner id formula; one process per runner id (newer instance
  wins); a revoke ends a key's runners, a demotion to guest keeps them; `my_agent` as the coding group's name and the `machine` key on events; the server-side `agent` subset
  `WebFetch`/`WebSearch`/`Task` exempt from R11 in runner runs; asks served inline on the waiting call's thread; My PC
  MCP/skills settings read-only in v1 (the runner's `runner.json` is edited by hand until part 3 gives it an editor);
  `/name` skill invocation stays server skills only (the model can still call `my_skills__use_skill`); runner groups
  are offered only with `strata_agent` (a chat with the coding tools switched off sees no My PC tools).
