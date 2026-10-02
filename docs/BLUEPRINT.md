# Strata-xeno system blueprint

How the engine and the server fit together: the processes, the life of a request, the memory hierarchy, the
checkpoints, every configuration surface and the diagnostics. Read it before changing anything structural; a new
engineer or an upstream reviewer should be able to start here.

**Keep this file current.** The rules are in [Keeping this file current](#keeping-this-file-current) below and in
`AGENTS.md`. A blueprint that describes last month's code is an instrument that returns a plausible wrong answer.

## Baseline and revision log

The sections below describe the code at the **baseline**. `file:line` references are at that commit unless a line
names another. Changes that are on other branches and not yet in the baseline are listed in
[Not yet in the baseline](#not-yet-in-the-baseline).

| date | baseline | what changed in this file | verified by |
|---|---|---|---|
| 2026-09-30 | `5c51574` on `xeno/exp-upstream-0.1.26-dyn` (upstream 0.1.26 + our dynamic experts + the #48 fix; the engine the D2x server runs on :8091) | extracted from `docs/reports/2026-09-30-fork-delta-and-blueprint.md` §5 (`534ffd3`) | two adversarial verifiers and a fix pass (that report's §9) |
| 2026-09-30 | the merge of `xeno/claude-merge-0.1.20` (`f0f5d7a`) into `xeno/exp-upstream-0.1.26-dyn` (#49 S7 prerequisite) | the line now carries every row of "Not yet in the baseline"; the rows stay listed until a build of this merge re-checks their `file:line` | not built yet; `pytest serve` 102 passed |
| 2026-10-01 | unchanged (the 0.1.30 merge `eb2ec1f` is not served yet) | the merge's row under "Not yet in the baseline"; its record is `docs/reports/2026-10-01-merge-upstream-0.1.30.md` | serve ABBA 20/20 identical outputs, `check_cache_slots` PASS (#56) |
| 2026-10-01 | unchanged (branch `xeno/ui-next-app-s0-s7`, not served yet) | the next web app, request history, per-card telemetry, the GGUF model info, the STATS line (UI S0-S7); rows in the file table and in the protocol section; record: `docs/reports/2026-10-01-ui-next-app.md` | serve tests and `bun test`; **the rebuilt engine has not run** |

**Labels.** `UPSTREAM` means byte-identical to upstream `4c68013`. `XENO` means a file new in our delta. `MIXED` means
an upstream file we modified. "Static reading" marks a statement from the code that no run has confirmed.

**Branches.** The baseline engine is the merged one (0.1.26-based). The daily branch `xeno/claude-merge-0.1.20` is
0.1.20-based: its engine differs as the fork-delta report §4 records, and its `serve/` has the work listed under
[Not yet in the baseline](#not-yet-in-the-baseline).

`UPSTREAM` means byte-identical to `4c68013`. `XENO` means a new file in our delta (145 files). `MIXED` means an
upstream file we modified (45 files). Source: `git diff --name-status 4c68013 5c51574`.

## 1. The model served (`include/strata/core/layout.hpp:27-58`)

- **Model:** Qwen3.8-Flash-Next, 48 layers.
- **Attention:** every 4th layer is full attention ("QSA", 12 layers); the other 36 are Gated-DeltaNet ("GDN").
- **Width:** `n_embd` 2560, hyper-connection `hc` = 4.
- **MoE:** on every layer, 512 experts, top-10, `n_ff` 640.
- **PLE:** a per-layer n-gram embedding table in GGUF shard 2 on SSD.
- **Drafter:** one MTP draft layer.
- **Sizes:** an expert blob is 1,382,400 B (`generate.cpp:652`); the Q2_0 arena is 34.0 GB (`setup.py`).

## 2. Process picture

```
 Claude Code / OpenAI client / web app
          |  HTTP (SSE)                          D2x: port 8091, launched by D:/Github/Strata/start-flash-next.ps1
          v                                      (sets CUDA_VISIBLE_DEVICES=1,0 -> CUDA0 = 5060 Ti, CUDA1 = 4070 SUPER)
 +----------------------- serve/server.py (Python, one process) [MIXED] ------------------+
 | Handler -> frontend.py (API -> chat messages, Qwen XML tool-call parser, template)      |
 |         -> Service.prepare (Jinja template + BPE tokenizer, tools/strata_tokenizer.py)  |
 |         -> Service.run (FIFO lock, LoopGuard [XENO], StopSequenceFilter [XENO], Detok)  |
 |         -> StrataEngine: stdin/stdout text protocol (GEN/GENI/STOP/QUIT <-> T/PP/DONE)  |
 |   optional: Vision, McpHub (serve/mcp.py), Telemetry, timeline [XENO]                   |
 +---------------------------------------+------------------------------------------------+
                                         | pipes (one resident sequence)
 +---------------------- strata.exe --serve (src/program/generate.cpp) [MIXED] -----------+
 | boot: weights -> session -> PLE -> MTP -> host arena -> CPU pool -> GPU0 cache (fill)  |
 |       -> 4070 tier (fill) [XENO] -> host load_rest [XENO] -> tail file [XENO] -> ...   |
 | per request: resume (live / checkpoint) -> prompt parts (windows | batched: borrow,    |
 |       split [XENO], wave [XENO]) -> checkpoints -> refill lent slots -> verify-window  |
 |       decode loop (MTP / suffix drafts) -> adaptive swaps between rounds -> DONE       |
 +------+-----------------------+----------------------+---------------------+----------+
        v                       v                      v                     v
  GPU0 5060 Ti (x4)      GPU1 4070 SUPER (x16)   CPU pool (13 workers   host RAM arena (pageable,
  trunk, KV, MTP,        exclusive expert tier,  + host thread,         reserve-only; host-owned
  expert cache           split-prefill experts   AVX-VNNI Q2 rows)      experts) + NVMe (pack, PLE,
                         [XENO]                  [MIXED]                tail file)
```

## 3. File responsibilities

| path | responsibility | owner |
|---|---|---|
| `src/program/generate.cpp` (6,969 lines) | engine driver: options, boot and placement, `--serve` loop, generate loop, adaptive tiers, checkpoints | MIXED +1,964/−111 |
| `src/prefill/prefill.cpp` (3,231) | batched prompt path: GEMMs, MMQ experts, streamed ring, stager, split, wave, Dm frontier, E-9 drafter K/V | MIXED +1,548/−205 |
| `src/prefill/frontier.cu` + `.hpp` | #41 Dm frontier | XENO (`78f373e`) |
| `src/prefill/moe_mmq.cu` + `.hpp` | llama.cpp MMQ for prompt experts; nsm=1 (`:153-154`) | MIXED |
| `src/prefill/kernels.cu`, `gemm.cu` | prompt kernels; `moe_routed_sum`, `moe_shared_finish`; `warm_cublas` | MIXED |
| `src/core/verify.cpp` + `.hpp` | speculative verify window: T tokens × 48 layers in one captured graph, bit-exact to greedy (`verify.hpp:1-20`) | MIXED |
| `src/core/mtp.cpp` + `.hpp` | MTP draft layer; `on_draft` hook | MIXED |
| `src/core/expert_source.cpp` + `.hpp` | expert sources; `expert_pool_dispatch_multi`; placement-first reads; exclusive release/recommit; NVMe tier | MIXED +562/−5 |
| `src/core/pinned.cu` + `.hpp` | host arena: VirtualAlloc, large pages, cudaHostRegister; `reserve_only`, `decommit_interior`, `commit_interior` | MIXED |
| `src/core/secondary_{arena,runner,vram}.cpp`, `secondary_{budget,profile}.hpp` | 4070 tier | XENO |
| `src/core/expert_cache.cpp` | GPU0 expert cache | UPSTREAM |
| `src/core/remote_experts.cpp` | upstream helper-GPU caches | MIXED (merge-only, 1 line) |
| `src/core/device.cu`, `device_main.cpp` | `strata-device`; our 4070 VRAM probe | MIXED |
| `src/kernels/cpu/q2_avx2.cpp`, `expert_layout.cpp` | Q2_0 rows; AVX-VNNI arm; ISA dispatch | MIXED (`f679806`) |
| `src/kernels/cpu/pool.cpp` + `.hpp` | CPU expert pool; P-cores first; priority; `rest()` | MIXED |
| `src/kernels/cuda/iq_kernels.cu` | native expert GPU kernels; CPU-order Q2_0 rows K0-K4 (`:332`) | MIXED |
| `src/kernels/cuda/s2_expert_grouped.cu` | grouped hit kernels; `moe_hit_merge_mapped` (`:764`) | MIXED |
| `src/kernels/ngram.cpp` | PLE table, `prefetch` (`:298`) | MIXED (`7134ce9`) |
| `src/platform/timeline.cpp`, `include/strata/timeline*.hpp` | #33 timeline | XENO |
| `src/platform/direct_file.cpp` | unbuffered overlapped reads (C-4 pool) | UPSTREAM |
| `serve/server.py` (1,903) | HTTP, engine client, OpenAI/Anthropic/MCP | MIXED |
| `serve/frontend.py` | messages, tool-call parser; billing strip; document blocks | MIXED |
| `serve/loop_guard.py`, `serve/pdf_blocks.py`, `serve/timeline.py` | loop guard; PDFs; server timeline lanes | XENO |
| `serve/history.py` | request history on disk: `requests-YYYY-MM.jsonl` summaries (kept), `detail/<id>.json.gz` (2 GB cap, oldest first); run-config key `"history"` `{enabled, dir, detail_cap_gb}`; `GET /metrics/requests[/<id>]` | XENO |
| `serve/gguf_info.py` | the model's name, source, variant and the quantization of each part (experts, attention, embeddings, PLE table ...) read from the GGUF headers named in the run config's `--native` / `--ple-gguf` (header only; bytes per part from the tensor offsets, bits per weight from the dims); `model_info` in `GET /metrics` | XENO |
| `serve/mcp_admin.py` (#79) | setting up MCP servers from the web app: `GET /mcp/config` (the servers of the run config, and of `--mcp-config` read-only, with their state and tools; the values of `env` and `headers` are a mask `********`, never the secret; the limits; `editable` and why not) and `POST /mcp/config` (`{"servers": {name: entry}, "settings": {timeout_s, max_result_chars, max_rounds}}`, a server not listed is deleted). Changing the list starts programs on this PC, so it is allowed only with an API key set and sent, or from this PC itself (loopback client address and a loopback `Host`) and from the app's own page (JSON, no foreign Origin); otherwise 403, and the page is read-only and says why; such a caller gets only each server's name, kind, state, tools and error from `GET /mcp/config` (no command, arguments, folder, address, env or headers: a token can sit in an argument or an address). Names are `[A-Za-z0-9_.-]{1,40}`; a program (`command`, `args` at most 64, `env`, `cwd`) or an address (`url` http/https, `headers`), never both; limits inside a range (timeout 1-600 s, result 1000-500000 characters, rounds 1-30). A mask sent back keeps the stored value, a changed value replaces it. The run config keeps its other keys, the key already in use (`mcp_servers` or `mcpServers`) is the one written, the original is copied once to `<config>.bak-mcp`, and the new file is moved over it from a temporary file. The hub is closed and started again from the file (`reload_hub`), so no restart; a tool call running at that moment ends with an error text. Needs `--config` (without it: read-only, 409 on POST). A chat request may carry `"strata_mcp_off": [server names]` next to `"strata_mcp": true`: those servers' tools are left out of that request (`McpHub.template_tools(skip_servers=...)`; anything but a list of names is ignored); the web app sends it from the per-server switches of the + menu's MCP list (kept in the browser, `mcpOff` of the sampling settings) | XENO |
| `serve/agent.py`, `serve/permissions.py`, `serve/shell.py`, `serve/agent_run.py`, `serve/judge.py`, `serve/agent_prompt.py` (#96) | the chat's coding tools, a port of Claude Code's: `AgentServer` is a built-in server of the MCP hub (like `skills`) with Claude Code's names and parameters (Read, Write, Edit, Glob, Grep, TodoWrite, ExitPlanMode; Bash, BashOutput, KillShell when there is a shell: Git Bash on Windows, else bash/PowerShell, `STRATA_SHELL` overrides), shown to the model without a prefix (`plain_names`) and not listed with the MCP servers (`hidden`). Every call goes through the gate in `AgentServer.call` (no context, no call): `permissions.decide(tool, args, Policy)` says allow/ask/deny with Claude Code's rule syntax (`Tool(specifier)`, `Bash(git commit:*)`, `Read(src/**)`; deny > allow > defaults): inside the chat's project folder files are free (not secrets, not `.git/` for writing), outside it and with no folder anything asks, a command asks unless it is a plain read-only one (a small shell reader finds chains, substitutions, redirections), mode `plan` changes nothing (ExitPlanMode asks for approval), mode `auto` marks what `serve/judge.py` may settle (never a dangerous command, a secret or `.git`), an unknown mode is the default one - there is no mode that switches the questions off. `Write`/`Edit` refuse a file not read in this chat or changed since (per-chat state keyed by `strata_agent.session`), `Edit` keeps CRLF, `Grep` refuses nested repetition (Python `re` cannot be interrupted), `Bash` runs a fresh shell in the folder, kills the whole process tree on timeout/cancel, keeps output to 30,000 characters, gives the command no `STRATA_*` variable. A request with `"strata_agent": {"cwd", "mode", "session", "allow", "deny"}` (anything odd ignored; JSON + own page + this PC or the API key, else 403; `"agent": false` in the run config: 409) gets the tools and the rules for the AI (`agent_prompt.build`, with the folder's `CLAUDE.md`/`AGENTS.md`); `run_with_mcp` runs them up to 100 rounds, drains `AgentRun` events into the stream (`permission` cards, `judging`/`judged`, `todos`, `mode`) while the tool waits for `POST /agent/permission {id, decision: allow|allow_chat|deny}` (`Broker`; same guard; 404 for an id not waiting; no answer in 10 minutes or a cancel is a refusal). Auto mode's judge is `Service.side_request` (not streamed, no tools, thinking off, `<severity>N</severity>` written by `forced_opening`, the existing side-request machinery) over the request goal and the call only, never tool results: severity 1-2 runs, 3 asks, 4-5 blocks, anything unclear asks. `GET /agent` says whether they are on, the shell, and whether this caller may use them | XENO |
| `serve/harness.py`, `serve/skills.py` (#94) | the skills and MCP servers of the other coding apps on this PC. `harness.scan(home, appdata)` reads, never writes, Claude Code (`~/.claude/skills`, the skills of the plugins `settings.json` enables via `plugins/installed_plugins.json`, `~/.claude.json` `mcpServers`), Claude Desktop, Codex (`~/.codex/skills`, `config.toml` `[mcp_servers]`), the shared `~/.agents/skills`, Antigravity (`~/.gemini/antigravity/skills`; its `mcp_config.json` if there is one, not seen on the PC it was written on), Gemini CLI (`~/.gemini/skills`, `settings.json`) and Cursor; a missing or broken file is an entry in `errors`, never an exception. The folder is `STRATA_HOME` (env) or the user's own, and tests and mock servers point it at a fake one. **Skills are on by default** at three levels, kept as off-lists in the run config: `"import": {"skills": {"enabled": true, "harness_off": [app ids], "off": {app id: [skill names]}}}`; an item found on a later rescan is on; the same skill name in several apps is one (the first app in the list wins, a copy that is off lets the next one in). They are the built-in server `skills` of the MCP hub (`SkillsServer`, no process): tools `find_skills(query)`, `use_skill(name)` (SKILL.md and the list of bundled files), `read_skill_file(name, path)`; the catalog is in the prompt only up to 15 skills, otherwise the model searches (on the PC it was written on: 462 skills in use, the three tool descriptions 466 characters). `read_skill_file` is fenced: relative path, no `..`, no hidden part, not a key by its name, the real path (links and junctions resolved) inside the skill's real folder, text under 100 KB. **MCP servers are only listed**, by app, and imported by a click: `POST /import {"import_mcp": {"harness", "name"}}` copies the entry (with its secrets, on the server) into the run config's `mcp_servers` (a taken name gets the app's name added; the same server already there, by command and arguments or by address, is `already`; SSE-only and no-command ones are `unsupported`; one off in its own app is imported with `disabled`), and the hub restarts; nothing is started by discovery. `GET /import` (`?rescan=1` reads the folders again; the result is cached) and `POST /import` (`{"skills": {...}}` strict, `{"import_mcp"}`, `{"rescan": true}`) follow the guard of `/mcp/config`: an API key set and sent, or this PC itself and the app's own page, else 403; without `--config` 409; a caller who may not change anything sees names, not commands, arguments, addresses, env or headers; no secret is in any response. a request with `"strata_skill": "<name>"` (the web app's `/name`; only a text counts) has the server call the skills server's `use_skill` and put its text as a system message in front of the last user message (`skills.invoked`, `skills.put_before_last_user`), 400 with the name when there is no such skill in use, 403 from a foreign Origin; `McpHub(builtins=...)` and `hub_from_config(..., builtins=...)` take the built-in servers; `mcp_admin.update_config` is the shared careful write | XENO |
| `serve/storage.py`, `serve/telemetry.py` | the disks behind the model (model, bus, media, which disk holds it; per-disk MB/s and OS read latency with psutil) and every GPU on its own (clocks, throttle reasons, PCIe link, a history per card); the cards are sampled at 5 Hz while a request runs, the history keeps 1 Hz | XENO |
| `serve/ui/**`, `serve/ui_hash.py` | the next web app (React + Vite + Tailwind, built with Bun; `dist/` committed, stale-checked by `serve/test_ui.py`); served at `/` (and at `/next/`); the classic app (`serve/web/`, untouched) at `/classic/`, and at `/` only when the run config has `"ui": "classic"` (`ui_choice`); the browser checks (`serve/ui/e2e/`, `bun run e2e`, Playwright as a dev dependency, against two mock servers on ports 18771 and 18772, not part of `bun test`: overflow and errors on every page in both languages at 1280 and 390 px, follow-scroll, opening the thinking, rewrite and take back, the orbs moving, the language switch, the thinking levels); `serve/ui/dev/mock_server.py` serves the app with no model or GPU for UI work (`STRATA_MOCK_LONG=1`: a long, slow answer, to check that the chat follows it; `STRATA_MOCK_MCP_CONFIG=file.json`: starts the MCP servers of that run config so the Settings page's MCP setup works against them, used by the `mcpset` browser check on a third mock, port 18773; `STRATA_HOME=folder`: the "other apps" Settings > Import reads are that fake folder, used by the `import` check on a fourth mock, port 18774, with a fake home the runner builds). Settings > MCP servers (#79, `McpSettings`): list with state and tools, add / edit / turn off / delete, paste a Claude Desktop block, the limits; read-only with the reason when `GET /mcp/config` says the caller may not change them. Pages (a prompt is shown as tokens read = prefill, and tokens from the conversation cache, never added together): Chat, Dashboard (one-screen overview of the others), Live, Requests, Hardware, Settings (a branched menu, `BranchedMenu`, of three sections: General = status marks, API key; MCP tools = servers, limits; Import = skills, MCP servers (#94: `ImportSettings`; skills with a switch for the whole import, each app and each skill, a filter and Rescan; MCP servers of the other apps listed by app with an Import button each, which is the only thing that starts one); the address says which topic shows, `#/settings/<status-marks|api-key|mcp-servers|mcp-limits|import-skills|import-mcp>`, an unknown one shows the first), About (the model and the server); The chat page has a sidebar (`Sidebar`, drawn with the same `BranchedMenu` as Settings: a project or a day is a section, a conversation a branch, the line goes to the open one; a drawer on a phone, hideable on a desktop, remembered in `strata.sidebar`) with Recents (the conversations that are in no project, newest first, as the branches of one section called Recents - not split by day) and Projects (named folders that group them); the sidebar's width stretches and shrinks (the list is taken away once it is a rail) and the phone's drawer slides; the conversations are kept in this browser only (#92, `lib/sessions.ts`, pure operations over `store`): `strata.chats` is the index (`active`, `items` with id, title, time, project, named; `projects`), `strata.chat` stays the OPEN conversation's messages (the key the classic app reads) and `strata.chat.<id>` holds the others (a conversation moves between the two when it is opened or left); a conversation is added with its first prompt (title = that prompt cut to about 40 characters, no model call) and one that is emptied is dropped; a conversation that existed at `strata.chat` before the index is migrated as the first item; a write the browser refuses (storage full) is reported once through `ChatController.onError`; switching is refused while an answer is written; the address `#/chat/<id>` opens one (an unknown id shows a new chat, the address is replaced, not added to) and the top navigation's Chat link goes back to the open one. Code in an answer is coloured like an IDE (`lib/highlight.ts`, highlight.js core with 27 languages and the usual short names; a block with no or an unknown language stays plain, no guessing; a block over 40,000 characters, or over 8,000 while it is still being written, stays plain; the result is the library's own spans around escaped text, tested to add no markup and to leave the text unchanged; colours are `--syn-*` tokens: VS Code Light+ in the light theme, Monokai - the theme of Claude Code's own code - in the dark one). The chat's composer is `PromptBar` (adapted from React Bits, see `serve/ui/REFERENCES.md`): the field, and one bar with the `+` menu (attach, new chat, save as Markdown), the thinking-effort slider (`reasoning_effort`) and Sampling. Security of the monitor (#71, review of 2026-10-01): with no API key a request whose `Host` is not this PC (an IP address, `localhost`, a name with no dot, `.local`, or a name in the run config's `allowed_hosts`) is refused with 421, which closes DNS rebinding; `POST /metrics/keep` needs JSON and the app's own origin, `/load` and `/unload` refuse a foreign origin; the new app's page is sent with `X-Frame-Options: DENY`, `frame-ancestors 'none'` and `nosniff`; history files are owner-only off Windows. The thinking levels offered are the model's own: the server probes its chat template at start (`ChatTemplate.efforts`) and lists them in `/metrics` `engine.efforts` (the shipped template: Off, Low, Medium, XHigh; "high" is XHigh's alias and is not offered), the chat offers those and maps a saved level onto one that exists (`serve/ui/src/lib/effort.ts`). Status marks, four choices (it starts as the second): the orbs; orbs with the loading style (the lattice of dots by the thinking, LatticeLoader as the app had it; the lattice also marks the thinking in the loading-only mode); the loading style alone (a lattice or matrix pattern for each of the nine forms, `components/loader.tsx`, `matrix.tsx`); or avatars instead (`serve/ui/src/lib/avatar.ts`, bot-avatars vendored in `src/vendor/bot-avatars`, loaded only when chosen; the choice is `localStorage` `strata.avatar` and which avatar `strata.avatar.type` (by status, one of eighteen, or random), in About; and in a menu from the header: one list, each way with a look at three of its marks and a line on what it is, arrow keys to move through it). Languages: English and Thai (`serve/ui/src/lib/i18n.ts`: `t("English text")` is the English and its key, the Thai is in `src/i18n/th-*.ts`, a test fails when a string in the source has no Thai entry; the choice is `localStorage` `strata.lang`, default Thai when the browser's language is Thai; a switch in the header; Thai text gets room for its marks under `:lang(th)`). Design system: one metrics poll for the whole app (`MetricsProvider`), orbs (`serve/ui/src/vendor/thinking-orbs`, MIT, local `fps` and `scale` patches; the empty chat's orb is 160 px and shows a random form every 5 s while the server is idle) that show what the server, a GPU, a reply, a tool call or a loading page is doing (idle = the searching globe turning slowly, every display frame for the server's orbs and 30 fps for the small GPU ones; a change of form is a 450 ms dissolve), with a shimmering status line, the agent's thinking as a `Thought` line (React Bits ThoughtLine) beside a `Lattice` (LatticeLoader) that runs the pattern of what the agent is doing and ends as a tick or a cross (`serve/ui/src/components/thought.tsx`), the + menu also has an MCP tools row (how many tools from how many servers, on or off, switched there; disabled with the reason when no server is set up),  a sent prompt can be rewritten (Edit: it and what came after are replaced) or, the last one, taken back (Undo: the prompt returns to the composer, its answer goes), both off while an answer is written (`ChatController.edit` / `undoLast`); the speed at which a prompt is read under that prompt in the chat (`serve/ui/src/lib/prefill.ts`: the mean over the last second from `/metrics` `live.prompt_read` while it reads, the engine's mean the moment it is read, then the final chunk's `timings` with tokens read and cached; an option in Sampling; the dev mock reads a prompt at `STRATA_MOCK_PREFILL_TPS`, default 300), the server's status is named by its phase (`live.phase`: reading the prompt, thinking, answering, writing a tool call; the orb takes the phase's form too) and a thinking window of fixed height that follows the end while it is written and fades only where there is more (`serve/ui/src/components/reason.tsx`), figures that only grow pop in and figures that go up and down turn reels (`pop.tsx`, `spin.tsx`, transitions.dev, see `serve/ui/REFERENCES.md`), the + menu is the button grown into the panel (`prompt-bar.css`), a sent prompt rises out of the composer and the empty-chat heading closes up (`lib/sendfx.ts`), a sticky navbar with a gliding pill, shared page heads, strips of figures instead of tables, charts that start at zero. `#/requests/trace` opens an engine `STRATA_TIMELINE` file in the browser (lanes, zoom, what a window spent its time on, a closed copy for ui.perfetto.dev; nothing is uploaded); `POST /metrics/keep {next: N}` keeps the full prompt of the next N requests in their detail file only (Q8). Chat parity done (UI S2); Live, Requests and the request page (hash routes `#/requests/<id>`) read `/metrics` and `/metrics/requests` (UI S5); Hardware, `#/hardware/gpu/<n>` and `#/hardware/ssd/<n>` list every GPU, the CPU kernels in use, RAM and the disks, each figure from an instrument or "not measured" (UI S6); the request page's "Who waited for whom" is a dependency definition over the STATS counters (UI S7, no hardware claim); the chat's coding tools (#96): `lib/agent.ts` (`agentRequest` makes `strata_agent` {cwd, mode, session, allow}; the rules a user chose "allow for this chat" are kept per chat under `strata.agent.rules`, at most 100 per chat and 50 chats), `Project.folder` in `lib/sessions.ts` (the folder the project's chats work in; chats in no project use `Settings.agentFolder`), `ChatController.answer` (POST /agent/permission; the card stays until the server takes the answer), `pages/chat/AgentCall.tsx` (a call of the coding tools shown as what it is: the command and its output, an Edit as a diff, the steps of TodoWrite; a question is a card on the call with Allow / Allow for this chat / Deny; auto mode's verdict), `components/AgentControls.tsx` (switch, mode ask/plan/auto, folder; in the + menu's Coding tools panel and in Settings > Coding tools), the browser checks `agent` (scripted server answers) and `agentdemo` (the real tools behind the mock with STRATA_MOCK_AGENT=1, where a message with "agent demo" makes the fake model use them) | XENO |
| `CMakeLists.txt` | our `strata_timeline`, `strata_secondary`, `strata_secondary_compute`, `STRATA_BUILD_XENO_TESTS` | MIXED (+179/−9) |
| `tests/xeno/**` (88 files) | parity tests, Python tests, `perf/` tools | XENO |

**CMake targets.**

- `strata` links `strata_engine strata_prefill strata_secondary strata_spec`, plus psapi.
- `strata_engine` links `strata_secondary_compute`.
- `strata_secondary` links cudart, and NVML when found (`STRATA_HAS_NVML`).
- `strata_timeline` is linked into `strata_core` and `strata_kernels_cpu`.

## 4. The life of a request

```
client            server.py                                  strata --serve (generate.cpp)                   devices
  | POST /v1/messages?beta=true (stream)
  |-------------->| _do_post: path split on '?' (:1458)
  |               | _anthropic (:1583): anthropic_to_messages
  |               |   - strip billing header (frontend.py:184)  [XENO]
  |               |   - document blocks (pdf_blocks.py:31)      [XENO]
  |               | Service.prepare (:741): render + encode
  |               | Service.run (:831): FIFO lock; restart dead engine
  |               | "GEN <max_new> [keys] <ids>\n" ---------->| next_line (:4871); parse keys (:4884-4924)
  |               |                                           | resume = live | longest prefix checkpoint (:5053-5087)
  |               |<---------------- "RESUME n" --------------| (:5146); apply_pending(true) (:5149)
  |               |                                           | parts [read_from,root_at) [..,turn_at) [..,n-1)
  |               |<--- "PP pos total ms tok/s" per chunk ----|   windows (<= --short-read 64) | batched:
  |               |                                           |   lend -> Prefill::run / run_wave  ---> GPU0 trunk,
  |               |                                           |   checkpoint_at(root/turn) (:5348)      4070 MoE
  |               |<---------------- "REUSED n" --------------| refill lent slots (tail file); (:5359)
  |               |<---------------- "T id" x (a+1) ----------| decode loop (:5402)                ---> GPU0 graph +
  | SSE events    | Detok -> OutputParser -> LoopGuard ->     |                                          4070 + CPU pool
  |<--------------| StopSequenceFilter -> content blocks      |
  |               |<-- "DONE gen prompt pms dms fin acc off reused hits look" (:5636)
  | message_delta | usage: input_tokens = n - reused, cache_read_input_tokens = reused (:1287-1291)
```

**Engine protocol** (`generate.cpp:4073-4089`, `:4816-4854`, `:4871-5712`):

- **Boot:** `INFO key=value…`, then `READY <ctx> stop`.
- **Server → engine:** `GEN` / `GENI` / `STOP` / `QUIT`.
- **Engine → server, per request:** `RESUME` → `PP`* → `REUSED` → `T`* → `STATS` → `DONE`, or `ERR`. Several paths
  `return 1` after `ERR`; the process exits and the server restarts it. `STATS key=value…` ([XENO], UI S4) is one
  line just before `DONE`: this request's decode counters (tiers, stage ms, NVMe), parsed into `engine.last["stats"]`;
  a reader that does not know it skips it. A second `INFO cpu_isa=… gpu_arch=…` is printed at boot.
- **Watchdog:** `STRATA_WATCHDOG_S` (60).

**Prefix reuse** (`:5044-5146`). `resume` is `live` if it is a prefix of the prompt, else the longest checkpoint that
is a prefix.

- Every checkpoint that is longer than `resume` **or is not a prefix of this prompt** is erased (`:5083-5085`). So an
  interleaved request that diverges early (a Claude Code side request) wipes the chain (fork-delta report §8 item 2).
- `resume == 0` clears all checkpoints (`:5088-5096`).
- A checkpoint resume restores GDN, PLE history and indexer tails; the KV cells are positional and trusted.

**Prompt parts** (`:5286-5356`).

- `turn_at` is the last `<|im_start|>`. `root_at` is the first, when the prompt is read from 0 and that position is
  ≥ 2048.
- A part of ≤ 64 tokens goes through verify windows. Otherwise the part is lent slots, then `Prefill::run` or
  `run_wave`.
- A part log line is printed (`8408f84`). Lent slots are refilled before the first decode window.

**Borrowing** (`plan_lend`, `:4131-4153`). The chunk is the largest of {8192, …, 256} whose buffers take at most 85 %
of the cache slots, or 90 % with pinned experts. The buffers are carved from the **last** GPU0 slots.

**Decode per round** (`:5361-5519`):

1. `apply_pending`.
2. `ver.run(T, …)`: host dispatch per layer (§5); the 4070's partials and the CPU rows merge.
3. Accept the longest matching prefix.
4. Start the adapt thread.
5. `ver.commit(a+1)`, then `ple_ahead.push` [XENO].
6. Emit `T` lines.
7. `mtp.draft`, whose `on_draft` does the PLE prefetch [XENO].
8. Join adapt; check EOS or STOP.

The window is up to 6 tokens with `--spec 4` plus the suffix drafter (`:1785-1788`).

## 5. Memory hierarchy and decode dispatch

```
                          +----------- GPU0: RTX 5060 Ti 16 GB (sm_120, PCIe x4) ----------------------------+
 routed (layer,expert) -->| trunk, native dense/head, KV (int8), GDN state, PLE hist, MTP (size UNMEASURED), logits|
 entries per window       | EXPERT CACHE (ranked profile prefix): exclusive owner by default [XENO]; last k    |
                          | slots = "lendable tail" = prompt-path buffers while a prompt is read [UPSTREAM     |
                          | borrow + XENO tail-file refill]                                                    |
                          +-------------------------------------------------------------------------------------+
                          +----------- GPU1: RTX 4070 SUPER 12 GB (sm_89, x16, display card) [XENO] ------------+
                          | SECONDARY ARENA: next-ranked experts, exclusive by default; runner workspace;       |
                          | split-prefill buffers, ring, stager; free floor (D2x: 640 MiB)                      |
                          +-------------------------------------------------------------------------------------+
                          +----------- host RAM 48 GB ----------------------------------------------------------+
                          | arena reserves address space for all 24,576 experts; only HOST-OWNED ones are       |
                          | committed (placement-first) [XENO]. CPU pool computes misses. --ram-cache-gib -> NVMe |
                          +-------------------------------------------------------------------------------------+
                          +----------- NVMe --------------------------------------------------------------------+
                          | GGUF shard 1 (experts + dense), shard 2 (PLE, direct reads), tail-<key>.bin (~3.5 GB)|
                          +-------------------------------------------------------------------------------------+
```

`expert_pool_dispatch_multi` (`expert_source.cpp:780-1075`):

```
ids (n_tok x k) --> usage[layer*512+e] += 1                                    (:812-814)
                --> plan: host_res[l,e] >= 0 -> kind 0: GPU0 hit (in the window graph)
                          miss, pcie share   -> kind 1: PCIe read (D2x: pcie_frac 0 -> never)
                    publish plan to GPU before CPU work                         (:820-898)
                --> upstream remote helpers (--expert-cache-device*)            (:905-913)   [UPSTREAM]
                --> secondary_res[l,e] >= 0 -> kind 2: 4070 launch              (:912-947)   [XENO]
                --> act quantize                                                (:948-955)
                --> NVMe tier: materialize_batch                                (:956-976)   [XENO]
                --> CPU jobs -> pool.run_split_multi_native (13 workers + host) (:978-1030)
                --> secondary_runner.finish(out)                                (:1036-1052) [XENO]
tier_entries[0..3] = primary / 4070 / pcie / cpu                               (:993, :1021)
```

- **Tier split** (MEASURED, dual D2 decode on the code prompt): primary 56.9 %, 4070 28.0 %, CPU 15.1 % (checkpoint
  report §3.1).
- **Measured sizes:**

| arm | GPU0 tier | 4070 tier | host-owned | working set | source |
|---|---|---|---|---|---|
| D2 (tier 8704) | 6,239 slots / 8.03 GiB | 6,602 / 8.50 GiB | 15.11 GiB | 16.4 GiB | checkpoint report §3.1, §5 |
| ranked profile, tier 6400 | — | 4,854 / 6.25 GiB | 17.36 GiB | 18.8-19.2 GiB | same §5; #45 m10 |
| D2x | UNMEASURED here | — | — | 19.2 GiB at 8K | #45 m11 |

**Boot order.**

| step | code | note |
|---|---|---|
| options, validation | `:1419-1869` | unknown flag = error |
| LAZY CUDA [XENO] | `:1413-1418` | EAGER removed (`c868593`) |
| `--pcie-frac` → 0 [XENO] | `:1909-1919` | #27 |
| exclusive and placement decisions [XENO] | `:1920-1970` | verified |
| MTP before the host arena | `:2332-2348` | "WDDM can refuse the draft weights after mapping tens of GiB of host pages" (`:2330-2331`) |
| host arena | `:2405-2440` | `pin_for_cuda = false` with the 4070 tier or exclusive primary |
| cache sizing | `:2619-2757` | ×3/4 retry (#60) |
| placement-first GPU0 fill [XENO] | `:2811-2963` | reader thread → 32 pinned → H2D → D2H → memcmp → release |
| 4070 tier [XENO] | `:2974-3160` | monitor thread `:3148` |
| `load_rest` [XENO] | `:3177-3190` | host-owned only |
| tail file [XENO] | `:3231-3265` | — |
| serve setup | `:4155-4870` | lend, lanes, verifier, drafter, adapt, stdin thread, READY |

**Ownership rules [XENO].**

- Exclusive primary is on when eligible. Exclusive secondary is on with the tier.
- Placement-first applies whenever any tier is exclusive.
- The lendable tail is refilled from the tail file.

**Adaptive swaps.**

- Code: serve `:4485-4728`, generate `:6150-6490`.
- Usage decays ×0.7 per adapt call. A swap needs usage ≥ 2 and a gain ≥ 1.5.
- **D2x in serve:** paired primary swaps 8 every round; the 4070 tier is static (`adapt_secondary` resolves to 0).

**The CPU pool under `--serve`** (`docs/reports/2026-09-30-cpu-pool-and-kernel-survey.md` §2): an idle worker spins
`_mm_pause` for 20 ms (`pool.hpp:167`, `STRATA_POOL_SPIN_US`) before it sleeps, and the serve loop never calls `rest()`
(only generate mode does, `generate.cpp:6592`), so during decode the 13 workers spin at priority 15 (HIGH class +
HIGHEST) on logical CPUs 2, 4, 6, 8, 10 and 12-19, with the host on 0. Between requests and during batched prompt reads
they sleep.

## 6. KV cache and checkpoints

- **QSA KV** (`--kv fp16|int8|q4_0|k8v4`): D2x uses int8, all in VRAM (`--kv-resident 0`).
- **GDN state:** one in-place buffer.
- **MTP K/V:** its own, over the last 32,768 cells.
- **Checkpoints** (`:1197-1266`, `conv_cache.hpp`):
  - Contents: ids plus GDN, PLE and indexer tails, about 118 MB each.
  - `checks[0]` is pinned; the rest rotate LRU; the cap is 6.
  - Created mid-prompt (every 16,384 fresh tokens, in `serve_chunk`, `:4432-4450`), at the root (`:5299-5306`) and at
    the turn (`:5290-5293`).
  - `checkpoint_at` dedupes **by length only** (`:4384-4385`), then runs `cudaDeviceSynchronize` + `checkpoint_save`
    (`:4396`).

## 7. Configuration surfaces

**The 25 XENO flags** (set difference against `4c68013`):

`--adapt-gate --adapt-secondary --cache-cpu-only --exclusive-primary-experts --exclusive-secondary-experts
--lock-cpu-experts --mmvq-exact --no-exclusive-primary-experts --no-exclusive-secondary-experts --no-tail-file
--ple-ahead --pool-priority --pool-rest --process-priority --profile-decode-range --profile-prefill-range
--ram-cache-gib --route-trace --secondary-async-launch --secondary-expert-mib --secondary-free-floor-mib
--secondary-graph --secondary-profile-timing --secondary-stage-only --tail-file`

No upstream flag was removed. Upstream defaults we changed through a flag: `--pcie-frac` (−1 → 0), and
`--adapt-swaps`/`--adapt-every` (8/1 under exclusive primary).

**Default-on or flagless behaviour changes against `4c68013`** (upstream rule 7, "no silent default changes"; option
defaults verified at `generate.cpp:181`, `:237`, `:239`, `:251`, `:258`, `:312`):

| change | where | way back to upstream's behaviour |
|---|---|---|
| MMQ `nsm = 1` on every device | `moe_mmq.cu:153-154` | `STRATA_MMQ_STREAM_K` set to any value |
| CPU-order Q2_0 GPU path and the `exp(double)` CPU SwiGLU (changes CPU-only output too) | `pool.cpp:419`; `verify.cpp:654`, `:694` | **no opt-out** |
| LAZY module loading instead of forced EAGER | `generate.cpp:1413-1418` | `CUDA_MODULE_LOADING=EAGER` in the environment |
| exclusive primary when eligible (`exclusive_mode = -1`); eligible by default for any Q2_0 native pack with a profile and a cache, since `pcie_frac` now defaults to 0 | `:251`, `:1931-1941` | `--no-exclusive-primary-experts` |
| placement-first whenever a tier is exclusive | `:1948` | follows the above |
| tail file | `:312` | `--no-tail-file` |
| `--pool-priority` 2 (HIGHEST) | `:237` | `--pool-priority 0` |
| `--pool-rest` 1 (generate mode only, P6) | `:239` | `--pool-rest 0` |
| decode PLE read-ahead | `:181` | `--ple-ahead 0` |
| first-chunk PLE read-ahead in the prompt path | `prefill.cpp:1940-1943` | `STRATA_PREFILL_PLE_AHEAD=0` |
| one gather launch per MMQ group | `prefill.cpp:2258-2262` | `STRATA_PREFILL_GROUP_GATHER=0` |
| AVX-VNNI Q2 row dispatch | `expert_layout.cpp:73-77` | `STRATA_FORCE_AVX2=1` (also turns off AVX-512) |
| copy-issuer thread on (upstream's D-5 default, kept) | `prefill.cpp:2250-2257` | `STRATA_PREFILL_ISSUER=0` (per the fork-delta report §4.3 B.2) |
| AVX2-only kernel notice for non-Q2_0 native packs dropped (log only) | `generate.cpp:1973-1978` | none (fork-delta report §4.1) |

**XENO environment variables:**

- Prompt-path features: `STRATA_PREFILL_EXPERT_SPLIT`, `STRATA_PREFILL_WAVE`, `STRATA_MMQ_STREAM_K`,
  `STRATA_DM_FRONTIER(_FRAC)`, `STRATA_EXPERT_ORDER`.
- Prompt-path diagnostics and A/B: `STRATA_PREFILL_{BUFFERS,COPY_THREAD,GROUP_GATHER,PLE_AHEAD,ROUTE_TRACE}`.
- Timeline: `STRATA_TIMELINE(_MAX_EVENTS)`.
- Exit hang: `STRATA_EXIT_TRACE`, `STRATA_EXP_QUICK_EXIT`.
- Placement and arena: `STRATA_ARENA_PIN`, `STRATA_HOT_TO_SECONDARY`.
- Probes: `STRATA_LINK_PROBE`, `STRATA_SECONDARY_POKE`.

**D2x** (`D:/Github/Strata/strata-flash-next-d2x.json`):

- **Flags** (read from the file during verification): `--pack D:\Github\Strata\packs\q2_0`, `--native <Q2_0 shard 1>`,
  `--ple-gguf <shard 2>`, `--expert-profile D:\Github\Strata\data
anked-exl3only-profile.bin` (untracked, 196,632 B,
  `git hash-object` `c7d37560`, 24,576 pairs; not HEAD's `a4ed7e5`), `--prefill auto` (borrowing on), `--spec 4`,
  `--spec-min-p 0.5`, `--mtp D:\Github\Strata\mtp
t`, `--kv int8`, `--max-context 131072`, `--pool-workers 13`,
  `--adapt-swaps 8`, `--adapt-every 1`, `--pcie-frac 0`, `--expert-cache 8000`, `--vram-reserve-mib 2400`,
  `--secondary-free-floor-mib 640`, `--secondary-expert-mib 6400`, `--exclusive-primary-experts`, `--pool-priority 2`,
  `--process-priority 2`. `cwd` is `D:\Github\Strata`, a checkout on branch `xeno/avxvnni` (`f679806`), so the
  profile, MTP and draft vocab come from that pre-merge-1 tree.
- **Config comment:** several GPUs in the server's `gpu` field make the server add `--layer-split`, so the dual setup
  comes from `CUDA_VISIBLE_DEVICES=1,0` in `start-flash-next.ps1`, not from `gpu`.
- **Environment:** `STRATA_PREFILL_EXPERT_SPLIT=1`, `STRATA_PREFILL_WAVE=1`.
- **Engine:** `dynfix.exe`, the merged engine plus `5c51574`.
- **Measured:** #45 m11 (`dyn.exe` sha256 `8462650abea384b2`): 8K read 4.12/4.15 s, decode code 83.6/86.5 and thai
  50.8/52.2, "An 8K read plus 256 tokens takes about 7.9 s".

**Build.** VS2022 with CUDA 13.3, Ninja, `-DSTRATA_BUILD_TESTS=OFF`, `-DSTRATA_BUILD_XENO_TESTS=ON`,
`"-DCMAKE_CUDA_ARCHITECTURES=89;120"` (`C:/Strata-exp/build-dyn.cmd`). So upstream's own C++/CUDA parity tests were
**not built or run** on D; #45 reports Python tests only (93). Upstream `3801f86` ("STRATA_BUILD_TESTS=ON works
without the unpublished tests/ tree") shows part of upstream's `tests/` is not published, which matters for rule 5
(a PR carries its parity test in upstream's layout).

## 8. Diagnostics

- **Per request:** a part log, request metrics by tier, decode hit rate.
- **At boot:** `mem_mark`, placement-first timings, tail-file lines.
- **Timeline:** `STRATA_TIMELINE`, read with `tests/xeno/perf/timeline.py`. `AGENTS.md` requires recording one timeline
  rather than hunting stage by stage.
- **Tests:** XENO C++/CUDA tests exist only with `STRATA_BUILD_XENO_TESTS` on Windows. #45 reports "Our Python tests
  pass (93)" on the merged tree.

## Not yet in the baseline

Work committed on other branches that changes what this blueprint describes. When the baseline moves past it, fold it
into the sections above and delete the row.

| branch, commit | area (section) | what changes |
|---|---|---|
| `xeno/claude-merge-0.1.20` `694ad82` (#46) | request life (§4), `frontend.py` | a `document` or `image` inside a `tool_result` reaches the model (`_tool_result_content`); images stay image items with vision, else a note |
| same, `12556b1`, `284bb19` (#49 S2, #48) | request life (§4), engine protocol | `StrataEngine.generate` polls the process at every quiet heartbeat (`QUIET_S`) and in the STOP-and-drain; a dead or restarting engine is `529 overloaded_error` on `/v1/messages`; each stdout pump owns its process and queue; `disable_parallel_tool_use` |
| same, `e63d359`, `284bb19` (#49 S3) | request life (§4) | thinking budget: `Service._generate` stops the engine once the budget is spent inside the thinking block, then continues from prompt + written + drained tokens + CLOSE; the DONE figures of both calls are merged; non-streamed requests are side requests (low effort, 1,024 budget, `STRATA_SIDE_BUDGET`) |
| same, `55eb129` (#49 S5) | diagnostics (§8) | the llama-server-shaped timing block per request (`serve/timing_line.py`); `STRATA_TRACE_SSE=<file>` |
| same, `c54c337`, `284bb19` (#49 S6) | request life (§4) | the FIFO lock became `RequestGate`: streamed requests (priority 0) before non-streamed (1); image encoding takes the slot at the request's priority |
| same, `80cd0b3` (#49 S8) | configuration (§7) | `"sampling": {"preset": "deterministic" / "balanced" / "reasoning"}` in the run config |
| same, `1620329`, `284bb19` (#49 S4, server) | configuration (§7), request life (§4) | `serve/cjk_guard.py`: `"cjk_guard": true` writes the Han ids next to the config and starts the engine with `--ban-ids`; `ban=1` per request when the current turn has no Han and does not name Chinese; `cjk_chars` in `/metrics` |
| same, `5f12d89` (#49 S4, engine; built and checked: `sampler_parity`, engine seam check) | engine protocol, decode (§4, §5) | `--ban-ids FILE`, INFO `ban=<count>`, GEN/GENI key `ban=1`; `SamplerParams.ban` (a device bitmap; the sampler kernels are templated so no ban is the old code); the verifier re-picks a banned request's window |
| `xeno/exp-upstream-0.1.26-dyn`, upstream PR #175 (`9bc9a8e`, `fcf93ec`, ported; #49 S7) | engine protocol and checkpoints (§4, §6) | the GEN key `cache_slot=0..3` (the server sends `strata_cache_slot`, default 0): an inactive slot's positional KV, recurrent state and prefix checkpoints are snapshotted to a delete-on-close file and restored on return; INFO `cache_slots` / `cache_storage`; layer-split engines advertise one slot. The server picks the slot itself (`CacheSlots` in `serve/server.py`): a prompt family (its first 512 tokens) keeps its slot, a new family takes a free one or the least recently used; a client's own `strata_cache_slot` wins |
| same (#49 S7 follow-up, decided by the developer 2026-09-30) | request life (§4) | the classifier's fast stage: when the last user message (thinking off) says the reply "MUST begin with <X>" or "Respond with <X>... ONLY" (the real fast stage: `<severity>N</severity>`), the server writes `<X>` for the model and the model continues it (`serve/forced_opening.py`, `_opening` in `Service._generate`); such a request, when greedy, is answered again from its last answer instead of generated again (`Service.replay_key`, 4 kept; never for any other request, so a repeated benchmark prompt is always timed). `STRATA_DEBUG` also logs, per request, the tokens it shares with its family's last prompt and its turn boundaries. The server sends that shared length as the GEN key `ckpt_at=P`, and the engine reads the prompt in one more part (`prompt part shared`) and keeps a checkpoint at P when it lies between the resume point (or the root) and the last turn start: a transcript that grows inside one message (the classifier's) had no checkpoint past the root |
| same (2026-09-30, from live use) | process start, request life (§3, §4) | the port is bound before the model loads: `loading_server` answers every POST with 529 `overloaded_error` and every GET with 503 until `serve()` takes the port (a refused connection made Claude Code back off, once to "retry in 29m"; EXL3's `a04b381` did the same). A side request (`think_budget.is_side`) is now non-streamed **and without tools**: Claude Code resends a main turn without streaming after a failed stream, and that one's thinking was closed at 1,024 tokens |
| same, `6f646e8` (#55 W7, 2026-10-01) | model (§1, the drafter), setup | `data/draft_vocab.bin` (the MTP draft head's token subset) gains all 5,741 Thai tokens: 40,525 -> 46,252 ids, draft head 68.0 -> 77.6 MiB. The old subset held 14 Thai tokens, so Thai answers drafted almost nothing; Thai decode +20-35 %, the same output. `tools/draft_vocab.py` (upstream 0.1.27 `319e4ef`) with a `thai` script. Since the 0.1.30 merge (#56), setup's `refresh_draft_vocab` (upstream 0.1.27) installs it at setup and at every start over a shipped subset (the old 40,525-id one included); `--draft-vocab en` (`data/draft_vocab_en.bin`) keeps the English/code one; a subset made by hand is kept |
| `xeno/exp-upstream-0.1.30-dyn` (#56, 2026-10-01): upstream v0.1.30 merged into the 0.1.26 line | every section: the engine, the sampler, the prompt path's loans, setup | upstream 0.1.27-0.1.30 under the fork: sampled top-k split across the GPU (#197; the CJK ban ported into its kernels), the multi-GPU session carve and per-stage prompt loans (#216; a layer split with prompt borrowing is refused, as the fork's wave lanes live in CUDA0's loan only), the conversation cache (#189, opt-in, off), rope scaling (#84, opt-in), idle unload (#208; `RequestGate` gained a non-blocking acquire), 1024-token streaming off the split layout (the split layout keeps 2048), #210 tool-call parsing, #154 fixes. The fork keeps its MTP/arena block (moved after upstream's session allocation), its lending and wave, its cache slots (now carrying the checkpoint's `dead`/`block_pos`) and `serve_fatal()`. Greedy output identical to the 0.1.26 line on the #54 set. Full record: `docs/reports/2026-10-01-merge-upstream-0.1.30.md` |

## Keeping this file current

**Update this file in the same commit as the change** when a change does any of these:

1. adds, removes or renames a process, a thread that does work every request or every round, or a file with a
   responsibility of its own (§2, §3);
2. changes the life of a request: the HTTP path, the engine protocol (`GEN` keys, reply lines), prefix reuse, the
   prompt parts, the decode round (§4);
3. changes where experts live or how decode dispatches them: tiers, profiles, swaps, the CPU pool (§5);
4. changes the KV cache or the checkpoints (§6);
5. adds, removes or changes the default of a flag, a JSON config key or an environment variable (§7);
6. adds or changes a diagnostic: a log line, a counter, a timeline span, a test harness (§8);
7. merges upstream or moves the served engine to another branch or commit (the baseline).

**How:**

- Edit the section the change belongs to. Keep every statement tied to a `file:line`, a commit or a log line; mark a
  statement no run has confirmed as "static reading".
- A change on a branch that is not the baseline goes into [Not yet in the baseline](#not-yet-in-the-baseline) first,
  one row per commit or feature, and moves into the sections when the baseline includes it.
- When the baseline moves (a merge, a new served engine), re-check every `file:line` that the moved code touches,
  then add a row to the revision log with the new commit and what was re-checked.
- Do not copy measurements here: link the issue or report that holds them. This file says how the system works; the
  register and the reports say how fast it is.
- When a change in the list above needs no edit, the commit message says why in a trailer:
  `Blueprint: n/a - <why>`. `tools/hooks/commit-msg` (`git config core.hooksPath tools/hooks`) refuses a commit
  whose diff adds or removes an engine flag, a GEN key, a `STRATA_*` variable, a run-config key or a source file
  without this file or that trailer.
