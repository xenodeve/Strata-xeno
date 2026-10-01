# The next web app and the deep monitor (UI S0-S7): what was built, what was measured, what is not verified

Date: 2026-10-01. Branch `xeno/ui-next-app-s0-s7` (off `xeno/exp-upstream-0.1.30-dyn`, 526a56c), worktree
`C:\Strata-exp\src-dyn`. The daily worktree (`src-daily`) and the daily server (:8091) were not touched.
Input: the 2026-10-01 handoff and survey (grill Q1-Q11). No PRD or issue exists yet: `/to-prd` and `/to-issues` are the
developer's to run, and no PR opens before an issue does.

## What exists

| Slice | What | Where | Verified by |
|---|---|---|---|
| S0 | the Monitor lists every visible GPU when the config has no `"gpu"` key (the D2x bug: card 0 only) | `serve/server.py` `monitor_gpus`, `serve/telemetry.py` | 6 unit tests; real NVML showed both cards |
| S1 | the next app at `/next/`, the classic app at `/classic/` (untouched), `dist/` committed and checked for staleness | `serve/ui/`, `serve/ui_hash.py`, `serve/test_ui.py` | tests; a stale source fails the test |
| S2 | Chat parity (stream, thinking, MCP, attachments, sampling, shared settings, export/undo), About, `"ui": "next"` switch | `serve/ui/src/pages/chat/`, `lib/chat.ts` | browser run on a mock; `bun test` for markdown escaping and the API message replay |
| S3 | history on disk, dialect, prefill speed, decode over a 16-token window, keep-full-prompt on request | `serve/history.py`, `GET /metrics/requests[/<id>]`, `POST /metrics/keep` | 40+ unit and server tests |
| S4 | a `STATS key=value` line before `DONE`; `INFO cpu_isa`, `INFO gpu_arch` | `src/program/generate.cpp` | **compiled and linked, never run** (see below) |
| S5 | Live, Requests, request page | `serve/ui/src/pages/` | browser run on a mock |
| S6 | Hardware, GPU page, SSD page; telemetry per card (clocks, throttle, PCIe link), disks behind the model | `serve/telemetry.py`, `serve/storage.py` | unit tests; real NVML and real disk listing |
| S7 (UI part) | "who waited for whom" over the STATS counters, stated as a dependency definition | `serve/ui/src/lib/stall.ts` | `bun test` |
| extra | Dashboard (one-screen overview); the model's name and the quantization of each part from the GGUF headers; prefill shown apart from the conversation cache; smooth open/close motion | `pages/Dashboard.tsx`, `serve/gguf_info.py`, `components/motion.tsx` | tests; the real model's headers read in 0.1 s |

Test counts at the last commit: `python -m unittest` over `serve.test_server test_history test_ui test_telemetry
test_gguf_info test_timeline test_detok test_mcp` 208 tests OK (3 skipped); `bun test` in `serve/ui` 21 tests OK.

## Measured

- **Browser cost** (the handoff's rule: measure once, add a Quiet mode only if above noise). Renderer CPU from CDP
  `TaskDuration`, three rounds with the arms alternated in one session: blank page 0.00 %, Live page idle 0.18-0.37 % of
  one core, Live page while requests stream continuously 1.40-1.61 %. Below noise, so **no Quiet mode**. The browser's GPU
  cost was not measured by this method and is not claimed.
- **Real model headers** (`C:\AI\models\ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO\Q2_0`, 2 shards, 66.4 GB): experts Q2_0 at
  2.25 bits per weight (34.0 GB), PLE table IQ4_NL at 4.50 (28.8 GB), attention a mix (BF16, Q3_K, IQ4_XS, Q4_K, ...) at
  5.30, embeddings Q3_K, output Q5_K/BF16; 3.00 bits per weight over the whole model.
- **Real GPUs**: NVML numbers the 4070 SUPER as 0 and the 5060 Ti as 1, CUDA the other way round (`CUDA_VISIBLE_DEVICES=1,0`).
  That is why `INFO gpu_arch` is `sm<N>@<card name>` and the page matches by name. At idle the 5060 Ti's link read
  Gen 1 x4 against a maximum of Gen 4 x16 (idle power saving: the page says the link drops when idle).
- The daily server's `/metrics` showed `gpu_count: 1` (the S0 bug, in production) and `psutil: false`, so per-disk
  throughput is **not measured** there. The Hardware page says so instead of showing a zero.

## Not verified, and why

- **The rebuilt engine has not run.** `build-dyn/strata.exe` compiles and links with the STATS and INFO changes; the
  daily server holds both GPUs, so nothing ran it. Still to do, with :8091 stopped by the developer: the serve-mode
  byte-compare that the old `DONE` line is unchanged; a real STATS line from a request; `INFO gpu_arch` names against NVML names.
- **Not done, and why:** per-request n-gram/PLE statistics (`ReaderStats` is written by the I/O worker thread and
  `stats()` is documented as safe only with no ticket in flight, so a copy at the end of a request needs a lock in the
  reader, and that cannot be tested without the GPUs); PCIe copy timing per card (CUDA events at the `cudaMemcpyAsync`
  sites); per-round timestamps and their same-session ABBA cost (S7); the Trace-next-N viewer and Perfetto export (S8);
  the Nsight import (S9); sampled kernel latency (S10, opt-in); the React Bits components (the handoff says to look at
  each live first).
- **Flaky under load.** One full test run, made while an engine build and a mock server were using the machine, failed 2
  tests; the same set passed on the next two runs (208 OK). The failing test names were not captured.

## Decisions made while building (not in the handoff)

- `"ui": "classic"` stays the default for `/`. The handoff said to switch `/` once Chat reached parity, but the
  Monitor had no successor until S5/S6, so the switch is one config key (`"ui": "next"`).
- `STATS` goes **before** `DONE`, not after: the server's reader returns at `DONE`, so a line after it would be read as
  the next request's. Readers that skip unknown lines are unaffected.
- A sixth page, Dashboard, was added at the developer's request during the session.
- One history row per engine call: an MCP request calls `Service.run` once per tool round, so later rounds get the id
  with `-2`, `-3`.

## Security notes

`security-review` over this branch found nothing at confidence 8 or above. The one thing worth the developer's eye:
with no API key set, `/metrics` and `/metrics/requests` carry `preview`, the first 200 characters of the last user
message (the handoff's Q8 decision). The server already warns that without a key anyone on the network can use it; a
server reachable beyond localhost should set `api_key`. The full prompt is kept only after `POST /metrics/keep` and goes
to the request's detail file, never to the summary row.

## How to look without the GPUs

`python serve/ui/dev/mock_server.py [port] [model.gguf ...]` serves the app with no model: the hardware, routes and
history are real, the answer text and the engine's STATS and prefill chunks are labelled fixtures.
