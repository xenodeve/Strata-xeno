# The next web app and the deep monitor: what was built, what was measured, what is not verified

Date: 2026-10-01, brought up to date the same day. Branch `xeno/ui-next-app-s0-s7` (off `xeno/exp-upstream-0.1.30-dyn`,
526a56c), worktree `C:\Strata-exp\src-dyn`. The daily worktree (`src-daily`) and the daily server (:8091) were not touched.
Input: the 2026-10-01 handoff and survey (grill Q1-Q11). The PRD is #65 and the work that is left is its sub-issues
#66-#78.

## What exists

| Slice | What | Where | Verified by |
|---|---|---|---|
| S0 | the Monitor lists every visible GPU when the config has no `"gpu"` key (the D2x bug: card 0 only) | `serve/server.py` `monitor_gpus`, `serve/telemetry.py` | unit tests; real NVML showed both cards |
| S1 | the next app (served at `/` since #67, and at `/next/`), the classic app at `/classic/` (untouched), `dist/` committed and checked for staleness | `serve/ui/`, `serve/ui_hash.py`, `serve/test_ui.py` | tests; a stale source fails the test |
| S2 | Chat parity (stream, thinking, MCP, attachments, sampling, shared settings, export), About | `serve/ui/src/pages/chat/`, `lib/chat.ts` | browser runs on a mock; `bun test` for the controller over a stream cut at awkward places |
| S3 | history on disk, dialect, prefill per chunk, decode over a 16-token window, keep-full-prompt on request | `serve/history.py`, `GET /metrics/requests[/<id>]`, `POST /metrics/keep` | unit and server tests |
| S4 | a `STATS key=value` line before `DONE`; `INFO cpu_isa`, `INFO gpu_arch` | `src/program/generate.cpp` | **compiled and linked, never run** (#66) |
| S5 | Live, Requests, request page | `serve/ui/src/pages/` | browser runs on a mock |
| S6 | Hardware, GPU page, SSD page; telemetry per card (clocks, throttle, PCIe link), disks behind the model | `serve/telemetry.py`, `serve/storage.py` | unit tests; real NVML and real disk listing |
| S7 (UI part) | "who waited for whom" over the STATS counters, stated as a dependency definition | `serve/ui/src/lib/stall.ts` | `bun test` |
| S8 (viewer part) | `#/requests/trace` opens an engine `STRATA_TIMELINE` file in the browser (lanes, zoom, what a window spent its time on, a closed copy for Perfetto); nothing is uploaded | `serve/ui/src/pages/Trace.tsx`, `lib/trace.ts` | `bun test`; a real 162,925-span file parsed and drawn |
| beyond the plan | Dashboard; the model's name and quantization from the GGUF headers; prefill shown apart from the conversation cache; the composer bar (attach, new chat, save, thinking effort, sampling); the thinking line with its lattice; orbs for every state; the prompt-read speed under a prompt; rewrite and take back a prompt; the model's own thinking levels (`engine.efforts`); English and Thai with a switch (a test fails on a string with no Thai); gentle open, close, stretch and close-up motion | `serve/ui/src/`, `serve/gguf_info.py`, `serve/server.py` | `bun test`, `serve.test_server`, `serve.test_ui`, browser runs on a mock |

Test runs at the last commit: `bun test` in `serve/ui` 72 tests OK; `python -m unittest serve.test_server` 133 OK,
`serve.test_ui` 11 OK.

## Measured

- **Real model headers** (`C:\AI\models\ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO\Q2_0`, 2 shards, 66.4 GB): experts Q2_0 at
  2.25 bits per weight (34.0 GB), PLE table IQ4_NL at 4.50 (28.8 GB), attention a mix (BF16, Q3_K, IQ4_XS, Q4_K, ...) at
  5.30, embeddings Q3_K, output Q5_K/BF16; 3.00 bits per weight over the whole model.
- **Real GPUs**: NVML numbers the 4070 SUPER as 0 and the 5060 Ti as 1, CUDA the other way round (`CUDA_VISIBLE_DEVICES=1,0`).
  That is why `INFO gpu_arch` is `sm<N>@<card name>` and the page matches by name. At idle the 5060 Ti's link read
  Gen 1 x4 against a maximum of Gen 4 x16 (idle power saving: the page says the link drops when idle).
- The daily server's `/metrics` showed `gpu_count: 1` (the S0 bug, in production) and `psutil: false`, so per-disk
  throughput is **not measured** there (#69). The Hardware page says so instead of showing a zero.
- The model's chat template accepts low, medium and xhigh (its default) and `enable_thinking=false`; "high" renders the
  same as xhigh. The app used to offer "High", which the server turned into xhigh without saying.

## Not verified, and why

- **The browser's cost is not known.** An early measurement (renderer CPU from CDP over a still page, then a streaming one)
  was made before the idle orbs moved, and says nothing about the animated ones. The later attempt used a headless
  Chromium without a GPU, which runs the display loop far faster than a screen would (it is not tied to a display), so its
  figure overstates a real page by a large factor and is **not claimed**. Measuring it once on a real GPU browser, and a
  Quiet switch if it is above noise, is #70. Until then the idle orbs of the small GPU cards are capped at 30 frames a second.
- **The rebuilt engine has not run.** `build-dyn/strata.exe` compiles and links with the STATS and INFO changes; the
  daily server holds both GPUs, so nothing ran it (#66, needs :8091 stopped by the developer, and #59).
- **Not done:** per-request n-gram/PLE statistics (#73), PCIe copy timing per card (#74), per-round timestamps and their
  same-session ABBA cost (#75), "trace the next N requests" on the engine side (#76), the Nsight import (#77), sampled
  kernel latency (#78), per-disk throughput from the OS (#69).
- **Never read by a native Thai reader.** The Thai was written and checked by tests (completeness, placeholders) and by
  screenshots for clipping and overflow, not by a person who reads Thai.
- **Flaky under load.** One full test run, made while an engine build and a mock server were using the machine, failed 2
  tests; the same set passed on the next runs. The failing test names were not captured.

## Decisions made while building (not in the handoff)

- `/` serves the new app since chat parity (#67); `"ui": "classic"` in the run config brings the classic app back to `/`.
- `STATS` goes **before** `DONE`, not after: the server's reader returns at `DONE`, so a line after it would be read as
  the next request's. Readers that skip unknown lines are unaffected.
- A sixth page, Dashboard, was added at the developer's request during the session.
- One history row per engine call: an MCP request calls `Service.run` once per tool round, so later rounds get the id
  with `-2`, `-3`.
- The answer's footer is stored as figures and worded when shown, so it follows the language; a conversation saved by an
  older version keeps its old text.
- The app is English and Thai; the default is Thai when the browser says so. Technical nouns (prompt, token, prefill,
  decode, cache, context, GPU) and anything the server or the model supplies stay as they are.

## Security notes

`security-review` over the branch found nothing at confidence 8 or above at the time. The one thing worth the developer's
eye: with no API key set, `/metrics` and `/metrics/requests` carry `preview`, the first 200 characters of the last user
message (the handoff's Q8 decision); whether to hide it is #68. The server already warns that without a key anyone on the
network can use it; a server reachable beyond localhost should set `api_key`. The full prompt is kept only after
`POST /metrics/keep` and goes to the request's detail file, never to the summary row. A fuller review of the app as it is
now (the key in the browser, what is stored, the keep endpoint, the history files, the file reader, the markdown renderer) is #71.

## How to look without the GPUs

`python serve/ui/dev/mock_server.py [port] [model.gguf ...]` serves the app with no model: the hardware, routes and
history are real, the answer text and the engine's STATS and prefill chunks are labelled fixtures. The mock thinks for about
twenty seconds (`STRATA_MOCK_THINK_MS`) and reads a prompt at `STRATA_MOCK_PREFILL_TPS` (default 300 tokens a second).
