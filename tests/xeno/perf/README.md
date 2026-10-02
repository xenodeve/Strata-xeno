# tests/xeno/perf — the measurement tools

These are the scripts behind the 2026-09-28/29 reports in `docs/reports/`. They are kept here so that every speed claim can be re-run from the repo (issue #24). The parser test is `tests/xeno/test_ab_parse.py`.

| script | what it does |
|---|---|
| `ab.py` | Same-session ABBA runner (see below). |
| `launch_ab.py` | Starts `ab.py` fully detached, so it survives the shell that launched it. |
| `route_tools.py` | Routing traces from `strata generate --dump-routing` (format 2, #93): counts, STRP profiles, static-tier simulation. |
| `blend_profile.py` | Blends EXL3 router counts with Strata traces, and evaluates the blend on held-out benchmark traces. |
| `adapt_sim.py` | Compares static placement, an oracle, and decayed-LFU swaps with a per-round budget. |
| `make_trace_prompts.py`, `trace-prompts/` | Training prompts for routing traces, disjoint from the benchmark prompts. |
| `collect_traces.py` | One traced run per training and benchmark prompt. |
| `h2_nsys.py` | Nsight Systems capture of the verifier decode, then kernel time per device and per class. |
| `gguf_sizes.py` | Tensor bytes from GGUF headers, split into experts and trunk. |
| `timeline.py` | The whole-pipeline latency budget from one `STRATA_TIMELINE` run (#33). See below. |

## `ab.py`

`ab.py` runs every prompt through the arms in an ABBA order. For each run it records:

- the exe's sha256,
- the greedy output,
- the per-stage counters.

The summary then states whether the outputs were identical across arms.

Measurement rules from AGENTS.md:

- **Priority.** Every arm runs at `--pool-priority 2` and in the HIGH process priority class. `@normal` in an arm's args opts that arm out.
- **Exe snapshots.** Each arm's exe is copied into the output directory first, so a rebuild in place cannot change it mid-run.
- **Completion.** Wait for the `DONE` line before a build or another measurement.

The prompts are the codex reference runs: `%TEMP%/strata-codex-dispatch-detail-{code,thai}256/on.command.json` and `strata-codex-compact-parity-{sky,long}256`. These directories are not in the repo. Each holds the exact `strata generate` command and its `prompt.ids`.

```sh
python tests/xeno/perf/launch_ab.py my-ab "A=path/old.exe" "B=path/new.exe|--some-flag 1" --prompts code,thai --order A,B,B,A
python tests/xeno/perf/ab.py --summarize my-ab code,thai
```

## `timeline.py` - where the time goes, in one run (#33)

`STRATA_TIMELINE=<file>` makes the engine record every thread and GPU lane of the pipeline on one clock, as a
Chrome trace JSON file. Under `serve/server.py`, the server writes `<file>.server.json` beside it. The analyzer turns
the file into a budget:

```sh
STRATA_TIMELINE=run.json strata.exe ...            # generate, or --serve (the file grows after every request)
python tests/xeno/perf/timeline.py run.json         # + --json summary.json, --perfetto closed.json
```

| lane | what it records |
|---|---|
| `main` | startup steps (the `mem_mark` points); per request: the request, the prompt parts, and every decode round. Inside a round: `adapt apply`, the verify window, `emit tokens`, `mtp draft`, `adapt join`. Inside the window: `verify stage`, `verify launch`, then per layer `wait gpu` (the doorbell) and `cpu experts`, with the dispatch stages (`dispatch plan`, `act quantize`, `dispatch jobs`, `cpu pool`, `4070 finish`) and the pool phases (`pool gate/up`, `pool quantize`, `pool down`). Then `verify tail`, `head sampling` and `verify commit`. In a prompt: `prefill run`, then per chunk and per layer: `router sync`, `host grouping`, `expert launches`, `copy issue`, `stager wait` and `wait issuer`. |
| `gpu0 compute (prefill)` | device time of every prompt-path phase, per layer. These are the `STRATA_PREFILL_TIMING` marks: a gap in which the stream waits is charged to the phase that waited. |
| `gpu0 copy engine (prefill)` | the device time of every expert copy (`copy pinned` / `copy staged` / `copy peer`), with its entry index and layer. |
| `gpu1 4070 experts` | the 4070's share of each decode layer (device time). |
| `prefill copy issuer`, `prefill stager N`, `pool worker N`, `4070 launcher`, `adapt`, `ple read-ahead` | the helper threads: issuer waits, stager memcpy and buffer waits, each worker's `wake` (from the publish) and `drain`, and the 4070 enqueue. |
| `server ...` | `http request`, `template+tokenize`, `queue wait`, `engine request`, `first token`. |

The report covers:

- **Prompt path:**
  - GPU time by phase and by layer.
  - The host thread's budget.
  - The copy engine's busy time, with every idle gap split into three kinds: `not issued yet (host)` (plus what the issuing thread was doing instead), `in the issue call`, and `issued, not started (device)`.
  - Copy durations by the compute phase that runs at the same time.
  - The largest gaps, by layer.
- **Decode:** ms per round of every span, the unaccounted rest, the slowest layers, the GPU lanes' busy time per round, and the pool's wake-up latency and stragglers.
- **Helper threads and lane utilisation.**
- **Server requests,** from HTTP to first token.

**Cost.** The timeline changes the thing it measures, so it is for splitting a stage, like Nsight Systems, and not for
A/B speed numbers. Those still come from paired, profiler-off counters (AGENTS.md). On the 8,023-token prompt, a
same-session ABBA gave prefill at 12.78/12.16 s with the timeline on against 11.95/11.55 s off (about +6 %), with
identical output. Decode was within noise (`%TEMP%/strata-claude-stage/TL*.stdout`). With the timeline off, the exe
matches the pre-timeline build within the run-to-run spread (8K prefill 12.27 vs 12.07 s, code256 decode 84.26 vs
84.42 tok/s, 4+4 alternating runs). Details are in `docs/reports/2026-09-29-pipeline-timeline.md`.

**A copy's start is recorded right before the copy, never before a host wait.** Recorded before the stager wait,
it charged the host's wait to the copy engine, and it produced a plausible false finding: "copies stall during
`gdn`" (see the report's correction). Keep every new device span's start event after any host-side block.

The GPU lanes are placed on the host clock by an anchor event. WDDM batches submissions, so the anchor forces the
submission and keeps the narrowest of eight tries. Without that, spans landed 4 ms early (`xeno_timeline_gpu`).
