# tests/xeno/perf — the measurement tools

These are the scripts behind the 2026-09-28/29 reports in `docs/reports/`. They are kept here so that every speed claim can be re-run from the repo (issue #24). The parser test is `tests/xeno/test_ab_parse.py`.

| script | what it does |
|---|---|
| `ab.py` | Same-session ABBA runner (see below). |
| `launch_ab.py` | Starts `ab.py` fully detached, so it survives the shell that launched it. |
| `route_tools.py` | Routing traces from `strata generate --route-trace`: counts, STRP profiles, static-tier simulation. |
| `blend_profile.py` | Blends EXL3 router counts with Strata traces, and evaluates the blend on held-out benchmark traces. |
| `adapt_sim.py` | Compares static placement, an oracle, and decayed-LFU swaps with a per-round budget. |
| `make_trace_prompts.py`, `trace-prompts/` | Training prompts for routing traces, disjoint from the benchmark prompts. |
| `collect_traces.py` | One traced run per training and benchmark prompt. |
| `h2_nsys.py` | Nsight Systems capture of the verifier decode, then kernel time per device and per class. |
| `gguf_sizes.py` | Tensor bytes from GGUF headers, split into experts and trunk. |

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
