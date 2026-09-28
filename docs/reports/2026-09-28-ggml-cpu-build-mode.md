# ggml-cpu SIMD mode, and why accepted token IDs drifted

## Question

On 2026-09-28 every run of the unchanged baseline (`f653657`) produced outputs that differ from Codex's accepted 256-token IDs recorded the same morning. The first differing token was code 237, Thai 60, sky 5 and long 29. K0 was excluded as the cause because A and B agreed with each other. The next suspect was the ggml-cpu build mode, because the builds genuinely differ there:

- Codex's build dirs (`afk-build`, `afk-dual-build`, `gpu-bottleneck-*`) compile ggml-cpu with `-DGGML_CPU_GENERIC` (no SIMD).
- Claude's (`afk-claude-build`, `q2-*`) compile it with `/arch:AVX2 GGML_AVX2 GGML_FMA GGML_F16C`.

The cause of the split: `CMakeFiles/*/CMakeSystem.cmake` records `CMAKE_SYSTEM_PROCESSOR ""` in Codex's builds and `"AMD64"` in Claude's. The likely reason is that `PROCESSOR_ARCHITECTURE` was missing from the configuring shell's environment. ggml's `ggml_get_system_arch` then does not recognise x86 and falls back to the generic backend. `GGML_NATIVE=ON` in both.

**Not affected:** the native Q2_0 CPU expert kernel is Strata's own AVX-VNNI code. Every log in both groups reports `native Q2_0 expert rows use AVX-VNNI`. ggml-cpu is used only for auxiliary work (for example `from_float` / `vec_dot` in `native_expert.cpp`).

## ggml-cpu mode: three builds of the same K0 source

**Builds** (all from `ab93dfd` source, verified in `build.ninja`):
- **G** `q2-kernel-generic-build`: configured with `PROCESSOR_ARCHITECTURE` unset → `GGML_CPU_GENERIC`.
- **X** `q2-kernel-build`: `/arch:AVX2`.
- **V** `q2-kernel-vnni-build`: `/arch:AVX2 GGML_AVX_VNNI` (`-DGGML_AVX_VNNI=ON`).

**Runs:**
- Order G X V V X G, profiler off, command as in `2026-09-28-q2-rows-per-warp.md`.
- Artefacts: `%TEMP%\strata-claude-ggml` (code; runs 6 and later failed to start with `0xC0000142` after the orchestrating shell was reaped for low system RAM, so code has G×1, X×2, V×2) and `%TEMP%\strata-claude-ggml-thai` (all six).

| Prompt | Arm | tok/s (runs) | wait for rings | pool | CPU pool | CPU rows GB/s |
|---|---|---|---:|---:|---:|---:|
| code | G | 66.2 | 21.2 | 21.5 | 15.9 | 21.2 |
| code | X | 67.1 (66.4, 67.8) | 21.3 | 20.6 | 15.0 | 22.5 |
| code | V | 65.6 (66.5, 64.8) | 21.5 | 21.7 | 15.9 | 21.3 |
| thai | G | 37.3 (35.6, 39.0) | 14.6 | 18.6 | 14.4 | 21.7 |
| thai | X | 37.8 (38.6, 36.9) | 14.6 | 18.2 | 13.8 | 22.4 |
| thai | V | 38.6 (38.1, 39.2) | 14.5 | 17.4 | 13.1 | 23.6 |

**Result:**
- All three modes produced **identical tokens** in every run (code and Thai).
- Throughput differs by **< 4 %**, below the 13.6 % gate. No stage moved beyond run-to-run spread.
- **The ggml-cpu SIMD mode affects neither output nor speed on this workload. It is not the cause of the drift, and it is not worth tuning.**

## The drift is run-environment, not build

Codex's `gpu-bottleneck-build\strata.exe` (sha256 `7945d11d…`) is unchanged since 06:57. Rerunning it with the byte-identical command of `strata-codex-secondary-unprofiled-code256` (artefacts `%TEMP%\strata-claude-oldexe`):

| | this morning (Codex) | 11:10 rerun |
|---|---|---|
| output vs the morning run | — | **first differs at token 237** |
| prefill `experts streamed` | 1928 | **1923** (resident 2362 both) |
| 4070 SUPER lower free at boot | 1.05 GiB | 1.29 GiB |
| load speed | 0.69 GiB/s | 0.98 GiB/s |

**What this shows:**
- The same binary on the same inputs produced different tokens at a different time. The first observable divergence is already in prefill, where a different set of experts was routed.
- Within one session the output is stable: every run on 2026-09-28 from ~09:40 onwards (A, B, G, X, V, and both Codex exes) produced identical tokens per prompt.
- The mechanism is **unproved**. A boot-state-dependent choice in prefill (for example a GPU library picking an algorithm by available memory) is a hypothesis to test, not a finding.

## Rules that follow

- **Judge parity against a baseline run in the same session** (ABBA with the unchanged build), never against token IDs recorded in an earlier session. An old accepted-ID file is a regression hint, not a gate.
- **Record the exe sha256 with every run** and check it before reusing a build dir. `gpu-bottleneck-compact-build` was rebuilt after its accepted runs, so its current binary (`5284946…`) is not the one that produced them (`a3774b6…`).
- **ggml-cpu mode is irrelevant for speed.** Leave it as configured, but know that `CMAKE_SYSTEM_PROCESSOR ""` silently yields `GGML_CPU_GENERIC`.
- **Open item: prefill routing depends on boot state.** Find out what differs in prefill between boots, because it also changes speculative acceptance and therefore speed comparisons across sessions.
