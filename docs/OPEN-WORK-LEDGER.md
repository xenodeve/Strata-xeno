# Open work

The authoritative scope is Flash-Next PRD #1 and the approved build handoff of 2026-09-27. The developer selected Strata and authorized AFK implementation on branches; no PR, merge to main, or default-profile change is authorized.

| Phase | Engine issue | Measurement issue | State |
|---|---|---|---|
| 1 Cache correctness / serving fixes | xenodeve/Strata-xeno#1 | xenodeve/Qwen3.8-Flash-Next-Tuning#10 | Cache parity green on four 256-token prompts; upstream serving fixes and review pending |
| 2 CPU affinity / architecture build | #2 | #11 | Pending |
| 3 Static second GPU | #3 | #12 | Depends on 1–2 |
| 4 Exclusive swaps | #4 | #13 | Depends on 3 |
| 5 Second-GPU prefill | #5 | #14 | Depends on 3–4 |
| 6 Claude Code compatibility | #6 | #15 | Pending |
| 7 Long context / sessions | #7 | #16 | Depends on memory fit |
| 8 Speculation / sampler | #8 | #17 | Depends on correctness |
| 9 Telemetry / default decision | #9 | #18 | Final measured gate |

Use the source-audit report at Flash-Next commit `8bf2c853417c5d381c1a4d4d1c20c3f55b5ad1c0`, not the superseded `ced0445` report. The actual runtime includes whole-token and verifier-window graphs, and adaptive replacement above the cache allocator.
