# Open work

Strata-xeno PRD: xenodeve/Qwen3.8-Flash-Next-Tuning#1. Engine implementation and measurement issues are paired. The approved 2026-09-27 handoff is the phase scope.

| Phase | Engine | Measurement | Current state |
|---|---|---|---|
| 1 GPU expert correctness | xenodeve/Strata-xeno#1 | xenodeve/Qwen3.8-Flash-Next-Tuning#10 | Parked: preserve old CPU tokens or adopt bit-exact canonical exp contract; candidate local branch xeno/afk-phase1 at 8c28ad0, not pushed |
| 2 CPU core policy and dual architecture build | #2 | #11 | Built and reviewed on xeno/afk-phase2; PR not opened |
| 3 Static 4070 expert tier | #3 | #12 | Depends on Phase 1 decision |
| 4 Exclusive swaps | #4 | #13 | Depends on Phase 3 |
| 5 4070 prefill | #5 | #14 | Depends on Phase 3 |
| 6 Claude Code serving | #6 | #15 | Upstream #7–#9, count_tokens, billing-header normalization, thinking signature, loop guard, and document blocks green; #10 parked for token-5 divergence; further compatibility open |
| 7 Long context | #7 | #16 | Depends on memory fit |
| 8 Speculation and sampler | #8 | #17 | Depends on correctness |
| 9 Telemetry and default decision | #9 | #18 | Final measured gate |

The source audit at Flash-Next commit `8bf2c853417c5d381c1a4d4d1c20c3f55b5ad1c0` supersedes the earlier report at `ced0445`.
