# Open work

Strata-xeno PRD: xenodeve/Qwen3.8-Flash-Next-Tuning#1. Engine implementation and measurement issues are paired. The approved 2026-09-27 handoff is the phase scope.

| Phase | Engine | Measurement | Current state |
|---|---|---|---|
| 1 GPU expert correctness | xenodeve/Strata-xeno#1 | xenodeve/Qwen3.8-Flash-Next-Tuning#10 | Developer accepted Q2_0 `double-exp` output as current baseline (ADR 0001); candidate branch xeno/afk-phase1 pushed at 9ffb846; cache parity green; integration review pending |
| 2 CPU core policy and dual architecture build | #2 | #11 | Built and reviewed on xeno/afk-phase2; PR not opened |
| 3 Static 4070 expert tier | #3 | #12 | Opt-in secondary Q2 compute matches accepted Phase 1 baseline on four 256-token prompts; 256 MiB tier helped 3/4 rough pairs, code slower. 100 ms lifetime monitor saw >=9.18 GiB lower free in full sky run and exits owning process on sampled breach. Graceful fallback, paired ABBA, long-session and default-choice gates remain open |
| 4 Exclusive swaps | #4 | #13 | Static opt-in primary ownership released 6.43 GiB host pages and lowered private commit 43.37→36.94 GiB with sky 96/96 parity; newcomer/victim swaps and recovery still open |
| 5 4070 prefill | #5 | #14 | Depends on Phase 3 |
| 6 Claude Code serving | #6 | #15 | Upstream #7–#9, count_tokens, billing-header normalization, thinking signature, loop guard, stop sequences, dead-child health, and document blocks green; #10 parked for token-5 divergence; further compatibility open |
| 7 Long context | #7 | #16 | Depends on memory fit |
| 8 Speculation and sampler | #8 | #17 | Depends on correctness |
| 9 Telemetry and default decision | #9 | #18 | Final measured gate |

The source audit at Flash-Next commit `8bf2c853417c5d381c1a4d4d1c20c3f55b5ad1c0` supersedes the earlier report at `ced0445`.
