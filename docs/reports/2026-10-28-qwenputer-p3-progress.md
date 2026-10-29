# P3 Native CUDA circuit checkpoint

The actual compiled FFN/GDN tensor path now runs on the primary
RTX 5060 Ti. The language decoder is not integrated and free-running execution is
P4 work. No host calculator or opcode emulator supplies these CUDA predictions.
P3's listed NVIDIA gates passed; the local checkpoint keeps later gates open.

| Gate | Actual result | Evidence in task temporary directory |
|---|---|---|
| Executor stub RED | `executor_init:not_implemented`, exit 1 | `p3-step-red.log` |
| First real circuit replay, 17*23 | 1,543 supplied steps; 4,502,474 residual and 3,211,264 state values, zero mismatches | `p3-multiply-01.log` |
| Full frozen development replays | 79 cases, 74,704 steps, zero residual/state/prediction mismatches; 826,326 lag values match | `p3-native-replays-01/report.json` and per-case logs |
| Full GPU token embedding | 263,254,200 values bit-identical to verified CPU, machine state unchanged | `p3-feature-green.log` |
| Analytic FFN/GDN controls, no firmware | Five cases, query 0/1/2/128, keep/half/clear, beta 0/1, key/no-key, zero norm and ordered collisions: pass | `p3-analytic-*.log` |
| Full final-build repeat | Same 79 cases / 74,704 steps / 826,326 lag values pass again | `p3-final-repeat-02/report.json` |
| Coefficient negative control | Altered owner-node coefficients produce 1,483 mismatches and numeric error 16, exit 1 as expected | `p3-negative-coefficient.log` |
| Compute Sanitizer | Real six-step store-word case: memcheck, initcheck, synccheck all zero errors; racecheck zero hazards | `p3-sanitizer-pipe-{tool}.log` |
| Final Python/native input suite | 94 passed, 4 skipped, 99 subtests, 16.59 seconds | `p3-final-python-suite-02.log` |

The 79 cases are 29 math/error cases and 50 instruction/fault fixtures, including
all eight synthetic fault cases. The math corpus retains 21 solved cases and
eight explicit errors/refusals; a firmware refusal is not counted as solved math.
GPU-predicted text includes `391`, `0.3` and `-3` for the corresponding development
cases. These are supplied-trace replay checks, not free-running acceptance.

Broad-replay executable SHA256:
`3f6f18acd8abe95ce6279db18839ed518481a51682d595d56523156951e530b8`.
Final repeat/sanitizer executable SHA256:
`932fa7b049c2e7e61c2d9db21299df325e9b336d6cee1710388de0b4bb35a9fe`.
GPU feature-test executable SHA256:
`357b8b048c49f228f7d38d2dff34d06bc3810feea587afc83ba3110edba6691d`.
Artifact SHA256 remains
`edcd063736a3b3575cf81b5bac524943e0694e23c12af0cd261b44b150e704b0`.
Current circuit allocations are 31,787,872 bytes for one mutable state set. Graph
driver resources and later shadow/base sets are additional; this is not a full
integrated-model fit measurement.

The sparse GPU path uses explicit rounded FP32 multiply/add/sqrt/divide,
per-target no-FMA/no-FTZ flags, ordered FFN/output reductions and per-stage lag
history. It uses no floating-point atomics. Host schedule construction verifies
the original and level-batched input producers before GPU allocation. Original
dimension levels alone are insufficient because final code outputs overwrite
some token-feature dimensions.

## Sanitizer transport diagnosis

The initial memcheck could not attach and the headless wrapper timed out after
240 seconds, then drained its owned process tree. This was not a memory-check
pass. With the same short analytic fixture and executable, default IPC reproduced
the attach failure; the documented per-process environment setting
`NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE=named-pipes` passed. All four
tools then passed on the real full-circuit store-word replay. The commands use
`--launch-timeout 10 --kill --error-exitcode 99` so connection failure cannot
silently leave a target running. No firewall, HAGS, TDR or security settings changed.
The installed NVIDIA Compute Sanitizer documentation describes this override for
local frontend/target connection problems. The lower-level cause of the default
transport failure was not established.

Sanitizer coverage is the six-step full circuit plus an analytic memcheck control,
not every possible program. Racecheck does not prove global floating-point
determinism; the ordered reduction contract and exact repeat checks cover that.

## Scope and next phase

The final Python suite supplied the pinned firmware and real Gyro tokenizer data.
Its four skips are two P0 GPU-probe integration tests (not selected in that unit
invocation), privileged symlink creation and sandbox-denied junction creation.
Prior P0 real probe and P1 junction evidence remains recorded; these skips are
not new pass claims. The earlier narrower invocation (`p3-final-python-suite.log`)
had 81 passes / six skips because firmware/tokenizer inputs and the two related
tokenizer test modules were not selected. Raw logs from both invocations remain.

P4 adds bounded device-driven trace execution. No held-out acceptance cases have
been executed or used for tuning. No language-model parity, utilization or
practical speedup is claimed here. Diagnostic replay timing includes per-token
submission and residual capture; reference/checksum comparisons run outside it.
The new code is isolated from the default engine and served hub configuration.
CUDA-only tests are selected with `STRATA_BUILD_QWENPUTER_NATIVE_TESTS=OFF` by
default. HIP/SYCL Native execution remains unsupported and untested; their default
engines do not compile or link this optional library.
