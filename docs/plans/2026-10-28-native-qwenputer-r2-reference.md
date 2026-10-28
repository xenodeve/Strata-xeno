> Frozen R2 reference. R3 supersedes the Swift test baseline with verified Gyro-M; old missing-Swift findings are historical. Technical contracts and source citations below are preserved verbatim. This local reference is not a new publication.

# PRD: Native Qwenputer for Strata Flash-Next

**Date:** 2026-10-28  
**Revision:** R2 after parent review: readability, D2x compatibility, policy references, dual-architecture builds and numerical/measurement gates.  
**Status:** R2 plan/seams approved; P0 authorized and in progress.  
**Tracker:** [PRD #276](https://github.com/xenodeve/Strata-xeno/issues/276), `ready-for-agent`; body read back and verified.  
**Implementation plan:** [task-by-task plan](../superpowers/plans/2026-10-28-native-qwenputer.md)

<!-- lang:en -->
## Problem Statement

Strata serves Qwen3.8 Flash-Next with GPU/RAM/NVMe expert placement, but has no compiled-computation circuit in its decoder. The user wants accurate arithmetic produced by a real circuit inside inference, with the least practical latency, then a possible Universal phase. A host calculator, an emulator answering production requests, or downloading the merged Qwen3.5 model would not satisfy this Native stage.

The user approved a primary-GPU native C++/CUDA sparse FFN/GDN executor, separate circuit weights, GPU-driven trace/token selection, and frozen language state during machine-only steps. This approves the architecture, not an unmeasured speed claim.

## Solution

Load a versioned circuit artifact once, validate it before device allocation, and execute its compiled FFN/GDN operations within Strata's decode controller. A bounded standalone CUDA conditional graph generates internal trace tokens and machine-owned text. Only text enters the language model; trace steps do not advance its cache, sampling position, penalty history or PLE history. Feed the machine's completed response into the exact language decode path, then resume normal generation and MTP.

The production path is compiled sparse tensor arithmetic, not instruction emulation. Python and a RISC-V reference may compile artifacts or generate test oracles offline only. Artifact packaging is independent of steady-state latency when both loaders produce identical resident tensors, placement and graphs.

Keep the feature opt-in. Disabled execution retains the previous prompt, expert budgeting, protocol and token behavior. Enabling it preserves offloading mechanisms but reserves real additional GPU state; any resulting expert-slot/latency cost must be measured and reported.

## User Stories

1. As a Strata user, I want exact supported arithmetic so that the answer is computed rather than guessed.
2. As a user, I want the model to use the circuit internally so that no external calculator service is needed.
3. As a user, I want the model to continue reasoning after the result so that calculation is part of one answer.
4. As a user, I want ordinary text generation to retain its behavior when the feature is disabled.
5. As a user, I want normal text overhead with the feature enabled measured separately from calculation time.
6. As a user, I want existing GPU/RAM/NVMe expert offloading preserved so that my large model remains usable.
7. As a user, I want computation status shown immediately so that I know the model is working.
8. As a user, I want refresh/reconnection to retain that status and the running answer.
9. As an API client, I want native calls kept internal so that my application does not receive an external calc tool request.
10. As an API client, I want non-native tools to keep their existing dispatch behavior.
11. As a user, I want previous tool-call text and quoted text to be read as history, not executed on prefill.
12. As a user, I want overlong or unsupported input rejected explicitly so that truncation cannot produce a plausible wrong answer.
13. As a user, I want cancellation and limits to stop computation without publishing a partial number as success.
14. As a user, I want errors distinguished from arithmetic results and from a broken engine.
15. As a user, I want consecutive calculations to retain correct machine state and return ownership correctly.
16. As a user, I want conversation switching and prefix reuse to preserve the correct internal calculation history.
17. As a user, I want saved sessions to restore both model and circuit state or refuse incompatible data before mutation.
18. As an operator, I want artifacts bound to the correct tokenizer, firmware and semantics so that incompatible packs fail clearly.
19. As an operator, I want malformed artifact/firmware files refused before large allocation or GPU upload.
20. As an operator, I want bounded GPU work that keeps Windows responsive without changing TDR or security settings.
21. As an operator, I want startup, peak memory and expert-hit changes reported on the actual serving hardware.
22. As a developer, I want exact circuit replay against an independent oracle and tests for every ownership transition.
23. As a developer, I want disabled parity and same-session latency evidence before any rollout decision.
24. As a developer, I want the Native contract documented so that a later Universal phase does not weaken it.

## Verified Baseline and Research Boundaries

- Source baseline: `e5737a27f485773a7ce22784fef6af7b480dcc9a`; tracked tree was clean. Existing untracked work is preserved.
- Pinned upstream: `divagr18/qwenputer@ca13310e2f9f788d1dd3feeea8213c1b0c23913c`; source was read, not executed. Its compiler/core is distinct from the dense Qwen3.5 merged loader [U1-U4].
- GPU inventory: physical GPU 1 is RTX 5060 Ti 16,311 MiB; physical GPU 0 is RTX 4070 SUPER 12,282 MiB. Existing D2x mapping makes the 5060 the logical primary. Do not infer device identity from an index alone.
- Existing build cache names CUDA 13.3, MSVC 14.44 and architecture 120; that cache is not sufficient evidence for a full dual-GPU D2x binary. Full baseline/candidate builds must include sm_89 and sm_120, as documented by pinned CMake lines 9 and 156-159. Installed headers contain conditional/WHILE definitions, but no conditional create/instantiate/run probe has executed.
- Local Python is 3.11.9; torch 2.14.0+cpu, transformers 5.14.1, numpy 2.4.6, safetensors 0.8.0. The upstream HF extra asks for newer torch/transformers. The planned offline compiler must first establish its minimal dependencies; a full HF/GPU-PyTorch install is not presumed necessary.
- D2x tokenizer has 248,320 entries. Its 554-byte descriptor has a string `model: gpt2`; actual vocabulary/merges/types are separate files. Upstream `Text` expects an HF Tokenizer JSON. A Strata tokenizer adapter is required, rather than passing the descriptor unchanged.
- Observed special IDs are tool_call 248058, /tool_call 248059, tool_response 248066, /tool_response 248067, think 248068, /think 248069. These are evidence for this pack, not constants for arbitrary models.
- Configured D2x executable exists (SHA256 `da93215c3eda1ae9e8e1d50ec7872696a9d502d51e9690fabdae48f499b005ed`), expert pack and tokenizer exist. The configured Swift native/PLE first GGUF shard is absent at its configured path; that directory currently contains mmproj files. End-to-end tests are blocked until the required model inputs are located or separately approved for restoration/download.
- Baseline protocol/session/timeline tests ran: 65 passed, 1 skipped, 19 subtests passed, exit 0. Initial sandbox loopback failures were isolated and the identical suite passed with per-command loopback access. This is neither GPU parity nor a Native test.
- The current D2x config uses --secondary-expert-mib 4100, --spec 4, STRATA_PREFILL_EXPERT_SPLIT=1 and STRATA_PREFILL_WAVE=1; --layer-split, --batch, --slots and --pipeline-windows are absent. This supported dual-GPU expert/prefill offload route is distinct from splitting decoder layers. Its elastic VRAM cache path refuses fork tiers; use a static circuit reserve first [R1,R3].
- No Native throughput, memory peak, call-use rate or arithmetic success rate has been measured. Shared-link assistant messages remain unavailable; the user's supplied text and approved architecture are the authority.

## Implementation Decisions

1. **Native scope and ownership.** Support the existing single-decoder Flash-Next D2x path, including secondary expert offload, MTP and suffix drafting. GPU-generated internal trace IDs use a separate token kind, never an extension exposed to client token IDs. Explicitly refuse feature-on decoder layer-split, multi-request batch and pipelined-window modes before allocation in v1; this does not refuse the current D2x secondary expert tier or STRATA_PREFILL_EXPERT_SPLIT path. Feature-off support remains unchanged.
2. **Offline compiler.** Reuse the pinned RV32IM/Zba/bignum FFN/GDN compiler with a Strata vocabulary adapter and a reviewed native-call eligibility change. Preserve the circuit's arithmetic parser. Compile/park firmware offline and export a ready snapshot; production startup never runs a host emulator. Preserve provenance and required notices. A 29,932-byte pinned calc-bridge ELF fixture exists upstream, but matching source/build/provenance must be verified before use.
3. **Artifact ABI.** Propose little-endian `STQPUT01`, schema 1.0, checked section directory and typed FP32/int32 arrays. Include compiler commit plus local patch digest, numeric ABI, model contract/tokenizer digests, firmware SHA256, op ordering, dimensions, CSR/scatter arrays, feature-slot descriptors, byte/printable maps, initial state and declared resource limits. No pickle, executable payload or arbitrary Python callback is accepted.
4. **Feature lowering.** Export 16-byte text rows plus original byte lengths and special/printable maps, and lower only known circuit feature kinds. Do not materialize a dense vocabulary-by-residual embedding matrix. Prove all exported token-feature values against the compiler over the supported trace vocabulary and all 248,320 text IDs.
5. **Loader.** Validate magic/version/endian, file/section integrity, overflow-safe sizes, indices, row pointers, finite coefficients, head layout and configured memory caps before GPU allocation/upload. Firmware ingestion also checks ELF32 little-endian RISC-V, entry, segment bounds/overlaps and filesz/memsz. Limit auxiliary host memory during validation. Reject unsupported ABI/binding before state mutation.
6. **Exact operations.** Preserve coefficient bits, sparse products, ReLU, decay, beta/write/read rules, argmax tie rules, lag history, normalization and rounded sqrt/division. Start with deterministic per-output reductions and fixed dot-product trees. Upstream FP32 atomics do not establish determinism. No quantization, TF32, fast-math, lossy state, or reduction reordering without an exactness proof and oracle gate.
7. **Device control.** Use a standalone top-level conditional WHILE graph on the primary device, with an initialized condition every launch and a finite chunk step/output budget. No host callback, allocation/free, CDP or device-graph launch occurs in its body. Do not clone the conditional graph or graft it as a child of the verifier graph. v1 serializes the active executor; parked sessions own separate saved states and graph addresses are rebound/rebuilt when storage changes.
8. **Feasibility gate.** Before relying on conditional graphs, run real Windows create/instantiate/launch/relaunch tests for zero/one/N trips, stop conditions and repeated sessions on the installed driver. If unsupported, stop and ask the user about a fixed-count regular graph with GPU active/done masks and host supervision per bounded chunk. That fallback remains native arithmetic but is not silently substituted for the approved runtime.
9. **Transactional decode.** Define ENTER after a consumed input row r, where the circuit owns the next token. Commit at most r+1 input rows, not the emitted-token count. A speculative shadow circuit evaluates the candidate window; normal acceptance and the first valid takeover clamp the prefix. Commit the circuit by replaying only that accepted prefix from its retained base state; discard the rejected shadow suffix. Validate this correctness-first cost before replacing it with a journal optimization. Fence outstanding commit/cross-stream work before handoff, snapshot or host access.
10. **Language state.** Machine-only trace steps do not change language KV/GDN/PLE, logical positions, RNG counters or penalty history. Machine-owned text advances language state once in its original order. Initially consume that text through exact verifier windows; ordinary batched prefill is allowed only after equivalence is proved. Keep the machine output unpublished until complete success, then resume language sampling/MTP at the correct position.
11. **Bridge contract.** V1 uses the existing model's native XML call syntax, with an explicitly enrolled internal calc capability and eligibility/role gates. Other tool names and historical/quoted examples cannot start a machine burst. Preserve upstream expression semantics, including Python floor division/modulo, rational division, limited non-negative powers, gcd and decimal +/−/* where supported. The pinned source has 80 capture tokens, 16 bytes per row, a 255-byte expression payload and MAXD=62; reject any overflowing/truncated input, unsupported operand/operator or embedded NUL before it can be treated as successful arithmetic. Any extension changes the compiler ABI and oracle corpus.
12. **API and canonical history.** Internal call/result syntax must never leak into external tool dispatch. Preserve a verified mapping from visible assistant messages to the canonical internal token transcript per cache slot. Bind it to message-prefix hashes, artifact identity and session identity. Matching continuations reuse canonical tokens; ambiguous/untrusted histories cold-read safely and never splice guessed internals. Prefill history updates bridge history without replaying earlier calculations. This map must survive the session persistence modes supported in v1.
13. **Sessions and failures.** Save only at a coherent quiescent boundary in v1; an active burst save is explicitly busy/refused or waits within an existing bounded queue. Saved identity includes artifact/numeric ABI, compiler patch, tokenizer, firmware and state geometry. Store RAM/PC/registers/counters, lag history, bridge/ownership and canonical-history metadata; never graph handles. A partial device restore invalidates the entire context. Limit/cancel/fault discards unpublished output and restores the retained coherent transaction boundary or invalidates the session; it never silently continues from partial state.
14. **Scheduling and resources.** Compute actual resident weights, committed/shadow/init states, histories, FIFO, scratch, graphs and transient restore peaks before reserving GPU memory and filling expert caches. Active execution stays on the primary GPU; NVMe holds artifacts and cold snapshots only. Preserve the display card's 2.5 GB total headroom and the existing offload mechanism. Report against the existing 42 GB private-commit budget (AGENTS.md:295-299) and measured baseline, explicitly noting a baseline already over budget. The 2.5 GB rule is total display headroom, including current desktop use (AGENTS.md:323-324), not a requirement for 2.5 GB free VRAM. Do not edit served profiles/launchers.
15. **Observability and rollout.** Reuse the existing engine protocol, timeline and live-run status path. Emit computing/return/error transitions when they begin; refresh must show the current step. Add timeline spans and paired counters for bridge, trial/commit, entry, trace, result replay and host waits. Keep native enabled only in an experimental test config until user approval of evidence; disabled behavior remains unchanged.

## Testing Decisions and Proposed Primary Seam

The primary behavioral seam is one consumed text window plus its next-token ownership decision: accepted text in, language/circuit ownership and visible output/state out. Test it through the existing verifier/commit and service request boundaries, rather than exposing many new public APIs. User review of this PRD includes checking that this seam matches the expected native behavior; no additional architecture interview is presumed.

Two lower seams are necessary evidence, not substitutes for the feature: artifact validation must prove refusal before GPU mutation, and circuit replay must prove that real native tensor operations produce the trace/result. HTTP tests alone cannot prove either. Prefer existing conversation validation/transfer, protocol, tokenizer, timeline and slots patterns.

Good tests assert outputs, state ownership/fingerprints, counters or explicit refusal reasons. A deliberate negative control must fail for the named semantic reason; missing scripts/imports or unexecuted tests are not red/green evidence. Oracle calculators/emulators are test-only and must be visibly separate from the production executable's dispatch path. Freeze compiler/patch/corpus digests and seeds, separate development fixtures from held-out free-running acceptance cases, and use an independent integer/Fraction arithmetic oracle not derived from upstream firmware/IR. Exclude oracle/reference/checksum verification from timed turn intervals and compare results offline after timing; assert that the production benchmark path never invokes host arithmetic/reference.

## Change Inventory and Search Boundary

Paths below are proposed change sites, not existing implementation claims; baseline anchors refer to e5737a27.
| Path | Responsibility/change | Verification |
|---|---|---|
| `tools/qwenputer/{export,tokenizer,oracle}.py` (new), `third_party/qwenputer/` (new) | Pinned compiler/provenance, Strata adapter, artifact and test-only oracle | G1/G2 |
| `include/strata/artifact/qwenputer.hpp`, `src/artifact/qwenputer.cpp` (new) | Bounded ABI parser, binding, resource estimate | G1 malformed fixtures/no-allocation |
| `include/strata/kernels/qwenputer.hpp`, `src/kernels/qwenputer.cu` (new) | Exact deterministic sparse FFN/GDN and device token features/control | G2 native step/state replay |
| `include/strata/core/qwenputer.hpp`, `src/core/qwenputer.cpp` (new) | Session/transaction ownership, top-level graph, limits | G0/G3/G5 |
| `include/strata/core/verify.hpp:1-23,66-96`, `src/core/verify.cpp:2012,2407-2433` | Accepted input boundary and async commit dependencies | G3 every ENTER row |
| `src/program/generate.cpp:12672-12679,13903-13928,15509-15543` | Prompt, serve/generate loops, draft suffix discard/replay | G3/G6; no missed loop |
| `src/program/generate.cpp:5919-5935`, cache budget/init sites | Static reserve before expert placement; no elastic assumption | G8 slots/peak/hit split |
| `include/strata/core/{conversation_cache,conversation_file,conversation_snapshot}.hpp`, corresponding `src/core/` sources | Native identity/state, coherent save/restore limits | G4; invalid/partial restore |
| `include/strata/core/coupled_draft.hpp:19-35`, MTP/penalty callers | Preserve counters/history; change only if integration requires it | G3 seeded continuation |
| `serve/native_transcript.py` (new), `serve/server.py:1723-1854,3913-3972,4328` | Native event protocol, canonical prefix mapping, internal ownership | G4/G5/G6 |
| `serve/frontend.py:622-637,832`, `serve/runs.py` | Internal/native vs external calls; reconnect metadata | G5 streaming fragmentation |
| `serve/ui/src/lib/status.ts:37`, i18n/event consumers and tests | Immediate continuous computing/return/error status | G5 real rendered refresh |
| `include/strata/timeline*.hpp`, `serve/timeline.py`, `tests/xeno/perf/timeline.py`, `tests/xeno/test_timeline.py` | Extend analyzer first; spans/counters at new stages | G8 timeline fixtures/off-cost |
| `tests/qwenputer/` (new), `serve/test_native_qwenputer.py` (new), `tests/xeno/perf/qwenputer.py` (new) | Component, ownership, service and measurement gates | G0-G8 |
| `CMakeLists.txt`, `docs/BLUEPRINT.md`, new native guide and reports | Opt-in build/flag/protocol docs, backend guards, evidence | G6/G7/G9 |

Search used git grep/git ls-files across engine, headers, serving, tokenizer, tests, CMake and blueprint (rg unavailable), plus pinned upstream compiler/runtime/bridge/firmware source and primary CUDA/Windows docs. Dynamic imports, all remote issue history, other drives/caches and unavailable backend toolchains were not exhaustively surveyed. Existing unrelated untracked reports/builds/profiles were not incorporated into baseline code.

## Acceptance Gates

| ID | Required evidence; no invented speed target |
|---|---|
| G0 Feasibility | Actual conditional graph probe on primary: 0, 1, N, relaunch/reset, full-buffer, stop and bounded cancel; runtime/driver/device/OS recorded. Unsupported runtime stops the selected path. |
| G1 Artifact | Reproducible schema/export hashes; all feature-map values match compiler; license/provenance manifest; malformed/oversize/incompatible packs rejected before device allocation. |
| G2 Native exactness | Zero semantic trace/result mismatches against the pinned IR/reference corpus and an independent exact arithmetic oracle; every instruction family/error path, repeated runs and frozen held-out free-running cases/seeds are covered separately from development fixtures. Deterministic reduction proof/tests cover colliding outputs. Compute Sanitizer memcheck/initcheck/synccheck and applicable racecheck must run; unsupported paths are reported. Racecheck does not prove global-atomic determinism. |
| G3 Ownership | ENTER at every input-window position, rejected drafts, first/last rows, suffix and MTP transitions; no machine trace advances language state/RNG/penalties/PLE; completed result replays once. |
| G4 History/session | Two interleaved conversations, cached continuation, cold history, history without hidden fragments, save/restore identity mismatch and partial transfer failure; no stale circuit state or hidden-prefix corruption. |
| G5 Safety/status | Bounds, cancel, malformed UTF-8/NUL/overlong/nested/reserved tokens, unsupported syntax, huge powers, fault and GPU failure produce explicit terminal reasons; no partial numeric success; immediate status and refresh are verified. |
| G6 Disabled parity | Fresh unchanged baseline vs candidate disabled, same session greedy raw-token parity over code/Thai/ordinary/long-prefix turns; disabled prompt/budget/protocol/snapshot serialization and no circuit allocation match. |
| G7 Backend/build | Full D2x builds include sm_89 and sm_120, verified binary coverage and an actual secondary-tier smoke before ABBA. Build every backend touched by shared files and preserve Q2 parity (AGENTS.md:44). CUDA real tests come first; unavailable HIP/SYCL execution is explicit implementation-validation debt, not a passed check or a blocker to plan review. |
| G8 Performance | Timeline first; profiler-off ABBA for disabled parity/ordinary enabled overhead and native arithmetic turns. Record same-run stage counters, hashes, priorities, hit split, draft acceptance, startup/peak RAM/VRAM, entry/exit/host waits, disturbed runs and DONE. Timed intervals exclude oracle/reference/checksum verification; compare results offline and assert no host arithmetic/reference in the production path. Report distributions; the existing 13.6% rule (AGENTS.md:320-321) is a repo heuristic, not statistical significance, and under-gate effects remain unproved without consistent paired repetition. |
| G9 Release | No served config/default/deployment change from this PRD. User reviews correctness, resource costs and measured latency before opt-in rollout. Universal remains deferred. |

Arithmetic corpus includes signed add/sub/mul, exact rational division, negative floor/modulo, gcd, precedence/powers, decimals within firmware support, 62-digit capacity boundaries, divide-by-zero, capacity/complexity errors, and multi-call continuation. Separate a forced native-call proof from spontaneous model call-use evaluation; neither implies general coding/3D quality improvement.

## Delivery Order and Dependencies

| Phase | Deliverable | Depends on |
|---|---|---|
| P0 | Preflight, real conditional-graph smoke probe, frozen baseline manifest and model-input inventory | Plan review; task-owned probe only |
| P1 | Pinned licensed compiler adapter, fixture/oracle bundle and reproducible artifact export | P0; specific third-party execution approval |
| P2 | CPU artifact loader/validator and exact token-feature lowering | P1 schema and fixtures |
| P3 | Deterministic single-step CUDA FFN/GDN executor | P2; compiler oracles |
| P4 | Bounded standalone GPU trace/token-selection graph | P0 runtime capability, P3 |
| P5 | Transactional verifier/commit integration, MTP/suffix ownership and prompt-history mode | P4 |
| P6 | Snapshot identity, save/restore and canonical transcript storage | P5 |
| P7 | API-native ownership, live status, refresh and ordinary-tool compatibility | P5/P6 |
| P8 | Full backend/parity/quality tests, D2x timelines and profiler-off ABBA; review report | P0-P7 and available model inputs |

Each phase is test-first, ends with its own evidence and blueprint update, and remains local until publishing is separately authorized. The companion implementation plan names proposed files, exact seams and per-phase tests; this PRD's decisions describe contracts rather than unstable file locations.

## Out of Scope

- Universal model adapters, learned model training, arbitrary architecture loading, a merged Qwen3.5 runtime, and general 3D/coding benchmark claims.
- Host calculator/emulator production fallback, arbitrary C/Python execution, filesystem/network tools or additional privileges.
- Dynamic D2x expert-cache resizing, a circuit on the display GPU, active working-state paging during bursts, or a default change justified by estimates.
- V1 concurrent batch/layer-split/pipelined machine execution and mid-burst session serialization; supported combinations must be explicit.
- Changing serving configs/launchers, TDR/driver/security settings, installing large dependencies/models, publishing tracker entries, push/PR or deployment during this planning/review phase.

## Risks, Approvals and Further Notes

- **First technical stop:** conditional graph feasibility on this Windows driver. Probe failure requires a user decision on the documented native chunk fallback, not silent CPU execution.
- **First end-to-end blocker:** the configured Swift native/PLE GGUF is missing at its path. Locate an existing approved copy first; otherwise present exact files, sizes, source and disk/memory impact for approval. Do not substitute a different quantization/profile or download the merged Qwen3.5 model.
- **Third-party execution:** before running the pinned compiler/reference/firmware or a downloaded binary, present the exact commit/paths, audited imports/commands, purpose, expected writes and resource requirements for action-time approval. Reading source and writing this plan did not execute it.
- **Dependencies:** use existing minimal NumPy-based export where viable. Any install, RISC-V toolchain/WSL use, firmware rebuild or model download is separately reviewed; the existence of wsl.exe does not prove a working Linux toolchain.
- **Resource decisions:** actual artifact/shadow/graph sizes and ordinary-text overhead are unknown. If a reserve cannot fit the approved memory/headroom constraints, report the measured alternatives; do not silently shrink context, change experts/defaults or move the circuit.
- **Licensing:** source MIT notice is verified. Check firmware/libc/libgcc and binary provenance separately; any checkpoint-derived tensors must retain their applicable weight license. Do not assume all generated/binary inputs inherit one license.
- **Planning workflow:** Approved R2 PRD published as [#276](https://github.com/xenodeve/Strata-xeno/issues/276), with full English/Thai requirements and verified body. Implementation issue breakdown remains separate; no PR/push/merge is authorized.
- **Fast mode:** request Fast for non-Astra models where supported. Local defaults read Sol/medium/priority, but the active turn's model/tier is not exposed and no tier setter is available. Do not claim it enabled.
- **Rollback:** feature disabled restores the original runtime path; damaged contexts are discarded, not resumed. Removing the experimental config/artifact leaves served profiles untouched. No roll-forward/default promise follows from a component benchmark.

## Evidence Index

- R1: [Strata blueprint](../BLUEPRINT.md) baseline/revision log and elastic-cache refusal; [ADR 0001](../adr/0001-q2-exp-baseline.md) Q2 correctness contract.
- R4: Existing local policy at [AGENTS.md:44](../../AGENTS.md#L44) (backend builds), [295-299](../../AGENTS.md#L295) (42 GB private commit), [320-321](../../AGENTS.md#L320) (13.6% noise heuristic), [323-324](../../AGENTS.md#L323) (2.5 GB total display headroom); [CMakeLists.txt:9,156-159](../../CMakeLists.txt#L9) (dual architecture configuration). These constraints are not newly proposed thresholds.
- R2: [Discovery and actual local test/resource evidence](../reports/2026-10-28-native-qwenputer-discovery.md); logs at `%TEMP%/strata-qwenputer-discovery-20261028/`.
- R3: [Pinned verifier contract](https://github.com/xenodeve/Strata-xeno/blob/e5737a27f485773a7ce22784fef6af7b480dcc9a/include/strata/core/verify.hpp#L1), [commit](https://github.com/xenodeve/Strata-xeno/blob/e5737a27f485773a7ce22784fef6af7b480dcc9a/src/core/verify.cpp#L2407), [snapshot](https://github.com/xenodeve/Strata-xeno/blob/e5737a27f485773a7ce22784fef6af7b480dcc9a/include/strata/core/conversation_snapshot.hpp#L50), [sampling counters](https://github.com/xenodeve/Strata-xeno/blob/e5737a27f485773a7ce22784fef6af7b480dcc9a/include/strata/core/coupled_draft.hpp#L19).
- U1: [Pinned RV chat](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/chipformer/rv/chat.py#L100) full merged loader; lines 172-184 host token selection; 324-426 ownership.
- U2: [Pinned sparse executor](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/chipformer/rv/tritonexec.py#L47) atomic scatter, rounded normalization and state layout.
- U3: [Pinned bridge](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/chipformer/rv/bridge.py#L45), [core features](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/chipformer/rv/core.py#L109), [ELF loader](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/chipformer/rv/elf.py).
- U4: [Pinned calc firmware](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/chipformer/rv/env/calc/calc.c#L1), [firmware build](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/chipformer/rv/env/calc/build_calc.sh), [MIT license](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/LICENSE).
- C1: [CUDA 13.3 conditional graph requirements](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-node-body-graph-requirements) and installed `driver_types.h`; header presence is not a runtime pass.
- C2: [CUDA conditional-node API restrictions](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/structcudaConditionalNodeParams.html).
- W1: [Microsoft WDDM TDR](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/timeout-detection-and-recovery): the default two seconds concerns scheduling/preemption timeout; it is not a blanket per-graph duration guarantee. Do not change TDR settings.
<!-- lang:end -->

<!-- lang:th -->
# PRD: Native Qwenputer สำหรับ Strata Flash-Next

**Revision:** R2 หลัง parent review: readability, D2x compatibility, policy references, dual-architecture builds และ numerical/measurement gates  
**สถานะ:** อนุมัติแผน/จุดเชื่อมต่อ R2 แล้ว; อนุมัติ P0 และกำลังดำเนินการ  
**Tracker:** [PRD #276](https://github.com/xenodeve/Strata-xeno/issues/276), `ready-for-agent`; อ่านกลับและตรวจ body แล้ว  
**Implementation plan:** [แผนราย task](../superpowers/plans/2026-10-28-native-qwenputer.md)

## ปัญหาที่ต้องแก้

Strata ให้บริการ Qwen3.8 Flash-Next โดยวาง experts ใน GPU/RAM/NVMe แต่ยังไม่มีวงจรที่ compile เป็น weights อยู่ใน decoder ผู้ใช้ต้องการ arithmetic ที่ถูกต้องจากวงจรจริงระหว่าง inference โดยมี latency ต่ำที่สุดที่ทำได้ แล้วจึงพิจารณาระยะ Universal การใช้เครื่องคิดเลขบน host, ให้ emulator ตอบคำขอจริง หรือดาวน์โหลด merged Qwen3.5 model ไม่ตรงกับระยะ Native นี้

ผู้ใช้อนุมัติ native C++/CUDA sparse FFN/GDN executor บน GPU หลัก, circuit weights แยก, การเดิน trace/เลือก token บน GPU และการพัก language state ระหว่าง machine-only steps แล้ว นี่คือการอนุมัติ architecture ไม่ใช่การรับรองความเร็วที่ยังไม่ได้วัด

## แนวทางแก้

โหลด circuit artifact ที่มี version ครั้งเดียว ตรวจความถูกต้องก่อนจัดสรร device memory แล้วประเมิน FFN/GDN ที่ compile ไว้ภายในตัวควบคุม decode ของ Strata ใช้ standalone CUDA conditional graph ที่จำกัดงานต่อรอบเพื่อสร้าง internal trace tokens และข้อความที่วงจรเป็นเจ้าของ เฉพาะข้อความเท่านั้นที่เข้าโมเดลภาษา; trace steps ไม่ขยับ cache, sampling position, penalty history หรือ PLE history ของโมเดล ป้อน response ที่วงจรสร้างเสร็จแล้วกลับผ่าน language decode path ที่รักษาผลเดิม จากนั้นจึงให้โมเดลและ MTP ทำงานต่อ

เส้นทางตอบจริงเป็น sparse tensor arithmetic ที่ compile ไว้ ไม่ใช่การ emulate instructions Python และ RISC-V reference ใช้ compile artifacts หรือสร้าง test oracles แบบ offline ได้เท่านั้น รูปแบบไฟล์ artifact ไม่กำหนด steady-state latency หาก loaders ทั้งสองแบบให้ resident tensors, placement และ graphs เหมือนกัน

feature ต้อง opt-in เมื่อปิดต้องคง prompt, expert budgeting, protocol และพฤติกรรม token เดิม เมื่อเปิดยังคงกลไก offloading แต่ต้องจอง GPU state เพิ่มจริง; หากทำให้ expert slots ลดลงหรือ latency เพิ่ม ต้องวัดและรายงาน

## User Stories

1. ในฐานะผู้ใช้ Strata ฉันต้องการ arithmetic ที่รองรับและให้ผล exact เพื่อให้คำตอบมาจากการคำนวณแทนการเดา
2. ในฐานะผู้ใช้ ฉันต้องการให้โมเดลใช้วงจรภายใน เพื่อไม่ต้องมีบริการเครื่องคิดเลขภายนอก
3. ในฐานะผู้ใช้ ฉันต้องการให้โมเดลคิดต่อหลังได้ผล เพื่อให้การคำนวณอยู่ในคำตอบเดียวกัน
4. ในฐานะผู้ใช้ ฉันต้องการให้การสร้างข้อความปกติคงพฤติกรรมเดิมเมื่อปิด feature
5. ในฐานะผู้ใช้ ฉันต้องการให้วัด overhead ของข้อความปกติเมื่อเปิด feature แยกจากเวลาคำนวณ
6. ในฐานะผู้ใช้ ฉันต้องการคง GPU/RAM/NVMe expert offloading เพื่อให้ยังใช้โมเดลใหญ่ได้
7. ในฐานะผู้ใช้ ฉันต้องการให้แสดงสถานะคำนวณทันที เพื่อรู้ว่าโมเดลกำลังทำงาน
8. ในฐานะผู้ใช้ ฉันต้องการให้ refresh/reconnect แล้วยังคงสถานะและคำตอบที่กำลังทำอยู่
9. ในฐานะ API client ฉันต้องการให้ native calls อยู่ภายใน เพื่อให้แอปไม่รับคำขอเรียก external calc tool
10. ในฐานะ API client ฉันต้องการให้ tools ที่ไม่ใช่ native คงการ dispatch เดิม
11. ในฐานะผู้ใช้ ฉันต้องการให้อ่าน tool-call text เดิมและข้อความที่ยกมาเป็น history โดยไม่รันระหว่าง prefill
12. ในฐานะผู้ใช้ ฉันต้องการให้ปฏิเสธ input ที่ยาวเกินหรือไม่รองรับอย่างชัดเจน เพื่อไม่ให้การตัดข้อความกลายเป็นคำตอบผิดที่ดูน่าเชื่อ
13. ในฐานะผู้ใช้ ฉันต้องการให้ cancellation และ limits หยุดการคำนวณโดยไม่รายงานตัวเลขบางส่วนว่าเป็นผลสำเร็จ
14. ในฐานะผู้ใช้ ฉันต้องการแยก errors ออกจากผล arithmetic และอาการ engine เสีย
15. ในฐานะผู้ใช้ ฉันต้องการให้คำนวณต่อเนื่องแล้ว machine state และการคืน ownership ยังถูกต้อง
16. ในฐานะผู้ใช้ ฉันต้องการให้การสลับบทสนทนาและ prefix reuse รักษา internal calculation history ที่ถูกต้อง
17. ในฐานะผู้ใช้ ฉันต้องการให้ saved sessions คืนทั้ง model และ circuit state หรือปฏิเสธข้อมูลที่เข้ากันไม่ได้ก่อนเปลี่ยน state
18. ในฐานะผู้ดูแล ฉันต้องการให้ artifacts ผูกกับ tokenizer, firmware และ semantics ที่ถูกต้อง เพื่อให้ packs ที่เข้ากันไม่ได้ถูกปฏิเสธชัดเจน
19. ในฐานะผู้ดูแล ฉันต้องการให้ปฏิเสธ artifact/firmware ที่ผิดรูปก่อนจัดสรรหน่วยความจำมากหรืออัปโหลดเข้า GPU
20. ในฐานะผู้ดูแล ฉันต้องการจำกัดงาน GPU เพื่อให้ Windows ยังตอบสนอง โดยไม่แก้ TDR หรือการตั้งค่าความปลอดภัย
21. ในฐานะผู้ดูแล ฉันต้องการรายงาน startup, peak memory และ expert-hit changes บนเครื่องที่ใช้ให้บริการจริง
22. ในฐานะนักพัฒนา ฉันต้องการ exact circuit replay เทียบ independent oracle และ tests ของทุก ownership transition
23. ในฐานะนักพัฒนา ฉันต้องการ disabled parity และหลักฐาน latency ใน session เดียวกันก่อนตัดสินใจ rollout
24. ในฐานะนักพัฒนา ฉันต้องการบันทึกสัญญาของ Native เพื่อให้ระยะ Universal ภายหลังไม่ลดข้อกำหนดนี้

## Baseline ที่ตรวจแล้วและขอบเขต research

- Source baseline คือ `e5737a27f485773a7ce22784fef6af7b480dcc9a`; tracked tree สะอาดเมื่อเริ่ม และเก็บ untracked work เดิมไว้
- ตรึง upstream ที่ `divagr18/qwenputer@ca13310e2f9f788d1dd3feeea8213c1b0c23913c`; อ่าน source แต่ยังไม่รัน ส่วน compiler/core แยกจาก dense Qwen3.5 merged loader [U1-U4]
- GPU inventory: physical GPU 1 เป็น RTX 5060 Ti 16,311 MiB; physical GPU 0 เป็น RTX 4070 SUPER 12,282 MiB D2x mapping เดิมให้ 5060 เป็น logical primary ห้ามสรุปตัวตน device จาก index อย่างเดียว
- build cache เดิมระบุ CUDA 13.3, MSVC 14.44 และ architecture 120 ซึ่งยังไม่ใช่หลักฐานว่ามี full dual-GPU D2x binary ครบ baseline/candidate builds ต้องมีทั้ง sm_89 และ sm_120 ตาม pinned CMake lines 9 และ 156-159 headers ที่ติดตั้งมี conditional/WHILE definitions แต่ยังไม่ได้รัน probe ที่ create/instantiate/run conditional graph
- Python ในเครื่องคือ 3.11.9; torch 2.14.0+cpu, transformers 5.14.1, numpy 2.4.6, safetensors 0.8.0 HF extra ของ upstream ต้องการ torch/transformers ใหม่กว่า offline compiler ที่วางแผนต้องยืนยัน dependencies ขั้นต่ำก่อน โดยไม่สมมติว่าต้องติดตั้ง HF/GPU-PyTorch ทั้งชุด
- D2x tokenizer มี 248,320 entries descriptor ขนาด 554 bytes มี `model: gpt2` เป็น string; vocabulary/merges/types จริงแยกไฟล์ ขณะที่ upstream `Text` ต้องการ HF Tokenizer JSON จึงต้องมี Strata tokenizer adapter แทนการส่ง descriptor เดิมเข้าไป
- special IDs ที่ตรวจได้คือ tool_call 248058, /tool_call 248059, tool_response 248066, /tool_response 248067, think 248068, /think 248069 ตัวเลขนี้เป็นหลักฐานของ pack นี้ ไม่ใช่ constants สำหรับโมเดลอื่น
- D2x executable ที่ config ระบุมีอยู่ (SHA256 `da93215c3eda1ae9e8e1d50ec7872696a9d502d51e9690fabdae48f499b005ed`), expert pack และ tokenizer มีอยู่ แต่ Swift native/PLE first GGUF shard ไม่อยู่ในตำแหน่งที่ config อ้าง; directory นั้นมี mmproj files การทดสอบเต็มระบบจึงติดจนกว่าจะพบ model inputs ที่ต้องใช้ หรือได้รับอนุมัติแยกให้คืนไฟล์/ดาวน์โหลด
- รัน baseline protocol/session/timeline tests แล้ว: 65 passed, 1 skipped, 19 subtests passed, exit 0 ตรวจแยกแล้วว่า loopback failures ใน sandbox หายเมื่อรันชุดเดิมด้วยสิทธิ์ loopback เฉพาะคำสั่ง นี่ไม่ใช่ GPU parity หรือ Native test
- D2x config ปัจจุบันใช้ --secondary-expert-mib 4100, --spec 4, STRATA_PREFILL_EXPERT_SPLIT=1 และ STRATA_PREFILL_WAVE=1; ไม่มี --layer-split, --batch, --slots หรือ --pipeline-windows เส้นทาง dual-GPU expert/prefill offload นี้ต้องรองรับ และแยกจากการแบ่ง decoder layers เส้นทาง elastic VRAM cache ปฏิเสธ fork tiers จึงเริ่มด้วย static circuit reserve [R1,R3]
- ยังไม่ได้วัด Native throughput, memory peak, อัตราที่โมเดลเลือกใช้วงจร หรืออัตรา arithmetic สำเร็จ คำตอบฝั่ง assistant ในลิงก์แชร์ยังอ่านไม่ได้ ให้ยึดข้อความที่ผู้ใช้ส่งมาและ architecture ที่อนุมัติเป็นหลัก

## การตัดสินใจด้าน implementation

1. **ขอบเขต Native และ ownership** รองรับ single-decoder Flash-Next D2x เดิม รวม secondary expert offload, MTP และ suffix drafting internal trace IDs ที่ GPU สร้างต้องเป็น token kind แยก ไม่เพิ่มเป็น client token IDs ใน v1 ให้ปฏิเสธ feature-on ที่ใช้ decoder layer-split, multi-request batch และ pipelined-window modes ก่อน allocation อย่างชัดเจน; ไม่ได้ปฏิเสธ D2x secondary expert tier หรือ STRATA_PREFILL_EXPERT_SPLIT ปัจจุบัน เมื่อปิด feature ยังรองรับเท่าเดิม
2. **Offline compiler** ใช้ RV32IM/Zba/bignum FFN/GDN compiler ที่ตรึง commit พร้อม Strata vocabulary adapter และการเปลี่ยน native-call eligibility ที่ผ่าน review รักษา arithmetic parser ของวงจร compile/park firmware แบบ offline แล้ว export snapshot ที่พร้อมใช้; production startup ไม่รัน host emulator เก็บ provenance และ notices ที่ต้องมี upstream มี calc-bridge ELF fixture ขนาด 29,932 bytes แต่ต้องยืนยัน source/build/provenance ที่ตรงกันก่อนใช้
3. **Artifact ABI** เสนอ little-endian `STQPUT01`, schema 1.0 พร้อม section directory ที่ตรวจได้และ typed FP32/int32 arrays เก็บ compiler commit กับ local patch digest, numeric ABI, model contract/tokenizer digests, firmware SHA256, op ordering, dimensions, CSR/scatter arrays, feature-slot descriptors, byte/printable maps, initial state และ resource limits ที่ประกาศไว้ ไม่รับ pickle, executable payload หรือ arbitrary Python callback
4. **Feature lowering** export text rows ขนาด 16 bytes พร้อมความยาว bytes เดิมและ special/printable maps แล้ว lower เฉพาะ circuit feature kinds ที่รู้จัก ไม่สร้าง dense vocabulary-by-residual embedding matrix พิสูจน์ exported token-feature values เทียบ compiler ครบ supported trace vocabulary และ text IDs ทั้ง 248,320 ตัว
5. **Loader** ตรวจ magic/version/endian, file/section integrity, sizes ที่ป้องกัน integer overflow, indices, row pointers, finite coefficients, head layout และ memory caps ที่ตั้งไว้ ก่อน GPU allocation/upload การรับ firmware ต้องตรวจ ELF32 little-endian RISC-V, entry, segment bounds/overlaps และ filesz/memsz ด้วย จำกัด auxiliary host memory ระหว่าง validation ปฏิเสธ ABI/binding ที่ไม่รองรับก่อนเปลี่ยน state
6. **Exact operations** รักษา coefficient bits, sparse products, ReLU, decay, beta/write/read rules, argmax tie rules, lag history, normalization และ rounded sqrt/division เริ่มด้วย deterministic per-output reductions และ dot-product trees ที่กำหนดไว้ FP32 atomics ของ upstream ไม่ได้พิสูจน์ determinism ห้าม quantization, TF32, fast-math, lossy state หรือเปลี่ยนลำดับ reduction โดยไม่มี exactness proof และ oracle gate
7. **Device control** ใช้ standalone top-level conditional WHILE graph บน primary device โดยตั้ง condition ทุก launch และจำกัด chunk steps/output budget ภายใน body ไม่มี host callback, allocation/free, CDP หรือ device-graph launch ไม่ clone conditional graph หรือแทรกเป็น child ของ verifier graph v1 ให้ active executor ทำงานทีละตัว; parked sessions มี saved states แยก และ rebind/rebuild graph addresses เมื่อ storage เปลี่ยน
8. **Feasibility gate** ก่อนพึ่ง conditional graphs ต้องทดสอบ create/instantiate/launch/relaunch บน Windows จริงสำหรับ zero/one/N trips, stop conditions และ repeated sessions ด้วย driver ที่ติดตั้ง หากไม่รองรับให้หยุดและถามผู้ใช้เรื่อง fixed-count regular graph ที่มี GPU active/done masks และให้ host กำกับต่อ bounded chunk ทางสำรองนั้นยังเป็น native arithmetic แต่ไม่แทน runtime ที่อนุมัติแบบเงียบ ๆ
9. **Transactional decode** กำหนด ENTER หลัง consumed input row r ซึ่งวงจรเป็นเจ้าของ token ถัดไป commit ได้ไม่เกิน r+1 input rows ไม่ใช่นับ emitted tokens ใช้ speculative shadow circuit ประเมิน candidate window; normal acceptance และ takeover แรกที่ใช้ได้จำกัด prefix จากนั้น commit circuit โดย replay เฉพาะ accepted prefix จาก base state ที่เก็บไว้ แล้วทิ้ง rejected shadow suffix ต้องวัดต้นทุนของวิธีที่เน้น correctness นี้ก่อนเปลี่ยนเป็น journal optimization fence commit/cross-stream work ที่ค้างก่อน handoff, snapshot หรือ host access
10. **Language state** machine-only trace steps ไม่เปลี่ยน language KV/GDN/PLE, logical positions, RNG counters หรือ penalty history machine-owned text ขยับ language state ครั้งเดียวตามลำดับเดิม เริ่มป้อนข้อความนั้นผ่าน exact verifier windows; ใช้ batched prefill ปกติได้ต่อเมื่อพิสูจน์ equivalence แล้ว เก็บ machine output ไว้โดยยังไม่เผยแพร่จนจบสำเร็จ จากนั้นจึง resume language sampling/MTP ในตำแหน่งที่ถูกต้อง
11. **Bridge contract** v1 ใช้ native XML call syntax ของโมเดลเดิม พร้อม internal calc capability ที่ประกาศชัดและ eligibility/role gates tool names อื่นกับตัวอย่างใน history/ข้อความที่ยกมาต้องไม่เริ่ม machine burst รักษา expression semantics ของ upstream รวม Python floor division/modulo, rational division, powers ที่เป็นจำนวนเต็มไม่ติดลบขนาดจำกัด, gcd และ decimal +/−/* ที่รองรับ source ที่ตรึงมี 80 capture tokens, 16 bytes ต่อ row, expression payload 255 bytes และ MAXD=62; ต้องปฏิเสธ input ที่ overflow/ถูกตัด, operands/operators ที่ไม่รองรับ หรือ embedded NUL ก่อนจะนับเป็น arithmetic สำเร็จ การขยายขอบเขตต้องเปลี่ยน compiler ABI และ oracle corpus
12. **API และ canonical history** internal call/result syntax ต้องไม่หลุดเข้า external tool dispatch เก็บ mapping ที่ยืนยันได้จาก visible assistant messages ไปยัง canonical internal token transcript ต่อ cache slot ผูกกับ message-prefix hashes, artifact identity และ session identity continuation ที่ตรงจึง reuse canonical tokens; history ที่กำกวมหรือเชื่อถือไม่ได้ต้อง cold-read อย่างปลอดภัยและไม่แทรก internals จากการเดา prefill history อัปเดต bridge history โดยไม่คำนวณงานเก่าซ้ำ mapping นี้ต้องอยู่ใน session persistence modes ที่ v1 รองรับ
13. **Sessions และ failures** v1 save ได้เฉพาะ coherent quiescent boundary; การ save ระหว่าง active burst ต้องตอบ busy/refused ชัดเจนหรือรอผ่าน bounded queue เดิม saved identity รวม artifact/numeric ABI, compiler patch, tokenizer, firmware และ state geometry เก็บ RAM/PC/registers/counters, lag history, bridge/ownership และ canonical-history metadata แต่ไม่เก็บ graph handles การ restore device บางส่วนแล้วล้มเหลวทำให้ context ทั้งหมดใช้ต่อไม่ได้ limit/cancel/fault ต้องทิ้ง output ที่ยังไม่เผยแพร่ แล้วคืน coherent transaction boundary ที่เก็บไว้หรือ invalidate session ห้ามรันต่อจาก partial state แบบเงียบ ๆ
14. **Scheduling และ resources** คำนวณ resident weights, committed/shadow/init states, histories, FIFO, scratch, graphs และ transient restore peaks จริงก่อน reserve GPU memory และเติม expert cache active execution อยู่บน primary GPU; NVMe เก็บ artifacts กับ cold snapshots เท่านั้น รักษา display card headroom รวม 2.5 GB และกลไก offload เดิม รายงานเทียบ private-commit budget เดิม 42 GB (AGENTS.md:295-299) กับ baseline ที่วัดจริง โดยระบุชัดหาก baseline เกิน budget อยู่แล้ว กฎ 2.5 GB หมายถึง display headroom รวมการใช้ปัจจุบันของ desktop (AGENTS.md:323-324) ไม่ใช่ข้อกำหนดว่าต้องมี free VRAM 2.5 GB ไม่แก้ served profiles/launchers
15. **Observability และ rollout** ใช้ engine protocol, timeline และ live-run status path เดิม ส่ง computing/return/error transitions ตั้งแต่เริ่มขั้นตอน; refresh ต้องเห็นขั้นตอนปัจจุบัน เพิ่ม timeline spans กับ paired counters ของ bridge, trial/commit, entry, trace, result replay และ host waits เปิด native เฉพาะ experimental test config จนกว่าผู้ใช้อนุมัติหลักฐาน; เมื่อปิดต้องคงพฤติกรรมเดิม

## การทดสอบและจุดเชื่อมต่อหลักที่เสนอ

จุดเชื่อมต่อหลักที่ใช้ทดสอบพฤติกรรมคือ consumed text window หนึ่งชุดกับการตัดสินว่าใครเป็นเจ้าของ token ถัดไป: รับ accepted text แล้วตรวจ language/circuit ownership, visible output และ state ทดสอบผ่าน verifier/commit กับ service request boundaries เดิม แทนการเพิ่ม public APIs หลายตัว การ review PRD นี้รวมถึงตรวจว่าจุดเชื่อมต่อนี้ตรงกับพฤติกรรม native ที่ต้องการ โดยไม่สมมติว่าต้องเริ่มสัมภาษณ์ architecture ใหม่

จุดทดสอบระดับล่างอีกสองจุดจำเป็นต่อหลักฐาน ไม่ใช่ตัวแทน feature: artifact validation ต้องพิสูจน์การปฏิเสธก่อนเปลี่ยน GPU state และ circuit replay ต้องพิสูจน์ว่า native tensor operations จริงสร้าง trace/ผลลัพธ์ HTTP tests อย่างเดียวพิสูจน์สองเรื่องนี้ไม่ได้ ใช้รูปแบบ conversation validation/transfer, protocol, tokenizer, timeline และ slots เดิมเป็นหลัก

test ที่ดีตรวจ outputs, state ownership/fingerprints, counters หรือ refusal reasons ที่ชัดเจน negative control ที่จงใจทำให้เสียต้อง fail ด้วยเหตุผลทาง semantics ที่ test ระบุ; script/import ที่หายหรือ tests ที่ยังไม่รันไม่ใช่หลักฐาน red/green oracle calculators/emulators ใช้เฉพาะ test และต้องแยกจาก production executable dispatch path ให้เห็นชัด ตรึง compiler/patch/corpus digests และ seeds แยก development fixtures จาก held-out free-running acceptance cases และใช้ independent integer/Fraction arithmetic oracle ที่ไม่ได้มาจาก upstream firmware/IR ช่วงจับเวลา turn ต้องไม่รวม oracle/reference/checksum verification โดยเทียบผลแบบ offline หลังหยุดเวลา และตรวจยืนยันว่า production benchmark path ไม่เรียก host arithmetic/reference

## ตำแหน่งที่จะเปลี่ยนและขอบเขตการค้น

รายการนี้คือตำแหน่งที่เสนอว่าจะเปลี่ยน ไม่ใช่อ้างว่ามี implementation แล้ว; baseline anchors อิง e5737a27
| Path | หน้าที่/สิ่งที่จะเปลี่ยน | การตรวจ |
|---|---|---|
| `tools/qwenputer/{export,tokenizer,oracle}.py` (ใหม่), `third_party/qwenputer/` (ใหม่) | pinned compiler/provenance, Strata adapter, artifact และ test-only oracle | G1/G2 |
| `include/strata/artifact/qwenputer.hpp`, `src/artifact/qwenputer.cpp` (ใหม่) | bounded ABI parser, binding, resource estimate | G1 malformed fixtures/no-allocation |
| `include/strata/kernels/qwenputer.hpp`, `src/kernels/qwenputer.cu` (ใหม่) | exact deterministic sparse FFN/GDN และ device token features/control | G2 native step/state replay |
| `include/strata/core/qwenputer.hpp`, `src/core/qwenputer.cpp` (ใหม่) | session/transaction ownership, top-level graph, limits | G0/G3/G5 |
| `include/strata/core/verify.hpp:1-23,66-96`, `src/core/verify.cpp:2012,2407-2433` | accepted input boundary และ async commit dependencies | G3 ENTER ทุก row |
| `src/program/generate.cpp:12672-12679,13903-13928,15509-15543` | prompt, serve/generate loops, ทิ้ง/replay draft suffix | G3/G6; ไม่ตกหล่น loop |
| `src/program/generate.cpp:5919-5935`, cache budget/init sites | static reserve ก่อน expert placement; ไม่พึ่ง elastic | G8 slots/peak/hit split |
| `include/strata/core/{conversation_cache,conversation_file,conversation_snapshot}.hpp`, sources คู่กันใน `src/core/` | native identity/state, coherent save/restore limits | G4; invalid/partial restore |
| `include/strata/core/coupled_draft.hpp:19-35`, MTP/penalty callers | คง counters/history; เปลี่ยนเฉพาะเมื่อ integration ต้องใช้ | G3 seeded continuation |
| `serve/native_transcript.py` (ใหม่), `serve/server.py:1723-1854,3913-3972,4328` | native event protocol, canonical prefix mapping, internal ownership | G4/G5/G6 |
| `serve/frontend.py:622-637,832`, `serve/runs.py` | แยก internal/native จาก external calls; reconnect metadata | G5 streaming fragmentation |
| `serve/ui/src/lib/status.ts:37`, i18n/event consumers และ tests | สถานะ computing/return/error ที่ต่อเนื่องและเริ่มทันที | G5 real rendered refresh |
| `include/strata/timeline*.hpp`, `serve/timeline.py`, `tests/xeno/perf/timeline.py`, `tests/xeno/test_timeline.py` | เพิ่ม analyzer ก่อน; spans/counters ของขั้นใหม่ | G8 timeline fixtures/off-cost |
| `tests/qwenputer/` (ใหม่), `serve/test_native_qwenputer.py` (ใหม่), `tests/xeno/perf/qwenputer.py` (ใหม่) | component, ownership, service และ measurement gates | G0-G8 |
| `CMakeLists.txt`, `docs/BLUEPRINT.md`, native guide และ reports ใหม่ | opt-in build/flag/protocol docs, backend guards, evidence | G6/G7/G9 |

ค้นด้วย git grep/git ls-files ใน engine, headers, serving, tokenizer, tests, CMake และ blueprint (ไม่มี rg) รวม pinned upstream compiler/runtime/bridge/firmware source และ primary CUDA/Windows docs ยังไม่ได้สำรวจ dynamic imports, remote issue history ทั้งหมด, drives/caches อื่นและ backend toolchains ที่เข้าถึงไม่ได้อย่างครบถ้วน ไม่ได้นำ untracked reports/builds/profiles เดิมที่ไม่เกี่ยวข้องมาเป็น baseline code

## เกณฑ์ยอมรับ

| ID | หลักฐานที่ต้องมี; ไม่ตั้งเป้าความเร็วจากการเดา |
|---|---|
| G0 Feasibility | conditional graph probe จริงบน primary: 0, 1, N, relaunch/reset, full-buffer, stop และ bounded cancel; บันทึก runtime/driver/device/OS หาก runtime ไม่รองรับให้หยุดเส้นทางที่เลือก |
| G1 Artifact | schema/export hashes ที่ทำซ้ำได้; feature-map values ทั้งหมดตรง compiler; license/provenance manifest; malformed/oversize/incompatible packs ถูกปฏิเสธก่อน device allocation |
| G2 Native exactness | semantic trace/result mismatches เป็นศูนย์เมื่อเทียบ pinned IR/reference corpus และ independent exact arithmetic oracle ครบ instruction families/error paths, repeated runs และ frozen held-out free-running cases/seeds ที่แยกจาก development fixtures หลักฐาน/tests ของ deterministic reduction ครอบคลุม outputs ที่ชนกัน ต้องรัน Compute Sanitizer memcheck/initcheck/synccheck กับ applicable racecheck และรายงาน paths ที่ไม่รองรับ racecheck ไม่ได้พิสูจน์ global-atomic determinism |
| G3 Ownership | ENTER ทุก input-window position, rejected drafts, first/last rows, suffix และ MTP transitions; machine trace ไม่ขยับ language state/RNG/penalties/PLE; completed result replay เพียงครั้งเดียว |
| G4 History/session | สลับสองบทสนทนา, cached continuation, cold history, history ที่ไม่มี hidden fragments, save/restore identity mismatch และ partial transfer failure; ไม่มี circuit state ค้างหรือ hidden-prefix เสีย |
| G5 Safety/status | bounds, cancel, malformed UTF-8/NUL/overlong/nested/reserved tokens, unsupported syntax, huge powers, fault และ GPU failure ให้ terminal reasons ชัดเจน; ไม่มี partial numeric success; ตรวจ immediate status และ refresh จริง |
| G6 Disabled parity | fresh unchanged baseline เทียบ candidate disabled ใน session เดียวกันด้วย greedy raw-token parity ครบ code/Thai/ordinary/long-prefix turns; disabled prompt/budget/protocol/snapshot serialization และการไม่ allocate circuit ต้องตรงเดิม |
| G7 Backend/build | full D2x builds มี sm_89 กับ sm_120 ตรวจ binary coverage และทำ actual secondary-tier smoke ก่อน ABBA ต้อง build ทุก backend ที่ shared files แตะและรักษา Q2 parity (AGENTS.md:44) เริ่ม CUDA real tests ก่อน; HIP/SYCL execution ที่ยังทำไม่ได้เป็น implementation-validation debt ชัดเจน ไม่ใช่ check ที่ผ่านหรือ blocker ต่อ plan review |
| G8 Performance | ใช้ timeline ก่อน; profiler-off ABBA สำหรับ disabled parity/ordinary enabled overhead และ native arithmetic turns บันทึก same-run stage counters, hashes, priorities, hit split, draft acceptance, startup/peak RAM/VRAM, entry/exit/host waits, disturbed runs และ DONE ช่วงจับเวลาไม่รวม oracle/reference/checksum verification ให้เทียบผล offline และยืนยันว่า production path ไม่เรียก host arithmetic/reference รายงาน distributions; กฎเดิม 13.6% (AGENTS.md:320-321) เป็น repo heuristic ไม่ใช่ statistical significance และผลต่ำกว่าเกณฑ์ยังไม่พิสูจน์หากไม่มี paired repetition ที่สม่ำเสมอ |
| G9 Release | PRD นี้ไม่เปลี่ยน served config/default/deployment ผู้ใช้ตรวจ correctness, resource costs และ latency ที่วัดจริงก่อน opt-in rollout Universal ยังเลื่อนไประยะถัดไป |

arithmetic corpus รวม signed add/sub/mul, exact rational division, negative floor/modulo, gcd, precedence/powers, decimals ที่ firmware รองรับ, ขอบเขต 62 digits, divide-by-zero, capacity/complexity errors และ multi-call continuation แยก forced native-call proof จากการประเมินว่าโมเดลเลือกใช้วงจรเองได้ดีเพียงใด; ทั้งสองไม่ใช่หลักฐานว่าคุณภาพ coding/3D โดยรวมดีขึ้น

## ลำดับงานและ dependencies

| Phase | ผลงาน | ต้องมีมาก่อน |
|---|---|---|
| P0 | preflight, real conditional-graph smoke probe, frozen baseline manifest และ model-input inventory | review แผน; ใช้เฉพาะ probe ที่สร้างสำหรับงานนี้ |
| P1 | compiler adapter ที่ตรึง commit/มี license, fixture/oracle bundle และ artifact export ที่ทำซ้ำได้ | P0; อนุมัติ third-party execution แบบเฉพาะ |
| P2 | CPU artifact loader/validator และ exact token-feature lowering | schema และ fixtures จาก P1 |
| P3 | deterministic single-step CUDA FFN/GDN executor | P2; compiler oracles |
| P4 | bounded standalone GPU trace/token-selection graph | runtime capability จาก P0, P3 |
| P5 | transactional verifier/commit integration, MTP/suffix ownership และ prompt-history mode | P4 |
| P6 | snapshot identity, save/restore และ canonical transcript storage | P5 |
| P7 | API-native ownership, live status, refresh และ ordinary-tool compatibility | P5/P6 |
| P8 | full backend/parity/quality tests, D2x timelines และ profiler-off ABBA; review report | P0-P7 และ model inputs ที่พร้อม |

ทุก phase ทำ test-first จบด้วยหลักฐานของตัวเองและอัปเดต blueprint โดยยังเป็นงาน local จนกว่าจะอนุมัติ publishing แยก companion implementation plan ระบุ proposed files, จุดเชื่อมต่อที่แน่นอนและ tests ต่อ phase; decisions ใน PRD นี้อธิบาย contracts แทนตำแหน่งไฟล์ที่อาจเปลี่ยน

## นอกขอบเขต

- Universal model adapters, การ train โมเดล, โหลด architecture ใดก็ได้, merged Qwen3.5 runtime และข้ออ้าง benchmark 3D/coding โดยรวม
- production fallback ไป host calculator/emulator, arbitrary C/Python execution, filesystem/network tools หรือสิทธิ์เพิ่มเติม
- dynamic D2x expert-cache resizing, วงจรบน display GPU, paging active working state ระหว่าง burst หรือเปลี่ยน default ด้วยค่าประมาณ
- v1 concurrent batch/layer-split/pipelined machine execution และ mid-burst session serialization; ต้องประกาศ combinations ที่รองรับชัดเจน
- เปลี่ยน serving configs/launchers, TDR/driver/security settings, ติดตั้ง dependencies/models ขนาดใหญ่, publish tracker entries, push/PR หรือ deploy ระหว่าง planning/review ระยะนี้

## ความเสี่ยง การอนุมัติ และหมายเหตุ

- **จุดหยุดทางเทคนิคแรก:** conditional graph ใช้งานได้จริงกับ Windows driver นี้หรือไม่ หาก probe ไม่ผ่านต้องให้ผู้ใช้ตัดสินใจเรื่อง native chunk fallback ที่บันทึกไว้ ไม่เปลี่ยนไปรัน CPU แบบเงียบ ๆ
- **blocker แรกของ end-to-end:** Swift native/PLE GGUF ที่ config อ้างไม่อยู่ในตำแหน่งนั้น ให้หา approved copy ที่มีอยู่ก่อน; หากไม่มีต้องเสนอ files, sizes, source และผลต่อ disk/memory ที่แน่นอนเพื่อขออนุมัติ ไม่เปลี่ยน quantization/profile หรือดาวน์โหลด merged Qwen3.5 model แทน
- **Third-party execution:** ก่อนรัน pinned compiler/reference/firmware หรือ downloaded binary ต้องเสนอ commit/paths ที่แน่นอน, imports/commands ที่ตรวจแล้ว, จุดประสงค์, สิ่งที่จะเขียนและ resource requirements เพื่อขออนุมัติเมื่อจะรันจริง การอ่าน source และเขียนแผนนี้ยังไม่ได้ execute สิ่งเหล่านั้น
- **Dependencies:** ใช้ minimal NumPy-based export ที่มีอยู่หากทำได้ การ install, ใช้ RISC-V toolchain/WSL, rebuild firmware หรือดาวน์โหลดโมเดลต้อง review แยก; การมี wsl.exe ไม่พิสูจน์ว่า Linux toolchain ใช้งานได้
- **Resource decisions:** ยังไม่รู้ขนาด artifact/shadow/graphs จริงและ ordinary-text overhead หาก reserve ไม่ลงตัวกับ memory/headroom ที่อนุมัติ ต้องเสนอทางเลือกพร้อมผลวัด ไม่ลด context, เปลี่ยน experts/defaults หรือย้ายวงจรแบบเงียบ ๆ
- **Licensing:** ตรวจ source MIT notice แล้ว ต้องตรวจ firmware/libc/libgcc และ binary provenance แยก; tensors ที่มาจาก checkpoint ต้องคง weight license ที่เกี่ยวข้อง ไม่สมมติว่า generated/binary inputs ทุกอย่างใช้ license เดียวกัน
- **Planning workflow:** เผยแพร่ PRD R2 ที่อนุมัติเป็น [#276](https://github.com/xenodeve/Strata-xeno/issues/276) พร้อมข้อกำหนดอังกฤษ/ไทยครบและตรวจ body แล้ว การแตก implementation issues แยกต่างหาก; ไม่อนุมัติ PR/push/merge
- **Fast mode:** ขอใช้ Fast สำหรับ models ที่ไม่ใช่ Astra ในจุดที่รองรับ local defaults ที่อ่านได้คือ Sol/medium/priority แต่ไม่เปิดเผย model/tier ของ active turn และไม่มี tier setter จึงห้ามอ้างว่าเปิดสำเร็จ
- **Rollback:** ปิด feature แล้วกลับเส้นทาง runtime เดิม; context ที่เสียต้องทิ้ง ไม่ resume การนำ experimental config/artifact ออกไม่เปลี่ยน served profiles component benchmark ไม่ใช่คำมั่นว่าจะ rollout หรือเปลี่ยน default

## ดัชนีหลักฐาน

- R1: [Strata blueprint](../BLUEPRINT.md) baseline/revision log และ elastic-cache refusal; [ADR 0001](../adr/0001-q2-exp-baseline.md) สัญญา Q2 correctness
- R4: policy เดิมใน [AGENTS.md:44](../../AGENTS.md#L44) (backend builds), [295-299](../../AGENTS.md#L295) (private commit 42 GB), [320-321](../../AGENTS.md#L320) (noise heuristic 13.6%), [323-324](../../AGENTS.md#L323) (display headroom รวม 2.5 GB); [CMakeLists.txt:9,156-159](../../CMakeLists.txt#L9) (dual architecture configuration) ทั้งหมดเป็นข้อกำหนดเดิม ไม่ใช่ thresholds ใหม่ที่แผนเสนอ
- R2: [ผลสำรวจและ local test/resource evidence จริง](../reports/2026-10-28-native-qwenputer-discovery.md); logs ที่ `%TEMP%/strata-qwenputer-discovery-20261028/`
- R3: [Pinned verifier contract](https://github.com/xenodeve/Strata-xeno/blob/e5737a27f485773a7ce22784fef6af7b480dcc9a/include/strata/core/verify.hpp#L1), [commit](https://github.com/xenodeve/Strata-xeno/blob/e5737a27f485773a7ce22784fef6af7b480dcc9a/src/core/verify.cpp#L2407), [snapshot](https://github.com/xenodeve/Strata-xeno/blob/e5737a27f485773a7ce22784fef6af7b480dcc9a/include/strata/core/conversation_snapshot.hpp#L50), [sampling counters](https://github.com/xenodeve/Strata-xeno/blob/e5737a27f485773a7ce22784fef6af7b480dcc9a/include/strata/core/coupled_draft.hpp#L19)
- U1: [Pinned RV chat](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/chipformer/rv/chat.py#L100) full merged loader; lines 172-184 เป็น host token selection; 324-426 เป็น ownership
- U2: [Pinned sparse executor](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/chipformer/rv/tritonexec.py#L47) atomic scatter, rounded normalization และ state layout
- U3: [Pinned bridge](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/chipformer/rv/bridge.py#L45), [core features](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/chipformer/rv/core.py#L109), [ELF loader](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/chipformer/rv/elf.py)
- U4: [Pinned calc firmware](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/chipformer/rv/env/calc/calc.c#L1), [firmware build](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/chipformer/rv/env/calc/build_calc.sh), [MIT license](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/LICENSE)
- C1: [CUDA 13.3 conditional graph requirements](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-node-body-graph-requirements) และ `driver_types.h` ที่ติดตั้ง; การมี header ไม่ใช่ runtime pass
- C2: [ข้อจำกัด CUDA conditional-node API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/structcudaConditionalNodeParams.html)
- W1: [Microsoft WDDM TDR](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/timeout-detection-and-recovery): default สองวินาทีเป็น scheduling/preemption timeout ไม่ใช่การรับรองเวลาต่อ graph แบบครอบจักรวาล ห้ามแก้ TDR settings
<!-- lang:end -->

