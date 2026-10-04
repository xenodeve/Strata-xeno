# Codebase review vs upstream v0.1.38

Date: 2026-10-04. Reviewed fork: `ad28ddb7c347da415221818bcdc3b9eba0daf267`. Upstream: `99f3dbd0b21d1401b3769e0c0d963913607f380b` (v0.1.38).

## 1. Scope and conclusion

This is a focused, read-only mechanism audit of the handoff candidates in prefill, expert loading, configuration and administrative HTTP routes. It is not an exhaustive audit of every changed file. Source and call sites were read through GitHub at the pinned commits; no engine build, GPU run, daily server, configuration edit or ABBA was performed. Historical measurements below belong to their linked issues, not to this review.

Read as context: AGENTS.md, BLUEPRINT.md, the 0.1.30/0.1.34/0.1.37/0.1.38 merge reports, OPEN-WORK-LEDGER.md, and #136 with its result comments. The blueprint is a mechanism map, not a substitute for the pinned source when line numbers have moved.

**Conclusion:** there are useful convergence opportunities, but the handoff's similar-looking mechanisms are not all redundant. Converge the grouped-copy primitive, redundant configuration decisions and common loader contracts. Keep the fork's ring scheduling, split/wave behavior and placement-first ownership. Do not equate a smaller diff with lower latency.

Six traced findings below map to five tracker items. #153, #154, #155 and #156 already existed when deduplicated; this review created #161 for loss of GGUF failure diagnostics. All are proposals, not approval to modify the engine or security boundary.

## 2. Priority: measured latency before cleanup

The source-only findings in section 3 establish maintenance, behavior or diagnostic costs, not a measured D2x speedup. Existing latency work should therefore retain priority over cosmetic refactoring.

- **Existing #152, coordinated with #149:** the reported KV-streaming drafter fallback is a concrete upstream-convergence candidate. Its source timelines report `on_chunk (drafter)` at 15.6 versus 207.9 ms for a 9,927-token prefix and 1.6 versus 16.7 ms for a 399-token part. These compare streaming modes in the recorded session; they are not measurements of a newly enabled batch-ring implementation. Upstream's batched ring support is merged but gated in the fork. The owner must measure main-model hashes, draft acceptance, replies and whole-turn time before changing that default. This review does not change or independently reproduce the drafter path. [#152](https://github.com/xenodeve/Strata-xeno/issues/152), [#149](https://github.com/xenodeve/Strata-xeno/issues/149).
- **Existing #40 and the remaining #136 CPU work:** router synchronization, the remaining per-layer pool barrier and host dispatch planning need current same-session attribution. The original #136 timing table predates several improvements; it must not be used as the current post-merge budget or added across overlapping lanes. [#136 and comments](https://github.com/xenodeve/Strata-xeno/issues/136), [#40](https://github.com/xenodeve/Strata-xeno/issues/40).
- **Do not re-propose shipped work:** #136's comments record #133's removal of the split's CUDA0-resident relay, #139's tail-reader thread choice, #142's opportunistic omission of unrouted copies, #145's background checkpoint write, and #147's fused CPU-pool phase. The results do not authorize deleting their A/B levers or replacing opportunistic lookahead with a wait for routing.

Within unmeasured cleanup work, R1 and R2 address recurring merge surfaces; R4 is a smaller policy-consolidation change. R5 is independently actionable behavior at a security boundary and needs an explicit policy decision. R6 is a narrow diagnostic fix. No numerical latency ranking among those items is justified by this review.

## 3. Traced findings

### R1. Two grouped-gather primitives, not two copies at runtime

**Kind:** redundant implementation / improvement. **Decision:** converge on upstream's primitive; keep the fork's scheduler. **Tracker:** [#154](https://github.com/xenodeve/Strata-xeno/issues/154).

**Fork:** [moe_mmq.hpp:75-100](https://github.com/xenodeve/Strata-xeno/blob/ad28ddb/include/strata/prefill/moe_mmq.hpp#L75-L100), [moe_mmq.cu:280-345](https://github.com/xenodeve/Strata-xeno/blob/ad28ddb/src/prefill/moe_mmq.cu#L280-L345), and the live streamed call at [prefill.cpp:3290-3370](https://github.com/xenodeve/Strata-xeno/blob/ad28ddb/src/prefill/prefill.cpp#L3290-L3370). **Upstream:** [moe_mmq.cu:229-245](https://github.com/Niko1221/Strata/blob/99f3dbd/src/prefill/moe_mmq.cu#L229-L245).

The fork's pointer-array overload and upstream's `GatherGroup` overload implement the same aligned group-copy job. Their argument lists put `down_off` and `gu_half_bytes` in different positions although both are `size_t`. The fork overload is live; retaining the upstream overload does not mean the engine executes both. Existing #154 also inventories the split call at `prefill.cpp:1232` and non-CUDA definitions.

**Concrete cost:** two kernel/wrapper implementations and two public signatures to reconcile when layouts change, plus a same-type argument-order hazard. There is no measured extra runtime copy or extra launch from their coexistence.

**Proposed change:** construct a compact `GatherGroup` with `first=0`, preserve the already-shifted output destinations, and use the upstream aligned primitive. Preserve the fork's per-expert fallback when the upstream function returns `false`; preserve no-op and unsupported-size behavior. Do not replace the surrounding ring walk: the current code waits for the last required copy, queues the gather, records slot-use events, then publishes consumption. Returning a slot before its copy consumer is finished would invalidate the ownership contract.

**Safety proof:** byte-level group tests covering partial groups, nonzero destination offsets, unequal role offsets, mixed resident/ring sources, unaligned inputs and fallback; ring exhaustion and mid-group flush; both GPUs; then the common STATE_HASH and greedy ABBA gates. Keep the grouped-gather A/B lever. Event coalescing is separate work requiring measurement, not an incidental part of this refactor.

### R2. GGUF role geometry is duplicated; the two reader policies are not interchangeable

**Kind:** redundant logic / improvement. **Decision:** share geometry, keep ownership-aware transport. **Tracker:** [#155](https://github.com/xenodeve/Strata-xeno/issues/155).

**Fork:** generic loader [expert_source.cpp:2534-2690](https://github.com/xenodeve/Strata-xeno/blob/ad28ddb/src/core/expert_source.cpp#L2534-L2690), placement-first reader and caller [3240-3340](https://github.com/xenodeve/Strata-xeno/blob/ad28ddb/src/core/expert_source.cpp#L3240-L3340). **Upstream:** generic reader [2388-2535](https://github.com/Niko1221/Strata/blob/99f3dbd/src/core/expert_source.cpp#L2388-L2535).

The inspected buffered, Windows-unbuffered and placement-first worker bodies repeat gate/up/down lengths and destination offsets. Upstream still has separate buffered and unbuffered geometry, so adopting upstream wholesale does not itself solve that repetition. #155 records the broader geometry inventory and the existing `expert_ranges` reuse.

However, `load_experts_gguf_direct` reads in 32-expert groups, skips a fully excluded group before I/O, and can scatter into `dst_of` slab addresses. `load_rest` combines GPU-exclusive and NVMe exclusions and commits only the host-owned destinations. The upstream generic reader uses 16-expert chunks and contiguous arena destinations. A blind substitution would lose behavior required by placement-first and capacity mode.

**Concrete cost:** at least the three inspected worker bodies separately derive role geometry. Shard/layout changes must be carried through each. There is no evidence that both loaders run for one D2x load, nor a measured turn-latency saving from merging them.

**Proposed change:** first reuse a narrow role-range description, checking the existing helper before introducing another one. Unify transport only if skip-before-I/O, slab destinations, batching, aligned EOF handling and error contracts remain equivalent. Do not restore eager loading of the whole arena.

**Safety proof:** synthetic per-role shard boundaries, unequal role sizes, truncated/missing files, arena and slab destinations, GPU/NVMe exclusions; byte comparison of loaded experts; cold-start stage measurements and private-commit peak. Finish with the common served-path gates if implementation is approved.

### R3. The generic GGUF loader carries an unused ownership extension

**Kind:** unnecessary branch/API surface. **Decision:** adopt the upstream generic signature after a complete caller check; keep the live placement-first skip. **Tracker:** part of [#155](https://github.com/xenodeve/Strata-xeno/issues/155), not a duplicate issue.

**Fork:** the signature and two skip-branch sites in [expert_source.cpp:2534-2690](https://github.com/xenodeve/Strata-xeno/blob/ad28ddb/src/core/expert_source.cpp#L2534-L2690), its explicit-null production call around [2781](https://github.com/xenodeve/Strata-xeno/blob/ad28ddb/src/core/expert_source.cpp#L2770-L2790), and its public declaration in `include/strata/core/expert_source.hpp`. **Upstream:** [expert_source.cpp:2388-2389](https://github.com/Niko1221/Strata/blob/99f3dbd/src/core/expert_source.cpp#L2388-L2389), which has `threads, bool unbuffered` without `skip`.

The inspected production call passes `skip=nullptr`; `tests/core/expert_layout_test.cpp` calls the default-null form. Ownership-aware loading is handled by `load_rest` through the separate direct reader, where its skip mask is essential. This distinguishes a removable generic-loader extension from an actively used memory-ownership feature.

**Concrete cost:** an extra public parameter, an explicit-null argument and two per-expert skip-check sites that have already collided with upstream's signature evolution. Binary-size and latency effects were not measured.

**Proposed change and proof:** verify every caller at the implementation commit, including tests and downstream/internal API users, then restore the generic API to upstream and remove only its unreachable non-null behavior. Preserve `unbuffered` argument meaning; a positional argument must not accidentally change meaning during migration. Run synthetic byte-parity tests for both generic read modes and the placement-first/capacity tests. Do not remove `load_rest`'s skip mask.

### R4. Resolve expert-split once; preserve both peer refusal predicates

**Kind:** redundant policy parsing / improvement. **Decision:** keep fork behavior with one resolved configuration value. **Tracker:** [#156](https://github.com/xenodeve/Strata-xeno/issues/156).

**Fork:** [generate.cpp:2480-2500](https://github.com/xenodeve/Strata-xeno/blob/ad28ddb/src/program/generate.cpp#L2480-L2500), [2660-2690](https://github.com/xenodeve/Strata-xeno/blob/ad28ddb/src/program/generate.cpp#L2660-L2690), and [prefill.cpp:2235-2260](https://github.com/xenodeve/Strata-xeno/blob/ad28ddb/src/prefill/prefill.cpp#L2235-L2260). **Upstream equivalent:** none for this fork expert-split setting; #156 identifies upstream's separate peer refusal at `generate.cpp:1773`.

`STRATA_PREFILL_EXPERT_SPLIT` is parsed for option refusal, parsed again for setup/sizing, and separately cached in prefill execution. The adjacent peer refusals do different jobs: one excludes layer-split/remote combinations; the other excludes the fork's secondary/exclusive/expert-split combinations. They are not duplicate predicates that can be deleted interchangeably.

**Concrete cost:** three copies of the activation rule and distributed configuration decisions. This is not three environment parses per token and is not an established latency bottleneck.

**Proposed change:** pass a narrow immutable resolved option through existing setup rather than building a general environment framework. Preserve absent/empty/zero/nonzero `atoi` behavior unless a separate compatibility change is approved. Preserve both refusal conditions, diagnostic precedence, secondary eligibility, partial MMQ coverage and wave's dependency on split. No A/B lever or serving-profile edit.

**Safety proof:** red-first option/refusal matrix; split/wave on and off; unavailable or non-exclusive secondary tier; partial MMQ layers and relevant size thresholds. Verify equivalent decisions, then the common state/reply gates.

### R5. Administrative routes disagree on configured trusted Origins

**Kind:** redundant security policy with observable behavior divergence. **Decision:** converge only after the intended administrative policy is explicit. **Tracker:** [#153](https://github.com/xenodeve/Strata-xeno/issues/153).

**Fork:** `_foreign_origin` at [server.py:2884-2890](https://github.com/xenodeve/Strata-xeno/blob/ad28ddb/serve/server.py#L2884-L2890), its `/load` and `/unload` gate at [3430-3434](https://github.com/xenodeve/Strata-xeno/blob/ad28ddb/serve/server.py#L3430-L3434), and `_own_page` at [3551-3566](https://github.com/xenodeve/Strata-xeno/blob/ad28ddb/serve/server.py#L3551-L3566). **Upstream:** [server.py:2220-2250](https://github.com/Niko1221/Strata/blob/99f3dbd/serve/server.py#L2220-L2250) and [2335-2365](https://github.com/Niko1221/Strata/blob/99f3dbd/serve/server.py#L2335-L2365).

The first fork gate ignores `trusted_origins`; the subsequent upstream-derived gate explicitly accepts them. The versioned lifecycle routes do not pass through the same extra unversioned gate. The source therefore establishes inconsistent treatment of a configured trusted reverse-proxy page.

Existing #153 reports a mock-only HTTP reproduction: trusted JSON gets 403 on the fork's unversioned routes and 200 upstream, while the fork's versioned routes accept it. This review independently traced the source but did not run that reproducer. This is not evidence of an untrusted-origin or authentication bypass.

**Concrete cost:** two independent administrative Origin decisions and rejection of a configured trusted caller on one route family. No meaningful D2x latency benefit is established.

**Proposed change and proof:** decide whether administrative operations accept configured trusted origins, then use one policy across both families. Keep authentication, Host checks and refusal-before-side-effect ordering. Test own/trusted/foreign/null/absent Origin, JSON/plain content, key/no key and all four endpoints. Explicitly preserve or document 403-versus-415 precedence. Do not treat CORS permission as administrative authorization or relax the boundary as incidental cleanup.

### R6. The placement-first GGUF reader discards the actual failure cause

**Kind:** improvement / diagnostic defect. **Decision:** adopt upstream's error-reporting contract, not its transport. **Tracker:** [#161](https://github.com/xenodeve/Strata-xeno/issues/161), created by this review.

**Fork:** [expert_source.cpp:3240-3340](https://github.com/xenodeve/Strata-xeno/blob/ad28ddb/src/core/expert_source.cpp#L3240-L3340). **Upstream analogue:** [2388-2483](https://github.com/Niko1221/Strata/blob/99f3dbd/src/core/expert_source.cpp#L2388-L2483).

The direct reader reduces open/allocation/submit/wait/completion/short-read failures to `bad=true`, discards the worker's detailed `e`, and returns negative `seconds`. `load_rest` then emits only `load_rest: the GGUF read failed`. Upstream's unbuffered generic reader preserves the first worker error under a mutex and returns `LoadStats.ok=false`, `error`, and negative `seconds`.

**Failure is detected; the cause is lost.** This is source-traced, not a reproduced disk failure or a claim that a failed load is silently accepted.

**Concrete cost:** several distinct boot failures become one indistinguishable message, losing shard and operation context. No turn-latency saving is claimed.

**Proposed change:** carry the first specific error through the existing result and caller, including shard/phase and layer/role/range/status where available. Preserve the negative-seconds contract for current callers. Keep batching, ownership, alignment, destinations and successful bytes unchanged; separate this patch from R2's transport convergence.

**Safety proof:** fault-injection tests distinguish failure classes, valid aligned reads beyond EOF still pass when payload is complete, and a failed load does not clear deferred state. Successful byte parity and the common served-path gates remain required after approval.

## 4. Candidates that should not become blind deletions

**PLE read-ahead — traced, keep the fork-specific scheduling.** [Fork prefill.cpp:1990-2275](https://github.com/xenodeve/Strata-xeno/blob/ad28ddb/src/prefill/prefill.cpp#L1990-L2275) has one pending future/take lifecycle, not two independent reads of every chunk. Its idempotent take adopts the upstream landing rule; compare [upstream prefill.cpp:1370-1515](https://github.com/Niko1221/Strata/blob/99f3dbd/src/prefill/prefill.cpp#L1370-L1515). The fork additionally handles wave-strided lookahead and the A/B lever. Renaming or replacing the body wholesale would not remove demonstrated duplicate I/O and could lose the wave stride. No new removal issue is warranted from this evidence.

**A/B levers — retain.** The handoff expressly distinguishes an unused production option from a deliberately retained comparison arm. #136's result comments still name `STRATA_SLOT_SYNC`, `STRATA_POOL_FUSED` and other levers in recently settled experiments. This review does not establish a retirement decision for any lever. A future retirement proposal needs its originating issue, final experiment and supported-config coverage, not just a search for its presence in D2x.

**Peer naming — disposition based on merge documentation, not a completed full-path audit here.** The 0.1.38 merge report distinguishes the fork's secondary tier from upstream `PeerExperts`/PeerPrefill and records the fork's refusal of the incompatible prefill combination. Similar names alone do not prove duplicate ownership or computation. No API rename or implementation deletion is proposed by this report.

**Literal Q2_0 value 42 — candidate deferred.** This review did not complete an occurrence-by-occurrence comparison of the CUDA file and the named constant's dependency placement. The handoff's approximate count is not promoted to a verified finding. In particular, do not couple such cleanup to the Q2_0 drafter investigation in #149 or imply it improves IQ2_XS D2x latency.

## 5. Implementation gates and remaining coverage

For any later approved engine change: issue, isolated branch/worktree, red-first behavioral tests under `tests/xeno/` with CMake registration as needed, relevant blueprint update, review, and PR. Require byte-identical main-model `STRATA_STATE_HASH` at `max_tokens=1` plus greedy same-session ABBA against the agreed deployed executable. Use both priority flags at 2 for performance arms, profiler-off timing and counters from the same runs. Timelines explain mechanisms; they do not replace unprofiled comparisons.

Loader changes additionally require cold-start/private-commit measurements and capacity-mode coverage. Gather changes require ring-lifetime and both-device byte checks. Security changes require the HTTP authorization matrix before any implementation is accepted. Nothing here authorizes editing a serving profile, launcher, the daily server, `serve/ui/`, or another agent's drafter work.

Not completed in this pass: an exhaustive call graph and dead-code census across every fork file, full secondary-runner/pool/security-boundary audits, every environment lever's retirement history, CUDA compilation, runtime parity, GPU sanitizer checks and latency experiments. Existing tracker measurements and mock-test results are cited as prior evidence only.

The deliverable is this documentation-only report and the tracker mapping above. It establishes specific convergence candidates and rejects unsafe simplifications; it does not certify engine correctness or claim a performance improvement.
