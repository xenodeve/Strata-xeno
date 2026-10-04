# Codebase review against upstream v0.1.38 — 2026-10-04

The most useful consolidation is to adopt upstream's grouped-copy primitive while preserving the fork's streamed/split scheduling. The largest repeated ownership code is the paired primary swap transition in the serve and generate loops. Placement-first loading, split/wave, the secondary tier and their A/B alternatives remain load-bearing; replacing them wholesale with upstream would remove requirements that upstream does not implement.

Two server-policy inconsistencies were reproduced without a model: trusted origins are rejected by the extra `/load`/`/unload` gate, and a supposedly read-only Git command can select a repository outside the coding project's boundary. No engine, server, config or launcher was changed. This review does not demonstrate a new latency win.

## Baselines, scope and method

| Item | Pinned value |
|---|---|
| Fork (F) | `ad28ddb7c347da415221818bcdc3b9eba0daf267`, origin/main, including #140/PR #151 |
| Upstream (U) | `99f3dbd0b21d1401b3769e0c0d963913607f380b`, v0.1.38 |
| Comparison | `git diff 99f3dbd...ad28ddb`; upstream is an ancestor, so the two-dot tree comparison agrees |
| Delta | 422 reachable commits; 643 files, +98,440/-1,616 |
| Engine plus reviewed server boundary | 70 files, +11,854/-1,541 (`src include serve/server.py serve/permissions.py`; this inventory includes excluded drafter files) |
| Review branch/worktree | `docs/2026-10-04-codebase-review`, `C:\Strata-exp\src-review` |
| Served profile, read only | `D:\Github\Strata\strata-flash-next-d2x.json`: Swift 1.5 IQ2_XS, split/wave enabled, KV int8, resident 65,536, context 262,144, paired primary swaps 8/every 1, secondary 6,400 MiB |

All F/U line references below refer to these pinned trees, not today's moving main or the blueprint's older baseline. The review followed callers, ownership and error paths rather than treating added lines as disposable. Standards and originating-issue checks ran in separate agents; the primary reviewer re-read the reported source before accepting findings and independently handled trust boundaries and tracker writing.

Inputs: `AGENTS.md`, `docs/BLUEPRINT.md`, the v0.1.30/.34/.37/.38 merge reports, `docs/OPEN-WORK-LEDGER.md`, issue #136 and subsystem issues including their amendments. The blueprint explicitly retains an older served baseline; its line numbers were treated as discovery pointers and checked against F. Its baseline reconciliation is already tracked in #110.

This is a targeted structural review of the largest engine divergences and server trust boundaries, not an exhaustive audit of all 643 files. UI, setup, website assets and MTP/drafter internals were excluded. #149 and the newer #152 remain with their existing owner. No daily server, engine run, GPU test, build or benchmark was started. The generic `simplify` and `security-review` skills were not found in the installed skill locations checked; this report therefore claims a manual simplification/security pass, not execution of those unavailable skills.

**Evidence labels:** **traced** means this session read the real source path; **reproduced** adds an executed probe. **Hypothesis** marks an unmeasured runtime effect or recommendation whose safety still needs testing. Historical measurements are attributed to their original artifacts and never combined into a new A/B result.

## Latency priority before cleanup priority

Existing measured work remains ahead of stylistic consolidation. In the historical `C:\Strata-exp\merge-137-record\tlsum-142.txt`, prompt run2 reads399 batched tokens in1,050 ms: host `expert launches`366.1 ms, `router sync`283.1 ms, `split wait issuer`245.7 ms, GPU `wait host`254.6 ms. These are overlapping/attributed spans from that run, not independently additive savings or timings of ad28ddb.

| Order | Work and upstream comparison | Decision |
|---|---|---|
| 1 | Decode host dispatch/per-layer pool barrier: #136 item9; fork CPU one-phase implementation is additional to upstream's native multi-token pool | Keep the measured fused/reference modes; continue the existing measurement work. A code extraction alone does not remove this cost |
| 2 | Device-side route grouping: #40; upstream's group gather still consumes host-built groups | Keep #40 as the measured bottleneck work; deduplicating gather APIs does not eliminate router synchronization |
| 3 | Group release events: F per-slot records versus U one record per group | Count/time the current event/reuse work first (#136 follow-up); potential savings are a **hypothesis**, not366ms or245ms |
| 4 | Paired adaptive publication: U#463 waits for non-paired pending copies; F paired stages still query completion | Reuse #61/#143 for the determinism investigation; latency/greedy consequences need a paired trace. Do not insert a wait during refactoring |
| External owner | #152, the KV-streaming drafter prompt pass; upstream's batched-ring path is already present | Reference existing work only; no new drafter investigation or change in this review |

After these, rank consolidation by the repeated merge surface: `generate.cpp` ownership transitions; grouped-copy primitives; expert reader geometry; split configuration; secondary completion; small numerical-predicate cleanup. Server trust findings have a separate correctness priority and are not presented as performance wins.

The reported merge history supports this ordering: v0.1.30 collided on both adaptive loops, v0.1.34 ported per-role GGUF changes through multiple fork readers, v0.1.37 collided on the restructured prompt path, and v0.1.38 reported 15 conflicted files / 72 diff3 hunks, including 30 in `prefill.cpp` and 14 in `generate.cpp`. Those conflict counts are the merge reports' records, not a new trial merge.

## Findings

### R1. Per-slot release events remain after grouped gathers

**Kind:** improvement; redundant synchronization submission. **Evidence:** source **traced**; latency benefit **hypothesis**. **Tracker:** [#136](https://github.com/xenodeve/Strata-xeno/issues/136).

F `src/prefill/prefill.cpp:1232-1237` gathers a split subgroup once, then records one `my_used` event per streamed entry. F `:3327-3330` does the same in the CUDA0 streamed walk. U `src/prefill/prefill.cpp:2404-2407` records one release event and maps every slot in the group to it; the copy issuer uses that mapping at U `:2355,2361`. For a subgroup with q streamed slots, F submits q records where U submits1: q-1 extra event submissions, with no event-specific time measured here.

**Decision:** evaluate convergence with upstream's release mapping, independently of R3's primitive consolidation. The fork's split has two consumers: F `:1166` selects a lane's used array and F `:2430-2431` waits for both lanes before slot reuse. A single shared map copied from U is insufficient. Keep the fork's two-consumer lifetime contract.

**Proof before a change:** obtain current per-lane counts/timeline attribution and a measurable ceiling before building. Characterize small rings, mid-group flush, mixed resident/streamed entries, skipped jobs and both wave lanes; every reused slot must wait for the event that actually covers its gather. Then group-byte parity and the main-model/greedy gates below. Do not revive #29's superseded original checklist as a missing-spec accusation.

### R2. Paired primary ownership transitions are implemented twice

**Kind:** redundant code. **Evidence:** **traced**; extraction safety unverified. **Tracker:** [#157](https://github.com/xenodeve/Strata-xeno/issues/157); related behavior #61/#143.

F `src/program/generate.cpp:6458-6517` and `:8659-8724` repeat stage3 publication/host release and stage2 copy-home/acquire/H2D transitions. Selection and stage1 also repeat at F `:6521-6563` and `:8739-8781`. This is two roughly 60-line transition bodies plus repeated planning; an ownership or error-path correction has two change sites in the14-hunk merge hotspot.

U `src/program/generate.cpp:4777` and `:6437-6455` has corresponding mode-local non-paired pending publication, but no exclusive paired ownership protocol. **Decision:** keep the fork's protocol and extract identical primary transitions in a separate approved refactor. Leave mode orchestration, per-stage instrumentation, stage handling and generate-only paired secondary behavior outside that seam.

Keep acquisition before home-copy, held-slot lifetime across parallel copies, victim admission before CPU publication, incoming host release after H2D and lent-tail exclusions. Characterize each state transition and error before moving one seam per commit. Runtime/numerical behavior must remain unchanged.

There is also a traced limit to U#463: F `apply_pending` returns when `pending` is empty (`:6391,8478`), while paired H2D entries live in `ps_h2d` and publish after non-blocking queries (`:6460,8662`). U's wait at `:6590` therefore does not make F's paired path timing-independent. That is a mechanism observation, not a reproduced cause of #61/#143 or #149; any deterministic-policy change requires separate measurements and approval.

### R3. Two grouped-copy primitives serve one job

**Kind:** redundant code and an unnecessary unused overload. **Evidence:** **traced**. **Tracker:** [#154](https://github.com/xenodeve/Strata-xeno/issues/154).

F `include/strata/prefill/moe_mmq.hpp:82,92` declares two `gather_native_group` overloads. F `src/prefill/moe_mmq.cu:302-321` uses compact pointers and internally falls back; F `:323-339` is U's `GatherGroup` form, equivalent to U `moe_mmq.cu:229-245`. Both have their own grouped-copy kernel form. Only the fork overload is called, at F `prefill.cpp:1232,3327` and `tests/xeno/gather_group_parity.cu:43`; the U overload has no caller in this tree. Adjacent size_t parameters swap `down_off` and `gu_half_bytes`, creating an undetectable-at-compile-time argument-order hazard.

**Decision:** adopt U's primitive, retain F's streamed/split walk. A compact group with `first=0,n=pending_count` plus the existing shifted destinations represents F's subgroup. U returns false for unsupported alignment/shape without launching; the caller must explicitly retain the per-expert fallback. Converting to absolute group indices while also retaining destination shifts would double-offset the output.

**Cost:** two signatures/kernel forms, an unused production API and its non-CUDA stub, plus repeated future-merge resolutions. No acceleration is promised. **Safety:** red-first byte comparisons for unaligned inputs, distinguishable role offsets, partial groups at nonzero destination positions, mixed ring/cache pointers and ring-induced flushes. Existing group-size tests do not establish all those new cases. Run parity on sm_89/sm_120, fingerprints and same-session greedy ABBA; retain GROUP_GATHER's A/B lever. Keep R1 out of the primitive-only commit.

### R4. GGUF readers duplicate geometry, but have different ownership contracts

**Kind:** redundant geometry and unnecessary generic skip surface. **Evidence:** **traced**. **Tracker:** [#155](https://github.com/xenodeve/Strata-xeno/issues/155).

F `src/core/expert_source.cpp:2567,2647,2863,2890,3259` independently derive gate/up/down lengths and destination offsets. The existing `expert_ranges` helper at F `:2882` is already consumed at F `:3615`. U `expert_source.cpp:2421-2422,2499-2500` also repeats geometry, so wholesale U adoption does not solve this duplication.

F `load_experts_gguf` (`:2534`) has a skip argument/branches, but its production caller (`:2781`) passes null and `tests/core/expert_layout_test.cpp:192` uses the default null argument. No repository caller supplies a non-null skip. U's API (`:2388-2389`) has no skip. That generic extension can return toward U when its call inventory still proves there is no ownership-aware consumer.

The live `load_rest` reader is different: F `:3240` uses32-expert aligned DirectFile batches, skips entirely GPU-owned batches before reading (`:3274`) and scatters through slab `dst_of` (`:3288,3324`). U uses16-expert Windows ReadFile batches into contiguous blob offsets. Replacing F with U as-is would lose the CPU-owned-only commit/slab contract from #4, or write into pages that should remain uncommitted.

**Decision:** converge geometry first; keep the fork's ownership-aware reader until a shared lower-level transport demonstrably represents both destinations, skip-before-read and error semantics. Do not change chunk/thread counts as incidental cleanup. **Cost:** at least five geometry sites and two unbuffered worker bodies; recurring shard/format merge edits, startup/private-commit risk. No measured per-turn saving.

**Safety:** role shards crossing within a layer, unequal role sizes, short/error reads, arena/slab destinations, exclusion of GPU/NVMe experts, loaded-byte comparison and cold-start private-commit/stage measurements. Then D2x fingerprints and same-session greedy ABBA. Never reintroduce eager whole-arena loading merely to remove a function.

### R5. Split configuration has three independent parsers

**Kind:** redundant policy. **Evidence:** **traced**. **Tracker:** [#156](https://github.com/xenodeve/Strata-xeno/issues/156).

F `src/program/generate.cpp:2490` reads `STRATA_PREFILL_EXPERT_SPLIT` for peer refusal; F `:2670` reads it for split layout/wave setup; F `src/prefill/prefill.cpp:2245` independently caches it for execution. U has no fork expert-split setting; its own peer refusal is at U `generate.cpp:1718-1723`.

**Decision:** resolve the fork option once and hand that immutable decision to its consumers through a narrow existing setup seam. Preserve absent/0/nonzero atoi behavior, exclusive-secondary eligibility, wave dependency, split/routed thresholds and every A/B lever. Do not add a generic environment/configuration framework for one setting.

The adjacent refusal blocks F `generate.cpp:2480,2489` check different combinations: upstream layer/remote-cache conflicts versus fork secondary-tier/exclusive/split conflicts. They are separate constraints, not duplicate refusals to remove. Their presentation can be consolidated after one split decision exists.

**Cost:** three parser/policy sites and two validation surfaces, not a demonstrated runtime bottleneck. **Safety:** configuration/refusal tests for every combination, then split/wave/partial-MMQ fingerprint and greedy ABBA gates.

### R6. SecondaryRunner repeats its completion tail

**Kind:** redundant code. **Evidence:** **traced**; no stale-state bug demonstrated. **Tracker:** [#159](https://github.com/xenodeve/Strata-xeno/issues/159).

F `src/core/secondary_runner.cpp:363-374,396-412` repeats timeline closure, completion-event recording, STRATA_SECONDARY_POKE parsing/query, pending publication and launch accounting. U has no `SecondaryRunner`; its closest grouped compute call is U `src/core/verify.cpp:758`. U's PeerExperts is an additional decode-tier design with a different ownership contract, not a drop-in replacement.

**Decision:** retain the runner and graph/direct/profile/async modes; share only the completion tail when approved. The enqueue bodies differ in error checks, markers and grouped bounds (`entries,entries` at F `:346` versus `host_count_,dst_.size()` at F `:386`), so a larger shared enqueue helper is not automatically justified.

**Cost:** two completion protocols and two static caches for the same setting. No speed claim. **Safety:** completion-event failure/pending characterization, Q2 and IQ graph/direct parity, multi-token/format changes, timeline counters, main-model fingerprints and greedy ABBA. Retain SECONDARY_POKE.

### R7. Administrative Origin policy is applied twice and disagrees

**Kind:** redundant security policy and behavior improvement. **Evidence:** **traced and reproduced**. **Tracker:** [#153](https://github.com/xenodeve/Strata-xeno/issues/153).

F `serve/server.py:3430-3434` calls `_foreign_origin` (`:2884-2890`) and then `_own_page` (`:3551-3566`). The first ignores trusted_origins; the second honors them. U uses only `_own_page` at U `server.py:2230`, with its trusted-origin clause at U `:2349-2352`.

With mock engine side effects replaced by recording functions, configured trusted Origin `https://strata.example.com` and JSON receive403 from F's `/load` and `/unload`,200 from U, and200 from F's `/v1/load`/`/v1/unload`. Untrusted JSON is403 in both trees. Own/no-Origin JSON is200. The probe ran 32 cases per tree. Foreign/trusted plain-content response ordering also differs (403/415).

**Decision:** converge on one explicit administrative-origin policy after deciding whether the fork intends to trust configured proxy origins for both route families. Do not weaken the boundary as incidental refactoring. Preserve or deliberately document response ordering, authentication and refusal before side effects. This finding does not establish an untrusted-origin bypass.

**Cost:** duplicate policy and a trusted-proxy route inconsistency; no measured D2x latency saving. **Safety:** own/trusted/foreign/null/no-Origin × JSON/plain × key/no-key route tests. No security code was changed.

### R8. Read-only Git classification drops a repository selector's outside path

**Kind:** correctness improvement at a fork-only trust boundary. **Evidence:** **traced and reproduced**. **Tracker:** existing [#110 evidence comment](https://github.com/xenodeve/Strata-xeno/issues/110#issuecomment-5978825518).

F `serve/permissions.py:298-311` recognizes Git show and checks a list of bad flags. F `:318` then discards arguments beginning with `-`, including `--git-dir=<outside>/.git`. The remaining `HEAD:marker.txt` is checked as a filesystem name under the project (`:322-326`). F `:469-470` permits it before the plan-mode refusal. `serve/agent.py:255-275` forwards an allowed call without asking; `serve/shell.py:121` gives the unchanged command to the selected shell.

In benign temporary repositories, `git --git-dir=<outside>/.git show HEAD:marker.txt` receives allow in normal, plan and auto modes and Git reads `BENIGN_OUTSIDE_SENTINEL` outside the project. Direct cat of that outside file asks/plan denies; separated `--git-dir <outside>/.git` does not receive automatic allow. No actual secret was read and this is not a claim that every Git argument variant is exploitable.

U has no `serve/permissions.py` or coding-tool policy equivalent. **Decision:** retain the fork feature, validate repository/work-tree selection against actual command semantics or exclude those selectors from automatic read-only classification. The established residual item in #110 owns follow-up; no duplicate issue was opened.

**Cost:** loss of the intended project-read permission boundary, independent of latency. **Safety:** red-first policy and actual-executor tests with benign outside fixtures, both flag forms, every permission mode, ordinary in-project positive controls, protected/secret policy and command-deny precedence. No security code was changed.

### R9. Fork CPU-order predicates use an unnamed Q2_0 type

**Kind:** improvement. **Evidence:** **traced**. **Tracker:** [#158](https://github.com/xenodeve/Strata-xeno/issues/158).

F `src/kernels/cuda/iq_kernels.cu:1760,1779,1784,1786,1793` has seven literal42 occurrences in five CPU-order decision lines. F `include/strata/core/placement_formats.hpp:15` already names kGgmlQ2_0. U uses numeric IDs in its common format registries (`iq_kernels.cu:495,510-512`) but has no fork x_scales CPU-order branch.

**Decision:** name only the fork's special numerical predicates consistently, using a narrow constant dependency. Preserve U's numeric registry style; a full registry rewrite increases conflicts without addressing this fork contract. **Cost:** seven decision literals and recurring merge review of the CPU-order gate; maintenance only. **Safety:** build both architectures, native_q2_pool_hit_parity/relevant grouped parity, fingerprints and greedy ABBA. Never alter CPU-order arithmetic during naming cleanup.

### R10. Capacity-mode spans attach post-miss counts to pre-miss intervals

**Kind:** diagnostic improvement. **Evidence:** source **traced**; runtime reproduction outstanding. **Tracker:** [#160](https://github.com/xenodeve/Strata-xeno/issues/160).

F `src/core/expert_source.cpp:2169-2170` times the resident run using the current njobs. Deferred entries increase njobs at F `:2175-2179`; `cpu pool misses` carries their count at F `:2184`. The later `cpu pool` event uses the increased total at F `:2235`, although its interval covers only the first run. `dispatch jobs` at F `:2234` likewise attaches the final count to an earlier interval. With d deferred new jobs, the resident span's job argument overstates resident work by d. Timeline args are generic a/b fields (`include/strata/timeline.hpp:40`); no analyzer's duration or wall-time sum is claimed wrong.

U has one pool run (`expert_source.cpp:2068-2069`) and no deferred/timeline equivalent. D2x has no NVMe expert tier, so this observation is capacity-mode-only. **Decision:** first reproduce the emitted argument mismatch at the real dispatch seam; if confirmed, freeze the resident count before deferred additions while keeping execution and numeric behavior unchanged. **Cost:** misleading diagnostic cardinality, not measured serving latency.

**Safety:** actual emitted events for mixed/zero/all misses and source ownership/byte parity. The existing `tests/xeno/nvme_overlap.cpp` exercises source begin/end state, not this event argument; do not claim it proves the new assertion. Add the dispatch-level test red first, then the normal engine correctness gates after approval.

## Candidates retained or rejected

| Mechanism / candidate | Traced comparison and decision |
|---|---|
| “PLE read-ahead twice” | Rejected as duplicate execution. F `prefill.cpp:2177-2219` has one future/take/upload; F `:2572` lands it at the same rule as U `:1649`. U `:1433` is the equivalent ple_land. Keep the already-converged semantics, fork PLE_AHEAD arm and wave stride (`F:2213` c0+stride versus `U:1453` c0+m.T) |
| End-of-stage PLE take | F `:2059` requires LB<=1 and LE>1; F `:3474` cannot receive pending rows for a stage ending before layer1. The analogous U eligibility/landing structure has the same redundant fallback. Keep the harmless upstream-shaped no-op for now rather than invent a dead-path cleanup with no measured value |
| `--ple-inflight`64 versus 256 | Deliberate queue-depth divergence; U's default256 is an A/B arm, not evidence that F64 should be removed. No new SSD measurement here |
| Upstream PeerPrefill versus fork split/wave | Keep F. F `prefill.cpp:2367` skips CUDA0-owned expert copies and `:3469` publishes fallback-layer wave handoffs. U's PeerPrefill at `:1103` lacks this owner/wave contract. #115 amended #113's full-split restriction; #142 permits ahead-of-routing copies. Replacing those rules using stale acceptance lists would regress approved behavior |
| set_peer / set_peer_tier | F `prefill.cpp:1852-1859` explicitly declines upstream peer prompt sharing; F `:1654` sets the fork's secondary ownership tier. Keep these distinct capabilities and the refusal. Their similar names are a documented naming hazard, not proof that one function or tier is dead |
| Secondary tier / PeerExperts / remote experts | Keep separate supported modes. F `generate.cpp:2480-2499` explicitly refuses their conflicting combinations. Shared grouped kernels already receive upstream#363; ownership/slot layout is not thereby equivalent |
| Q2 CPU-order path versus upstream fused IQ path | Keep both. F `iq_kernels.cu:1779-1793` distinguishes the numerical contract; replacing it with U's general fused path would violate the accepted Q2 baseline. R9 only names predicates |
| Fused CPU pool and its reference path | Keep both and POOL_FUSED. #147's historical pool ABBA ends `DONE exit0` with identical outputs. A deliberate measured A/B alternative is not dead code |
| Parked cache slots versus upstream conversation-cache options | Keep F's physical checkpoint/slot lifecycle. U `generate.cpp` has conversation_cache_slots/cache_slot_off options, not F's CacheSlot ownership/async parking mechanism. #145's SLOT_SYNC arm remains intentional |
| Capacity, placement and memory-budget helpers | Keep used ownership, slab, admission, mirror and secondary-budget mechanisms. Searches found no U set_capacity/materialize_begin equivalent in src/include; U's file/read APIs do not implement the fork capacity contract. No new dead helper established |
| Default-on optimization levers | Retain SLOT_SYNC (#145), POOL_FUSED (#147), SPLIT_ROUTED_MAX (#142), IO_THREADS (#139), GROUP_GATHER (#29), PLE_AHEAD (#31) and the other supported diagnostics. A settled default is not permission to retire its A/B arm; no lever-retirement proposal was made |
| Blueprint older baseline | Discovery-pointer issue already recorded in #110, not a new engine finding. This report uses fresh pinned references; it does not silently move the served baseline or rewrite old measurements |

## Independent review axes

### Standards

0 new hard documented-standard violations;2 conservative judgment calls from the reviewed engine subset. The role geometry is repeated across expert readers (R4), and graph/direct secondary enqueue/completion repeats common work (R6). The parent narrowed the latter recommendation to completion only because bounds, markers and error checks differ. The additional capacity diagnostic observation is R10, explicitly without a model reproduction. Intentional A/B modes and used ownership/budget helpers were not treated as speculative generality.

### Spec

0 new hard missing/incorrect/scope-creep requirements demonstrated in the assigned prefill/reader scope;4 traced conclusions. Group primitives can converge with fallback and destination guards (R3). PLE is already one integrated path, including #31's A/B arm. Placement-first scatter/skip behavior implements #4 and cannot be replaced by U's eager contiguous API. Split/wave implements the amended #115/#133/#142 contracts; earlier full-split/exact-copy checklist text is superseded. The group-release mapping is an upstream opportunity, not a proven unfulfilled #29 acceptance item.

**Axis totals:** Standards0 hard/2 judgment calls (largest: reader geometry); Spec0 hard/4 traced conclusions (largest risk: replacing an ownership contract with a superficially equivalent upstream API). These counts cover the independent axes, not all parent-reviewed server findings above.

## Verification, implementation gates and artifacts

Executed in this session:

- Confirmed both refs, ancestry/diff scope, clean review tree before documentation and origin/main still at ad28ddb during review.
- Source reads and caller searches for every accepted finding; the parent checked the child evidence against both pinned trees. One preliminary upstream refusal line in #156 was wrong; it was corrected from1773 to1718-1723 after reading the actual source, in both language sections before completion.
- A final repository-wide caller search also found the default-argument GGUF test call at `tests/core/expert_layout_test.cpp:192`. The earlier wording "only repository caller" was corrected in R4/#155; the narrower conclusion that no caller supplies a non-null skip remains supported.
- `python -B C:/Strata-exp/review-origin-probe.py <tree>`: 32 HTTP cases in each pinned tree, mock only. Results at `C:\Strata-exp\review-origin-fork.jsonl` and `review-origin-upstream.jsonl`.
- `python -B C:/Strata-exp/review-permission-probe.py`: 12 policy controls across normal/plan/auto plus an actual benign external-Git read. This executes no user-supplied or secret-reading command.
- `python -B -m unittest serve.test_permissions serve.test_server.SharedSettings serve.test_server.WebSecurity -q`: 57 tests, OK; log `C:\Strata-exp\review-existing-tests.log`. The existing green tests do not cover the two reproduced gaps.
- GitHub follow-ups #153-#160 created with English/full Thai mirrors; #110 receives the scoped security evidence. Existing perf/determinism work is referenced rather than duplicated.

**Not run:** C++/CUDA build, ctest, full pytest, GPU parity, STATE_HASH, greedy engine ABBA or latency benchmarks. Historical merge-report105/107 and pytest1091 figures are not this session's test results. The tracked changes are Markdown report/index only, so engine TDD/build/parity is future acceptance work, not a claimed green gate for this review.

For any later approved engine change: issue → branch → meaningful red-first tests under tests/xeno registered in CMakeLists.txt; byte-identical main-model STATE_HASH at max_tokens=1 using the same-session baseline; greedy raw-token ABBA against the deployed exe; paired stage counters/timeline and both `--pool-priority 2`/`--process-priority 2` in every performance arm; exact exe hashes. Do not compare sessions, change the serving config/launcher or touch the daily server. Changes to numerical policy, ownership or security need their own explicit decision; do not mix them with relocation.

Working evidence and issue snapshots remain under `C:\Strata-exp\review-*`. The report/index are the durable repository artifacts; the GitHub issues carry the proposed implementation inventory and acceptance criteria. Skill-rule deviations are reported separately on xenodeve/xeno-skills according to the global session-report rule, not presented as engine defects.

Skill feedback actually filed: delayed start-before-exploration ordering ([existing #134 comment](https://github.com/xenodeve/xeno-skills/issues/134#issuecomment-5979020145)); the prematurely published source anchor subsequently corrected ([#397](https://github.com/xenodeve/xeno-skills/issues/397)); the conflict between the T4 delegation prohibition and code-review's required parallel axes ([#398](https://github.com/xenodeve/xeno-skills/issues/398)); the source/include-only caller survey that missed a test ([existing #105 comment](https://github.com/xenodeve/xeno-skills/issues/105#issuecomment-5979084379)); and the omitted ranked hypotheses during the review probes ([#399](https://github.com/xenodeve/xeno-skills/issues/399)). No local Obsidian library note was written because this session ran in Strata-xeno. No other rule failures were identified in the final self-check; this is an observation, not an automated compliance guarantee.

**Verdict:** retain the served architecture, converge narrow duplicated primitives/state transitions through approved parity-gated follow-ups, and settle the reproduced server policies before treating that boundary as covered by its existing green tests.
