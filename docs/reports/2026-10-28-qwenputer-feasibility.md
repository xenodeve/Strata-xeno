# Native Qwenputer P0 feasibility — 2026-10-28

P0 is the only authorized implementation phase. The compiled arithmetic circuit, model bridge and decoder integration are not implemented by this diagnostic. PRD: [#276](https://github.com/xenodeve/Strata-xeno/issues/276), approved R2; the English/Thai published body was read back exactly (62,037 characters, SHA256 `066617000239adb3708d2b7fc2d5e0220dbcf0feb6f57fb4c07dfe1ea0e3d59e`). No implementation issue breakdown, PR, push or merge was performed.

## Source, isolation and instructions

- Baseline `e5737a27f485773a7ce22784fef6af7b480dcc9a`; branch `xeno/native-qwenputer-p0`, isolated under `D:/Github/Strata/.worktrees/native-qwenputer-p0`. Main's unrelated files remain outside this worktree.
- Loaded the installed `xeno-skills/skills/t4/using-t4/SKILL.md` and relevant planning, TDD, execution and memory instructions. The local `C:/Users/xenod/.agents/skills/to-prd/SKILL.md` requires tracker publication with `ready-for-agent`; issue #276 corrects the earlier local-only deliverable. `/to-issues` remains separate. The approved `/to-prd`-first order is retained.
- App-managed worktree creation is unavailable for this durable task; manual Git fallback was used after confirming `.worktrees` is ignored and the target absent. No existing checkout was reset.
- Evidence/build directory: `C:/Users/xenod/AppData/Local/Temp/strata-qwenputer-p0-20261028`. No downloaded third-party compiler, reference, firmware or model was executed. Existing vendor sources were reused; no install/download of weights occurred.
- Fast-mode preference is recorded; active task model/service tier cannot be verified or changed through available supported controls. No change is claimed.

## Hardware and build provenance

Primary is physical GPU 1, RTX 5060 Ti, UUID `GPU-059b90e2-2b5c-00b8-f3ba-f6dea8de083e`, 16,311 MiB. Secondary/display is physical GPU 0, RTX 4070 SUPER, UUID `GPU-fba37e4b-ea9e-66e9-c3fd-a16b2e833bc4`, 12,282 MiB. Both report WDDM. Engine mapping is `CUDA_VISIBLE_DEVICES=1,0`; the final probe checks the resulting primary UUID. Driver package is 621.01.

CPU is i5-13500, 20 logical processors; physical RAM 51,208,699,904 bytes. Preflight available physical memory was 25,140,346,880 bytes before and 25,238,638,592 after. These are snapshots, not measured model peaks. The 42 GB Strata private-commit budget and 2.5 GB total display headroom policy remain unchanged; no full model memory-compliance result is available.

Build compiler: CUDA 13.3.73 in the v13.3 toolkit (SDK version.json 13.3.1), MSVC 19.44.35228.0, Ninja, Release. Actual runtime/driver API values and loaded CUDART identity are recorded separately in the probe JSON; they must not be inferred from a directory name. The generic ggml CPU mode warning is recorded, not treated as a speed claim.

Unchanged baseline built 152/152 targets with root CMake hash `5b21527aaabdbd52b30cc3d10d8300cb32c58b342b459d40bfb328076bbb2382` and no tracked diff before build. Executable is 108,903,936 bytes, SHA256 `3948784248ce3193ddd636792ce6f377fdb5e727eea5b241dcb279b8b8e46042`. `cuobjdump --list-elf` found 59 sm_89 and 59 sm_120 entries. `--help` exited 0; executable hash remained unchanged. This is not a secondary-expert inference smoke.

Final candidate build/probe identities and observed outcomes are appended below.

## Historical Swift inventory (superseded by Gyro-M)

Profile `D:/Github/Strata/strata-flash-next-d2x.json` SHA256 is `07557c23cbac89c467c627f5723ba14a5142084ddb33ff7a1f88e7b28714f535`. `inputs.json` pins every configured executable, tokenizer, expert pack, MTP and profile input plus vendor files. Full files are streamed through SHA256; directory digests combine sorted relative names, byte counts and individual file digests. Search scanned 172,818 files without errors in existing model storage, staging, this model's HF cache and existing engine storage. No exact shard filename or matching content-addressed blob filename was found. This does not claim a scan of every drive.

| Role | Exact path / identity | SHA256 or tree digest |
|---|---|---|
| Served executable | `C:/Strata-exp/run-d2x-260/strata.exe` | `da93215c3eda1ae9e8e1d50ec7872696a9d502d51e9690fabdae48f499b005ed` |
| Tokenizer | `D:/Github/Strata/packs/swift15-iq2_xs/tokenizer` | `9016263eb8b85a4f9fc43c456c42426954836339945945f11f58a6ae992e0f99` |
| Expert pack | `D:/Github/Strata/packs/swift15-iq2_xs` | `0cdc7fa04c61a0cf4686815f0cfba498a7c072891367edfe23ae33a8802daa57` |
| MTP | `D:/Github/Strata/mtp/rt` | `6d5680c0a9346d807da0535fb7e6bbd9fdd79d47ad64edc6f84d648394c52aab` |
| Expert profile | `D:/Github/Strata/data/ranked-exl3only-profile.bin` | `a796734ba1489f10a2fe3daec34071da2ebd5add66a963d9a0469f4af925393d` |
| Existing ggml source | `D:/Github/Strata/third_party/llama.cpp/ggml` | `6872d48be8fb540ad4d82a12ff6ea51f2c7762cece2b69ea1752eb5dd3ddeb98` |

The parent Git repository's HEAD reported from a non-independent vendor directory is not a ggml revision proof. The version file pins `3cf03257f219afbe7334045ff7c6a06ac68c627d`; the actual existing source tree is separately hashed above.

Both required files belong at the configured native/PLE directory `C:/AI/models/ukisai-Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS/`:

| Filename | Exact bytes | SHA256 |
|---|---:|---|
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` | 39,788,473,344 | `ca3b302d3d1eb6eddce3ea4acc79854e6cc5fdc0583e374a6a8b6bb67327743b` |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf` | 28,363,693,824 | `e4996a6de18fd49c58d9a0040fcfd73787db6407c51973f29884bee7b14c9dbb` |

Total 68,152,167,168 bytes (63.47 GiB). Local SHA256SUMS agrees with pinned [Hugging Face metadata](https://huggingface.co/api/models/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d?recursive=true&expand=false), retrieved as metadata only. Neither observed C: nor D: free space can hold this pair. Do not delete user files, change profiles/quantization or download silently. This was a historical Swift limitation. The user subsequently selected Gyro-M; no Swift restoration is needed or proposed.

## Scope and verification limits

The task-owned GPU diagnostic increments a counter and checksum to validate graph control flow. Its host checksum assertions are test oracles, not production arithmetic, and its elapsed times are diagnostic observations, not an arithmetic latency benchmark. It allocates before launch, initializes the condition on every launch, uses one same-device WHILE body and no body host callbacks/allocations. Cancellation is requested from a separate stream; maximum work is 4,096 finite steps. Rebuild and current-device-switch checks do not prove integration with an actual language decoder.

No greedy raw-token parity, model load, served D2x secondary expert smoke, Qwenputer circuit replay, ordinary-turn overhead, native arithmetic success rate, timeline or ABBA was run. No performance/default change is proposed. No prefill/cache/decoder/server code changed. HIP and SYCL toolchains are unavailable; their build validation is explicit debt under AGENTS.md, not a pass. No target machine security or TDR settings changed.

## Reproduction and deviations

Commands use `tools/qwenputer/headless.py`, which invokes existing tools with CREATE_NO_WINDOW, loads vcvars in a child only and prepends CUDA bin plus bin/x64. Full engine configurations use quoted `-DCMAKE_CUDA_ARCHITECTURES=89;120`; the standalone primary-only diagnostic may use 120. Exact argv and DONE exit markers are kept in each log.

- Default sandbox MSVC configure failed C1902; the identical installed toolchain configure passed with scoped elevated access. No PDB/toolchain/security modification was made.
- An initial unquoted PowerShell CMake argument split the ggml path and configured llama.cpp. That configure directory is rejected as baseline evidence. A new correctly quoted Strata directory was configured, and source/project/architectures were verified.
- Do not rebuild the completed baseline directory after candidate CMake edits. Its recorded executable remains immutable.
- The Bash SDD workspace helper cannot start in the sandbox; equivalent plan-scoped ledger files were created directly. No completion was inferred from a failed helper.

P0 is not a release or readiness approval for P1. The Gyro-M profile choice and model validation, plus backend validation debt, remain explicit gates. The only public issue operation is the approved PRD #276; implementation tickets and publication beyond it remain separate.


## Current Gyro-M test baseline (user steering, local-only)

The user selected Gyro-M for daily-use testing. No profile/default was edited and no model was loaded. `gyro-inspection.json` parses only the GGUF header using the existing project reader; `gyro-inputs.json` streams full SHA256 once per unique input (native/PLE aliases share the same file). Both metadata and hashing completed. The whole 91.97 GB GGUF was read for its digest; this was not inferred from a filename or remote checksum.

Verified metadata: GGUF v3, general.architecture=qwen4exp, name Qwen3.8-Flash-Next Gyro-M, 48 blocks, hidden size 2560, 512 experts / 10 used / expert FFN 640, native context 262144. Tensor counts: Q8_0=667, F32=389, TQ2_T=144, BF16=24. Pack format v5 declares Hadamard-folded TQ type144 experts; hadamard.txt exists. This is not Swift IQ2_XS.

Tokenizer: gpt2/qwen35 descriptor, vocabulary248320, separate vocabulary/merges/types. MTP uses the existing rt dense/experts/draft-vocab files. Both profiles reference it with spec4; runtime numerical compatibility still requires execution. Geometry and file existence are not a compatibility pass.

Hub evidence distinguishes key3/native262K from key3y/512K YaRN. No active --config could be identified by the permitted process inspection. Native262K uses5 workers and run-gyro-d145r; 512K uses13 workers, YaRN scale2 and run-gyro-reader. Both preserve secondary expert offload, prefill split/wave, KV int8/resident65536, display reserve3277, MTP/suffix and an expert-profile-save target. Profile selection was sent as a concise pending user question; no model-dependent comparison was launched while waiting.

Any later controlled run must copy/freeze the learned expert profile into task-owned files and redirect its save destination so a baseline arm cannot mutate the candidate's input or the daily-use learned profile. Use a private port if a server is needed; do not stop/reconfigure the daily-use server. Test configuration copies must preserve the selected serving settings and document any justified difference, including vision companion/accounting.

| Current input | SHA256 / tree digest |
|---|---|
| `D:\Github\Strata\strata-gyro-m-d2x.json` | `06ef847e063bb9643c7bf3f486e2ffdadc29737493268238c21f59188d5f6c5b` |
| `D:\Github\Strata\strata-gyro-m-d2x-512k.json` | `8da3719abc96fe0597f5e84ecf326f12c880729ab52c9ce54229e1b5ca4a1d15` |
| `C:\Strata-exp\run-gyro-d145r\strata.exe` (144387584 bytes) | `8fbf47992bb7539eaa6cb049efa6423f09927ecf2cee4c9cecd321479926b300` |
| `D:\Github\Strata\packs\gyro-m\tokenizer` (10093356 bytes) | `cbad9f317a1b84da005fcecbd6b3d4423d7ce3f53cc45b0ccab8cbb5aeb22b88` |
| `D:\Github\Strata\packs\gyro-m` (4962575916 bytes) | `75bfe5952f2affc039067f5f3b96e1495d044c9f990349794bb86da872f0d009` |
| `C:\AI\models\agentionai-Qwen3.8-Flash-Next-Gyro-GGUF\Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf` (91965774272 bytes) | `7aa36d15131d255179ab6c87fa9c9f81a0b541fc9e81e959d4c3e29eb20feb5f` |
| `D:\Github\Strata\mtp\rt` (824236860 bytes) | `6d5680c0a9346d807da0535fb7e6bbd9fdd79d47ad64edc6f84d648394c52aab` |
| `D:\Github\Strata\data\gyro-m-learned-profile.bin` (196632 bytes) | `3e7699a41dd3acc063541af2119079bc5174c67e811cf26527de512858e7cdba` |
| `C:\Strata-exp\run-gyro-reader\strata.exe` (127988224 bytes) | `4eca09d932ed1a706c35aa8094e70529d712704efe290ac744fb3551cbb1e209` |

No revised PRD or local model/profile details were published. Parent reports that automatic approval review rejected the proposed public Gyro-M issue edit because authorization to disclose local details was missing. This execution neither retried that publication nor uploaded R3. GitHub #276 remains the previously authorized R2 PRD.

## Final component results

| P0 check | Observed result |
|---|---|
| Unchanged engine build | PASS152/152, sm89+120,59 ELF entries each, help0, immutable hash recorded above |
| Candidate engine and diagnostic build | PASS154/154, sm89+120,59 ELF entries each, help0 |
| Real graph create/instantiate | PASS on RTX5060Ti; primary UUID checked from JSON |
| Graph control flow | PASS0/1/17, repeat5/repeat0, output-bound3, cancel, current-device switch, recreated graph1/0 |
| Negative control | PASS detector: real GPU increment2 produces one.steps2/expected1, ok=false, exit1 as intended |
| Final full helper/GPU suite | PASS9 tests,3.814s,0 failures, DONE exit0 |
| Gyro input inventory/full hashes | PASS, all configured engine/model/tokenizer/pack/MTP/profile inputs exist |
| Profile selection | RESOLVED:Gyro-M262K/5workers selected by user |
| Model/secondary tier inference smoke, greedy parity, ABBA | Baseline startup attempted and failed before token0; actual secondary execution/candidate/parity/ABBA unpassed; see current attempt below |
| Full model resource gate | NOT RUN; inventory snapshots and diagnostic process metrics are not full model peaks |
| HIP/SYCL | NOT RUN: toolchains unavailable; explicit validation debt |

- Candidate probe: `C:\Users\xenod\AppData\Local\Temp\strata-qwenputer-p0-20261028\candidate-build\tests\qwenputer\qwenputer_conditional_probe.exe`, 102912 bytes, SHA256 `af91478941e6c6331597ddf9dc0b870f2ff120cee7eca40956491b8865b405fc`.
- Candidate engine: `C:\Users\xenod\AppData\Local\Temp\strata-qwenputer-p0-20261028\candidate-build\strata.exe`, 108903424 bytes, SHA256 `550499c3ea61f4077e4c5fca61fd202889498a156fc8d6c0b44c53ee2a68040e`.

Probe API values: headers=13030, runtime=13050, driver_api=13050; observed values are retained without assuming they equal the compiler version. CUDART DLL-name lookup was empty, so a loaded DLL identity is not established by it. Primary free VRAM before/after=15908995072/15908995072 bytes; this net observation is not a peak. Process private at end=511991808, peak working set=232329216 bytes. Cancel completed at 369 of4096 steps. Timings in probe.json are diagnostic observations only.

## Independent review and remaining work

A fresh-context Astra reviewer read the P0 code/diff and requirements without edits or execution. No Critical findings. One Important finding was verified: the old launcher wrote DONE after killing only the immediate child, while a grandchild survived. Its real parent/grandchild test failed before the fix. The launcher now assigns a waiting worker to a task-owned Windows kill-on-close job before releasing a stdin handshake. Commands and descendants inherit that job. Timeout or normal completion terminates/drains the job before DONE. Assignment failure cannot start the tool. The regression test and final9-test suite passed.

Deferred minor: the probe source comment says recreated allocations have fresh addresses; CUDA may reuse addresses. The test establishes destruction/recreation/state reset, not address relocation. No relocation claim is made here.

The reviewer set aside P1+ production arithmetic/integration because outside authorized P0; final-source GPU/provenance checks because then pending (now executed above); Gyro model runs because profile selection remains pending; HIP/SYCL because unavailable (still release-validation debt). None is silently counted as passed.

R3 local reading draft applies the actual installed kien-thai skill (technical explainer, topic-first/condition-first structure, clear paragraphs, consistent developer terms, no polite padding or recap). Its full24/10/9 coverage and verbatim preserved R2 contracts are recorded separately. Native-model availability check contacted localhost11434 only and was refused; no model/service was started or downloaded. The skill's direct drafting fallback was used.

Follow-up requested by parent: corrected the probe comment to describe destruction/recreation and reset only; allocator addresses may be reused, so this is not an address-relocation test. The diff changes comments only and no rebuild/runtime result is claimed for that wording edit.


## Selected 262K model smoke attempt (2026-10-28, private evidence)

The user selected the native262K/five-worker profile. Its SHA256 remained06ef847e063bb9643c7bf3f486e2ffdadc29737493268238c21f59188d5f6c5b. Runtime test copies redirect both learned-profile paths to task-owned files; source profile and original learned-profile hashes were unchanged afterward. Exact tokenizer vocabulary, merges and token types matched the GGUF; special IDs were checked, not inferred from count. Details are in gyro-tokenizer-binding.json.

Attempt01 stopped before launching the engine because Windows rounded the requested job commit limit to page size. Attempt02 verified the actual41999998976-byte limit (requested42,000,000,000), then ran the unchanged baseline with the selected profile, identical engine arguments/env and a21-token greedy smoke prompt, max-new16. No server, vision encoder, MCP or API listener was launched; vision accounting settings were retained. This is component smoke, not a full served-turn benchmark.

Attempt02 exited1 after14.641s, before any generated token, with placement-first load: VirtualAlloc(MEM_COMMIT) failed with Windows error1455. The resource controller itself did not signal its physical-floor/timeout stop. Sampled engine private peak33,921,974,272bytes, working-set peak5,388,955,648bytes; system commit peak78,573,899,776 of90,968,305,664bytes; minimum available physical19,596,664,832bytes. The job's committed-memory guard and sampled PrivateUsage are not interchangeable evidence; this attempt does not prove that the repo's private-commit budget is inherently exceeded. Do not silently increase limits or shrink the selected profile.

The startup loaded MTP weights and staged5607 secondary experts/6.82GiB in SECONDARY COMPUTE mode. This proves secondary-tier setup, not executed secondary compute or TQ2_T/MTP numerical correctness. No inference/parity/ABBA pass is claimed. Stock-vendor baseline logs also state no layers onMMQ and expert split cannot run, so this binary cannot be presented as validating the daily-use Gyro prefill split.

The selected profile's deployment comment records run-gyro-d145r/sha8fbf47992bb7 as xeno/defer-m145@9c3f16f8949473fbd3b3531608b4d2c930040308, defaultggml plus trellis overlay. That commit/branch exists locally. This differs from plan baselinee573 and its stock vendor. No exact matching deployed binary was found among top-levelbuild*/strata.exe candidates; profile comments are provenance evidence, not a reproducible binary-source proof by themselves. A user choice between the daily-use branch and the originalmain baseline has been requested; no source checkout switch, merge or further model retry was done while pending.

No task-owned model process remains running. Candidate model inference, actual secondary execution, greedy parity, resource fit and performance gates remain unpassed. Existing component build/probe evidence retains its original identity.

## Authorized R4 public synchronization

The later user authorization superseded the earlier public-PRD restriction. Existingissue#276 now matches the public-safe R4 Markdown exactly; bodySHA256 1dcf885081d171b87e8f07026877a3a81bd9ca713e492bcd8bdf1126515f8dba. Its bilingual normativeR2 technical appendix is https://github.com/xenodeve/Strata-xeno/issues/276#issuecomment-6479163682. No private paths/UUIDs/local resource logs were included. Three Mermaid diagrams passedMermaid11.16.0 parse pluslocalheadlessChrome render and visual inspection; actualGitHub renderedSVGs were not verified. No unrelated issue, repositorypush,PR,merge,deployment orlibraryupload occurred.
