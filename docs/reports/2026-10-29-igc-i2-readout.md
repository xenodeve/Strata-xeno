# I2 diagnostic: final-head readout fits training data but fails held-out operands

I2 is not complete. A 334,516-parameter readout can memorize the small training
set, but its operand tuples do not generalize to any of the 12 positive held-out
examples. This identifies a failed variant, not a functioning latent calculator
or evidence that activation mapping is impossible.

## Collection and identities

The isolated main-based candidate uses the actual selected Gyro-M 262K / five-worker
Hub server/settings. It is not the deployed daily executable. The existing dump
flags now capture the first verifier window's owning residual and mixed inputs,
paired logits, actual full token prefix and versioned position/geometry metadata.
Only a completed owned HTTP request publishes a dataset record. The reader refuses
wrong indices, malformed/truncated tensors, mismatched prefixes, nonfinite/stale
zero data and source identities that changed. This is opt-in diagnostic I/O, not a
serving or latency experiment.

The frozen corpus has 64 generated prompts: 48 training and 16 template-held-out,
balanced across no-computation/add/subtract/multiply. Positive operands have
magnitudes up to 99. The training templates include English and Thai; the 16 held-out
examples use different English templates. This is not held-out Thai coverage or
62-digit mapper validation. Negative prompts contain quoted arithmetic without a
request to calculate. Model responses are limited to one token for observation;
their answer accuracy is not the readout result.

Full content hashing of 22 enumerated native/pack/tokenizer/MTP/profile/build/server
inputs took 55.672 seconds; the native model still matches its pinned digest. The
manifest does not claim to identify every process DLL. The successful collection
took 250.125 seconds, including startup (ready after 19.641 seconds), diagnostic
copies, timeline, JSON and passive memory polling. No speed claim follows.

| Identity | SHA256 |
|---|---|
| Candidate executable | `813be71fa59a4ae6034bcead14207b77580f484fe84dea371e88bf0bd19cae31` |
| Enumerated source manifest | `5113a3e739cc3724ab486b099ba26ec9140e5b406d4b677af02a8e42578903a5` |
| Frozen corpus rows | `fce8f8c80d3c370804fec675dd99065ef1922afb562d3990ea9b386ed4df954b` |
| Trained readout state | `a3f39f9e2fb68455eeeb49b403fd6b3c5379320c385da22f36a2224569d28d47` |

All 64 completed observations validate. The owned job drained; source file
size/timestamp checks and served/original-learned-profile content checks stayed
unchanged. No private-commit cap was installed. Sampled process private commit
peaked at 47,006,109,696 bytes; minimum system commit headroom was 14,601,654,272
bytes and physical availability 9,413,672,960 bytes. These samples are distinct
from an OS-recorded lifetime peak. The 24 GiB physical admission is the selected
Hub launcher's own rule. Additional disclosed diagnostic system safety floors
were retained, with no workload/cache/offload setting change.

The first collection retained 45 successful observations before its 180-second
harness deadline. It had no engine error and drained safely. Its data was not
silently combined into the successful cohort. Observed request cadence justified
a bounded 600-second second cohort (outer 660 seconds), with unchanged resource
preflight and emergency floors. The second cohort completed once, without another
model retry. Both raw attempts remain available.

## Declared learning experiment

Input: one actual 2,560-dimensional `head_mixed` vector after final HC, at the fixed
answer-start anchor. A CPU MLP (2560 → 128 → 52, GELU) predicts nine categorical
heads: no-call/operator, each operand's sign/length, and two digits per operand.
Training normalization is fitted only on training rows. Operand losses are masked
for negative examples. No prompt text/token IDs or expected arithmetic answer is
fed to the MLP. The native Gyro weights and all activations remain frozen.

Seed 20261029, AdamW lr=0.001/weight_decay=0.01, 300 epochs, four CPU threads,
PyTorch 2.14.0+cpu; training deadline 120 seconds, outer bound 150 seconds. The fit
completed in 0.563 seconds, excluding import/collection. Hyperparameters were
chosen before evaluating held-out results; there was no held-out optimizer step
or parameter selection. Initial loss 13.494837, final training loss 0.002966.

| Check | Train | Template-held-out |
|---|---:|---:|
| No-call/operator class correct | 48/48 | 12/16 |
| Exact operand tuple, positive prompts | 36/36 | 0/12 |
| Correct operator and exact operands, positive prompts | 36/36 | 0/12 |
| False calls on negative prompts | 0/12 | 0/4 |
| Missed positive calls | 0/36 | 1/12 |

Held-out confusion (rows=true, columns=predicted; none/add/subtract/multiply):
`[[4,0,0,0],[1,2,1,0],[0,0,4,0],[0,0,2,2]]`.
Four negative examples do not establish a deployment false-trigger bound. Exact
operand scoring includes canonical signs, lengths and every digit; a correct
operator with a wrong digit is not a valid invocation.

## What this does and does not establish

The dataset uses genuine deployed-quantization activations and the readout's own
parameters receive gradients. It has no gradient through HC, attention, GDN, MoE
or the earlier frozen Gyro layers. It is neither the paper's attention over
pre-anchor states nor an output mapper. There is no learned invocation, decoder
injection, internal reasoning loop or model benefit demonstrated here.

Training fit plus held-out failure does not separate insufficient data from the
single-final-vector representation or model architecture. The next controlled
ablation should expose bounded selected pre-anchor states and compare against
this diagnostic variant, with a new untouched evaluation set frozen before making
new generalization claims. The current 16 examples are now known diagnostic data,
not a reusable unseen gate after architecture tuning. No hidden-state cache for
the entire 262K context is justified by this result.

Source inspection found the existing synchronized `Prefill::on_chunk` callback
supplies final multi-stream rows and absolute positions. Existing all-row debug
dumping rounds to BF16 and lacks per-request framing. A bounded F32 selected-row
diagnostic can use the callback without changing model arithmetic; it still needs
prefix/row identity, cache-missing masks and real-model checks. This is the next
I2 implementation experiment, not an already validated capability.

## Checks and evidence

The CUDA D2x build retains `89;120` and completed successfully. HIP/SYCL/Linux
validation remains unrun. Timeline analyzer: 24 tests pass. Readout/observation/
corpus tests: nine pass, covering train-only normalization, wrong digit and
noncanonical zero, false calls, prefix/geometry/finite data and frozen splits.
The unrelated five benchmark accounting tests also still pass.

Private evidence names: `igc-i2-source-identities.json`, `igc-i2-source-hash.log`,
`igc-i2-observations-01/` (partial), `igc-i2-observations-02/` (complete),
`igc-i2-collection-02.log`, `igc-i2-readout-01/{report.json,predictions.json,readout.safetensors}`
and `igc-i2-readout.log`. Collection and fit wrappers both end `DONE exit=0`.
The full project still includes I1b, I3/I4/I5 and programmable sandbox/debug,
geometry, store and later-model capability gates; this probe closes none of them.

## Follow-up: bounded F32 pre-anchor observation

The new opt-in collector uses both the existing fenced prefill callback and
`Verifier::copy_head_input` before prompt-window commit/reuse. The latter matters:
the selected Hub's `short-read=256` reads these short prompts through verifier
windows. No short-read/cache/offload setting was changed to obtain observations.

`igc_observation.hpp` retains at most 80 selected rows immediately preceding the
anchor, four HC streams × 2560 F32 values per row. The largest wire payload is
3,278,124 bytes. Metadata binds each row's absolute position, actual token and
source path. Missing cached-prefix rows are omitted and exposed explicitly by
the reader; they are not reconstructed from KV, filled from old requests or
silently treated as observed zeros. The archive is a versioned data-only format,
not a program or adapter artifact. Default serving allocates no row buffers;
the existing dump flag opts into copying/writing. These diagnostics do not inject
activations, alter arithmetic or establish default-path token parity.

CPU contract RED failed on the initially absent header, then the `/W4 /WX` host
build and runtime contract passed. The independent Python wire fixture matches
the C++ output byte-for-byte. Four reader tests pass for sparse/missing rows,
header/position/token/source corruption, nonfinite data, duplicates and lengths.
The host contract also tests reset isolation, failed owner copy and out-of-range
prefixes. Timeline analyzer RED/GREEN adds host `igc.preanchor.observe/write`
accounting, explicitly separate from GPU time; all 25 analyzer tests pass.

Real-model two-request smoke on unchanged Hub settings passes:

| Request | Prompt tokens | Observed pre-anchor rows | Paths | Missing rows |
|---|---:|---:|---|---:|
| Calculate `17*23` | 65 | 64, positions 0–63 | serial verifier | 0 |
| Quoted arithmetic, do not calculate | 72 | 71, positions 0–70 | serial verifier | 0 |

All rows are finite and nonzero, F32, position/token-bound; both owned requests
complete. This verifies the real short-read path, **not** the batched/pipelined/
layer-split path or a real cached-missing-row case. Those coverage gaps remain.
The source callback path is implemented but has not been exercised in this smoke.

Candidate SHA256 `f8030aa2d26e48e8383e58a15fb12d49a468bb94bdc66495caa6b1ed46f483a0`;
first archive `d69a8cafe7f2e4cae65be5c05f1f093d88d745aa75004baec1523e1f6c91e733`;
second `ebcd2294a0fe7cd15300012687057567d5db18375425a163810e5a1580302108`.
Evidence: `igc-preanchor-smoke-01/`, `igc-preanchor-{host-red,host-green,host-test,engine-build,smoke}.log`.
The D2x `89;120` build and smoke end `DONE exit=0`; owned cleanup and served/original
learned-profile checks pass. Smoke wall 29.375 seconds, ready 20.203 seconds;
sampled private peak 46,575,521,792 bytes, min commit headroom 14,366,564,352 bytes,
min physical available 9,971,019,776 bytes. These instrumented values are not
paired speed or memory-win claims. No mapper training on these new rows is claimed
yet; I2 remains open and the next ablation needs its own frozen protocol.

## Frozen activation-attention experiment outcome

The new cohort completes 64 requests in 242.187 seconds (ready 19.422 seconds),
with 4,120 observed F32 pre-anchor rows, no missing rows and source path 1 only
(the actual verifier short-read route). Owned cleanup, enumerated source-stat
checks and served/original learned-profile hashes all pass. Its source manifest
retains the previous full model/pack/MTP hashes for unchanged size/timestamps and
rehashes the new executable; this is explicit incremental provenance, not another
fresh full hash of the 91.97 GB model. Original manifests/data are preserved.

Protocol/code were committed before fitting (`aab9555f`, `e3ad3d92`). The new
evaluation-only 64-row manifest, seed20261102, SHA256
`0b6158b0df2ce66edca7bad158b75e32b36e93543a40230891d8d761933bc5ef`,
has not been collected or evaluated. The sixteen former held-out cases are now
known diagnostic data. They are not relabeled as unseen after tuning.

| Variant, same new cohort | Train operator | Train exact positive tuples | Diagnostic operator | Diagnostic exact positive tuples |
|---|---:|---:|---:|---:|
| Final-vector MLP, 334,516 params | 48/48 | 36/36 | 14/16 | 0/12 |
| Activation attention, 227,396 params | 48/48 | 36/36 | 12/16 | 0/12 |

Both have zero false calls on the four diagnostic negatives; the denominator is
too small for a deployment bound. Attention fits in 17.672 seconds on four CPU
threads, 300 fixed epochs, loss13.834379→0.003129. It has no gradient through
frozen Gyro and no output mapper. Its preregistered development stop rule fires:
do not consume the new evaluation set when diagnostic joint tuples remain zero.
Neither variant passes I2 or is eligible for integration/rollout.

Attention state SHA256 `61783ba2088a76ebc11fd4775f6cce22bde99c2027cb7b1f1f88a7b20bf3c4c5`;
same-cohort baseline state `e4b504dbd73ce590e47f4dd328ff2b63cbf0305baa015ae255c35597d0180b1f`;
protocol SHA256 `d6e2d8117d911c0428b0613825a7b8a4dcfb67fcaae55840a83571ea1fdccbc8`;
source manifest `8feeca33e405f189684ae448b8c365f8fcdc327040a302ae09a9d745e4efd2fc`.
Evidence: `igc-i2-preanchor-fit-01/`, `igc-preanchor-experiment-01/`,
`igc-preanchor-{fit-collection,final-vector-baseline,attention-fit}.log`, all DONE0.
Three input-head mask/refusal/gradient tests pass; they concern readout autograd,
not a native frozen-model backward path.

### Diagnosis before the next change

Only 47 distinct absolute operand values occur among 36 positive fit examples;
12 absolute values in the diagnostic set were not seen as numbers in training.
Per-positive attention field correctness (n=12): operator8, lhs/rhs sign9/10,
lhs/rhs length11/9, lhs ones/tens2/0, rhs ones/tens0/1. This localizes the failure
to digit extraction as well as routing; it does not prove data coverage is the
sole cause. Exact causal-prefix comparisons among early prompt positions also
show final-HC residual variation (756 comparisons; relative-L2 median0.07833,
max0.18141) under the adaptive profile. Its cause is not established; do not call
this a model/capture bug or explain it from an unrelated run.

[IGC §3.3](https://arxiv.org/html/2501.00684v1#S3.SS3) reports best results at early
layer1 and a left-aligned digit representation. Our successful capture seam is
currently final HC, not that early seam. The actual Gyro tokenizer separately
encodes each digit (`17`→`1`,`7`; `123456`→six digit tokens), unlike the paper's
Llama grouping. Its alignment result cannot simply be transferred as a proven
fix. Next investigate an explicitly named early native activation boundary
without disabling actual engine fusion, and keep learned digit layout separate
from the unchanged little-endian exact-executor ABI. No new speed benchmark is
scheduled ahead of usable integrated IGC.

The reused build folder also needs an archival distinction. Exact f803 executable
bytes are now retained in a private content-addressed cache before another build.
Offline source resolution verifies historical cached content by SHA256; a cache
name alone is insufficient. It is not permission to use a different current Hub
baseline. Three RED/GREEN tests cover changed live files, exact frozen fallback,
wrong cache bytes and content-identical timestamp changes. The earlier 813 PE was
not archived before replacement; its original logs/source commit/data remain,
but exact-binary replay of that earlier cohort is unavailable. This does not
retroactively alter its observed run or invent a replacement executable identity.
