# Controlled runtime arithmetic reaches the native decoder

## What runs

The version2 diagnostic fixture stores the canonical request prefix digest and a
160-byte typed `ExactRequest`, rather than the earlier precomputed `ExactResult`.
It contains operands and operation/ownership metadata, no answer or output token.
On the matched request's first decode window, the engine enqueues one I1a exact
CUDA execution and a bounded result broadcast on the verifier stream. The learned
native output mapper and Q8 head consume the device result on that stream. There
is no host arithmetic, formatted result prefill or host numeric readback.

Version1 remains explicitly output-only replay for historical reproducibility.
Both versions require the diagnostic build and two explicit artifact/fixture
flags; no serving defaults change. Unmatched requests stage an empty failure.
The controller still supplies oracle operands and fixed answer-start selection.
This does not establish learned I2, spontaneous use or dynamic reasoning triggers.

The broadcast leaves row0 untouched and copies rows1..63, with explicit stream
and count1..64 validation. The native fixture reader also rejects unsupported
domain/operator/reserved fields, wrong owner/anchor and noncanonical integers.
The mapper status fence and fresh startup content hashing remain diagnostic
costs, not a production latency result.

## Verification

The full CUDA89;120 build succeeds. All30 typed runtime results match the earlier
independently checked I1a result bytes. Native exact/serializer/fixture/broadcast
tests pass. Native output replay still matches the421 frozen output positions and
their Q8-head decisions. Broadcast graph A/B/A and error-path tests pass under
memcheck, initcheck, synccheck and racecheck with zero errors/warnings. These checks
do not waive the separate earlier Qwenputer WHILE instrumentation gap. HIP, SYCL
and Linux are not run.

The actual Hub Gyro-M262K/five-worker profile, Vision, original GPU mapping,
MTP suffix, offload/cache and priorities are retained. Each of three sequential
arms receives the same30 known arithmetic questions plus two unmatched ordinary
prompts, temperature0/max96. Expert-profile inputs are identical task-owned copies.
No private-commit cap or served configuration is introduced. This is a controlled
functional run, not a speed benchmark or fresh acceptance corpus.

| arm | training questions | known development questions | token limits |
|---|---:|---:|---:|
| pre-output-hook build |23/24|6/6|1|
| current build, diagnostic off |23/24|6/6|1|
| current build, runtime diagnostic on |24/24|6/6|0|

Enabled examples: `92 - 18` produces `The result is 74.`; `12 * 64` produces
`12 * 64 = 768`. Both match the independent expected value. The frozen new rubric
separates correct explicit/equation results from the old one-integer format score:
enabled29/30 satisfy that narrower format, while all30 have consistent correct
numeric results. Original earlier reports remain unchanged. Offline checking never
executes generated arithmetic and is outside the inference path.

Raw-token parity remains **failed**, including an unmatched Thai control. The
earlier same-prehook-binary control already reproduced variability before this
hook; it does not establish its root cause or excuse the release gate. No general
quality gain, token/latency improvement, disabled parity or I3 completion follows
from this small controlled result. The next integration work is learned readiness
and operand extraction, followed by ownership/lifecycle and larger output tests.

Total235.844s includes three model boots, content checks and cleanup; it is not
inference latency. Sampled maximum single-process private commit46,821,982,208B,
minimum available physical10,372,239,360B and system commit headroom13,937,270,784B.
No emergency floor or timeout is reached; task tree is drained.

## Frozen evidence

* Candidate `74c104b24727202f156c0326afc1f18e586cce29b9b6bffff495d1385d089258`.
* Pre-hook `ae6c485fb6b765ae4005183bac1712508b801dbfb1da35ee46bae50966597d7d`
  is the earlier main-based diagnostic build, not untouched main.
* Fixture5868B/30 requests
  `3f9b5ea2eb8700da189b04d2748784b9252dfd5dc5118babd4e167aeff06cde9`.
* Output artifact `43cf5c8d341bf91966ffc6fa22aba67eb3e04be6a3dcfd694af4b21fb1c62c2c`.
* Runner `a90885f66af77ab82f03ad51de49d6447134f98c2a9c6c36c143fdc5864b1b7c`.
* Pre-run protocol `aa47ee68610b9db890e4db6b0011730f5ad0da9e9fe4a94816517fb61a5d58a5`.

Private evidence: `igc-runtime-executor-build-01.log`,
`igc-runtime-native-output-probe-build.log`, `igc-runtime-validation-01/` (commands,
binary hashes and sanitizer logs), `igc-runtime-model-01/` (inputs/report/memory,
per-arm records/config/engine/Vision logs) and corresponding headless logs, DONE0.
[Frozen protocol](../plans/2026-10-29-igc-controlled-runtime-executor.md).
