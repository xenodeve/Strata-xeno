# I2: expanded supervision on the verified CUDA backend

The contextual CPU arm nearly fit training data but still failed exact operands
and the unseen English negative-label template. This next recipe broadens
instruction/negation supervision and fitting effort. It is not an isolated
causal test of one factor or a serving speed comparison. Prior failed arms and
their unchanged protocols remain available.

## Frozen inputs and separation

`expanded_dataset()` freezes seed 20261031, 10,240 fit rows and three 128-row
development axes. Canonical row SHA256:
`fe0e9d86210b4c0c7ef544a7b4a5982dc9c8f53c3a3eebada7a2e34b2356fa7c`.
It adds varied English/Thai arithmetic and non-calculation instructions, including
labels, filenames, quotes, data and character-preservation requests. Exact
development templates remain excluded from fitting. Their semantics/templates
are developer-known from prior diagnostics, so do not call them untouched tests.

Keep the eight absolute withheld values and the three seen/new value/template
axes. Exclude original pilot, fresh evaluation64 and all previous development
operand pairs from every new row. Preserve sign, length, decimal-slot coverage
and independently sampled expression styles. The old `dataset()` output remains
byte-identical by canonical hash; historical caches are not rewritten.

Prepare only the required native embedding rows as before. Verify the selected
tokenizer/wrapper against all 64 real prefixes, every expression inside the
80-row pre-anchor window, source identities, and all native Q8_0 cache rows
against independent FP16-scale/int8 decoding. Only F32 activations enter the
network; cache indices are bookkeeping. No full-model body is run in this seam.

## CUDA qualification and fixed fit

An already installed official PyTorch 2.11.0+cu130 runtime supports primary
RTX 5060 Ti/SM120. An eight-sample, same-state CPU/CUDA check matched predictions,
maximum output difference 2.7657e-5. All parameter gradients passed mixed
per-component tolerances `atol=1e-6, rtol=1e-3` and an additional maximum absolute
bound of 1e-4; an injected wrong-gradient control failed as intended. Earlier
relative-only/vector-norm checks incorrectly rejected mathematically zero and
saturated tiny gradients; those logs remain preserved. This is own-head numerical
qualification, not backward through Gyro or an output mapper.

Train the same query-free, one-contextual-block, 96,964-parameter architecture
from a fresh CPU initialization with seed 20261031, then move it to CUDA.
Use 70 epochs, batch 64, AdamW lr0.001/weight decay0.01, `foreach=False`,
`fused=False`, four host threads and deterministic CPU-seeded batch order.
Disable TF32, flash/memory-efficient SDP and MHA fastpath; use math SDP and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. No mixed precision or base-weight updates.

Keep the 600-second fit/660-second outer deadline, one heavy job, owned cleanup,
and a training-only 2 GiB free-GPU admission check. This is not a Hub baseline
restriction or process private-commit cap. No installation/download or ExLlama
code import. Record runtime/device, all hashes, loss history, CPU commitment and
GPU peak allocated/reserved memory. Persist final epoch only, without selecting
on development metrics. Do not extend epochs or tune thresholds after results.

## Quality and stop

Report training and each development axis separately: operator/readiness,
canonical exact operands/tuples, nonpadding digits, signs/lengths, reversals,
swaps, invalid encodings and false/missed calls. Do not repair wrong outputs.
Eligibility requires all 7,680 fitting positives, all 96 positives on each
development axis and no false calls. Otherwise keep fresh evaluation64 untouched
and preserve the failed state. Further changes need a new declared experiment.

If eligible, lock weights before the original one-time fresh64 evaluation.
Its finite two-digit gate remains all 48 positive tuples and no false calls on
16 negatives. This does not close 62-digit coverage, output mapping, repeated
triggers or integrated usability. Validate learned weights in the native runtime
before claiming integration. The next step after usable integration is actual
Hub end-to-end benchmarking, followed by evidence-led optimization.
