# Native Qwenputer artifact format 1.0

This is the data-only ABI for the opt-in Native Flash-Next experiment tracked in
[#276](https://github.com/xenodeve/Strata-xeno/issues/276). It contains a compiled
sparse FFN/GDN circuit, not a language-model checkpoint, Python object, opcode
interpreter or host arithmetic implementation. The format is under development;
no generated artifact is approved for distribution or daily use yet.

## Container

All integers and IEEE-754 binary32 coefficient bits are little-endian. A reader
must validate the entire container, binding and sparse indices on CPU before any
GPU allocation or mutation of a live session. The hard file limit is 256 MiB and
the section limit is 32. Empty typed sections are allowed.

The 64-byte header is `<8sHHIIIQ32s>`: magic `STQPUT01`, major 1, minor 0,
endian marker `0x01020304`, section count, aligned header bytes, exact file bytes,
and SHA256 of all bytes after the header. Each 80-byte section entry is
`<16sQQII32s8s>`: NUL-padded ASCII name, offset, byte size, record stride, flags
(zero), section SHA256 and eight reserved zero bytes. Names use
`[a-z][a-z0-9_]{0,14}` and are strictly sorted. Entries are followed by zero
padding to a 64-byte boundary; sections are contiguous in this order, individually
aligned/padded to 64 bytes. Size must be divisible by stride. No aliases, gaps,
overlaps, trailing data, nonzero padding, unknown flags or unsupported version
are accepted. SHA256 checks integrity; they are not a publisher signature.

## Sections and records

| Section | Record | Meaning |
|---|---|---|
| `config` | 16 uint32 | dimensions, trace vocabulary, text vocabulary, value-head count, FFN-node count, maximum sublayer, state float count, TCO ID, TCC ID, newline ID, capture rows (80), row width (16), expression bytes (255), head width (128), lag count (3), owner dimension (`UINT32_MAX` only in diagnostic fixtures) |
| `identity` | 6 × 32 bytes | math ABI, compiler source manifest, firmware, boot state, model, complete tokenizer binding SHA256, in that order |
| `dims` | `<2i>` | kind (one=0, token=1, FFN=2, head=3), final placement level |
| `features` | `<4i>` | destination dimension, feature opcode, two integer parameters |
| `exprs` | `<2i>` | start/count into `terms`; preserve term order |
| `terms` | `<if>` | residual dimension, exact FP32 coefficient bits |
| `chans` | `<i4f>` | expression index and current/lag1/lag2/lag3 taps |
| `maps` | `<2i>` | key row/value column and channel index; z maps use expression index |
| `ffns` | `<6i>` | original node index, sublayer, output dimension, into flag, term start/count |
| `ffn_terms` | `<fii>` | coefficient, a-expression index, b-expression index, in reduction order |
| `heads` | `<21if>` | original node index, sublayer, value-head index, into flag, rows, columns, q start/count, k start/count, v start/count, z start/count, output start/count, decay expression, beta expression (`-1` means one), initial-row start/count, state offset in floats, tiny-output threshold |
| `head_levels` | uint8 | one per value head: bit=0, byte=1, decay=2 |
| `outputs` | `<iif>` | value column, residual destination, weight, in scatter/reduction order |
| `init` | `<2i128f>` | value-head index, row index, 128 initial state values |
| `keep` | int32 | retained output code dimensions, then tau/mu; order is load-bearing |
| `text_rows` | 16 bytes | compiler projection for every original text ID |
| `text_lens` | uint32 | original byte length for every ID; a projection never authorizes truncation |
| `text_types` | uint8 | original Strata token type for every ID |
| `printable` | int32 | 256 exact byte IDs followed by tool-response open/close and think open/close |
| `meta` | ASCII JSON bytes | human-readable provenance, names, ABI policy and limits; never executable |

Every index/count/range must be revalidated by the native loader, including empty
maps. Every head has 128 × 128 FP32 state cells. Key/query maps and initial row
indices are in 0…127. Keys permit zero or one positive row; queries may have
0…128 positive rows. The initial compiler uses two-hot ROM/DROM reads, but the ABI
does not silently select only two of a larger valid query. Unknown sections are
rejected by the native schema reader even if the generic container reader can
enumerate them. The loader derives all memory costs rather than trusting claims
in `meta`.

## Token features

Input tokens have a separate kind (Trace or Text), never an unchecked offset
supplied by an API client. Trace IDs are contiguous in the pinned family/control
order; text IDs retain the complete Gyro tokenizer binding. Opcode numbers are
ZERO=0, ONE=1, FAMILY=2, BIT=3, DIGIT=4, CODE=5, IS_TEXT=6, MU=7, TAU=8, TCO=9,
TCC=10, TEXT_BIT=11. Parameters identify a family/bit, digit, output-code bit,
or text-row byte/bit. Family indices and sizes are the pinned `rv/spec.py`
order. Output code uses five low family bits followed by sixteen payload bits;
printable text has output family index 20. Nonprintable text has zero code.

ONE is one for both supported token kinds. TAU is -1 on Trace and `2^-40` on
Text; MU is one only on Trace. All unknown feature names fail export. Before an
artifact is accepted, the exporter exhaustively compares every lowered feature
with the pinned original callback over all 230,324 Trace IDs and all text IDs.
The proof digest covers FP32 columns in feature order, Trace then Text.

## Numerical and execution contract

Coefficients must be exactly representable as finite FP32, including values such
as `-2^82` and `2^-41`; int32 numerator/denominator pairs cannot represent the
actual circuit. Sparse indices are int32. Preserve ordered multiplication and
addition with separate rounding; no FMA contraction, FTZ, TF32, global fast-math,
quantization or unordered floating-point atomics. FFNs compute
`sum(c * a * relu(b))`; heads apply lagged projections, decay (keep/half/clear),
beta-controlled row overwrite, ordered query sum, ballast normalization using
columns 0/1, output gates and ordered scatter. The ABI string and its SHA256 are
stored in the artifact. This policy is a target contract; Native exactness must
still pass the real IR/state oracles in P3/P4.

Compiler/export/oracle processes start with `PYTHONHASHSEED=0` and confirm
`sys.flags.hash_randomization == 0` before importing upstream. The pinned source
contains set iteration that changes sparse expression term order across random
Python hash seeds. Pinning the process seed preserves upstream bytes while
freezing reduction order; setting the environment after Python starts is refused.
Artifact provenance records this compiler environment. Historical draft exports
without this freeze are not reproducibility controls.

Layer batching may combine independent work only if it preserves this reduction
contract. Lag history belongs to the input residual at the corresponding head
stage; it must not be shifted after that stage's outputs have already mutated it.
Initial state, committed state, trial state and lag history have separate ownership.
Runtime graph handles are never serialized.

## Capture, scope and provenance

Outside a live enrolled calc call, long language tokens remain valid. Within the
capture, reject rows wider than 16 bytes, reserved tokens, NUL, invalid UTF-8,
more than 80 rows, malformed/nested XML or an expression beyond 255 bytes. Total
XML capture and expression payload have separate budgets. Never clamp or truncate
input. Only newly generated enrolled calls can execute; history and quoted/fenced
examples cannot. A failure or limit never publishes a partial answer as success.

`MAXD=62` is an internal capacity, not a guarantee that every expression with a
62-digit answer succeeds. The pinned firmware's `mul` rejects when operand lengths
sum above 62 before trimming; `add_mag` reserves an extra carry digit. Actual
reference diagnosis found `10**61`, a 62-digit literal plus zero, and that literal
times one return `error`; the literal itself and its subtraction by one succeed.
Oracles retain the independent exact mathematical expectation and a separate
known-refusal classification. Such a case is not counted as mathematical success.
Fixing/rebuilding firmware requires a new artifact identity and the applicable
execution approval; a host calculation must never silently replace its error.

Pinned compiler source is MIT with byte-level Git blob/SHA256 verification before
import. Generated artifacts currently carry a distribution hold because linked
firmware runtime provenance/notices are not fully established. Local execution
approval for a digest does not approve redistribution or another firmware digest.

The artifact format can represent another approved compiled program without a
host opcode interpreter. General program selection/loading, model-addressable
filesystem/memory and Universal model adapters require additional explicit
capability contracts and tests; the current calc eligibility gate is not evidence
that those capabilities are integrated.
