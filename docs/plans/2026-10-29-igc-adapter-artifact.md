# IGC diagnostic neural-weight artifact

Native integration needs a bounded, identity-bound weight artifact rather than
importing Torch or Python into serving. Prepare the exporter/host validator while
the frozen input fit runs; native operator/loader tests follow after the heavy job.

Version1 is **diagnostic-only**. No current mapper has passed the full I2/I3/use
gates. A diagnostic artifact cannot be enabled in a serving loader or relabeled
eligible merely because a component fit succeeds. Its flags reject production
use. The two fixed architectures are the current query-free contextual input
readout and the late gated result residual, with separately enumerated semantics.

## Binary contract

Little-endian fixed256-byte header: magic `STIGCM01`; twelve uint32 fields
(version, kind, flags, tensor count, embedding width, HC, context window, maximum
digits, exact ABI, numeric domain, operation mask, reserved0); uint64 payload bytes;
five32-byte digests (base model, tokenizer binding, training state, training protocol,
neural schema);32 reserved zero bytes. Kind1 is the contextual input architecture,
kind2 the gated output architecture. Only diagnostic flag1 is accepted in v1.

Tensor descriptors are40 bytes: uint32 canonical slot/rank/four dimensions,
uint64 payload-relative offset/count. Require exact generated schema order/shape,
canonical16-byte offsets, no overlap, zero padding, F32 finite values and no
trailing content. Bound the whole file to8MiB before reading and refuse reparse
or non-regular files. Tail32 bytes are SHA256 of all preceding bytes; callers also
pin the full artifact SHA256 and the expected model/tokenizer identities before
exposing any weights. A checksum is content identity, not an authorization claim.

Derive tensor names/shapes from the fixed first-party model classes and include
explicit operator semantics in the schema digest. Validate LayerNorm epsilon,
GELU/dropout/norm-first and attention geometry so a shape-preserving semantic
change cannot silently keep the ABI. Code generation/native parity will consume
this same schema; do not maintain an unguarded second name list.

Input kind explicitly records the current2-digit/80-row experimental scope;
the exact executor's62-digit capacity does not enlarge the learned input's range.
Output kind permits the typed62-digit feature shape but has no result-use claim.
Before production, add malformed/OOD/range refusal tests and qualify actual short,
prefill and cached-prefix activation sources, hard-off parity and MTP integration.

Change inventory: `igc/adapter_artifact.py` exporter/host validator and boundary
tests; BLUEPRINT responsibility/evidence. Use safe named tensors, no pickle or
code payload. Native C++ loader/kernels, qualified learned artifacts, production
configuration and serving activation remain separate required steps.
