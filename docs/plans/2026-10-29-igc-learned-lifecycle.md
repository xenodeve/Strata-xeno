# Learned wrapper lifecycle without a full-model load

Issue #276, G3-G5/I2-I4 follow-up after `ada2e825`. Exercise the actual
`IgcLearnedDiagnostic` public lifecycle and head hook using one previously
inspected genuine activation record and its matching canonical prompt tokens.
Content-bind the existing deployed model/tokenizer read-only; do not allocate
language-model weights or invoke a host calculator. Keep fresh acceptance sealed.

The first seam is cancellation after a native result has been produced. It must
clear unpublished result state before another ordinary/captured head call;
cancel is idempotent, and a new `begin()` must be required before restaging.
Test red first against a context-only cancellation stub. Then add request-entry
reset, unavailable/foreign history, same-prefix reuse and release/reprepare
controls through the real hook. Preserve stream/fence ownership and stable graph
addresses, destroying captured graphs before releasing their storage.

Include cancellation before the first stage and rejection during graph capture.
For the known capture refusal, end capture and successfully retry revocation
before any further head. This is not recovery from an arbitrary CUDA fault;
failed revocation must never be treated as permission to resume computation.

This is a component lifecycle test, not live language generation, actual API
cancellation, a generalization test, a full session checkpoint or benchmark.
Actual source hashing reads the existing GGUF but never loads its weights.
Record fixture/model/bundle/binary identities, sanitizer scope, resource use and
unrun gates in the report, Blueprint and bilingual Issue #276 comment. No Hub
profile, serving default, system setting or unrelated process changes.
