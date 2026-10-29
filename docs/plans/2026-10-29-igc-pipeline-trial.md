# Pipeline trial checkpoint for accepted-prefix integration

Issue #276, G3/I4 prerequisite. The current first-anchor connection remains
unchanged. Add an opt-in component primitive for a caller to retain the pipeline
base, execute a candidate window, then keep or discard the candidate module
state. This does not decide which language rows are accepted.

The seam is the public `LatentPipeline` API, using the existing actual four-network
bundle and old inspected activation fixtures. Test red first. A trial checkpoints
the bounded mutable invocation/readout/row-owner buffers on the same created
CUDA stream. Neural weights and graph addresses stay unchanged. Mapper scratch
is stateless between calls and is recomputed. Trial storage is allocated lazily
outside capture and reused; first-anchor users that never start a trial allocate
nothing extra.

One active trial is allowed. Nested trials and staging a new request during a
trial fail before mutation. Finishing accepted keeps the candidate; finishing
rejected restores the base. The caller must discard rejected language outputs
and fence its stream before reusing external inputs. Partial-prefix acceptance
requires abort followed by replay of only the accepted prefix, not committing a
whole rejected suffix. This primitive does not implement KV/GDN rollback,
dynamic learned triggers, snapshot persistence or conversation restore.

Tests cover pending-to-applied abort/replay, commit followed by graph repeats,
owner changes across requests, wrong stream/nesting, hard-off and stable bounded
trial storage. Use small representative old records; arithmetic correctness is
already checked independently and is not recomputed by host production code.
Sanitizer outcomes and compiled hashes are recorded separately. No speed claim,
Hub setting or default changes. Update Blueprint and tracker with actual scope.
