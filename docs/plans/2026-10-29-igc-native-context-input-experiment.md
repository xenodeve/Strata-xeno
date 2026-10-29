# I2: frozen contextual-feature development experiment

The optional native prefix extractor now passes full-model first-attention
fidelity, windows one/eight and reset isolation. A 64-request sizing run completes
in 4.25s including loading/repeat/archive validation, using 711,867,136 explicit
owned device bytes plus 66,498,560 native dense bytes and a 675,430,400-byte mapped
embedding. Sampled child private-commit peak is 2,469,007,360 bytes. Minimum system
commit headroom is 59,879,272,448; physical available 31,470,936,064 bytes. This
prices offline data preparation, not production latency or total VRAM.

## Frozen comparison

Use the existing expanded v2 corpus exactly: 10,240 training and 384 known
development rows, row SHA256
`fe0e9d86210b4c0c7ef544a7b4a5982dc9c8f53c3a3eebada7a2e34b2356fa7c`.
Keep the original prompt wrapper/tokenizer, native geometry/weights and all
exclusions. Preserve fresh acceptance64 untouched. It is not optimizer data,
development data or a release test used for model selection.

Only the features change from the completed cosine embedding control: actual
first-native-attention HC4 F32 rows replace pointwise native embedding HC4 F32
rows. This is the pre-MoE seam already qualified, not a full transformer layer.
Keep learned position features, query-free contextual 96,964-parameter model,
seed20261031, batch64, AdamW .001/.01 and global cosine .001 to .00001, 70 epochs,
same training shuffle and final-checkpoint-only protocol. No trainable Gyro
weights, token IDs as neural features, text parser, symbolic operand substitution
or result mapper. Report fit/development metrics and per-field errors separately.

## Data and resource contract

`tools/igc/context_cache.py` owns collection and validation. Recreate the same
64 existing real prefixes first, then tokenize the frozen corpus. Every relevant
expression must lie in the last80 pre-anchor rows; prefixes are bounded to256.
Run chunks of at most64 with the pinned optional executable, reset/repeat and
commit fences. Validate every archive's token/position/source/shape/finite values.
Consolidate raw F32 values losslessly into a disk-backed NPY plus explicit missing
mask and per-record archive hashes; remove only redundant task-owned temporary
archives after their consolidated bytes are flushed. No arbitrary recursive
deletion. Never train an incomplete cache.

The exact padded F32 payload is 34,812,723,200 bytes (10,624 x80 x4 x2560 x4).
Check current free disk against payload plus4GiB margin, physical available8GiB
and commitment12GiB before collection. These are disclosed offline diagnostic
admission checks, not Hub settings. Sample actual process/system memory per chunk;
stop only the owned child on timeout or an emergency2GiB system-commit/1GiB
physical floor. One heavy job, headless, no private-commit cap. Bound collection
to1,200s and each chunk90s. Cache is file-backed and flushed per chunk; it is not
a 35GB private RAM allocation or a permanently pinned hidden-state store.

During training, copy only a bounded minibatch from the read-only F32 mmap to the
GPU. No whole-corpus device copy. Validate regular non-reparse files, exact sizes,
corpus/protocol/source/binary identities and content hashes before fitting. Fit
deadline1,200s/outer1,260s; this larger offline bound accommodates actual feature
transfer, without changing the data/model/optimizer recipe. Retain failures and
final state; do not choose an earlier checkpoint on development performance.

## Completion and next step

Input generalization is still unpassed until the declared exact tuple/no-false-
call gate passes. Check correct operators, signs, lengths and nonpadding digits,
then the three development axes. Do not consume fresh64 merely because fit is
perfect. The context feature result alone does not establish model answer use,
full62-digit support, dynamic triggers or end-to-end benefit. Continue I3 training
infrastructure independently; actual Hub evaluation follows usable integration.

Change sites: cache collector/reader and tests; existing training CLI receives an
explicit context-cache option while preserving its default embedding path;
BLUEPRINT records offline responsibilities; reports/issue comments carry outcome.

## Bounded-I/O correction after the interrupted capture

The initial collector stops after9,408 completed rows on its physical emergency
floor. Commitment remains ample. A bounded1GiB read reproduction shows the
whole-file mapping adds1,048,739,840 bytes to the collector working set while
private commit stays fixed; closing the map releases it. Reading the same bytes
through a reused64-row buffer instead bounds the working set near238MB. This
supports fixing the retained mapping lifetime, without lowering quality, deleting
data, raising floors or changing Hub settings.

Replace the large persistent mapping with `NpyRows`, which reads/writes at most
64 exact F32 rows through ordinary bounded I/O. Keep file format, activations,
corpus, model and fitting recipe unchanged. Journal identities before collection,
record metadata per completed chunk and sample the collector as well as its child.
Training uses the same bounded reader rather than retaining a whole-file map.

Resume the owned interrupted cache only after validating its frozen corpus,
completed chunk inputs/output markers, every persisted row/mask/finite/padding
value, current source/binary identities and exact committed boundary in the
preserved failure log. Independently recapture both ends (128 rows) and require
bitwise equality before adding the remaining rows. Hash recovered F32 content
now; explicitly do not claim the original per-archive hashes survived, because
the initial collector journaled them only on final completion. Preserve original
failed source/log/memory and incomplete last-chunk files. Completion report names
this recovery provenance. If any check fails, refuse the recovered cache.

The older embedding contextual-block protocol keeps its original filename and
byte identity. This native-feature experiment now has a distinct filename; the
temporary naming collision did not modify historical weights, reports or data.
