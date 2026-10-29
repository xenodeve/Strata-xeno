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
