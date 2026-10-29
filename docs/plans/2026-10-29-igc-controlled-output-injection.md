# I3 controlled native decoder injection

The native mapper passes numerical/head-decision/regular-graph and sanitizer
qualification. Next test autonomous completion, using known checked operands to
isolate I3 while I2 remains unpassed. Do not describe this as spontaneous learned
input selection or a usable Hub feature.

Add an opt-in diagnostic build seam immediately after final HC mixing and before
native head quantization in `Verifier`. Configure before init/graph capture;
prepare weights/workspace on its explicit stream before capture, then enqueue
only native forward in captured graphs. Default builds do not compile this seam.
The diagnostic object outlives all graph replays and drains before freeing its
buffers; one object/stream, no layer split, pipeline windows, batch or replicas.
Ordinary D2x secondary expert offload and the selected MTP suffix remain intact.

Use two explicit diagnostic CLI paths (artifact and fixture). The bounded fixture
contains model/tokenizer/artifact digests plus up to64 canonical prompt-prefix
hashes and their previously checked typed exact results. Validate structure and
content before GPU upload. Fresh content-bind the actual GGUF(s) and tokenizer
with `VerifiedInputs`; retain read handles. The outer harness separately binds
pack/MTP/config/build/profile sources, as that class does not enumerate them.
No arithmetic or answer text is computed by the host matcher. The exact result
was produced by I1a and compared offline with an independent oracle.

At the first decode window, match the complete canonical input prefix and require
result anchor=`prefix_count-1`. Copy its fixed typed result/owner arrays once on
the verifier stream. Unmatched prompts stage an empty upstream-failure result and
retain baseline behavior. NN operations stay in the graph with stable addresses;
device result status controls active/inactive rows. Subsequent windows do not
copy/recompute/serialize results or advance extra language positions. This is a
pure late-head change, not a persistent language KV/GDN write.

For this instrumented pilot, copy/check device statuses after each completed
window and fail closed before publishing that window on unexpected failure.
This diagnostic fence is disclosed, not presented as production latency. The
normal request loop retains greedy MTP acceptance/commit, penalties, positions,
stop tokens and cancellation. Snapshot/rollback/controller ownership for future
dynamic calls remains I4; this fixture cannot waive those gates.

Freeze initial autonomous tests before running:24 fit/six known development
prompts, max96 generated tokens, temperature0, actual Hub Gyro262K/5-worker
configuration including Vision and inherited GPU mapping. Compare candidate
without flags, candidate unmatched fixture and enabled matched fixture in one
session; record raw IDs, final numeric answer, complete/limit/fault outcomes,
model/profile/build hashes and memory. This is functional testing, not ABBA speed
evidence. Test malformed fixture rejection before model allocation and wrong
prefix/anchor isolation. Do not consume acceptance64 or deploy this pilot.

Fresh GGUF hashing and per-window status diagnostics add startup/runtime costs;
include them in accounting and remove diagnostic-only work before practical
benchmarking. Usable integration still requires eligible learned input mapping,
broader output coverage, transaction/lifecycle checks and fair end-to-end evidence.
