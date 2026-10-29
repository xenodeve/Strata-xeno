# Native input forward qualification

Implement the existing diagnostic kind1 query-free contextual `InputReadout`
schema without changing its weights, supervision or data. Support either genuine
native embedding rows or first-attention rows, right-aligned80×4×2560 states,
explicit valid-row mask and the fixed positional indices0..79. No token IDs,
literal parser, final-query shortcut or oracle operands are native mapper inputs.

Reproduce per-HC LayerNorm/projection, learned positions, one pre-norm64-wide
four-head transformer with128-wide exact-GELU FFN, nine learned queries and the
52 raw output logits. This is float neural forward, not exact arithmetic or
proof the diagnostic weights are eligible. Compare against frozen Torch math
forward and saved field decisions for both existing artifact states. CPU/GPU
rounding need not be bitwise identical; require atol0.001/rtol0.0001 and identical
field argmax decisions on tested observations, then broaden all development rows.

Load validated kind1 weights once and reserve capacity1..64 on one explicit
created stream. No allocation, synchronization, host arithmetic or result readback
inside `run`. Hard off bypasses before arguments/device work. Validate mask bytes,
at least one observed row and finiteness of valid states. Ignore absent-row bytes
just as Torch masks them. Failed rows must preserve caller logit bytes, including
NaNs, while returning a distinct status; finite-overflow weights/states also fail
closed. Normal/graph/repeated execution, capacity1/64 and alternating inputs must
agree; run all applicable ComputeSanitizer tools. Report exact memory reservation.

Use the existing strict bounded model/tokenizer/state/protocol/schema artifact
loader. Native probe inputs are bounded first-party binary observations, not code.
Build CUDA89;120; execute primary GPU only. HIP/SYCL/Linux remain explicit debt.
This component does not establish learned controller ownership, readiness, typed
request assembly, full-model utilization or I2 completion. Connect those only
after this forward is qualified, keeping failed diagnostic models non-production.
