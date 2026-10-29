# I3 late output-mapper fit, development pilot

Freeze this recipe before inspecting development predictions. Use the completed
30-question/421-position native capture (corpus SHA256
`913fbdf842ce9cc55862200167147dee26f116f81e99ea5792e49ccb817725e3`).
Fit only the 24 training questions, evaluate the 6 development questions after
the final epoch. Do not consume input acceptance64. This isolates output use with
oracle operands and checked I1a results while I2 remains unpassed.

First independently reconstruct every canonical prefix and target from the
frozen corpus, native wrapper and bound tokenizer. Validate native completion,
mixed/logit/prefix hashes and result ownership. Before any optimizer step, project
all421 mixed rows through the qualified native Q8 head and require all248320
logits per row to match captured bytes. A failure stops fitting.

Only `ResultMapper` trains at the final2560-wide head input. No base weights or
body activations receive gradients. Use native quantized forward plus the
explicit dequantized-head transpose STE; this is a surrogate derivative, not a
claim of exact backward through quantization or the full Gyro body.

Seed20261105, deterministic F32, TF32 off, 4 CPU threads, existing CUDA Torch
runtime. Batch16 positions, shuffle training positions every epoch with a fixed
generator; 70 epochs AdamW lr0.001 weight_decay0.01, global cosine decay to0.00001,
no gradient clipping. Fit deadline1800s, outer1980s includes qualification and
cleanup. Record source/protocol/library/model/corpus digests, final weights only,
loss history, finite gradient checks and memory. No memory cap or served changes.

Train the full next-token cross entropy, including numeric symbols, words and
turn closure. No expected token IDs or formatted result enter mapper features.
Teacher forcing includes only previous correct completion tokens. The checked
typed numeric result remains bound to the initial invocation/anchor throughout
each completion. The mapper has result digits/sign plus current mixed activation,
not a supplied output digit index.

After the final epoch, report token accuracy/CE, numeric-symbol accuracy and
per-question complete and numeric-only teacher-forced sequence agreement.
Report training and development separately. Evaluate unchanged base/hard-off,
failed-result bypass, learned correct result, zero numeric content and content
swapped with a different question. Counterfactual content is a diagnostic view,
not an accepted production result or relaxed ownership validation. Hard-off and
failed-result paths must preserve exact mixed bytes and baseline predictions.

The ordinary teacher-forced words are a separate non-regression measure; fewer
numeric errors alone is insufficient. No result of this pilot establishes
free-running full-number accuracy, spontaneous invocation, native NN adapter
execution, production transaction correctness or Hub speed. Those require the
later in-engine injection and model tests. This dataset covers at most two-digit
operands/four-digit results, irrespective of the result interface's62-digit bound.
