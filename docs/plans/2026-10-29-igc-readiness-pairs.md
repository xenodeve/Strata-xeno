# Final-activation readiness, matched-intent diagnostic

The current I2 embedding readout fits training but misroutes unseen quoted or
negated prompts. Its inspected development set has no matched positive/negative
pair. Existing final-head pilot activations separated readiness better, but only
on a small previously inspected corpus. Test this mechanism directly, without
changing operand extraction, the numeric ABI or the language model.

Freeze320 examples with seed20261107:96 training operand pairs and48 development
pairs, each appearing once as a calculation request and once as a copy/quotation
request, plus16 ordinary/unsupported negatives per split. Pairs, templates and
all previous arithmetic/input/output/fresh-acceptance pairs are disjoint across
splits. The domain is the current two-digit signed integer add/subtract/multiply
pilot. Unsupported here means outside that initial module, not outside future
programmable or geometry capabilities. Labels are binary readiness; no answers,
generated result text or test logits are training features.

Collect one actual head-mixed-after-final-HC2560-vector at fixed answer start per
prompt, using the selected unchanged Hub Gyro262K/five-worker/Vision/MTP/D2x
configuration and canonical wrapper. One generated token completes each owned
request; it is not a target or input to training. Require recorded boundary,
prefix, source/tensor hashes and completed ownership. Keep prior capture modes
unchanged; use an explicitly named readiness-anchor mode. Diagnostic admission,
capture and timeout remain separate from Hub behavior and timing evidence.

Train a separate2560→128→2 GELU readout, normalization fitted on training rows
only, seed20261107, CPU four threads, deterministic F32. Full-batch300 epochs,
AdamW lr0.001/weight_decay0.01, no clipping/checkpoint selection or development
fitting, bounded120s. Evaluate the final argmax only: positive recall, negative
false calls, complete matched-pair success and per-template/language confusion.
All development positives and negatives must classify correctly before considering
this small diagnostic a pass. Failure remains a diagnostic; fresh acceptance64
stays untouched. Reproduce saved predictions and separate operator/operand errors
when combining with the existing extraction component later.

This is fixed-anchor readiness research, not spontaneous multi-call integration,
causal proof of hidden-state quality, full-range coverage or a latency benchmark.
No base weights change, no backward through Gyro and no serving/default activation.
