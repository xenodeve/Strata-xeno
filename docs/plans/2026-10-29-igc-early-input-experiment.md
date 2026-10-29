# I2: early native input-state experiment

The final-HC attention readout fit its 48 training samples but extracted zero of
12 exact known-diagnostic positive tuples. This experiment tests the input seam;
it does not assume the earlier seam solves generalization. All sixteen diagnostic
samples are already known. They are development evidence, never fresh evaluation.

## Frozen comparison

Collect the unchanged 64-row corpus (`fce8f8c8…`) once with the same Gyro-M
262K/five-worker Hub profile, caching, offload and MTP settings. Observe three
actual native tensors from that cohort: final-HC residual, embedding broadcast,
and the first native attention output before layer-zero MoE combination. These
names describe Strata's fused execution points; the third is not asserted to be
the paper's complete layer-one output. Preserve fusions and configuration.

The two-request smoke with executable
`ae6c485fb6b765ae4005183bac1712508b801dbfb1da35ee46bae50966597d7d`
validated all 135 embedding rows against the native Q8_0 embedding control and
independent block decoding: zero differing values. All three taps matched on
the 45 common-prefix rows. This validates that cohort, not every prefix/cache
mode or the unexplained variation in earlier observations.

Compare the same 227,396-parameter attention architecture on each stage. Keep
the final mixed vector as the query conditioning in all three arms. Keep seed
20261029, AdamW learning rate 0.001/weight decay 0.01, 300 full-batch epochs,
four CPU threads, deterministic operations, 180-second fit/210-second outer
bounds. No labels, raw text, token IDs or arithmetic answers enter the network.
Token IDs only bind observations and validate capture fidelity. Fit normalization
on the 48 training rows. Missing/padded observations remain masked.

Save each arm's weights, stage identity, protocol/corpus/source/executable hashes
and all predictions. Do not tune architecture, epochs or thresholds after viewing
an arm. Compare field errors as well as whole tuples, no-call/operator results,
missed calls and false calls. An exact arithmetic executor cannot repair wrong
operands; a wrong sign, length or digit is a failed invocation.

## Development stop and untouched evaluation

The development eligibility rule remains full training-positive exactness and at
least one known-diagnostic exact positive tuple. If no arm meets it, retain the
new 64-row evaluation (`0b6158b0…`) unconsumed and report the failure. If multiple
arms qualify, select by diagnostic exact routed tuples, then fewer false calls,
then embedding/attention/final in that fixed tie order. Lock weights before a
single evaluation. No claim of generalization comes from choosing on known data.

Fresh feasibility acceptance remains 48/48 positive exact tuples and no false
call on 16 negatives. This is a finite two-digit gate, not 62-digit coverage or
daily-use readiness. No architecture/default activation follows from a failed
or partial gate. I3 output mapping, I4 committed-state integration and subsequent
actual Hub end-to-end quality/latency/token/resource benchmarks remain required.

Capture is instrumented diagnostics with bounded cleanup and resource telemetry;
it is not a speed baseline. The task wrapper adds no private-commit cap. Keep the
selected launcher's 24 GiB physical admission and disclosed emergency bounds,
without changing serving settings. Use exactly one heavy job and preserve raw
failed or partial runs.
